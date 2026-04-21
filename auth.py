"""
auth.py — Authentication module with Google OAuth 2.0, email/password, JWT tokens, and MongoDB.

Provides:
  - MongoDB user database (Atlas cloud or local instance)
  - Google OAuth 2.0 login flow
  - Email + password registration and login (bcrypt hashed)
  - JWT token creation and verification
  - FastAPI router with /auth/* endpoints

Endpoints:
  POST /auth/register         → Create account with email + password, return JWT
  POST /auth/login            → Sign in with email + password, return JWT
  GET  /auth/google           → Redirect to Google consent screen
  GET  /auth/google/callback  → Handle OAuth callback, issue JWT, redirect to frontend
  GET  /auth/me               → Return current user profile (requires Bearer token)

The JWT_SECRET_KEY must be identical across both backends so that a token
issued by port 8000 is also valid on port 8001 (used in compare mode).
"""

from __future__ import annotations

import os
import re
import time
from typing import Any, Dict, Optional
from urllib.parse import urlencode

import bcrypt
import httpx
import jwt
from bson import ObjectId
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError, OperationFailure

# ── Configuration (read from environment / .env) ────────────────────────────
GOOGLE_CLIENT_ID: str = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET: str = os.getenv("GOOGLE_CLIENT_SECRET", "")
JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "")
FRONTEND_URL: str = os.getenv("FRONTEND_URL", "http://localhost:8080")

# MongoDB connection string (must be provided via .env / environment)
MONGODB_URI: str = os.getenv("MONGODB_URI", "").strip()
MONGODB_DB_NAME: str = os.getenv("MONGODB_DB_NAME", "emotiontrack")

# Google OAuth endpoints
_GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
_GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
_GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"

# MongoDB client and collection (initialized in init_db)
_mongo_client: Optional[MongoClient] = None
_users_collection = None
_analysis_collection = None

# JWT lifetime: 7 days
_JWT_EXPIRY_SECONDS = 7 * 24 * 3600

# Minimum password length
_MIN_PASSWORD_LENGTH = 8

# Basic email regex (good enough for validation — not RFC-perfect)
_EMAIL_RE = re.compile(r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$")


# ── Pydantic request models ─────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: str
    password: str
    name: str


class LoginRequest(BaseModel):
    email: str
    password: str


# ── Database ─────────────────────────────────────────────────────────────────

def init_db() -> None:
    """Connect to MongoDB and ensure indexes exist."""
    global _mongo_client, _users_collection, _analysis_collection

    if not MONGODB_URI:
        raise RuntimeError(
            "MONGODB_URI is not configured. Set it in .env or as an environment variable."
        )

    print(f"[auth] Connecting to MongoDB...")
    _mongo_client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
    # Fail fast if MongoDB is unreachable instead of surfacing errors later.
    _mongo_client.admin.command("ping")
    db = _mongo_client[MONGODB_DB_NAME]
    _users_collection = db["users"]
    _analysis_collection = db["analysis_reports"]

    # Legacy cleanup: remove explicit null google_id values from older writes.
    # This prevents duplicate-key collisions on unique google_id indexes.
    _users_collection.update_many({"google_id": None}, {"$unset": {"google_id": ""}})

    # Create indexes (idempotent — safe to call multiple times).
    _users_collection.create_index("email", unique=True)

    # Replace legacy sparse index with a partial index that only applies to
    # string google_id values. This allows many email-only users (no google_id).
    try:
        _users_collection.drop_index("google_id_1")
    except OperationFailure:
        pass
    _users_collection.create_index(
        "google_id",
        unique=True,
        partialFilterExpression={"google_id": {"$type": "string"}},
    )

    # Report/history storage for completed analysis sessions.
    _analysis_collection.create_index([("user_id", 1), ("created_at", -1)])
    _analysis_collection.create_index([("session_id", 1), ("backend", 1)])

    print(f"[auth] MongoDB connected — database: {MONGODB_DB_NAME}, collection: users")


def get_analysis_collection():
    """Return the MongoDB collection used for persisted analysis reports."""
    if _analysis_collection is None:
        raise RuntimeError("Database is not initialized. Call init_db() first.")
    return _analysis_collection


def _user_doc_to_dict(doc: dict) -> Dict[str, Any]:
    """Convert a MongoDB document to a user dict with string id."""
    if doc is None:
        return None
    return {
        "id": str(doc["_id"]),
        "email": doc.get("email", ""),
        "name": doc.get("name", ""),
        "picture": doc.get("picture", ""),
        "password_hash": doc.get("password_hash"),
        "google_id": doc.get("google_id"),
        "auth_provider": doc.get("auth_provider", "email"),
        "created_at": doc.get("created_at", 0),
        "last_login": doc.get("last_login", 0),
    }


# ── Password helpers ──────────────────────────────────────────────────────────

def _hash_password(password: str) -> str:
    """Hash a plaintext password with bcrypt."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def _verify_password(password: str, password_hash: str) -> bool:
    """Check a plaintext password against a bcrypt hash."""
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


# ── User database operations ─────────────────────────────────────────────────

def _get_or_create_google_user(
    google_id: str, email: str, name: str, picture: str
) -> Dict[str, Any]:
    """Find or create a user from Google OAuth. Links accounts if email exists."""
    now = time.time()

    # Check if user exists by google_id
    doc = _users_collection.find_one({"google_id": google_id})

    if doc:
        # Existing Google user — update profile and last_login
        _users_collection.update_one(
            {"_id": doc["_id"]},
            {"$set": {"last_login": now, "name": name, "picture": picture}}
        )
        doc = _users_collection.find_one({"_id": doc["_id"]})
        return _user_doc_to_dict(doc)

    # Check if an email-only account exists — link it
    doc = _users_collection.find_one({"email": email})

    if doc:
        # Link Google to existing email account
        _users_collection.update_one(
            {"_id": doc["_id"]},
            {"$set": {
                "google_id": google_id,
                "name": name,
                "picture": picture,
                "auth_provider": "both",
                "last_login": now
            }}
        )
        doc = _users_collection.find_one({"_id": doc["_id"]})
        return _user_doc_to_dict(doc)

    # Brand new user via Google
    new_user = {
        "email": email,
        "name": name,
        "picture": picture,
        "password_hash": None,
        "google_id": google_id,
        "auth_provider": "google",
        "created_at": now,
        "last_login": now,
    }
    result = _users_collection.insert_one(new_user)
    doc = _users_collection.find_one({"_id": result.inserted_id})
    return _user_doc_to_dict(doc)


def _create_email_user(email: str, name: str, password: str) -> Dict[str, Any]:
    """Create a new user with email + password. Raises on duplicate email."""
    now = time.time()
    password_hash = _hash_password(password)

    new_user = {
        "email": email,
        "name": name,
        "picture": "",
        "password_hash": password_hash,
        "auth_provider": "email",
        "created_at": now,
        "last_login": now,
    }

    try:
        result = _users_collection.insert_one(new_user)
    except DuplicateKeyError as exc:
        details = getattr(exc, "details", {}) or {}
        key_pattern = details.get("keyPattern", {})
        if "email" in key_pattern or "email" in str(exc):
            raise ValueError("email_taken")
        raise

    doc = _users_collection.find_one({"_id": result.inserted_id})
    return _user_doc_to_dict(doc)


def _verify_email_login(email: str, password: str) -> Optional[Dict[str, Any]]:
    """Verify email + password. Returns user dict or None."""
    doc = _users_collection.find_one({"email": email})

    if not doc:
        return None

    # User exists but signed up with Google only (no password set)
    if not doc.get("password_hash"):
        return None

    if not _verify_password(password, doc["password_hash"]):
        return None

    # Update last_login
    _users_collection.update_one(
        {"_id": doc["_id"]},
        {"$set": {"last_login": time.time()}}
    )

    doc = _users_collection.find_one({"_id": doc["_id"]})
    return _user_doc_to_dict(doc)


# ── JWT helpers ──────────────────────────────────────────────────────────────

def create_token(user_id: str, email: str, name: str) -> str:
    """Create a signed JWT for the given user."""
    payload = {
        "sub": user_id,  # Already a string (MongoDB ObjectId converted)
        "email": email,
        "name": name,
        "iat": time.time(),
        "exp": time.time() + _JWT_EXPIRY_SECONDS,
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm="HS256")


def verify_token(token: str) -> Optional[Dict[str, Any]]:
    """Verify and decode a JWT. Returns the payload dict or None."""
    if not token or not JWT_SECRET_KEY:
        return None
    try:
        return jwt.decode(token, JWT_SECRET_KEY, algorithms=["HS256"])
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        return None


def get_user_from_token(token: str) -> Optional[Dict[str, Any]]:
    """Verify a JWT and fetch the full user record from the database."""
    payload = verify_token(token)
    if not payload:
        return None
    user_id = payload.get("sub")
    if not user_id:
        return None

    try:
        doc = _users_collection.find_one({"_id": ObjectId(user_id)})
    except Exception:
        return None

    return _user_doc_to_dict(doc) if doc else None


# ── FastAPI Router ───────────────────────────────────────────────────────────

auth_router = APIRouter(prefix="/auth", tags=["Authentication"])


# ── Email / password endpoints ────────────────────────────────────────────────

@auth_router.post("/register")
async def register(body: RegisterRequest):
    """Create a new account with email + password and return a JWT."""
    email = body.email.strip().lower()
    name = body.name.strip()
    password = body.password

    if not _EMAIL_RE.match(email):
        raise HTTPException(status_code=400, detail="Invalid email address.")
    if len(password) < _MIN_PASSWORD_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Password must be at least {_MIN_PASSWORD_LENGTH} characters.",
        )
    if not name:
        raise HTTPException(status_code=400, detail="Name is required.")

    try:
        user = _create_email_user(email, name, password)
    except ValueError as e:
        if str(e) == "email_taken":
            raise HTTPException(status_code=409, detail="An account with this email already exists.")
        raise

    token = create_token(user["id"], user["email"], user["name"])
    print(f"[auth] New user registered: {email}")
    return {"token": token, "user": {
        "id": user["id"], "email": user["email"],
        "name": user["name"], "picture": user["picture"],
    }}


@auth_router.post("/login")
async def login(body: LoginRequest):
    """Sign in with email + password and return a JWT."""
    email = body.email.strip().lower()
    password = body.password

    if not email or not password:
        raise HTTPException(status_code=400, detail="Email and password are required.")

    user = _verify_email_login(email, password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    token = create_token(user["id"], user["email"], user["name"])
    print(f"[auth] User signed in: {email}")
    return {"token": token, "user": {
        "id": user["id"], "email": user["email"],
        "name": user["name"], "picture": user["picture"],
    }}


# ── Google OAuth endpoints ────────────────────────────────────────────────────

def _get_callback_url(request: Request) -> str:
    """Build the Google OAuth callback URL from the current request."""
    base = str(request.base_url).rstrip("/")
    return f"{base}/auth/google/callback"


@auth_router.get("/google")
async def google_login(request: Request):
    """Redirect the user to Google's OAuth consent screen."""
    if not GOOGLE_CLIENT_ID:
        raise HTTPException(
            status_code=500,
            detail="GOOGLE_CLIENT_ID is not configured. Set it in .env.",
        )

    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": _get_callback_url(request),
        "response_type": "code",
        "scope": "openid email profile",
        "access_type": "offline",
        "prompt": "consent",
    }
    return RedirectResponse(f"{_GOOGLE_AUTH_URL}?{urlencode(params)}")


@auth_router.get("/google/callback")
async def google_callback(request: Request, code: str = Query(...)):
    """Exchange the Google authorization code for user info and issue a JWT."""
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        raise HTTPException(
            status_code=500,
            detail="Google OAuth credentials are not configured.",
        )

    callback_url = _get_callback_url(request)

    # Step 1: Exchange authorization code for an access token
    async with httpx.AsyncClient() as client:
        token_resp = await client.post(
            _GOOGLE_TOKEN_URL,
            data={
                "code": code,
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "redirect_uri": callback_url,
                "grant_type": "authorization_code",
            },
        )
    if token_resp.status_code != 200:
        raise HTTPException(
            status_code=400,
            detail="Failed to exchange authorization code with Google.",
        )

    tokens = token_resp.json()
    access_token = tokens.get("access_token")
    if not access_token:
        raise HTTPException(status_code=400, detail="No access token received.")

    # Step 2: Fetch the user's profile from Google
    async with httpx.AsyncClient() as client:
        userinfo_resp = await client.get(
            _GOOGLE_USERINFO_URL,
            headers={"Authorization": f"Bearer {access_token}"},
        )
    if userinfo_resp.status_code != 200:
        raise HTTPException(
            status_code=400,
            detail="Failed to fetch user info from Google.",
        )

    userinfo = userinfo_resp.json()
    google_id = userinfo.get("id")
    email = userinfo.get("email", "")
    name = userinfo.get("name", "")
    picture = userinfo.get("picture", "")

    if not google_id or not email:
        raise HTTPException(
            status_code=400, detail="Incomplete user info from Google."
        )

    # Step 3: Create or update the user in the local database
    user = _get_or_create_google_user(google_id, email, name, picture)
    print(f"[auth] User authenticated: {email}")

    # Step 4: Issue a JWT and redirect to the frontend
    token = create_token(user["id"], user["email"], user["name"])
    return RedirectResponse(f"{FRONTEND_URL}/auth/callback?token={token}")


@auth_router.get("/me")
async def get_current_user(request: Request):
    """Return the authenticated user's profile."""
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization header.")

    token = auth_header.split(" ", 1)[1]
    user = get_user_from_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired token.")

    return {
        "id": user["id"],
        "email": user["email"],
        "name": user["name"],
        "picture": user["picture"],
        "auth_provider": user.get("auth_provider", "google"),
    }
