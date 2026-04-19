"""
auth.py — Authentication module with Google OAuth 2.0, email/password, JWT tokens, and SQLite.

Provides:
  - SQLite user database  (file-based — no external service needed)
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
import sqlite3
import time
from typing import Any, Dict, Optional
from urllib.parse import urlencode

import bcrypt
import httpx
import jwt
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

HERE = os.path.dirname(os.path.abspath(__file__))

# ── Configuration (read from environment / .env) ────────────────────────────
GOOGLE_CLIENT_ID: str = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET: str = os.getenv("GOOGLE_CLIENT_SECRET", "")
JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "")
FRONTEND_URL: str = os.getenv("FRONTEND_URL", "http://localhost:8080")

# Google OAuth endpoints
_GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
_GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
_GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"

# SQLite database — one file next to this module
DB_PATH = os.path.join(HERE, "auth.db")

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

_NEW_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    email           TEXT    UNIQUE NOT NULL,
    name            TEXT    NOT NULL DEFAULT '',
    picture         TEXT    NOT NULL DEFAULT '',
    password_hash   TEXT,
    google_id       TEXT    UNIQUE,
    auth_provider   TEXT    NOT NULL DEFAULT 'email',
    created_at      REAL    NOT NULL,
    last_login      REAL    NOT NULL
)
"""


def init_db() -> None:
    """Create or migrate the users table."""
    conn = sqlite3.connect(DB_PATH)

    # Check if table exists
    table_exists = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='users'"
    ).fetchone()

    if not table_exists:
        conn.execute(_NEW_SCHEMA)
        conn.commit()
        conn.close()
        print("[auth] SQLite database ready (new schema).")
        return

    # Check if migration is needed (old schema lacks password_hash column)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(users)").fetchall()}
    if "password_hash" not in columns:
        print("[auth] Migrating database to new schema (adding email/password support)...")
        conn.execute("ALTER TABLE users RENAME TO _users_old")
        conn.execute(_NEW_SCHEMA)
        conn.execute(
            """
            INSERT INTO users (id, email, name, picture, google_id, auth_provider, created_at, last_login)
            SELECT id, email, name, picture, google_id, 'google', created_at, last_login
            FROM _users_old
            """
        )
        conn.execute("DROP TABLE _users_old")
        conn.commit()
        print("[auth] Migration complete.")

    conn.close()
    print("[auth] SQLite database ready.")


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
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    now = time.time()

    # Check if user exists by google_id
    row = conn.execute(
        "SELECT * FROM users WHERE google_id = ?", (google_id,)
    ).fetchone()

    if row:
        # Existing Google user — update profile
        conn.execute(
            "UPDATE users SET last_login = ?, name = ?, picture = ? WHERE google_id = ?",
            (now, name, picture, google_id),
        )
    else:
        # Check if an email-only account exists — link it
        row = conn.execute(
            "SELECT * FROM users WHERE email = ?", (email,)
        ).fetchone()

        if row:
            # Link Google to existing email account
            conn.execute(
                "UPDATE users SET google_id = ?, name = ?, picture = ?, auth_provider = 'both', last_login = ? WHERE email = ?",
                (google_id, name, picture, now, email),
            )
        else:
            # Brand new user via Google
            conn.execute(
                "INSERT INTO users (email, name, picture, google_id, auth_provider, created_at, last_login) "
                "VALUES (?, ?, ?, ?, 'google', ?, ?)",
                (email, name, picture, google_id, now, now),
            )
    conn.commit()

    row = conn.execute(
        "SELECT * FROM users WHERE email = ?", (email,)
    ).fetchone()
    user = dict(row)
    conn.close()
    return user


def _create_email_user(email: str, name: str, password: str) -> Dict[str, Any]:
    """Create a new user with email + password. Raises on duplicate email."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    now = time.time()

    # Check if email is already taken
    existing = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    if existing:
        conn.close()
        raise ValueError("email_taken")

    password_hash = _hash_password(password)
    conn.execute(
        "INSERT INTO users (email, name, password_hash, auth_provider, created_at, last_login) "
        "VALUES (?, ?, ?, 'email', ?, ?)",
        (email, name, password_hash, now, now),
    )
    conn.commit()

    row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    user = dict(row)
    conn.close()
    return user


def _verify_email_login(email: str, password: str) -> Optional[Dict[str, Any]]:
    """Verify email + password. Returns user dict or None."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    if not row:
        conn.close()
        return None

    user = dict(row)
    conn.close()

    # User exists but signed up with Google only (no password set)
    if not user.get("password_hash"):
        return None

    if not _verify_password(password, user["password_hash"]):
        return None

    # Update last_login
    conn = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE users SET last_login = ? WHERE id = ?", (time.time(), user["id"]))
    conn.commit()
    conn.close()

    return user


# ── JWT helpers ──────────────────────────────────────────────────────────────

def create_token(user_id: int, email: str, name: str) -> str:
    """Create a signed JWT for the given user."""
    payload = {
        "sub": str(user_id),
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
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM users WHERE id = ?", (int(user_id),)).fetchone()
    conn.close()
    return dict(row) if row else None


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
