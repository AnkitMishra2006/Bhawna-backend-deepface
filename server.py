"""
server.py — FastAPI WebSocket server for real-time emotion tracking (DeepFace backend).

This server is a drop-in replacement for ../backend/server.py.
The WebSocket protocol is IDENTICAL — the frontend only needs to change the URL:
    ws://localhost:8000/ws/{id}   ← custom-trained EmotionNet
    ws://localhost:8001/ws/{id}   ← this server (DeepFace)

Start this server:
    uvicorn server:app --host 0.0.0.0 --port 8001 --reload

── Why DeepFace ──────────────────────────────────────────────────────────────
DeepFace (github.com/serengil/deepface) wraps several production-grade
pre-trained models. For emotion recognition it ships a mini_XCEPTION model
trained on the FER2013 dataset (35,887 labelled face photographs).
It supports 7 emotions: angry, disgust, fear, happy, neutral, sad, surprise —
identical to the custom model.

DeepFace handles the full pipeline internally:
  face detection → alignment → normalisation → emotion inference

This means there is no model.py or train.py in this directory — DeepFace
replaces all of that.

── First-run behaviour ───────────────────────────────────────────────────────
On the very first startup DeepFace downloads its model weights from GitHub
(~100 MB) and stores them in ~/.deepface/weights/.  This is automatic.
Subsequent startups load from the local cache and take only a few seconds.

── WebSocket Protocol ────────────────────────────────────────────────────────
Client → Server  (per frame, every ~200 ms):
    {
        "type":      "frame",
        "data":      "<base64-encoded JPEG/PNG — may include data URI prefix>",
        "timestamp": 1.23       ← seconds into the video
    }

Server → Client  (response to each frame):
    {
        "type":             "result",
        "face_detected":    true,
        "raw_scores":       {"angry": 1.2, "happy": 84.1, ...},   ← 0–100 %
        "smoothed_scores":  {"angry": 0.9, "happy": 79.4, ...},   ← 15-frame avg
        "dominant_emotion": "happy",
        "confidence":       79.4,
        "timestamp":        1.23,
        "model":            "deepface"      ← extra field for comparison UI
    }
    or, if no face found:
    {
        "type":          "result",
        "face_detected": false,
        "timestamp":     1.23,
        "model":         "deepface"
    }

Client → Server  (when video ends / user stops):
    {"type": "end"}

Server → Client  (final AI / template report):
    {
        "type": "report",
        "text": "Throughout the session..."
    }

── Health check ──────────────────────────────────────────────────────────────
GET  /health  →  {
    "status": "ok",
    "model_loaded": true,
    "model_source": "deepface",
    "detector_backend": "opencv",
    "classes": [...],
    "window_size": 15
}

── LLM report (optional) ────────────────────────────────────────────────────
Set GEMINI_API_KEY in a .env file to enable AI-generated reports using Google
Gemini (free tier — no billing required).  Without a key the server falls
back to a rich template report.

── Detector backend ─────────────────────────────────────────────────────────
Set DEEPFACE_DETECTOR in .env to control accuracy vs speed tradeoff:
  opencv      — Haar cascades, ~5 ms per frame, decent accuracy (default)
  mediapipe   — ~15 ms per frame, excellent accuracy, recommended for webcam
  ssd         — ~20 ms per frame, good accuracy
  mtcnn       — ~50 ms per frame, very accurate, slower on CPU
  retinaface  — ~100 ms per frame, best accuracy, slowest on CPU
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import tempfile
import time
from collections import deque
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from deepface import DeepFace
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# ── Environment ──────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(HERE, ".env"))

# ── Authentication (Google OAuth + JWT + MongoDB) ───────────────────────────
from auth import auth_router, get_user_from_token, init_db

# ── Config (overridable via .env) ─────────────────────────────────────────────
# Detector backend used by DeepFace for face detection.
# See module docstring for options and speed/accuracy tradeoffs.
DETECTOR_BACKEND: str = os.getenv("DEEPFACE_DETECTOR", "opencv").lower()


# ── Globals ───────────────────────────────────────────────────────────────────
# DeepFace manages its own model weights internally (in ~/.deepface/weights/).
# We track whether the warm-up succeeded so health check can report it.
DEEPFACE_READY: bool = False

# The 7 emotion labels DeepFace returns — identical to the custom model.
CLASS_NAMES: List[str] = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Per-session state: key = session_id string from the WebSocket URL
SESSIONS: Dict[str, dict] = {}

# Sliding window size: 15 frames ≈ 3 seconds at 5 fps (per project spec)
WINDOW_SIZE: int = 15

# Asyncio semaphore — allows at most N concurrent DeepFace inference calls.
# TF2 eager mode is generally thread-safe for inference, but a semaphore of 4
# prevents overloading the CPU with many simultaneous heavy operations while
# still serving multiple sessions in parallel.
_INFERENCE_SEM: Optional[asyncio.Semaphore] = None  # initialised in lifespan

# Whisper speech-to-text model (loaded at startup, None if disabled)
WHISPER_MODEL: Optional[Any] = None


# ── Lifespan events ──────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Warm up DeepFace and optionally load Whisper on startup.
    """
    global DEEPFACE_READY, _INFERENCE_SEM, WHISPER_MODEL

    # ── Initialise authentication database ────────────────────────────────────
    init_db()

    _INFERENCE_SEM = asyncio.Semaphore(4)

    print(f"[startup] DeepFace backend: {DETECTOR_BACKEND}")
    print("[startup] Warming up DeepFace emotion model (may download weights on first run)...")

    loop = asyncio.get_running_loop()
    ready = await loop.run_in_executor(None, _warmup_deepface)

    if ready:
        DEEPFACE_READY = True
        print("[startup] DeepFace ready.")
    else:
        print("[startup] WARNING: DeepFace warm-up failed — check logs above.")

    # Load Whisper speech-to-text model (optional)
    whisper_model_name = os.getenv("WHISPER_MODEL", "base")
    try:
        import whisper as _whisper
        print(f"[startup] Loading Whisper model '{whisper_model_name}'...")
        WHISPER_MODEL = await loop.run_in_executor(None, _whisper.load_model, whisper_model_name)
        print(f"[startup] Whisper '{whisper_model_name}' ready.")
    except ImportError:
        print("[startup] openai-whisper not installed — audio transcription disabled.")
    except Exception as exc:
        print(f"[startup] Whisper load failed ({exc}) — audio transcription disabled.")

    yield  # server is running

    # ── Shutdown ──────────────────────────────────────────────────────────────
    print("[shutdown] Resources released.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Emotion Tracker API — DeepFace", version="1.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # tighten to your frontend origin in production
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)


def _warmup_deepface() -> bool:
    """
    Synchronous warm-up: analyse a 100×100 blank BGR image.
    Returns True if warmup succeeded, False otherwise.
    """
    try:
        dummy = np.zeros((100, 100, 3), dtype=np.uint8)
        # enforce_detection=False so it doesn't raise on a blank image
        DeepFace.analyze(
            img_path=dummy,
            actions=["emotion"],
            enforce_detection=False,
            detector_backend=DETECTOR_BACKEND,
            silent=True,
        )
        return True
    except Exception as exc:
        print(f"[warmup] Error: {exc}")
        return False


# ── Frame decoding ─────────────────────────────────────────────────────────────

def _decode_frame(b64_data: str) -> Optional[np.ndarray]:
    """
    Decode a base64-encoded image string (with or without a data URI prefix
    like "data:image/jpeg;base64,…") into a BGR numpy array.

    Returns None on any failure so the caller can gracefully return
    a "no face detected" result instead of crashing.
    """
    try:
        # Strip the optional data URI scheme prefix sent by Canvas.toDataURL()
        if "," in b64_data:
            b64_data = b64_data.split(",", 1)[1]
        raw_bytes = base64.b64decode(b64_data)
        buf       = np.frombuffer(raw_bytes, dtype=np.uint8)
        frame     = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return frame   # None if imdecode failed (corrupt / empty data)
    except Exception:
        return None


# ── DeepFace inference ────────────────────────────────────────────────────────

def _run_deepface(frame_bgr: np.ndarray) -> Tuple[Optional[Dict[str, float]], float]:
    """
    Run DeepFace emotion analysis on a BGR numpy array.

    Returns a (scores, face_confidence) tuple:
      - scores:          dict of {emotion: 0–100 %}, or None if no face found
      - face_confidence:  DeepFace's detection confidence in [0, 1]

    The face_confidence is analogous to MediaPipe's detection score in the
    custom backend.  It lets the caller weight this frame's contribution to
    the smoothing window: a marginal detection (confidence ~0.55) should
    influence the live chart less than a clear front-on face (~0.95).

    Design decisions:
    ─────────────────
    • enforce_detection=True  — raises ValueError when no face is found.
      This is cleaner than enforce_detection=False which analyses the whole
      image as a phantom "face" and returns garbage scores.

    • expand_percentage=15    — tells DeepFace to grow the detected bounding
      box by 15% on each side before cropping (matching the padding used in
      the custom backend's _detect_and_crop_face function so the two systems
      see comparable face contexts).

    • align=True              — DeepFace aligns the face to a canonical
      orientation using eye landmarks before running the emotion model.
      This materially improves accuracy for tilted or rotated heads.

    • silent=True             — suppresses DeepFace's progress bar spam.

    • When multiple faces are detected the face with the highest
      face_confidence score is used (typically the main subject).

    Emotion key names returned by DeepFace:
        'angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'
    These are identical to the custom model's class names (case-insensitive).
    """
    try:
        results = DeepFace.analyze(
            img_path=frame_bgr,
            actions=["emotion"],
            enforce_detection=True,
            detector_backend=DETECTOR_BACKEND,
            align=True,
            expand_percentage=15,
            silent=True,
        )

        # results is always a list (one entry per detected face)
        if not results:
            return None, 0.0

        # If multiple faces detected, pick the most confident one
        best = max(results, key=lambda r: r.get("face_confidence", 0.0))

        # Extract detection confidence — analogous to MediaPipe's score[0]
        det_score: float = float(best.get("face_confidence", 1.0))

        # DeepFace emotion values are already in 0–100 range
        raw_emotions: dict = best["emotion"]   # e.g. {"angry": 1.2, "happy": 84.1, …}

        # Normalise key order to match CLASS_NAMES, round to 2dp
        scores = {
            emotion: round(float(raw_emotions.get(emotion, 0.0)), 2)
            for emotion in CLASS_NAMES
        }
        return scores, det_score

    except ValueError:
        # "Face could not be detected." — normal when person looks away
        return None, 0.0
    except Exception as exc:
        # Unexpected errors (model loading failure, corrupt image, etc.)
        # Log but don't crash the WebSocket handler
        print(f"[deepface] Unexpected error during analysis: {exc}")
        return None, 0.0


def _run_deepface_tta(frame_bgr: np.ndarray) -> Tuple[Optional[Dict[str, float]], float]:
    """
    Run DeepFace with horizontal-flip Test-Time Augmentation (TTA).

    This mirrors the TTA approach in the custom backend's _run_model_tta:
    analyse the original frame and a horizontally flipped copy, then average
    the emotion scores.  Because faces are roughly left–right symmetric for
    most emotions, this reduces per-frame variance and gives more stable
    probability estimates.

    Unlike the custom backend (which batches both crops in one forward pass),
    DeepFace requires two separate analyze() calls — one for each orientation.
    This doubles the per-frame inference cost.  In practice the extra latency
    is acceptable because:
      • The semaphore already limits concurrency to 4.
      • Even with 2× overhead the worst-case RT (~200 ms on CPU) still fits
        within the 200 ms frame budget at 5 fps.
      • The stability improvement materially reduces chart flicker.

    Fallback: if the flipped frame fails to detect a face (e.g. asymmetric
    occlusion), the original-only result is returned unchanged.

    Returns (scores, avg_confidence) — same contract as _run_deepface.
    """
    scores_orig, conf_orig = _run_deepface(frame_bgr)
    if scores_orig is None:
        return None, 0.0

    # Horizontal mirror — DeepFace will re-detect and re-align the flipped face
    flipped = np.fliplr(frame_bgr).copy()
    scores_flip, conf_flip = _run_deepface(flipped)

    if scores_flip is None:
        # Face not detected in the mirror — fall back to original only
        return scores_orig, conf_orig

    # Average emission scores from both orientations
    avg_scores = {
        name: round((scores_orig[name] + scores_flip[name]) / 2.0, 2)
        for name in CLASS_NAMES
    }
    avg_conf = (conf_orig + conf_flip) / 2.0
    return avg_scores, avg_conf


# ── Sliding-window smoothing ──────────────────────────────────────────────────

def _smooth_window(
    window: deque,
    weights: deque,
    ema_alpha: float = 0.3,
) -> Dict[str, float]:
    """
    Compute smoothed emotion scores using an exponentially weighted moving average,
    where each frame is additionally weighted by its detection confidence score.

    This replaces the previous uniform box average and mirrors the custom backend's
    improved smoothing.  Two changes make a noticeable difference in real-time use:

    1. EMA (alpha=0.3): recent frames carry exponentially more weight than older
       ones, so the chart responds faster to genuine emotion changes while still
       smoothing out single-frame noise (blinks, micro-expressions).

    2. Confidence weighting: DeepFace's face_confidence (0–1) gates each frame's
       contribution.  A marginal detection (confidence ~0.55) counts less than a
       crisp front-on face (~0.95), reducing chart jitter from uncertain frames.
    """
    if not window:
        return {name: 0.0 for name in CLASS_NAMES}

    acc      = {name: 0.0 for name in CLASS_NAMES}
    acc_w    = 0.0
    momentum = 1.0 - ema_alpha

    for scores, w in zip(window, weights):          # oldest → newest
        scaled_w = ema_alpha * w
        for name in CLASS_NAMES:
            acc[name] = scaled_w * scores[name] + momentum * acc[name]
        acc_w = scaled_w + momentum * acc_w

    if acc_w < 1e-9:
        return {name: 0.0 for name in CLASS_NAMES}

    return {name: round(acc[name] / acc_w, 2) for name in CLASS_NAMES}


# ── Audio transcription ─────────────────────────────────────────────────────────────

def _transcribe_audio(audio_chunks: List[bytes]) -> Optional[str]:
    """
    Concatenate accumulated audio chunks and transcribe with Whisper.

    The chunks are raw bytes from MediaRecorder (WebM/Opus container).
    Concatenating in order is valid: the first chunk carries the container
    header; subsequent chunks are continuation data.  The assembled file is
    written to a temp directory, transcribed, then immediately deleted.

    Returns the transcript string, or None if Whisper is unavailable, no
    audio was received, or transcription fails.  Never raises — failures are
    logged and swallowed so report generation always succeeds.
    """
    if WHISPER_MODEL is None or not audio_chunks:
        return None

    tmp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
            for chunk in audio_chunks:
                f.write(chunk)
            tmp_path = f.name

        # fp16=False avoids errors on CPU-only machines (fp16 requires CUDA)
        result = WHISPER_MODEL.transcribe(tmp_path, fp16=False)
        text = result.get("text", "").strip()
        return text if text else None

    except Exception as exc:
        print(f"[whisper] Transcription error: {exc}")
        return None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str) -> None:
    """
    Main real-time analysis endpoint.

    One WebSocket connection per analysis session.  Each connection:
      • Receives "frame" messages at ~5 fps, analyses each one, sends back results
      • Receives a single "end" message when the video/recording finishes
      • Generates and sends a session summary report, then closes cleanly
    """
    # ── Authenticate via query-string token ────────────────────────────────
    token = websocket.query_params.get("token")
    user = get_user_from_token(token) if token else None
    if not user:
        await websocket.accept()
        await websocket.send_text(json.dumps({
            "type": "auth_error",
            "message": "Authentication required. Please log in first.",
        }))
        await websocket.close(code=4001)
        return

    await websocket.accept()

    SESSIONS[session_id] = {
        "window":       deque(maxlen=WINDOW_SIZE),  # up to 15 most-recent raw score dicts
        "det_weights":  deque(maxlen=WINDOW_SIZE),  # detection confidence per frame
        "history":      [],                          # (timestamp, dominant, smoothed) per frame
        "start":        time.monotonic(),
        "audio_chunks": [],                          # raw WebM bytes from audio_chunk messages
    }
    session = SESSIONS[session_id]

    try:
        while True:
            text = await websocket.receive_text()

            try:
                msg = json.loads(text)
            except json.JSONDecodeError:
                await websocket.send_text(
                    json.dumps({"type": "error", "message": "Invalid JSON."})
                )
                continue

            msg_type = msg.get("type")

            if msg_type == "frame":
                if _INFERENCE_SEM is None:
                    await websocket.send_text(
                        json.dumps({"type": "error", "message": "Server not ready."})
                    )
                    continue
                # Acquire semaphore BEFORE dispatching to thread pool so we
                # don't queue more work than TF can handle concurrently.
                async with _INFERENCE_SEM:
                    loop     = asyncio.get_running_loop()
                    response = await loop.run_in_executor(
                        None, _process_frame_sync, msg, session
                    )
                await websocket.send_text(json.dumps(response))

            elif msg_type == "audio_chunk":
                # Accumulate raw audio bytes for end-of-session transcription.
                raw = msg.get("data", "")
                if raw:
                    if "," in raw:
                        raw = raw.split(",", 1)[1]
                    try:
                        session["audio_chunks"].append(base64.b64decode(raw))
                    except Exception:
                        pass   # silently discard corrupt chunks

            elif msg_type == "end":
                loop = asyncio.get_running_loop()
                transcript = await loop.run_in_executor(
                    None, _transcribe_audio, session["audio_chunks"]
                )
                if transcript:
                    print(f"[{session_id}] Transcript ({len(transcript)} chars): "
                          f"{transcript[:80]}{'...' if len(transcript) > 80 else ''}")
                report_text = await _generate_report(session, transcript)
                await websocket.send_text(
                    json.dumps({"type": "report", "text": report_text})
                )
                break   # close connection gracefully after report is sent

            else:
                await websocket.send_text(
                    json.dumps({
                        "type":    "error",
                        "message": f"Unknown message type: {msg_type!r}",
                    })
                )

    except WebSocketDisconnect:
        pass   # client navigated away or closed the tab — silent cleanup
    finally:
        SESSIONS.pop(session_id, None)


def _process_frame_sync(msg: dict, session: dict) -> dict:
    """
    Synchronous per-frame pipeline (runs in a thread pool via run_in_executor).

    Pipeline:
        base64 string
        → BGR numpy array              (_decode_frame)
        → {emotion: score}, confidence (_run_deepface_tta — TTA with horizontal flip)
        → sliding window advance       (confidence-weighted EMA)
        → smoothed scores              (_smooth_window)
        → response dict

    This function must NOT use await — it is called from a regular thread.
    """
    timestamp: float = float(msg.get("timestamp", 0.0))
    model_tag: str   = "deepface"

    if not DEEPFACE_READY:
        return {
            "type":      "error",
            "message":   "DeepFace is not ready yet. Please wait a moment and retry.",
            "timestamp": timestamp,
            "model":     model_tag,
        }

    # ── 1. Decode the incoming base64 frame ───────────────────────────────────
    frame = _decode_frame(msg.get("data", ""))
    if frame is None:
        return {
            "type":          "result",
            "face_detected": False,
            "timestamp":     timestamp,
            "model":         model_tag,
        }

    # ── 2. Run DeepFace with TTA (original + horizontal flip, averaged) ────────
    raw_scores, det_score = _run_deepface_tta(frame)
    if raw_scores is None:
        return {
            "type":          "result",
            "face_detected": False,
            "timestamp":     timestamp,
            "model":         model_tag,
        }

    # ── 3. Advance the sliding window and compute temporally smoothed scores ───
    # Detection confidence gates how much this frame contributes to the EMA.
    session["window"].append(raw_scores)
    session["det_weights"].append(det_score)
    smoothed_scores = _smooth_window(session["window"], session["det_weights"])

    dominant   = max(smoothed_scores, key=smoothed_scores.get)
    confidence = smoothed_scores[dominant]

    # ── 4. Append to session history for end-of-session report ────────────────
    session["history"].append((timestamp, dominant, dict(smoothed_scores)))

    return {
        "type":             "result",
        "face_detected":    True,
        "raw_scores":       raw_scores,
        "smoothed_scores":  smoothed_scores,
        "dominant_emotion": dominant,
        "confidence":       confidence,
        "timestamp":        timestamp,
        "model":            model_tag,
    }


# ── Report generation ─────────────────────────────────────────────────────────

async def _generate_report(session: dict, transcript: Optional[str] = None) -> str:
    """
    Build an end-of-session emotional summary.

    Tries an LLM-powered report first (if GEMINI_API_KEY is set in the
    environment).  Falls back to a deterministic template report that is still
    detailed and human-readable — and works with zero API keys.
    """
    history: List[Tuple] = session["history"]

    if not history:
        return "No emotion data was captured during this session."

    duration: float = history[-1][0] - history[0][0] if len(history) > 1 else 0.0
    total_frames    = len(history)

    emotion_counts  = {name: 0 for name in CLASS_NAMES}
    for _, dominant, _ in history:
        emotion_counts[dominant] += 1

    emotion_pct = {
        e: round(c / total_frames * 100, 1)
        for e, c in emotion_counts.items()
    }
    sorted_emotions  = sorted(emotion_pct.items(), key=lambda x: x[1], reverse=True)
    dominant_overall = sorted_emotions[0][0]

    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        try:
            llm_text = await _llm_report(history, emotion_pct, duration, api_key, transcript)
            if llm_text:
                return llm_text
        except Exception as exc:
            print(f"[report] LLM call failed ({exc}). Falling back to template.")

    return _template_report(duration, sorted_emotions, dominant_overall, history, transcript)


async def _llm_report(
    history:     List[Tuple],
    emotion_pct: dict,
    duration:    float,
    api_key:     str,
    transcript:  Optional[str] = None,
) -> Optional[str]:
    """
    Call Google Gemini to generate a natural-language session report.

    Requires GEMINI_API_KEY in the environment.  The free tier is sufficient —
    no billing setup is required.  Get a key at aistudio.google.com/apikey.

    Override the default model with the GEMINI_MODEL env var.
    Default: gemini-2.0-flash  (free tier, fast, high quality).
    """
    try:
        import google.generativeai as genai
    except ImportError:
        print("[report] 'google-generativeai' package not installed — skipping LLM report.")
        return None

    model_name   = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel(model_name)

    n    = len(history)
    step = max(1, n // 20)
    timeline = "\n".join(
        f"  t={ts:.1f}s → {dom}"
        for ts, dom, _ in history[::step]
    )
    dist_block = "\n".join(
        f"  {e}: {p}%"
        for e, p in sorted(emotion_pct.items(), key=lambda x: x[1], reverse=True)
        if p > 0
    )

    transcript_block = (
        f"\n\nAudio transcript of what was said during the session:\n\"{transcript}\""
        if transcript else ""
    )

    prompt = (
        f"You are analysing facial emotion data captured from a video session using "
        f"the DeepFace library's pre-trained emotion recognition model.\n\n"
        f"Session duration  : {duration:.0f} seconds\n"
        f"Frames analysed   : {n}\n\n"
        f"Emotion distribution (% of frames where each emotion was dominant):\n{dist_block}\n\n"
        f"Emotional timeline (sampled every ~{step} frames):\n{timeline}"
        f"{transcript_block}\n\n"
        "Write a concise 2–3 paragraph report describing the person's emotional journey. "
        "Cover which emotions dominated, any notable transitions, and an overall characterisation."
        + (
            " Where relevant, connect the facial expressions to what was said in the transcript."
            if transcript else ""
        )
        + " Use a warm, observational tone. Address the subject directly "
        "(\"You appeared...\" / \"Throughout the session...\"). "
        "Do not make clinical diagnoses."
    )

    response = await gemini_model.generate_content_async(
        prompt,
        generation_config=genai.types.GenerationConfig(
            max_output_tokens=600,
            temperature=0.7,
        ),
    )
    return response.text.strip()


def _template_report(
    duration:         float,
    sorted_emotions:  list,
    dominant_overall: str,
    history:          List[Tuple],
    transcript:       Optional[str] = None,
) -> str:
    """
    Deterministic template-based report — no API keys required.

    Generated from the session statistics (dominant emotion, distribution,
    first-half vs second-half comparison).
    """
    top    = sorted_emotions[0]
    second = next((e for e in sorted_emotions[1:] if e[1] > 5.0),  None)
    third  = next((e for e in sorted_emotions[2:] if e[1] > 3.0),  None)

    # — Paragraph 1: distribution overview ————————————————————————————————————
    p1 = (
        f"Throughout the {duration:.0f}-second session, your facial expressions were analysed "
        f"across {len(history)} video frames using DeepFace's pre-trained emotion model. "
        f"Your predominant emotion was **{top[0]}**, detected in {top[1]}% of the frames."
    )
    if second:
        p1 += f" **{second[0].capitalize()}** was also notably present ({second[1]}%)"
        p1 += (f", followed by **{third[0]}** at {third[1]}%." if third else ".")

    # — Paragraph 2: temporal arc ──────────────────────────────────────────────
    mid         = len(history) // 2
    first_half  = [dom for _, dom, _ in history[:mid]]
    second_half = [dom for _, dom, _ in history[mid:]]

    first_dom  = max(set(first_half),  key=first_half.count)  if first_half  else dominant_overall
    second_dom = max(set(second_half), key=second_half.count) if second_half else dominant_overall

    if first_dom != second_dom:
        p2 = (
            f"The session showed a clear emotional arc: the first half was dominated by "
            f"**{first_dom}**, while the second half shifted toward **{second_dom}**. "
            "Such transitions often reflect natural changes in engagement, comfort, or "
            "response to evolving video content."
        )
    else:
        p2 = (
            f"Your emotional state remained consistently **{first_dom}** from beginning to end, "
            "indicating a stable baseline throughout — typical of someone who is focused, "
            "engaged, or at ease during the recording."
        )

    # — Paragraph 3: technical context ────────────────────────────────────────
    p3 = (
        "This analysis was performed using DeepFace with its mini_XCEPTION emotion model, "
        "which applies face detection, geometric alignment, and seven-class emotion "
        "classification to each video frame. A 3-second (15-frame) sliding window averages "
        "the per-frame scores to smooth over blinks and transitional micro-expressions. "
        "The live chart above reflects the full moment-by-moment emotional trajectory "
        "for all seven emotions — angry, disgust, fear, happy, neutral, sad, and surprise."
    )

    paragraphs = [p1, p2]
    if transcript:
        p_transcript = (
            f"**What you said:** \"{transcript}\"\n\n"
            "Your spoken words provide additional context for the facial expressions observed above."
        )
        paragraphs.append(p_transcript)
    paragraphs.append(p3)
    return "\n\n".join(paragraphs)


# ── Health check ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict:
    return {
        "status":              "ok",
        "model_loaded":        DEEPFACE_READY,
        "model_source":        "deepface",
        "detector_backend":    DETECTOR_BACKEND,
        "classes":             CLASS_NAMES,
        "window_size":         WINDOW_SIZE,
        "audio_transcription": WHISPER_MODEL is not None,
        "whisper_model":       os.getenv("WHISPER_MODEL", "base") if WHISPER_MODEL else None,
    }
