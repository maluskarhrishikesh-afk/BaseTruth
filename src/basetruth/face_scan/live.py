"""Dedicated live Face Scan session orchestration.

This module owns the non-persistent live Face Scan flow so it does not need to
reuse the much heavier Video KYC session contract. It reuses the existing
challenge engine from ``kyc.liveness`` but adds Face Scan-specific live
authenticity signals and a canonical result payload for live mode.
"""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import secrets
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from basetruth.face_scan.service import (
    FACE_SCAN_MODEL_VERSION,
    FACE_SCAN_RULES_VERSION,
    FACE_SCAN_SCHEMA_VERSION,
)
from basetruth.kyc.liveness import (
    analyze_challenge, extract_features, face_geometry_valid,
    is_face_stable, compute_face_texture_score, compute_face_brightness,
    is_yaw_motion_natural,
    MIN_FACE_DETECTION_CONFIDENCE, MIN_FACE_AREA_RATIO,
    FACE_STABLE_FRAMES_REQUIRED, FACE_STABILITY_YAW_VARIANCE_MIN,
    CHALLENGE_TIMEOUT_SECONDS,
)
from basetruth.logger import get_logger
from basetruth.vision.face import get_face_analyzer, get_mediapipe_faces
from basetruth.face_scan import narrative as _narrative_mod
import basetruth.face_scan.ml_scorer_live as _ml_scorer_live

log = get_logger(__name__)

_CV2_RESIZE = getattr(cv2, "resize")
_CV2_INTER_AREA = getattr(cv2, "INTER_AREA")
_CV2_CVT_COLOR = getattr(cv2, "cvtColor")
_CV2_COLOR_BGR2GRAY = getattr(cv2, "COLOR_BGR2GRAY")
_CV2_LAPLACIAN = getattr(cv2, "Laplacian")
_CV2_CV_64F = getattr(cv2, "CV_64F")
_CV2_CANNY = getattr(cv2, "Canny")
_CV2_IMDECODE = getattr(cv2, "imdecode")
_CV2_IMREAD_COLOR = getattr(cv2, "IMREAD_COLOR")

FACE_SCAN_LIVE_SESSION_TTL = timedelta(minutes=20)

# Hard-abort: consecutive frames with no face before the session is terminated.
# At ~10 FPS this equals ~5 seconds of continuous face absence — far beyond any
# normal camera repositioning. After this threshold the session is useless.
MAX_CONSECUTIVE_NO_FACE_FRAMES: int = 50

# Hard-abort: stability gate static-source rejections (zero yaw variance or
# artificial periodic jitter) before terminating the session. A genuine user
# struggling with lighting never reaches this count; only a persistent spoof does.
MAX_STATIC_SOURCE_REJECTIONS: int = 12

# Early-exit replay: once all_frame_history reaches this many frames, run a quick
# repeat-frame check. If repeat_frame_score already exceeds REPLAY_ABORT_SCORE_THRESHOLD,
# terminate immediately — no point letting an attacker continue probing.
REPLAY_ABORT_FRAME_THRESHOLD: int = 30
REPLAY_ABORT_SCORE_THRESHOLD: float = 80.0

# Grace period (seconds) after a challenge passes where no-face frames are treated
# as normal repositioning and do NOT count toward the consecutive-abort counter.
# Genuine users take 1-3 seconds to reorient from one turn to the next, so we
# give them a clean window before the abort clock starts counting again.
CHALLENGE_TRANSITION_GRACE_SECONDS: float = 2.5

DEFAULT_FACE_SCAN_CHALLENGES: List[str] = ["blink", "nod", "turn_left", "turn_right"]
FACE_SCAN_CHALLENGE_LABELS: Dict[str, str] = {
    "look_straight": "LOOK AT THE CAMERA",
    "blink": "BLINK ONCE",
    "turn_left": "TURN TO YOUR LEFT",
    "turn_right": "TURN TO YOUR RIGHT",
    "nod": "NOD ONCE",
}
FACE_SCAN_CHALLENGE_INSTRUCTIONS: Dict[str, str] = {
    "look_straight": "Look into the camera and hold still — your face will be captured automatically.",
    "blink": "Blink naturally — close both eyes fully and open them. We need 2 blinks.",
    "turn_left": "Slowly turn your head to YOUR LEFT. Hold that position. Then look straight ahead.",
    "turn_right": "Slowly turn your head to YOUR RIGHT. Hold that position. Then look straight ahead.",
    "nod": "Slowly look DOWN — tilt your chin toward your chest. Hold it. Then look back up.",
}

# Per-challenge timeout overrides (seconds).  Hold-and-return challenges need more
# time than instant challenges — a user who holds for 4 seconds and then returns
# uses about 6-7 seconds total, well within these limits.
_CHALLENGE_TIMEOUTS: Dict[str, float] = {
    "look_straight": 15.0,
    "blink": 15.0,
    "nod": 30.0,
    "turn_left": 30.0,
    "turn_right": 30.0,
}

_LIVE_FRAME_RUNTIME_ERRORS = (AttributeError, RuntimeError, TypeError, ValueError, OSError)

_face_lock = threading.Lock()

# Known virtual / software camera device-label substrings used for injection attacks.
# Matched case-insensitively against the video track label reported by the browser.
_VIRTUAL_CAMERA_TOKENS: frozenset = frozenset({
    "obs", "virtual", "manycam", "snap camera",
    "droidcam", "epoccam", "ivcam", "xsplit",
    "mmhmm", "iriun", "camo", "e2esim",
    "ndi virtual input", "wirecast", "logitech capture",
})

_FACE_SCAN_LIVE_PAGE_HTML = """<!DOCTYPE html>
<html lang=\"en\">
<head>
<meta charset=\"UTF-8\"/>
<meta name=\"viewport\" content=\"width=device-width,initial-scale=1.0\"/>
<title>BaseTruth · Face Scan Live</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0f172a;color:#e2e8f0;min-height:100vh;display:flex;justify-content:center;padding:1rem}
.shell{width:100%;max-width:480px}
.logo{margin:1rem 0 .8rem;text-align:center;font-size:1.35rem;font-weight:800;background:linear-gradient(135deg,#6366f1,#8b5cf6);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.card{background:#1e293b;border:1px solid #334155;border-radius:16px;padding:1.2rem 1rem;margin-bottom:.8rem}
.btn{display:block;width:100%;padding:.8rem;background:linear-gradient(135deg,#4f46e5,#6366f1);color:#fff;border:none;border-radius:10px;font-size:1rem;font-weight:700;cursor:pointer;margin-top:.8rem}
.video-wrap{position:relative;width:100%;border-radius:12px;overflow:hidden;background:#000;aspect-ratio:4/3;margin-top:.6rem}
video{width:100%;height:100%;object-fit:cover;transform:scaleX(-1)}
.oval{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);width:52%;aspect-ratio:3/4;border:3px solid rgba(99,102,241,.65);border-radius:50%;pointer-events:none}
.badge{position:absolute;top:.6rem;right:.6rem;padding:.28rem .7rem;border-radius:99px;font-size:.72rem;font-weight:700;background:rgba(148,163,184,.18);color:#cbd5e1;border:1px solid rgba(148,163,184,.3)}
.ch-card{background:linear-gradient(135deg,rgba(99,102,241,.14),rgba(139,92,246,.09));border:1px solid rgba(99,102,241,.38);border-radius:12px;padding:.9rem 1rem;margin-top:.9rem;text-align:center}
.ch-label{font-size:1.1rem;font-weight:800;color:#c4b5fd;margin-bottom:.35rem;letter-spacing:.04em}
.ch-inst{font-size:.84rem;color:#94a3b8;line-height:1.55}
.prog-wrap{background:#0f172a;border-radius:99px;height:7px;margin-top:.7rem;overflow:hidden}
.prog-fill{height:100%;border-radius:99px;background:linear-gradient(90deg,#6366f1,#8b5cf6);transition:width .35s ease}
.fb{text-align:center;font-size:.88rem;margin-top:.6rem;min-height:1.2em;color:#94a3b8}
.res-card{border-radius:12px;padding:1.1rem;border:1px solid rgba(99,102,241,.35);background:rgba(15,23,42,.85)}
.res-title{font-size:1.1rem;font-weight:800;margin-bottom:.45rem}
.res-meta{font-size:.82rem;color:#94a3b8;line-height:1.7}
.list{margin-top:.8rem;padding-left:1.05rem;color:#cbd5e1;line-height:1.65;font-size:.84rem}
</style>
</head>
<body>
<div class=\"shell\">
    <div class=\"logo\">BaseTruth Face Scan</div>
    <div id=\"intro\" class=\"card\">
        <h2 style=\"font-size:1.12rem;font-weight:700;margin-bottom:.5rem\">Live Face Authenticity Check</h2>
        <p style=\"font-size:.84rem;color:#94a3b8;line-height:1.6\">Complete the short camera challenge. BaseTruth will verify liveness, temporal consistency, and replay heuristics from your live session.</p>
        <button id=\"btn-start\" class=\"btn\">Start Live Face Scan</button>
    </div>
    <div id=\"live\" class=\"card\" style=\"display:none\">
        <div class=\"video-wrap\"><video id=\"vid\" autoplay muted playsinline></video><div class=\"oval\"></div><div class=\"badge\" id=\"face-badge\">Searching...</div></div>
        <div class=\"ch-card\"><div class=\"ch-label\" id=\"ch-label\">Please wait...</div><div class=\"ch-inst\" id=\"ch-inst\">Starting camera...</div><div class=\"prog-wrap\"><div id=\"prog-fill\" class=\"prog-fill\" style=\"width:0%\"></div></div></div>
        <div class=\"fb\" id=\"fb\"></div>
    </div>
    <div id=\"result\" class=\"card\" style=\"display:none\"></div>
</div>
<script>
const SESSION_ID = '__SESSION_ID__';
const CHALLENGES = __CHALLENGES_JSON__;
const LABELS = {look_straight:'LOOK AT THE CAMERA', blink:'BLINK ONCE', turn_left:'TURN TO YOUR LEFT', turn_right:'TURN TO YOUR RIGHT', nod:'NOD ONCE'};
const INSTR = {look_straight:'Look into the camera and hold still — your face will be captured automatically.', blink:'Blink once: slowly close both eyes completely, then open them wide.', turn_left:'Slowly turn your head to YOUR left. Hold the turn briefly, then return.', turn_right:'Slowly turn your head to YOUR right. Hold the turn briefly, then return.', nod:'Slowly nod your head down once, then look back at the camera.'};
let ws=null, stream=null, captureTimer=null, reconnectTimer=null;
const CAPTURE_MS = 100;
const RECONNECT_DELAY_MS = 1200;
const MAX_RECONNECT_ATTEMPTS = 8;
let reconnectAttempts = 0;
let sessionFinished = false;
// Sticky feedback: when the server sends feedback_sticky=true (e.g. wrong-direction
// correction), lock the displayed feedback for STICKY_HOLD_MS so the user has time
// to read it before the next frame status overwrites it.
const STICKY_HOLD_MS = 2500;
let stickyFeedbackUntil = 0;

function show(id){['intro','live','result'].forEach(x=>{const el=document.getElementById(x); if(el) el.style.display = x===id ? 'block' : 'none';});}
function feedback(msg){const el=document.getElementById('fb'); if(el) el.textContent = msg || '';}

async function start(){
    show('live');
    sessionFinished = false;
    reconnectAttempts = 0;
    try{
        stream = await navigator.mediaDevices.getUserMedia({
            video:{
                facingMode:'user',
                width:{ideal:1280},
                height:{ideal:720},
            },
            audio:false
        });
        const vid=document.getElementById('vid'); vid.srcObject=stream;
        await vid.play();
        connectSocket();
    }catch(err){
        document.getElementById('result').innerHTML = '<div class="res-card"><div class="res-title" style="color:#f87171">Camera unavailable</div><div class="res-meta">'+String(err)+'</div></div>';
        show('result');
    }
}

function socketUrl(){
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    return proto + '://' + location.host + '/api/v1/face-scan/ws/' + SESSION_ID;
}

function connectSocket(){
    ws = new WebSocket(socketUrl());
    ws.onopen = () => {
        reconnectAttempts = 0;
        const vid=document.getElementById('vid');
        // Read the actual camera device label from the active video track.
        // Virtual / software cameras (OBS, Snap Camera, etc.) report their tool name
        // here, which the server uses for virtual-camera detection.
        let deviceLabel = '';
        try {
            const tracks = stream ? stream.getVideoTracks() : [];
            if (tracks.length > 0) deviceLabel = tracks[0].label || '';
        } catch(_) {}
        ws.send(JSON.stringify({type:'meta', camera_width: vid.videoWidth || 0, camera_height: vid.videoHeight || 0, observed_fps: 1000 / CAPTURE_MS, user_agent:navigator.userAgent, platform:navigator.platform || '', device_label: deviceLabel}));
        startCapture();
    };
    ws.onmessage = (event) => { try{ handle(JSON.parse(event.data)); }catch(_err){} };
    ws.onclose = () => {
        if(sessionFinished){
            stopCapture();
            return;
        }
        if(captureTimer){clearInterval(captureTimer);captureTimer=null;}
        if(reconnectAttempts >= MAX_RECONNECT_ATTEMPTS){
            feedback('Connection interrupted. Reopen this page to continue before the session expires.');
            return;
        }
        reconnectAttempts += 1;
        feedback('Connection interrupted — reconnecting…');
        reconnectTimer = setTimeout(()=>connectSocket(), RECONNECT_DELAY_MS);
    };
}

function startCapture(){
    const canvas=document.createElement('canvas');
    const ctx=canvas.getContext('2d');
    const vid=document.getElementById('vid');
    captureTimer=setInterval(()=>{
        if(!ws || ws.readyState!==1 || !vid.videoWidth) return;
        canvas.width=640;
        canvas.height=Math.round(640 * vid.videoHeight / vid.videoWidth);
        ctx.save();
        ctx.scale(-1,1);
        ctx.drawImage(vid,-canvas.width,0,canvas.width,canvas.height);
        ctx.restore();
        canvas.toBlob(blob=>{
            if(!blob) return;
            const fr=new FileReader();
            fr.onloadend=()=>{
                const b64=String(fr.result).split(',')[1];
                // Include a high-resolution client-side timestamp so the server can
                // cross-check frame capture timing against its own receive timestamps
                if(ws && ws.readyState===1) ws.send(JSON.stringify({type:'frame', data:b64, captured_at_ms:performance.now()}));
            };
            fr.readAsDataURL(blob);
        }, 'image/jpeg', 0.82);
    }, CAPTURE_MS);
}

function stopCapture(){
    if(captureTimer){clearInterval(captureTimer);captureTimer=null;}
    if(reconnectTimer){clearTimeout(reconnectTimer);reconnectTimer=null;}
    if(stream){stream.getTracks().forEach(t=>t.stop());stream=null;}
}

function handle(msg){
    if(msg.type==='status'){
        document.getElementById('face-badge').textContent = msg.face_detected ? 'Face detected' : 'Searching...';
        const challenge = msg.challenge || CHALLENGES[Math.min(msg.challenges_completed || 0, CHALLENGES.length - 1)] || 'look_straight';
        document.getElementById('ch-label').textContent = LABELS[challenge] || 'FOLLOW THE INSTRUCTION';
        document.getElementById('ch-inst').textContent = INSTR[challenge] || msg.feedback || '';
        const total = Math.max(1, msg.total_challenges || CHALLENGES.length || 1);
        const done = msg.challenges_completed || 0;
        document.getElementById('prog-fill').style.width = ((done / total) * 100) + '%';
        // Sticky feedback: if the server flags a correction message, lock it for
        // STICKY_HOLD_MS so the very next frame status does not overwrite it.
        if(msg.feedback_sticky && msg.feedback){
            feedback(msg.feedback);
            stickyFeedbackUntil = performance.now() + STICKY_HOLD_MS;
        } else if(performance.now() >= stickyFeedbackUntil){
            feedback(msg.feedback || '');
        }
        // If the lock just expired, clear it so future messages flow normally.
        return;
    }
    if(msg.type==='processing'){
        // All challenges passed — server is now running the result computation
        // (LLM narrative call). Stop sending frames and show a waiting message.
        stopCapture();
        document.getElementById('face-badge').textContent = 'All done!';
        document.getElementById('ch-label').textContent = 'CHALLENGES COMPLETE';
        document.getElementById('ch-inst').textContent = msg.message || 'Verifying your results\u2026 Please wait.';
        document.getElementById('prog-fill').style.width = '100%';
        feedback('');
        return;
    }
    if(msg.type==='result'){
        sessionFinished = true;
        stopCapture();
        const evidence = (msg.evidence || []).map(item => '<li>' + item + '</li>').join('');
        document.getElementById('result').innerHTML = '<div class="res-card"><div class="res-title">' + (msg.verdict || 'RESULT') + '</div><div class="res-meta">Risk Score: <strong>' + Number(msg.risk_score_0_100 || 0).toFixed(1) + '/100</strong><br>Confidence: <strong>' + Number(msg.confidence_0_100 || 0).toFixed(1) + '/100</strong><br>' + (msg.honest_review || '') + '</div><ul class="list">' + evidence + '</ul></div>';
        show('result');
        return;
    }
    if(msg.type==='error'){
        sessionFinished = true;
        stopCapture();
        document.getElementById('result').innerHTML = '<div class="res-card"><div class="res-title" style="color:#f87171">Session error</div><div class="res-meta">' + (msg.message || 'Unexpected error.') + '</div></div>';
        show('result');
    }
}

document.getElementById('btn-start').addEventListener('click', start);
</script>
</body>
</html>"""


def _default_environment() -> Dict[str, Any]:
    """Return the default live-session environment envelope."""
    return {
        "platform": "web",
        "browser": None,
        "os": None,
        "camera_resolution": None,
        "observed_fps": None,
        "frame_drop_rate": None,
        "virtual_camera_suspected": False,
    }


@dataclass
class FaceScanLiveSession:
    """Holds the full state for one live Face Scan session."""

    session_id: str
    challenges: List[str]

    status: str = "waiting"  # waiting | active | completed | failed | expired
    current_challenge_idx: int = 0
    challenge_frame_history: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    all_frame_history: List[Dict[str, Any]] = field(default_factory=list)
    # Consecutive frames that passed all face-validation checks (geometry, confidence,
    # size, centering, single face).  Challenges only start once this reaches
    # FACE_STABLE_FRAMES_REQUIRED.  Resets to zero on any validation failure.
    face_stable_frames: int = 0
    # Yaw values collected during the stability window.  When the window completes,
    # variance is checked to confirm micro-movement (a real person always has small
    # involuntary oscillations; a static screen/photo has zero variance).
    face_stable_yaw_buffer: List[float] = field(default_factory=list)
    challenge_results: List[Dict[str, Any]] = field(default_factory=list)
    challenge_snapshots: List[Dict[str, Any]] = field(default_factory=list)
    # Each entry records one wrong-motion event: challenge name, what the user did
    # wrong, and the frame index at which it was detected. This is a forensic signal:
    # genuine real-time interaction produces self-corrections; scripted attacks are
    # typically robotically perfect with zero wrong attempts.
    challenge_wrong_actions: List[Dict[str, Any]] = field(default_factory=list)
    # Whether a natural blink (EAR dip) was observed during the stability window.
    # A real person blinks involuntarily; a static screen or photo never blinks.
    # Not a hard fail — a soft signal that informs the risk audit trail.
    blink_observed_in_stability: bool = False
    # Monotonic timestamp (time.monotonic()) of when the current challenge was
    # presented to the user.  Zero means no challenge has started yet.
    # Used by the timeout enforcement logic to reset slow or stuck challenges.
    challenge_started_at: float = 0.0
    # Consecutive frames where no valid face was detected. Resets to 0 the moment
    # a face IS found. Triggers hard-abort when it exceeds MAX_CONSECUTIVE_NO_FACE_FRAMES.
    consecutive_frames_without_face: int = 0
    # Count of stability-gate rejections for static-source reasons (zero yaw variance
    # or artificial jitter) across the whole session. Triggers abort at MAX_STATIC_SOURCE_REJECTIONS.
    static_source_rejections: int = 0
    # Per-challenge reaction times (ms): time from when the challenge appeared to when
    # the user completed it. Bots react near-instantly; humans take 600–3000 ms.
    challenge_reaction_times: List[float] = field(default_factory=list)
    # Monotonic timestamp of the most recent challenge advance. Used to give the
    # user a grace window between turn challenges where no-face frames are treated
    # as repositioning rather than as consecutive-abort signals.
    last_challenge_advance_at: float = 0.0
    frames_received: int = 0
    frames_without_face: int = 0
    last_live_frame_bytes: Optional[bytes] = None
    best_live_frame_bytes: Optional[bytes] = None
    last_face_box: Optional[List[int]] = None
    environment: Dict[str, Any] = field(default_factory=_default_environment)
    result: Optional[Dict[str, Any]] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc) + FACE_SCAN_LIVE_SESSION_TTL)

    def is_expired(self) -> bool:
        return datetime.now(timezone.utc) > self.expires_at

    @property
    def current_challenge(self) -> Optional[str]:
        if self.current_challenge_idx < len(self.challenges):
            return self.challenges[self.current_challenge_idx]
        return None

    @property
    def all_done(self) -> bool:
        return self.current_challenge_idx >= len(self.challenges)

    def current_frame_history(self) -> List[Dict[str, Any]]:
        key = f"ch_{self.current_challenge_idx}"
        return self.challenge_frame_history.setdefault(key, [])

    def advance_challenge(self, snapshot: Dict[str, Any] | None = None) -> None:
        """Record the current challenge as passed and move to the next one.

        The stability gate is only reset when advancing FROM a turn challenge
        (turn_left or turn_right).  These leave significant yaw residual
        (typically ±0.20) that would otherwise immediately fire the wrong-
        direction guard on the very next challenge before the user has had any
        chance to act.

        For still challenges (look_straight, blink, nod) the yaw shift is
        negligible.  Resetting the gate for these transitions forces an
        unnecessary re-stabilisation window: the user starts performing the
        next motion during the gate (which rejects those frames), and then
        has to perform it again once the challenge finally opens — this is
        the direct cause of needing 4-5 attempts per challenge.
        """
        # Capture the challenge that just passed before incrementing the index
        # so we can decide whether the stability gate needs a reset below.
        completed_challenge = self.current_challenge

        if completed_challenge is not None:
            self.challenge_results.append(
                {
                    "index": self.current_challenge_idx,
                    "challenge": completed_challenge,
                    "passed": True,
                }
            )
            if snapshot:
                self.challenge_snapshots.append(snapshot)

        self.current_challenge_idx += 1
        self.challenge_frame_history[f"ch_{self.current_challenge_idx}"] = []

        # Only reset the stability gate after a turn challenge.  Turn challenges
        # leave the head pointed sideways (large yaw residual).  The gate
        # requires |yaw| <= FACE_STABILITY_YAW_MAX (0.12), so it forces the
        # user back to neutral before the next challenge opens — preventing the
        # wrong-direction guard from firing on the residual yaw of the previous
        # turn.  Resetting for non-turn challenges causes needless re-stabilisation
        # and is the primary reason challenges used to take 4-5 attempts.
        _TURN_CHALLENGES = frozenset({"turn_left", "turn_right"})
        if completed_challenge in _TURN_CHALLENGES:
            self.face_stable_frames = 0
            self.face_stable_yaw_buffer.clear()
            self.blink_observed_in_stability = False

    def to_status_dict(self) -> Dict[str, Any]:
        """Return the public live Face Scan session contract."""
        return {
            "session_id": self.session_id,
            "status": self.status,
            "challenges": self.challenges,
            "current_challenge": self.current_challenge,
            "current_instruction": FACE_SCAN_CHALLENGE_INSTRUCTIONS.get(self.current_challenge or "", ""),
            "challenges_completed": self.current_challenge_idx,
            "total_challenges": len(self.challenges),
            "challenge_results": self.challenge_results,
            "result": self.result,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "environment": self.environment,
            "best_frame_available": bool(self.best_live_frame_bytes),
            "challenge_snapshots_available": bool(self.challenge_snapshots),
        }


class FaceScanLiveSessionStore:
    """Thread-safe in-memory store for Face Scan live sessions."""

    def __init__(self) -> None:
        self._sessions: Dict[str, FaceScanLiveSession] = {}
        self._lock = threading.Lock()

    def create(self, challenges: List[str]) -> FaceScanLiveSession:
        sid = secrets.token_urlsafe(16)
        session = FaceScanLiveSession(session_id=sid, challenges=challenges)
        with self._lock:
            self._sessions[sid] = session
        return session

    def get(self, session_id: str) -> Optional[FaceScanLiveSession]:
        with self._lock:
            session = self._sessions.get(session_id)
            if session and session.is_expired() and session.status not in ("completed", "failed"):
                session.status = "expired"
            return session

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    """Clamp a numeric value to a bounded range."""
    return max(low, min(high, float(value)))


def _scale_risk(value: float, low: float, high: float, inverse: bool = False) -> float:
    """Map a raw measurement to a 0-100 risk scale."""
    if high <= low:
        return 0.0
    if inverse:
        if value <= low:
            return 100.0
        if value >= high:
            return 0.0
        return _clamp((high - value) * 100.0 / (high - low))

    if value <= low:
        return 0.0
    if value >= high:
        return 100.0
    return _clamp((value - low) * 100.0 / (high - low))


def _average_hash(gray: np.ndarray, size: int = 8) -> str:
    """Compute a small average-hash fingerprint for replay detection."""
    resized = _CV2_RESIZE(gray, (size, size), interpolation=_CV2_INTER_AREA)
    mean_val = float(resized.mean())
    bits = "".join("1" if px >= mean_val else "0" for px in resized.flatten())
    width = max(1, len(bits) // 4)
    return format(int(bits, 2), f"0{width}x")


def _perceptual_hash(gray: np.ndarray, size: int = 8) -> str:
    """Compute a DCT-based perceptual hash (pHash) of a grayscale face crop.

    Unlike average hash, pHash uses the Discrete Cosine Transform to capture the
    dominant frequency structure of the image. Because it works in the frequency
    domain rather than on raw pixel values, it is robust to tiny noise additions,
    small brightness shifts, and minor pixel shifts — all tricks an attacker might
    use to defeat average-hash similarity checks while replaying the same recording.
    The top-left 8×8 block of a 32×32 DCT contains the lowest (most stable) spatial
    frequencies, so two visually identical frames produce almost the same hash even
    after re-encoding or adding imperceptible noise.
    """
    # Resize to 32×32 first to give the DCT room to separate low from high frequencies.
    resized = _CV2_RESIZE(gray, (32, 32), interpolation=_CV2_INTER_AREA).astype(np.float32)
    # 2-D DCT: concentrates signal energy in the top-left corner.
    dct_block = cv2.dct(resized)
    # Keep only the top-left size×size block (lowest frequencies).
    dct_low = dct_block[:size, :size].flatten()
    # Skip index 0 (DC coefficient — raw average brightness) which swamps everything else.
    mean_val = float(np.mean(dct_low[1:]))
    bits = "".join("1" if v >= mean_val else "0" for v in dct_low)
    width = max(1, len(bits) // 4)
    return format(int(bits, 2), f"0{width}x")


def _difference_hash(gray: np.ndarray, size: int = 8) -> str:
    """Compute a difference hash (dHash) of a grayscale face crop.

    dHash encodes the horizontal gradient pattern of the image: each bit records
    whether a pixel is brighter or darker than its right-side neighbour. This makes
    it sensitive to structural / spatial differences while being largely insensitive
    to uniform brightness changes or mild compression artefacts. Combining dHash
    with pHash gives a second independent structural check so an attacker would need
    to simultaneously defeat both to avoid detection.
    """
    # 9 wide × 8 tall: comparing each pixel to its right neighbour produces 8×8 = 64 bits.
    resized = _CV2_RESIZE(gray, (size + 1, size), interpolation=_CV2_INTER_AREA)
    # Bit is 1 where the left pixel is brighter than its right neighbour.
    diff = resized[:, :-1] > resized[:, 1:]
    bits = "".join("1" if b else "0" for b in diff.flatten())
    width = max(1, len(bits) // 4)
    return format(int(bits, 2), f"0{width}x")


def _hamming_distance(hash_a: str, hash_b: str) -> int:
    """Return the Hamming distance between two hex-encoded average hashes."""
    a_bits = bin(int(hash_a, 16))[2:].zfill(len(hash_a) * 4)
    b_bits = bin(int(hash_b, 16))[2:].zfill(len(hash_b) * 4)
    return sum(ch_a != ch_b for ch_a, ch_b in zip(a_bits, b_bits))


def _is_repeat_frame_pair(frame_a: Dict[str, Any], frame_b: Dict[str, Any]) -> bool:
    """Return True if two frames are suspiciously similar across all available hash types.

    We check three independent fingerprints: average hash (aHash), perceptual hash
    (pHash), and difference hash (dHash). Each one attacks the similarity check from
    a different angle. An attacker who adds tiny noise to defeat aHash will still be
    caught by pHash (frequency domain) and dHash (gradient domain). A pair counts as
    a suspected repeat if pHash agrees when available, or if aHash alone matches when
    the newer hashes are absent (backward-compatible fallback for existing sessions
    and test fixtures that only carry the legacy frame_hash field).
    """
    ahash_a = str(frame_a["frame_hash"])
    ahash_b = str(frame_b["frame_hash"])
    a_repeat = _hamming_distance(ahash_a, ahash_b) <= 1

    phash_a = str(frame_a.get("frame_hash_phash", ""))
    phash_b = str(frame_b.get("frame_hash_phash", ""))
    has_phash = bool(phash_a and phash_b)
    p_repeat = _hamming_distance(phash_a, phash_b) <= 1 if has_phash else False

    dhash_a = str(frame_a.get("frame_hash_dhash", ""))
    dhash_b = str(frame_b.get("frame_hash_dhash", ""))
    has_dhash = bool(dhash_a and dhash_b)
    d_repeat = _hamming_distance(dhash_a, dhash_b) <= 1 if has_dhash else False

    if has_phash:
        # pHash is the most noise-robust signal; trust it as the primary check.
        # If pHash agrees, the frames are very likely identical in content.
        # Fall back to aHash+dHash consensus when pHash alone doesn't fire.
        return p_repeat or (a_repeat and (not has_dhash or d_repeat))
    # Legacy path: only aHash available (test fixtures, pre-upgrade sessions).
    return a_repeat


def _clip_box(box: np.ndarray, img: np.ndarray) -> tuple[int, int, int, int]:
    """Clip a bounding box to the image size."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box.astype(int).tolist()
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))
    return x1, y1, x2, y2


def _attach_ear_if_possible(img: np.ndarray, faces: List[Any]) -> None:
    """Attach MediaPipe EAR values to face objects when available."""
    try:
        mp_faces = get_mediapipe_faces(img)
    except _LIVE_FRAME_RUNTIME_ERRORS as exc:  # noqa: BLE001
        log.debug("Live EAR attachment failed; continuing without EAR.", extra={"error": str(exc)})
        return

    if not mp_faces:
        return

    ear = getattr(mp_faces[0], "ear", None)
    if ear is None:
        return

    for face in faces:
        face.ear = ear


def _detect_faces(img: np.ndarray, attach_ear: bool = True) -> List[Any]:
    """Detect faces for the live Face Scan frame.

    attach_ear controls whether MediaPipe is run to compute EAR (Eye Aspect Ratio).
    EAR is only needed for the blink challenge. Skipping it for other challenges halves
    per-frame processing time because it avoids running a second detector pipeline.
    """
    try:
        app = get_face_analyzer()
    except ImportError:
        log.info("Face Scan live analysis falling back to MediaPipe because InsightFace is unavailable.")
    else:
        try:
            with _face_lock:
                faces = list(app.get(img))
            if faces:
                if attach_ear:
                    _attach_ear_if_possible(img, faces)
                return faces
        except _LIVE_FRAME_RUNTIME_ERRORS as exc:  # noqa: BLE001
            log.warning(
                "Face Scan live InsightFace detection failed; using MediaPipe fallback.",
                extra={"error": str(exc)},
            )

    try:
        return list(get_mediapipe_faces(img))
    except _LIVE_FRAME_RUNTIME_ERRORS as exc:  # noqa: BLE001
        log.error("Face Scan live detection failed in both detectors.", extra={"error": str(exc)})
        return []


def _extract_live_frame_metrics(img: np.ndarray, face: Any, frame_index: int) -> Dict[str, Any]:
    """Extract liveness features plus extra live-authenticity metrics from one frame."""
    features = dict(extract_features(face))
    x1, y1, x2, y2 = _clip_box(np.asarray(face.bbox), img)
    gray = _CV2_CVT_COLOR(img, _CV2_COLOR_BGR2GRAY)
    face_crop_gray = gray[y1:y2, x1:x2]
    frame_area = float(max(img.shape[0] * img.shape[1], 1))
    bbox_area_ratio = float(max((x2 - x1) * (y2 - y1), 1) / frame_area)
    laplacian_var = float(_CV2_LAPLACIAN(face_crop_gray, _CV2_CV_64F).var())
    brightness_mean = float(np.mean(face_crop_gray))
    edge_density = float(_CV2_CANNY(face_crop_gray, 80, 160).mean() / 255.0)
    frame_hash = _average_hash(face_crop_gray)
    frame_hash_phash = _perceptual_hash(face_crop_gray)
    frame_hash_dhash = _difference_hash(face_crop_gray)
    det_score = float(getattr(face, "det_score", 0.95) or 0.95)

    metrics: Dict[str, Any] = {
        **features,
        "frame_index": frame_index,
        "laplacian_var": laplacian_var,
        "brightness_mean": brightness_mean,
        "edge_density": edge_density,
        "bbox_area_ratio": bbox_area_ratio,
        "frame_hash": frame_hash,
        "frame_hash_phash": frame_hash_phash,
        "frame_hash_dhash": frame_hash_dhash,
        "detector_confidence": det_score,
        "face_box": [x1, y1, x2, y2],
    }

    # Run FFT screen-frequency analysis on every 5th frame only to limit CPU usage.
    # A real face produces low mid-frequency energy; a filmed screen shows periodic
    # Moiré peaks from the display pixel grid (see _compute_fft_grid_peak_ratio).
    if frame_index % 5 == 0:
        metrics["fft_grid_peak_ratio"] = _compute_fft_grid_peak_ratio(face_crop_gray)

    return metrics


def handle_live_meta(session: FaceScanLiveSession, data: Dict[str, Any]) -> None:
    """Capture browser-provided environment metadata for the live session."""
    width = data.get("camera_width")
    height = data.get("camera_height")
    if isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0:
        session.environment["camera_resolution"] = [width, height]

    fps = data.get("observed_fps")
    if isinstance(fps, (int, float)) and fps > 0:
        session.environment["observed_fps"] = round(float(fps), 2)

    ua = str(data.get("user_agent") or "")
    if ua:
        session.environment["browser"] = ua[:180]

    platform = str(data.get("platform") or "")
    if platform:
        session.environment["os"] = platform[:80]

    # Check the video track label for known virtual/software camera tool names.
    # Any match flags the session for review — these tools are commonly used to
    # inject pre-recorded or deepfake video instead of a live camera stream.
    device_label = str(data.get("device_label") or "")
    label_lower = device_label.lower()
    if any(token in label_lower for token in _VIRTUAL_CAMERA_TOKENS):
        session.environment["virtual_camera_suspected"] = True


def _build_abort_result(session: "FaceScanLiveSession", reason: str) -> Dict[str, Any]:
    """Build and store a terminal LIVENESS_FAILED result for an aborted live session.

    Called when a hard-abort condition fires mid-session (consecutive no-face frames,
    persistent static-source, or early replay detection). Sets session.result and
    session.status so reconnect logic in the API returns the right response.
    """
    result: Dict[str, Any] = {
        "filename": f"face_scan_live_{session.session_id}.jpg",
        "scan_type": "face_scan",
        "mode": "live",
        "schema_version": FACE_SCAN_SCHEMA_VERSION,
        "verdict": "LIVENESS_FAILED",
        "risk_score_0_100": 100.0,
        "confidence_0_100": 0.0,
        "confidence_reason": reason,
        "overall_explanation": "Session terminated: " + reason,
        "honest_review": reason + ". Please start a new session.",
        "evidence": ["Session terminated early: " + reason + "."],
        "trace": {
            "decision_trace_id": f"fs_live_{uuid.uuid4().hex[:12]}",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "processing_time_ms": int((time.time() - session.created_at.timestamp()) * 1000),
            "rules_version": FACE_SCAN_RULES_VERSION,
            "model_version": FACE_SCAN_MODEL_VERSION,
        },
        "environment": session.environment,
        "checks": {},
        "narrative_pending": False,
    }
    session.result = result
    session.status = "failed"
    return {"type": "result", **result}


def _no_face_tick(session: "FaceScanLiveSession") -> Optional[Dict[str, Any]]:
    """Increment no-face counters and hard-abort the session if absence is too long.

    Increments both the total frames_without_face counter (used in end-of-session
    confidence scoring) and the consecutive_frames_without_face counter (used for
    the hard-abort guard). The consecutive counter is reset to 0 in
    process_live_frame_message the moment a face IS detected.

    Returns a terminal abort dict when the threshold is exceeded, or None to let
    the caller continue with its normal no-face status response.
    """
    session.frames_without_face += 1
    # During the challenge transition grace period, no-face frames are expected —
    # the user is physically moving from one turn position to the next. Do not count
    # them toward the consecutive-abort clock; just let them pass silently.
    if (
        session.last_challenge_advance_at > 0.0
        and (time.monotonic() - session.last_challenge_advance_at) < CHALLENGE_TRANSITION_GRACE_SECONDS
    ):
        return None
    session.consecutive_frames_without_face += 1
    if session.consecutive_frames_without_face >= MAX_CONSECUTIVE_NO_FACE_FRAMES:
        log.warning(
            "Live session aborted: consecutive no-face frames exceeded threshold.",
            extra={
                "session_id": session.session_id,
                "consecutive_no_face": session.consecutive_frames_without_face,
                "threshold": MAX_CONSECUTIVE_NO_FACE_FRAMES,
            },
        )
        return _build_abort_result(
            session,
            "No face detected for too long \u2014 ensure you are well-lit and centred in the oval",
        )
    return None


def _compute_temporal_consistency(history_groups: List[List[Dict[str, Any]]]) -> Dict[str, float]:
    """Estimate frame-to-frame motion consistency within completed challenge windows."""
    usable_groups = [group for group in history_groups if len(group) >= 3]
    if not usable_groups:
        flattened = [frame for group in history_groups for frame in group]
        if len(flattened) < 6:
            return {"score_0_100": 45.0, "yaw_jerk": 0.0, "pitch_jerk": 0.0, "nose_jitter": 0.0}
        usable_groups = [flattened]

    weighted_yaw_jerk = 0.0
    weighted_pitch_jerk = 0.0
    weighted_nose_jitter = 0.0
    total_weight = 0.0

    for group in usable_groups:
        yaws = np.asarray([float(frame["yaw"]) for frame in group], dtype=np.float32)
        pitches = np.asarray([float(frame["pitch"]) for frame in group], dtype=np.float32)
        noses = np.asarray([float(frame["nose_rel_x"]) for frame in group], dtype=np.float32)

        yaw_jerk = float(np.mean(np.abs(np.diff(np.diff(yaws))))) if len(yaws) >= 3 else 0.0
        pitch_jerk = float(np.mean(np.abs(np.diff(np.diff(pitches))))) if len(pitches) >= 3 else 0.0
        nose_jitter = float(np.mean(np.abs(np.diff(np.diff(noses))))) if len(noses) >= 3 else 0.0
        weight = float(len(group))

        weighted_yaw_jerk += yaw_jerk * weight
        weighted_pitch_jerk += pitch_jerk * weight
        weighted_nose_jitter += nose_jitter * weight
        total_weight += weight

    yaw_jerk = weighted_yaw_jerk / max(total_weight, 1.0)
    pitch_jerk = weighted_pitch_jerk / max(total_weight, 1.0)
    nose_jitter = weighted_nose_jitter / max(total_weight, 1.0)
    score = _clamp((yaw_jerk * 240.0) + (pitch_jerk * 240.0) + (nose_jitter * 320.0))
    return {
        "score_0_100": round(score, 2),
        "yaw_jerk": round(yaw_jerk, 4),
        "pitch_jerk": round(pitch_jerk, 4),
        "nose_jitter": round(nose_jitter, 4),
    }


def _compute_replay_heuristics(history: List[Dict[str, Any]]) -> Dict[str, float]:
    """Estimate replay risk from repeated frames and brightness flicker.

    Frames are compared at a ~300 ms gap (not consecutive) using the server-side
    receive timestamps recorded on every processed frame. At typical webcam FPS
    (8–15 fps), consecutive frames are naturally near-identical for any real person
    who is momentarily still; waiting 300 ms gives organic micro-movements time to
    accumulate and create measurable visual change.

    The Hamming threshold is tightened to ≤ 1 (≥ 98.4% bit-identical). Real people
    produce at least 2–4 bit flips over 300 ms from breathing, micro-expressions, and
    detector landmark noise. A looped or injected pre-recorded stream produces Hamming 0
    nearly every pair because the identical encoded frames repeat on a fixed clock.
    """
    if len(history) < 4:
        return {"score_0_100": 10.0, "repeat_frame_score": 0.0, "flicker_score": 0.0, "brightness_instability": 0.0}

    hashes = [str(frame["frame_hash"]) for frame in history]  # kept for flicker / future use
    times = [float(f.get("server_recv_mono", 0.0)) for f in history]
    has_timestamps = any(t != 0.0 for t in times)

    TARGET_DELTA_S = 0.30  # compare frames ~300 ms apart
    low_distance_pairs = 0
    n_pairs = 0

    if has_timestamps:
        # For each frame, find the closest frame approximately 300 ms later.
        # Search up to 20 frames ahead to handle varying FPS without missing the target.
        for i in range(len(history) - 1):
            t_target = times[i] + TARGET_DELTA_S
            best_j = -1
            best_diff = float("inf")
            for j in range(i + 1, min(i + 20, len(history))):
                diff = abs(times[j] - t_target)
                if diff < best_diff:
                    best_diff = diff
                    best_j = j
                if times[j] > t_target + TARGET_DELTA_S:
                    break  # overshot by a full extra step — stop searching
            if best_j != -1 and best_diff <= 0.20:  # within ±200 ms of the 300 ms target
                n_pairs += 1
                # Use all available hash types so noise-added replays are still caught.
                if _is_repeat_frame_pair(history[i], history[best_j]):
                    low_distance_pairs += 1
    else:
        # No timestamps available: fall back to comparing every 3rd frame
        # (approximates ~375 ms at 8 FPS, ~200 ms at 15 FPS)
        step = 3
        for idx in range(len(history) - step):
            n_pairs += 1
            if _is_repeat_frame_pair(history[idx], history[idx + step]):
                low_distance_pairs += 1

    if n_pairs == 0:
        repeat_frame_score = 0.0
    else:
        repeat_ratio = low_distance_pairs / n_pairs
        repeat_frame_score = repeat_ratio * 100.0

    brightness = np.asarray([float(frame["brightness_mean"]) for frame in history], dtype=np.float32)
    brightness_instability = float(np.std(np.diff(brightness))) if len(brightness) >= 2 else 0.0
    flicker_score = _scale_risk(brightness_instability, 3.0, 14.0)

    score = _clamp((0.7 * repeat_frame_score) + (0.3 * flicker_score))
    return {
        "score_0_100": round(score, 2),
        "repeat_frame_score": round(repeat_frame_score, 2),
        "flicker_score": round(flicker_score, 2),
        "brightness_instability": round(brightness_instability, 4),
    }


def _compute_fft_grid_peak_ratio(gray_crop: np.ndarray) -> float:
    """Compute the spectral peak-concentration in the mid-frequency ring of a face crop.

    When a camera films a digital screen (phone, laptop, tablet), the LCD/OLED pixel
    grid produces a Moiré interference pattern. In the 2D FFT of the face region, this
    shows up as 2–4 bright, localised spots in the mid-frequency ring (corresponding to
    the horizontal and vertical pixel pitches of the screen).

    A real human face has organic, irregular texture. Even JPEG's 8×8 DCT compression
    blocks add energy in this same mid-frequency region — but that energy is spread
    broadly across many ring pixels, not concentrated in a few sharp spikes.

    We therefore measure PEAK CONCENTRATION: what fraction of the mid-frequency ring
    energy is held by just the top 2% of ring pixels? A filmed screen produces very
    concentrated energy (ratio ≈ 0.35–0.70); organic skin or JPEG artefacts produce
    diffusely spread energy (ratio ≈ 0.05–0.18).

    A Hanning window is applied before the FFT to suppress boundary leakage.
    """
    if gray_crop.size < 400:
        # Face crop too small for a meaningful frequency analysis
        return 0.0
    try:
        # Resize to 64×64 for fast, consistent FFT bin resolution across all frame sizes
        resized = _CV2_RESIZE(gray_crop, (64, 64), interpolation=_CV2_INTER_AREA).astype(np.float32)
        # Apply Hanning window to suppress spectral leakage from the image border
        win = np.outer(np.hanning(64), np.hanning(64))
        f = np.fft.fft2(resized * win)
        fshift = np.fft.fftshift(f)
        mag = np.abs(fshift)
        # Zero out the DC component (centre, always the largest single peak)
        cx, cy = 32, 32
        mag[cy - 3:cy + 3, cx - 3:cx + 3] = 0.0
        # Mid-frequency ring: radii 6–22 pixels in the 64×64 transform.
        # This range matches LCD/OLED pixel-grid spatial periods at typical selfie distances.
        y_g, x_g = np.ogrid[:64, :64]
        dist = np.sqrt((y_g - cy) ** 2 + (x_g - cx) ** 2)
        ring = (dist >= 6) & (dist <= 22)
        ring_pixels = mag[ring]
        ring_total = float(ring_pixels.sum())
        if ring_total < 1e-6:
            return 0.0
        # Peak concentration: fraction of ring energy in the top 2% of ring pixels.
        # Moiré from a screen concentrates energy in ~4 bright spots → high ratio.
        # Organic skin texture or JPEG blocks spread energy across all ring pixels → low ratio.
        sorted_ring = np.sort(ring_pixels)[::-1]
        top_n = max(1, int(len(sorted_ring) * 0.02))  # top 2% of ring pixels
        peak_concentration = float(sorted_ring[:top_n].sum() / ring_total)
        return float(np.clip(peak_concentration, 0.0, 1.0))
    except Exception:  # noqa: BLE001
        return 0.0


def _compute_screen_frequency_analysis(history: List[Dict[str, Any]]) -> Dict[str, float]:
    """Aggregate per-frame FFT grid-peak ratios to estimate screen-replay risk.

    Each frame contributes one `fft_grid_peak_ratio` value (computed in
    `_extract_live_frame_metrics` on every 5th frame to save CPU). We average
    those values across the session. A high mean indicates that most frames showed
    the periodic spectral signature of a digital screen; a low mean is consistent
    with a real human face filmed by a camera.

    Threshold basis: organic skin texture produces mean ratios of ~0.14–0.22;
    screen replays typically produce ~0.28–0.50, depending on camera resolution
    and the angle at which the screen is being filmed.
    """
    peaks = [float(f["fft_grid_peak_ratio"]) for f in history if "fft_grid_peak_ratio" in f]
    if not peaks:
        # No FFT measurements available — return neutral (no additional risk added)
        return {"score_0_100": 0.0, "mean_fft_grid_peak": 0.0}
    mean_peak = float(np.mean(peaks))
    # _scale_risk maps [0.20, 0.40] → [0, 100]; below 0.20 = organic / no screen risk.
    # These thresholds are calibrated for the peak-concentration metric:
    # organic faces + JPEG blocks: ~0.05–0.18; filmed screen Moiré: ~0.35–0.70.
    score = _scale_risk(mean_peak, 0.20, 0.40)
    return {
        "score_0_100": round(score, 2),
        "mean_fft_grid_peak": round(mean_peak, 4),
    }


def _compute_saccade_analysis(history: List[Dict[str, Any]]) -> Dict[str, float]:
    """Detect eye micro-jitter (saccades) to distinguish a live face from a static replay.

    Human eyes are never perfectly still — they continuously make tiny involuntary
    micro-movements called saccades (roughly every 100–200 ms). These appear as small,
    random fluctuations in the eye landmark positions even when the head is not moving.
    In a static photo, printed mask, or loop-replayed video, the eye positions are
    either frozen (near-zero variance) or repeat the same smooth pattern every cycle.

    We measure how much each eye landmark moves between consecutive frames, after
    removing the head-motion trend by linear detrending. The residual variance
    (standard deviation) captures the micro-jitter signal. Very low variance means
    the eyes are suspiciously still — consistent with a non-live source.

    Thresholds are tuned for InsightFace / MediaPipe 5-point landmarks at 640px
    width and 5–15 FPS (where detector noise alone contributes ~0.003–0.008 std).
    """
    # Need at least 6 frames to estimate variance reliably
    usable = [f for f in history if "left_eye_x_norm" in f and "right_eye_x_norm" in f]
    if len(usable) < 6:
        # Not enough eye data — return a neutral score that adds no risk
        return {"score_0_100": 20.0, "mean_eye_jitter": 0.0, "eye_stillness_risk": 0.0}

    # Collect normalised eye coordinates across the session
    lx = np.array([float(f["left_eye_x_norm"])  for f in usable], dtype=np.float32)
    rx = np.array([float(f["right_eye_x_norm"]) for f in usable], dtype=np.float32)
    ly = np.array([float(f["left_eye_y_norm"])  for f in usable], dtype=np.float32)
    ry = np.array([float(f["right_eye_y_norm"]) for f in usable], dtype=np.float32)

    # Remove the linear head-motion trend from each coordinate so that only the
    # micro-jitter component remains (head turns / nods move eyes as a rigid block)
    t = np.linspace(0.0, 1.0, len(usable))
    for coords in (lx, rx, ly, ry):
        p = np.polyfit(t, coords, 1)
        coords -= np.polyval(p, t)

    # Mean standard deviation across all four detrended eye-coordinate series
    mean_std = float(np.mean([lx.std(), rx.std(), ly.std(), ry.std()]))

    # Eye stillness risk: std < 0.0005 → frozen (likely static photo)
    # Normal detector noise + saccades for a live face: std > 0.003–0.005
    stillness_risk = _scale_risk(mean_std, 0.0005, 0.004, inverse=True)
    # Final score is dominated by stillness (too-still eyes = high risk)
    score = _clamp(stillness_risk)
    return {
        "score_0_100": round(score, 2),
        "mean_eye_jitter": round(mean_std, 5),
        "eye_stillness_risk": round(stillness_risk, 2),
    }


def _compute_frame_timing_jitter(history: List[Dict[str, Any]]) -> Dict[str, float]:
    """Detect suspiciously uniform frame delivery intervals using server-side timestamps.

    Real browsers on real hardware deliver frames with organic timing variance: the
    browser event loop, OS scheduling, JPEG encoding time, and WebSocket buffering all
    add unpredictable, variable delays. The Coefficient of Variation (CV = std / mean)
    of inter-frame intervals for a genuine session is typically 0.15–0.50.

    Replay injection tools (OBS virtual camera, pre-recorded video injectors) send
    frames at a perfectly uniform rate matching their configured capture interval.
    This produces a CV below 0.05 — the frames arrive with metronomic regularity that
    no real browser produces.

    Server-side monotonic clock timestamps are used (not client-reported) to prevent
    timestamp spoofing from the client.
    """
    times = [float(f["server_recv_mono"]) for f in history if "server_recv_mono" in f]
    if len(times) < 6:
        # Too few frames to measure timing statistics reliably
        return {"score_0_100": 0.0, "interval_std_ms": 0.0, "interval_cv": 0.0}

    intervals = np.diff(np.array(times)) * 1000.0  # convert seconds → milliseconds
    # Discard intervals longer than 2 s — these reflect reconnects or tab switches,
    # not the baseline delivery rhythm, and would inflate the variance artificially
    intervals = intervals[intervals < 2000.0]
    if len(intervals) < 5:
        return {"score_0_100": 0.0, "interval_std_ms": 0.0, "interval_cv": 0.0}

    mean_interval = float(np.mean(intervals))
    std_interval = float(np.std(intervals))
    # CV = std / mean; high CV = organic (real); low CV = robotic (injection tool)
    cv = std_interval / max(mean_interval, 1.0)

    # Risk: low CV (< 0.03) = almost perfect timer = injection tool
    # _scale_risk with inverse=True maps [0.03, 0.15] → 100→0; below 0.03 = max risk
    timing_risk = _scale_risk(cv, 0.03, 0.15, inverse=True)
    return {
        "score_0_100": round(timing_risk, 2),
        "interval_std_ms": round(std_interval, 2),
        "interval_cv": round(cv, 4),
    }


def _compute_depth_consistency(session: "FaceScanLiveSession") -> Dict[str, float]:
    """Verify 3D face depth by checking that eye separation decreases during head turns.

    When a real 3D face turns left or right, the nose moves toward the turned side
    and the far eye is progressively occluded behind the nose. This reduces the
    visible (apparent) interocular distance (IOD) relative to the face bounding-box
    width — the effect of perspective projection on a three-dimensional object.

    A flat 2D photo or printed mask that is physically rotated has no depth: both
    'eyes' are on the same plane, so their apparent separation stays constant no
    matter how much the object is tilted. The IOD / bbox-width ratio remains flat.

    We compute the Pearson correlation between |yaw| and IOD/bbox ratio across all
    frames captured during turn challenges. A real 3D face yields a NEGATIVE
    correlation (more yaw → smaller IOD). A flat source yields a correlation near
    zero or positive (IOD does not decrease with yaw).

    This check only activates when turn challenges are present and enough frames
    (≥ 6 with IOD data) were captured during those challenges.
    """
    # Gather frames from every turn challenge that was successfully passed
    turn_frames: List[Dict[str, Any]] = []
    for result in session.challenge_results:
        if result.get("challenge") in ("turn_left", "turn_right") and result.get("passed"):
            group = session.challenge_frame_history.get(f"ch_{result['index']}", [])
            turn_frames.extend(group)

    # Only use frames that have the interocular_px_norm field (set via extract_features)
    usable = [f for f in turn_frames if "interocular_px_norm" in f and "yaw" in f]
    if len(usable) < 6:
        # No turn challenges, or not enough landmark data — skip with neutral result
        return {"score_0_100": 0.0, "iod_yaw_correlation": 0.0, "flat_face_risk": 0.0}

    yaws = np.array([abs(float(f["yaw"]))              for f in usable], dtype=np.float32)
    iods = np.array([float(f["interocular_px_norm"])   for f in usable], dtype=np.float32)

    if float(iods.std()) < 1e-5:
        # IOD never changed at all during the turn — very suspicious (flat source)
        return {"score_0_100": 80.0, "iod_yaw_correlation": 0.0, "flat_face_risk": 80.0}

    # Pearson correlation: negative = 3D face (IOD shrinks as yaw grows)
    corr = float(np.corrcoef(yaws, iods)[0, 1])

    # Map the correlation [-1, +1] to a flat-face risk [0, 100]:
    #   corr  ≤ -0.50 → strong 3D depth behaviour → risk = 0
    #   corr  ≥  0.00 → IOD flat or growing with yaw → risk = 100
    #
    # Calibrated against real-session data:
    #   Real human faces: iod_yaw_corr typically -0.40 to -0.99 → risk 0
    #   Plastic doll:     iod_yaw_corr ≈ -0.12 (shallow 3D)     → risk ~77
    #   Flat photo/mask:  iod_yaw_corr ≈  0.00 to +0.30         → risk 100
    #
    # Previous thresholds (0.70, 1.20) had the safe zone as corr ≤ -0.30, which
    # was too lenient: a plastic doll with corr = -0.12 only scored ~37, staying
    # below the SUSPICIOUS verdict threshold of 65.  Tightening to (0.50, 1.00)
    # raises the same doll to ~77, crossing the existing 65-point threshold.
    flat_face_risk = _scale_risk(corr + 1.0, 0.50, 1.00)
    score = _clamp(flat_face_risk)
    return {
        "score_0_100": round(score, 2),
        "iod_yaw_correlation": round(corr, 4),
        "flat_face_risk": round(flat_face_risk, 2),
    }


def _compute_head_velocity_variance(history: List[Dict[str, Any]]) -> float:
    """Compute the variance of frame-to-frame yaw velocity across all history frames.

    Human head movements accelerate and decelerate naturally as muscles engage and
    relax, producing variable velocity between frames. A replay video maintains a
    constant playback frame-rate, yielding low-variance (uniform) yaw velocity.
    A truly static photo or printed mask has near-zero velocity variance throughout.

    Returns NaN when fewer than 4 frames are available (not enough to estimate variance).
    """
    yaws = np.array([float(f["yaw"]) for f in history if "yaw" in f], dtype=np.float32)
    if len(yaws) < 4:
        return float("nan")
    # First derivative: frame-to-frame change in yaw = angular velocity
    velocities = np.diff(yaws)
    return float(np.var(velocities))


def _compute_blink_duration_ms(session: "FaceScanLiveSession") -> float:
    """Estimate mean blink duration (ms) from EAR values in the blink challenge frames.

    A real human blink lasts 100\u2013400 ms. Deepfakes and replay videos either show no
    blink event at all, or have abnormally short (<50 ms) or long (>600 ms) closures
    due to frame interpolation or video editing artifacts.

    EAR data is only meaningful on frames where MediaPipe ran (blink challenge frames).
    On all other challenges, EAR defaults to 0.30 (open-eye), so we restrict analysis
    to the blink challenge history. Returns NaN if no blink challenge was completed or
    no blink event was detected in the challenge frames.
    """
    blink_frames: List[Dict[str, Any]] = []
    for result in session.challenge_results:
        if result.get("challenge") == "blink" and result.get("passed"):
            blink_frames = session.challenge_frame_history.get(f"ch_{result['index']}", [])
            break

    if len(blink_frames) < 3:
        return float("nan")

    BLINK_EAR_THRESHOLD = 0.20  # EAR below this = eyes are closing (blink in progress)

    ear_series = [
        (float(f["ear"]), float(f.get("server_recv_mono", 0.0)))
        for f in blink_frames
    ]
    in_blink = False
    blink_start = 0.0
    durations: List[float] = []

    for ear, ts in ear_series:
        if not in_blink and ear < BLINK_EAR_THRESHOLD:
            in_blink = True
            blink_start = ts
        elif in_blink and ear >= BLINK_EAR_THRESHOLD:
            in_blink = False
            if ts > 0 and blink_start > 0:
                dur_ms = (ts - blink_start) * 1000.0
                # Accept only realistic blink durations (40\u2013700 ms includes margin)
                if 40.0 < dur_ms < 700.0:
                    durations.append(dur_ms)

    return float(np.mean(durations)) if durations else float("nan")


def _compute_mean_reaction_latency_ms(session: "FaceScanLiveSession") -> float:
    """Return the mean time (ms) from challenge appearance to completion.

    Bots typically react in milliseconds (the detection algorithm fires on the
    very first injected frame). Genuine humans process the visual prompt and
    react in 600\u20133000 ms depending on challenge complexity.

    Requires at least 2 recorded reaction times to be meaningful (a single
    challenge may be fast for other reasons). Returns NaN otherwise.
    """
    times = session.challenge_reaction_times
    return float(np.mean(times)) if len(times) >= 2 else float("nan")


def _compute_quality_metrics(history: List[Dict[str, Any]]) -> Dict[str, float]:
    """Estimate live frame quality for confidence scoring."""
    if not history:
        return {
            "blur_risk_0_100": 100.0,
            "brightness_risk_0_100": 100.0,
            "face_size_risk_0_100": 100.0,
            "mean_laplacian_var": 0.0,
            "mean_brightness": 0.0,
            "mean_face_area_ratio": 0.0,
        }

    mean_laplacian = float(np.mean([float(frame["laplacian_var"]) for frame in history]))
    mean_brightness = float(np.mean([float(frame["brightness_mean"]) for frame in history]))
    mean_face_area = float(np.mean([float(frame["bbox_area_ratio"]) for frame in history]))
    # Thresholds recalibrated for live webcam face crops.
    # At rest, webcam face crops have Laplacian variance ~80-400 (sharp).
    # Motion blur during head turns: ~30-80. Very blurry/out-of-focus: <30.
    # The old (22, 120) thresholds were calibrated for static document scans and
    # placed nearly all webcam frames at risk=0. These (30, 200) thresholds produce
    # meaningful variation: slightly motion-blurred frames score 30-70%.
    blur_risk = _scale_risk(mean_laplacian, 30.0, 200.0, inverse=True)
    brightness_risk = _scale_risk(abs(mean_brightness - 128.0), 18.0, 90.0)
    face_size_risk = _scale_risk(mean_face_area, 0.05, 0.10, inverse=True)
    return {
        "blur_risk_0_100": round(blur_risk, 2),
        "brightness_risk_0_100": round(brightness_risk, 2),
        "face_size_risk_0_100": round(face_size_risk, 2),
        "mean_laplacian_var": round(mean_laplacian, 2),
        "mean_brightness": round(mean_brightness, 2),
        "mean_face_area_ratio": round(mean_face_area, 4),
    }


def _confidence_reason(quality: Dict[str, float], session: FaceScanLiveSession) -> str:
    """Explain in simple English why the live confidence landed where it did."""
    no_face_ratio = session.frames_without_face / max(session.frames_received, 1)
    if no_face_ratio >= 0.25:
        return "Face tracking was unstable for too much of the live session, so confidence is reduced."
    if quality["blur_risk_0_100"] >= 60.0:
        return "Motion blur or soft focus reduced confidence in the live result."
    if quality["brightness_risk_0_100"] >= 60.0:
        return "Lighting was too uneven for a fully reliable live decision."
    fps = session.environment.get("observed_fps")
    if isinstance(fps, (int, float)) and fps < 4.0:
        return "The browser delivered too few usable live frames for a high-confidence result."
    return "The live session captured enough stable, well-tracked frames for a reliable result."


def _build_evidence(
    session: FaceScanLiveSession,
    temporal: Dict[str, float],
    replay: Dict[str, float],
    quality: Dict[str, float],
    saccade: Dict[str, float],
    screen_fft: Dict[str, float],
    timing: Dict[str, float],
    depth: Dict[str, float],
    confidence: float,
) -> List[str]:
    """Build operator-facing evidence bullets for the live result."""
    evidence: List[str] = []
    completed = [item["challenge"] for item in session.challenge_results if item.get("passed")]
    if completed:
        evidence.append(f"Completed live challenges: {', '.join(completed)}.")
    # Self-correction forensic note: report wrong actions and what they tell us.
    # A genuine human occasionally misreads an instruction and self-corrects after
    # seeing the feedback. Scripted attacks are robotically perfect. Neither extreme
    # is definitive on its own, but it is useful context for an operator.
    wrong_actions = session.challenge_wrong_actions
    if wrong_actions:
        counts: Dict[str, int] = {}
        for wa in wrong_actions:
            counts[wa["challenge"]] = counts.get(wa["challenge"], 0) + 1
        summary = "; ".join(f"{ch} x{n}" for ch, n in counts.items())
        evidence.append(
            f"User made {len(wrong_actions)} wrong-motion attempt(s) before self-correcting ({summary}) "
            f"\u2014 consistent with genuine real-time interaction with challenge feedback."
        )
    if replay["score_0_100"] >= 60.0:
        evidence.append("Replay heuristics found repeated frames or unstable brightness patterns consistent with a screen attack.")
    elif replay["score_0_100"] <= 20.0:
        evidence.append("No strong repeated-frame or replay-screen pattern was found in the live capture.")
    if temporal["score_0_100"] >= 50.0:
        evidence.append("Face movement across frames showed elevated temporal instability during challenge responses.")
    else:
        evidence.append("Head and eye motion stayed reasonably consistent across the live challenge sequence.")
    # Saccade / eye micro-jitter
    if saccade["score_0_100"] >= 50.0:
        evidence.append("Eye landmark positions were unusually still across frames — consistent with a static photo or looped replay rather than a live face.")
    # Screen frequency (FFT Moiré)
    if screen_fft["score_0_100"] >= 35.0:
        evidence.append(f"FFT screen-frequency analysis detected periodic spectral peaks (mean grid ratio {screen_fft['mean_fft_grid_peak']:.3f}) consistent with a filmed digital display.")
    # Frame timing uniformity
    if timing["score_0_100"] >= 35.0:
        evidence.append(f"Frame delivery intervals were suspiciously uniform (CV={timing['interval_cv']:.3f}) — indicative of a replay injection tool rather than a live browser session.")
    # 3D depth from turn geometry
    if depth["score_0_100"] >= 35.0:
        evidence.append("Interocular distance did not decrease during head turns as expected for a 3D face — consistent with a flat 2D photo or printed mask.")
    elif depth["score_0_100"] > 0.0 and depth["iod_yaw_correlation"] < -0.2:
        evidence.append("Eye separation decreased appropriately during head turns, confirming 3D face depth.")
    if quality["blur_risk_0_100"] >= 60.0 or quality["brightness_risk_0_100"] >= 60.0:
        evidence.append("Frame quality reduced confidence in the live decision.")
    if confidence < 35.0:
        evidence.append("The live session did not provide enough strong signal quality for a high-confidence verdict.")
    return evidence


def _overall_explanation(verdict: str) -> str:
    """Return a short technical explanation for the live result."""
    if verdict == "LIVENESS_FAILED":
        return "The subject did not complete the required live challenge sequence successfully."
    if verdict == "DEEPFAKE":
        return "The live session passed the challenge flow but showed strong replay or spoof-risk signals across frames."
    if verdict == "SUSPICIOUS":
        return "The live session completed, but the temporal or replay signals warrant manual review."
    if verdict == "INCONCLUSIVE":
        return "The live session completed, but frame quality and tracking stability were too weak for a reliable decision."
    return "The live session completed with low replay risk and stable challenge-response motion."


def _honest_review(verdict: str) -> str:
    """Return a plain-English operator summary for the live result."""
    if verdict == "LIVENESS_FAILED":
        return "The person did not complete the live challenge sequence successfully. Ask them to retry the session."
    if verdict == "DEEPFAKE":
        return "The live session looks high risk. Treat it as a likely spoof or replay attempt until manual review clears it."
    if verdict == "SUSPICIOUS":
        return "The live session completed, but some replay or motion-consistency signals look suspicious. Review it manually."
    if verdict == "INCONCLUSIVE":
        return "The live session completed, but the video quality was not strong enough for a confident decision."
    return "The live session looks genuine based on the current challenge-response and replay checks."


def build_live_face_scan_result(session: FaceScanLiveSession) -> Dict[str, Any]:
    """Build the canonical Face Scan payload for a completed live session."""
    history = session.all_frame_history
    completed_groups = [
        list(session.challenge_frame_history.get(f"ch_{item['index']}", []))
        for item in session.challenge_results
        if item.get("passed")
    ]
    completed_groups = [group for group in completed_groups if group]

    # Temporal consistency should only be evaluated on STILL challenges (look_straight,
    # blink). During motion challenges (turn_left, turn_right, nod) the person is
    # intentionally accelerating and decelerating — the second-derivative (jerk) of yaw
    # and pitch will naturally be large even for a completely genuine person, and would
    # produce false SUSPICIOUS verdicts. A static photo or replay, by contrast, shows
    # near-zero jerk even during a look_straight hold because nothing moves at all.
    _STILL_CHALLENGES = frozenset({"look_straight", "blink"})
    still_groups = [
        list(session.challenge_frame_history.get(f"ch_{item['index']}", []))
        for item in session.challenge_results
        if item.get("passed") and item.get("challenge") in _STILL_CHALLENGES
    ]
    still_groups = [group for group in still_groups if group]
    # Fall back to all completed groups if no still-challenge data is available
    # (e.g. a session that only had turn challenges)
    temporal = _compute_temporal_consistency(still_groups or completed_groups or [history])

    replay = _compute_replay_heuristics(history)
    quality = _compute_quality_metrics(history)

    # ── New authenticity signals ────────────────────────────────────────────────
    saccade    = _compute_saccade_analysis(history)
    screen_fft = _compute_screen_frequency_analysis(history)
    timing     = _compute_frame_timing_jitter(history)
    depth      = _compute_depth_consistency(session)

    # ── Tier 1 ML signals ────────────────────────────────────────────────────
    # These are computed at result-build time from data already in session/history.
    # head_velocity_var: variance of yaw velocity — low = replay/robot, high = human
    # blink_dur_ms: mean blink duration from EAR — abnormal = deepfake
    # reaction_lat_ms: mean challenge reaction time — instant = bot, slow = human
    head_velocity_var = _compute_head_velocity_variance(history)
    blink_dur_ms      = _compute_blink_duration_ms(session)
    reaction_lat_ms   = _compute_mean_reaction_latency_ms(session)

    quality_risk = round(
        (0.5 * quality["blur_risk_0_100"]) + (0.3 * quality["brightness_risk_0_100"]) + (0.2 * quality["face_size_risk_0_100"]),
        2,
    )
    # Risk-score formula — weights revised to incorporate the 4 new signals.
    # replay and temporal remain the dominant factors; depth (3D flat-face check)
    # is raised from 2% to 8% because it is the most physically meaningful signal
    # against photo/palm/printed-mask attacks. Temporal and screen_fft are reduced
    # slightly to keep the total at 100%.
    risk_score = round(
        _clamp(
            (0.50 * replay["score_0_100"])
            + (0.22 * temporal["score_0_100"])
            + (0.10 * quality_risk)
            + (0.05 * saccade["score_0_100"])
            + (0.04 * screen_fft["score_0_100"])
            + (0.01 * timing["score_0_100"])
            + (0.08 * depth["score_0_100"])
        ),
        2,
    )
    no_face_ratio = session.frames_without_face / max(session.frames_received, 1)
    frame_count_penalty = _scale_risk(len(history), 10.0, 30.0, inverse=True)
    tracking_penalty = _clamp(no_face_ratio * 100.0)
    confidence = round(
        _clamp(
            100.0
            - (0.35 * quality["blur_risk_0_100"])
            - (0.25 * quality["brightness_risk_0_100"])
            - (0.20 * frame_count_penalty)
            - (0.20 * tracking_penalty),
            5.0,
            99.0,
        ),
        2,
    )

    if not session.all_done:
        verdict = "LIVENESS_FAILED"
    elif confidence < 35.0:
        verdict = "INCONCLUSIVE"
    elif replay["score_0_100"] > 80.0:
        verdict = "DEEPFAKE"
    elif max(replay["score_0_100"], temporal["score_0_100"]) >= 50.0 and risk_score >= 35.0:
        # 50.0 sub-score threshold: temporal alone (computed only on still challenges) must be
        # clearly elevated to trigger SUSPICIOUS. A score of 35-49 on still frames is
        # borderline and not strong enough standalone evidence without replay support.
        # The additional risk_score >= 35.0 guard prevents individual sub-score spikes
        # (e.g. elevated repeat_frame_score from dark-frame hash collisions in a dim
        # room) from overriding a low overall risk score and producing a false SUSPICIOUS
        # verdict for a genuine live user.  If the overall evidence only yields risk < 35,
        # the verdict should align with that low-risk assessment.
        verdict = "SUSPICIOUS"
    elif depth["score_0_100"] >= 65.0:
        # Hard flat-face gate: the 3D depth check strongly indicates a flat source
        # (photo, printed mask, or palm held flat). The geometry guard above blocks
        # most palm attacks at the per-frame level, but a partial or edge-on palm
        # may still produce a high depth score. Any session where turn challenges
        # were passed but the IOD correlation shows zero 3D depth must be escalated.
        verdict = "SUSPICIOUS"
    else:
        verdict = "GENUINE"

    completed = [item["challenge"] for item in session.challenge_results if item.get("passed")]
    last_face_box = session.last_face_box or [0, 0, 0, 0]
    mean_detector_conf = round(float(np.mean([float(frame["detector_confidence"]) for frame in history])) if history else 0.0, 4)

    # Compute observed_fps from actual server-receive timestamps rather than using
    # the browser-reported value. The browser hardcodes 1000/CAPTURE_MS = 10.0 and
    # always sends that value, giving zero variation in the CSV. The server-side
    # monotonic timestamps reveal actual delivery timing including network jitter.
    _ts_list = [float(f["server_recv_mono"]) for f in history if "server_recv_mono" in f]
    if len(_ts_list) >= 2:
        _ts_duration = _ts_list[-1] - _ts_list[0]
        if _ts_duration > 0.001:
            session.environment["observed_fps"] = round((len(_ts_list) - 1) / _ts_duration, 2)

    # frame_drop_rate: fraction of received frames where no valid face was detected.
    # A genuine user at a good camera typically drops 2-15% of frames (blink, turn).
    # Replay attacks and injected videos may drop 0% (every frame has a face) or
    # spike to high values when an injected face doesn't hit the detector reliably.
    session.environment["frame_drop_rate"] = round(
        session.frames_without_face / max(session.frames_received, 1), 4
    )

    result = {
        "filename": f"face_scan_live_{session.session_id}.jpg",
        "scan_type": "face_scan",
        "mode": "live",
        "schema_version": FACE_SCAN_SCHEMA_VERSION,
        "verdict": verdict,
        "risk_score_0_100": risk_score,
        "confidence_0_100": confidence,
        "confidence_reason": _confidence_reason(quality, session),
        "overall_explanation": _overall_explanation(verdict),
        "honest_review": _honest_review(verdict),
        "evidence": _build_evidence(session, temporal, replay, quality, saccade, screen_fft, timing, depth, confidence),
        "trace": {
            "decision_trace_id": f"fs_live_{uuid.uuid4().hex[:12]}",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "processing_time_ms": int((time.time() - session.created_at.timestamp()) * 1000),
            "rules_version": FACE_SCAN_RULES_VERSION,
            "model_version": FACE_SCAN_MODEL_VERSION,
        },
        "environment": session.environment,
        "checks": {
            "face_detection": {
                "status": "pass",
                "face_count": 1,
                "primary_face_box": last_face_box,
                "detector_confidence": mean_detector_conf,
                "frames_processed": len(history),
                "frames_without_face": session.frames_without_face,
                # Mean landmark confidence across all frames — a proxy for how
                # reliably MediaPipe could locate facial keypoints. Low values
                # indicate partial occlusion, unconventional angle, or low light.
                "mean_landmark_confidence": mean_detector_conf,
            },
            "quality_assessment": {
                "status": "review" if quality_risk >= 35.0 else "pass",
                **quality,
            },
            "temporal_consistency": {
                "status": "review" if temporal["score_0_100"] >= 35.0 else "pass",
                **temporal,
                # Velocity variance: high for natural human movement, low for replay/robot.
                # Stored as None when fewer than 4 frames are available (NaN in NumPy
                # serialises poorly; None becomes null in JSON and NaN in the CSV).
                "head_velocity_variance": None if (head_velocity_var != head_velocity_var) else round(float(head_velocity_var), 6),
            },
            "replay_heuristics": {
                "status": "review" if replay["score_0_100"] >= 35.0 else "pass",
                **replay,
            },
            "active_liveness": {
                "status": "pass" if session.all_done else "fail",
                "passed": session.all_done,
                "completed_challenges": completed,
                "challenge_count": len(session.challenges),
                "best_frame_available": bool(session.best_live_frame_bytes),
                # Wrong-action events recorded during the session. Presence of
                # self-corrections is a mild positive authenticity signal (genuine
                # real-time interaction with feedback); zero wrong attempts in a
                # complex session can hint at scripted execution.
                "wrong_action_count": len(session.challenge_wrong_actions),
                "wrong_actions": session.challenge_wrong_actions,
                # Mean blink duration from the EAR series during the blink challenge.
                # Real blinks: 100-400 ms. Deepfakes: absent, too short, or too long.
                # None = blink challenge not present or no blink detected.
                "blink_duration_ms": None if (blink_dur_ms != blink_dur_ms) else round(blink_dur_ms, 1),
                # Mean reaction latency from challenge appearance to completion.
                # None until at least 2 challenges have been timed.
                "challenge_reaction_latency_ms": None if (reaction_lat_ms != reaction_lat_ms) else round(reaction_lat_ms, 1),
            },
            # ── New signals ────────────────────────────────────────────────────
            "saccade_analysis": {
                "status": "review" if saccade["score_0_100"] >= 35.0 else "pass",
                **saccade,
            },
            "screen_frequency": {
                "status": "review" if screen_fft["score_0_100"] >= 35.0 else "pass",
                **screen_fft,
            },
            "frame_timing": {
                "status": "review" if timing["score_0_100"] >= 35.0 else "pass",
                **timing,
            },
            "depth_consistency": {
                "status": "review" if depth["score_0_100"] >= 35.0 else "pass",
                **depth,
            },
        },
        "artifacts": {
            "best_frame_available": bool(session.best_live_frame_bytes),
            "challenge_snapshots_available": bool(session.challenge_snapshots),
        },
    }

    # ── Optional ML risk score (replaces the heuristic formula when trained) ──
    # Build the feature vector from the checks dict that is already in the result
    # and ask the ML scorer to predict spoof probability.  If the model pkl is
    # absent (cold-start / not yet trained) predict() returns None and we keep
    # the heuristic risk_score computed above unchanged.
    _fv = _ml_scorer_live.build_feature_vector(result["checks"], session.environment)
    _ml_result = _ml_scorer_live.predict(_fv)
    if _ml_result is not None:
        # Replace the heuristic risk score with the ML probability.
        result["risk_score_0_100"] = _ml_result["score"]
        result["trace"]["model_version"] = "xgboost-live-v1"
        result["checks"]["scoring_method"] = "ML"
        log.info(
            "ml_scorer_live: ML score applied to live result.",
            extra={"ml_score": _ml_result["score"], "session_id": session.session_id},
        )
    else:
        result["checks"]["scoring_method"] = "heuristic"

    # Assign session.result NOW — before the LLM narrative call — so that any
    # Streamlit status-poll that fires during the ~6-8 s LLM call sees a non-null
    # result immediately once status becomes "completed". The dict is mutable, so
    # the honest_review key we update below propagates automatically to whatever
    # already holds a reference to this dict (including the caller's session.result).
    # Mark narrative_pending=True so the UI can display a "generating..." message
    # while the LLM call is still in progress.
    result["narrative_pending"] = True
    session.result = result

    # Enrich the honest_review with a plain-English narrative from Gemma4.
    # If Ollama is offline the function returns the existing rule-based text unchanged.
    narrative, narrative_source = _narrative_mod.generate_face_scan_narrative(result)
    result["honest_review"] = narrative
    result["narrative_source"] = narrative_source
    # Clear the flag — the UI will stop showing the spinner on the next poll.
    result["narrative_pending"] = False
    # Mark the session as fully completed so Streamlit polls pick up the right status
    # even when this function ran in a background executor after the WS handler exited.
    session.status = "completed"

    return result


def process_live_frame_message(session: FaceScanLiveSession, b64_frame: str) -> Dict[str, Any]:
    """Process one live Face Scan frame and return the current status or result."""
    session.frames_received += 1
    # Record server-side monotonic time at the moment this frame is received.
    # Used by _compute_frame_timing_jitter to detect suspiciously uniform delivery
    # intervals (a hallmark of injection tools that replay at a fixed clock rate).
    server_recv_ts = time.monotonic()

    if not isinstance(b64_frame, str):
        return {"type": "status", "face_detected": False, "feedback": "Decode error."}

    try:
        raw = base64.b64decode(b64_frame, validate=True)
        nparr = np.frombuffer(raw, np.uint8)
        img = _CV2_IMDECODE(nparr, _CV2_IMREAD_COLOR)
        if img is None:
            return {"type": "status", "face_detected": False, "feedback": "Invalid frame."}
    except binascii.Error:
        return {"type": "status", "face_detected": False, "feedback": "Decode error."}

    try:
        # Only run the MediaPipe EAR pipeline when processing a blink challenge.
        # For all other challenges EAR is unused; skipping it roughly halves the
        # per-frame detection time because it avoids a full second detector run.
        faces = _detect_faces(img, attach_ear=(session.current_challenge == "blink"))
    except _LIVE_FRAME_RUNTIME_ERRORS as exc:  # noqa: BLE001
        log.error("Live frame detection raised unexpectedly.", extra={"error": str(exc), "session_id": session.session_id})
        _abort = _no_face_tick(session)
        if _abort is not None:
            return _abort
        return {
            "type": "status",
            "face_detected": False,
            "challenge": session.current_challenge,
            "challenges_completed": session.current_challenge_idx,
            "total_challenges": len(session.challenges),
            "feedback": "Face detection had a temporary issue — hold still and try again.",
            "challenge_just_passed": False,
        }

    if not faces:
        _abort = _no_face_tick(session)
        if _abort is not None:
            return _abort
        session.face_stable_frames = 0
        session.face_stable_yaw_buffer.clear()
        # During a challenge transition the user is physically repositioning and
        # the face being briefly absent is expected. Show context-aware feedback
        # so they know what to do next rather than seeing a generic error.
        _in_transition = (
            session.last_challenge_advance_at > 0.0
            and (time.monotonic() - session.last_challenge_advance_at) < CHALLENGE_TRANSITION_GRACE_SECONDS
        )
        return {
            "type": "status",
            "face_detected": False,
            "challenge": session.current_challenge,
            "challenges_completed": session.current_challenge_idx,
            "total_challenges": len(session.challenges),
            "feedback": "Move back to centre and hold still for the next challenge." if _in_transition else "No face detected — move into the oval.",
            "challenge_just_passed": False,
        }

    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    # Do NOT reset consecutive_frames_without_face here. A marginal detection that
    # fails the geometry, confidence, or size gates below (e.g. an extreme profile
    # face during a turn that the detector caught at the edge of its range) should
    # NOT save the session from the no-face abort. We only reset after ALL three
    # hard rejection gates pass — see the comment below the area check.

    # Validate that the detected object is geometrically consistent with a human
    # face before doing anything with it.  Face detectors occasionally accept a
    # palm, hand, or printed object — this guard rejects those early so no
    # challenge progress is credited and the frame is counted as "no face".
    geom_valid, geom_reason = face_geometry_valid(face)
    if not geom_valid:
        session.frames_without_face += 1
        session.face_stable_frames = 0
        session.face_stable_yaw_buffer.clear()
        log.debug(
            "Face geometry check rejected detection; treating as no face.",
            extra={"reason": geom_reason, "session_id": session.session_id},
        )
        return {
            "type": "status",
            "face_detected": False,
            "challenge": session.current_challenge,
            "challenges_completed": session.current_challenge_idx,
            "total_challenges": len(session.challenges),
            "feedback": "Keep your face fully inside the oval — move hands away from the camera.",
            "challenge_just_passed": False,
        }

    session.last_live_frame_bytes = raw
    try:
        metrics = _extract_live_frame_metrics(img, face, session.frames_received)
    except _LIVE_FRAME_RUNTIME_ERRORS as exc:  # noqa: BLE001
        log.warning(
            "Live frame landmarks were unusable; skipping frame.",
            extra={"error": str(exc), "session_id": session.session_id},
        )
        return {
            "type": "status",
            "face_detected": False,
            "challenge": session.current_challenge,
            "challenges_completed": session.current_challenge_idx,
            "total_challenges": len(session.challenges),
            "feedback": "Keep your full face visible and look straight at the camera.",
            "challenge_just_passed": False,
        }

    session.last_face_box = metrics["face_box"]
    # Tag the frame with the server receive time so timing-jitter analysis can
    # measure inter-frame interval variance later (see _compute_frame_timing_jitter)
    metrics["server_recv_mono"] = server_recv_ts

    # Reject frames where the detector confidence is too low to trust.
    # Even after passing the geometry invariant check, a background object can
    # still produce a marginal detection with low confidence.  The pitch/yaw
    # features extracted from such frames are unreliable noise: their variance
    # across 4 frames is enough to spuriously satisfy the nod challenge.  Only
    # frames where the detector was sufficiently confident are added to history.
    if metrics["detector_confidence"] < MIN_FACE_DETECTION_CONFIDENCE:
        session.frames_without_face += 1
        session.face_stable_frames = 0
        session.face_stable_yaw_buffer.clear()
        log.debug(
            "Live frame rejected: detector confidence below threshold.",
            extra={"detector_confidence": metrics["detector_confidence"], "session_id": session.session_id},
        )
        return {
            "type": "status",
            "face_detected": False,
            "challenge": session.current_challenge,
            "challenges_completed": session.current_challenge_idx,
            "total_challenges": len(session.challenges),
            "feedback": "No face detected — look directly into the camera.",
            "challenge_just_passed": False,
        }

    # Reject frames where the face is too small relative to the full frame.
    # When the user steps back, ducks away, or a background object is detected,
    # the bbox_area_ratio is very small.  The person must be close enough to
    # the camera to fill a meaningful fraction of the frame (the oval acts as
    # the visual guide for this on the frontend).
    if metrics["bbox_area_ratio"] < MIN_FACE_AREA_RATIO:
        session.frames_without_face += 1
        session.face_stable_frames = 0
        session.face_stable_yaw_buffer.clear()
        log.debug(
            "Live frame rejected: face bounding box too small.",
            extra={"bbox_area_ratio": metrics["bbox_area_ratio"], "session_id": session.session_id},
        )
        return {
            "type": "status",
            "face_detected": False,
            "challenge": session.current_challenge,
            "challenges_completed": session.current_challenge_idx,
            "total_challenges": len(session.challenges),
            "feedback": "No face detected \u2014 position yourself in the oval.",
            "challenge_just_passed": False,
        }

    # Face passed all three hard rejection gates (geometry, confidence, area) —
    # the user is genuinely present and visible. Reset the consecutive no-face
    # counter now. Doing it here (not at detection time) prevents profile/partial
    # detections that fail validation from silently resetting the abort clock.
    session.consecutive_frames_without_face = 0

    # ── Pre-liveness face stability gate ─────────────────────────────────────
    # Expert-recommended pipeline:
    #   Detection → Validation → Tracking (stability) → Liveness Challenge
    #
    # Challenges only start once FACE_STABLE_FRAMES_REQUIRED consecutive frames
    # all pass the stricter stability check.  Any failure resets the counter so
    # a flickering or marginal detection cannot accumulate a partial count.
    # After the window fills, yaw variance AND jitter naturalness are checked to
    # confirm live micro-movement — a real person oscillates irregularly; a static
    # source or shaken screen has near-zero or perfectly-alternating yaw values.
    if session.face_stable_frames < FACE_STABLE_FRAMES_REQUIRED:
        # Compute texture score and brightness together — brightness drives the
        # adaptive texture threshold (lower in dark rooms to avoid false rejects).
        _texture = compute_face_texture_score(img, face.bbox)
        _brightness = metrics.get("brightness_mean", 128.0)
        stable_ok, stable_feedback = is_face_stable(
            face=face,
            face_count=len(faces),
            bbox_area_ratio=metrics["bbox_area_ratio"],
            confidence=metrics["detector_confidence"],
            nose_rel_x=metrics["nose_rel_x"],
            nose_rel_y=metrics.get("nose_rel_y", 0.50),
            yaw=metrics.get("yaw", 0.0),
            pitch=metrics.get("pitch", 0.0),
            texture_score=_texture,
            brightness_mean=_brightness,
        )
        if not stable_ok:
            # Log at INFO on the first rejection of each stability window so we can
            # diagnose what condition keeps failing without flooding logs on every frame.
            # Subsequent frames in the same reset-window (face_stable_frames == 0 after reset)
            # are suppressed.
            if session.face_stable_frames == 0:
                log.info(
                    "Face stability gate rejected frame (first in window).",
                    extra={
                        "session_id": session.session_id,
                        "reason": stable_feedback,
                        "confidence": round(metrics["detector_confidence"], 4),
                        "bbox_area_ratio": round(metrics["bbox_area_ratio"], 4),
                        "nose_rel_x": round(metrics["nose_rel_x"], 4),
                        "nose_rel_y": round(metrics.get("nose_rel_y", 0.50), 4),
                        "yaw": round(metrics.get("yaw", 0.0), 4),
                        "pitch": round(metrics.get("pitch", 0.0), 4),
                        "texture_score": round(_texture, 2),
                        "brightness_mean": round(_brightness, 2),
                        "face_count": len(faces),
                    },
                )
            else:
                log.debug(
                    "Face stability gate rejected frame.",
                    extra={
                        "session_id": session.session_id,
                        "reason": stable_feedback,
                        "confidence": round(metrics["detector_confidence"], 4),
                        "bbox_area_ratio": round(metrics["bbox_area_ratio"], 4),
                        "nose_rel_x": round(metrics["nose_rel_x"], 4),
                        "nose_rel_y": round(metrics.get("nose_rel_y", 0.50), 4),
                        "yaw": round(metrics.get("yaw", 0.0), 4),
                        "pitch": round(metrics.get("pitch", 0.0), 4),
                        "texture_score": round(_texture, 2),
                        "brightness_mean": round(_brightness, 2),
                        "face_count": len(faces),
                        "stable_frames_so_far": session.face_stable_frames,
                    },
                )
            # Reset on any validation failure — must be a continuous clean window.
            session.face_stable_frames = 0
            session.face_stable_yaw_buffer.clear()
            session.blink_observed_in_stability = False
            return {
                "type": "status",
                "face_detected": True,
                "face_stable": False,
                "face_stable_progress": 0,
                "face_stable_required": FACE_STABLE_FRAMES_REQUIRED,
                "challenge": session.current_challenge,
                "challenges_completed": session.current_challenge_idx,
                "total_challenges": len(session.challenges),
                "feedback": stable_feedback,
                "challenge_just_passed": False,
            }

        # Frame passed all validation checks — accumulate toward the quota.
        session.face_stable_frames += 1
        session.face_stable_yaw_buffer.append(float(metrics.get("yaw", 0.0)))
        # Track natural blink signal during the stability window.
        if metrics.get("ear", 0.30) < 0.20:
            session.blink_observed_in_stability = True
        log.debug(
            "Face stability progress.",
            extra={"stable_frames": session.face_stable_frames, "required": FACE_STABLE_FRAMES_REQUIRED, "session_id": session.session_id},
        )

        if session.face_stable_frames >= FACE_STABLE_FRAMES_REQUIRED:
            # Window complete — verify natural micro-movement via yaw variance.
            _yaw_var = float(np.var(session.face_stable_yaw_buffer))
            if _yaw_var < FACE_STABILITY_YAW_VARIANCE_MIN:
                # Static source suspected — reset and demand another window.
                session.face_stable_frames = 0
                session.face_stable_yaw_buffer.clear()
                session.blink_observed_in_stability = False
                # Count this static-source signal. If it fires too many times,
                # this is a persistent spoof attempt, not a struggling genuine user.
                session.static_source_rejections += 1
                log.warning(
                    "Face stability gate: yaw variance too low — static source suspected.",
                    extra={"yaw_var": _yaw_var, "threshold": FACE_STABILITY_YAW_VARIANCE_MIN, "rejections": session.static_source_rejections, "session_id": session.session_id},
                )
                if session.static_source_rejections >= MAX_STATIC_SOURCE_REJECTIONS:
                    log.warning(
                        "Live session aborted: persistent static-source rejections exceeded threshold.",
                        extra={"session_id": session.session_id, "rejections": session.static_source_rejections},
                    )
                    return _build_abort_result(
                        session,
                        "Persistent static-source or spoof-attempt detected — session terminated",
                    )
                return {
                    "type": "status",
                    "face_detected": True,
                    "face_stable": False,
                    "face_stable_progress": 0,
                    "face_stable_required": FACE_STABLE_FRAMES_REQUIRED,
                    "challenge": session.current_challenge,
                    "challenges_completed": session.current_challenge_idx,
                    "total_challenges": len(session.challenges),
                    "feedback": "No natural movement detected — ensure you are using a live camera.",
                    "challenge_just_passed": False,
                }
            # Anti-gaming: reject artificially periodic jitter (e.g. screen shaking).
            _natural_ok, _natural_msg = is_yaw_motion_natural(session.face_stable_yaw_buffer)
            if not _natural_ok:
                session.face_stable_frames = 0
                session.face_stable_yaw_buffer.clear()
                session.blink_observed_in_stability = False
                session.static_source_rejections += 1
                log.warning(
                    "Face stability gate: yaw jitter pattern too regular — artificial source suspected.",
                    extra={"session_id": session.session_id, "rejections": session.static_source_rejections},
                )
                if session.static_source_rejections >= MAX_STATIC_SOURCE_REJECTIONS:
                    log.warning(
                        "Live session aborted: persistent artificial-jitter rejections exceeded threshold.",
                        extra={"session_id": session.session_id, "rejections": session.static_source_rejections},
                    )
                    return _build_abort_result(
                        session,
                        "Persistent artificial-source jitter detected — session terminated",
                    )
                return {
                    "type": "status",
                    "face_detected": True,
                    "face_stable": False,
                    "face_stable_progress": 0,
                    "face_stable_required": FACE_STABLE_FRAMES_REQUIRED,
                    "challenge": session.current_challenge,
                    "challenges_completed": session.current_challenge_idx,
                    "total_challenges": len(session.challenges),
                    "feedback": _natural_msg,
                    "challenge_just_passed": False,
                }
            # All checks passed — start challenge timing and fall through to challenge logic.
            session.challenge_started_at = time.monotonic()
        else:
            # Still accumulating — inform the user to keep holding still.
            _remaining = FACE_STABLE_FRAMES_REQUIRED - session.face_stable_frames
            return {
                "type": "status",
                "face_detected": True,
                "face_stable": False,
                "face_stable_progress": session.face_stable_frames,
                "face_stable_required": FACE_STABLE_FRAMES_REQUIRED,
                "challenge": session.current_challenge,
                "challenges_completed": session.current_challenge_idx,
                "total_challenges": len(session.challenges),
                "feedback": f"Hold still… ({_remaining} more frames)",
                "challenge_just_passed": False,
            }

    session.current_frame_history().append(metrics)
    session.all_frame_history.append(metrics)

    # Early-exit replay check: once enough frames are in history, run a quick
    # repeat-frame analysis. A score this high this early means the session is a
    # clear replay attack — terminate immediately rather than letting it continue.
    #
    # IMPORTANT: skip this check during hold-based challenges (nod, turn_left,
    # turn_right).  These challenges require the user to hold a static pose for
    # ~4 seconds (40 frames), so consecutive frames are intentionally very similar.
    # Running the replay check during a hold phase produces a false-positive abort
    # because the repeat_frame_score rises even for a genuine live user.  The
    # stability gate (which fires BEFORE challenges start) has already confirmed
    # live yaw variance, so a replay could not have reached this point undetected.
    _HOLD_CHALLENGES = {"nod", "turn_left", "turn_right"}
    if (
        len(session.all_frame_history) == REPLAY_ABORT_FRAME_THRESHOLD
        and session.current_challenge not in _HOLD_CHALLENGES
    ):
        _early_replay = _compute_replay_heuristics(session.all_frame_history)
        if _early_replay["repeat_frame_score"] >= REPLAY_ABORT_SCORE_THRESHOLD:
            log.warning(
                "Live session aborted: early replay detection at frame threshold.",
                extra={
                    "session_id": session.session_id,
                    "repeat_frame_score": _early_replay["repeat_frame_score"],
                    "threshold": REPLAY_ABORT_SCORE_THRESHOLD,
                },
            )
            return _build_abort_result(session, "Replay attack detected — session terminated")

    current_challenge = session.current_challenge
    if current_challenge is None:
        # All challenges already passed in a previous frame — replay the result if
        # it is ready, otherwise tell the browser to keep waiting. This guard fires
        # when the browser sends extra frames before it receives the 'processing'
        # message and stops capture.
        if session.result is not None and not bool(session.result.get("narrative_pending")):
            return {"type": "result", **session.result}
        return {"type": "processing", "message": "All challenges completed. Verifying your results\u2026"}

    # Challenge timeout enforcement: if the user has not completed the current
    # challenge within the allowed time, reset the frame history and restart
    # the timer.  Hold-based challenges (nod, turns) get longer timeouts.
    # This prevents slow brute-force probing while giving real users enough time.
    _now = time.monotonic()
    _challenge_timeout = _CHALLENGE_TIMEOUTS.get(current_challenge, CHALLENGE_TIMEOUT_SECONDS)
    if session.challenge_started_at == 0.0:
        session.challenge_started_at = _now
    elif _now - session.challenge_started_at > _challenge_timeout:
        ch_key = f"ch_{session.current_challenge_idx}"
        session.challenge_frame_history[ch_key] = []
        session.challenge_started_at = _now
        log.info(
            "Live challenge timed out; resetting challenge history.",
            extra={"challenge": current_challenge, "timeout": _challenge_timeout, "session_id": session.session_id},
        )
        return {
            "type": "status",
            "face_detected": True,
            "challenge": current_challenge,
            "challenges_completed": session.current_challenge_idx,
            "total_challenges": len(session.challenges),
            "feedback": f"Time's up — please perform the '{current_challenge.replace('_', ' ')}' challenge again.",
            "challenge_just_passed": False,
        }

    try:
        analysis = analyze_challenge(session.current_frame_history(), current_challenge)
    except _LIVE_FRAME_RUNTIME_ERRORS as exc:  # noqa: BLE001
        log.error(
            "Live challenge analysis failed for one frame.",
            extra={
                "error": str(exc),
                "session_id": session.session_id,
                "challenge": current_challenge,
            },
        )
        return {
            "type": "status",
            "face_detected": True,
            "challenge": current_challenge,
            "challenges_completed": session.current_challenge_idx,
            "total_challenges": len(session.challenges),
            "feedback": "Processing hiccup — hold still and continue.",
            "challenge_just_passed": False,
        }

    if analysis.get("reset_needed") or analysis.get("wrong_motion"):
        wrong_motion = analysis.get("wrong_motion", "unknown")
        # Record this wrong-action event with enough context for the forensic report.
        session.challenge_wrong_actions.append({
            "challenge": current_challenge,
            "wrong_motion": wrong_motion,
            "frame_index": metrics["frame_index"],
        })
        if analysis.get("reset_needed"):
            ch_key = f"ch_{session.current_challenge_idx}"
            current_history = session.challenge_frame_history.get(ch_key, [])
            # Keep only the last 2 frames — they represent the face near neutral
            # (just returned from the wrong-direction peak) and are a clean baseline.
            session.challenge_frame_history[ch_key] = current_history[-2:] if len(current_history) >= 2 else []
            log.info(
                "Wrong-direction motion detected; challenge frame history reset.",
                extra={"session_id": session.session_id, "challenge": current_challenge, "wrong_motion": wrong_motion},
            )
        # feedback_sticky=True tells the browser to hold this message for 2.5 s so
        # the user has time to read it before the next frame status overwrites it.
        return {
            "type": "status",
            "face_detected": True,
            "challenge": current_challenge,
            "challenges_completed": session.current_challenge_idx,
            "total_challenges": len(session.challenges),
            "feedback": analysis["feedback"],
            "feedback_sticky": True,
            "challenge_just_passed": False,
        }

    just_passed = False
    if analysis["passed"]:
        if current_challenge == "look_straight" and not session.best_live_frame_bytes:
            session.best_live_frame_bytes = raw
        if session.best_live_frame_bytes is None:
            session.best_live_frame_bytes = raw

        # Record reaction latency before resetting challenge_started_at.
        # This measures the time from when the challenge prompt appeared to when
        # the user completed it. Bots respond near-instantly; humans take 0.6\u20133 s.
        if session.challenge_started_at > 0.0:
            _reaction_ms = round((_now - session.challenge_started_at) * 1000.0, 1)
            session.challenge_reaction_times.append(_reaction_ms)

        # Mark the challenge transition timestamp and clear the consecutive no-face
        # counter so the abort clock starts fresh for the next challenge. advance_challenge()
        # may reset face_stable_frames (for turn challenges), forcing a new stability gate
        # during which the user naturally has no face visible as they reorient.
        session.last_challenge_advance_at = _now
        session.consecutive_frames_without_face = 0

        snapshot = {
            "challenge": current_challenge,
            "frame_index": metrics["frame_index"],
            "face_box": metrics["face_box"],
        }
        session.advance_challenge(snapshot=snapshot)
        # Reset the challenge timer so the next challenge has a fresh timeout window.
        session.challenge_started_at = time.monotonic()
        just_passed = True
        if session.all_done:
            # Signal the WebSocket handler to build the result payload outside
            # this executor call so the browser gets an immediate acknowledgement
            # (and stops performing the last challenge) while the LLM narrative
            # (~6-8 s) runs in a separate step.
            session.status = "processing"
            return {"type": "processing", "message": "All challenges completed! Verifying your results\u2026"}

    return {
        "type": "status",
        "face_detected": True,
        "challenge": current_challenge,
        "challenges_completed": session.current_challenge_idx,
        "total_challenges": len(session.challenges),
        "feedback": analysis["feedback"],
        "challenge_just_passed": just_passed,
    }


def render_live_page_html(session_id: str, challenges: List[str]) -> str:
    """Return the lightweight customer-facing Face Scan live page HTML."""
    challenges_json = str(challenges).replace("'", '"')
    return _FACE_SCAN_LIVE_PAGE_HTML.replace("__SESSION_ID__", session_id).replace("__CHALLENGES_JSON__", challenges_json)