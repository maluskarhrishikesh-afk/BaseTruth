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
from basetruth.kyc.liveness import analyze_challenge, extract_features
from basetruth.logger import get_logger
from basetruth.vision.face import get_face_analyzer, get_mediapipe_faces

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
DEFAULT_FACE_SCAN_CHALLENGES: List[str] = ["blink", "turn_left", "nod"]
FACE_SCAN_CHALLENGE_LABELS: Dict[str, str] = {
    "look_straight": "LOOK AT THE CAMERA",
    "blink": "CLOSE YOUR EYES",
    "turn_left": "TURN YOUR HEAD LEFT",
    "turn_right": "TURN YOUR HEAD RIGHT",
    "nod": "NOD YOUR HEAD",
}
FACE_SCAN_CHALLENGE_INSTRUCTIONS: Dict[str, str] = {
    "look_straight": "Look directly into the camera and hold still.",
    "blink": "Slowly close both eyes fully, then open them again.",
    "turn_left": "Slowly turn your head to your left.",
    "turn_right": "Slowly turn your head to your right.",
    "nod": "Slowly nod your head down and then back up.",
}

_LIVE_FRAME_RUNTIME_ERRORS = (AttributeError, RuntimeError, TypeError, ValueError, OSError)

_face_lock = threading.Lock()

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
const LABELS = {look_straight:'LOOK AT THE CAMERA', blink:'CLOSE YOUR EYES', turn_left:'TURN YOUR HEAD LEFT', turn_right:'TURN YOUR HEAD RIGHT', nod:'NOD YOUR HEAD'};
const INSTR = {look_straight:'Look directly into the camera and hold still.', blink:'Slowly close both eyes fully, then open them again.', turn_left:'Slowly turn your head to your left.', turn_right:'Slowly turn your head to your right.', nod:'Slowly nod your head down and then back up.'};
let ws=null, stream=null, captureTimer=null, reconnectTimer=null;
const CAPTURE_MS = 125;
const RECONNECT_DELAY_MS = 1200;
const MAX_RECONNECT_ATTEMPTS = 8;
let reconnectAttempts = 0;
let sessionFinished = false;

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
        ws.send(JSON.stringify({type:'meta', camera_width: vid.videoWidth || 0, camera_height: vid.videoHeight || 0, observed_fps: 1000 / CAPTURE_MS, user_agent:navigator.userAgent, platform:navigator.platform || '', device_label:''}));
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
                if(ws && ws.readyState===1) ws.send(JSON.stringify({type:'frame', data:b64}));
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
        feedback(msg.feedback || '');
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
    challenge_results: List[Dict[str, Any]] = field(default_factory=list)
    challenge_snapshots: List[Dict[str, Any]] = field(default_factory=list)
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
        """Record the current challenge as passed and move to the next one."""
        if self.current_challenge is not None:
            self.challenge_results.append(
                {
                    "index": self.current_challenge_idx,
                    "challenge": self.current_challenge,
                    "passed": True,
                }
            )
            if snapshot:
                self.challenge_snapshots.append(snapshot)

        self.current_challenge_idx += 1
        self.challenge_frame_history[f"ch_{self.current_challenge_idx}"] = []

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


def _hamming_distance(hash_a: str, hash_b: str) -> int:
    """Return the Hamming distance between two hex-encoded average hashes."""
    a_bits = bin(int(hash_a, 16))[2:].zfill(len(hash_a) * 4)
    b_bits = bin(int(hash_b, 16))[2:].zfill(len(hash_b) * 4)
    return sum(ch_a != ch_b for ch_a, ch_b in zip(a_bits, b_bits))


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


def _detect_faces(img: np.ndarray) -> List[Any]:
    """Detect faces for the live Face Scan frame."""
    try:
        app = get_face_analyzer()
    except ImportError:
        log.info("Face Scan live analysis falling back to MediaPipe because InsightFace is unavailable.")
    else:
        try:
            with _face_lock:
                faces = list(app.get(img))
            if faces:
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
    det_score = float(getattr(face, "det_score", 0.95) or 0.95)

    return {
        **features,
        "frame_index": frame_index,
        "laplacian_var": laplacian_var,
        "brightness_mean": brightness_mean,
        "edge_density": edge_density,
        "bbox_area_ratio": bbox_area_ratio,
        "frame_hash": frame_hash,
        "detector_confidence": det_score,
        "face_box": [x1, y1, x2, y2],
    }


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

    device_label = str(data.get("device_label") or "")
    suspicious_virtual = any(token in device_label.lower() for token in ("obs", "virtual", "manycam", "snap camera"))
    if suspicious_virtual:
        session.environment["virtual_camera_suspected"] = True


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
    """Estimate replay risk from repeated frames and brightness flicker."""
    if len(history) < 4:
        return {"score_0_100": 10.0, "repeat_frame_score": 0.0, "flicker_score": 0.0, "brightness_instability": 0.0}

    hashes = [str(frame["frame_hash"]) for frame in history]
    low_distance_pairs = 0
    for idx in range(1, len(hashes)):
        if _hamming_distance(hashes[idx - 1], hashes[idx]) <= 3:
            low_distance_pairs += 1
    repeat_ratio = low_distance_pairs / max(len(hashes) - 1, 1)
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
    blur_risk = _scale_risk(mean_laplacian, 22.0, 120.0, inverse=True)
    brightness_risk = _scale_risk(abs(mean_brightness - 128.0), 18.0, 90.0)
    face_size_risk = _scale_risk(mean_face_area, 0.06, 0.18, inverse=True)
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
    confidence: float,
) -> List[str]:
    """Build operator-facing evidence bullets for the live result."""
    evidence: List[str] = []
    completed = [item["challenge"] for item in session.challenge_results if item.get("passed")]
    if completed:
        evidence.append(f"Completed live challenges: {', '.join(completed)}.")
    if replay["score_0_100"] >= 60.0:
        evidence.append("Replay heuristics found repeated frames or unstable brightness patterns consistent with a screen attack.")
    elif replay["score_0_100"] <= 20.0:
        evidence.append("No strong repeated-frame or replay-screen pattern was found in the live capture.")
    if temporal["score_0_100"] >= 45.0:
        evidence.append("Face movement across frames showed elevated temporal instability during challenge responses.")
    else:
        evidence.append("Head and eye motion stayed reasonably consistent across the live challenge sequence.")
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
    temporal = _compute_temporal_consistency(completed_groups or [history])
    replay = _compute_replay_heuristics(history)
    quality = _compute_quality_metrics(history)

    quality_risk = round(
        (0.5 * quality["blur_risk_0_100"]) + (0.3 * quality["brightness_risk_0_100"]) + (0.2 * quality["face_size_risk_0_100"]),
        2,
    )
    risk_score = round(_clamp((0.5 * replay["score_0_100"]) + (0.35 * temporal["score_0_100"]) + (0.15 * quality_risk)), 2)
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
    elif max(replay["score_0_100"], temporal["score_0_100"]) >= 35.0:
        verdict = "SUSPICIOUS"
    else:
        verdict = "GENUINE"

    completed = [item["challenge"] for item in session.challenge_results if item.get("passed")]
    last_face_box = session.last_face_box or [0, 0, 0, 0]
    mean_detector_conf = round(float(np.mean([float(frame["detector_confidence"]) for frame in history])) if history else 0.0, 4)
    fps = session.environment.get("observed_fps")
    if isinstance(fps, (int, float)) and session.frames_received > 0:
        frame_drop_rate = _clamp(max(0.0, float(fps) - len(history) / max(len(completed), 1)) * 10.0)
        session.environment["frame_drop_rate"] = round(frame_drop_rate, 2)

    return {
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
        "evidence": _build_evidence(session, temporal, replay, quality, confidence),
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
            },
            "quality_assessment": {
                "status": "review" if quality_risk >= 35.0 else "pass",
                **quality,
            },
            "temporal_consistency": {
                "status": "review" if temporal["score_0_100"] >= 35.0 else "pass",
                **temporal,
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
            },
        },
        "artifacts": {
            "best_frame_available": bool(session.best_live_frame_bytes),
            "challenge_snapshots_available": bool(session.challenge_snapshots),
        },
    }


def process_live_frame_message(session: FaceScanLiveSession, b64_frame: str) -> Dict[str, Any]:
    """Process one live Face Scan frame and return the current status or result."""
    session.frames_received += 1
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
        faces = _detect_faces(img)
    except _LIVE_FRAME_RUNTIME_ERRORS as exc:  # noqa: BLE001
        log.error("Live frame detection raised unexpectedly.", extra={"error": str(exc), "session_id": session.session_id})
        session.frames_without_face += 1
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
        session.frames_without_face += 1
        return {
            "type": "status",
            "face_detected": False,
            "challenge": session.current_challenge,
            "challenges_completed": session.current_challenge_idx,
            "total_challenges": len(session.challenges),
            "feedback": "No face detected — move into the oval.",
            "challenge_just_passed": False,
        }

    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
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
    session.current_frame_history().append(metrics)
    session.all_frame_history.append(metrics)

    current_challenge = session.current_challenge
    if current_challenge is None:
        session.status = "completed"
        session.result = build_live_face_scan_result(session)
        return {"type": "result", **session.result}

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

    just_passed = False
    if analysis["passed"]:
        if current_challenge == "look_straight" and not session.best_live_frame_bytes:
            session.best_live_frame_bytes = raw
        if session.best_live_frame_bytes is None:
            session.best_live_frame_bytes = raw

        snapshot = {
            "challenge": current_challenge,
            "frame_index": metrics["frame_index"],
            "face_box": metrics["face_box"],
        }
        session.advance_challenge(snapshot=snapshot)
        just_passed = True
        if session.all_done:
            session.status = "completed"
            session.result = build_live_face_scan_result(session)
            return {"type": "result", **session.result}

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