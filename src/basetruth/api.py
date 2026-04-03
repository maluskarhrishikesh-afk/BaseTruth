"""BaseTruth FastAPI REST layer.

Run with::

    uvicorn basetruth.api:app --host 0.0.0.0 --port 8502

Or from baseTruth root::

    python -m basetruth.api

Install the api extra first::

    pip install basetruth[api]

"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from fastapi import FastAPI, File, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field

    _FASTAPI_AVAILABLE = True

    # ── Request / response models (must be at module level so Pydantic v2
    # can resolve forward references when generating the OpenAPI schema) ──────

    class ScanPathRequest(BaseModel):
        path: str = Field(..., description="Absolute path to the document or structured JSON to scan.")

    class UpdateCaseRequest(BaseModel):
        status: Optional[str] = Field(None, description="new | triage | investigating | pending_client | closed")
        disposition: Optional[str] = Field(None, description="open | monitor | escalate | cleared | fraud_confirmed")
        priority: Optional[str] = Field(None, description="low | normal | high | critical")
        assignee: Optional[str] = Field(None, description="Investigator user name.")
        labels: Optional[List[str]] = Field(None, description="Free-form labels for routing and triage.")
        note_text: str = Field("", description="Text of the new note to append, if any.")
        note_author: str = Field("api", description="Author name for the note.")

    class CreateKYCSessionRequest(BaseModel):
        customer_name:           str            = Field("", description="Customer display name.")
        entity_ref:              str            = Field("", description="Entity / case reference ID.")
        challenges:              List[str]      = Field([], description="Liveness challenges to present.")
        reference_embedding_b64: Optional[str]  = Field(None, description="Base-64 ArcFace embedding from the reference ID document.")

except ImportError:
    _FASTAPI_AVAILABLE = False


_DEFAULT_ARTIFACT_ROOT = Path("artifacts")

# ---------------------------------------------------------------------------
# Customer-facing Video KYC HTML page (served at GET /kyc/{session_id})
# Placeholders replaced at request-time:
#   __SESSION_ID__        → session ID token
#   __CHALLENGES_COUNT__  → integer number of challenges
#   __CUSTOMER_NAME__     → customer display name (may be empty)
#   __CHALLENGES_JSON__   → JSON array of challenge names, e.g. ["blink","nod"]
# ---------------------------------------------------------------------------
_KYC_PAGE_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>BaseTruth · Video KYC</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
  background:#0f172a;color:#e2e8f0;min-height:100vh;
  display:flex;flex-direction:column;align-items:center;padding:1rem 0.75rem}
.logo{margin:1.2rem 0 0.8rem;font-size:1.35rem;font-weight:800;
  background:linear-gradient(135deg,#6366f1,#8b5cf6);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent}
.card{background:#1e293b;border:1px solid #334155;border-radius:16px;
  padding:1.4rem 1.25rem;width:100%;max-width:460px;margin-bottom:0.75rem}
.video-wrap{position:relative;width:100%;border-radius:12px;overflow:hidden;
  background:#000;aspect-ratio:4/3}
video{width:100%;height:100%;object-fit:cover;transform:scaleX(-1)}
.oval{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
  width:52%;aspect-ratio:3/4;border:3px solid rgba(99,102,241,.65);
  border-radius:50%;pointer-events:none}
.badge{position:absolute;top:.6rem;right:.6rem;padding:.28rem .7rem;
  border-radius:99px;font-size:.72rem;font-weight:700;backdrop-filter:blur(8px)}
.b-ok{background:rgba(34,197,94,.18);color:#4ade80;border:1px solid rgba(34,197,94,.3)}
.b-warn{background:rgba(234,179,8,.18);color:#facc15;border:1px solid rgba(234,179,8,.3)}
.b-idle{background:rgba(148,163,184,.18);color:#94a3b8;border:1px solid rgba(148,163,184,.3)}
.ch-card{background:linear-gradient(135deg,rgba(99,102,241,.14),rgba(139,92,246,.09));
  border:1px solid rgba(99,102,241,.38);border-radius:12px;
  padding:.9rem 1.1rem;margin-top:.9rem;text-align:center}
.ch-label{font-size:1.15rem;font-weight:800;color:#c4b5fd;margin-bottom:.35rem;
  letter-spacing:.04em}
.ch-inst{font-size:.84rem;color:#94a3b8;line-height:1.55}
.prog-wrap{background:#0f172a;border-radius:99px;height:7px;margin-top:.7rem;overflow:hidden}
.prog-fill{height:100%;border-radius:99px;background:linear-gradient(90deg,#6366f1,#8b5cf6);
  transition:width .35s ease}
.dots{display:flex;gap:.45rem;justify-content:center;margin-top:.65rem}
.dot{width:10px;height:10px;border-radius:50%;border:2px solid #475569;background:transparent;transition:all .2s}
.dot.active{border-color:#6366f1;background:#6366f1}
.dot.done{border-color:#4ade80;background:#4ade80}
.fb{text-align:center;font-size:.88rem;margin-top:.65rem;min-height:1.2em;
  color:#94a3b8;transition:color .25s}
.fb.pass{color:#4ade80}.fb.fail{color:#f87171}
.btn{display:block;width:100%;padding:.82rem;
  background:linear-gradient(135deg,#4f46e5,#6366f1);color:#fff;
  border:none;border-radius:10px;font-size:1rem;font-weight:700;
  cursor:pointer;margin-top:.9rem;transition:opacity .2s}
.btn:hover{opacity:.88}.btn:disabled{opacity:.45;cursor:not-allowed}
.res-card{border-radius:12px;padding:1.4rem;text-align:center;margin-top:.4rem}
.res-pass{background:rgba(34,197,94,.09);border:1px solid rgba(34,197,94,.38)}
.res-fail{background:rgba(239,68,68,.09);border:1px solid rgba(239,68,68,.38)}
.res-icon{font-size:2.8rem;margin-bottom:.6rem}
.res-title{font-size:1.3rem;font-weight:800;margin-bottom:.45rem}
.res-pass .res-title{color:#4ade80}.res-fail .res-title{color:#f87171}
.res-det{font-size:.84rem;color:#94a3b8;line-height:1.55}
.sec-note{font-size:.72rem;color:#475569;text-align:center;margin-top:.5rem;padding-bottom:1.5rem}
.info-row{font-size:.78rem;color:#64748b;text-align:center;
  background:#0f172a;border-radius:8px;padding:.55rem;margin:.5rem 0}
#s-idle,#s-verify,#s-result{display:none}
#s-idle.on,#s-verify.on,#s-result.on{display:block}
</style>
</head>
<body>
<div class="logo">🛡️ BaseTruth KYC</div>

<!-- IDLE -->
<div id="s-idle" class="card on">
  <h2 style="font-size:1.15rem;font-weight:700;margin-bottom:.45rem">Video Identity Verification</h2>
  <p style="font-size:.85rem;color:#94a3b8;line-height:1.6;margin-bottom:.9rem">
    You have been asked to complete a quick AI-powered identity check.<br>
    This takes about <strong style="color:#e2e8f0">30–60 seconds</strong> and runs entirely
    on our secure server. No data is shared with third parties.
  </p>
  <p style="font-size:.85rem;color:#94a3b8;line-height:1.6">
    <strong style="color:#c4b5fd">Prepare:</strong><br>
    · Good lighting — face a window or bright light source<br>
    · Position your face inside the oval guide when prompted<br>
    · Follow on-screen prompts carefully
  </p>
  <div class="info-row" id="cust-info"></div>
  <button class="btn" id="btn-start">Start Verification</button>
</div>

<!-- VERIFY -->
<div id="s-verify" class="card">
  <div class="video-wrap">
    <video id="vid" autoplay muted playsinline></video>
    <div class="oval"></div>
    <div class="badge b-idle" id="face-badge">Searching…</div>
  </div>
  <div class="ch-card" id="ch-card">
    <div class="ch-label" id="ch-label">Please wait…</div>
    <div class="ch-inst" id="ch-inst">Preparing your session.</div>
    <div class="prog-wrap"><div class="prog-fill" id="prog-fill" style="width:0%"></div></div>
    <div class="dots" id="dots"></div>
  </div>
  <div class="fb" id="fb-msg"></div>
</div>

<!-- RESULT -->
<div id="s-result" class="card">
  <div class="res-card" id="res-inner">
    <div class="res-icon" id="res-icon">⏳</div>
    <div class="res-title" id="res-title">Processing…</div>
    <div class="res-det" id="res-det"></div>
  </div>
</div>

<p class="sec-note">🔒 Video processed on BaseTruth secure servers. Not shared externally.</p>

<script>
const SESSION_ID       = '__SESSION_ID__';
const TOTAL_CHALLENGES = __CHALLENGES_COUNT__;
const CUSTOMER_NAME    = '__CUSTOMER_NAME__';
const CHALLENGES       = __CHALLENGES_JSON__;

const LABELS = {
  blink:      'CLOSE YOUR EYES',
  turn_left:  'TURN YOUR HEAD LEFT',
  turn_right: 'TURN YOUR HEAD RIGHT',
  nod:        'NOD YOUR HEAD',
};
const INSTR = {
  blink:      'Slowly close both eyes completely, then open them again',
  turn_left:  'Slowly turn your head to YOUR left (left ear toward camera)',
  turn_right: 'Slowly turn your head to YOUR right (right ear toward camera)',
  nod:        'Slowly nod your head down and then back up to center',
};

let ws = null, stream = null, captureTimer = null, done = 0;
let resultShown = false;  // guard: don't overwrite a result already displayed

const sIdle   = document.getElementById('s-idle');
const sVerify = document.getElementById('s-verify');
const sResult = document.getElementById('s-result');
const vid     = document.getElementById('vid');
const badge   = document.getElementById('face-badge');
const chLabel = document.getElementById('ch-label');
const chInst  = document.getElementById('ch-inst');
const prog    = document.getElementById('prog-fill');
const dots    = document.getElementById('dots');
const fb      = document.getElementById('fb-msg');

const ci = document.getElementById('cust-info');
if (CUSTOMER_NAME) ci.textContent = 'Session prepared for: ' + CUSTOMER_NAME;
else ci.style.display = 'none';

for (let i = 0; i < TOTAL_CHALLENGES; i++) {
  const d = document.createElement('div');
  d.className = 'dot'; d.id = 'dot' + i; dots.appendChild(d);
}

function show(name){
  [sIdle,sVerify,sResult].forEach(s=>s.classList.remove('on'));
  if(name==='idle')   sIdle.classList.add('on');
  if(name==='verify') sVerify.classList.add('on');
  if(name==='result') sResult.classList.add('on');
}

document.getElementById('btn-start').addEventListener('click', async ()=>{
  const btn = document.getElementById('btn-start');
  btn.disabled = true; btn.textContent = 'Opening camera…';
  try {
    stream = await navigator.mediaDevices.getUserMedia(
      {video:{facingMode:'user',width:{ideal:1280},height:{ideal:720}},audio:false});
    vid.srcObject = stream;
    await vid.play();
  } catch(e){
    alert('Camera access denied. Please allow camera access and reload the page.');
    btn.disabled = false; btn.textContent = 'Start Verification'; return;
  }
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(`${proto}//${location.host}/kyc/ws/${SESSION_ID}`);
  ws.onopen = ()=>{ show('verify'); startCapture(); };
  ws.onmessage = e=>{ try{ handle(JSON.parse(e.data)); }catch(_){} };
  ws.onerror = ()=>{ if(!resultShown) showResult(false,0,'Connection error — could not reach the server. Please try again.'); };
  ws.onclose = e=>{ if(!resultShown && e.code!==1000 && done<TOTAL_CHALLENGES) showResult(false,0,'Session disconnected.'); stopCapture(); };
});

function startCapture(){
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  captureTimer = setInterval(()=>{
    if(!ws || ws.readyState!==1) return;
    if(!vid.videoWidth) return;
    canvas.width  = 640;
    canvas.height = Math.round(640 * vid.videoHeight / vid.videoWidth);
    // Draw un-mirrored (raw camera data) — CSS mirrors the preview only
    ctx.drawImage(vid,0,0,canvas.width,canvas.height);
    canvas.toBlob(blob=>{
      if(!blob) return;
      const fr = new FileReader();
      fr.onloadend = ()=>{
        const b64 = fr.result.split(',')[1];
        if(ws && ws.readyState===1) ws.send(JSON.stringify({type:'frame',data:b64}));
      };
      fr.readAsDataURL(blob);
    },'image/jpeg',0.82);
  },310);
}

function stopCapture(){
  if(captureTimer){clearInterval(captureTimer);captureTimer=null;}
  if(stream){stream.getTracks().forEach(t=>t.stop());stream=null;}
}

function handle(msg){
  if(msg.type==='status')      updateUI(msg);
  else if(msg.type==='result'){ stopCapture(); if(ws)ws.close(1000); showResult(msg.passed,msg.display_score||0,msg.message||''); }
  else if(msg.type==='error'){  stopCapture(); showResult(false,0,msg.message||'Verification failed.'); }
}

function updateUI(msg){
  if(msg.face_detected){ badge.className='badge b-ok'; badge.textContent='✓ Face detected'; }
  else{                   badge.className='badge b-warn'; badge.textContent='Center your face'; }
  if(msg.challenge){
    chLabel.textContent = LABELS[msg.challenge] || msg.challenge.toUpperCase();
    chInst.textContent  = INSTR[msg.challenge]  || '';
  }
  done = msg.challenges_completed||0;
  const total = msg.total_challenges || TOTAL_CHALLENGES;
  prog.style.width = total>0 ? (done/total*100)+'%' : '0%';
  for(let i=0;i<total;i++){
    const d=document.getElementById('dot'+i);
    if(!d) continue;
    d.className = i<done ? 'dot done' : (i===done ? 'dot active' : 'dot');
  }
  if(msg.feedback){
    fb.textContent  = msg.feedback;
    fb.className    = 'fb' + (msg.challenge_just_passed ? ' pass' : '');
  }
}

function showResult(passed,score,message){
  resultShown = true;  // prevent onclose from re-showing
  show('result');
  const inner = document.getElementById('res-inner');
  const icon  = document.getElementById('res-icon');
  const title = document.getElementById('res-title');
  const det   = document.getElementById('res-det');
  if(passed){
    inner.className = 'res-card res-pass';
    icon.textContent = '✅';
    title.textContent = 'Identity Verified';
    det.innerHTML = 'Your identity has been successfully verified by BaseTruth AI.<br>'
      + '<span style="color:#4ade80">Match score: '+(score).toFixed(1)+'%</span><br><br>'
      + 'You may close this window.';
  } else {
    inner.className = 'res-card res-fail';
    icon.textContent = '❌';
    title.textContent = 'Verification Failed';
    det.innerHTML = (message || 'Verification could not be completed.')
      + '<br><br>Please contact the agent for assistance.';
  }
}

window.addEventListener('beforeunload',()=>{ stopCapture(); if(ws)ws.close(); });
</script>
</body>
</html>"""


def _service(artifact_root: str | Path | None = None) -> Any:
    from basetruth.service import BaseTruthService

    return BaseTruthService(artifact_root or _DEFAULT_ARTIFACT_ROOT)


def create_app(artifact_root: str | Path | None = None) -> Any:
    """Create and return the BaseTruth FastAPI application."""
    if not _FASTAPI_AVAILABLE:
        raise ImportError(
            "FastAPI is required for the API server. "
            "Install the BaseTruth api extra: pip install 'basetruth[api]'"
        )

    app = FastAPI(
        title="BaseTruth API",
        description=(
            "Explainable document integrity and fraud detection REST API. "
            "Scan documents, browse cases, update investigator workflows, "
            "and retrieve reports programmatically."
        ),
        version="0.1.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    svc = _service(artifact_root)

    # -------------------------------------------------------------------------
    # Endpoints
    # -------------------------------------------------------------------------

    @app.get("/api/v1/health", tags=["System"])
    def health() -> Dict[str, Any]:
        """Return product and service health information."""
        return {
            "status": "ok",
            "product": "BaseTruth",
            "version": "0.1.0",
            "artifact_root": str(svc.artifact_root),
        }

    @app.post("/api/v1/scan", tags=["Scan"])
    def scan_by_path(request: ScanPathRequest) -> Dict[str, Any]:
        """Scan a document at an existing path on the server file system."""
        path = Path(request.path)
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {request.path}")
        try:
            return svc.scan_document(path)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/api/v1/scan/upload", tags=["Scan"])
    async def scan_upload(file: UploadFile = File(...)) -> Dict[str, Any]:
        """Upload a document and scan it in one request."""
        import tempfile

        suffix = Path(file.filename or "upload").suffix or ".pdf"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = Path(tmp.name)
        try:
            return svc.scan_document(tmp_path)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass

    @app.get("/api/v1/reports", tags=["Reports"])
    def list_reports(
        kind: Optional[str] = Query(None, description="Filter by kind: verification | comparison"),
        risk_level: Optional[str] = Query(None, description="Filter by risk level: high | medium | low"),
    ) -> List[Dict[str, Any]]:
        """List all reports, optionally filtered by kind and risk level."""
        reports = svc.list_reports()
        if kind:
            reports = [r for r in reports if r.get("kind") == kind]
        if risk_level:
            reports = [r for r in reports if r.get("risk_level") == risk_level]
        return reports

    @app.get("/api/v1/cases", tags=["Cases"])
    def list_cases(
        status: Optional[str] = Query(None, description="Filter by workflow status."),
        priority: Optional[str] = Query(None, description="Filter by priority."),
        disposition: Optional[str] = Query(None, description="Filter by disposition."),
    ) -> List[Dict[str, Any]]:
        """List all cases, optionally filtered by workflow state."""
        cases = svc.list_cases()
        if status:
            cases = [c for c in cases if c.get("status") == status]
        if priority:
            cases = [c for c in cases if c.get("priority") == priority]
        if disposition:
            cases = [c for c in cases if c.get("disposition") == disposition]
        return cases

    @app.get("/api/v1/cases/{case_key}", tags=["Cases"])
    def get_case(case_key: str) -> Dict[str, Any]:
        """Get full case detail including workflow state and all linked reports."""
        try:
            return svc.get_case_detail(case_key)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Case not found: {case_key}")

    @app.patch("/api/v1/cases/{case_key}", tags=["Cases"])
    def update_case(case_key: str, request: UpdateCaseRequest) -> Dict[str, Any]:
        """Update the workflow state of a case, optionally appending a note."""
        return svc.update_case(
            case_key,
            status=request.status,
            disposition=request.disposition,
            priority=request.priority,
            assignee=request.assignee,
            labels=request.labels,
            note_text=request.note_text,
            note_author=request.note_author,
        )

    # ── Video KYC — challenge-based liveness + face-match ─────────────────────

    import asyncio as _asyncio
    import base64 as _base64
    import json as _json
    import random as _random
    import threading as _threading

    import cv2 as _cv2
    import numpy as _np
    from fastapi.responses import HTMLResponse as _HTMLResponse

    from basetruth.kyc.session import ALL_CHALLENGES, SessionStore
    from basetruth.kyc.liveness import analyze_challenge, extract_features, run_face_match
    from basetruth.vision.face import get_face_analyzer
    from basetruth.logger import get_logger as _get_logger

    _kyc_log = _get_logger("basetruth.kyc")

    # One store per application instance (survives across requests)
    _kyc_store    = SessionStore()
    _kyc_face_lock = _threading.Lock()

    def _process_kyc_frame(session: Any, b64_frame: str) -> Dict[str, Any]:
        """CPU-bound per-frame analysis — called in a thread-pool executor."""
        try:
            raw   = _base64.b64decode(b64_frame)
            nparr = _np.frombuffer(raw, _np.uint8)
            img   = _cv2.imdecode(nparr, _cv2.IMREAD_COLOR)
            if img is None:
                return {"type": "status", "face_detected": False, "feedback": "Invalid frame."}
        except Exception:
            return {"type": "status", "face_detected": False, "feedback": "Decode error."}

        face_app = get_face_analyzer()
        with _kyc_face_lock:
            faces = face_app.get(img)

        if not faces:
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
        # Store the last clear frame for later use in PDF reports
        session.last_live_frame_bytes = raw

        if session.all_done:
            return _finish_session(session, face)

        features = extract_features(face)
        history  = session.current_frame_history()
        history.append(features)

        current_ch = session.current_challenge
        analysis   = analyze_challenge(history, current_ch)
        just_passed = False
        if analysis["passed"]:
            session.advance_challenge()
            just_passed = True
            if session.all_done:
                return _finish_session(session, face)

        return {
            "type": "status",
            "face_detected": True,
            "challenge": current_ch,
            "challenges_completed": session.current_challenge_idx,
            "total_challenges": len(session.challenges),
            "feedback": analysis["feedback"],
            "challenge_just_passed": just_passed,
        }

    def _finish_session(session: Any, face: Any) -> Dict[str, Any]:
        """Called once all liveness challenges pass — runs the face-match check."""
        if session.reference_embedding_b64:
            match = run_face_match(face, session.reference_embedding_b64)
            session.status = "completed" if match["passed"] else "failed"
            session.result = match
            return {"type": "result", **match}
        # No reference embedding → liveness-only session
        result = {
            "passed": True,
            "match_score": 1.0,
            "display_score": 100.0,
            "cosine_similarity": 1.0,
            "message": "Liveness checks passed (no ID reference provided).",
        }
        session.status = "completed"
        session.result = result
        return {"type": "result", **result}

    @app.post("/kyc/sessions", tags=["Video KYC"])
    def create_kyc_session(req: CreateKYCSessionRequest) -> Dict[str, Any]:
        """Create a challenge-based Video KYC session. Returns a session URL."""
        challenges = req.challenges or _random.sample(ALL_CHALLENGES, k=2)
        session = _kyc_store.create(
            challenges=challenges,
            reference_embedding_b64=req.reference_embedding_b64,
            customer_name=req.customer_name,
            entity_ref=req.entity_ref,
        )
        _kyc_log.info(
            "KYC session created",
            extra={
                "session_id": session.session_id,
                "customer_name": req.customer_name,
                "entity_ref": req.entity_ref,
                "challenges": challenges,
            },
        )
        return {
            **session.to_status_dict(),
            "session_url": f"/kyc/{session.session_id}",
        }

    @app.get("/kyc/sessions/{session_id}", tags=["Video KYC"])
    def get_kyc_session_status(session_id: str) -> Dict[str, Any]:
        """Poll the status of a KYC session from the agent dashboard."""
        session = _kyc_store.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found or expired.")
        return session.to_status_dict()

    @app.get("/kyc/{session_id}", response_class=_HTMLResponse, tags=["Video KYC"])
    def kyc_session_page(session_id: str) -> Any:
        """Serve the customer-facing Video KYC browser page."""
        session = _kyc_store.get(session_id)
        if not session:
            return _HTMLResponse(
                "<html><body style='font-family:sans-serif;background:#0f172a;color:#f87171;"
                "display:flex;justify-content:center;align-items:center;height:100vh;margin:0'>"
                "<h2>Session not found or has expired.</h2></body></html>",
                status_code=404,
            )
        html = _KYC_PAGE_HTML.replace("__SESSION_ID__", session.session_id)
        html = html.replace("__CHALLENGES_COUNT__", str(len(session.challenges)))
        html = html.replace("__CUSTOMER_NAME__", session.customer_name or "")
        html = html.replace("__CHALLENGES_JSON__", _json.dumps(session.challenges))
        return _HTMLResponse(html)

    @app.websocket("/kyc/ws/{session_id}")
    async def kyc_websocket(websocket: WebSocket, session_id: str) -> None:
        """WebSocket: browser streams base64 JPEG frames; server replies with JSON status/result."""
        await websocket.accept()
        session = _kyc_store.get(session_id)
        if not session:
            _kyc_log.warning("KYC WS rejected — session not found", extra={"session_id": session_id})
            await websocket.send_json({"type": "error", "message": "Session not found or expired."})
            await websocket.close(code=1008)
            return
        if session.status not in ("waiting", "active"):
            _kyc_log.warning(
                "KYC WS rejected — wrong status",
                extra={"session_id": session_id, "status": session.status},
            )
            await websocket.send_json({"type": "error", "message": f"Session is {session.status}."})
            await websocket.close(code=1008)
            return

        session.status = "active"
        _kyc_log.info("KYC WebSocket connected", extra={"session_id": session_id})
        loop = _asyncio.get_running_loop()
        _clean_exit = False
        try:
            while True:
                try:
                    data = await _asyncio.wait_for(websocket.receive_json(), timeout=15.0)
                except _asyncio.TimeoutError:
                    # Client went silent — send a gentle nudge and keep waiting
                    try:
                        await websocket.send_json({"type": "status", "face_detected": False,
                                                   "feedback": "No frames received — check your camera."})
                    except Exception:
                        pass
                    continue
                except WebSocketDisconnect:
                    _clean_exit = True
                    break
                if data.get("type") != "frame":
                    continue
                b64_frame = data.get("data", "")
                if not b64_frame:
                    continue
                try:
                    result = await loop.run_in_executor(None, _process_kyc_frame, session, b64_frame)
                except Exception as _frame_exc:
                    # Surface the real error to the browser instead of silently disconnecting.
                    _err_msg = str(_frame_exc) or "Frame processing error."
                    _kyc_log.error(
                        "KYC frame processing error",
                        extra={"session_id": session_id, "error": _err_msg},
                    )
                    try:
                        await websocket.send_json({"type": "error", "message": _err_msg})
                    except Exception:
                        pass
                    _clean_exit = True
                    break
                await websocket.send_json(result)
                if result.get("type") == "result":
                    _kyc_log.info(
                        "KYC session result",
                        extra={
                            "session_id": session_id,
                            "passed": result.get("passed"),
                            "score": result.get("display_score"),
                        },
                    )
                    _clean_exit = True
                    break
        except Exception:
            pass
        finally:
            if session.status == "active":
                session.status = "failed" if not _clean_exit else session.status
            try:
                await websocket.close(code=1000)
            except Exception:
                pass

    # ── Entity registry endpoints ─────────────────────────────────────────

    @app.get("/api/v1/entities", tags=["Entities"])
    def list_entities(
        q: str = Query("", description="Search term — name, PAN, Aadhaar, email, or phone."),
        field: str = Query("all", description="Field to search: all | name | pan | aadhar | email | phone"),
        limit: int = Query(100, ge=1, le=1000, description="Maximum rows to return."),
    ) -> List[Dict[str, Any]]:
        """Search the entity registry.

        Returns the most-recent entities when no query is supplied.
        Each result includes scan count and latest risk level for quick triage.
        """
        try:
            from basetruth.store import search_entities
            return search_entities(query=q, search_field=field, limit=limit)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"DB unavailable: {exc}") from exc

    @app.get("/api/v1/entities/{entity_ref}", tags=["Entities"])
    def get_entity(entity_ref: str) -> Dict[str, Any]:
        """Return full detail for one entity including all scans summary."""
        try:
            from basetruth.store import get_entity_scans, search_entities
            matches = search_entities(query=entity_ref, search_field="all", limit=1)
            if not matches:
                raise HTTPException(status_code=404, detail=f"Entity not found: {entity_ref}")
            entity = matches[0]
            entity["scans"] = get_entity_scans(entity_ref)
            return entity
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"DB unavailable: {exc}") from exc

    @app.get("/api/v1/entities/{entity_ref}/scans", tags=["Entities"])
    def list_entity_scans(entity_ref: str) -> List[Dict[str, Any]]:
        """List all document scans for a specific entity (most-recent first).

        Each item includes the full JSON report so analysts can review the
        signals that led to a flag without needing to access the filesystem.
        """
        try:
            from basetruth.store import get_entity_scans
            scans = get_entity_scans(entity_ref)
            if not scans:
                # Could be entity not found or just no scans yet — disambiguate
                from basetruth.store import search_entities
                if not search_entities(query=entity_ref, search_field="all", limit=1):
                    raise HTTPException(status_code=404, detail=f"Entity not found: {entity_ref}")
            return scans
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"DB unavailable: {exc}") from exc

    @app.get("/api/v1/scans/{scan_id}/report.pdf", tags=["Reports"])
    def download_scan_pdf(scan_id: int) -> Any:
        """Download the PDF audit report for a specific scan.

        Returns the PDF binary so auditors can save it locally or attach it
        to a case without needing access to the server filesystem.

        Use this endpoint when explaining a fraud flag to an auditor:
        look up the entity, find the scan ID, call this endpoint, share the PDF.
        """
        try:
            from fastapi.responses import Response
            from basetruth.store import get_scan_pdf
            pdf_bytes = get_scan_pdf(scan_id)
            if not pdf_bytes:
                raise HTTPException(
                    status_code=404,
                    detail=f"PDF report not found for scan {scan_id}. "
                           "The scan may not have generated a PDF, or the DB is unavailable.",
                )
            return Response(
                content=pdf_bytes,
                media_type="application/pdf",
                headers={
                    "Content-Disposition": f'attachment; filename="scan_{scan_id}_report.pdf"',
                    "Content-Length": str(len(pdf_bytes)),
                },
            )
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"DB unavailable: {exc}") from exc

    @app.get("/api/v1/scans/recent", tags=["Reports"])
    def list_recent_scans(
        limit: int = Query(50, ge=1, le=500, description="Number of most-recent scans to return."),
    ) -> List[Dict[str, Any]]:
        """Return the most-recent scans across all entities with their risk levels.

        Useful for a fraud-monitoring dashboard.
        """
        try:
            from basetruth.store import list_recent_scans as _list
            return _list(limit=limit)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"DB unavailable: {exc}") from exc

    @app.get("/api/v1/db/stats", tags=["System"])
    def db_stats() -> Dict[str, Any]:
        """Return entity, scan, and high-risk counts from the database."""
        try:
            from basetruth.store import db_stats as _stats
            return _stats()
        except Exception as exc:
            return {"error": str(exc), "entities": 0, "scans": 0, "high_risk": 0}

    return app


# ---------------------------------------------------------------------------
# Module-level application instance for uvicorn / importability.
# Gracefully sets app=None when FastAPI is not installed.
# ---------------------------------------------------------------------------

try:
    app = create_app()
except ImportError:
    app = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# CLI entry for running the API server directly.
# ---------------------------------------------------------------------------

def _serve(host: str = "0.0.0.0", port: int = 8502, artifact_root: str | None = None) -> None:  # pragma: no cover
    try:
        import uvicorn  # type: ignore
    except ImportError:
        print(
            "uvicorn is required to run the BaseTruth API server. "
            "Install with: pip install 'basetruth[api]'",
            file=sys.stderr,
        )
        sys.exit(1)

    _app = create_app(artifact_root)
    uvicorn.run(_app, host=host, port=port, ws="websockets-sansio")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(prog="basetruth.api", description="BaseTruth API server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8502)
    parser.add_argument("--artifact-root", default=None)
    args = parser.parse_args()
    _serve(args.host, args.port, args.artifact_root)
