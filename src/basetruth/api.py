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

    class CreateKYCSessionRequest(BaseModel):
        customer_name:           str            = Field("", description="Customer display name.")
        entity_ref:              str            = Field("", description="Entity / case reference ID.")
        challenges:              List[str]      = Field([], description="Liveness challenges to present.")
        reference_embedding_b64: Optional[str]  = Field(None, description="Base-64 ArcFace embedding from the reference ID document.")
        # Address-proof fields — optional; supplied when the operator uploads an address proof
        # document before starting the Video KYC session.
        address_dtls:            Optional[Dict] = Field(None, description="Extracted fields from the address proof document (Aadhaar / Passport).")
        reference_doc_filename:  Optional[str]  = Field(None, description="Original filename of the reference identity document.")

    class WebRTCOfferRequest(BaseModel):
        sdp:  str = Field(..., description="SDP offer string from RTCPeerConnection.createOffer().")
        type: str = Field("offer", description="SDP type — always 'offer'.")

    # ── Response models — explicit schemas rendered in the OpenAPI / Swagger spec ──

    class HealthResponse(BaseModel):
        status:        str = Field(..., description="Service status — always 'ok' when healthy.", examples=["ok"])
        product:       str = Field(..., description="Product name.", examples=["BaseTruth"])
        version:       str = Field(..., description="API version string.", examples=["1.0.0"])
        artifact_root: str = Field(..., description="Filesystem path where scan artefacts are stored.")

    class ForensicScanResponse(BaseModel):
        filename:            str            = Field(..., description="Original filename of the uploaded document.")
        document_type:       str            = Field(..., description="Detected document type, e.g. Aadhaar, Passport, Payslip.", examples=["Aadhaar"])
        is_image_based:      bool           = Field(..., description="True when the file is a scanned image or image-PDF.")
        forensic_verdict:    str            = Field(..., description="Final verdict: GENUINE, UNCERTAIN, LIKELY TAMPERED, or TAMPERED.", examples=["GENUINE"])
        forgery_score_0_100: float          = Field(..., ge=0, le=100, description="Forgery probability — 0 means genuine, 100 means tampered.", examples=[12.5])
        overall_explanation: str            = Field(..., description="Technical explanation of the forensic signals found.")
        honest_review:       str            = Field(..., description="Plain-English verdict written for non-technical reviewers.")
        evidence:            List[str]      = Field(..., description="Bullet-point list of specific forensic evidence items.")
        layers:              Dict[str, Any] = Field(..., description="Raw per-layer metrics: ELA, DCT, clone detection, noise, metadata, AI artefacts, etc.")

    class DocumentExtractResponse(BaseModel):
        filename:         str            = Field(..., description="Original filename.")
        document_type:    str            = Field(..., description="Classified document type.", examples=["Payslip"])
        is_image_based:   bool           = Field(..., description="True for scanned images; False for digital PDFs.")
        confidence:       float          = Field(..., ge=0, le=1, description="Classification confidence (0–1).", examples=[0.94])
        extracted_fields: Dict[str, Any] = Field(..., description="Key-value pairs extracted from the document — name, DOB, ID number, amounts, etc.")

    class KYCSessionCreateResponse(BaseModel):
        session_id:    str       = Field(..., description="Unique session token.")
        status:        str       = Field(..., description="Session lifecycle status.", examples=["waiting"])
        session_url:   str       = Field(..., description="URL of the customer-facing KYC page. Send this to the end-user via email or SMS.")
        challenges:    List[str] = Field(..., description="Liveness challenges assigned to this session.", examples=[["blink", "nod"]])
        customer_name: str       = Field(default="", description="Customer display name, if provided.")
        entity_ref:    str       = Field(default="", description="Entity / case reference, if provided.")

    class DBStatsResponse(BaseModel):
        entities:  int = Field(..., description="Total entities in the registry.")
        scans:     int = Field(..., description="Total document scans stored.")
        high_risk: int = Field(..., description="Scans flagged as high risk.")

    class LocationData(BaseModel):
        """GPS coordinates sent by the customer browser in Step 3 of the KYC wizard."""
        lat:      float = Field(..., description="Latitude in decimal degrees.")
        lon:      float = Field(..., description="Longitude in decimal degrees.")
        accuracy: float = Field(0.0, description="Accuracy radius in metres reported by the browser.")

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
.logo{margin:1.2rem 0 0.5rem;font-size:1.35rem;font-weight:800;
  background:linear-gradient(135deg,#6366f1,#8b5cf6);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent}
.card{background:#1e293b;border:1px solid #334155;border-radius:16px;
  padding:1.4rem 1.25rem;width:100%;max-width:460px;margin-bottom:0.75rem}
/* ── Step indicator ──────────────────────────────────────────────── */
.step-bar{display:flex;align-items:center;justify-content:center;
  gap:.2rem;width:100%;max-width:460px;margin:.4rem 0 .65rem;
  padding:.6rem .75rem;background:#1e293b;border:1px solid #334155;border-radius:12px}
.s-item{display:flex;flex-direction:column;align-items:center;flex:1;gap:.15rem}
.s-num{width:22px;height:22px;border-radius:50%;display:flex;align-items:center;
  justify-content:center;font-size:.7rem;font-weight:800;
  background:#0f172a;color:#475569;border:1.5px solid #334155;transition:all .25s}
.s-lbl{font-size:.58rem;color:#475569;font-weight:600;letter-spacing:.03em;
  white-space:nowrap;transition:color .25s}
.s-sep{color:#334155;font-size:.75rem;flex-shrink:0;margin-bottom:.85rem}
.s-item.active .s-num{background:rgba(99,102,241,.2);color:#818cf8;border-color:#6366f1}
.s-item.active .s-lbl{color:#818cf8}
.s-item.done .s-num{background:rgba(34,197,94,.15);color:#4ade80;border-color:#4ade80}
.s-item.done .s-lbl{color:#4ade80}
/* ── Upload zone ─────────────────────────────────────────────────── */
.uz{border:2px dashed #334155;border-radius:12px;padding:1.4rem;text-align:center;
  cursor:pointer;transition:border-color .2s,background .2s;min-height:110px;
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  gap:.55rem;position:relative;margin:.7rem 0}
.uz:hover{border-color:#6366f1;background:rgba(99,102,241,.04)}
.uz.has-file{border-color:#4ade80;background:rgba(34,197,94,.04)}
.uz-icon{font-size:1.7rem;opacity:.65}
.uz-lbl{font-size:.82rem;color:#64748b;line-height:1.5}
.thumb{width:100%;max-height:130px;object-fit:contain;border-radius:8px;
  display:none;border:1px solid #334155;margin-top:.4rem}
/* ── Buttons ─────────────────────────────────────────────────────── */
.btn{display:block;width:100%;padding:.8rem;
  background:linear-gradient(135deg,#4f46e5,#6366f1);color:#fff;
  border:none;border-radius:10px;font-size:1rem;font-weight:700;
  cursor:pointer;margin-top:.75rem;transition:opacity .2s}
.btn:hover{opacity:.88}
.btn:disabled{opacity:.45;cursor:not-allowed}
.btn-skip{background:transparent;border:1px solid #334155;color:#64748b;
  font-weight:500;font-size:.82rem;padding:.55rem;margin-top:.35rem}
.btn-skip:hover{border-color:#4f46e5;color:#818cf8;opacity:1}
.btn-loc{background:linear-gradient(135deg,#0ea5e9,#0284c7)}
/* ── Location result ─────────────────────────────────────────────── */
.loc-box{background:#0f172a;border:1px solid #334155;border-radius:10px;
  padding:.7rem 1rem;margin-top:.65rem;font-size:.82rem;color:#94a3b8;line-height:1.6}
.loc-addr{color:#e2e8f0;font-weight:600;margin-bottom:.25rem}
/* ── Camera / liveness ───────────────────────────────────────────── */
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
.ch-label{font-size:1.15rem;font-weight:800;color:#c4b5fd;margin-bottom:.35rem;letter-spacing:.04em}
.ch-inst{font-size:.84rem;color:#94a3b8;line-height:1.55}
.prog-wrap{background:#0f172a;border-radius:99px;height:7px;margin-top:.7rem;overflow:hidden}
.prog-fill{height:100%;border-radius:99px;
  background:linear-gradient(90deg,#6366f1,#8b5cf6);transition:width .35s ease}
.dots{display:flex;gap:.45rem;justify-content:center;margin-top:.65rem}
.dot{width:10px;height:10px;border-radius:50%;border:2px solid #475569;
  background:transparent;transition:all .2s}
.dot.active{border-color:#6366f1;background:#6366f1}
.dot.done{border-color:#4ade80;background:#4ade80}
.fb{text-align:center;font-size:.88rem;margin-top:.6rem;min-height:1.2em;
  color:#94a3b8;transition:color .2s}
.fb.pass{color:#4ade80}
.fb.fail{color:#f87171}
/* ── Result ──────────────────────────────────────────────────────── */
.res-card{border-radius:12px;padding:1.4rem;text-align:center;margin-top:.4rem}
.res-pass{background:rgba(34,197,94,.09);border:1px solid rgba(34,197,94,.38)}
.res-fail{background:rgba(239,68,68,.09);border:1px solid rgba(239,68,68,.38)}
.res-icon{font-size:2.8rem;margin-bottom:.6rem}
.res-title{font-size:1.3rem;font-weight:800;margin-bottom:.45rem}
.res-pass .res-title{color:#4ade80}
.res-fail .res-title{color:#f87171}
.res-det{font-size:.84rem;color:#94a3b8;line-height:1.55}
.addr-row{background:#0f172a;border-radius:8px;padding:.5rem .75rem;
  margin-top:.65rem;font-size:.78rem;color:#64748b;text-align:left;line-height:1.7}
.addr-row strong{color:#94a3b8}
.sec-note{font-size:.72rem;color:#475569;text-align:center;
  margin-top:.5rem;padding-bottom:1.5rem}
</style>
</head>
<body>
<div class="logo">🛡️ BaseTruth KYC</div>

<!-- Step indicator — hidden until the wizard starts -->
<div class="step-bar" id="step-bar" style="display:none">
  <div class="s-item" id="st1"><div class="s-num">1</div><div class="s-lbl">Upload ID</div></div>
  <div class="s-sep">›</div>
  <div class="s-item" id="st2"><div class="s-num">2</div><div class="s-lbl">Address</div></div>
  <div class="s-sep">›</div>
  <div class="s-item" id="st3"><div class="s-num">3</div><div class="s-lbl">Location</div></div>
  <div class="s-sep">›</div>
  <div class="s-item" id="st4"><div class="s-num">4</div><div class="s-lbl">Verify</div></div>
</div>

<!-- IDLE: welcome screen -->
<div id="s-idle" class="card">
  <h2 style="font-size:1.15rem;font-weight:700;margin-bottom:.45rem">Video Identity Verification</h2>
  <p style="font-size:.85rem;color:#94a3b8;line-height:1.6;margin-bottom:.85rem">
    You have been asked to complete an AI-powered identity check.
    This takes about <strong style="color:#e2e8f0">1–2 minutes</strong> and runs fully in your browser.
  </p>
  <p style="font-size:.85rem;color:#94a3b8;line-height:1.65;margin-bottom:.9rem">
    <strong style="color:#c4b5fd">You will need:</strong><br>
    · Aadhaar card (front) <em>or</em> PAN card<br>
    · Aadhaar card (back) <em>or</em> Passport for address verification<br>
    · Camera permission for the face liveness check<br>
    · Location permission for address verification
  </p>
  <div id="cust-info" style="display:none;background:rgba(99,102,241,.1);border:1px solid rgba(99,102,241,.3);
    border-radius:8px;padding:.5rem .75rem;font-size:.84rem;color:#c4b5fd;margin-bottom:.75rem"></div>
  <button class="btn" id="btn-start">Begin Verification</button>
</div>

<!-- STEP 1: Upload ID document -->
<div id="s-1" class="card" style="display:none">
  <h3 style="font-size:1rem;font-weight:700;margin-bottom:.4rem">Step 1 · Upload Your ID</h3>
  <p style="font-size:.83rem;color:#94a3b8;line-height:1.55;margin-bottom:.1rem">
    Upload a clear, well-lit photo of your
    <strong style="color:#e2e8f0">Aadhaar card (front)</strong> or
    <strong style="color:#e2e8f0">PAN card</strong>.
    The face photo on the ID will be used to verify your identity.
  </p>
  <div class="uz" id="uz1">
    <input type="file" id="f-id" accept="image/jpeg,image/png,image/webp" style="display:none">
    <div class="uz-icon">🪪</div>
    <div class="uz-lbl" id="uz1-lbl">Tap to select photo</div>
    <img id="prev1" class="thumb" alt="ID preview">
  </div>
  <div class="fb" id="fb1"></div>
  <button class="btn" id="btn1-next" disabled>Continue →</button>
</div>

<!-- STEP 2: Upload address proof -->
<div id="s-2" class="card" style="display:none">
  <h3 style="font-size:1rem;font-weight:700;margin-bottom:.4rem">Step 2 · Address Proof</h3>
  <p style="font-size:.83rem;color:#94a3b8;line-height:1.55;margin-bottom:.1rem">
    Upload the <strong style="color:#e2e8f0">back side of your Aadhaar card</strong> or
    the <strong style="color:#e2e8f0">address page of your Passport</strong>.
    We will verify your registered address against your live location.
  </p>
  <div class="uz" id="uz2">
    <input type="file" id="f-addr" accept="image/jpeg,image/png,image/webp" style="display:none">
    <div class="uz-icon">📄</div>
    <div class="uz-lbl" id="uz2-lbl">Tap to select photo</div>
    <img id="prev2" class="thumb" alt="Address proof preview">
  </div>
  <div class="fb" id="fb2"></div>
  <button class="btn" id="btn2-next" disabled>Continue →</button>
  <button class="btn btn-skip" id="btn2-skip">Skip — proceed without address verification</button>
</div>

<!-- STEP 3: Share live GPS location -->
<div id="s-3" class="card" style="display:none">
  <h3 style="font-size:1rem;font-weight:700;margin-bottom:.4rem">Step 3 · Share Your Location</h3>
  <p style="font-size:.83rem;color:#94a3b8;line-height:1.55;margin-bottom:.75rem">
    Allow location access so we can verify your current address is within range
    of your registered address on the proof document.
    Your coordinates are processed securely and not stored after verification.
  </p>
  <button class="btn btn-loc" id="btn-loc">📍  Share My Location</button>
  <div class="loc-box" id="loc-box" style="display:none">
    <div class="loc-addr" id="loc-addr"></div>
    <div id="loc-match" style="font-size:.78rem;margin-top:.2rem"></div>
  </div>
  <div class="fb" id="fb3"></div>
  <button class="btn" id="btn3-next" style="display:none">Continue to Verification →</button>
  <button class="btn btn-skip" id="btn3-skip">Skip — proceed without location check</button>
</div>

<!-- STEP 4: Liveness challenge -->
<div id="s-4" class="card" style="display:none">
  <div class="video-wrap">
    <video id="vid" autoplay muted playsinline></video>
    <div class="oval"></div>
    <div class="badge b-idle" id="face-badge">Searching…</div>
  </div>
  <div class="ch-card">
    <div class="ch-label" id="ch-label">Please wait…</div>
    <div class="ch-inst"  id="ch-inst">Starting camera…</div>
    <div class="prog-wrap"><div class="prog-fill" id="prog-fill" style="width:0%"></div></div>
    <div class="dots" id="dots"></div>
  </div>
  <div class="fb" id="fb4"></div>
</div>

<!-- RESULT screen -->
<div id="s-result" class="card" style="display:none">
  <div class="res-card" id="res-inner">
    <div class="res-icon"  id="res-icon">⏳</div>
    <div class="res-title" id="res-title">Processing…</div>
    <div class="res-det"   id="res-det"></div>
  </div>
  <div class="addr-row" id="addr-summary" style="display:none"></div>
</div>

<p class="sec-note">🔒 All data is processed on BaseTruth secure servers and is not shared externally.</p>

<script>
// Session constants injected server-side at request time
const SESSION_ID       = '__SESSION_ID__';
const TOTAL_CHALLENGES = __CHALLENGES_COUNT__;
const CUSTOMER_NAME    = '__CUSTOMER_NAME__';
const CHALLENGES       = __CHALLENGES_JSON__;

// Human-readable labels and instructions for each challenge type
const LABELS = {
  blink:      'CLOSE YOUR EYES',
  turn_left:  'TURN YOUR HEAD LEFT',
  turn_right: 'TURN YOUR HEAD RIGHT',
  nod:        'NOD YOUR HEAD',
};
const INSTR = {
  blink:      'Slowly close both eyes completely, then open them again',
  turn_left:  'Slowly turn your head to YOUR left',
  turn_right: 'Slowly turn your head to YOUR right',
  nod:        'Slowly nod your head down then back up to center',
};

let ws=null, stream=null, captureTimer=null, resultShown=false;

// ── Screen manager ────────────────────────────────────────────────────────
function show(id){
  ['s-idle','s-1','s-2','s-3','s-4','s-result'].forEach(s=>{
    const el=document.getElementById(s);
    if(el) el.style.display = s===id ? 'block' : 'none';
  });
}

// Update step bar: mark steps < n as done, step n as active, rest default
function setStep(n){
  const bar=document.getElementById('step-bar');
  if(bar) bar.style.display='flex';
  [1,2,3,4].forEach(i=>{
    const el=document.getElementById('st'+i);
    if(!el) return;
    el.className='s-item'+(i===n?' active':i<n?' done':'');
  });
}

// Show feedback message with optional CSS class (pass, fail, or '')
function setFb(id,msg,cls){
  const el=document.getElementById(id);
  if(!el) return;
  el.textContent=msg;
  el.className='fb'+(cls?' '+cls:'');
}

// ── Customer name banner ──────────────────────────────────────────────────
if(CUSTOMER_NAME){
  const ci=document.getElementById('cust-info');
  ci.textContent='Session prepared for: '+CUSTOMER_NAME;
  ci.style.display='block';
}

// ── Idle → Step 1 ─────────────────────────────────────────────────────────
document.getElementById('btn-start').onclick=()=>{ show('s-1'); setStep(1); };

// ── Step 1: Upload ID document ────────────────────────────────────────────
const uz1=document.getElementById('uz1');
const fId=document.getElementById('f-id');
uz1.onclick=()=>fId.click();

fId.onchange=e=>{
  const file=e.target.files[0]; if(!file) return;
  const r=new FileReader();
  r.onload=ev=>{
    // Show thumbnail and enable Continue button once file is selected
    const img=document.getElementById('prev1');
    img.src=ev.target.result; img.style.display='block';
    document.getElementById('uz1-lbl').textContent=file.name;
    uz1.classList.add('has-file');
    document.getElementById('btn1-next').disabled=false;
    setFb('fb1','','');
  };
  r.readAsDataURL(file);
};

document.getElementById('btn1-next').onclick=async()=>{
  const file=fId.files[0]; if(!file) return;
  const btn=document.getElementById('btn1-next');
  btn.disabled=true; btn.textContent='Processing…';
  setFb('fb1','⏳ Extracting face from ID…','');

  const fd=new FormData(); fd.append('file',file);
  try{
    const resp=await fetch(`/kyc/sessions/${SESSION_ID}/upload-id`,{method:'POST',body:fd});
    const data=await resp.json();
    if(resp.ok && data.face_found){
      setFb('fb1','✓ ID processed — face extracted successfully','pass');
      setTimeout(()=>{ show('s-2'); setStep(2); },700);
    } else {
      const msg=data.detail||data.message||'Could not extract face. Try a clearer, well-lit photo.';
      setFb('fb1','✗ '+msg,'fail');
      btn.disabled=false; btn.textContent='Try Again';
    }
  } catch{
    setFb('fb1','✗ Upload failed. Please check your connection and try again.','fail');
    btn.disabled=false; btn.textContent='Continue →';
  }
};

// ── Step 2: Upload address proof ──────────────────────────────────────────
const uz2=document.getElementById('uz2');
const fAddr=document.getElementById('f-addr');
uz2.onclick=()=>fAddr.click();

fAddr.onchange=e=>{
  const file=e.target.files[0]; if(!file) return;
  const r=new FileReader();
  r.onload=ev=>{
    const img=document.getElementById('prev2');
    img.src=ev.target.result; img.style.display='block';
    document.getElementById('uz2-lbl').textContent=file.name;
    uz2.classList.add('has-file');
    document.getElementById('btn2-next').disabled=false;
    setFb('fb2','','');
  };
  r.readAsDataURL(file);
};

document.getElementById('btn2-next').onclick=async()=>{
  const file=fAddr.files[0]; if(!file) return;
  const btn=document.getElementById('btn2-next');
  btn.disabled=true; btn.textContent='Processing…';
  setFb('fb2','⏳ Extracting address from document…','');

  const fd=new FormData(); fd.append('file',file);
  try{
    const resp=await fetch(`/kyc/sessions/${SESSION_ID}/upload-address`,{method:'POST',body:fd});
    const data=await resp.json();
    if(resp.ok){
      setFb('fb2','✓ Address proof uploaded','pass');
      setTimeout(()=>{ show('s-3'); setStep(3); },700);
    } else {
      const msg=data.detail||data.message||'Could not process document. Try a clearer photo.';
      setFb('fb2','✗ '+msg,'fail');
      btn.disabled=false; btn.textContent='Try Again';
    }
  } catch{
    setFb('fb2','✗ Upload failed. Please check your connection and try again.','fail');
    btn.disabled=false; btn.textContent='Continue →';
  }
};

document.getElementById('btn2-skip').onclick=()=>{ show('s-3'); setStep(3); };

// ── Step 3: Share GPS location ────────────────────────────────────────────
document.getElementById('btn-loc').onclick=()=>{
  if(!navigator.geolocation){
    setFb('fb3','Location is not supported by your browser.','fail');
    document.getElementById('btn3-next').style.display='block';
    return;
  }
  const btn=document.getElementById('btn-loc');
  btn.disabled=true; btn.textContent='Getting location…';
  setFb('fb3','⏳ Requesting GPS coordinates…','');

  navigator.geolocation.getCurrentPosition(async pos=>{
    const {latitude:lat,longitude:lon,accuracy}=pos.coords;
    try{
      const resp=await fetch(`/kyc/sessions/${SESSION_ID}/location`,{
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({lat,lon,accuracy}),
      });
      const data=await resp.json();
      if(resp.ok){
        // Show reverse-geocoded address to the user
        const box=document.getElementById('loc-box');
        box.style.display='block';
        document.getElementById('loc-addr').textContent=data.address||`${lat.toFixed(4)}°N, ${lon.toFixed(4)}°E`;
        // Show comparison result if address text was available
        const matchEl=document.getElementById('loc-match');
        if(data.comparison){
          const r=data.comparison.result;
          matchEl.textContent = r==='match'   ? '✓ Address matches your proof document' :
                                r==='partial' ? '~ Partial address match' :
                                r==='mismatch'? '✗ Address does not match proof document' : '';
          matchEl.style.color = r==='match'?'#4ade80':r==='partial'?'#facc15':'#f87171';
          if(data.comparison.distance_m!=null){
            matchEl.textContent += ` (${Math.round(data.comparison.distance_m)} m)`;
          }
        }
        setFb('fb3','✓ Location captured','pass');
      } else {
        setFb('fb3','⚠ Location saved but address lookup failed.','');
      }
    } catch{
      setFb('fb3','⚠ Could not reach server. Proceeding without location.','');
    }
    document.getElementById('btn3-next').style.display='block';
  },
  err=>{
    // Geolocation error codes: 1=denied, 2=unavailable, 3=timeout
    const msgs={1:'Location access denied.',2:'Location unavailable.',3:'Location request timed out.'};
    setFb('fb3',(msgs[err.code]||'Location error.')+' You may skip this step.','');
    const btn=document.getElementById('btn-loc');
    btn.disabled=false; btn.textContent='📍  Try Again';
    document.getElementById('btn3-next').style.display='block';
  },
  {timeout:15000,maximumAge:0});
};

document.getElementById('btn3-next').onclick=()=>{ startLiveness(); };
document.getElementById('btn3-skip').onclick=()=>{ startLiveness(); };

// ── Step 4: Liveness — open WebSocket, start camera, run challenges ───────
async function startLiveness(){
  show('s-4'); setStep(4);

  // Build dot indicators for each challenge
  const dotsEl=document.getElementById('dots');
  dotsEl.innerHTML='';
  for(let i=0;i<TOTAL_CHALLENGES;i++){
    const d=document.createElement('div');
    d.className='dot'; d.id='dot'+i; dotsEl.appendChild(d);
  }

  // Start the camera stream
  try{
    stream=await navigator.mediaDevices.getUserMedia(
      {video:{facingMode:'user',width:{ideal:1280},height:{ideal:720}},audio:false});
    const vid=document.getElementById('vid');
    vid.srcObject=stream; await vid.play();
  } catch{
    document.getElementById('ch-label').textContent='Camera Access Denied';
    document.getElementById('ch-inst').textContent='Please allow camera access and reload the page.';
    return;
  }

  // Open WebSocket — the server drives the challenge sequence from here
  const proto=location.protocol==='https:'?'wss:':'ws:';
  ws=new WebSocket(`${proto}//${location.host}/kyc/ws/${SESSION_ID}`);
  ws.onopen=()=>{ startCapture(); };
  ws.onmessage=e=>{ try{ handle(JSON.parse(e.data)); } catch{} };
  ws.onerror=()=>{ if(!resultShown) showResult(false,0,'Connection error.'); };
  ws.onclose=e=>{ if(!resultShown && e.code!==1000) showResult(false,0,'Session disconnected.'); stopCapture(); };
}

// Capture a JPEG frame every ~310 ms and send it to the server as base64
function startCapture(){
  const canvas=document.createElement('canvas');
  const ctx=canvas.getContext('2d');
  const vid=document.getElementById('vid');
  captureTimer=setInterval(()=>{
    if(!ws||ws.readyState!==1||!vid.videoWidth) return;
    // Resize to 640 px wide to keep payload manageable
    canvas.width=640;
    canvas.height=Math.round(640*vid.videoHeight/vid.videoWidth);
    ctx.drawImage(vid,0,0,canvas.width,canvas.height);
    canvas.toBlob(blob=>{
      if(!blob) return;
      const fr=new FileReader();
      fr.onloadend=()=>{
        const b64=fr.result.split(',')[1];
        if(ws&&ws.readyState===1) ws.send(JSON.stringify({type:'frame',data:b64}));
      };
      fr.readAsDataURL(blob);
    },'image/jpeg',0.82);
  },310);
}

function stopCapture(){
  if(captureTimer){clearInterval(captureTimer);captureTimer=null;}
  if(stream){stream.getTracks().forEach(t=>t.stop());stream=null;}
}

// Route incoming WebSocket messages from the server
function handle(msg){
  if(msg.type==='status')       updateLivenessUI(msg);
  else if(msg.type==='result'){ stopCapture(); if(ws) ws.close(1000); showResult(msg.passed,msg.display_score||0,msg.message||'',msg); }
  else if(msg.type==='error'){  stopCapture(); showResult(false,0,msg.message||'Verification failed.'); }
  // Ignore nudge/unknown messages — server sends nudges when waiting for the next frame
}

// Update the liveness challenge UI with the latest server-side status
function updateLivenessUI(msg){
  const badge=document.getElementById('face-badge');
  if(msg.face_detected){badge.className='badge b-ok';badge.textContent='✓ Face detected';}
  else{badge.className='badge b-warn';badge.textContent='Centre your face';}

  if(msg.challenge){
    document.getElementById('ch-label').textContent=LABELS[msg.challenge]||msg.challenge.toUpperCase();
    document.getElementById('ch-inst').textContent=INSTR[msg.challenge]||'';
  }

  // Update progress bar and dots
  const done=msg.challenges_completed||0;
  const total=msg.total_challenges||TOTAL_CHALLENGES;
  document.getElementById('prog-fill').style.width=total>0?(done/total*100)+'%':'0%';
  for(let i=0;i<total;i++){
    const d=document.getElementById('dot'+i);
    if(d) d.className='dot'+(i<done?' done':i===done?' active':'');
  }

  if(msg.feedback){
    const fb4=document.getElementById('fb4');
    fb4.textContent=msg.feedback;
    fb4.className='fb'+(msg.challenge_just_passed?' pass':'');
  }
}

// Build and show the final result screen
function showResult(passed,score,message,fullResult){
  resultShown=true;
  show('s-result');
  const inner=document.getElementById('res-inner');
  const icon=document.getElementById('res-icon');
  const title=document.getElementById('res-title');
  const det=document.getElementById('res-det');

  if(passed){
    inner.className='res-card res-pass';
    icon.textContent='✅';
    title.textContent='Identity Verified';
    det.innerHTML='Your identity has been successfully verified.<br>'
      +'<span style="color:#4ade80">Face match score: '+(score).toFixed(1)+'%</span><br><br>'
      +'You may now close this window.';
  } else {
    inner.className='res-card res-fail';
    icon.textContent='❌';
    title.textContent='Verification Failed';
    det.innerHTML=(message||'Verification could not be completed.')
      +'<br><br>Please contact your agent for assistance.';
  }

  // Show address summary panel if the server returned address check fields
  if(fullResult&&(fullResult.address_match_result||fullResult.current_address_text)){
    const sumEl=document.getElementById('addr-summary');
    sumEl.style.display='block';
    const r=fullResult.address_match_result;
    const addr=fullResult.current_address_text||'';
    const dist=fullResult.address_distance_meters;
    let html='<strong>Address check:</strong> ';
    html += r==='match'   ? '<span style="color:#4ade80">✓ Match</span>' :
            r==='partial' ? '<span style="color:#facc15">~ Partial match</span>' :
            r==='mismatch'? '<span style="color:#f87171">✗ Mismatch</span>' :
                            '<span style="color:#64748b">Skipped</span>';
    if(dist!=null) html+=` <span style="color:#64748b">(${Math.round(dist)} m)</span>`;
    if(addr) html+=`<br><span style="color:#64748b">Live address: ${addr.substring(0,120)}</span>`;
    sumEl.innerHTML=html;
  }
}

window.addEventListener('beforeunload',()=>{ stopCapture(); if(ws) ws.close(); });
</script>
</body>
</html>
"""


# ── API long description (Markdown — rendered in the Swagger info block) ─────────
_API_DESCRIPTION = """
**BaseTruth** is an AI-powered document fraud detection and identity verification platform.
Use these APIs to integrate forensic document analysis, structured field extraction,
entity risk scoring, and challenge-based Video KYC into any workflow.

## Quick Start

```bash
# 1. Health check
curl http://localhost:8000/api/v1/health

# 2. Forensic tamper scan — get a 0-100 forgery score for any document
curl -X POST http://localhost:8000/api/v1/forensic-scan \\
     -F "file=@passport.jpg"

# 3. Field extraction — classify document and extract structured data
curl -X POST http://localhost:8000/api/v1/document-extract \\
     -F "file=@payslip.pdf"
```

## Forgery Score Bands

| Score  | Verdict             | Recommended Action             |
|--------|---------------------|--------------------------------|
| 0 – 24  | **GENUINE**        | No tampering detected          |
| 25 – 49 | **UNCERTAIN**      | Manual review recommended      |
| 50 – 74 | **LIKELY TAMPERED**| Strong forensic signals found  |
| 75 – 100| **TAMPERED**       | High-confidence forgery        |

## Forensic Layers

Each image scan runs **11 forensic layers** in parallel:
`ELA` · `DCT` · `Clone Detection` · `Noise` · `Metadata` · `Entropy` ·
`Color Anomaly` · `Edge Density` · `Saturation` · `Font Consistency` · `AI Artifact Detection`

PDF scans run **11 PDF-specific layers**:
`Incremental Updates` · `Metadata` · `Font Consistency` · `Hidden Text` · `Suspicious Objects` ·
`Content Consistency` · `Digital Signature` · `Page ELA` · `Embedded Image Noise` · `File Entropy` · `XRef Integrity`

## Error Format

All error responses follow [RFC 7807](https://www.rfc-editor.org/rfc/rfc7807):
```json
{ "detail": "Human-readable description of the error" }
```
"""

# ── OpenAPI tag groups — each maps to a sidebar section in the Swagger UI ──────
_TAGS_METADATA: list = [
    {
        "name": "System",
        "description": (
            "Health-check and infrastructure endpoints. "
            "Use `GET /api/v1/health` to confirm the service is reachable before submitting documents."
        ),
    },
    {
        "name": "Scan",
        "description": (
            "Core document analysis endpoints.\n\n"
            "- **Forensic Scan** — run tamper-detection (ELA, DCT, clone detection, AI artefacts) and get a 0–100 forgery score\n"
            "- **Document Extract** — classify a document and extract structured fields (name, DOB, ID number, amounts)\n"
            "- **Scan by Path** — scan a file already present on the server filesystem\n"
            "- **Scan by Upload** — upload and scan in a single multipart/form-data request"
        ),
    },
    {
        "name": "Reports",
        "description": (
            "Browse and download scan audit reports.\n\n"
            "- List all reports, optionally filtered by kind and risk level\n"
            "- Download individual PDF audit reports by scan ID\n"
            "- Browse the 50 most-recent scans across all entities"
        ),
    },
    {
        "name": "Entities",
        "description": (
            "Entity registry — search and retrieve person or company profiles.\n\n"
            "Each entity accumulates a risk profile as documents are scanned against it. "
            "Use these endpoints in KYC and due-diligence workflows."
        ),
    },
    {
        "name": "Video KYC",
        "description": (
            "Challenge-based liveness + face-match for remote identity verification.\n\n"
            "**Typical flow:**\n"
            "1. `POST /kyc/sessions` — create a session and receive a `session_url`\n"
            "2. Send `session_url` to the end-user via email or SMS\n"
            "3. User opens the URL and completes liveness challenges in their browser\n"
            "4. Poll `GET /kyc/sessions/{session_id}` until `status` is `completed` or `failed`\n\n"
            "Available challenges: `blink` · `turn_left` · `turn_right` · `nod`"
        ),
    },
]

# ── Custom Swagger UI — enterprise dark-themed HTML served at /api/docs ─────────
_SWAGGER_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>BaseTruth \u00b7 API Reference</title>
  <link rel="icon" type="image/svg+xml"
    href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E%3Ctext y='.9em' font-size='90'%3E&#128737;%3C/text%3E%3C/svg%3E" />
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.18.2/swagger-ui.css" />
  <style>
  *,*::before,*::after{box-sizing:border-box}
  html,body{margin:0;padding:0;background:#080d1a;min-height:100vh;-webkit-font-smoothing:antialiased}
  /* ── Header & Navigation ─────────────────────────────────────────────────── */
  .bt-header{
    position:sticky;top:0;z-index:1000;
    background:rgba(8,13,26,.94);
    backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);
    border-bottom:1px solid rgba(99,102,241,.16);
    padding:0 2rem;height:62px;
    display:flex;align-items:center;gap:1rem;
    box-shadow:0 1px 0 rgba(99,102,241,.1),0 6px 28px rgba(0,0,0,.4);
  }
  .bt-brand{display:flex;align-items:center;gap:.6rem;text-decoration:none;user-select:none}
  .bt-icon{
    width:28px;height:28px;
    background:linear-gradient(135deg,#4f46e5 0%,#7c3aed 100%);
    border-radius:7px;display:flex;align-items:center;justify-content:center;
    font-size:.88rem;line-height:1;flex-shrink:0;
    box-shadow:0 0 0 1px rgba(99,102,241,.35),0 3px 10px rgba(79,70,229,.3);
  }
  .bt-brand-name{font-size:.96rem;font-weight:700;letter-spacing:-.025em;color:#f1f5f9}
  .bt-badge{
    font-size:.58rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;
    color:#818cf8;background:rgba(99,102,241,.1);border:1px solid rgba(99,102,241,.22);
    padding:.14rem .42rem;border-radius:99px;margin-left:.2rem;
  }
  .bt-spacer{flex:1 1 auto}
  .bt-nav{display:flex;align-items:center;gap:.2rem}
  .bt-nav-link{
    display:inline-flex;align-items:center;gap:.38rem;
    padding:.35rem .72rem;border-radius:6px;
    font-size:.76rem;font-weight:500;
    color:#8896aa;
    text-decoration:none;border:1px solid transparent;
    transition:color .15s,background .15s,border-color .15s;white-space:nowrap;
  }
  .bt-nav-link:hover{color:#dde5f0;background:rgba(255,255,255,.05);border-color:rgba(255,255,255,.08)}
  .bt-nav-link.bt-active{color:#a5b4fc;background:rgba(99,102,241,.1);border-color:rgba(99,102,241,.25)}
  .bt-mono{font-family:'SF Mono','Fira Code',ui-monospace,monospace;font-size:.73rem;opacity:.85;letter-spacing:0}
  .bt-nav-sep{width:1px;height:18px;background:rgba(255,255,255,.09);margin:0 .35rem;flex-shrink:0}
  .bt-dot{
    display:inline-block;width:6px;height:6px;border-radius:50%;
    background:#22c55e;box-shadow:0 0 0 2px rgba(34,197,94,.2);
    animation:hb 2.4s ease-in-out infinite;flex-shrink:0;
  }
  @keyframes hb{0%,100%{box-shadow:0 0 0 2px rgba(34,197,94,.2)}50%{box-shadow:0 0 0 5px rgba(34,197,94,0)}}
  /* ── Hide default swagger topbar ───────────────────────────────────────── */
  .swagger-ui .topbar{display:none!important}
  /* ── Base surfaces ──────────────────────────────────────────────────────── */
  .swagger-ui,.swagger-ui .wrapper{background:#080d1a!important}
  .swagger-ui{color:#dde5f0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif}
  /* ── Info section ───────────────────────────────────────────────────────── */
  .swagger-ui .info{padding:2.5rem 0 2rem}
  .swagger-ui .info .title{color:#f0f6ff;font-size:1.85rem;font-weight:800;line-height:1.15;letter-spacing:-.03em}
  .swagger-ui .info p,.swagger-ui .info li{color:#8a9ab3;line-height:1.72;font-size:.88rem}
  .swagger-ui .info a{color:#818cf8}
  .swagger-ui .info code{background:#111b2e;color:#c4b5fd;padding:.1rem .35rem;border-radius:4px;font-size:.84em;border:1px solid rgba(99,102,241,.15)}
  .swagger-ui .info .base-url{background:#111b2e;border:1px solid #1d2d44;color:#4d6480;padding:.3rem .75rem;border-radius:6px;font-size:.78rem}
  .swagger-ui .info .version{background:rgba(99,102,241,.1);border:1px solid rgba(99,102,241,.25);color:#a5b4fc;padding:.18rem .55rem;border-radius:99px;font-size:.62rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em}
  /* ── Scheme container ───────────────────────────────────────────────────── */
  .swagger-ui .scheme-container{background:#0d1526;box-shadow:none;border-bottom:1px solid #1a2840;padding:1rem 2rem}
  .swagger-ui .schemes>label,.swagger-ui .servers>label{color:#94a3b8;font-size:.8rem;font-weight:600;letter-spacing:.04em}
  /* ── Tag headings ───────────────────────────────────────────────────────── */
  .swagger-ui .opblock-tag{border-color:#131f35!important;color:#e8eef8!important;font-size:.9rem;font-weight:700;padding:.75rem 0;letter-spacing:-.01em}
  .swagger-ui .opblock-tag:hover{background:transparent!important}
  .swagger-ui .opblock-tag small{color:#4d6480!important;font-size:.78rem;font-weight:400;margin-left:.5rem}
  .swagger-ui .opblock-tag a{color:#e8eef8!important}
  /* ── Operation blocks — shared ──────────────────────────────────────────── */
  .swagger-ui .opblock{background:#0f1829;border-radius:9px;margin-bottom:.4rem;box-shadow:0 1px 3px rgba(0,0,0,.35);border-width:1px!important;overflow:hidden;transition:box-shadow .18s}
  .swagger-ui .opblock:hover{box-shadow:0 2px 12px rgba(0,0,0,.5)}
  .swagger-ui .opblock .opblock-summary{border-radius:9px;padding:.5rem .8rem}
  .swagger-ui .opblock.is-open .opblock-summary{border-bottom-color:#1a2840!important;border-radius:9px 9px 0 0}
  .swagger-ui .opblock .opblock-summary-description{color:#7b8fa8!important;font-size:.8rem}
  .swagger-ui .opblock .opblock-summary-path{color:#dde5f0!important;font-size:.86rem;font-weight:500}
  .swagger-ui .opblock-body{background:#090f1d;border-radius:0 0 9px 9px;padding:1.1rem}
  .swagger-ui .opblock-description-wrapper p,.swagger-ui .opblock-title_normal p{color:#94a3b8}
  /* ── Method badges ──────────────────────────────────────────────────────── */
  .swagger-ui .opblock-summary-method{border-radius:5px!important;font-weight:700!important;font-size:.7rem!important;min-width:60px!important;text-align:center!important;letter-spacing:.03em!important}
  .swagger-ui .opblock-get{border-color:#0ea5e9!important}
  .swagger-ui .opblock-get .opblock-summary{background:rgba(14,165,233,.05)!important}
  .swagger-ui .opblock-get .opblock-summary-method{background:#0ea5e9!important;color:#fff!important}
  .swagger-ui .opblock-post{border-color:#6366f1!important}
  .swagger-ui .opblock-post .opblock-summary{background:rgba(99,102,241,.05)!important}
  .swagger-ui .opblock-post .opblock-summary-method{background:#6366f1!important;color:#fff!important}
  .swagger-ui .opblock-put{border-color:#f59e0b!important}
  .swagger-ui .opblock-put .opblock-summary{background:rgba(245,158,11,.05)!important}
  .swagger-ui .opblock-put .opblock-summary-method{background:#f59e0b!important;color:#fff!important}
  .swagger-ui .opblock-delete{border-color:#ef4444!important}
  .swagger-ui .opblock-delete .opblock-summary{background:rgba(239,68,68,.05)!important}
  .swagger-ui .opblock-delete .opblock-summary-method{background:#ef4444!important;color:#fff!important}
  .swagger-ui .opblock-patch{border-color:#14b8a6!important}
  .swagger-ui .opblock-patch .opblock-summary{background:rgba(20,184,166,.05)!important}
  .swagger-ui .opblock-patch .opblock-summary-method{background:#14b8a6!important;color:#fff!important}
  /* ── Parameters table ───────────────────────────────────────────────────── */
  .swagger-ui .parameters-container,.swagger-ui .table-container{background:transparent}
  .swagger-ui table thead tr{background:#080d1a!important}
  .swagger-ui table thead tr td,.swagger-ui .col_header td{color:#64748b!important;font-size:.7rem!important;text-transform:uppercase!important;letter-spacing:.06em!important;border-color:#334155!important}
  .swagger-ui table.parameters tbody tr td,.swagger-ui table.headers tbody tr td{color:#cbd5e1;border-bottom-color:#1e293b!important;padding:.55rem .5rem}
  .swagger-ui .parameter__name{color:#e2e8f0;font-weight:600;font-size:.86rem}
  .swagger-ui .parameter__name.required::after{color:#f87171}
  .swagger-ui .parameter__type{color:#818cf8;font-size:.78rem}
  .swagger-ui .parameter__in{color:#64748b;font-size:.7rem;font-style:italic;margin-left:.3rem}
  .swagger-ui .markdown p,.swagger-ui .renderedMarkdown p{color:#94a3b8;line-height:1.65}
  .swagger-ui .markdown code,.swagger-ui .renderedMarkdown code{background:#0f172a;color:#c4b5fd;padding:.1rem .3rem;border-radius:3px;font-size:.85em}
  /* ── Code blocks ────────────────────────────────────────────────────────── */
  .swagger-ui .highlight-code,.swagger-ui .highlight-code pre{background:#050a14!important;border-radius:7px;border:1px solid #1a2840}
  .swagger-ui .microlight,.swagger-ui pre{background:#050a14!important;color:#dde5f0!important;font-size:.78rem;border-radius:5px;padding:.6rem .8rem;line-height:1.6}
  /* ── Response section ───────────────────────────────────────────────────── */
  .swagger-ui .responses-inner h4,.swagger-ui .responses-inner h5{color:#94a3b8;font-size:.76rem;letter-spacing:.06em;text-transform:uppercase}
  .swagger-ui .response-col_status{color:#4ade80!important;font-weight:700;font-size:.86rem}
  .swagger-ui .response-col_links{color:#64748b}
  .swagger-ui .tab-header .tab-item h4 span{color:#94a3b8}
  .swagger-ui .tab-header .tab-item.active h4 span{color:#e2e8f0}
  .swagger-ui .opblock-body .tab-header .tab-item.active h4 span:after{background:#6366f1}
  /* ── Request body textarea ──────────────────────────────────────────────── */
  .swagger-ui .body-param__text{background:#050a14!important;color:#dde5f0!important;border:1px solid #1a2840!important;border-radius:6px;font-size:.79rem;line-height:1.6}
  /* ── Models / schemas ───────────────────────────────────────────────────── */
  .swagger-ui section.models{background:#0f1829;border:1px solid #1a2840;border-radius:9px;margin-top:1.5rem}
  .swagger-ui section.models h4{color:#dde5f0!important;font-weight:700;font-size:.88rem}
  .swagger-ui section.models .model-container{background:#080d1a;border:1px solid #131f35;border-radius:7px;margin:.4rem}
  .swagger-ui .model-box{background:#080d1a!important;padding:.6rem .8rem;border-radius:5px}
  .swagger-ui .model{color:#cbd5e1}
  .swagger-ui .model-title{color:#f1f5f9;font-weight:700}
  .swagger-ui .prop-type{color:#818cf8;font-size:.8rem}
  .swagger-ui .prop-format{color:#64748b;font-size:.76rem}
  .swagger-ui .prop-enum{color:#f472b6}
  .swagger-ui .model .property.primitive{color:#818cf8}
  .swagger-ui .model .property.string{color:#34d399}
  .swagger-ui .model .property.integer,.swagger-ui .model .property.number{color:#fb923c}
  .swagger-ui .model .property.boolean{color:#f472b6}
  /* ── Buttons ────────────────────────────────────────────────────────────── */
  .swagger-ui .btn.try-out__btn{background:transparent!important;color:#6366f1!important;border:1px solid #6366f1!important;border-radius:6px;font-weight:600;font-size:.76rem;padding:.28rem .7rem;transition:all .2s}
  .swagger-ui .btn.try-out__btn:hover{background:rgba(99,102,241,.1)!important}
  .swagger-ui .btn.execute{background:linear-gradient(135deg,#4f46e5,#6366f1)!important;color:#fff!important;border:none!important;border-radius:6px;font-weight:700;padding:.42rem 1.25rem;box-shadow:0 2px 10px rgba(79,70,229,.4);transition:all .2s}
  .swagger-ui .btn.execute:hover{box-shadow:0 4px 18px rgba(79,70,229,.55);transform:translateY(-1px)}
  .swagger-ui .btn.cancel{background:transparent!important;color:#94a3b8!important;border-color:#334155!important;border-radius:6px}
  .swagger-ui .btn.authorize{background:rgba(99,102,241,.1)!important;color:#a5b4fc!important;border-color:#6366f1!important;border-radius:6px;font-weight:600}
  .swagger-ui .btn.authorize svg{fill:#6366f1!important}
  .swagger-ui .btn.authorize.locked{background:rgba(74,222,128,.1)!important;color:#4ade80!important;border-color:#4ade80!important}
  /* ── Input fields ───────────────────────────────────────────────────────── */
  .swagger-ui input[type=text],.swagger-ui input[type=email],.swagger-ui input[type=password],.swagger-ui input[type=search],.swagger-ui textarea,.swagger-ui select{background:#080d1a!important;border:1px solid #1a2840!important;color:#dde5f0!important;border-radius:6px;transition:border-color .18s;font-family:inherit!important}
  .swagger-ui input:focus,.swagger-ui textarea:focus{border-color:#6366f1!important;outline:none;box-shadow:0 0 0 3px rgba(99,102,241,.12)!important}
  /* ── Authorization dialog ───────────────────────────────────────────────── */
  .swagger-ui .dialog-ux .modal-ux{background:#0f1829;border:1px solid #1a2840;border-radius:14px;box-shadow:0 30px 80px rgba(0,0,0,.7)}
  .swagger-ui .dialog-ux .modal-ux-header{background:#080d1a;border-bottom:1px solid #1a2840;border-radius:14px 14px 0 0;padding:1.1rem 1.5rem}
  .swagger-ui .dialog-ux .modal-ux-header h3{color:#f0f6ff;font-weight:700}
  .swagger-ui .dialog-ux .modal-ux-content{padding:1.5rem}
  .swagger-ui .auth-container h4,.swagger-ui .auth-container h5{color:#94a3b8;font-size:.76rem;letter-spacing:.06em;text-transform:uppercase}
  .swagger-ui .auth-container p,.swagger-ui .auth-container li{color:#64748b;font-size:.82rem}
  .swagger-ui .auth-container .wrapper{background:transparent;border:none;box-shadow:none;padding:0}
  /* ── Filter bar ─────────────────────────────────────────────────────────── */
  .swagger-ui .filter-container{background:#0d1526;padding:.8rem 2rem;border-bottom:1px solid #1a2840}
  .swagger-ui .filter .operation-filter-input{background:#080d1a!important;border-color:#1a2840!important;color:#dde5f0!important;border-radius:7px;padding:.45rem 1rem;width:100%;max-width:420px;font-size:.84rem}
  .swagger-ui .filter .operation-filter-input::placeholder{color:#4d6480}
  /* ── Misc ───────────────────────────────────────────────────────────────── */
  .swagger-ui .expand-methods svg,.swagger-ui .expand-operation svg,.swagger-ui .arrow{fill:#6366f1!important}
  .swagger-ui .loading-container{background:#080d1a}
  .swagger-ui .loading-container .loading::after{border-color:#6366f1 transparent #6366f1 transparent}
  .swagger-ui .errors-wrapper{background:rgba(220,38,38,.07);border:1px solid rgba(220,38,38,.22);border-radius:8px}
  ::-webkit-scrollbar{width:6px;height:6px}
  ::-webkit-scrollbar-track{background:#080d1a}
  ::-webkit-scrollbar-thumb{background:#1e3050;border-radius:99px}
  ::-webkit-scrollbar-thumb:hover{background:#2c4370}
  </style>
</head>
<body>
  <header class="bt-header">
    <a class="bt-brand" href="/api/docs" aria-label="BaseTruth API home">
      <span class="bt-icon">&#128737;&#65039;</span>
      <span class="bt-brand-name">BaseTruth</span>
      <span class="bt-badge">v1.0</span>
    </a>
    <span class="bt-spacer"></span>
    <nav class="bt-nav" aria-label="Documentation links">
      <a class="bt-nav-link bt-active" href="/api/docs" aria-current="page">&#128218; Interactive Docs</a>
      <a class="bt-nav-link" href="/api/redoc" target="_blank" rel="noopener">&#128196; ReDoc</a>
      <a class="bt-nav-link" href="/api/openapi.json" target="_blank" rel="noopener"><span class="bt-mono">{ }</span> OpenAPI JSON</a>
      <span class="bt-nav-sep"></span>
      <a class="bt-nav-link" href="/api/v1/health" target="_blank" rel="noopener"><span class="bt-dot"></span> Health</a>
    </nav>
  </header>

  <div id="swagger-ui"></div>

  <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.18.2/swagger-ui-bundle.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.18.2/swagger-ui-standalone-preset.js"></script>
  <script>
    window.onload = function() {
      SwaggerUIBundle({
        url: '/api/openapi.json',
        dom_id: '#swagger-ui',
        presets: [
          SwaggerUIBundle.presets.apis,
          SwaggerUIBundle.SwaggerUIStandalonePreset,
        ],
        plugins: [SwaggerUIBundle.plugins.DownloadUrl],
        layout: 'BaseLayout',
        deepLinking: true,
        displayRequestDuration: true,
        defaultModelsExpandDepth: 1,
        defaultModelExpandDepth: 2,
        docExpansion: 'list',
        filter: true,
        persistAuthorization: true,
        tryItOutEnabled: true,
        requestSnippetsEnabled: true,
        requestSnippets: {
          generators: {
            curl_bash:       { title: 'cURL',       syntax: 'bash'       },
            node_fetch:      { title: 'Node.js',    syntax: 'javascript' },
            python_requests: { title: 'Python',     syntax: 'python'     },
          },
          defaultExpanded: false,
        },
        tagsSorter: 'alpha',
        operationsSorter: 'alpha',
      });
    };
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
        title="BaseTruth",
        summary="Document Fraud Detection & Identity Verification API",
        description=_API_DESCRIPTION,
        version="1.0.0",
        contact={
            "name": "BaseTruth Support",
            "url": "https://basetruth.ai",
            "email": "support@basetruth.ai",
        },
        license_info={
            "name": "Proprietary — All rights reserved",
            "url": "https://basetruth.ai/legal",
        },
        openapi_tags=_TAGS_METADATA,
        docs_url=None,          # replaced by our custom branded endpoint below
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

    # ── Custom Swagger UI — dark-themed, BaseTruth branded ────────────────────
    from fastapi.responses import HTMLResponse as _HTMLResp

    @app.get("/api/docs", include_in_schema=False)
    def swagger_ui() -> Any:
        """Serve the custom BaseTruth Swagger UI."""
        return _HTMLResp(_SWAGGER_HTML)

    svc = _service(artifact_root)

    # -------------------------------------------------------------------------
    # Endpoints
    # -------------------------------------------------------------------------

    @app.get(
        "/api/v1/health",
        tags=["System"],
        summary="Service Health Check",
        response_model=HealthResponse,
        responses={200: {"description": "Service is healthy"}},
    )
    def health() -> Dict[str, Any]:
        """Ping the BaseTruth API to confirm it is reachable and return version information.

        Use this endpoint before submitting documents to verify the service is online.
        """
        return {
            "status": "ok",
            "product": "BaseTruth",
            "version": "1.0.0",
            "artifact_root": str(svc.artifact_root),
        }

    @app.post(
        "/api/v1/scan",
        tags=["Scan"],
        summary="Scan Document by Server Path",
        responses={
            404: {"description": "File not found at the specified path"},
            500: {"description": "Scan engine error"},
        },
    )
    def scan_by_path(request: ScanPathRequest) -> Dict[str, Any]:
        """Scan a document that already exists on the server filesystem.

        Provide the absolute server-side path to the file. The full forensic
        and field-extraction pipeline runs and returns the scan result as JSON.
        """
        path = Path(request.path)
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {request.path}")
        try:
            return svc.scan_document(path)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post(
        "/api/v1/scan/upload",
        tags=["Scan"],
        summary="Scan Uploaded Document",
        responses={
            400: {"description": "Unsupported file type"},
            500: {"description": "Scan engine error"},
        },
    )
    async def scan_upload(file: UploadFile = File(..., description="Document file to scan (PDF or image).")) -> Dict[str, Any]:
        """Upload a document and scan it in a single request.

        Accepts any supported format: PDF, JPG, PNG, BMP, TIFF, GIF, WEBP.
        The full forensic pipeline runs on the uploaded bytes and the result
        is returned immediately. The file is not persisted after the scan.
        """
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

    @app.post(
        "/api/v1/document-extract",
        tags=["Scan"],
        summary="Extract Document Fields",
        response_model=DocumentExtractResponse,
        responses={
            500: {"description": "Classification or extraction engine error"},
        },
    )
    async def extract_document_endpoint(file: UploadFile = File(..., description="Document to classify and extract fields from (PDF or image).")) -> Dict[str, Any]:
        """Classify a document and extract its structured fields.

        This endpoint runs the same pipeline as the **Scan Document** screen:
        - **Step 1** — classify document type using Gemma AI (Aadhaar, Passport, Payslip, etc.)
        - **Step 2** — extract structured fields with PaddleOCR (scanned) or PyMuPDF (digital PDF)

        Returns the document type, classification confidence, and a key-value
        map of all extractable fields (name, DOB, ID number, amounts, employer, etc.).
        No data is persisted to the database.
        """
        from basetruth.ui.pages.scan import extract_document  # noqa: PLC0415

        file_bytes = await file.read()
        filename = file.filename or "document"
        result = extract_document(file_bytes, filename)
        if result.get("error"):
            raise HTTPException(status_code=500, detail=result["error"])
        # Strip private meta-keys (prefixed with _) before returning to the caller
        display_fields = {
            k: v for k, v in result.get("extracted_fields", {}).items()
            if not k.startswith("_")
        }
        return {
            "filename": result["filename"],
            "document_type": result["document_type"],
            "is_image_based": result["is_image_based"],
            "confidence": result["confidence"],
            "extracted_fields": display_fields,
        }

    @app.post(
        "/api/v1/forensic-scan",
        tags=["Scan"],
        summary="Forensic Tamper Analysis",
        response_model=ForensicScanResponse,
        responses={
            400: {"description": "Unsupported file type"},
            500: {"description": "Forensic engine error"},
        },
    )
    async def forensic_scan_endpoint(file: UploadFile = File(..., description="Document to analyse for tampering (PDF or image).")) -> Dict[str, Any]:
        """Upload a document and run full forensic tamper-detection analysis.

        Runs the same 11-layer forensic pipeline as the **Forensic Scan** UI screen:

        **Image layers:** ELA · Metadata · Entropy · Noise · DCT · Clone Detection ·
        Color Anomaly · Edge Density · Saturation · Font Consistency · AI Artifact Detection

        **PDF layers:** Incremental Updates · Metadata · Font Consistency · Hidden Text ·
        Suspicious Objects · Content Consistency · Digital Signature · Page ELA ·
        Embedded Image Noise · File Entropy · XRef Integrity

        Returns a `forgery_score_0_100` from 0 (genuine) to 100 (tampered),
        a plain-English `honest_review` for non-technical reviewers,
        and the raw per-layer `layers` metrics for deep investigation.
        No data is persisted.
        """
        from basetruth.ui.pages.forensics_utils import ForensicAnalyzer  # noqa: PLC0415

        file_bytes = await file.read()
        filename = file.filename or "document"
        result = ForensicAnalyzer.analyze_document(file_bytes, filename)
        if result.get("error"):
            raise HTTPException(status_code=500, detail=result["error"])

        # Expose Gemma4 visual detective output under a clean API-facing key.
        # Internal dict key is 'visual_clues'; published JSON key is 'visual_intelligence'.
        raw_vc = result.get("visual_clues") or {}
        # Strip the internal _unavailable sentinel — API consumers see null instead.
        visual_intelligence = None if raw_vc.get("_unavailable") else raw_vc

        return {
            "filename": result["filename"],
            "document_type": result["document_type"],
            "is_image_based": result["is_image_based"],
            "forensic_verdict": result["forensic_verdict"],
            "forgery_score_0_100": result["forgery_score_0_100"],
            "overall_explanation": result["overall_explanation"],
            # Plain-English LLM verdict written for non-technical reviewers.
            # Mirrors the 'Honest Review' card shown on the Forensic Scan UI page.
            "honest_review": result.get("honest_review", ""),
            "evidence": result["evidence"],
            "layers": result["layers"],
            # Gemma4 'Logical Detective' visual intelligence report.
            # Contains document_type, findings (area / clue / suspicion_level / reason),
            # and overall_assessment.  Null when Ollama is offline.
            "visual_intelligence": visual_intelligence,
        }

    @app.get(
        "/api/v1/reports",
        tags=["Reports"],
        summary="List Scan Reports",
        responses={
            503: {"description": "Database unavailable"},
        },
    )
    def list_reports(
        kind: Optional[str] = Query(None, description="Filter by report kind: `verification` or `comparison`."),
        risk_level: Optional[str] = Query(None, description="Filter by risk level: `high`, `medium`, or `low`."),
    ) -> List[Dict[str, Any]]:
        """Return all scan reports, optionally filtered by kind and risk level.

        When no filters are supplied the full report list is returned.
        Each item includes the entity reference, document type, verdict, score,
        and the ISO 8601 timestamp of the scan.
        """
        reports = svc.list_reports()
        if kind:
            reports = [r for r in reports if r.get("kind") == kind]
        if risk_level:
            reports = [r for r in reports if r.get("risk_level") == risk_level]
        return reports

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
    from basetruth.vision.face import get_face_analyzer, get_mediapipe_faces
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

        try:
            face_app = get_face_analyzer()
            with _kyc_face_lock:
                faces = face_app.get(img)
            # InsightFace kps are eye-centre points only — they cannot compute EAR.
            # Run MediaPipe in parallel to get Eye Aspect Ratio (needed for blink challenge).
            try:
                _mp_faces = get_mediapipe_faces(img)
                if _mp_faces:
                    _ear = _mp_faces[0].ear
                    for _f in faces:
                        _f.ear = _ear
            except Exception:
                pass  # EAR unavailable; blink will fall back to det_score path
        except ImportError:
            # InsightFace not available (Python 3.13+) — use MediaPipe as fallback.
            faces = get_mediapipe_faces(img)
        except Exception:
            # Any other init error (model download, ONNX issue) — fall back gracefully.
            faces = get_mediapipe_faces(img)

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
            try:
                return _finish_session(session, face)
            except Exception as _fin_exc:
                _kyc_log.error("KYC finish_session error", extra={"error": str(_fin_exc)})
                session.status = "failed"
                return {"type": "error", "message": "Face match failed — please retake the session."}

        try:
            features = extract_features(face)
        except Exception:
            # kps unavailable (face too small / angled) — skip this frame silently
            return {
                "type": "status",
                "face_detected": True,
                "challenge": session.current_challenge,
                "challenges_completed": session.current_challenge_idx,
                "total_challenges": len(session.challenges),
                "feedback": "Hold still — aligning to your face…",
                "challenge_just_passed": False,
            }
        history  = session.current_frame_history()
        history.append(features)

        current_ch = session.current_challenge
        analysis   = analyze_challenge(history, current_ch)
        just_passed = False
        if analysis["passed"]:
            session.advance_challenge()
            just_passed = True
            if session.all_done:
                try:
                    return _finish_session(session, face)
                except Exception as _fin_exc:
                    _kyc_log.error("KYC finish_session error", extra={"error": str(_fin_exc)})
                    session.status = "failed"
                    return {"type": "error", "message": "Face match failed — please retake the session."}

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
        """Called once all liveness challenges pass — runs the face-match check.

        Builds the final result dict that is sent back to the customer browser
        and stored in session.result.  Attaches address comparison fields so
        the customer result screen can display the address verification outcome.
        """
        if session.reference_embedding_b64:
            try:
                match = run_face_match(face, session.reference_embedding_b64)
            except Exception as exc:
                _kyc_log.error("run_face_match error", extra={"error": str(exc)})
                match = {
                    "passed": False,
                    "match_score": 0.0,
                    "display_score": 0.0,
                    "cosine_similarity": 0.0,
                    "threshold": 0.40,
                    "message": "Face match error — please retry.",
                }
            session.status = "completed" if match["passed"] else "failed"
            session.result = match
            result = {"type": "result", **match}
        else:
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
            result = {"type": "result", **result}

        # Attach address verification fields so the result screen can display them
        result["address_match_result"] = session.address_match_result or "skipped"
        result["address_distance_meters"] = session.address_distance_meters
        result["current_address_text"] = session.current_address_text

        return result

    @app.post(
        "/kyc/sessions",
        tags=["Video KYC"],
        summary="Create Video KYC Session",
        response_model=KYCSessionCreateResponse,
        responses={
            422: {"description": "Invalid challenge name supplied"},
        },
    )
    def create_kyc_session(req: CreateKYCSessionRequest) -> Dict[str, Any]:
        """Create a new challenge-based Video KYC session.

        Returns a `session_url` to send to the end-user. The user opens the URL
        in their browser and completes the liveness challenges. Poll
        `GET /kyc/sessions/{session_id}` to track completion.

        **Available challenges:** `blink` · `turn_left` · `turn_right` · `nod`

        If `challenges` is omitted, 2 challenges are selected at random.
        If `reference_embedding_b64` is provided, a face-match check against the
        ID document photo is performed after liveness passes.
        """
        challenges = req.challenges or _random.sample(ALL_CHALLENGES, k=2)
        session = _kyc_store.create(
            challenges=challenges,
            reference_embedding_b64=req.reference_embedding_b64,
            customer_name=req.customer_name,
            entity_ref=req.entity_ref,
            address_dtls=req.address_dtls,
            reference_doc_filename=req.reference_doc_filename or "",
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

    @app.get(
        "/kyc/sessions/{session_id}",
        tags=["Video KYC"],
        summary="Get KYC Session Status",
        responses={
            404: {"description": "Session not found or expired"},
        },
    )
    def get_kyc_session_status(session_id: str) -> Dict[str, Any]:
        """Poll the status of a Video KYC session.

        Returns `status` = `waiting` | `active` | `completed` | `failed`.
        When `status` is `completed` or `failed`, the `result` object contains
        `passed` (bool), `match_score` (0–1), and `display_score` (0–100).
        """
        session = _kyc_store.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found or expired.")
        return session.to_status_dict()

    @app.get("/kyc/{session_id}", response_class=_HTMLResponse, tags=["Video KYC"], include_in_schema=False)
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

    # ── KYC Step helpers — called from the new HTTP endpoints ────────────────

    def _extract_face_from_image_bytes(image_bytes: bytes) -> Optional[str]:
        """Detect the largest face in the uploaded ID document image and extract
        its embedding so it can be used as the face-match reference during liveness.

        Tries InsightFace first (buffalo_l); falls back to MediaPipe when InsightFace
        is unavailable (Python 3.13+) or raises unexpectedly.

        Returns the normed embedding as a base64-encoded float32 byte string,
        or None when no face is found or the image cannot be decoded.
        """
        import base64 as _b64  # noqa: PLC0415

        nparr = _np.frombuffer(image_bytes, _np.uint8)
        img = _cv2.imdecode(nparr, _cv2.IMREAD_COLOR)
        if img is None:
            _kyc_log.warning("_extract_face_from_image_bytes: cv2 could not decode image")
            return None

        # Try InsightFace first; fall back to MediaPipe on ImportError or error
        try:
            face_app = get_face_analyzer()
            with _kyc_face_lock:
                faces = face_app.get(img)
        except ImportError:
            faces = get_mediapipe_faces(img)
        except Exception as exc:
            _kyc_log.warning("InsightFace failed for ID image, using MediaPipe: %s", exc)
            faces = get_mediapipe_faces(img)

        if not faces:
            _kyc_log.debug("_extract_face_from_image_bytes: no face found in uploaded ID")
            return None

        # Pick the largest face by bounding-box area (most prominent face on the ID)
        def _area(f: Any) -> float:
            bb = f.bbox
            return float((bb[2] - bb[0]) * (bb[3] - bb[1]))

        face = max(faces, key=_area)
        emb = getattr(face, "normed_embedding", None)
        if emb is None:
            return None
        return _b64.b64encode(emb.astype("float32").tobytes()).decode()

    def _extract_address_text(image_bytes: bytes) -> str:
        """OCR an address-proof document image and return the raw text.

        Tries pytesseract (Tesseract wrapper) — returns an empty string if
        pytesseract or Tesseract is not installed.  Failure is non-critical:
        the address comparison will fall back to GPS-only when no text is
        available.
        """
        try:
            import io as _io  # noqa: PLC0415

            import pytesseract  # noqa: PLC0415
            from PIL import Image as _PILImg  # noqa: PLC0415

            img = _PILImg.open(_io.BytesIO(image_bytes))
            text = pytesseract.image_to_string(img, lang="eng")
            extracted = text.strip()
            _kyc_log.debug("_extract_address_text: extracted %d chars via OCR", len(extracted))
            return extracted
        except Exception as exc:
            _kyc_log.debug("_extract_address_text: OCR unavailable or failed (non-critical): %s", exc)
            return ""

    @app.post(
        "/kyc/sessions/{session_id}/upload-id",
        tags=["Video KYC"],
        summary="Upload Customer ID Document",
        responses={
            404: {"description": "Session not found or expired"},
            400: {"description": "No face found in the uploaded image"},
        },
    )
    async def kyc_upload_id(session_id: str, file: UploadFile) -> Dict[str, Any]:
        """Receive the customer's ID document photo (Step 1 of the KYC wizard).

        Extracts the face from the photo and stores the embedding as the
        face-match reference for the subsequent liveness check.  Overwrites
        any reference embedding the operator pre-seeded at session creation.

        Returns ``{"face_found": true}`` on success or raises HTTP 400 when
        no face can be detected.
        """
        session = _kyc_store.get(session_id)
        if not session or session.is_expired():
            raise HTTPException(status_code=404, detail="Session not found or expired.")

        image_bytes = await file.read()
        emb_b64 = await _asyncio.get_event_loop().run_in_executor(
            None, _extract_face_from_image_bytes, image_bytes
        )

        if emb_b64 is None:
            _kyc_log.warning(
                "kyc_upload_id: no face found in uploaded ID",
                extra={"session_id": session_id, "doc_filename": file.filename},
            )
            raise HTTPException(
                status_code=400,
                detail=(
                    "No face found in the uploaded image. "
                    "Please use a clear, well-lit photo of your Aadhaar or PAN card."
                ),
            )

        # Replace the reference embedding with the one extracted from the customer's ID
        session.reference_embedding_b64 = emb_b64
        # Store filename for audit trail
        session.reference_doc_filename = file.filename or "uploaded_id"
        _kyc_log.info(
            "kyc_upload_id: face embedding stored",
            extra={"session_id": session_id, "doc_filename": file.filename},
        )
        return {"face_found": True}

    @app.post(
        "/kyc/sessions/{session_id}/upload-address",
        tags=["Video KYC"],
        summary="Upload Address Proof Document",
        responses={
            404: {"description": "Session not found or expired"},
        },
    )
    async def kyc_upload_address(session_id: str, file: UploadFile) -> Dict[str, Any]:
        """Receive the customer's address proof photo (Step 2 of the KYC wizard).

        Runs OCR to extract the raw text from the document and stores it in the
        session for later address comparison once the customer shares their GPS
        location in Step 3.  OCR failure is non-critical — the address step is
        still marked as received so the wizard can continue.

        Returns ``{"address_extracted": bool, "hint": "..."}`` where
        ``address_extracted`` is True when OCR produced non-empty text.
        """
        session = _kyc_store.get(session_id)
        if not session or session.is_expired():
            raise HTTPException(status_code=404, detail="Session not found or expired.")

        image_bytes = await file.read()
        address_text = await _asyncio.get_event_loop().run_in_executor(
            None, _extract_address_text, image_bytes
        )

        # Store whatever we extracted; empty string is valid (OCR not available)
        session.address_dtls = {
            "raw_text": address_text,
            "filename": file.filename or "address_proof",
        }
        session.address_proof_filename = file.filename or "address_proof"
        _kyc_log.info(
            "kyc_upload_address: address proof stored",
            extra={
                "session_id": session_id,
                "doc_filename": file.filename,
                "text_len": len(address_text),
            },
        )
        return {
            "address_extracted": bool(address_text),
            "hint": "Address text extracted." if address_text else "OCR unavailable — address will be compared via GPS only.",
        }

    @app.post(
        "/kyc/sessions/{session_id}/location",
        tags=["Video KYC"],
        summary="Submit Customer GPS Location",
        responses={
            404: {"description": "Session not found or expired"},
        },
    )
    def kyc_save_location(session_id: str, data: LocationData) -> Dict[str, Any]:
        """Receive the customer's GPS coordinates (Step 3 of the KYC wizard).

        Reverse-geocodes the coordinates to get a human-readable address, then
        compares it against the address extracted from the proof document (if
        available) using Jaccard token overlap and PIN code matching.  If the
        proof address can also be forward-geocoded, the physical distance in
        metres is computed via the Haversine formula.

        All geocoding calls degrade gracefully — a network failure here never
        blocks the wizard from proceeding to the liveness step.

        Returns a JSON object with keys:
          - ``address``: human-readable string of the current location, or empty
          - ``comparison``: address comparison result dict, or null when skipped
        """
        from basetruth.kyc.address_match import (  # noqa: PLC0415
            calculate_distance,
            compare_addresses,
            geocode_address,
            reverse_geocode,
        )

        session = _kyc_store.get(session_id)
        if not session or session.is_expired():
            raise HTTPException(status_code=404, detail="Session not found or expired.")

        # Persist the raw GPS coordinates for audit trail
        session.current_location_json = {
            "lat": data.lat,
            "lon": data.lon,
            "accuracy": data.accuracy,
        }

        # Reverse-geocode to a readable address string (may return None on failure)
        live_addr: str = reverse_geocode(data.lat, data.lon) or ""
        session.current_address_text = live_addr

        comparison: Optional[Dict[str, Any]] = None
        proof_text: str = (session.address_dtls or {}).get("raw_text", "")

        if proof_text and live_addr:
            # Text-based comparison: Jaccard + PIN + state signals
            comparison = compare_addresses(proof_text, live_addr)
            session.address_match_result = comparison["result"]

            # Try to enrich with physical GPS distance via forward geocoding
            proof_gps = geocode_address(proof_text)
            if proof_gps:
                dist_m = calculate_distance(data.lat, data.lon, proof_gps[0], proof_gps[1])
                session.address_distance_meters = dist_m
                comparison["distance_m"] = dist_m
                # 500 m rule: if close enough, override text-only mismatch → match
                if dist_m <= 500:
                    session.address_match_result = "match"
                    comparison["result"] = "match"

        _kyc_log.info(
            "kyc_save_location: location stored",
            extra={
                "session_id": session_id,
                "lat": data.lat,
                "lon": data.lon,
                "address_match": session.address_match_result,
            },
        )
        return {"address": live_addr, "comparison": comparison}

    # ── ML Feature Extraction WebSocket ─────────────────────────────────────

    def _do_ml_extract(data_types: list, progress_cb: Any, cancel_event: Any = None) -> dict:
        """Run forensic feature extraction (Steps 1–6 of the full pipeline) in a
        thread-pool executor, streaming per-file progress via *progress_cb*.

        data_types: list containing 'images' and/or 'pdfs'.
        progress_cb(msg: dict) is called after every individual file so the WebSocket
        handler can forward it to the browser in real time.
        cancel_event: optional threading.Event; when set, the loop exits after the
        current file and returns a partial result with cancelled=True.

        Returns a summary dict: {total_rows, total_failed, elapsed_s, cancelled}.
        """
        import csv as _csv  # noqa: PLC0415
        import time as _time  # noqa: PLC0415
        from pathlib import Path as _P  # noqa: PLC0415

        _r = _P(__file__).resolve().parent.parent.parent  # repo root

        # Output CSV paths — same locations as the pipeline script uses.
        _img_csv = str(_r / "fraud_model" / "data" / "training_data_image.csv")
        _pdf_csv = str(_r / "fraud_model" / "data" / "training_data_pdf.csv")

        # Image column list (matches collect_training_samples._COLUMNS)
        # PDF column list (matches collect_training_samples_pdf._COLUMNS)
        # Instead of duplicating — and inevitably drifting out of sync with — the
        # canonical _COLUMNS list in each collection script, we import it directly.
        import sys as _sys  # noqa: PLC0415
        _scripts_dir = str(_r / "fraud_model" / "scripts")
        if _scripts_dir not in _sys.path:
            _sys.path.insert(0, _scripts_dir)
        from collect_training_samples import _COLUMNS as _IMG_COLS  # noqa: PLC0415
        from collect_training_samples_pdf import _COLUMNS as _PDF_COLS  # noqa: PLC0415

        # Build the list of extraction steps based on requested data_types.
        # Each step is:  (step_num, label_name, folder_path, label_int, csv_path, is_image, append)
        _steps: list = []
        step_num = 0

        if "images" in data_types:
            for folder_name, label_int, label_name in [
                ("original_images",         0, "ORIGINAL images"),
                ("original_derived_images", 1, "ORIGINAL DERIVED images"),
                ("tampered_images",         2, "TAMPERED images"),
                ("tampered_derived_images", 3, "TAMPERED DERIVED images"),
            ]:
                step_num += 1
                _steps.append((
                    step_num, label_name,
                    _r / "fraud_model" / "sample" / folder_name,
                    label_int, _img_csv, True,
                    step_num > 1,   # append = True for steps 2+ within same CSV
                ))

        if "pdfs" in data_types:
            pdf_step_start = step_num
            for folder_name, label_int, label_name in [
                ("original_pdfs",   0, "ORIGINAL PDFs"),
                ("tampered_pdfs",   1, "TAMPERED PDFs"),
            ]:
                step_num += 1
                _steps.append((
                    step_num, label_name,
                    _r / "fraud_model" / "sample" / folder_name,
                    label_int, _pdf_csv, False,
                    step_num > pdf_step_start + 1,   # append = True for step 2 of PDF
                ))

        total_rows = 0
        total_failed = 0
        t0 = _time.time()
        total_steps = len(_steps)

        for (snum, sname, folder_path, label_int, csv_path, is_image, append_csv) in _steps:
            # Stop early if the client has requested cancellation.
            if cancel_event and cancel_event.is_set():
                break
            # Notify the client that this step is starting and what's in the folder.
            if not folder_path.is_dir():
                progress_cb({
                    "type": "step_skip", "step_num": snum, "total_steps": total_steps,
                    "step_name": sname, "reason": f"Folder not found: {folder_path.name}",
                })
                continue

            # Count supported files in the folder before starting the step.
            if is_image:
                _exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
                files = sorted(p for p in folder_path.iterdir() if p.suffix.lower() in _exts)
            else:
                files = sorted(p for p in folder_path.iterdir() if p.suffix.lower() == ".pdf")

            if not files:
                progress_cb({
                    "type": "step_skip", "step_num": snum, "total_steps": total_steps,
                    "step_name": sname, "reason": f"No files found in {folder_path.name}",
                })
                continue

            progress_cb({
                "type": "step_start", "step_num": snum, "total_steps": total_steps,
                "step_name": sname, "folder": folder_path.name, "file_count": len(files),
                "label": label_int,
            })

            # Make sure the output CSV directory exists.
            _P(csv_path).parent.mkdir(parents=True, exist_ok=True)
            write_header = not append_csv or not _P(csv_path).exists()
            cols = _IMG_COLS if is_image else _PDF_COLS

            rows_written = 0
            rows_failed  = 0

            with open(csv_path, "a" if append_csv else "w", newline="", encoding="utf-8") as _f:
                writer = _csv.DictWriter(_f, fieldnames=cols)
                if write_header:
                    writer.writeheader()

                for idx, fpath in enumerate(files, 1):
                    # Check per-file so cancellation is responsive even in large folders.
                    if cancel_event and cancel_event.is_set():
                        break
                    try:
                        if is_image:
                            from basetruth.analysis.image_forensics_detect import (  # noqa: PLC0415
                                run_forensics as _run_img,
                            )
                            result  = _run_img(str(fpath))
                        else:
                            from basetruth.analysis.pdf_forensics_detect import (  # noqa: PLC0415
                                run_pdf_forensics as _run_pdf,
                            )
                            result  = _run_pdf(str(fpath))

                        summary = result.get("scan_summary", {})
                        score   = float(summary.get("forgery_score_0_100", 0.0))
                        verdict = str(summary.get("forensic_verdict", "UNKNOWN"))
                        fsize   = int(summary.get("file_size_bytes", fpath.stat().st_size))
                        layers  = result.get("layers", {})

                        # Import the feature-extraction helper from the collection script.
                        # _scripts_dir is already on sys.path (added at the top of this function).
                        if is_image:
                            from collect_training_samples import _extract_features as _eff  # noqa: PLC0415
                        else:
                            from collect_training_samples_pdf import _extract_features as _eff  # noqa: PLC0415

                        row = _eff(
                            layers=layers, score=score, verdict=verdict,
                            filename=fpath.name, file_size=fsize,
                        )
                        row["label"] = label_int
                        writer.writerow(row)
                        _f.flush()
                        rows_written += 1
                        total_rows   += 1

                        progress_cb({
                            "type": "file_done", "step_num": snum, "step_name": sname,
                            "file": fpath.name, "index": idx, "total": len(files),
                            "verdict": verdict, "score": round(score, 1),
                            "label": label_int, "ok": True,
                        })

                    except Exception as _file_err:
                        rows_failed  += 1
                        total_failed += 1
                        progress_cb({
                            "type": "file_done", "step_num": snum, "step_name": sname,
                            "file": fpath.name, "index": idx, "total": len(files),
                            "verdict": "ERROR", "score": 0.0,
                            "label": label_int, "ok": False, "error": str(_file_err),
                        })

            progress_cb({
                "type": "step_done", "step_num": snum, "step_name": sname,
                "rows_written": rows_written, "rows_failed": rows_failed,
            })

        return {
            "total_rows": total_rows,
            "total_failed": total_failed,
            "elapsed_s": round(_time.time() - t0, 1),
            "cancelled": bool(cancel_event and cancel_event.is_set()),
        }

    @app.websocket("/api/v1/ml/extract/ws")
    async def ml_extract_websocket(websocket: WebSocket) -> None:
        """WebSocket: stream forensic feature-extraction progress in real time.

        Client sends a single JSON config message::

            {"data_types": ["images", "pdfs"]}   # or just ["images"] or ["pdfs"]

        Server streams JSON progress messages::

            {"type": "step_start", "step_num": 1, "total_steps": 6, "step_name": "ORIGINAL images",
             "folder": "original_images", "file_count": 5, "label": 0}
            {"type": "file_done",  "step_num": 1, "file": "doc.jpg", "index": 3, "total": 5,
             "verdict": "ORIGINAL", "score": 12.3, "label": 0, "ok": true}
            {"type": "step_done",  "step_num": 1, "rows_written": 5, "rows_failed": 0}
            {"type": "step_skip",  "step_num": 2, "reason": "Folder not found"}
            {"type": "all_done",   "total_rows": 20, "total_failed": 0, "elapsed_s": 45.2}
        """
        await websocket.accept()
        _ext_log = _get_logger("basetruth.ml_extract")
        _ext_log.info("Feature extraction WebSocket connected")

        # Read config from client with a 5 s timeout.
        try:
            config = await _asyncio.wait_for(websocket.receive_json(), timeout=5.0)
        except Exception as _cfg_err:
            await websocket.send_json({"type": "error", "message": f"Expected JSON config: {_cfg_err}"})
            await websocket.close(code=1008)
            return

        data_types = config.get("data_types", ["images"])
        if not isinstance(data_types, list):
            data_types = ["images"]
        # Allowlist — only accept the two known types.
        data_types = [d for d in data_types if d in {"images", "pdfs"}]
        if not data_types:
            await websocket.send_json({"type": "error", "message": "Specify at least one of: images, pdfs."})
            await websocket.close(code=1008)
            return

        _ext_log.info("Starting feature extraction", extra={"data_types": data_types})
        loop         = _asyncio.get_running_loop()
        msg_q: _asyncio.Queue = _asyncio.Queue()
        # cancel_event is shared with the executor thread so we can signal early stop.
        cancel_event = _threading.Event()

        def _ext_cb(msg: dict) -> None:
            """Called from the sync extraction thread — bridge to the async queue."""
            loop.call_soon_threadsafe(msg_q.put_nowait, msg)

        extract_fut = loop.run_in_executor(None, _do_ml_extract, data_types, _ext_cb, cancel_event)

        # Drain the queue while extraction runs, sending every message to the browser.
        # If the client closes the connection (Stop button), WebSocketDisconnect is raised
        # inside send_json — we catch it, signal the executor thread, and exit cleanly.
        try:
            while not extract_fut.done():
                try:
                    msg = await _asyncio.wait_for(msg_q.get(), timeout=0.15)
                    await websocket.send_json(msg)
                except _asyncio.TimeoutError:
                    pass

            # Flush any remaining messages that arrived just before the future completed.
            while not msg_q.empty():
                await websocket.send_json(msg_q.get_nowait())

            summary = extract_fut.result()
            if summary.get("cancelled"):
                await websocket.send_json({"type": "cancelled", **summary})
                _ext_log.info("Feature extraction cancelled by client", extra=summary)
            else:
                await websocket.send_json({"type": "all_done", **summary})
                _ext_log.info("Feature extraction finished", extra=summary)
        except WebSocketDisconnect:
            # Client clicked Stop — signal the extraction thread to finish cleanly.
            _ext_log.warning("Extraction client disconnected — setting cancel flag")
            cancel_event.set()
            return
        except Exception as _ext_err:
            _ext_log.error("Feature extraction failed", extra={"error": str(_ext_err)})
            try:
                await websocket.send_json({"type": "error", "message": str(_ext_err)})
            except Exception:
                pass

        try:
            await websocket.close(code=1000)
        except Exception:
            pass

    # ── ML Training WebSocket ────────────────────────────────────────────────

    def _do_ml_train(model_type: str, progress_cb: Any) -> Dict[str, Any]:
        """Run training for one model type in a thread-pool executor.

        This is a plain synchronous function (no asyncio) so it can be
        submitted to run_in_executor without blocking the event loop.
        model_type must be 'image' or 'pdf'.
        progress_cb(step, pct) is called at each training milestone.
        """
        from pathlib import Path as _P  # noqa: PLC0415
        # Resolve repo root from this file: src/basetruth/api.py → 2 levels up
        _r = _P(__file__).resolve().parent.parent.parent

        if model_type == "image":
            from basetruth.analysis.ml_scorer import train as _train_img  # noqa: PLC0415
            csv = str(_r / "fraud_model" / "data" / "training_data_image.csv")
            pkl = str(_r / "fraud_model" / "models" / "ml_scorer_image.pkl")
            return _train_img([csv], pkl, progress_cb=progress_cb)
        elif model_type == "pdf":
            from basetruth.analysis.ml_scorer_pdf import train_pdf as _train_pdf  # noqa: PLC0415
            csv = str(_r / "fraud_model" / "data" / "training_data_pdf.csv")
            pkl = str(_r / "fraud_model" / "models" / "ml_scorer_pdf.pkl")
            return _train_pdf([csv], pkl, progress_cb=progress_cb)
        else:
            raise ValueError(f"Unknown model type: {model_type!r}")

    @app.websocket("/api/v1/ml/train/ws")
    async def ml_train_websocket(websocket: WebSocket) -> None:
        """WebSocket: stream ML training progress in real time.

        Client sends a single JSON message with which model(s) to train::

            {"models": ["image", "pdf"]}   # or just ["image"] or ["pdf"]

        Server streams JSON log messages::

            {"type": "log",   "model": "image", "step": "Fold 1/5 done...", "pct": 32}
            {"type": "done",  "model": "image", "metrics": {...}}
            {"type": "error", "model": "image", "message": "Training aborted: ..."}

        After all models finish a final ``{"type": "all_done"}`` is sent and
        the connection is closed cleanly.
        """
        await websocket.accept()
        _ml_log = _get_logger("basetruth.ml_train")
        _ml_log.info("ML training WebSocket connected")

        # Read configuration from the client (5 s timeout)
        try:
            config = await _asyncio.wait_for(websocket.receive_json(), timeout=5.0)
        except Exception as _cfg_err:
            await websocket.send_json({"type": "error", "message": f"Expected JSON config: {_cfg_err}"})
            await websocket.close(code=1008)
            return

        models_to_train = config.get("models", ["image"])
        if not isinstance(models_to_train, list):
            models_to_train = ["image"]
        # Only allow known types — guard against injection of arbitrary model names
        models_to_train = [m for m in models_to_train if m in {"image", "pdf"}]
        if not models_to_train:
            await websocket.send_json({"type": "error", "message": "Specify at least one of: image, pdf."})
            await websocket.close(code=1008)
            return

        loop = _asyncio.get_running_loop()

        for model_type in models_to_train:
            _ml_log.info("Starting training", extra={"model": model_type})

            # A thread-safe async queue bridges the sync training thread (which calls
            # progress_cb) and this async coroutine (which sends to the WebSocket).
            msg_queue: _asyncio.Queue = _asyncio.Queue()

            def _make_cb(_mt: str, _q: _asyncio.Queue) -> Any:
                """Return a closure that puts progress messages on the async queue.

                The training function runs in a thread, so we use
                call_soon_threadsafe to schedule the queue.put_nowait on the
                event loop — this is the only thread-safe way to communicate
                from a sync thread to an async coroutine.
                """
                def _cb(step: str, pct: int) -> None:
                    loop.call_soon_threadsafe(
                        _q.put_nowait,
                        {"type": "log", "model": _mt, "step": step, "pct": pct},
                    )
                return _cb

            progress_cb = _make_cb(model_type, msg_queue)

            # Submit the blocking training call to the default thread pool so
            # the event loop stays responsive while training runs in the background.
            train_fut = loop.run_in_executor(None, _do_ml_train, model_type, progress_cb)

            # Drain the message queue while training is still running.  The 0.15 s
            # timeout means we check train_fut.done() roughly 6× per second.
            while not train_fut.done():
                try:
                    msg = await _asyncio.wait_for(msg_queue.get(), timeout=0.15)
                    await websocket.send_json(msg)
                except _asyncio.TimeoutError:
                    pass

            # Drain any messages that arrived in the final moments before done.
            while not msg_queue.empty():
                await websocket.send_json(msg_queue.get_nowait())

            # Retrieve the training result (or re-raise any exception).
            try:
                result = train_fut.result()
                await websocket.send_json({"type": "done", "model": model_type, "metrics": result})
                _ml_log.info("Training finished", extra={"model": model_type, "metrics": result})
            except Exception as _train_err:
                _ml_log.error("Training failed", extra={"model": model_type, "error": str(_train_err)})
                await websocket.send_json({"type": "error", "model": model_type, "message": str(_train_err)})

        await websocket.send_json({"type": "all_done"})
        try:
            await websocket.close(code=1000)
        except Exception:
            pass

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

    # ── Video KYC — WebRTC signaling (lower latency, lower bandwidth) ─────

    @app.post("/kyc/webrtc/{session_id}/offer", tags=["Video KYC"], include_in_schema=False)
    async def kyc_webrtc_offer(session_id: str, req: WebRTCOfferRequest) -> Dict[str, Any]:
        """WebRTC signaling endpoint — accept an SDP offer and return an SDP answer.

        The browser creates an RTCPeerConnection, adds the camera video track,
        generates an SDP offer and POSTs it here.  The server returns an SDP
        answer.  After ICE exchange completes the browser streams video directly
        over WebRTC instead of the WebSocket JPEG loop, giving lower latency
        and better CPU/bandwidth efficiency.

        Results are delivered back to the browser via a polling endpoint at
        GET /kyc/sessions/{session_id} — the browser polls every 500 ms.

        Requires: ``pip install aiortc``
        Falls back gracefully if aiortc is not installed (returns 501).
        """
        try:
            from aiortc import RTCPeerConnection, RTCSessionDescription  # type: ignore
        except ImportError:
            from fastapi.responses import JSONResponse as _JR
            return _JR(
                status_code=501,
                content={
                    "error": "aiortc not installed",
                    "detail": "Install with: pip install aiortc",
                },
            )

        session = _kyc_store.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found or expired.")
        if session.status not in ("waiting", "active"):
            raise HTTPException(status_code=409, detail=f"Session is {session.status}.")

        session.status = "active"

        pc = RTCPeerConnection()
        loop = _asyncio.get_running_loop()

        @pc.on("track")
        def on_track(track: Any) -> None:
            if track.kind != "video":
                return

            async def _consume() -> None:
                """Pull frames from the WebRTC video track and process them."""
                try:
                    while True:
                        frame = await _asyncio.wait_for(track.recv(), timeout=5.0)
                        # Convert aiortc VideoFrame → numpy BGR
                        img = frame.to_ndarray(format="bgr24")
                        import cv2 as _cv2_rtc
                        import base64 as _b64_rtc
                        _, buf = _cv2_rtc.imencode(".jpg", img, [_cv2_rtc.IMWRITE_JPEG_QUALITY, 80])
                        b64 = _b64_rtc.b64encode(buf).decode("utf-8")
                        result = await loop.run_in_executor(
                            None, _process_kyc_frame, session, b64
                        )
                        if result.get("type") == "result":
                            session.webrtc_result = result
                            _kyc_log.info(
                                "KYC WebRTC session result",
                                extra={
                                    "session_id": session_id,
                                    "passed": result.get("passed"),
                                    "score": result.get("display_score"),
                                },
                            )
                            break
                except _asyncio.TimeoutError:
                    _kyc_log.warning("WebRTC track receive timeout", extra={"session_id": session_id})
                except Exception as exc:  # noqa: BLE001
                    _kyc_log.error("WebRTC frame error", extra={"session_id": session_id, "error": str(exc)})

            _asyncio.ensure_future(_consume())

        offer = RTCSessionDescription(sdp=req.sdp, type=req.type)
        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        # Store the PC so it can be closed later
        if not hasattr(session, "webrtc_pcs"):
            session.webrtc_pcs = []
        session.webrtc_pcs.append(pc)

        _kyc_log.info("WebRTC offer accepted", extra={"session_id": session_id})
        return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

    @app.get("/kyc/webrtc/{session_id}/result", tags=["Video KYC"], include_in_schema=False)
    def kyc_webrtc_result(session_id: str) -> Dict[str, Any]:
        """Poll for the result of a WebRTC KYC session.

        Returns the current session status and, once complete, the face-match
        result.  The browser polls this endpoint every 500 ms after establishing
        the WebRTC connection.
        """
        session = _kyc_store.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found.")
        result = getattr(session, "webrtc_result", None)
        return {
            "status": session.status,
            "result": result,
        }

    # ── Entity registry endpoints ─────────────────────────────────────────

    @app.get(
        "/api/v1/entities",
        tags=["Entities"],
        summary="Search Entity Registry",
        responses={
            503: {"description": "Database unavailable"},
        },
    )
    def list_entities(
        q: str = Query("", description="Search term — name, PAN, Aadhaar, email, or phone number."),
        field: str = Query("all", description="Field to search: `all` | `name` | `pan` | `aadhar` | `email` | `phone`."),
        limit: int = Query(100, ge=1, le=1000, description="Maximum rows to return (1–1000)."),
    ) -> List[Dict[str, Any]]:
        """Search the entity registry by name, PAN, Aadhaar, email, or phone.

        Returns the most-recent entities when no query is supplied.
        Each result includes total scan count and latest risk level for quick triage.
        """
        try:
            from basetruth.store import search_entities
            return search_entities(query=q, search_field=field, limit=limit)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"DB unavailable: {exc}") from exc

    @app.get(
        "/api/v1/entities/{entity_ref}",
        tags=["Entities"],
        summary="Get Entity Profile",
        responses={
            404: {"description": "Entity not found"},
            503: {"description": "Database unavailable"},
        },
    )
    def get_entity(entity_ref: str) -> Dict[str, Any]:
        """Return the full profile for a single entity, including all scan summaries.

        The `entity_ref` is the unique identifier assigned when the entity was
        first created (typically PAN number, Aadhaar, or a case reference).
        """
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

    @app.get(
        "/api/v1/entities/{entity_ref}/scans",
        tags=["Entities"],
        summary="List Entity Scan History",
        responses={
            404: {"description": "Entity not found"},
            503: {"description": "Database unavailable"},
        },
    )
    def list_entity_scans(entity_ref: str) -> List[Dict[str, Any]]:
        """List all document scans for a specific entity, most-recent first.

        Each item includes the full JSON forensic report so analysts can review
        the signals that triggered a flag without needing filesystem access.
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

    @app.get(
        "/api/v1/scans/{scan_id}/report.pdf",
        tags=["Reports"],
        summary="Download Scan PDF Report",
        responses={
            200: {"content": {"application/pdf": {}}, "description": "PDF audit report"},
            404: {"description": "Report not found for this scan ID"},
            503: {"description": "Database unavailable"},
        },
    )
    def download_scan_pdf(scan_id: int) -> Any:
        """Download the PDF audit report for a specific scan.

        Returns the binary PDF so auditors can save it locally or attach it
        to a case without needing filesystem access. Look up the `scan_id`
        from the entity scan history endpoint.
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

    @app.get(
        "/api/v1/scans/recent",
        tags=["Reports"],
        summary="List Recent Scans",
        responses={
            503: {"description": "Database unavailable"},
        },
    )
    def list_recent_scans(
        limit: int = Query(50, ge=1, le=500, description="Number of most-recent scans to return (1–500)."),
    ) -> List[Dict[str, Any]]:
        """Return the most-recent document scans across all entities.

        Each item includes entity reference, document type, verdict, forgery score,
        and timestamp. Useful for a real-time fraud-monitoring dashboard.
        """
        try:
            from basetruth.store import list_recent_scans as _list
            return _list(limit=limit)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"DB unavailable: {exc}") from exc

    @app.get(
        "/api/v1/db/stats",
        tags=["System"],
        summary="Database Statistics",
        response_model=DBStatsResponse,
    )
    def db_stats() -> Dict[str, Any]:
        """Return aggregate counts from the database.

        Returns the total number of entities, document scans, and high-risk
        flags. Use this to populate a management dashboard or monitor growth.
        """
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
