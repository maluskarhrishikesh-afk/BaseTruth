"""Face Scan page — run deepfake and liveliness analysis on one uploaded face image.

This page allows agents to manually inspect images for face integrity and liveness.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import streamlit as st
import requests

from basetruth.face_scan.service import run_face_scan_static
from basetruth.logger import get_logger
from basetruth.service import BaseTruthService
from basetruth.ui.components import _page_title, _db_available_cached
from basetruth.db import list_face_scan_live_results
from basetruth.ui.pages.scan import (
    _RESULT_CSS,
    _syntax_highlight_json,
)
from basetruth.ui.pages.video_kyc import _API_INTERNAL, _API_EXTERNAL

log = get_logger(__name__)

_VERDICT_BADGE: Dict[str, str] = {
    "GENUINE": "🟢 GENUINE",
    "SUSPICIOUS": "🟠 SUSPICIOUS",
    "DEEPFAKE": "🔴 DEEPFAKE",
    "INCONCLUSIVE": "⚪ INCONCLUSIVE",
    "LIVENESS_FAILED": "🟡 LIVENESS FAILED",
    "UNAVAILABLE": "⚪ UNAVAILABLE",
}

_VERDICT_BORDER: Dict[str, str] = {
    "GENUINE": "rgba(34,197,94,0.28)",
    "SUSPICIOUS": "rgba(249,115,22,0.28)",
    "DEEPFAKE": "rgba(239,68,68,0.28)",
    "INCONCLUSIVE": "rgba(148,163,184,0.18)",
    "LIVENESS_FAILED": "rgba(234,179,8,0.24)",
    "UNAVAILABLE": "rgba(148,163,184,0.18)",
}

def _verdict_color(verdict: str) -> str:
    verdict_upper = (verdict or "").upper()
    if verdict_upper == "GENUINE":
        return "#22c55e"
    if verdict_upper == "SUSPICIOUS":
        return "#f97316"
    if verdict_upper == "DEEPFAKE":
        return "#ef4444"
    if verdict_upper == "LIVENESS_FAILED":
        return "#eab308"
    return "#94a3b8"

def _page_face_scan(_service: BaseTruthService = None) -> None:  # type: ignore[assignment]
    """Render the Face Scan page."""
    st.markdown(_RESULT_CSS, unsafe_allow_html=True)
    st.markdown(_page_title("🥸", "Face Scan"), unsafe_allow_html=True)

    with st.expander("ℹ️ How this screen works", expanded=False):
        st.markdown(
            """
<div style="margin:10px 0">
<div class="bt-step-row"><div class="bt-step-num">1</div>
Upload one face image (e.g. selfie or ID photo).</div>
<div class="bt-step-row"><div class="bt-step-num">2</div>
BaseTruth auto-detects the primary face in the image.</div>
<div class="bt-step-row"><div class="bt-step-num">3</div>
The static Face Scan engine runs deterministic face-authenticity checks and returns a production-grade result payload.</div>
<div class="bt-step-row"><div class="bt-step-num">4</div>
You get a verdict, risk score, confidence score, evidence, and a downloadable JSON result.</div>
</div>

> No data is saved from this page. Use this as an instant face integrity checker.
""",
            unsafe_allow_html=True,
        )

    tab_static, tab_live = st.tabs(["🖼️ Static Photo Scan", "🎥 Live Camera Challenge"])

    with tab_static:
        upload = st.file_uploader(
            "Drop a face image here to run deepfake analysis",
            type=None,
            accept_multiple_files=False,
            key="face_scan_upload",
            help="Supported: JPG, PNG, WebP",
        )

        if upload is not None:
            file_bytes = upload.read()
            filename = upload.name or "face_image"
            cache_key = f"face_result_{filename}_{len(file_bytes)}"

            if st.session_state.get(cache_key) is None:
                with st.spinner("🥸 Running static face scan..."):
                    result = run_face_scan_static(file_bytes, filename)
                st.session_state[cache_key] = result
            else:
                result = st.session_state[cache_key]

            if result.get("error"):
                st.error(f"Face analysis failed: {result['error']}")
            else:
                _render_static_result(result, filename)

    with tab_live:
        st.markdown(
            "Use your webcam to perform an interactive Face Scan live session with dedicated liveness, temporal-consistency, and replay checks."
        )
        
        CH_LABELS = {
            "look_straight": "Look straight",
            "blink": "Close eyes / Blink",
            "turn_left": "Turn left",
            "turn_right": "Turn right",
            "nod": "Nod head"
        }
        challenges = st.multiselect(
            "Select Challenges",
            ["look_straight", "blink", "turn_left", "turn_right", "nod"],
            default=["look_straight", "blink", "nod"],
            format_func=lambda x: CH_LABELS.get(x, x),
            key="face_scan_challenges"
        )

        if st.button("Generate Live Challenge Link", type="primary", key="start_live_btn"):
            payload = {"challenges": challenges}
            try:
                resp = requests.post(f"{_API_INTERNAL}/api/v1/face-scan/sessions", json=payload, timeout=10)
                resp.raise_for_status()
                session_data = resp.json()
                st.session_state.face_scan_live_sid = session_data["session_id"]
                st.session_state.face_scan_live_url = session_data["session_url"]
            except requests.RequestException as e:
                st.error(f"Failed to start live session: {e}")

        if "face_scan_live_sid" in st.session_state:
            sid = st.session_state.face_scan_live_sid
            session_path = st.session_state.get("face_scan_live_url", f"/face-scan/live/{sid}")
            ui_url = f"{_API_EXTERNAL}{session_path}"
            
            import time
            _should_auto_refresh = False
            
            try:
                status_resp = requests.get(f"{_API_INTERNAL}/api/v1/face-scan/sessions/{sid}", timeout=5)
                if status_resp.status_code == 200:
                    status_data = status_resp.json()
                    status = status_data.get("status", "waiting")
                    
                    if status == "expired":
                        # Session TTL has elapsed — clean it up and prompt a fresh start.
                        st.warning("⏰ This session has expired (sessions last 20 minutes). Start a new one below.")
                        del st.session_state.face_scan_live_sid
                        st.session_state.pop("face_scan_live_url", None)
                        if st.button("Start New Session", key="expired_new_btn"):
                            st.rerun()
                    elif status in ["completed", "failed", "processing"]:
                        st.subheader("Live Session Result")
                        result = status_data.get("result")
                        # Determine if challenges are done but the LLM narrative is still running
                        narrative_pending = bool(result.get("narrative_pending", False)) if result else False
                        is_verifying = (status == "processing") or narrative_pending
                        if status == "completed" and result and not is_verifying:
                            st.success("✅ Live Face Scan completed successfully.")
                            _render_static_result(result, result.get("filename", f"face_scan_live_{sid}.jpg"))
                        elif is_verifying:
                            # Challenges done, LLM narrative still being generated
                            st.info("⏳ Verifying results and generating response…")
                            _should_auto_refresh = True
                        else:
                            st.error("❌ Live Face Scan failed or ended before completion.")
                            st.json(status_data)
                        
                        if not _should_auto_refresh and st.button("Start New Session"):
                            del st.session_state.face_scan_live_sid
                            st.session_state.pop("face_scan_live_url", None)
                            st.rerun()
                    else:
                        completed = int(status_data.get("challenges_completed", 0) or 0)
                        total = int(status_data.get("total_challenges", 0) or 0)
                        current = status_data.get("current_instruction") or status_data.get("current_challenge") or "Follow the on-screen instruction."
                        st.info(f"⏳ Live session is active. Progress: {completed}/{total} challenges completed.")
                        st.caption(f"Current challenge: {current}")
                        st.success("Open the live camera page in a new tab if it is not already running.")
                        st.link_button("🎥 Open Live Face Scan", ui_url, type="primary")
                        _should_auto_refresh = True
                        
                        if st.button("Cancel Session"):
                            del st.session_state.face_scan_live_sid
                            st.session_state.pop("face_scan_live_url", None)
                            st.rerun()
            except requests.RequestException as e:
                st.error(f"Failed to fetch session status: {e}")
                if st.button("Reset Live Session"):
                    del st.session_state.face_scan_live_sid
                    st.session_state.pop("face_scan_live_url", None)
                    st.rerun()
                    
            if _should_auto_refresh:
                time.sleep(2)
                st.rerun()

        # ── Recent Live Scan Recordings ─────────────────────────────────────
        # Persistent panel querying the DB so operators can watch or delete
        # recordings from past sessions — even after Streamlit session state
        # has been cleared by navigation, refresh, or "Start New Session".
        st.markdown("---")
        with st.expander("📋 Recent Live Scan Recordings", expanded=True):
            if not _db_available_cached():
                st.warning("Database unavailable — cannot load recent recordings.", icon="⚠️")
            else:
                _recent_rows = list_face_scan_live_results(limit=10)
                _rows_with_video = [r for r in _recent_rows if r.video_key]
                if not _rows_with_video:
                    st.info("No recorded live scan sessions yet.", icon="ℹ️")
                else:
                    # Render each recorded session as a compact row with Watch action.
                    for _row in _rows_with_video:
                        _rec_url_key = f"rec_video_url_{_row.session_id}"
                        _rec_deleted_key = f"rec_video_deleted_{_row.session_id}"
                        _sid_short = (_row.session_id[:16] + "\u2026") if len(_row.session_id) > 16 else _row.session_id
                        _ts = _row.created_at.strftime("%Y-%m-%d %H:%M") if _row.created_at else "N/A"
                        _v_label = _VERDICT_BADGE.get(str(_row.verdict or "").upper(), _row.verdict or "—")
                        _c1, _c2, _c3, _c4 = st.columns([4, 2, 2, 2])
                        _c1.code(_sid_short, language=None)
                        _c2.markdown(_v_label)
                        _c3.caption(_ts)
                        if st.session_state.get(_rec_deleted_key):
                            _c4.caption("🗑️ Deleted")
                        elif _c4.button("▶ Watch", key=f"rec_watch_{_row.session_id}"):
                            try:
                                _vid_r = requests.get(
                                    f"{_API_INTERNAL}/api/v1/face-scan/sessions/{_row.session_id}/video",
                                    timeout=10,
                                )
                                if _vid_r.status_code == 200:
                                    st.session_state[_rec_url_key] = _vid_r.json().get("video_url")
                                else:
                                    st.error(f"Could not load video: {_vid_r.status_code} {_vid_r.text}", icon="❌")
                            except requests.RequestException as _rv_exc:
                                st.error(f"API error: {_rv_exc}", icon="❌")
                        # Show the player inline when a URL has been fetched for this row.
                        _rec_url = st.session_state.get(_rec_url_key)
                        if _rec_url:
                            st.video(_rec_url)
                            # Allow the user to close/refresh the player without deleting.
                            if st.button("✖ Close player", key=f"rec_close_{_row.session_id}"):
                                st.session_state.pop(_rec_url_key, None)
                                st.rerun()
                            st.markdown("---")

def _render_static_result(result: Dict[str, Any], filename: str) -> None:
    verdict = str(result.get("verdict", "INCONCLUSIVE") or "INCONCLUSIVE").upper()
    score = float(result.get("risk_score_0_100", 0.0) or 0.0)
    confidence = float(result.get("confidence_0_100", 0.0) or 0.0)
    
    verdict_label = _VERDICT_BADGE.get(verdict, verdict)
    v_color = _verdict_color(verdict)
    v_border = _VERDICT_BORDER.get(verdict, "rgba(148,163,184,0.18)")
    icon = "🧑‍🦲"

    # ── Classification banner (mirrors Scan Document) ──────────────────────
    st.markdown(
        f"""
        <div class="bt-classify-banner">
          <div class="bt-classify-icon">{icon}</div>
          <div class="bt-classify-detail">
            <div class="bt-classify-type" style="color:{v_color};">{verdict_label}</div>
            <div class="bt-classify-meta">
                Face Scan &nbsp;·&nbsp; Risk Score: <strong>{score:.1f}/100</strong>
                &nbsp;·&nbsp; Confidence: <strong>{confidence:.1f}/100</strong>
                &nbsp;·&nbsp; Static Heuristic Engine
                &nbsp;·&nbsp; File: <strong>{filename}</strong>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    payload: Dict[str, Any] = result

    # ── Honest Review card ──────────────────────────────────────────────────
    review_border = v_color
    review_text = str(result.get("honest_review") or result.get("overall_explanation") or "")
    
    st.markdown(
        f"""
        <div style="
            border-radius: 12px;
            border-left: 4px solid {review_border};
            background: rgba(15, 23, 42, 0.85);
            padding: 16px 22px;
            margin-bottom: 18px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.18);
        ">
            <div style="
                font-size: 0.72rem;
                font-weight: 700;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                color: {review_border};
                margin-bottom: 8px;
            ">🔎 Agent Review</div>
            <div style="
                font-size: 0.92rem;
                line-height: 1.65;
                color: #e2e8f0;
            ">{review_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Session Recording panel ─────────────────────────────────────────────
    # Only visible when a video was recorded for this session (live mode only).
    # The video_key is attached to the result dict in build_live_face_scan_result.
    video_key = result.get("video_key")
    session_id_for_video = result.get("trace", {}).get("decision_trace_id", "")
    # Use the filename to derive the session_id for API calls (filename contains it).
    _live_filename = str(result.get("filename", ""))
    _live_sid = None
    if _live_filename.startswith("face_scan_live_"):
        _live_sid = _live_filename.replace("face_scan_live_", "").replace(".jpg", "")

    if video_key and _live_sid:
        with st.expander("🎥 Session Recording", expanded=True):
            # State keys scoped to this session_id so multiple results don't clash.
            _video_url_key = f"face_scan_video_url_{_live_sid}"
            _video_deleted_key = f"face_scan_video_deleted_{_live_sid}"

            if st.session_state.get(_video_deleted_key):
                st.info("🗑️ Recording has been deleted.", icon="ℹ️")
            else:
                # Load or refresh the presigned URL (1-hour expiry).
                if st.button("▶ Watch Recording", key=f"face_scan_video_load_{_live_sid}"):
                    try:
                        _vid_resp = requests.get(
                            f"{_API_INTERNAL}/api/v1/face-scan/sessions/{_live_sid}/video",
                            timeout=10,
                        )
                        if _vid_resp.status_code == 200:
                            st.session_state[_video_url_key] = _vid_resp.json().get("video_url")
                        else:
                            st.error(f"Could not load video: {_vid_resp.status_code} {_vid_resp.text}")
                    except requests.RequestException as _ve:
                        st.error(f"Failed to reach API: {_ve}")

                presigned = st.session_state.get(_video_url_key)
                if presigned:
                    st.video(presigned)

                st.caption(
                    f"Recorded session · auto-deleted after {__import__('os').environ.get('FACE_SCAN_VIDEO_RETENTION_DAYS', '90')} days"
                )

                # Delete recording — requires a confirmation checkbox before firing.
                st.markdown("---")
                _confirm_key = f"face_scan_video_confirm_del_{_live_sid}"
                confirmed = st.checkbox("I confirm I want to permanently delete this recording", key=_confirm_key)
                if confirmed:
                    if st.button("🗑 Delete Recording", type="secondary", key=f"face_scan_video_del_{_live_sid}"):
                        try:
                            _del_resp = requests.delete(
                                f"{_API_INTERNAL}/api/v1/face-scan/sessions/{_live_sid}/video",
                                timeout=10,
                            )
                            if _del_resp.status_code == 200:
                                st.success("Recording deleted successfully.")
                                st.session_state[_video_deleted_key] = True
                                st.session_state.pop(_video_url_key, None)
                                st.rerun()
                            else:
                                st.error(f"Delete failed: {_del_resp.status_code} {_del_resp.text}")
                        except requests.RequestException as _de:
                            st.error(f"Failed to reach API: {_de}")

    highlighted = _syntax_highlight_json(payload)
    st.markdown(
        f"""
        <div class="bt-scan-result-wrap" style="border-color:{v_border};">
            <div class="bt-scan-result-header">
                <span class="bt-doc-type-badge">Face Scan</span>
                <span class="bt-scan-badge" style="color:{v_color}; font-weight:700;">{verdict_label}</span>
            </div>
            <div class="bt-scan-json">{highlighted}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Download + reset row ───────────────────────────────────────────────
    col_dl, col_reset = st.columns([3, 1])
    with col_dl:
        st.download_button(
            "⬇ Download face analysis result as JSON",
            data=json.dumps(payload, indent=2, ensure_ascii=False),
            file_name=f"{Path(filename).stem}_face_result.json",
            mime="application/json",
            use_container_width=True,
            key=f"face_scan_dl_btn_{filename}",
        )
    with col_reset:
        if st.button("🔄 Reset", use_container_width=True, key=f"face_scan_reset_btn_{filename}",
                     help="Clear result and analyze another face"):
            for k in [k for k in st.session_state if k.startswith("face_scan_") or k.startswith("face_result_")]:
                del st.session_state[k]
            st.rerun()
