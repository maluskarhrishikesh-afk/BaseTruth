"""Video KYC page — create secure sessions, schedule appointments, and
conduct in-person verification.

Architecture overview:
  Tab 1  Start KYC Session  — Upload reference ID -> extract ArcFace embedding
                               -> POST /kyc/sessions to API -> share link with
                               customer -> poll status every 2 s -> result + PDF.

  Tab 2  Schedule            — Generate a .ics calendar invite that uses the
                               BaseTruth KYC URL as the meeting link so the
                               customer can open it directly in their browser.

  Tab 3  In-Person Verify    — Webcam-only check for face-to-face interactions
                               (uses st.camera_input, runs locally in Streamlit).
"""
from __future__ import annotations

import io
import os
import socket
import subprocess
import sys
import textwrap
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import requests
import streamlit as st

from basetruth.ui.components import (
    _DB_IMPORTS_OK,
    _page_title,
    _render_entity_link_widget,
    db_available,
    save_identity_check,
)

# ---------------------------------------------------------------------------
# API communication helpers
# ---------------------------------------------------------------------------

# Streamlit runs inside the UI container; it talks to the API container via
# Docker DNS.  In local development both default to localhost.
_API_INTERNAL = os.getenv("BT_API_INTERNAL_URL", "http://localhost:8000")
_API_EXTERNAL = os.getenv("BT_API_EXTERNAL_URL", "http://localhost:8000")
_API_PORT = 8000


@st.cache_resource
def _ensure_local_api() -> bool:
    """Auto-start the FastAPI server when running locally (outside Docker).

    In Docker, BT_API_INTERNAL_URL is set to the service name so we skip.
    In local dev mode the env var is absent and we spawn uvicorn once so
    all /kyc/* endpoints are available immediately.

    Returns True once the server is accepting connections, False on timeout.
    """
    if os.getenv("BT_API_INTERNAL_URL"):
        return True  # Docker / explicit config — server managed externally

    def _port_open() -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as _s:
            _s.settimeout(0.5)
            return _s.connect_ex(("127.0.0.1", _API_PORT)) == 0

    if _port_open():
        return True

    # Spawn uvicorn as a background process.
    # --ws websockets-sansio avoids the HTTP 403 bug in the legacy websockets
    # implementation that ships as the default in uvicorn ≤ 0.42 when paired
    # with websockets 13.x.
    subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn", "basetruth.api:app",
            "--host", "127.0.0.1",
            "--port", str(_API_PORT),
            "--ws", "websockets-sansio",
            "--log-level", "warning",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Wait up to 20 s for the server to accept connections
    deadline = time.time() + 20
    while time.time() < deadline:
        if _port_open():
            return True
        time.sleep(0.5)

    return False


def _api_post(path: str, payload: Dict) -> Optional[Dict]:
    try:
        resp = requests.post(f"{_API_INTERNAL}{path}", json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:  # noqa: BLE001
        st.error(f"API error: {exc}")
        return None


def _api_get(path: str) -> Optional[Dict]:
    try:
        resp = requests.get(f"{_API_INTERNAL}{path}", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:  # noqa: BLE001
        st.error(f"API error: {exc}")
        return None


# ---------------------------------------------------------------------------
# ICS calendar invite generator
# ---------------------------------------------------------------------------

def _make_ics(
    customer_name: str,
    agent_name: str,
    meeting_link: str,
    start_dt: datetime,
    duration_minutes: int,
    description: str,
) -> bytes:
    """Generate a .ics calendar invite as UTF-8 bytes (RFC 5545)."""
    end_dt = start_dt + timedelta(minutes=duration_minutes)
    fmt = "%Y%m%dT%H%M%SZ"
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//BaseTruth//Video KYC//EN",
        "CALSCALE:GREGORIAN",
        "METHOD:REQUEST",
        "BEGIN:VEVENT",
        f"UID:{uuid.uuid4()}@basetruth",
        f"DTSTAMP:{datetime.now(timezone.utc).strftime(fmt)}",
        f"DTSTART:{start_dt.strftime(fmt)}",
        f"DTEND:{end_dt.strftime(fmt)}",
        f"SUMMARY:Video KYC Session -- {customer_name}",
        f"ORGANIZER;CN={agent_name}:mailto:noreply@basetruth.local",
        f"ATTENDEE;CN={customer_name};ROLE=REQ-PARTICIPANT;RSVP=TRUE:mailto:customer@placeholder",
        "STATUS:CONFIRMED",
    ]
    desc_safe     = description.replace("\n", "\\n").replace(",", "\\,").replace(";", "\\;")
    location_safe = meeting_link.replace(",", "\\,").replace(";", "\\;")
    for key, val in [("DESCRIPTION", desc_safe), ("LOCATION", location_safe)]:
        raw    = f"{key}:{val}"
        folded = "\r\n ".join(textwrap.wrap(raw, 75, break_long_words=True, break_on_hyphens=False))
        lines.append(folded)
    lines += ["END:VEVENT", "END:VCALENDAR"]
    return ("\r\n".join(lines) + "\r\n").encode("utf-8")


# ===========================================================================
# Tab 1 -- Start KYC Session
# ===========================================================================

def _tab_start_session() -> None:
    st.markdown(
        """
        <div style="background:linear-gradient(135deg,#1e293b,#0f172a);
                    border:1px solid #334155;border-radius:12px;
                    padding:1rem 1.25rem;margin-bottom:1.2rem">
        <h4 style="color:#e2e8f0;margin:0 0 .4rem">Why a dedicated Video SDK?</h4>
        <p style="color:#94a3b8;font-size:.85rem;margin:0;line-height:1.6">
        Third-party video platforms (Zoom, Teams) block server-side frame access -- making
        real AI liveness and face-match impossible.<br>
        <strong style="color:#c4b5fd">BaseTruth Video KYC</strong> streams frames through
        our own WebSocket layer so RetinaFace + ArcFace run on every frame with full control
        and zero third-party data sharing.
        </p></div>
        """,
        unsafe_allow_html=True,
    )

    # 0. Link entity
    forced_ref, extra_identity = None, None
    if _DB_IMPORTS_OK and db_available():
        forced_ref, extra_identity = _render_entity_link_widget("vkyc_start", mandatory=False)
        st.divider()

    # 1. Reference ID upload
    st.subheader("1  Upload Reference ID")
    doc_file = st.file_uploader(
        "Upload ID document (Aadhaar / PAN / Passport photo)",
        type=["jpg", "jpeg", "png", "webp"],
        key="vkyc_ref_doc",
    )
    if doc_file:
        import base64  # noqa: PLC0415
        import cv2  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
        from basetruth.vision.face import get_face_analyzer  # noqa: PLC0415

        face_app = get_face_analyzer()
        nparr = np.frombuffer(doc_file.getvalue(), np.uint8)
        img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        try:
            faces = face_app.get(img)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Face detection failed: {exc}")
            faces = []

        if faces:
            face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            emb  = face.normed_embedding
            st.session_state["vkyc_ref_emb_b64"] = base64.b64encode(
                emb.astype("float32").tobytes()
            ).decode()
            st.success("Reference face extracted from ID.")
        else:
            st.error("No face found in the uploaded document.")
            st.session_state.pop("vkyc_ref_emb_b64", None)

    # 2. Session parameters
    st.divider()
    st.subheader("2  Session Setup")
    c1, c2 = st.columns(2)
    with c1:
        customer_name = st.text_input(
            "Customer name", placeholder="e.g. Rahul Sharma", key="vkyc_cust_name"
        )
    with c2:
        entity_ref_input = st.text_input(
            "Entity / Case ref", placeholder="e.g. ENT-001", key="vkyc_entity_ref"
        )

    with st.expander("Challenge selection (optional)", expanded=False):
        ALL_CH = ["blink", "turn_left", "turn_right", "nod"]
        CH_LABELS = {
            "blink":      "Close eyes",
            "turn_left":  "Turn head left",
            "turn_right": "Turn head right",
            "nod":        "Nod head",
        }
        selected = st.multiselect(
            "Pick 1-4 challenges (leave empty for 2 random)",
            options=ALL_CH,
            format_func=lambda c: CH_LABELS.get(c, c),
            key="vkyc_challenges",
        )
    challenges = selected or []

    # 3. Create session
    st.divider()
    st.subheader("3  Create and Share")

    ref_b64 = st.session_state.get("vkyc_ref_emb_b64")
    if not ref_b64:
        st.info("Upload a reference ID above to enable face-match (optional -- liveness-only is also supported).")

    if st.button("Create Secure KYC Session", type="primary", use_container_width=True):
        payload: Dict[str, Any] = {
            "customer_name":           customer_name.strip(),
            "entity_ref":              entity_ref_input.strip() or (forced_ref or ""),
            "challenges":              challenges,
            "reference_embedding_b64": ref_b64,
        }
        resp = _api_post("/kyc/sessions", payload)
        if resp:
            sid = resp["session_id"]
            session_url = f"{_API_EXTERNAL}/kyc/{sid}"
            st.session_state["vkyc_active_sid"]      = sid
            st.session_state["vkyc_session_url"]     = session_url
            st.session_state["vkyc_session_created"] = True
            st.session_state["vkyc_doc_filename"]    = doc_file.name if doc_file else ""
            st.session_state["vkyc_forced_ref"]      = forced_ref
            st.session_state["vkyc_extra_identity"]  = extra_identity
            st.rerun()

    # 4. Monitor active session
    if not st.session_state.get("vkyc_session_created"):
        return

    sid         = st.session_state.get("vkyc_active_sid", "")
    session_url = st.session_state.get("vkyc_session_url", "")

    st.success("Session created -- share this URL with your customer:")
    st.code(session_url, language="text")
    st.markdown(
        f'<a href="{session_url}" target="_blank" '
        f'style="display:inline-block;padding:.5rem 1rem;background:#4f46e5;color:#fff;'
        f'border-radius:8px;text-decoration:none;font-weight:600">Open KYC Page</a>',
        unsafe_allow_html=True,
    )
    st.caption("The customer opens this link in their browser -- no app or plugin needed.")
    st.divider()

    # Live status poll (auto-refresh every 2 s while active)
    status_resp = _api_get(f"/kyc/sessions/{sid}")
    if not status_resp:
        return

    status   = status_resp.get("status", "unknown")
    ch_done  = status_resp.get("challenges_completed", 0)
    ch_total = status_resp.get("total_challenges", len(challenges) or 2)
    result   = status_resp.get("result")

    col_s, col_p = st.columns([3, 2])
    with col_s:
        status_colors = {
            "waiting":   ("Yellow circle", "Waiting for customer"),
            "active":    ("Blue circle", "Session in progress"),
            "completed": ("Green circle", "Completed"),
            "failed":    ("Red circle", "Failed"),
            "expired":   ("Black circle", "Expired"),
        }
        icon, label = status_colors.get(status, ("White circle", status))
        st.metric("Session status", label)
    with col_p:
        st.metric("Challenges", f"{ch_done} / {ch_total} done")

    if status in ("waiting", "active"):
        st.progress(ch_done / max(ch_total, 1), text="Liveness challenges progress")
        with st.spinner("Waiting for customer to complete verification..."):
            time.sleep(2)
        st.rerun()

    elif status == "completed" and result:
        passed     = result.get("passed", False)
        disp_score = result.get("display_score", result.get("match_score", 0) * 100)
        cosine_sim = result.get("cosine_similarity", 0.0)

        if passed:
            st.success(f"Identity Verified -- Face match score: {disp_score:.1f}%")
        else:
            st.error(f"Verification Failed -- Score: {disp_score:.1f}%")
            st.caption(result.get("message", ""))

        if _DB_IMPORTS_OK and db_available():
            _kyc_persist(
                result, sid, status_resp,
                st.session_state.get("vkyc_doc_filename", ""),
                st.session_state.get("vkyc_forced_ref"),
                st.session_state.get("vkyc_extra_identity"),
                cosine_sim,
            )

        if st.button("Start New Session", use_container_width=True):
            for k in ["vkyc_active_sid", "vkyc_session_url", "vkyc_session_created",
                      "vkyc_ref_emb_b64", "vkyc_doc_filename"]:
                st.session_state.pop(k, None)
            st.rerun()

    elif status in ("failed", "expired"):
        msg = "Session expired -- please create a new one." if status == "expired" \
              else "Verification failed. Please retry."
        st.error(msg)
        if st.button("Start New Session", use_container_width=True):
            for k in ["vkyc_active_sid", "vkyc_session_url", "vkyc_session_created",
                      "vkyc_ref_emb_b64", "vkyc_doc_filename"]:
                st.session_state.pop(k, None)
            st.rerun()


def _kyc_persist(
    result: Dict,
    session_id: str,
    status_resp: Dict,
    doc_filename: str,
    forced_ref: Optional[str],
    extra_identity: Optional[Dict],
    cosine_sim: float,
) -> None:
    """Save KYC result to DB and render the PDF report."""
    disp_score = float(result.get("display_score", result.get("match_score", 0) * 100))
    is_match   = bool(result.get("passed", False))

    vkyc_result = {
        "is_match":          is_match,
        "confidence":        cosine_sim,
        "cosine_similarity": cosine_sim,
        "display_score":     disp_score,
        "threshold":         0.40,
        "liveness_passed":   True,
        "liveness_state":    "challenge_response",
        "match":             is_match,
        "session_id":        session_id,
        "challenges":        status_resp.get("challenges", []),
    }

    entity_ref  = forced_ref or status_resp.get("entity_ref", "")
    entity_name = ""
    if extra_identity:
        entity_name = (
            f"{extra_identity.get('first_name', '')} "
            f"{extra_identity.get('last_name', '')}".strip()
        )
    elif status_resp.get("customer_name"):
        entity_name = status_resp["customer_name"]

    vkyc_pdf: Optional[bytes] = None
    try:
        from basetruth.reporting.pdf import render_identity_check_pdf  # noqa: PLC0415
        vkyc_pdf = render_identity_check_pdf(
            check_type="video_kyc",
            result=vkyc_result,
            entity_ref=entity_ref,
            entity_name=entity_name,
            doc_filename=doc_filename,
        )
    except Exception:  # noqa: BLE001
        pass

    saved = save_identity_check(
        check_type="video_kyc",
        result=vkyc_result,
        forced_entity_ref=forced_ref,
        extra_identity=extra_identity,
        doc_filename=doc_filename,
        pdf_bytes=vkyc_pdf,
    )
    if saved:
        st.info(
            f"KYC result saved -- ID: {saved['id']}, "
            f"Entity: {saved.get('entity_ref', 'unlinked')}"
        )
    if vkyc_pdf:
        st.download_button(
            "Download KYC Report (PDF)",
            data=vkyc_pdf,
            file_name=f"video_kyc_{entity_ref or 'report'}.pdf",
            mime="application/pdf",
            key="vkyc_pdf_dl",
        )


# ===========================================================================
# Tab 2 -- Schedule Appointment
# ===========================================================================

def _tab_schedule() -> None:
    st.markdown(
        """
        <div style="background:linear-gradient(135deg,#1e293b,#0f172a);
                    border:1px solid #334155;border-radius:12px;
                    padding:1rem 1.25rem;margin-bottom:1.2rem">
        <h4 style="color:#e2e8f0;margin:0 0 .4rem">Schedule a Video KYC Appointment</h4>
        <p style="color:#94a3b8;font-size:.85rem;margin:0;line-height:1.6">
        Create a KYC session first (tab 1) to get the secure URL, then paste it as the
        Meeting Link below. The customer receives a calendar invite -- when they click
        the join link at the scheduled time, they land directly on the BaseTruth KYC page
        in their browser. No Zoom or Teams account required.
        </p></div>
        """,
        unsafe_allow_html=True,
    )

    _default_link = st.session_state.get("vkyc_session_url", "")

    with st.form("vkyc_schedule_form", clear_on_submit=False):
        st.subheader("Session Details")
        c1, c2 = st.columns(2)
        with c1:
            customer_name = st.text_input("Customer Name *", placeholder="e.g. Rahul Sharma")
            agent_name    = st.text_input("Agent / Your Name *", placeholder="e.g. Priya Mehta")
        with c2:
            session_date = st.date_input("Session Date *", min_value=datetime.today().date())
            session_time = st.time_input("Session Time (IST) *", value=None)

        duration = st.selectbox(
            "Duration",
            options=[15, 20, 30, 45, 60],
            index=2,
            format_func=lambda x: f"{x} minutes",
        )
        meeting_link = st.text_input(
            "KYC Session URL / Meeting Link *",
            value=_default_link,
            placeholder="http://localhost:8000/kyc/...  or  https://your-domain/kyc/...",
            help=(
                "Create a KYC session in the Start KYC Session tab to get this URL. "
                "Alternatively paste a Zoom / Teams link if using external video."
            ),
        )
        notes = st.text_area(
            "Additional Notes (optional)",
            placeholder="e.g. Please keep your Aadhaar card and PAN card handy.",
            height=70,
        )
        submitted = st.form_submit_button(
            "Generate Calendar Invite", type="primary", use_container_width=True
        )

    if submitted:
        errors = []
        if not customer_name.strip():  errors.append("Customer Name is required.")
        if not agent_name.strip():     errors.append("Agent Name is required.")
        if not meeting_link.strip():   errors.append("Meeting / KYC URL is required.")
        if session_time is None:       errors.append("Session Time is required.")
        for e in errors:
            st.error(e)
        if errors:
            return

        ist_offset   = timezone(timedelta(hours=5, minutes=30))
        start_dt     = datetime(
            session_date.year, session_date.month, session_date.day,
            session_time.hour, session_time.minute, tzinfo=ist_offset,
        )
        start_dt_utc = start_dt.astimezone(timezone.utc)

        description = textwrap.dedent(f"""\
            Video KYC Session -- BaseTruth AI Identity Verification

            Customer : {customer_name.strip()}
            Agent    : {agent_name.strip()}
            Join URL : {meeting_link.strip()}

            What to prepare:
            - Original Aadhaar Card (physical or digital)
            - Original PAN Card
            - Good lighting and a stable internet connection
            - A device with a front-facing camera and a modern browser (Chrome, Safari, Edge)

            {('Notes: ' + notes.strip()) if notes.strip() else ''}

            Click the Join URL at the scheduled time to begin the verification.
            The AI-powered check takes 30-60 seconds in your browser -- no app needed.
        """).strip()

        ics_bytes = _make_ics(
            customer_name.strip(), agent_name.strip(), meeting_link.strip(),
            start_dt_utc, duration, description,
        )
        st.success(
            f"Calendar invite ready for {customer_name.strip()} -- "
            f"{session_date.strftime('%d %b %Y')} at "
            f"{session_time.strftime('%I:%M %p')} IST ({duration} min)"
        )

        col_dl, col_info = st.columns([1, 1])
        with col_dl:
            st.download_button(
                "Download .ics (Calendar Invite)",
                data=ics_bytes,
                file_name=f"vkyc_{customer_name.strip().replace(' ', '_')}.ics",
                mime="text/calendar",
                use_container_width=True,
            )
        with col_info:
            st.info("Forward the .ics to the customer via email or WhatsApp.", icon="info")

        with st.expander("Email invite text (copy & paste)", expanded=False):
            email_body = textwrap.dedent(f"""\
                Subject: Video KYC -- {session_date.strftime('%d %b %Y')} at {session_time.strftime('%I:%M %p')} IST

                Dear {customer_name.strip()},

                Your Video KYC session has been scheduled:

                Date & Time : {session_date.strftime('%d %B %Y')} at {session_time.strftime('%I:%M %p')} IST
                Duration    : {duration} minutes
                Join Link   : {meeting_link.strip()}

                What to keep ready:
                  - Original Aadhaar Card (physical or digital)
                  - Original PAN Card
                  - Good lighting and a stable internet connection
                  - A laptop or mobile with a front camera and Chrome / Safari / Edge

                {('Notes: ' + notes.strip()) if notes.strip() else ''}

                At the scheduled time, simply click the Join Link above.
                The verification runs entirely in your browser -- no app download needed.

                Regards,
                {agent_name.strip()}
            """).strip()
            st.code(email_body, language="text")


# ===========================================================================
# Tab 3 -- In-Person Webcam Verify
# ===========================================================================

def _tab_conduct() -> None:
    with st.expander("How it works", expanded=False):
        st.markdown(
            "For **face-to-face** or on-site KYC, use this tab.\n\n"
            "1. Upload a reference ID document to extract the face embedding.\n"
            "2. Use the camera input to capture a live photo.\n"
            "3. The system runs RetinaFace + ArcFace to compare the live face with the ID.\n\n"
            "For the liveness heuristic, slightly turn your head left or right before capturing."
        )

    forced_ref, extra_identity = None, None
    if _DB_IMPORTS_OK and db_available():
        forced_ref, extra_identity = _render_entity_link_widget("vkyc_conduct", mandatory=True)
        st.divider()

    st.subheader("1  Reference ID")
    doc_file = st.file_uploader(
        "Upload ID Document",
        type=["jpg", "jpeg", "png", "webp"],
        key="vk_doc",
    )
    if doc_file:
        import cv2  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
        from basetruth.vision.face import get_face_analyzer  # noqa: PLC0415

        face_app = get_face_analyzer()
        nparr    = np.frombuffer(doc_file.getvalue(), np.uint8)
        img      = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        try:
            faces = face_app.get(img)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Face detection failed: {exc}")
            faces = []

        if faces:
            face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            st.session_state["vkyc_ref_emb"] = face.normed_embedding
            st.success("Reference face extracted.")
        else:
            st.error("No face detected in the uploaded document.")
            st.session_state.pop("vkyc_ref_emb", None)

    st.divider()
    st.subheader("2  Live Capture")

    reference_emb = st.session_state.get("vkyc_ref_emb")
    if reference_emb is None:
        st.warning("Upload a Reference ID Document above first.")
        return

    st.info("For liveness detection, slightly turn your head before capturing.", icon="info")
    camera_photo = st.camera_input("Capture live photo", key="vkyc_camera")

    if camera_photo is not None:
        if _DB_IMPORTS_OK and db_available() and not forced_ref and not extra_identity:
            st.warning("Please link an entity (mandatory).")
            st.stop()

        import cv2  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        nparr    = np.frombuffer(camera_photo.getvalue(), np.uint8)
        live_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if live_img is None:
            st.error("Failed to decode the captured image.")
            return

        with st.spinner("Running RetinaFace + ArcFace..."):
            from basetruth.vision.face import get_face_analyzer  # noqa: PLC0415
            face_app = get_face_analyzer()
            try:
                faces = face_app.get(live_img)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Face detection failed: {exc}")
                return

        if not faces:
            st.error("No face detected. Please try again with better lighting.")
            return

        primary = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        box     = primary.bbox.astype(int)

        liveness_state   = "Center"
        head_turn_passed = False
        if primary.kps is not None:
            left_eye_x  = primary.kps[0][0]
            right_eye_x = primary.kps[1][0]
            nose_x      = primary.kps[2][0]
            dist_l = abs(nose_x - left_eye_x)
            dist_r = max(abs(right_eye_x - nose_x), 1.0)
            ratio  = dist_l / dist_r
            if ratio > 1.6:
                liveness_state, head_turn_passed = "Turned Right", True
            elif ratio < 0.6:
                liveness_state, head_turn_passed = "Turned Left", True

        emb      = primary.normed_embedding
        sim      = float(np.dot(emb, reference_emb))
        score    = min(max((sim - (-0.5)) / (1.0 - (-0.5)) * 100, 0), 100)
        is_match = sim >= 0.40

        color  = (0, 255, 0) if is_match else (0, 0, 255)
        label  = f"VERIFIED: {score:.1f}%" if is_match else f"MISMATCH: {score:.1f}%"
        cv2.rectangle(live_img, (box[0], box[1]), (box[2], box[3]), color, 3)
        cv2.putText(live_img, label,
                    (box[0], box[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        live_txt = f"Liveness: {liveness_state} {'(PASS)' if head_turn_passed else ''}"
        cv2.putText(live_img, live_txt,
                    (box[0], box[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        if primary.kps is not None:
            for p in primary.kps.astype(int):
                cv2.circle(live_img, (p[0], p[1]), 3, (255, 0, 0), cv2.FILLED)

        annotated_rgb = cv2.cvtColor(live_img, cv2.COLOR_BGR2RGB)
        st.image(annotated_rgb, caption="Annotated Result", use_container_width=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            if is_match:
                st.success(f"Identity Match\nScore: {score:.1f}%")
            else:
                st.error(f"Identity Mismatch\nScore: {score:.1f}%")
        with col2:
            if head_turn_passed:
                st.success(f"Liveness Passed\nHead: {liveness_state}")
            else:
                st.warning(f"Liveness Inconclusive\nHead: {liveness_state}\nTurn head and retake.")
        with col3:
            overall = "PASS" if (is_match and head_turn_passed) else "FAIL"
            if overall == "PASS":
                st.success("KYC Verdict: PASS")
            else:
                st.error("KYC Verdict: FAIL")
            st.caption(f"Cosine similarity: {sim:.4f}")

        if _DB_IMPORTS_OK and db_available():
            vkyc_result: Dict[str, Any] = {
                "is_match":          bool(is_match),
                "confidence":        float(sim),
                "cosine_similarity": float(sim),
                "display_score":     float(score),
                "threshold":         0.40,
                "liveness_state":    liveness_state,
                "liveness_passed":   bool(head_turn_passed),
                "match":             bool(is_match),
            }
            _ref_for_pdf  = forced_ref or ""
            _name_for_pdf = ""
            if extra_identity:
                _name_for_pdf = (
                    f"{extra_identity.get('first_name', '')} "
                    f"{extra_identity.get('last_name', '')}".strip()
                )
            vkyc_pdf: Optional[bytes] = None
            try:
                from basetruth.reporting.pdf import render_identity_check_pdf  # noqa: PLC0415
                vkyc_pdf = render_identity_check_pdf(
                    check_type="video_kyc",
                    result=vkyc_result,
                    entity_ref=_ref_for_pdf,
                    entity_name=_name_for_pdf,
                    doc_filename=doc_file.name if doc_file else "",
                )
            except Exception:  # noqa: BLE001
                pass

            vkyc_saved = save_identity_check(
                check_type="video_kyc",
                result=vkyc_result,
                forced_entity_ref=forced_ref,
                extra_identity=extra_identity,
                doc_filename=doc_file.name if doc_file else "",
                pdf_bytes=vkyc_pdf,
            )
            if vkyc_saved:
                st.info(
                    f"Video KYC result saved "
                    f"(ID: {vkyc_saved['id']}, "
                    f"Entity: {vkyc_saved.get('entity_ref', 'unlinked')})"
                )
            if vkyc_pdf:
                st.download_button(
                    "Download Video KYC Report (PDF)",
                    data=vkyc_pdf,
                    file_name=f"video_kyc_{st.session_state.get('vkyc_entity_ref', 'report')}.pdf",
                    mime="application/pdf",
                    key="vkyc_pdf_dl_conduct",
                )


# ===========================================================================
# Main page entry point
# ===========================================================================

def _page_video_kyc() -> None:
    # Ensure the local API server is running before any tab renders.
    # This is a no-op when BT_API_INTERNAL_URL is set (Docker mode).
    if not _ensure_local_api():
        st.error(
            "Could not start the local API server on port 8000. "
            "Run `uvicorn basetruth.api:app --port 8000` in a separate terminal "
            "and reload this page."
        )

    st.markdown(_page_title("Video KYC", "Video KYC"), unsafe_allow_html=True)
    st.caption(
        "AI-powered identity verification -- create sessions, schedule appointments, "
        "and conduct in-person checks."
    )

    tab_start, tab_schedule, tab_conduct = st.tabs([
        "Start KYC Session",
        "Schedule Appointment",
        "In-Person Verify",
    ])

    with tab_start:
        _tab_start_session()

    with tab_schedule:
        _tab_schedule()

    with tab_conduct:
        _tab_conduct()
