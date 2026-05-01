"""Video KYC page — unified session setup, scheduling, and status monitoring.

Architecture overview:
  Tab 1  Session Setup & Schedule  — Fill in customer details, optionally upload
                                     a reference ID for face-match, pick an
                                     appointment date/time, then click one button
                                     to create the KYC session AND generate the
                                     calendar invite + email invite in one step.

  Tab 2  Session Status            — Poll the active session for live status;
                                     save the completed result to the database
                                     and download the PDF report.

Customer-side flow (what the customer does after opening the link):
  1. Upload ID — Aadhaar front or PAN card
  2. Upload address proof — Aadhaar back or Passport back
  3. Share live location — captures current GPS coordinates
  4. Liveness check — auto-captures frames and compares with ID photo;
     live location is also matched against the address on the proof
     document (accepted within ~500 m).

Note: In-Person Verify has been removed — use the Identity Verification
screen for face-to-face checks.
"""
from __future__ import annotations

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

from basetruth.integrations.email_invite import KYCEmailSender
from basetruth.ui.components import (
    _DB_IMPORTS_OK,
    _db_available_cached,
    _page_title,
    _render_entity_link_widget,
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
        resp = requests.post(f"{_API_INTERNAL}{path}", json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:  # noqa: BLE001
        st.error(f"API error: {exc}")
        return None


def _api_get(path: str) -> Optional[Dict]:
    try:
        resp = requests.get(f"{_API_INTERNAL}{path}", timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:  # noqa: BLE001
        st.error(f"API error: {exc}")
        return None


def _api_get_raw(path: str) -> Optional[bytes]:
    """GET raw bytes from an API endpoint (used for image downloads).
    Returns None silently when the endpoint is unavailable — callers
    treat a missing best-frame as a non-critical condition.
    """
    try:
        resp = requests.get(f"{_API_INTERNAL}{path}", timeout=5)
        if resp.status_code == 200:
            return resp.content
    except Exception:  # noqa: BLE001
        pass
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
# Section 1 — Session Setup & Schedule  (unified operator form)
# ===========================================================================

def _section_setup_and_schedule() -> None:
    """Unified operator form: fill in customer details, optionally upload a
    reference ID for face-match, pick an appointment time, then create the
    KYC session.  The session URL, calendar invite (.ics), and email invite
    are all generated in one step so the operator can share everything from
    a single screen.
    """
    # ── If a session already exists show a reset prompt instead of the form ──
    if st.session_state.get("vkyc_session_created"):
        # The form is hidden while a session is active.
        # The share panel (rendered below) displays the active session details.
        return

    # ── Entity link (optional) ────────────────────────────────────────────
    forced_ref, extra_identity = None, None
    if _DB_IMPORTS_OK and _db_available_cached():
        forced_ref, extra_identity = _render_entity_link_widget("vkyc_start", mandatory=False)
        st.divider()

    # ── Step 1: Customer & session fields ────────────────────────────────
    st.subheader("1  Session Details")
    c1, c2 = st.columns(2)
    with c1:
        customer_name = st.text_input(
            "Customer name *", placeholder="e.g. Rahul Sharma", key="vkyc_cust_name"
        )
        customer_email = st.text_input(
            "Customer email (for invite)", placeholder="e.g. rahul@example.com",
            key="vkyc_cust_email",
        )
    with c2:
        entity_ref_input = st.text_input(
            "Entity / Case ref", placeholder="e.g. BT-000001", key="vkyc_entity_ref"
        )
        agent_name = st.text_input(
            "Your name (agent)", placeholder="e.g. Priya Mehta", key="vkyc_agent_name_input"
        )

    with st.expander("Challenge selection (optional)", expanded=False):
        st.caption(
            "🔍 **Look straight at the camera** is always included as the mandatory "
            "first challenge — this captures a clean frontal selfie for face comparison. "
            "Select additional challenges below (leave empty for 2 random)."
        )
        ALL_CH = ["blink", "turn_left", "turn_right", "nod"]
        CH_LABELS = {
            "blink":      "Close eyes",
            "turn_left":  "Turn head left",
            "turn_right": "Turn head right",
            "nod":        "Nod head",
        }
        selected = st.multiselect(
            "Pick 1-4 additional challenges (leave empty for 2 random)",
            options=ALL_CH,
            format_func=lambda c: CH_LABELS.get(c, c),
            key="vkyc_challenges",
        )
    challenges = selected or []

    # ── Step 2: Appointment time ─────────────────────────────────────────
    st.divider()
    st.subheader("2  Appointment Time")
    a1, a2, a3 = st.columns(3)
    with a1:
        session_date = st.date_input(
            "Date *", min_value=datetime.today().date(), key="vkyc_date"
        )
    with a2:
        session_time = st.time_input("Time (IST) *", value=None, key="vkyc_time")
    with a3:
        duration = st.selectbox(
            "Duration", options=[15, 20, 30, 45, 60], index=2,
            format_func=lambda x: f"{x} min", key="vkyc_duration",
        )

    # ── Step 3: Reference ID upload (optional — enables face-match) ──────
    st.divider()
    st.subheader("3  Reference ID  (optional)")
    st.caption(
        "Upload the customer's Aadhaar or PAN card to enable face-match during the session. "
        "Leave empty for liveness-only verification."
    )
    doc_file = st.file_uploader(
        "Upload ID document (Aadhaar front or PAN card with photo)",
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
            import base64 as _b64  # noqa: PLC0415
            st.session_state["vkyc_ref_emb_b64"] = _b64.b64encode(
                emb.astype("float32").tobytes()
            ).decode()
            st.session_state["vkyc_doc_bytes"] = doc_file.getvalue()
            st.success("✅ Reference face extracted — face-match will run during the session.")
        else:
            st.error("No face found in the uploaded document. Try a clearer photo.")
            st.session_state.pop("vkyc_ref_emb_b64", None)
            st.session_state.pop("vkyc_doc_bytes", None)
    elif not st.session_state.get("vkyc_ref_emb_b64"):
        st.info(
            "No reference ID uploaded — liveness-only mode. "
            "Upload an ID above to also verify face identity.",
            icon="ℹ️",
        )

    # ── Step 4: Create session ────────────────────────────────────────────
    st.divider()
    st.subheader("4  Create and Share")

    if st.button(
        "📅 Schedule Appointment & Create Session",
        type="primary",
        use_container_width=True,
    ):
        # Validate required fields before calling the API
        errors = []
        if not customer_name.strip():
            errors.append("Customer name is required.")
        if session_time is None:
            errors.append("Appointment time is required.")
        for e in errors:
            st.error(e)
        if errors:
            return

        # Create the KYC session via the FastAPI backend
        ref_b64 = st.session_state.get("vkyc_ref_emb_b64")
        payload: Dict[str, Any] = {
            "customer_name":           customer_name.strip(),
            "entity_ref":              entity_ref_input.strip() or (forced_ref or ""),
            "challenges":              challenges,
            "reference_embedding_b64": ref_b64,
        }
        resp = _api_post("/kyc/sessions", payload)
        if not resp:
            return  # _api_post already showed the error banner

        sid         = resp["session_id"]
        session_url = f"{_API_EXTERNAL}/kyc/{sid}"

        # Build the .ics calendar invite bytes
        ist_offset   = timezone(timedelta(hours=5, minutes=30))
        start_dt     = datetime(
            session_date.year, session_date.month, session_date.day,
            session_time.hour, session_time.minute, tzinfo=ist_offset,
        )
        start_dt_utc = start_dt.astimezone(timezone.utc)
        description  = (
            f"Video KYC Session — BaseTruth AI Identity Verification\n\n"
            f"Customer : {customer_name.strip()}\n"
            f"Join URL : {session_url}\n\n"
            "What the customer needs to do when they open the link:\n"
            "  1. Upload Aadhaar (front) or PAN card\n"
            "  2. Upload address proof (Aadhaar back or Passport back)\n"
            "  3. Allow location access for address verification\n"
            "  4. Complete the liveness challenge\n\n"
            "What to have ready:\n"
            "  - Aadhaar card and/or PAN card\n"
            "  - Good lighting and a stable internet connection\n"
            "  - A device with a front-facing camera and a modern browser"
        )
        ics_bytes = _make_ics(
            customer_name.strip(),
            agent_name.strip() or "BaseTruth Agent",
            session_url,
            start_dt_utc,
            duration,
            description,
        )

        # Persist all session data so both tabs can read it
        st.session_state.update({
            "vkyc_active_sid":       sid,
            "vkyc_session_url":      session_url,
            "vkyc_session_created":  True,
            "vkyc_doc_filename":     doc_file.name if doc_file else "",
            "vkyc_forced_ref":       forced_ref,
            "vkyc_extra_identity":   extra_identity,
            "vkyc_saved_remote":     False,
            "vkyc_ics_bytes":        ics_bytes,
            "vkyc_customer_name":    customer_name.strip(),
            "vkyc_customer_email":   customer_email.strip(),
            # Store under vkyc_agent_name (different from widget key vkyc_agent_name_input)
            "vkyc_agent_name":       agent_name.strip() or "BaseTruth Agent",
            "vkyc_appt_date_str":    session_date.strftime("%d %b %Y"),
            "vkyc_appt_time_str":    session_time.strftime("%I:%M %p"),
            "vkyc_duration_min":     duration,
            "vkyc_challenges_used":  challenges,
        })
        st.session_state.pop("vkyc_saved_remote_ref", None)
        st.rerun()


def _section_share_panel() -> None:
    """Shown after the session is created.  Displays the session URL, the
    calendar invite download, and a ready-to-send email template so the
    operator can share everything with the customer in one step.
    """
    if not st.session_state.get("vkyc_session_created"):
        return

    session_url  = st.session_state.get("vkyc_session_url", "")
    ics_bytes    = st.session_state.get("vkyc_ics_bytes")
    cust_name    = st.session_state.get("vkyc_customer_name", "Customer")
    cust_email   = st.session_state.get("vkyc_customer_email", "")
    agent_name   = st.session_state.get("vkyc_agent_name", "")
    date_str     = st.session_state.get("vkyc_appt_date_str", "")
    time_str     = st.session_state.get("vkyc_appt_time_str", "")
    duration_min = st.session_state.get("vkyc_duration_min", 30)

    st.divider()
    st.subheader("✅ Session Created — Share with Customer")

    # Session URL display
    st.markdown("**Session Link**")
    st.code(session_url, language="text")
    st.markdown(
        f'<a href="{session_url}" target="_blank" '
        f'style="display:inline-block;padding:.45rem .9rem;background:#4f46e5;color:#fff;'
        f'border-radius:8px;text-decoration:none;font-size:.85rem;font-weight:600">'
        f'🔗 Open KYC Page</a>',
        unsafe_allow_html=True,
    )
    st.caption(
        "The customer opens this link on their phone or laptop. They upload their ID "
        "and address proof, share their live location, then complete the liveness "
        "challenge — no app download needed."
    )

    st.divider()

    # Email body template (used by both the mailto link and the copy-paste expander)
    email_body = (
        f"Dear {cust_name},\n\n"
        f"Your Video KYC session has been scheduled:\n\n"
        f"  Date & Time : {date_str} at {time_str} IST\n"
        f"  Duration    : {duration_min} minutes\n"
        f"  Join Link   : {session_url}\n\n"
        "When you open the link you will need to:\n"
        "  1. Upload your Aadhaar card (front) or PAN card\n"
        "  2. Upload your address proof (Aadhaar back or Passport back)\n"
        "  3. Allow location access so we can verify your current address\n"
        "  4. Complete the liveness check (follow the on-screen prompts)\n\n"
        "The entire process takes under 2 minutes in your browser.\n\n"
        f"Regards,\n{agent_name}"
    )

    # Calendar invite download + mailto: link
    col_ics, col_mail = st.columns(2)
    with col_ics:
        if ics_bytes:
            safe_name = cust_name.replace(" ", "_")
            st.download_button(
                "📅 Download Calendar Invite (.ics)",
                data=ics_bytes,
                file_name=f"vkyc_{safe_name}.ics",
                mime="text/calendar",
                use_container_width=True,
            )
            st.caption("Attach this .ics file when emailing the customer.")
    with col_mail:
        import urllib.parse  # noqa: PLC0415
        # Build a mailto: link so the operator can open their email client pre-filled.
        # This avoids needing SMTP credentials in BaseTruth.
        subject = f"Your Video KYC Session — {date_str} at {time_str} IST"
        mailto = (
            f"mailto:{cust_email}"
            f"?subject={urllib.parse.quote(subject)}"
            f"&body={urllib.parse.quote(email_body)}"
        )
        st.markdown(
            f'<a href="{mailto}" '
            f'style="display:inline-block;width:100%;padding:.45rem .9rem;'
            f'background:#0ea5e9;color:#fff;border-radius:8px;text-decoration:none;'
            f'font-size:.85rem;font-weight:600;text-align:center">'
            f'📧 Open Email Client</a>',
            unsafe_allow_html=True,
        )
        if not cust_email:
            st.caption("Add customer email in Step 1 to pre-fill the address.")
        else:
            st.caption(f"Opens your email client addressed to {cust_email}.")

    with st.expander("📋 Email invite text (copy & paste)", expanded=False):
        st.code(email_body, language="text")

    # ── Direct email send (only shown when SMTP is configured) ───────────
    # Instantiate the sender once to check if SMTP env vars are set.
    # If not configured, we show an info note instead of a broken button.
    _sender = KYCEmailSender()
    if _sender.is_configured():
        st.divider()
        st.markdown("**Send invite directly to customer**")
        st.caption(
            "This will send the email with the session link and calendar invite "
            "(.ics attachment) directly to the customer's inbox."
        )
        
        # If the email was missed in Step 1 (e.g. due to UI focus loss), allow entering it here.
        if not cust_email:
            cust_email = st.text_input("Customer email", placeholder="e.g. rahul@example.com", key="vkyc_fallback_email")

        # Track whether the email has already been sent this session so the
        # button changes to a success state and the operator can re-send if needed.
        if st.session_state.get("vkyc_email_sent"):
            st.success(
                f"Invite already sent to **{cust_email or 'customer'}**. "
                "Click below to send again if needed.",
                icon="✅",
            )
        if cust_email:
            if st.button(
                "📧 Send Email Invite",
                key="vkyc_send_email_btn",
                use_container_width=True,
                type="primary",
            ):
                with st.spinner(f"Sending invite to {cust_email}…"):
                    ok, err = _sender.send_kyc_invite(
                        to_email=cust_email,
                        customer_name=cust_name,
                        agent_name=agent_name,
                        session_url=session_url,
                        date_str=date_str,
                        time_str=time_str,
                        duration_min=duration_min,
                        ics_bytes=ics_bytes,
                    )
                if ok:
                    st.session_state["vkyc_email_sent"] = True
                    # Also save it so it persists if the app reruns
                    st.session_state["vkyc_customer_email"] = cust_email
                    st.success(
                        f"Email invite sent to **{cust_email}** with the session link "
                        "and calendar attachment.",
                        icon="✅",
                    )
                    st.rerun()
                else:
                    st.error(f"Could not send email: {err}")
    else:
        with st.expander("ℹ️ Enable direct email sending", expanded=False):
            st.markdown(
                "Set the following environment variables to send invites directly "
                "from BaseTruth without opening your email client:\n\n"
                "```\n"
                "BT_SMTP_HOST=smtp.gmail.com\n"
                "BT_SMTP_PORT=587\n"
                "BT_SMTP_USER=you@gmail.com\n"
                "BT_SMTP_PASSWORD=<app_password>\n"
                "BT_EMAIL_FROM=BaseTruth KYC <you@gmail.com>\n"
                "```\n\n"
                "For Gmail, generate an App Password under **Security → App Passwords**."
            )

    st.divider()
    if st.button("🔄 Start a New Session", use_container_width=True, key="vkyc_new_session_btn"):
        _clear_session_state()
        st.rerun()


# ===========================================================================
# Session-state reset helper
# ===========================================================================

def _clear_session_state() -> None:
    """Reset all vkyc_* session-state keys to allow starting a fresh session."""
    for k in [
        "vkyc_active_sid", "vkyc_session_url", "vkyc_session_created",
        "vkyc_ref_emb_b64", "vkyc_doc_filename", "vkyc_doc_bytes",
        "vkyc_saved_remote", "vkyc_saved_remote_ref",
        "vkyc_ics_bytes", "vkyc_customer_name", "vkyc_customer_email",
        "vkyc_agent_name", "vkyc_agent_name_input",
        "vkyc_appt_date_str", "vkyc_appt_time_str",
        "vkyc_duration_min", "vkyc_forced_ref", "vkyc_extra_identity",
        "vkyc_challenges_used", "vkyc_email_sent",
        # Document upload state (new enriched KYC flow)
        "vkyc_aadhaar_bytes", "vkyc_aadhaar_filename", "vkyc_aadhaar_qr",
        "vkyc_pan_bytes", "vkyc_pan_filename", "vkyc_pan_data", "vkyc_pan_sig_bytes",
        "vkyc_addr_proof_bytes", "vkyc_addr_proof_filename", "vkyc_addr_dtls",
        "vkyc_live_selfie_bytes", "vkyc_face_match_result",
        # Applicant details form (auto-filled from docs)
        "vkyc_first_name", "vkyc_last_name", "vkyc_pan_number",
        "vkyc_aadhaar_uid", "vkyc_phone", "vkyc_email_input", "_vkyc_auto_key",
    ]:
        st.session_state.pop(k, None)


# ===========================================================================
# KYC persistence helper (shared by the status section below)
# ===========================================================================


def _build_kyc_save_artifacts(
    result: Dict,
    session_id: str,
    status_resp: Dict,
    doc_filename: str,
    forced_ref: Optional[str],
    extra_identity: Optional[Dict],
    cosine_sim: float,
    aadhar_dtls: Optional[Dict] = None,
    pan_dtls: Optional[Dict] = None,
    address_dtls: Optional[Dict] = None,
) -> tuple[Dict[str, Any], str, Optional[bytes]]:
    """Build the Video KYC persistence payload and optional PDF report.

    Merges liveness result with operator-supplied document data (Aadhaar QR,
    PAN extraction, address proof) into a single result dict, then renders
    the PDF report.  The caller passes the face-match cosine_sim when the
    operator has clicked 'Match Face'; otherwise the session's liveness score
    is used.
    """
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
        # Enriched document payloads from operator uploads
        "aadhar_dtls":       aadhar_dtls or {},
        "pan_dtls":          pan_dtls or {},
        "address_dtls":      address_dtls or {},
        # Address comparison from the session
        "isAddressMatch":    status_resp.get("isAddressMatch", "skipped"),
        "kyc_comments":      status_resp.get("kyc_comments", ""),
        "current_location":          status_resp.get("current_location", ""),
        "address_distance_meters":  status_resp.get("address_distance_meters"),
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

    return vkyc_result, entity_ref, vkyc_pdf


# ===========================================================================
# Section 2 — Session Status  (document review + live polling + save)
# ===========================================================================

def _section_session_status() -> None:  # noqa: PLR0912,PLR0915
    """Tab 2: full document review + liveness monitoring surface.

    Mirrors the Identity Verification screen experience:
    - Operator uploads Aadhaar (QR decoded), PAN (extracted + signature),
      and address proof (address extracted) while the session is running or
      after it completes.
    - Live selfie is fetched from the session once liveness challenges finish.
    - Address comparison from the customer's GPS vs. the proof document is shown.
    - Operator fills in / confirms applicant details (auto-filled from docs).
    - Match Face runs ArcFace comparison between the Aadhaar card and the
      live selfie.
    - Save to Database persists all enriched data in one upsert.
    """
    # Import document extraction helpers from the Identity Verification page
    # (they live there to avoid duplicating the logic).
    from basetruth.ui.pages.identity import (  # noqa: PLC0415
        _parse_aadhaar_qr,
        _extract_pan_info,
        _crop_pan_signature,
        render_aadhaar_qr_fields,
        render_pan_fields,
    )

    if not st.session_state.get("vkyc_session_created"):
        st.info(
            "No active session yet. Go to **Session Setup & Schedule** to create one.",
            icon="ℹ️",
        )
        return

    sid        = st.session_state.get("vkyc_active_sid", "")
    challenges = st.session_state.get("vkyc_challenges_used", [])

    status_resp = _api_get(f"/kyc/sessions/{sid}")
    if not status_resp:
        return

    if status_resp.get("aadhaar_qr") and "vkyc_aadhaar_qr" not in st.session_state:
        st.session_state["vkyc_aadhaar_qr"] = status_resp["aadhaar_qr"]
    if status_resp.get("pan_data") and "vkyc_pan_data" not in st.session_state:
        st.session_state["vkyc_pan_data"] = status_resp["pan_data"]
    if status_resp.get("pan_sig_b64") and "vkyc_pan_sig_bytes" not in st.session_state:
        import base64
        st.session_state["vkyc_pan_sig_bytes"] = base64.b64decode(status_resp["pan_sig_b64"])
    if status_resp.get("address_dtls") and "vkyc_addr_dtls" not in st.session_state:
        st.session_state["vkyc_addr_dtls"] = status_resp["address_dtls"]
    
    if status_resp.get("aadhaar_b64") and "vkyc_aadhaar_bytes" not in st.session_state:
        import base64
        st.session_state["vkyc_aadhaar_bytes"] = base64.b64decode(status_resp["aadhaar_b64"])
    if status_resp.get("pan_b64") and "vkyc_pan_bytes" not in st.session_state:
        import base64
        st.session_state["vkyc_pan_bytes"] = base64.b64decode(status_resp["pan_b64"])
    if status_resp.get("address_proof_b64") and "vkyc_addr_proof_bytes" not in st.session_state:
        import base64
        st.session_state["vkyc_addr_proof_bytes"] = base64.b64decode(status_resp["address_proof_b64"])

    status   = status_resp.get("status", "unknown")
    ch_done  = status_resp.get("challenges_completed", 0)
    ch_total = status_resp.get("total_challenges", len(challenges) or 2)
    result   = status_resp.get("result")

    # Set this flag early so we can fire st.rerun() after ALL widgets have
    # rendered (Streamlit requires all widgets to be drawn before rerun).
    # When the session is waiting/active, we poll every 2 s so document
    # uploads from the customer's liveness page appear automatically.
    _should_auto_refresh = status in ("waiting", "active")

    # ── Session Monitor header ────────────────────────────────────────────
    status_labels = {
        "waiting":   "⏳ Waiting for customer",
        "active":    "🔵 Session in progress",
        "completed": "✅ Completed",
        "failed":    "❌ Failed",
        "expired":   "⚫ Expired",
    }
    col_s, col_p = st.columns([3, 2])
    with col_s:
        st.metric("Session status", status_labels.get(status, status))
    with col_p:
        st.metric("Challenges", f"{ch_done} / {ch_total} done")

    if status in ("waiting", "active"):
        # Show progress bar while the customer completes the liveness challenge.
        # The auto-refresh fires at the bottom of this function so Aadhaar /
        # selfie uploads from the customer's liveness page appear here without
        # the operator needing to manually reload the page.
        st.progress(ch_done / max(ch_total, 1), text="Liveness challenges progress")

    if status in ("failed", "expired"):
        msg = "Session expired — please create a new one." if status == "expired" \
              else "Verification failed. Please retry."
        st.error(msg)
        if st.button("🔄 Start New Session", use_container_width=True, key="vkyc_restart_failed"):
            _clear_session_state()
            st.rerun()
        return

    # ── Liveness result banner ────────────────────────────────────────────
    if result:
        passed     = result.get("passed", False)
        disp_score = result.get("display_score", result.get("match_score", 0) * 100)
        if passed:
            st.success(f"Liveness Passed — Score: {disp_score:.1f}%")
        else:
            st.warning(f"Liveness completed with score: {disp_score:.1f}%")
            st.caption(result.get("message", ""))

    st.divider()

    # Prefetch the best liveness frame early so it appears in the selfie column on
    # the first render after the customer completes the challenge, without a reload.
    if not st.session_state.get("vkyc_live_selfie_bytes"):
        frame_bytes = _api_get_raw(f"/kyc/sessions/{sid}/best-frame")
        if frame_bytes:
            st.session_state["vkyc_live_selfie_bytes"] = frame_bytes

    # ── Identity Documents — 3 side-by-side sections: Aadhaar | PAN | Selfie ──
    # All three come from the customer's session upload — no operator re-upload needed.
    st.subheader("📄 Identity Documents")
    st.caption(
        "Aadhaar, PAN, and selfie come from what the customer submitted through the session link."
    )

    col_a, col_p_doc, col_s = st.columns(3)

    # ── Column 1: Aadhaar Card (from customer's session upload — no re-upload needed) ──
    with col_a:
        st.markdown("**📄 Aadhaar Card**")
        # Decode Aadhaar QR if we have the image bytes but not the QR result yet.
        # The bytes are synced from the customer's upload via the session API above.
        if "vkyc_aadhaar_qr" not in st.session_state and st.session_state.get("vkyc_aadhaar_bytes"):
            with st.spinner("Decoding Aadhaar QR..."):
                qr = _parse_aadhaar_qr(st.session_state["vkyc_aadhaar_bytes"])
                st.session_state["vkyc_aadhaar_qr"] = qr

        if st.session_state.get("vkyc_aadhaar_bytes"):
            st.image(
                st.session_state["vkyc_aadhaar_bytes"],
                caption="Aadhaar Card",
                use_container_width=True,
            )
            qr = st.session_state.get("vkyc_aadhaar_qr", {})
            if qr.get("qr_found"):
                if qr.get("qr_type") == "secure":
                    st.info(
                        "Secure Aadhaar QR detected (2018+). Demographic data is "
                        "cryptographically signed and cannot be displayed offline.",
                        icon="🔒",
                    )
                else:
                    # Show a success banner matching the Identity Verification screen
                    if qr.get("qr_type") == "gemma4":
                        st.success("Aadhaar details extracted via Gemma4 fallback")
                    else:
                        st.success("✅ QR decoded successfully")
                    # Use the shared utility so Video KYC and Identity Verification
                    # always show the same 9 Aadhaar fields in the same format.
                    render_aadhaar_qr_fields(qr)
            elif qr.get("qr_found") is False:
                st.warning("QR not found in the uploaded Aadhaar.", icon="⚠️")
        else:
            # Customer has not yet uploaded Aadhaar through the session link
            st.info(
                "Aadhaar not yet received. The image and QR data will appear here "
                "once the customer uploads it through the session link.",
                icon="⏳",
            )

    # ── Column 2: PAN Card (from customer's session upload — no re-upload needed) ──
    with col_p_doc:
        st.markdown("**💳 PAN Card**")
        # PAN bytes, extracted data, and signature are all synced from the
        # customer's liveness-page upload via the session API at the top of
        # this function — no operator re-upload is required.
        if st.session_state.get("vkyc_pan_bytes"):
            st.image(
                st.session_state["vkyc_pan_bytes"],
                caption="PAN Card",
                use_container_width=True,
            )
            pan_d = st.session_state.get("vkyc_pan_data", {})
            sig_b = st.session_state.get("vkyc_pan_sig_bytes")
            # render_pan_fields already shows the success/warning banner internally
            render_pan_fields(pan_d, sig_b)
        else:
            # Customer has not yet uploaded PAN through the session link
            st.info(
                "PAN not yet received. The image and extracted fields will appear "
                "here once the customer uploads it through the session link.",
                icon="⏳",
            )

    # ── Column 3: Live Selfie (captured during the customer's liveness session) ──
    with col_s:
        st.markdown("**🤳 Live Selfie**")
        live_selfie_col = st.session_state.get("vkyc_live_selfie_bytes")
        if live_selfie_col:
            st.image(
                live_selfie_col,
                caption="Best frame captured during liveness challenge",
                use_container_width=True,
            )
        else:
            st.info(
                "Live selfie not yet available. It will appear here once the "
                "customer completes the liveness challenge.",
                icon="⏳",
            )

    # ── Additional Sections: Address Proof | Current Location | Placeholder ──
    st.divider()
    col_addr, col_loc, col_ph = st.columns(3)

    # ── Column 1: Address Proof (from customer's session upload) ──
    with col_addr:
        st.markdown("**🏠 Address Proof**")
        
        # Extract address if we have bytes but no details yet
        if "vkyc_addr_dtls" not in st.session_state and st.session_state.get("vkyc_addr_proof_bytes"):
            with st.spinner("Extracting address from document..."):
                try:
                    from basetruth.ui.pages.scan import extract_document  # noqa: PLC0415
                    addr_result = extract_document(
                        st.session_state["vkyc_addr_proof_bytes"], "address_proof"
                    )
                    st.session_state["vkyc_addr_dtls"] = addr_result.get("extracted_fields", {})
                except Exception:  # noqa: BLE001
                    st.session_state["vkyc_addr_dtls"] = {}

        if st.session_state.get("vkyc_addr_proof_bytes"):
            st.image(st.session_state["vkyc_addr_proof_bytes"], caption="Address Proof (Aadhaar back / Passport)", use_container_width=True)
            
            addr_d = st.session_state.get("vkyc_addr_dtls", {})
            if addr_d:
                # Non-address fields to exclude so we don't show Aadhaar numbers,
                # names, dates, or the internal filename/raw_text keys.
                _SKIP_ADDR_KEYS = {
                    "filename", "raw_text", "document_type", "extraction_confidence",
                    "aadhar_number", "aadhaar_number", "pan_number",
                    "name", "full_name", "father_name", "care_of",
                    "dob", "date_of_birth", "year_of_birth", "gender",
                }
                # Use the pre-cleaned raw_text first (populated by the API upload
                # endpoint via _extract_address_text which already strips non-addr fields).
                address_text = addr_d.get("raw_text", "")
                if not address_text:
                    # Fall back to the dedicated address field if raw_text is absent
                    for _ak in ("full_address", "address", "complete_address"):
                        _av = addr_d.get(_ak, "")
                        if _av and len(str(_av)) > 20:
                            address_text = str(_av).strip()
                            break
                if not address_text:
                    # Last resort: join only address-component fields
                    address_text = " ".join(
                        str(v) for k, v in addr_d.items()
                        if not k.startswith("_") and k not in _SKIP_ADDR_KEYS and v
                    ).strip()
                if address_text:
                    st.info(f"Extracted address:\n{address_text}", icon="🏠")
        else:
            st.info(
                "Address Proof not yet received. The image and extracted address will appear "
                "here once the customer uploads it through the session link.",
                icon="⏳",
            )

    # ── Column 2: Current Location & Address Match ──
    with col_loc:
        st.markdown("**📍 Current Location (Live GPS)**")
        current_addr = status_resp.get("current_location", "")
        addr_match   = status_resp.get("isAddressMatch", status_resp.get("address_match_result", "skipped"))
        dist_m       = status_resp.get("address_distance_meters")

        if current_addr or addr_match not in ("skipped", ""):
            st.write(current_addr or "Not captured")

            if dist_m is not None:
                st.metric("Distance from Proof", f"{dist_m:.0f} m")

            _match_labels = {
                "match":     "✅ Address Matched",
                "mismatch":  "❌ Address Mismatch",
                "partial":   "⚠️ Partial Match",
                "skipped":   "⏭️ Not Verified",
            }
            st.info(_match_labels.get(addr_match, addr_match or "⏭️ Not Verified"), icon="📍")
        else:
            st.info(
                "Location not yet captured. It will appear here once the customer shares it.",
                icon="⏳"
            )

    # ── Column 3: Liveness Challenge Results ──
    with col_ph:
        st.markdown("**🎯 Liveness Challenge Results**")
        _challenges_list   = status_resp.get("challenges", [])
        _challenge_results = status_resp.get("challenge_results", [])
        _ch_done           = status_resp.get("challenges_completed", 0)
        _ch_labels_map = {
            "look_straight": "Look at Camera",
            "blink":         "Close Eyes",
            "turn_left":     "Turn Head Left",
            "turn_right":    "Turn Head Right",
            "nod":           "Nod Head",
        }
        # Build a set of challenge names that have been recorded as passed
        _passed_names = {r["challenge"] for r in _challenge_results if r.get("passed")}
        if _challenges_list:
            # When status is "waiting" and no challenges have been attempted yet,
            # show a clear "not started" message rather than falsely marking the
            # first challenge as "in progress" (which misleads the operator).
            if status == "waiting" and not _passed_names:
                st.info("Liveness challenge not yet started.", icon="⏳")
            else:
                for _ci, _ch in enumerate(_challenges_list):
                    _label = _ch_labels_map.get(_ch, _ch.replace("_", " ").title())
                    if _ch in _passed_names:
                        st.success(f"✅ {_label}")
                    elif _ci == _ch_done and status == "active":
                        # Only show "in progress" when session is actively running
                        st.info(f"⏳ {_label} (in progress…)")
                    else:
                        st.caption(f"○ {_label}")
        elif status in ("waiting", "active"):
            st.info("Waiting for customer to start liveness check.", icon="⏳")
        else:
            st.caption("No challenge data recorded.")

    st.divider()

    # ── Document Cross-Checks ──────────────────────────────────────────────
    # Mirror the Identity Verification screen: compare the name and DOB from
    # the Aadhaar QR against PAN card extraction so the operator can spot
    # discrepancies before saving the KYC result.
    qr_data  = st.session_state.get("vkyc_aadhaar_qr", {})
    pan_data = st.session_state.get("vkyc_pan_data", {})

    if qr_data or pan_data:
        from basetruth.analysis.identity_checks import (  # noqa: PLC0415
            compare_first_last_names,
            compare_dob_values,
        )
        st.subheader("Document Cross-Checks")
        _aadhaar_name_xc = (
            qr_data.get("name", "") if qr_data.get("qr_type") in ("xml", "gemma4") else ""
        )
        _pan_name_xc    = pan_data.get("full_name") or pan_data.get("name", "")
        _aadhaar_dob_xc = qr_data.get("dob") or qr_data.get("yob", "")
        _pan_dob_xc     = pan_data.get("date_of_birth", "")
        _name_check     = compare_first_last_names(_aadhaar_name_xc, _pan_name_xc)
        _dob_check      = compare_dob_values(_aadhaar_dob_xc, _pan_dob_xc)

        chk_cols = st.columns(2)
        with chk_cols[0]:
            if _name_check.get("passed") is True:
                st.success(
                    "**First Name & Last Name Match: PASS**  \n"
                    f"Aadhaar: *{_name_check['aadhaar_first_name']} {_name_check['aadhaar_last_name']}*  \n"
                    f"PAN: *{_name_check['pan_first_name']} {_name_check['pan_last_name']}*"
                )
            elif _name_check.get("passed") is False:
                st.error(
                    "**First Name & Last Name Match: FAIL**  \n"
                    f"{_name_check['message']}"
                )
            elif _aadhaar_name_xc or _pan_name_xc:
                st.info(_name_check.get("message", "Name comparison unavailable."))
            else:
                st.caption("Aadhaar and PAN documents are needed to compare names.")

        with chk_cols[1]:
            if _dob_check.get("passed") is True:
                st.success(
                    "**DOB Match: PASS**  \n"
                    f"{_dob_check['message']}"
                )
            elif _dob_check.get("passed") is False:
                st.error(
                    "**DOB Match: FAIL**  \n"
                    f"{_dob_check['message']}"
                )
            elif _aadhaar_dob_xc or _pan_dob_xc:
                st.info(_dob_check.get("message", "DOB comparison unavailable."))
            else:
                st.caption("Aadhaar and PAN documents are needed to compare date of birth.")

    st.divider()

    # ── Applicant Details form (auto-filled from documents) ────────────────
    st.subheader("Applicant Details")
    st.info(
        "Fields marked **auto-filled** are extracted from the documents. "
        "Please provide Phone and Email manually.",
        icon="ℹ️",
    )

    # Extract preferred name from Aadhaar QR (primary) or PAN (fallback)
    _qr_name_ap   = (
        qr_data.get("name", "") if qr_data.get("qr_type") in ("xml", "gemma4") else ""
    )
    _pan_name_ap  = (pan_data.get("full_name") or pan_data.get("name") or "").strip()
    _preferred_name_ap = _qr_name_ap or _pan_name_ap
    _name_parts_ap = _preferred_name_ap.split(maxsplit=1)
    _default_fn_ap   = _name_parts_ap[0] if _name_parts_ap else ""
    _default_ln_ap   = _name_parts_ap[1] if len(_name_parts_ap) > 1 else ""
    _default_pan_ap  = pan_data.get("pan_number", "")
    _default_aadh_ap = qr_data.get("uid", "")

    # Show father's name / DOB caption from PAN if available
    if pan_data.get("father_name") or pan_data.get("date_of_birth"):
        meta_bits = []
        if pan_data.get("father_name"):
            meta_bits.append(f"Father's name: **{pan_data['father_name']}**")
        if pan_data.get("date_of_birth"):
            meta_bits.append(f"DOB: **{pan_data['date_of_birth']}**")
        st.caption("  |  ".join(meta_bits))

    # Re-fill auto fields whenever document availability changes (e.g. customer
    # uploads Aadhaar after the operator already opened the tab).
    _auto_key_vkyc = (
        "_vkyc_auto_"
        f"{'1' if st.session_state.get('vkyc_aadhaar_bytes') else '0'}_"
        f"{'1' if st.session_state.get('vkyc_pan_bytes') else '0'}"
    )
    if st.session_state.get("_vkyc_auto_key") != _auto_key_vkyc:
        if _default_fn_ap:
            st.session_state["vkyc_first_name"] = _default_fn_ap
        if _default_ln_ap:
            st.session_state["vkyc_last_name"] = _default_ln_ap
        if _default_pan_ap:
            st.session_state["vkyc_pan_number"] = _default_pan_ap
        if _default_aadh_ap:
            st.session_state["vkyc_aadhaar_uid"] = _default_aadh_ap
        st.session_state["_vkyc_auto_key"] = _auto_key_vkyc

    mc1, mc2 = st.columns(2)
    first_name = mc1.text_input(
        "First name \u00a0\u2605 required",
        key="vkyc_first_name",
        disabled=True,
        help="Auto-filled from Aadhaar QR / PAN. Documents must be received to populate.",
    )
    last_name = mc2.text_input(
        "Last name \u00a0\u2605 required",
        key="vkyc_last_name",
        disabled=True,
        help="Auto-filled from Aadhaar QR / PAN. Documents must be received to populate.",
    )
    mc3, mc4 = st.columns(2)
    pan_number = mc3.text_input(
        "PAN number \u00a0\u2605 required",
        key="vkyc_pan_number",
        placeholder="ABCDE1234F",
        disabled=True,
        help="Auto-filled from PAN card OCR. PAN card must be received to populate.",
    )
    aadhaar_uid = mc4.text_input(
        "Aadhaar number \u00a0\u2605 required",
        key="vkyc_aadhaar_uid",
        placeholder="1234 5678 9012",
        disabled=True,
        help="Auto-filled from Aadhaar QR code. Aadhaar card must be received to populate.",
    )
    mc5, mc6 = st.columns(2)
    # Pre-populate email from the session's customer email if not already entered
    if "vkyc_email_input" not in st.session_state and st.session_state.get("vkyc_customer_email"):
        st.session_state["vkyc_email_input"] = st.session_state["vkyc_customer_email"]
    email_val = mc5.text_input(
        "Email  *(enter manually)*",
        key="vkyc_email_input",
        placeholder="applicant@email.com",
    )
    phone_val = mc6.text_input(
        "Phone  *(enter manually)*",
        key="vkyc_phone",
        placeholder="+91 98765 43210",
    )

    # Required-field caption — shows which fields will populate when documents arrive
    _required_missing = []
    if not str(st.session_state.get("vkyc_first_name", "")).strip():
        _required_missing.append("First name")
    if not str(st.session_state.get("vkyc_last_name", "")).strip():
        _required_missing.append("Last name")
    if not str(st.session_state.get("vkyc_pan_number", "")).strip():
        _required_missing.append("PAN number")
    if not str(st.session_state.get("vkyc_aadhaar_uid", "")).strip():
        _required_missing.append("Aadhaar number")
    if _required_missing:
        st.caption(
            f"⚠️ The following fields will be auto-filled when documents are received: "
            f"**{', '.join(_required_missing)}**"
        )

    # Entity link widget — reuses the same component as Identity Verification
    forced_ref, extra_identity = None, None
    if _DB_IMPORTS_OK and _db_available_cached():
        forced_ref, extra_identity = _render_entity_link_widget("vkyc_status", mandatory=False)

    # Fall back to the entity ref / identity set when the session was created
    if not forced_ref:
        forced_ref = st.session_state.get("vkyc_forced_ref")
    if not extra_identity:
        extra_identity = st.session_state.get("vkyc_extra_identity")

    # Build extra_identity from form fields if not supplied by entity link widget.
    # IMPORTANT: keys must match the Entity model columns (pan_number, aadhar_number)
    # or _find_or_create_entity will fail to match/create the entity correctly.
    if not extra_identity:
        extra_identity = {
            k: v for k, v in {
                "first_name":    first_name.strip(),
                "last_name":     last_name.strip(),
                "pan_number":    pan_number.strip(),
                "aadhar_number": aadhaar_uid.strip().replace(" ", ""),
                "phone":         phone_val.strip(),
                "email":         email_val.strip(),
            }.items() if v
        } or None

    st.divider()

    # ── Face Match ────────────────────────────────────────────────────────
    aadhaar_bytes = st.session_state.get("vkyc_aadhaar_bytes")
    live_selfie   = st.session_state.get("vkyc_live_selfie_bytes")

    if aadhaar_bytes and live_selfie:
        st.subheader("🔍 Face Match")
        st.caption(
            "Compare the face on the uploaded Aadhaar card against the live selfie "
            "captured during the liveness challenge."
        )
        if st.button(
            "Run Face Match  🔍",
            key="vkyc_match_face_btn",
            type="primary",
            use_container_width=True,
        ):
            with st.spinner("Running face detection and ArcFace matching..."):
                try:
                    from basetruth.vision.face import compare_faces  # noqa: PLC0415
                    fm_result = compare_faces(aadhaar_bytes, live_selfie)
                    st.session_state["vkyc_face_match_result"] = fm_result
                except Exception as exc:  # noqa: BLE001
                    st.session_state["vkyc_face_match_result"] = {"error": str(exc)}
            st.rerun()

        fm = st.session_state.get("vkyc_face_match_result")
        if fm:
            if fm.get("error"):
                st.error(f"Face matching failed: {fm['error']}")
            else:
                # compare_faces returns 'confidence' (cosine similarity) and 'display_score'
                fm_score  = float(fm.get("display_score", 0))
                fm_conf   = float(fm.get("confidence", 0))
                fm_thresh = float(fm.get("threshold", 0.40))
                fm_match  = bool(fm.get("match", False))

                # Show annotated face images side-by-side — matches Identity Verification layout
                r1, r2 = st.columns(2)
                with r1:
                    if fm.get("doc_annotated_rgb") is not None:
                        st.image(
                            fm["doc_annotated_rgb"],
                            caption="Face detected on Aadhaar",
                            use_container_width=True,
                        )
                with r2:
                    if fm.get("selfie_annotated_rgb") is not None:
                        st.image(
                            fm["selfie_annotated_rgb"],
                            caption="Face detected in selfie",
                            use_container_width=True,
                        )

                if fm_match:
                    st.success(
                        f"### ✅ IDENTITY MATCH — {fm_score:.1f}% confidence\n"
                        "The face on the Aadhaar card matches the live selfie."
                    )
                    st.success("**Photo Match: PASS** — Aadhaar photo and selfie match.")
                else:
                    st.error(
                        f"### 🚨 IDENTITY MISMATCH — {fm_score:.1f}% confidence\n"
                        "The faces DO NOT match. Possible fraud risk."
                    )
                    st.error("**Photo Match: FAIL** — Aadhaar photo and selfie do not match.")
                st.caption(
                    f"Cosine similarity: {fm_conf:.3f}  (threshold: {fm_thresh:.2f})"
                )

        st.divider()
    elif live_selfie and not aadhaar_bytes:
        st.info("Upload Aadhaar card above to enable face match.", icon="ℹ️")
    elif aadhaar_bytes and not live_selfie:
        st.info(
            "Live selfie not yet available. Face match will be enabled once the "
            "liveness session captures a high-confidence frame.",
            icon="ℹ️",
        )

    # ── Build save artifacts ──────────────────────────────────────────────
    if not result:
        # Session still in progress — fire a timed rerun so document uploads
        # from the customer's liveness page (Aadhaar, selfie) propagate here
        # automatically without the operator having to reload the page.
        if _should_auto_refresh:
            time.sleep(2)
            st.rerun()
        return

    # Use face match cosine similarity when available; fall back to session score
    fm           = st.session_state.get("vkyc_face_match_result", {})
    cosine_sim   = result.get("cosine_similarity", 0.0)
    if fm and not fm.get("error"):
        cosine_sim = float(fm.get("cosine_similarity", cosine_sim))

    aadhar_dtls_val = st.session_state.get("vkyc_aadhaar_qr") \
        if st.session_state.get("vkyc_aadhaar_qr", {}).get("qr_found") else None
    pan_dtls_val    = st.session_state.get("vkyc_pan_data")
    addr_dtls_val   = st.session_state.get("vkyc_addr_dtls")

    vkyc_result, entity_ref, vkyc_pdf = _build_kyc_save_artifacts(
        result       = result,
        session_id   = sid,
        status_resp  = status_resp,
        doc_filename = st.session_state.get("vkyc_doc_filename", ""),
        forced_ref   = forced_ref,
        extra_identity = extra_identity,
        cosine_sim   = cosine_sim,
        aadhar_dtls  = aadhar_dtls_val,
        pan_dtls     = pan_dtls_val,
        address_dtls = addr_dtls_val,
    )

    # ── Save to Database ──────────────────────────────────────────────────
    if _DB_IMPORTS_OK and _db_available_cached():
        if st.session_state.get("vkyc_saved_remote"):
            st.success(
                f"Saved to database — Entity: **{st.session_state.get('vkyc_saved_remote_ref') or 'unlinked'}**"
            )
        elif st.button("💾 Save to Database", key="vkyc_remote_save_btn", use_container_width=True):
            with st.spinner("Saving Video KYC result to database..."):
                saved = save_identity_check(
                    check_type="video_kyc",
                    result=vkyc_result,
                    forced_entity_ref=forced_ref,
                    extra_identity=extra_identity,
                    doc_filename=st.session_state.get("vkyc_doc_filename", ""),
                    pdf_bytes=vkyc_pdf,
                    doc_bytes=st.session_state.get("vkyc_doc_bytes"),
                    selfie_bytes=live_selfie,
                    aadhar_dtls=aadhar_dtls_val,
                    pan_dtls=pan_dtls_val,
                    aadhaar_bytes=st.session_state.get("vkyc_aadhaar_bytes"),
                    aadhaar_filename=st.session_state.get("vkyc_aadhaar_filename", ""),
                    pan_bytes=st.session_state.get("vkyc_pan_bytes"),
                    pan_signature_bytes=st.session_state.get("vkyc_pan_sig_bytes"),
                    address_proof_bytes=st.session_state.get("vkyc_addr_proof_bytes"),
                    address_proof_filename=st.session_state.get("vkyc_addr_proof_filename", ""),
                )
                if saved:
                    st.session_state["vkyc_saved_remote"]     = True
                    st.session_state["vkyc_saved_remote_ref"] = saved.get("entity_ref")
                    st.rerun()
                else:
                    st.error(
                        "Video KYC completed but could not be saved to the database. "
                        "Check the Logs screen for details."
                    )
    else:
        st.info("Database is offline — result not persisted.", icon="ℹ️")

    if vkyc_pdf and st.session_state.get("vkyc_saved_remote"):
        st.download_button(
            "⬇ Download KYC Report (PDF)",
            data=vkyc_pdf,
            file_name=f"video_kyc_{entity_ref or 'report'}.pdf",
            mime="application/pdf",
            key="vkyc_pdf_dl",
        )

    if st.button("🔄 Start New Session", use_container_width=True, key="vkyc_restart_btn"):
        _clear_session_state()
        st.rerun()

    # For completed sessions where the Aadhaar hasn't appeared yet (race
    # condition: customer uploaded just as the operator opened the tab),
    # offer a lightweight one-click refresh so documents sync without a
    # full page reload.
    if status == "completed" and not st.session_state.get("vkyc_aadhaar_bytes"):
        st.info(
            "Aadhaar not yet synced for this session. Click **Refresh Documents** "
            "to load the image the customer submitted.",
            icon="ℹ️",
        )
        if st.button("🔄 Refresh Documents", key="vkyc_refresh_docs_btn", use_container_width=True):
            st.rerun()





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

    st.markdown(_page_title("🎥", "Video KYC"), unsafe_allow_html=True)
    st.caption(
        "Create a secure remote session, schedule the appointment, and share "
        "the link with the customer — all in one step."
    )

    tab_setup, tab_status = st.tabs([
        "📅 Session Setup & Schedule",
        "📊 Session Status",
    ])

    with tab_setup:
        _section_setup_and_schedule()
        _section_share_panel()

    with tab_status:
        _section_session_status()

