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
        session_url = st.session_state.get("vkyc_session_url", "")
        st.success("Session is active — share this link with the customer:")
        st.code(session_url, language="text")
        st.markdown(
            f'<a href="{session_url}" target="_blank" '
            f'style="display:inline-block;padding:.45rem .9rem;background:#4f46e5;color:#fff;'
            f'border-radius:8px;text-decoration:none;font-weight:600">🔗 Open KYC Page</a>',
            unsafe_allow_html=True,
        )
        if st.button("🔄 Start a New Session", use_container_width=True):
            _clear_session_state()
            st.rerun()
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
                    st.success(
                        f"Email invite sent to **{cust_email}** with the session link "
                        "and calendar attachment.",
                        icon="✅",
                    )
                else:
                    st.error(f"Could not send email: {err}")
        else:
            st.info(
                "Add the customer email in Step 1 to enable direct sending.",
                icon="ℹ️",
            )
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
) -> tuple[Dict[str, Any], str, Optional[bytes]]:
    """Build the Video KYC persistence payload and optional PDF report."""
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

    return vkyc_result, entity_ref, vkyc_pdf


# ===========================================================================
# Section 2 — Session Status  (live polling + save)
# ===========================================================================

def _section_session_status() -> None:
    """Tab 2: polls the active session every 2 s while it is running, shows
    the result when completed, and provides Save to Database and PDF download.
    """
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

    status   = status_resp.get("status", "unknown")
    ch_done  = status_resp.get("challenges_completed", 0)
    ch_total = status_resp.get("total_challenges", len(challenges) or 2)
    result   = status_resp.get("result")

    col_s, col_p = st.columns([3, 2])
    with col_s:
        status_labels = {
            "waiting":   "⏳ Waiting for customer",
            "active":    "🔵 Session in progress",
            "completed": "✅ Completed",
            "failed":    "❌ Failed",
            "expired":   "⚫ Expired",
        }
        st.metric("Session status", status_labels.get(status, status))
    with col_p:
        st.metric("Challenges", f"{ch_done} / {ch_total} done")

    if status in ("waiting", "active"):
        # Auto-refresh every 2 s while the customer is active
        st.progress(ch_done / max(ch_total, 1), text="Liveness challenges progress")
        with st.spinner("Waiting for customer to complete verification..."):
            time.sleep(2)
        st.rerun()

    elif status == "completed" and result:
        passed     = result.get("passed", False)
        disp_score = result.get("display_score", result.get("match_score", 0) * 100)
        cosine_sim = result.get("cosine_similarity", 0.0)

        if passed:
            st.success(f"Identity Verified — Face match score: {disp_score:.1f}%")
        else:
            st.error(f"Verification Failed — Score: {disp_score:.1f}%")
            st.caption(result.get("message", ""))

        vkyc_result, entity_ref, vkyc_pdf = _build_kyc_save_artifacts(
            result,
            sid,
            status_resp,
            st.session_state.get("vkyc_doc_filename", ""),
            st.session_state.get("vkyc_forced_ref"),
            st.session_state.get("vkyc_extra_identity"),
            cosine_sim,
        )

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
                        forced_entity_ref=st.session_state.get("vkyc_forced_ref"),
                        extra_identity=st.session_state.get("vkyc_extra_identity"),
                        doc_filename=st.session_state.get("vkyc_doc_filename", ""),
                        pdf_bytes=vkyc_pdf,
                        doc_bytes=st.session_state.get("vkyc_doc_bytes"),
                    )
                    if saved:
                        st.session_state["vkyc_saved_remote"] = True
                        st.session_state["vkyc_saved_remote_ref"] = saved.get("entity_ref")
                        st.rerun()
                    else:
                        st.error(
                            "Video KYC completed but could not be saved to the database. "
                            "Check the Logs screen for details."
                        )
        else:
            st.info("Database is offline — result not persisted.")

        if vkyc_pdf and st.session_state.get("vkyc_saved_remote"):
            st.download_button(
                "⬇ Download KYC Report (PDF)",
                data=vkyc_pdf,
                file_name=f"video_kyc_{entity_ref or 'report'}.pdf",
                mime="application/pdf",
                key="vkyc_pdf_dl",
            )

        if st.button("🔄 Start New Session", use_container_width=True):
            _clear_session_state()
            st.rerun()

    elif status in ("failed", "expired"):
        msg = "Session expired — please create a new one." if status == "expired" \
              else "Verification failed. Please retry."
        st.error(msg)
        if st.button("🔄 Start New Session", use_container_width=True):
            _clear_session_state()
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

