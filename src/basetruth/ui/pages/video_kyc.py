"""Video KYC page — schedule sessions and conduct real-time liveness + face match."""
from __future__ import annotations

import io
import textwrap
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

import streamlit as st

from basetruth.ui.components import (
    _DB_IMPORTS_OK,
    _page_title,
    _render_entity_link_widget,
    db_available,
    save_identity_check,
)

# ---------------------------------------------------------------------------
# ICS calendar invite generator (no external library needed)
# ---------------------------------------------------------------------------

def _make_ics(
    customer_name: str,
    agent_name: str,
    meeting_link: str,
    start_dt: datetime,
    duration_minutes: int,
    description: str,
) -> bytes:
    """Generate a .ics calendar invite as raw bytes."""
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
        f"SUMMARY:Video KYC Session — {customer_name}",
        f"ORGANIZER;CN={agent_name}:mailto:noreply@basetruth.local",
        f"ATTENDEE;CN={customer_name};ROLE=REQ-PARTICIPANT;RSVP=TRUE:mailto:customer@placeholder",
        "STATUS:CONFIRMED",
    ]

    # Fold long lines per RFC 5545 (max 75 octets)
    desc_safe = description.replace("\n", "\\n").replace(",", "\\,").replace(";", "\\;")
    location_safe = meeting_link.replace(",", "\\,").replace(";", "\\;")
    for key, val in [("DESCRIPTION", desc_safe), ("LOCATION", val if (val := location_safe) else "")]:
        raw = f"{key}:{val}"
        # fold at 75 chars
        folded = "\r\n ".join(textwrap.wrap(raw, 75, break_long_words=True, break_on_hyphens=False))
        lines.append(folded)

    lines += ["END:VEVENT", "END:VCALENDAR"]
    return ("\r\n".join(lines) + "\r\n").encode("utf-8")


# ---------------------------------------------------------------------------
# Schedule tab
# ---------------------------------------------------------------------------

def _tab_schedule() -> None:
    st.markdown(
        """
        <div style="background:linear-gradient(135deg,#0f172a 0%,#1e293b 100%);
                    border:1px solid #334155;border-radius:12px;padding:1.2rem 1.4rem;
                    margin-bottom:1.2rem;">
        <h4 style="color:#e2e8f0;margin:0 0 0.5rem;">How Video KYC works in the market</h4>
        <p style="color:#94a3b8;font-size:0.88rem;margin:0;line-height:1.6;">
        The most modern approach used by banks and fintechs today is a <strong style="color:#c4b5fd">
        scheduled live video call</strong> where a KYC agent verifies identity in real time.<br>
        The customer receives a calendar invite with a secure video link (Zoom, Teams, or Google Meet).
        During the call the agent uses an AI tool to run face-match and liveness checks.<br>
        This is exactly what BaseTruth enables — you schedule the session here, the customer joins
        the video call, and the <em>Conduct Verification</em> tab performs the AI checks live.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("vkyc_schedule_form", clear_on_submit=False):
        st.subheader("Session Details")
        c1, c2 = st.columns(2)
        with c1:
            customer_name = st.text_input("Customer Name *", placeholder="e.g. Rahul Sharma")
            agent_name = st.text_input(
                "Agent / Your Name *", placeholder="e.g. Priya Mehta"
            )
        with c2:
            session_date = st.date_input("Session Date *", min_value=datetime.today().date())
            session_time = st.time_input("Session Time (IST) *", value=None)

        duration = st.selectbox(
            "Duration",
            options=[15, 20, 30, 45, 60],
            index=2,
            format_func=lambda x: f"{x} minutes",
        )

        platform = st.selectbox(
            "Video Platform",
            options=["Zoom", "Microsoft Teams", "Google Meet", "Other"],
        )
        meeting_link = st.text_input(
            "Meeting / Join Link *",
            placeholder="https://zoom.us/j/... or https://teams.microsoft.com/...",
            help=(
                "Create a meeting in Zoom/Teams/Meet first, copy the join link, "
                "then paste it here. BaseTruth will embed it in the calendar invite."
            ),
        )

        notes = st.text_area(
            "Additional Notes (optional)",
            placeholder="e.g. Please keep your Aadhaar card and PAN card handy.",
            height=80,
        )

        submitted = st.form_submit_button("📅 Generate Calendar Invite", type="primary", use_container_width=True)

    if submitted:
        errors = []
        if not customer_name.strip():
            errors.append("Customer Name is required.")
        if not agent_name.strip():
            errors.append("Agent Name is required.")
        if not meeting_link.strip():
            errors.append("Meeting link is required.")
        if session_time is None:
            errors.append("Session Time is required.")

        if errors:
            for e in errors:
                st.error(e)
            return

        # Build UTC datetime (assume IST = UTC+5:30)
        ist_offset = timezone(timedelta(hours=5, minutes=30))
        start_dt_ist = datetime(
            session_date.year, session_date.month, session_date.day,
            session_time.hour, session_time.minute, tzinfo=ist_offset,
        )
        start_dt_utc = start_dt_ist.astimezone(timezone.utc)

        description = textwrap.dedent(f"""\
            Video KYC Session scheduled via BaseTruth.

            Customer: {customer_name.strip()}
            Agent: {agent_name.strip()}
            Platform: {platform}
            Join Link: {meeting_link.strip()}

            What to prepare:
            - Original Aadhaar Card (physical or digital)
            - Original PAN Card
            - Good lighting and a stable internet connection

            {('Notes: ' + notes.strip()) if notes.strip() else ''}

            The agent will use BaseTruth AI to verify your identity during this call.
            This is a secure, automated check — no data is shared externally.
        """).strip()

        ics_bytes = _make_ics(
            customer_name=customer_name.strip(),
            agent_name=agent_name.strip(),
            meeting_link=meeting_link.strip(),
            start_dt=start_dt_utc,
            duration_minutes=duration,
            description=description,
        )

        st.success(
            f"✅ Calendar invite ready for **{customer_name.strip()}** — "
            f"{session_date.strftime('%d %b %Y')} at "
            f"{session_time.strftime('%I:%M %p')} IST ({duration} min)"
        )

        col_dl, col_copy = st.columns([1, 1])

        with col_dl:
            st.download_button(
                label="⬇️ Download .ics (Calendar Invite)",
                data=ics_bytes,
                file_name=f"vkyc_{customer_name.strip().replace(' ', '_')}.ics",
                mime="text/calendar",
                use_container_width=True,
                help=(
                    "Open this file on your computer or phone to add the event to "
                    "your calendar. Forward the same file to the customer so they "
                    "can add it to their Google Calendar / Outlook / Apple Calendar."
                ),
            )

        with col_copy:
            st.info("Forward the downloaded .ics file to your customer via email or WhatsApp.", icon="📧")

        with st.expander("📋 Email invite text (copy & paste)", expanded=False):
            email_body = textwrap.dedent(f"""\
                Subject: Video KYC Session — {session_date.strftime('%d %b %Y')} at {session_time.strftime('%I:%M %p')} IST

                Dear {customer_name.strip()},

                Your Video KYC session has been scheduled. Please find the details below:

                Date & Time : {session_date.strftime('%d %B %Y')} at {session_time.strftime('%I:%M %p')} IST
                Duration    : {duration} minutes
                Platform    : {platform}
                Join Link   : {meeting_link.strip()}

                What to keep ready:
                  • Original Aadhaar Card (physical or digital copy)
                  • Original PAN Card
                  • Good lighting and a stable internet connection

                {('Additional Notes: ' + notes.strip()) if notes.strip() else ''}

                A calendar invite (.ics file) is attached. Click it to add this event
                to your Google Calendar / Outlook / Apple Calendar.

                Regards,
                {agent_name.strip()}
            """).strip()
            st.code(email_body, language="text")

        with st.expander("ℹ️ How to share in Zoom / Teams / Google Meet"):
            st.markdown(
                """
                **Zoom** — Open Zoom → Schedule a Meeting → copy the Join URL → paste it above.

                **Microsoft Teams** — Open Teams → Calendar → New Meeting → copy Join Link → paste above.

                **Google Meet** — Open Google Calendar → create event → add Google Meet → copy link → paste above.

                After generating the invite, send the **.ics file** to the customer.
                They click it and it appears directly in their calendar with the join link.
                """
            )


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------

def _page_video_kyc() -> None:
    st.markdown(_page_title("🎥", "Video KYC"), unsafe_allow_html=True)
    st.caption("Schedule sessions and conduct AI-powered live identity verification.")

    tab_schedule, tab_conduct = st.tabs(["📅 Schedule Session", "🎥 Conduct Verification"])

    with tab_schedule:
        _tab_schedule()

    with tab_conduct:
        _tab_conduct()


def _tab_conduct() -> None:

    with st.expander("ℹ️ How it works", expanded=False):
        st.markdown(
            "This interface captures frames from your webcam and runs the "
            "**RetinaFace** + **ArcFace** pipeline directly on each frame.\n\n"
            "1. Upload a reference ID document so the system can extract the face embedding.\n"
            "2. Use the camera input below to capture a live photo.\n"
            "3. The system will detect liveness cues and compare the live face to the ID.\n\n"
            "To test **Liveness Detection**, turn your head slightly left or right before capturing."
        )

    # 0) Entity selector
    forced_ref = None
    extra_identity = None
    if _DB_IMPORTS_OK and db_available():
        forced_ref, extra_identity = _render_entity_link_widget("vkyc", mandatory=True)
        st.divider()

    # 1) Upload Reference Document
    st.subheader("1. Setup Reference ID")
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
        nparr = np.frombuffer(doc_file.getvalue(), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        faces = face_app.get(img)

        if len(faces) > 0:
            face = max(
                faces,
                key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            )
            st.session_state["vkyc_ref_emb"] = face.normed_embedding
            st.success(
                "✅ ID subject successfully extracted. You may begin the live capture."
            )
        else:
            st.error("❌ No face detected in the uploaded ID document.")
            st.session_state.pop("vkyc_ref_emb", None)

    # 2) Live Camera Capture & Processing
    st.divider()
    st.subheader("2. Live Liveness Test & Face Match")

    reference_emb = st.session_state.get("vkyc_ref_emb")

    if reference_emb is None:
        st.warning(
            "Please upload a Reference ID Document first to generate the cryptographic session."
        )
        return

    st.info(
        "📸 Use the camera below to take a live photo. "
        "For **liveness detection**, slightly turn your head left or right before capturing."
    )

    camera_photo = st.camera_input("Capture live photo", key="vkyc_camera")

    if camera_photo is not None:
        if not forced_ref and not extra_identity:
            st.warning("Please link an entity or enter applicant details (mandatory).")
            st.stop()

        import cv2  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        nparr = np.frombuffer(camera_photo.getvalue(), np.uint8)
        live_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if live_img is None:
            st.error("Failed to decode the captured image.")
            return

        with st.spinner("Running RetinaFace detection + ArcFace matching …"):
            from basetruth.vision.video_kyc import VideoKYCProcessor  # noqa: PLC0415

            processor = VideoKYCProcessor()
            processor.set_reference_embedding(reference_emb)

            try:
                faces = processor.face_app.get(live_img)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Face detection failed: {exc}")
                return

            if not faces:
                st.error(
                    "❌ No face detected in the captured frame. "
                    "Please try again — make sure your face is clearly visible."
                )
                return

            primary = max(
                faces,
                key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            )
            box = primary.bbox.astype(int)

            # --- Liveness check (head-turn heuristic via eye/nose keypoints) ---
            liveness_state = "Center"
            head_turn_passed = False
            if primary.kps is not None:
                left_eye_x = primary.kps[0][0]
                right_eye_x = primary.kps[1][0]
                nose_x = primary.kps[2][0]
                dist_left = abs(nose_x - left_eye_x)
                dist_right = max(abs(right_eye_x - nose_x), 1.0)
                ratio = dist_left / dist_right
                if ratio > 1.6:
                    liveness_state = "Turned Right"
                    head_turn_passed = True
                elif ratio < 0.6:
                    liveness_state = "Turned Left"
                    head_turn_passed = True

            # --- Identity matching ---
            emb = primary.normed_embedding
            sim = float(np.dot(emb, reference_emb))
            score = min(max((sim - (-0.5)) / (1.0 - (-0.5)) * 100, 0), 100)
            is_match = sim >= 0.40

            # --- Annotate the image ---
            color = (0, 255, 0) if is_match else (0, 0, 255)
            label = f"VERIFIED: {score:.1f}%" if is_match else f"MISMATCH: {score:.1f}%"
            cv2.rectangle(
                live_img, (box[0], box[1]), (box[2], box[3]), color, 3
            )
            cv2.putText(
                live_img, label, (box[0], box[1] - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
            )
            live_txt = (
                f"Liveness: {liveness_state} "
                f"{'(PASS)' if head_turn_passed else ''}"
            )
            cv2.putText(
                live_img, live_txt, (box[0], box[1] - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2,
            )
            if primary.kps is not None:
                for p in primary.kps.astype(int):
                    cv2.circle(live_img, (p[0], p[1]), 3, (255, 0, 0), cv2.FILLED)

            # --- Display results ---
            annotated_rgb = cv2.cvtColor(live_img, cv2.COLOR_BGR2RGB)
            st.image(annotated_rgb, caption="Annotated Result", use_container_width=True)

            col1, col2, col3 = st.columns(3)
            with col1:
                if is_match:
                    st.success(f"✅ **Identity Match**\n\nScore: **{score:.1f}%**")
                else:
                    st.error(f"🚨 **Identity Mismatch**\n\nScore: **{score:.1f}%**")
            with col2:
                if head_turn_passed:
                    st.success(
                        f"✅ **Liveness Passed**\n\nHead: **{liveness_state}**"
                    )
                else:
                    st.warning(
                        f"⚠️ **Liveness Inconclusive**\n\nHead: **{liveness_state}**\n\n"
                        "Turn your head and retake."
                    )
            with col3:
                overall = "PASS" if (is_match and head_turn_passed) else "FAIL"
                if overall == "PASS":
                    st.success("✅ **KYC Verdict: PASS**")
                else:
                    st.error("❌ **KYC Verdict: FAIL**")
                st.caption(f"Cosine similarity: {sim:.4f}")

            # -- Persist Video KYC to DB + Generate PDF -------------------------
            if _DB_IMPORTS_OK and db_available():
                vkyc_result: Dict[str, Any] = {
                    "is_match": bool(is_match),
                    "confidence": float(sim),
                    "cosine_similarity": float(sim),
                    "display_score": float(score),
                    "threshold": 0.40,
                    "liveness_state": liveness_state,
                    "liveness_passed": bool(head_turn_passed),
                    "match": bool(is_match),
                }

                _ref_for_pdf = forced_ref or ""
                _name_for_pdf = ""
                if extra_identity:
                    _name_for_pdf = (
                        f"{extra_identity.get('first_name', '')} "
                        f"{extra_identity.get('last_name', '')}".strip()
                    )

                try:
                    from basetruth.reporting.pdf import (  # noqa: PLC0415
                        render_identity_check_pdf,
                    )

                    vkyc_pdf = render_identity_check_pdf(
                        check_type="video_kyc",
                        result=vkyc_result,
                        entity_ref=_ref_for_pdf,
                        entity_name=_name_for_pdf,
                        doc_filename=doc_file.name if doc_file else "",
                    )
                except Exception:  # noqa: BLE001
                    vkyc_pdf = None

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
                        file_name=(
                            f"video_kyc_report_"
                            f"{st.session_state.get('vkyc_entity_ref', 'unlinked')}.pdf"
                        ),
                        mime="application/pdf",
                        key="vkyc_pdf_dl",
                    )
