"""Video KYC page — real-time liveness detection and face matching via webcam."""
from __future__ import annotations

from typing import Any, Dict

import streamlit as st

from basetruth.ui.components import (
    _DB_IMPORTS_OK,
    _page_title,
    _render_entity_link_widget,
    db_available,
    save_identity_check,
)


def _page_video_kyc() -> None:
    st.markdown(_page_title("🎥", "Video KYC (Real-Time)"), unsafe_allow_html=True)
    st.caption("Perform live liveness detection and face matching via webcam.")

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
