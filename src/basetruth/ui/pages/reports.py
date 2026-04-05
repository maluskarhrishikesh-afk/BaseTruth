"""Reports page — one consolidated PDF per entity, plus source-document ZIP export."""
from __future__ import annotations

import io
import zipfile

import streamlit as st

from basetruth.service import BaseTruthService
from basetruth.ui.components import (
    _DB_IMPORTS_OK,
    _db_available_cached,
    _page_title,
    get_all_entities_with_scans,
    get_entity_layered_analysis,
    get_entity_identity_checks,
    get_entity_scans,
    minio_delete_object,
    minio_get_object,
    minio_list_entity_objects,
    minio_upload,
)

_CONSOLIDATED_PDF_KEY = "{entity_ref}/consolidated_report.pdf"
_LAYERED_PDF_FALLBACK_KEY = "{entity_ref}/layered_analysis_report.pdf"


def _is_audit_source_object(key: str) -> bool:
    normalized = key.replace("\\", "/")
    filename = normalized.rsplit("/", 1)[-1].lower()
    if filename == "consolidated_report.pdf":
        return False
    if filename.endswith("_report.pdf"):
        return False
    if "/case_reports/" in normalized:
        return False
    return True


def _page_reports(service: BaseTruthService) -> None:
    _ = service
    st.markdown(_page_title("📊", "Reports"), unsafe_allow_html=True)

    with st.expander("ℹ️ How to use this screen", expanded=False):
        st.markdown(
            """
This screen keeps one final report per applicant and nothing else.

- `Generate / Refresh Consolidated Report` builds a single PDF covering every saved activity for that applicant.
- If a newer activity is added later, generating again replaces the older consolidated PDF.
- `Download All Source Documents (ZIP)` bundles the applicant's uploaded evidence files for audit.
"""
        )

    if not _DB_IMPORTS_OK or not _db_available_cached():
        st.info("📴 Database is offline. Connect PostgreSQL to view consolidated reports.")
        return

    entities = get_all_entities_with_scans()
    if not entities:
        st.info(
            "No data yet. Use Identity Verification, Video KYC, Scan Document, or Bulk Scan first."
        )
        return

    search = st.text_input(
        "🔍 Filter applicants",
        placeholder="Name, PAN, email, or BT-reference...",
        key="reports_search",
    ).strip().lower()

    filtered = [
        entity
        for entity in entities
        if not search
        or search in (entity.get("name") or "").lower()
        or search in (entity.get("pan_number") or "").lower()
        or search in (entity.get("email") or "").lower()
        or search in (entity.get("entity_ref") or "").lower()
    ]

    st.caption(f"{len(filtered)} applicant(s) shown")
    st.divider()

    for entity in filtered:
        entity_ref = entity["entity_ref"]
        entity_name = entity.get("name") or entity_ref
        pan_number = entity.get("pan_number") or ""
        email = entity.get("email") or ""
        linked_scans = entity.get("scans") or []
        identity_checks = get_entity_identity_checks(entity_ref)
        face_checks = [check for check in identity_checks if check.get("check_type") == "face_match"]
        kyc_checks = [check for check in identity_checks if check.get("check_type") == "video_kyc"]

        subtitle = "  ·  ".join(filter(None, [pan_number, email]))
        summary_parts = []
        if face_checks:
            summary_parts.append(f"{len(face_checks)} face match")
        if kyc_checks:
            summary_parts.append(f"{len(kyc_checks)} Video KYC")
        if linked_scans:
            summary_parts.append(f"{len(linked_scans)} scan(s)")
        summary = ", ".join(summary_parts) or "no saved activity"

        label = f"👤 **{entity_name}** — {entity_ref}"
        if subtitle:
            label += f"  ·  {subtitle}"
        label += f"  ·  *{summary}*"

        with st.expander(label, expanded=False):
            if not face_checks and not kyc_checks and not linked_scans:
                st.info("No verification activity recorded for this applicant.")
                continue

            col1, col2, col3 = st.columns(3)
            col1.metric("Face Match Checks", len(face_checks))
            col2.metric("Video KYC Sessions", len(kyc_checks))
            col3.metric("Document Scans", len(linked_scans))
            st.divider()

            action_col1, action_col2 = st.columns(2)

            with action_col1:
                if st.button(
                    "📄 Generate / Refresh Consolidated Report",
                    key=f"reports_generate_{entity_ref}",
                    use_container_width=True,
                ):
                    with st.spinner("Building consolidated PDF..."):
                        try:
                            from basetruth.reporting.pdf import render_consolidated_entity_pdf  # noqa: PLC0415

                            pdf_bytes = render_consolidated_entity_pdf(
                                entity=entity,
                                scans=get_entity_scans(entity_ref) or linked_scans,
                                identity_checks=identity_checks,
                            )
                            pdf_key = _CONSOLIDATED_PDF_KEY.format(entity_ref=entity_ref)
                            minio_delete_object(pdf_key)
                            upload_ok = minio_upload(pdf_key, pdf_bytes, "application/pdf")
                            st.session_state[f"reports_pdf_{entity_ref}"] = pdf_bytes
                            if upload_ok:
                                st.success("✅ Consolidated report generated and saved.")
                            else:
                                st.warning(
                                    "Report generated, but MinIO upload failed. Use the download button below."
                                )
                        except Exception as exc:  # noqa: BLE001
                            st.error(f"Failed to generate report: {exc}")

                pdf_bytes = st.session_state.get(f"reports_pdf_{entity_ref}")
                if not pdf_bytes:
                    pdf_bytes = minio_get_object(_CONSOLIDATED_PDF_KEY.format(entity_ref=entity_ref))
                    if pdf_bytes:
                        st.session_state[f"reports_pdf_{entity_ref}"] = pdf_bytes
                if pdf_bytes:
                    st.download_button(
                        "⬇ Download Consolidated Report (PDF)",
                        data=pdf_bytes,
                        file_name=f"consolidated_report_{entity_ref}.pdf",
                        mime="application/pdf",
                        key=f"reports_download_pdf_{entity_ref}",
                        use_container_width=True,
                    )

                layered_state = get_entity_layered_analysis(entity_ref).get("report_state") or {}
                layered_key = layered_state.get("minio_key") or _LAYERED_PDF_FALLBACK_KEY.format(entity_ref=entity_ref)
                layered_pdf = minio_get_object(layered_key) if layered_key else None
                if layered_pdf:
                    st.download_button(
                        "⬇ Download Final Layered Report (PDF)",
                        data=layered_pdf,
                        file_name=f"layered_analysis_{entity_ref}.pdf",
                        mime="application/pdf",
                        key=f"reports_download_layered_pdf_{entity_ref}",
                        use_container_width=True,
                    )

            with action_col2:
                if st.button(
                    "📦 Download All Source Documents (ZIP)",
                    key=f"reports_zip_build_{entity_ref}",
                    use_container_width=True,
                ):
                    with st.spinner("Collecting source documents..."):
                        try:
                            objects = [
                                obj
                                for obj in minio_list_entity_objects(entity_ref)
                                if _is_audit_source_object(obj["key"])
                            ]
                            if not objects:
                                st.warning("No uploaded source documents were found for this applicant.")
                            else:
                                zip_buffer = io.BytesIO()
                                with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
                                    for obj in objects:
                                        obj_key = obj["key"]
                                        obj_bytes = minio_get_object(obj_key)
                                        if obj_bytes:
                                            archive.writestr(obj_key[len(entity_ref) + 1 :], obj_bytes)
                                st.session_state[f"reports_zip_{entity_ref}"] = zip_buffer.getvalue()
                                st.success(f"✅ ZIP ready — {len(objects)} file(s)")
                        except Exception as exc:  # noqa: BLE001
                            st.error(f"Failed to build ZIP: {exc}")

                zip_bytes = st.session_state.get(f"reports_zip_{entity_ref}")
                if zip_bytes:
                    st.download_button(
                        "⬇ Download ZIP",
                        data=zip_bytes,
                        file_name=f"documents_{entity_ref}.zip",
                        mime="application/zip",
                        key=f"reports_download_zip_{entity_ref}",
                        use_container_width=True,
                    )
