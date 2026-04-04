"""Reports page â€” one consolidated PDF per entity, ZIP of all source documents."""
from __future__ import annotations

import io
import zipfile
from typing import Any, Dict

import streamlit as st

from basetruth.service import BaseTruthService
from basetruth.ui.components import (
    _DB_IMPORTS_OK,
    _db_available_cached,
    _page_title,
    get_all_entities_with_scans,
    get_entity_identity_checks,
    get_entity_scans,
    minio_delete_object,
    minio_get_object,
    minio_list_entity_objects,
    minio_upload,
)

_CONSOLIDATED_PDF_KEY = "{entity_ref}/consolidated_report.pdf"


def _page_reports(service: BaseTruthService) -> None:
    st.markdown(_page_title("ðŸ“Š", "Reports"), unsafe_allow_html=True)

    with st.expander("â„¹ï¸ How to use this screen", expanded=False):
        st.markdown(
            """
**One consolidated report per applicant** â€” covering all their verification activity.

- **Generate / Refresh Consolidated Report** â€” builds a single PDF covering Identity
  Verification, Video KYC, and Document Scans.  Each new generation replaces the
  previous version so only the latest report is kept.
- **Download All Documents (ZIP)** â€” bundles every source document uploaded for that
  applicant (Aadhaar, PAN, selfies, payslips, bank statements, etc.) into a single
  ZIP file for audit purposes.
"""
        )

    if not _DB_IMPORTS_OK or not _db_available_cached():
        st.info(
            "ðŸ“´ Database is offline. Connect PostgreSQL to view entity reports."
        )
        return

    entities = get_all_entities_with_scans()
    if not entities:
        st.info(
            "No data yet. Use **Scan**, **Bulk Scan**, or **Identity Verification** "
            "to add records."
        )
        return

    # â”€â”€ Search / filter â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    search = st.text_input(
        "ðŸ” Filter applicants",
        placeholder="Name, PAN, email, or BT-referenceâ€¦",
        key="rpt_search",
    ).strip().lower()

    filtered = [
        e for e in entities
        if not search
        or search in (e.get("name") or "").lower()
        or search in (e.get("pan_number") or "").lower()
        or search in (e.get("email") or "").lower()
        or search in (e.get("entity_ref") or "").lower()
    ]

    st.caption(f"{len(filtered)} applicant(s) shown")
    st.divider()

    # â”€â”€ Per-entity cards â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    for ent in filtered:
        ref        = ent["entity_ref"]
        name       = ent["name"] or ref
        pan        = ent.get("pan_number") or ""
        email      = ent.get("email") or ""
        ent_scans  = ent.get("scans") or []

        # Fetch identity checks for this entity
        idv_checks = get_entity_identity_checks(ref) if _DB_IMPORTS_OK and _db_available_cached() else []
        face_checks = [c for c in idv_checks if c.get("check_type") == "face_match"]
        kyc_checks  = [c for c in idv_checks if c.get("check_type") == "video_kyc"]

        sub = "  Â·  ".join(filter(None, [pan, email]))
        summary_parts = []
        if face_checks:
            summary_parts.append(f"{len(face_checks)} face match")
        if kyc_checks:
            summary_parts.append(f"{len(kyc_checks)} Video KYC")
        if ent_scans:
            summary_parts.append(f"{len(ent_scans)} scan(s)")
        summary_str = ", ".join(summary_parts) or "no activity yet"

        expander_label = (
            f"ðŸ‘¤ **{name}** â€” {ref}"
            + (f"  Â·  {sub}" if sub else "")
            + f"  Â·  *{summary_str}*"
        )

        with st.expander(expander_label, expanded=False):
            if not face_checks and not kyc_checks and not ent_scans:
                st.info("No verification activity recorded for this entity.")
                continue

            # â”€â”€ Summary table â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Face Match Checks", len(face_checks))
            col_b.metric("Video KYC Sessions", len(kyc_checks))
            col_c.metric("Document Scans", len(ent_scans))

            st.divider()

            btn_col1, btn_col2 = st.columns(2)

            # â”€â”€ Generate / Refresh consolidated PDF â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            with btn_col1:
                if st.button(
                    "ðŸ“„ Generate / Refresh Consolidated Report",
                    key=f"gen_report_{ref}",
                    use_container_width=True,
                ):
                    with st.spinner("Building consolidated PDFâ€¦"):
                        try:
                            from basetruth.reporting.pdf import (  # noqa: PLC0415
                                render_consolidated_entity_pdf,
                            )

                            # Fetch full scan list for the entity
                            full_scans = get_entity_scans(ref) or ent_scans

                            pdf_bytes = render_consolidated_entity_pdf(
                                entity=ent,
                                scans=full_scans,
                                identity_checks=idv_checks,
                            )

                            pdf_key = _CONSOLIDATED_PDF_KEY.format(entity_ref=ref)
                            # Delete previous version
                            minio_delete_object(pdf_key)
                            # Upload new version
                            upload_ok = minio_upload(pdf_key, pdf_bytes, "application/pdf")

                            st.session_state[f"consolidated_pdf_{ref}"] = pdf_bytes
                            if upload_ok:
                                st.success("âœ… Consolidated report generated and saved.")
                            else:
                                st.warning(
                                    "Report generated but could not be uploaded to MinIO. "
                                    "Use the download button below."
                                )
                        except Exception as exc:  # noqa: BLE001
                            st.error(f"Failed to generate report: {exc}")

            # â”€â”€ Download consolidated PDF â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            _pdf = st.session_state.get(f"consolidated_pdf_{ref}")
            if not _pdf:
                # Try to fetch existing one from MinIO
                _pdf_key = _CONSOLIDATED_PDF_KEY.format(entity_ref=ref)
                try:
                    _pdf = minio_get_object(_pdf_key)
                    if _pdf:
                        st.session_state[f"consolidated_pdf_{ref}"] = _pdf
                except Exception:  # noqa: BLE001
                    pass

            if _pdf:
                with btn_col1:
                    st.download_button(
                        "â¬‡ Download Consolidated Report (PDF)",
                        data=_pdf,
                        file_name=f"consolidated_report_{ref}.pdf",
                        mime="application/pdf",
                        key=f"dl_consolidated_{ref}",
                        use_container_width=True,
                    )

            # â”€â”€ Download All Documents (ZIP) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            with btn_col2:
                if st.button(
                    "ðŸ“¦ Download All Documents (ZIP)",
                    key=f"zip_docs_{ref}",
                    use_container_width=True,
                ):
                    with st.spinner("Collecting documents from storageâ€¦"):
                        try:
                            objects = minio_list_entity_objects(ref)
                            if not objects:
                                st.warning(
                                    "No source documents found in storage for this entity."
                                )
                            else:
                                zip_buf = io.BytesIO()
                                with zipfile.ZipFile(
                                    zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED
                                ) as zf:
                                    for obj in objects:
                                        obj_key = obj["key"]
                                        obj_data = minio_get_object(obj_key)
                                        if obj_data:
                                            # Preserve relative path inside ZIP
                                            arc_name = obj_key[len(ref) + 1:]  # strip entity_ref/
                                            zf.writestr(arc_name, obj_data)

                                zip_bytes = zip_buf.getvalue()
                                st.session_state[f"zip_docs_{ref}"] = zip_bytes
                                st.success(f"âœ… ZIP ready â€” {len(objects)} file(s)")
                        except Exception as exc:  # noqa: BLE001
                            st.error(f"Failed to build ZIP: {exc}")

            _zip = st.session_state.get(f"zip_docs_{ref}")
            if _zip:
                with btn_col2:
                    st.download_button(
                        "â¬‡ Download ZIP",
                        data=_zip,
                        file_name=f"documents_{ref}.zip",
                        mime="application/zip",
                        key=f"dl_zip_{ref}",
                        use_container_width=True,
                    )

