"""Scan page (single document)."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict

import streamlit as st

from basetruth.service import BaseTruthService
from basetruth.ui.components import (
    _DB_IMPORTS_OK,
    _db_available_cached,
    _page_title,
    _render_entity_link_widget,
    _render_report_summary,
    _save_uploaded_files,
    minio_upload,
    save_scan_to_db,
)


def _scan_source_content_type(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return "application/pdf"
    if path.suffix.lower() == ".json":
        return "application/json"
    return "application/octet-stream"


def _page_scan(service: BaseTruthService) -> None:
    st.markdown(_page_title("🔍", "Scan Document"), unsafe_allow_html=True)

    with st.expander("ℹ️ How to use this screen", expanded=False):
        st.markdown(
            """
**Scan** verifies a single document end-to-end.

1. Upload the file (PDF, image, or LiteParse JSON) or paste a local path.
2. *(Optional)* Expand **"Associate with a person"** to link the result to an applicant's
   profile — this prevents duplicate entries in Records when the document doesn't include
   PAN or Aadhaar.
3. Hit **Run scan** and review the Truth Score, risk level, and forensic signals.
4. Download the JSON or PDF report to share with colleagues or attach to a loan file.
"""
        )

    upload = st.file_uploader(
        "Drop a file here — PDF, JSON (LiteParse or structured), or image",
        type=None,
        accept_multiple_files=False,
        label_visibility="visible",
    )
    path_input = st.text_input("Or enter an existing file path on disk")
    st.caption(
        "Note: scanned identity documents (Aadhaar, PAN card) are image-only PDFs with no text layer. "
        "LiteParse requires ImageMagick for OCR on those files. "
        "If ImageMagick is not installed, BaseTruth uses a text-extraction fallback that extracts "
        "metadata and structure but cannot read the printed text from image-only pages."
    )

    forced_ref, extra_identity = _render_entity_link_widget("scan")

    if st.button("Run scan ->", type="primary"):
        report: Dict[str, Any] | None = None
        scan_error: str = ""
        source_path: Path | None = None
        with st.spinner("Running BaseTruth scan..."):
            try:
                if upload is not None:
                    temp_dir = Path(tempfile.mkdtemp(prefix="bt_upload_"))
                    source_path = _save_uploaded_files([upload], temp_dir)[0]
                    report = service.scan_document(
                        source_path,
                        forced_entity_ref=forced_ref or None,
                        extra_identity=extra_identity or None,
                        persist_to_db=False,
                    )
                elif path_input.strip():
                    source_path = Path(path_input.strip())
                    report = service.scan_document(
                        source_path,
                        forced_entity_ref=forced_ref or None,
                        extra_identity=extra_identity or None,
                        persist_to_db=False,
                    )
                else:
                    st.warning("Provide an uploaded file or a local file path.")
            except FileNotFoundError as exc:
                scan_error = f"File not found: {exc}"
            except Exception as exc:  # noqa: BLE001
                scan_error = (
                    f"Scan failed: {exc}\n\n"
                    "Common causes:\n"
                    "- ImageMagick is not installed (needed for image-only PDFs such as "
                    "Aadhaar / PAN card scans).  Install from https://imagemagick.org/\n"
                    "- The PDF is password-protected or corrupt.\n"
                    "- An unexpected LiteParse or OCR error occurred."
                )

        if scan_error:
            st.error(scan_error)
            return

        if report and source_path:
            st.session_state["scan_pending_report"] = report
            st.session_state["scan_pending_source_path"] = str(source_path)
            st.session_state["scan_pending_forced_ref"] = forced_ref
            st.session_state["scan_pending_extra_identity"] = extra_identity
            st.session_state["scan_saved"] = False
            st.session_state.pop("scan_saved_ref", None)

    report = st.session_state.get("scan_pending_report")
    if report:
        artifacts = report.get("artifacts", {})
        summary = report.get("structured_summary", {})
        is_fallback = artifacts.get("parse_fallback") or summary.get("parse_fallback")
        is_image_only = artifacts.get("is_image_only_pdf") or summary.get(
            "is_image_only_pdf"
        )
        ocr_engine = artifacts.get("ocr_engine", "")

        if is_fallback:
            fallback_reason = (
                artifacts.get("parse_fallback_reason")
                or summary.get("parse_fallback_reason", "")
            ).split("|")[0].strip()

            if is_image_only and ocr_engine == "pytesseract":
                st.info(
                    "**Image-only PDF detected** -- LiteParse required ImageMagick which is "
                    "not installed. BaseTruth used **Tesseract OCR** as a fallback and "
                    "successfully extracted text from the document.  "
                    "Field extraction quality may differ from a full LiteParse scan."
                )
            elif is_image_only and ocr_engine == "unavailable":
                st.warning(
                    "**Image-only PDF -- OCR required for full extraction.**  "
                    "This document (e.g. Aadhaar card, PAN card) contains no embedded "
                    "text layer. PDF metadata forensics ran, but field-level extraction "
                    "was not possible without OCR.\n\n"
                    "**To fix this, choose ONE option:**\n\n"
                    "**Option A (recommended) -- Install ImageMagick:**  \n"
                    "Download from https://imagemagick.org/script/download.php#windows  \n"
                    "Restart the terminal after install, then re-scan.\n\n"
                    "**Option B -- Install Tesseract + Poppler:**  \n"
                    "1. Tesseract: https://github.com/UB-Mannheim/tesseract/wiki  \n"
                    "   Add its folder to your system PATH  \n"
                    "2. Poppler: https://github.com/oschwartz10612/poppler-windows/releases  \n"
                    "   Add poppler/bin to your system PATH  \n"
                    "3. In the BaseTruth folder run:  \n"
                    "   `.venv\\Scripts\\pip install pytesseract pdf2image`  \n"
                    "4. Re-scan the document."
                )
            elif is_image_only:
                st.warning(
                    "**Image-only PDF** -- no embedded text found after OCR attempt.  "
                    "PDF metadata forensics ran in full.  "
                    f"Reason: {fallback_reason or 'unknown'}."
                )
            else:
                st.warning(
                    "**Partial scan** -- LiteParse could not process this document "
                    f"({fallback_reason or 'reason unknown'}).  "
                    "BaseTruth used text-extraction fallback (PyMuPDF).  "
                    "PDF metadata forensics and structural checks ran in full.  "
                    "To enable full LiteParse scans: install ImageMagick from "
                    "https://imagemagick.org/script/download.php#windows"
                )

        st.divider()
        st.subheader("Scan Result")
        _render_report_summary(report)
        if artifacts.get("verification_json_path"):
            st.caption(f"Report saved to: {artifacts['verification_json_path']}")

        scan_saved = st.session_state.get("scan_saved", False)
        scan_saved_ref = st.session_state.get("scan_saved_ref")
        if scan_saved:
            st.success(f"Saved to database — Entity: **{scan_saved_ref or 'unlinked'}**")
        elif _DB_IMPORTS_OK and _db_available_cached():
            if st.button("💾 Save to Database", key="scan_save_btn", use_container_width=True):
                with st.spinner("Saving scan result to database..."):
                    pdf_bytes = None
                    pdf_path_str = artifacts.get("pdf_report_path", "")
                    if pdf_path_str and Path(pdf_path_str).exists():
                        pdf_bytes = Path(pdf_path_str).read_bytes()

                    saved = save_scan_to_db(
                        report,
                        pdf_bytes=pdf_bytes,
                        forced_entity_ref=st.session_state.get("scan_pending_forced_ref") or None,
                        extra_identity=st.session_state.get("scan_pending_extra_identity") or None,
                        layered_screen_name="Scan Document",
                    )
                    if saved:
                        source_path_str = st.session_state.get("scan_pending_source_path", "")
                        if source_path_str:
                            source_path = Path(source_path_str)
                            if source_path.exists() and saved.get("entity_ref"):
                                minio_upload(
                                    f"{saved['entity_ref']}/{source_path.name}",
                                    source_path.read_bytes(),
                                    _scan_source_content_type(source_path),
                                )
                        report["_entity_ref"] = saved.get("entity_ref")
                        st.session_state["scan_pending_report"] = report
                        st.session_state["scan_saved"] = True
                        st.session_state["scan_saved_ref"] = saved.get("entity_ref")
                        st.rerun()
                    else:
                        st.error(
                            "Scan completed but could not be saved to the database. "
                            "Check the Logs screen for details."
                        )
        else:
            st.info(
                "Database is offline — scan saved to disk only. Start PostgreSQL to persist results."
            )

        stem = Path(report.get("source", {}).get("name", "report")).stem
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                "⬇ Download JSON report",
                data=json.dumps(report, indent=2, ensure_ascii=False),
                file_name=f"{stem}_verification.json",
                mime="application/json",
                use_container_width=True,
            )
        with col_dl2:
            pdf_path_str = artifacts.get("pdf_report_path", "")
            if pdf_path_str and Path(pdf_path_str).exists():
                st.download_button(
                    "⬇ Download PDF report",
                    data=Path(pdf_path_str).read_bytes(),
                    file_name=f"{stem}_report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            else:
                st.button(
                    "PDF report (generating...)",
                    disabled=True,
                    use_container_width=True,
                )
