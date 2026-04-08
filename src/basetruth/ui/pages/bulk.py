"""Bulk scan page — forensics-only, no OCR pipeline.

Each uploaded document is analysed through the 11-layer forensic engine
(image_forensics_detect.py). Results are shown immediately after scanning
and saved to the database on request.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

from basetruth.logger import get_logger
from basetruth.ui.components import (
    _DB_IMPORTS_OK,
    _db_available_cached,
    _page_title,
    _render_entity_link_widget,
    _save_uploaded_files,
    minio_upload,
    save_scan_to_db,
)

log = get_logger(__name__)

# Supported image file extensions that the forensic engine can analyse directly.
_IMAGE_EXTS: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# Filename-keyword to document type mapping for labelling scans without OCR.
_FILENAME_DOC_TYPES: List[tuple[str, str]] = [
    ("payslip", "payslip"),
    ("salary", "payslip"),
    ("bank", "bank_statement"),
    ("statement", "bank_statement"),
    ("pan", "pan_card"),
    ("aadhar", "aadhaar"),
    ("aadhaar", "aadhaar"),
    ("passport", "passport"),
    ("form16", "form16"),
    ("form_16", "form16"),
    ("offer", "offer_letter"),
    ("appointment", "offer_letter"),
    ("employment", "employment_letter"),
    ("increment", "increment_letter"),
    ("gift", "gift_letter"),
    ("utility", "utility_bill"),
    ("electricity", "utility_bill"),
    ("property", "property_agreement"),
    ("agreement", "property_agreement"),
    ("degree", "degree_certificate"),
    ("marksheet", "marksheet"),
    ("certificate", "certificate"),
    ("photo", "photograph"),
    ("photograph", "photograph"),
    ("signature", "signature"),
    ("cheque", "cancelled_cheque"),
    ("hospital", "hospital_bill"),
    ("invoice", "invoice"),
]


def _guess_doc_type(filename: str) -> str:
    """Guess document type from filename keywords.

    We no longer run OCR on bulk uploads, so we cannot read the document content
    to classify it.  Instead we look for keywords in the filename (e.g. 'payslip',
    'pan', 'bank') and map them to standard document type strings.  This is used
    only for labelling the scan row in the database — the actual forensic analysis
    does not depend on the document type.
    Falls back to 'document' if nothing matches.
    """
    lower = filename.lower()
    for keyword, doc_type in _FILENAME_DOC_TYPES:
        if keyword in lower:
            return doc_type
    return "document"


def _sha256_of_file(path: Path) -> str:
    """Compute SHA-256 hash of a file in 64 KB chunks.

    We hash in chunks (rather than reading the whole file at once) so that
    large multi-MB scans don't spike memory usage.  The hash is stored in
    the database as source_sha256 so duplicate uploads can be detected.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _bulk_source_content_type(path: Path) -> str:
    """Choose the correct MIME type when uploading a source file to MinIO."""
    if path.suffix.lower() == ".pdf":
        return "application/pdf"
    if path.suffix.lower() in _IMAGE_EXTS:
        return "image/" + path.suffix.lower().lstrip(".")
    return "application/octet-stream"


def _cleanup_bulk_temp_dir(temp_dir_str: str) -> None:
    """Delete the temporary directory created for uploaded files.

    We write uploaded files to a temp directory so the forensic engine can
    open them as real file paths.  This helper removes the directory after
    the user clicks Save, freeing up disk space.  The 'bt_bulk_' name prefix
    check is a safety guard so we never accidentally delete the wrong directory.
    Called after a successful save so large files are not left on disk.
    """
    if not temp_dir_str:
        return
    temp_dir = Path(temp_dir_str)
    if not temp_dir.exists() or not temp_dir.name.startswith("bt_bulk_"):
        return
    shutil.rmtree(temp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Forensics card rendering
# ---------------------------------------------------------------------------

_FORENSIC_VERDICT_BADGE: Dict[str, str] = {
    "ORIGINAL": "🟢 ORIGINAL",
    "UNCERTAIN": "🟡 UNCERTAIN",
    "LIKELY TAMPERED": "🟠 LIKELY TAMPERED",
    "TAMPERED": "🔴 TAMPERED",
}

_LAYER_STATUS_ICON: Dict[str, str] = {
    "CLEAN": "✅",
    "SUSPICIOUS": "⚠️",
    "N/A": "➖",
    "ERROR": "❓",
}


def _render_forensics_detail(forensics: Dict[str, Any], fname: str) -> None:
    """Render a full forensics breakdown for one scanned file.

    Shows the overall verdict + score at the top, an evidence list, and an
    expandable 11-layer breakdown. Same layout as the Scans approval screen.
    """
    summary = forensics.get("scan_summary", {})
    verdict = summary.get("forensic_verdict", "—")
    score = summary.get("forgery_score_0_100", 0)
    explanation = summary.get("overall_explanation", "")
    evidence = summary.get("evidence", [])
    layers = forensics.get("layers", {})

    col1, col2, col3 = st.columns(3)
    col1.metric("Forensic Verdict", _FORENSIC_VERDICT_BADGE.get(verdict, verdict))
    col2.metric("Forgery Score", f"{score:.1f} / 100")
    col3.metric("File", summary.get("format", "—"))

    if explanation:
        st.info(explanation)

    if evidence:
        with st.expander("🔍 Evidence", expanded=False):
            for ev in evidence:
                st.markdown(f"- {ev}")

    if layers:
        with st.expander("🧪 All 11 Forensic Layers", expanded=False):
            for layer_key in sorted(layers.keys()):
                layer = layers[layer_key]
                icon = _LAYER_STATUS_ICON.get(layer.get("status", ""), "➖")
                name = layer.get("name", layer_key)
                plain = layer.get("plain_english", "")
                status = layer.get("status", "N/A")
                col_a, col_b = st.columns([1, 4])
                col_a.markdown(f"**{icon} {name}**  \n`{status}`")
                col_b.caption(plain or "—")

    with st.expander("📥 Download raw JSON", expanded=False):
        st.download_button(
            "⬇ forensics JSON",
            data=json.dumps(forensics, indent=2, ensure_ascii=False),
            file_name=f"{Path(fname).stem}_forensics.json",
            mime="application/json",
            key=f"dl_json_{fname}",
        )


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------


def _page_bulk(_service: Any = None) -> None:
    """Render the Bulk Scan page.

    _service is kept for backwards compatibility with app.py call signature
    but is no longer used. Each uploaded file is first classified by Gemma4
    (via Ollama) to determine the document type and whether it is an image-
    based scan or a digitally-created PDF. Based on the classification:

    - Image files (.jpg, .png, …) → 11-layer image forensics via run_forensics().
    - PDFs identified as scanned/image-based → run_forensics_on_pdf().
    - PDFs identified as structured/digital (payslip, offer letter, etc.) →
      placeholder result (structured-PDF forensics coming in a future release).

    When Ollama is unavailable Gemma4 classification is skipped and the legacy
    filename-based document type guessing is used as fallback.
    """
    st.markdown(_page_title("📦", "Bulk Scan"), unsafe_allow_html=True)

    with st.expander("ℹ️ How to use this screen", expanded=False):
        st.markdown(
            """
**Bulk Scan** lets you run the 11-layer forensic engine on an entire document folder at once.

1. Upload all the applicant's documents (payslips, PAN, Aadhaar, bank statements, etc.).
2. Expand **"Associate documents with a person"** and enter the applicant's PAN, email, or phone
   so every document in this batch links to the same profile in Document Intelligence.
3. Hit **Run bulk scan**. Gemma4 first classifies each document to determine its type and
   whether it is a scanned image or a digital PDF. Then:
   - **Image files** and **scanned PDFs** are checked for tampering using: ELA, metadata,
     noise, DCT, clone detection, colour anomaly, edge analysis, saturation, font consistency,
     AI artefact detection, and file entropy.
   - **Digital PDFs** (payslips, offer letters, form16, etc.) are flagged with a placeholder —
     structured-PDF forensic analysis will be available in a future release.
4. Review the forensic verdict for each file, then click **Save to Database** to persist.
"""
        )

    uploads = st.file_uploader(
        "Upload multiple documents",
        type=None,
        accept_multiple_files=True,
        key="bulk_uploads",
    )
    folder_input = st.text_input(
        "Or scan all files from a folder on disk",
        key="bulk_folder_input",
    )

    # Clear stale results when the user uploads a new set of files
    _current_upload_names: tuple = tuple(f.name for f in (uploads or []))
    _last_upload_names: tuple = tuple(st.session_state.get("_bt_last_upload_names", ()))
    if _current_upload_names != _last_upload_names:
        st.session_state["_bt_last_upload_names"] = _current_upload_names
        if _current_upload_names:
            for _k in ("bt_bulk_reports", "bt_bulk_errors", "bt_bulk_saved", "bt_bulk_saved_ref"):
                st.session_state.pop(_k, None)

    # Mark the page as having unsaved uploads (used by navigate-away guard)
    if not st.session_state.get("bt_bulk_saved", False) and (
        uploads or folder_input.strip() or st.session_state.get("bt_bulk_reports")
    ):
        st.session_state["_bulk_has_uploads"] = True

    bulk_forced_ref, bulk_extra_identity = _render_entity_link_widget("bulk")

    if st.button("Run bulk scan →", type="primary"):
        for _k in ("bt_bulk_reports", "bt_bulk_errors", "bt_bulk_saved", "bt_bulk_saved_ref"):
            st.session_state.pop(_k, None)
        log.info(
            "Bulk scan started",
            extra={
                "uploaded_count": len(uploads or []),
                "has_folder_input": bool(folder_input.strip()),
                "has_forced_entity": bool(bulk_forced_ref),
            },
        )
        with st.spinner("Preparing files…"):
            paths: List[Path] = []
            stale_temp = str(st.session_state.get("bt_bulk_temp_dir") or "")
            if stale_temp:
                _cleanup_bulk_temp_dir(stale_temp)
                st.session_state.pop("bt_bulk_temp_dir", None)
            if uploads:
                temp_dir = Path(tempfile.mkdtemp(prefix="bt_bulk_"))
                st.session_state["bt_bulk_temp_dir"] = str(temp_dir)
                paths.extend(_save_uploaded_files(list(uploads), temp_dir))
            if folder_input.strip():
                folder_path = Path(folder_input.strip())
                if folder_path.is_dir():
                    paths.extend([
                        p for p in sorted(folder_path.rglob("*"))
                        if p.suffix.lower() in _IMAGE_EXTS or p.suffix.lower() == ".pdf"
                    ])
                else:
                    st.warning(f"Folder not found: {folder_input.strip()}")
            if not paths:
                log.warning("Bulk scan blocked — no files provided")
                st.warning("Provide uploaded files or a folder path.")
                st.stop()

        _new_reports: List[Dict[str, Any]] = []
        _new_errors: List[str] = []
        _scan_status = st.empty()
        _prog = st.progress(0)

        # ── Phase 1: Gemma4 batch classification ─────────────────────────────
        # Ask Gemma4 to classify every document in one Ollama call so we know
        # whether each PDF is a scanned/image-based file (→ run image forensics)
        # or a digitally-created structured document like a payslip or offer
        # letter (→ forensics placeholder for now).
        _classify_map: Dict[str, Dict[str, Any]] = {}
        _classification_info = st.empty()
        _classification_info.info(f"🧠 Classifying {len(paths)} document(s) with Gemma4…")
        try:
            from basetruth.integrations.pdf import get_document_image_bytes  # noqa: PLC0415
            from basetruth.integrations.ollama import classify_documents_batch  # noqa: PLC0415

            # Rasterise every file to JPEG so Gemma4 can inspect the visual content.
            # For images this is a direct resize; for PDFs page-1 is rendered to JPEG.
            _preview_bytes: List[bytes] = [
                get_document_image_bytes(p) or b"" for p in paths
            ]
            _filenames = [p.name for p in paths]

            # One Ollama call classifies all documents — returns list of dicts
            # with keys: filename, document_type, is_image_based, confidence.
            _classifications = classify_documents_batch(_preview_bytes, _filenames)
            _classify_map = {c["filename"]: c for c in _classifications}
            log.info(
                "Bulk scan: Gemma4 classification complete",
                extra={"file_count": len(_classifications)},
            )
        except Exception as _ce:  # noqa: BLE001
            # Gemma4 / Ollama unavailable — fall back to filename-based type guessing
            # and treat all PDFs as image-based (existing behaviour).
            log.warning(
                "Bulk scan: Gemma4 classification skipped (Ollama unavailable?): %s", _ce
            )
        _classification_info.empty()

        # ── Phase 2: Per-file forensic analysis ──────────────────────────────
        for i, p in enumerate(paths):
            # Retrieve Gemma4 classification for this file (may be absent if Ollama is down)
            _clsf = _classify_map.get(p.name, {})
            _ai_doc_type = _clsf.get("document_type", "") or ""
            _is_image_based: bool = _clsf.get("is_image_based", True)
            _classify_confidence: float = float(_clsf.get("confidence", 0.0))

            # Use Gemma4's document_type when it is confident; otherwise fall
            # back to guessing from the filename (original behaviour).
            _final_doc_type = (
                _ai_doc_type
                if _ai_doc_type and _ai_doc_type != "unknown" and _classify_confidence > 0.4
                else _guess_doc_type(p.name)
            )

            _scan_status.info(f"🔬 Forensic scan {i + 1}/{len(paths)}: **{p.name}**…")
            try:
                # Import the forensic engine lazily to avoid a heavy top-level import
                from basetruth.analysis.image_forensics_detect import (  # noqa: PLC0415
                    run_forensics,
                    run_forensics_on_pdf,
                )

                if p.suffix.lower() in _IMAGE_EXTS:
                    # Raw image file — always run image forensics directly
                    forensics_result = run_forensics(str(p))

                elif p.suffix.lower() == ".pdf" and not _is_image_based and _classify_confidence > 0.5:
                    # PDF classified as a digitally-created structured document
                    # (payslip, offer letter, form16, etc.) by Gemma4.
                    # Deep forensic analysis for structured PDFs is not yet implemented —
                    # record a placeholder so the scan is saved and visible without crashing.
                    log.info(
                        "Bulk scan: structured PDF — forensics placeholder",
                        extra={
                            "file": p.name,
                            "doc_type": _final_doc_type,
                            "confidence": _classify_confidence,
                        },
                    )
                    forensics_result = {
                        "scan_summary": {
                            "forensic_verdict": "N/A",
                            "forgery_score_0_100": 0,
                            "overall_explanation": (
                                f"Structured PDF analysis is not yet available for "
                                f"'{_final_doc_type}' documents. Deep forensic "
                                "inspection of digitally-created PDFs will be added "
                                "in a future release."
                            ),
                            "format": "PDF (structured / digital)",
                            "evidence": [
                                "Gemma4 classified this document as a digitally-created PDF.",
                                "Image-layer forensics (ELA, clone, noise) are not applicable "
                                "to digitally-created PDFs and will be skipped.",
                            ],
                        },
                        "layers": {},
                        "_placeholder": True,
                    }

                else:
                    # PDF (scanned/image-based, or Gemma4 was not confident enough) —
                    # render page 1 and run the full 11-layer image forensic pipeline.
                    forensics_result = run_forensics_on_pdf(str(p))

                _fsummary = forensics_result.get("scan_summary", {})
                log.info(
                    "Bulk scan: forensics complete",
                    extra={
                        "file": p.name,
                        "verdict": _fsummary.get("forensic_verdict", ""),
                        "score": _fsummary.get("forgery_score_0_100", 0),
                        "doc_type": _final_doc_type,
                        "is_image_based": _is_image_based,
                    },
                )
                # Build the minimal report dict that save_scan_to_db expects.
                # source → file identity; document_type → Gemma4 classification or
                # filename guess; _layered_forensics → full 11-layer result.
                report: Dict[str, Any] = {
                    "source": {
                        "name": p.name,
                        "path": str(p),
                        "sha256": _sha256_of_file(p),
                    },
                    "document_type": _final_doc_type,
                    "_layered_forensics": forensics_result,
                    "_gemma4_classification": _clsf or None,
                }
                _new_reports.append(report)

            except Exception as exc:  # noqa: BLE001
                log.error("Bulk scan document failed", extra={"path": str(p), "error": str(exc)}, exc_info=True)
                _new_errors.append(f"{p.name}: {exc}")
                _scan_status.error(f"❌ {i + 1}/{len(paths)}: {p.name} — {exc}")

            _prog.progress((i + 1) / len(paths))

        _scan_status.empty()
        _prog.empty()

        st.session_state["bt_bulk_reports"] = _new_reports
        st.session_state["bt_bulk_errors"] = _new_errors
        st.session_state["bt_bulk_source_paths"] = [str(p) for p in paths]
        st.session_state["bt_bulk_forced_ref"] = bulk_forced_ref
        st.session_state["bt_bulk_extra_identity"] = bulk_extra_identity
        st.session_state["bt_bulk_saved"] = False
        st.session_state["_bulk_has_uploads"] = True
        st.session_state.pop("bt_bulk_saved_ref", None)
        log.info(
            "Bulk scan completed",
            extra={"report_count": len(_new_reports), "error_count": len(_new_errors)},
        )

    if "bt_bulk_reports" not in st.session_state:
        return

    reports: List[Dict[str, Any]] = st.session_state["bt_bulk_reports"]
    errors: List[str] = st.session_state["bt_bulk_errors"]

    st.success(f"Scanned {len(reports)} document(s).")
    if not st.session_state.get("bt_bulk_saved"):
        st.info("Review results below, then click **Save to Database** at the bottom.")

    if errors:
        with st.expander(f"{len(errors)} document(s) had errors"):
            for err in errors:
                st.error(err)

    # ── Per-document forensic result cards ───────────────────────────────────
    st.subheader("Forensic Results")
    for idx, r in enumerate(reports):
        fname = r.get("source", {}).get("name", f"document-{idx + 1}")
        doc_type = r.get("document_type", "document")
        forensics = r.get("_layered_forensics") or {}
        _fsummary = forensics.get("scan_summary", {})
        _verdict = _fsummary.get("forensic_verdict", "—")
        _score = _fsummary.get("forgery_score_0_100")
        _score_str = f"{_score:.0f}/100" if _score is not None else "—"
        _icon = {
            "ORIGINAL": "✅", "UNCERTAIN": "🟡", "LIKELY TAMPERED": "🟠", "TAMPERED": "🔴",
        }.get(_verdict, "📄")

        with st.expander(
            f"{_icon} {fname}  —  {_FORENSIC_VERDICT_BADGE.get(_verdict, _verdict)}"
            f"  |  Score: {_score_str}  |  Type: {doc_type}",
            expanded=False,
        ):
            if forensics:
                _render_forensics_detail(forensics, fname)
            else:
                st.warning("No forensics data for this file.")

    # ── Save to Database ──────────────────────────────────────────────────────
    st.divider()
    if st.session_state.get("bt_bulk_saved"):
        st.success(
            f"Saved to database — Entity: **{st.session_state.get('bt_bulk_saved_ref') or 'unlinked'}**"
        )
    elif _DB_IMPORTS_OK and _db_available_cached():
        st.info("Click **Save to Database** to persist this batch.")
        if st.button("💾 Save to Database", key="bulk_save_btn", use_container_width=True):
            with st.spinner("Saving to database…"):
                save_errors: List[str] = []
                source_paths = [
                    Path(p_str) for p_str in st.session_state.get("bt_bulk_source_paths", [])
                ]
                batch_entity_ref: str | None = st.session_state.get("bt_bulk_forced_ref") or None
                extra_identity = st.session_state.get("bt_bulk_extra_identity") or None
                log.info(
                    "Bulk save started",
                    extra={"report_count": len(reports), "forced_entity_ref": batch_entity_ref or ""},
                )

                for idx, report in enumerate(reports):
                    saved = save_scan_to_db(
                        report,
                        forced_entity_ref=batch_entity_ref,
                        extra_identity=extra_identity,
                        layered_screen_name="Bulk Scan",
                    )
                    if not saved:
                        src_name = report.get("source", {}).get("name", f"document-{idx + 1}")
                        save_errors.append(src_name)
                        log.error("Bulk save failed for report", extra={"source_name": src_name})
                        continue

                    resolved_ref = saved.get("entity_ref")
                    if resolved_ref and batch_entity_ref is None:
                        batch_entity_ref = resolved_ref

                    # Upload the original source file to MinIO under {entity_ref}/{filename}
                    # so it can be previewed later on the Scans and Document Intelligence screens.
                    if idx < len(source_paths):
                        source_path = source_paths[idx]
                        if resolved_ref and source_path.exists():
                            minio_upload(
                                f"{resolved_ref}/{source_path.name}",
                                source_path.read_bytes(),
                                _bulk_source_content_type(source_path),
                            )

                st.session_state["bt_bulk_reports"] = reports

                if save_errors:
                    log.warning(
                        "Bulk save partially failed",
                        extra={"failed_documents": save_errors, "entity_ref": batch_entity_ref or ""},
                    )
                    st.error(
                        "⚠️ Some documents could not be saved: "
                        + ", ".join(save_errors)
                        + ". Check the Logs screen for details."
                    )
                else:
                    temp_dir_str = str(st.session_state.get("bt_bulk_temp_dir") or "")
                    _cleanup_bulk_temp_dir(temp_dir_str)
                    st.session_state.pop("bt_bulk_temp_dir", None)
                    st.session_state["bt_bulk_saved"] = True
                    st.session_state["bt_bulk_saved_ref"] = batch_entity_ref
                    st.session_state.pop("_bulk_has_uploads", None)
                    log.info(
                        "Bulk save completed",
                        extra={"entity_ref": batch_entity_ref or "", "report_count": len(reports)},
                    )
                    st.rerun()
    else:
        st.warning("Database is offline — connect PostgreSQL to save results.")
