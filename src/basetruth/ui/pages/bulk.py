"""Bulk scan page — forensics-only, no OCR pipeline.

Each uploaded document is first classified by Gemma4 (via Ollama) to determine
its type and whether it is a scanned image or a digitally-created PDF. Then:

- Image files (.jpg, .png, …) and scanned PDFs:
    Analysed by the 11-layer image forensic engine (image_forensics_detect.py).

- Structured/digital PDFs (payslips, offer letters, bank statements, etc.):
    Analysed by the 11-layer PDF forensic engine (pdf_forensics_detect.py)
    which applies incremental-update detection, metadata analysis, font
    consistency checking, hidden-text detection, suspicious-object detection,
    content structure analysis, digital-signature integrity, page-render ELA,
    embedded-image noise, file entropy, and object/xref integrity.

Results are shown immediately after scanning and saved to the database on request.
"""
from __future__ import annotations

import concurrent.futures as _cf
import hashlib
import json
import multiprocessing as _mp
import os
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


def _resolve_bulk_doc_type(path: Path, classification: Dict[str, Any]) -> str:
    """Choose the best document type label for one bulk-uploaded file.

    Gemma4 classification is preferred when confidence is good enough. When the
    model is unavailable or uncertain, we fall back to filename heuristics so the
    rest of the pipeline can continue without blocking the operator.
    """
    ai_doc_type = str(classification.get("document_type") or "").strip()
    confidence = float(classification.get("confidence") or 0.0)
    if ai_doc_type and ai_doc_type != "unknown" and confidence > 0.4:
        return ai_doc_type
    return _guess_doc_type(path.name)


def _process_bulk_document(path_str: str, classification: Dict[str, Any]) -> Dict[str, Any]:
    """Run one document's forensics and extraction inside a worker process.

    Keeping each document in a separate worker process prevents heavy OCR/CV
    libraries from sharing mutable state across files and lets multiple uploads
    run at the same time after the batch classification step finishes.
    """
    path = Path(path_str)
    final_doc_type = _resolve_bulk_doc_type(path, classification)
    is_image_based = bool(classification.get("is_image_based", True))
    classify_confidence = float(classification.get("confidence") or 0.0)

    log.info(
        f"Bulk Scanning Started: Picked up document '{path.name}' for analysis. Assumed document type is '{final_doc_type}'.",
        extra={
            "file": path.name,
            "doc_type": final_doc_type,
            "is_image_based": is_image_based,
            "classify_confidence": classify_confidence,
        },
    )

    try:
        from basetruth.analysis.image_forensics_detect import (  # noqa: PLC0415
            run_forensics,
            run_forensics_on_pdf,
        )
        from basetruth.analysis.pdf_forensics_detect import (  # noqa: PLC0415
            run_pdf_forensics,
        )

        if path.suffix.lower() in _IMAGE_EXTS:
            log.info(
                f"Forensics Engine Started: Running the 11-layer image forensics pipeline on '{path.name}'.",
                extra={"file": path.name, "doc_type": final_doc_type},
            )
            forensics_result = run_forensics(str(path))
        elif path.suffix.lower() == ".pdf":
            if not is_image_based:
                log.info(
                    f"PDF Forensics Engine Started: Document '{path.name}' looks like a structured digital PDF. Running PDF forensic checks.",
                    extra={
                        "file": path.name,
                        "doc_type": final_doc_type,
                        "confidence": classify_confidence,
                    },
                )
                forensics_result = run_pdf_forensics(str(path))
            else:
                log.info(
                    f"Image-Based PDF Engine Started: Document '{path.name}' is an image inside a PDF. Extracting pages to run image forensics.",
                    extra={"file": path.name, "doc_type": final_doc_type},
                )
                forensics_result = run_forensics_on_pdf(str(path))
        else:
            forensics_result = {}

        _fs = forensics_result.get("scan_summary", {})

        no_extract_types = {
            "photograph", "signature", "cancelled_cheque", "document", "unknown",
        }

        log.info(
            f"Forensics Analysis Complete: Finished analyzing '{path.name}'. Verdict is '{_fs.get('forensic_verdict', 'UNKNOWN')}'.",
            extra={
                "file": path.name,
                "verdict": _fs.get("forensic_verdict", ""),
                "forgery_score": _fs.get("forgery_score_0_100", 0),
            },
        )

        doc_extraction: Dict[str, Any] = {}
        if final_doc_type not in no_extract_types:
            log.info(
                f"Data Extraction Phase: Starting field extraction using AI for '{path.name}'.",
                extra={"file": path.name, "doc_type": final_doc_type},
            )
            try:
                from basetruth.integrations.document_extract import (  # noqa: PLC0415
                    extract_document_fields,
                )

                doc_extraction = extract_document_fields(
                    str(path),
                    doc_type=final_doc_type,
                    filename=path.name,
                )
                log.info(
                    f"Data Extraction Phase: Finished extracting data fields from '{path.name}'. Found {len([k for k, v in doc_extraction.items() if not k.startswith('_') and v])} data fields.",
                    extra={
                        "file": path.name,
                        "fields_extracted": len([k for k, v in doc_extraction.items() if not k.startswith("_") and v]),
                    },
                )
            except Exception as ext_exc:  # noqa: BLE001
                log.warning(
                    f"Bulk scan worker: field extraction failed (non-fatal) — {path.name}: {ext_exc}",
                    extra={"file": path.name, "error": str(ext_exc)},
                )

        report: Dict[str, Any] = {
            "source": {
                "name": path.name,
                "path": str(path),
                "sha256": _sha256_of_file(path),
            },
            "document_type": final_doc_type,
            "_layered_forensics": forensics_result,
            "_gemma4_classification": classification or None,
            "_document_extraction": doc_extraction,
        }
        return {"ok": True, "path": str(path), "report": report}
    except Exception as exc:  # noqa: BLE001
        # Include the filename and exception class in the message itself so that
        # the Log Analyser can show a useful one-liner without needing to expand
        # the full JSON payload.
        log.error(
            f"Bulk Scanning Failed: Something went wrong while analyzing '{path.name}'. Error details: {type(exc).__name__} - {exc}",
            extra={"path": str(path), "file": path.name, "error": str(exc), "exc_type": type(exc).__name__},
            exc_info=True,
        )
        return {"ok": False, "path": str(path), "error": str(exc), "file_name": path.name}


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
      11-layer PDF forensics via run_pdf_forensics() from pdf_forensics_detect.py.

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
   - **Digital PDFs** (payslips, offer letters, form16, bank statements, etc.) are analysed
     using the 11-layer PDF forensic pipeline: incremental-update detection, metadata
     fingerprinting, font consistency, hidden-text detection, suspicious-object detection,
     content structure, digital-signature integrity, page-render ELA, embedded-image noise,
     file entropy, and object/xref integrity.
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

            _classifications = classify_documents_batch(_preview_bytes, _filenames)
            _classify_map = {c["filename"]: c for c in _classifications}
            log.info(
                f"Bulk Scanning Preparation: Successfully classified {len(_classifications)} documents using Gemma4 AI.",
                extra={"file_count": len(_classifications)},
            )
        except Exception as _ce:  # noqa: BLE001
            # Gemma4 / Ollama unavailable — fall back to filename-based type guessing
            # and treat all PDFs as image-based (existing behaviour).
            log.warning(
                "Bulk scan: Gemma4 classification skipped (Ollama unavailable?): %s", _ce
            )
        _classification_info.empty()

        # ── Phase 2: Per-file worker processes ─────────────────────────────
        _scan_status.info(
            f"⚙️ Processing {len(paths)} document(s) in parallel worker process(es)…"
        )
        _reports_by_path: Dict[str, Dict[str, Any]] = {}
        _worker_count = min(len(paths), max(1, os.cpu_count() or 1))
        _mp_context = _mp.get_context("spawn")

        with _cf.ProcessPoolExecutor(
            max_workers=_worker_count,
            mp_context=_mp_context,
            max_tasks_per_child=1,
        ) as _pool:
            _future_map = {
                _pool.submit(_process_bulk_document, str(p), _classify_map.get(p.name, {})): p
                for p in paths
            }
            for i, _future in enumerate(_cf.as_completed(_future_map), start=1):
                p = _future_map[_future]
                try:
                    _result = _future.result()
                except Exception as exc:  # noqa: BLE001
                    # Log with the filename and exception type in the message so the
                    # Log Analyser shows a meaningful one-liner (not just a bare label).
                    log.error(
                        f"Bulk scan failed: {p.name} — {type(exc).__name__}: {exc}",
                        extra={"path": str(p), "file": p.name, "error": str(exc), "exc_type": type(exc).__name__},
                        exc_info=True,
                    )
                    _new_errors.append(f"{p.name}: {exc}")
                    _scan_status.error(f"❌ {i}/{len(paths)}: {p.name} — {exc}")
                    _prog.progress(i / len(paths))
                    continue

                if _result.get("ok"):
                    _report = _result["report"]
                    _reports_by_path[_result["path"]] = _report
                    _fsummary = (_report.get("_layered_forensics") or {}).get("scan_summary", {})
                    # Extract the document extraction result from the report (may be absent for non-bulk paths)
                    _doc_extraction = _report.get("_document_extraction") or {}
                    log.info(
                        f"Bulk Scanning Successful: Completely finished processing '{p.name}'.",
                        extra={
                            "file": p.name,
                            "verdict": _fsummary.get("forensic_verdict", ""),
                            "score": _fsummary.get("forgery_score_0_100", 0),
                            "doc_type": _report.get("document_type", "document"),
                            "unavailable": _doc_extraction.get("_unavailable", False),
                            "error": _doc_extraction.get("error", ""),
                        },
                    )
                    _scan_status.info(f"✅ Completed {i}/{len(paths)}: **{p.name}**")
                else:
                    _error_text = str(_result.get("error") or "unknown error")
                    _new_errors.append(f"{p.name}: {_error_text}")
                    _scan_status.error(f"❌ {i}/{len(paths)}: {p.name} — {_error_text}")

                _prog.progress(i / len(paths))

        _new_reports = [
            _reports_by_path[str(p)]
            for p in paths
            if str(p) in _reports_by_path
        ]

        _scan_status.empty()
        _prog.empty()

        st.session_state["bt_bulk_reports"] = _new_reports
        st.session_state["bt_bulk_errors"] = _new_errors
        st.session_state["bt_bulk_source_paths"] = [
            str((report.get("source") or {}).get("path") or "")
            for report in _new_reports
        ]
        st.session_state["bt_bulk_forced_ref"] = bulk_forced_ref
        st.session_state["bt_bulk_extra_identity"] = bulk_extra_identity
        st.session_state["bt_bulk_saved"] = False
        st.session_state["_bulk_has_uploads"] = True
        log.info(
            f"Bulk Scanning Batch Complete: Processed {len(_new_reports)} documents successfully. {len(_new_errors)} failed.",
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

            # ── Field extraction status ───────────────────────────────────
            # Show the operator whether Gemma4 successfully extracted document
            # fields, or whether extraction was skipped/failed.
            _ext = r.get("_document_extraction") or {}
            _raw_ocr_markdown_path = str(_ext.get("_raw_ocr_markdown_path") or "")
            if _raw_ocr_markdown_path:
                st.caption(f"Raw marksheet OCR markdown: {_raw_ocr_markdown_path}")
            if not _ext:
                # Empty dict — an exception prevented extraction from running at all
                st.warning(
                    "📋 Field extraction did not run for this document. "
                    "Check the Logs screen for details.",
                    icon="⚠️",
                )
            elif _ext.get("_unavailable"):
                # Ollama was offline when extraction was attempted
                st.info(
                        "📋 Field extraction skipped — Gemma4 (Ollama) is not running. "
                    "Start Ollama and re-scan to extract structured fields.",
                    icon="ℹ️",
                )
            elif _ext.get("error"):
                # Extraction ran but returned a hard error
                st.warning(
                    f"📋 Field extraction failed: {_ext.get('error', 'unknown error')}",
                    icon="⚠️",
                )
            else:
                # Gemma4 returned actual data — show a summary of extracted fields
                _ext_display = {
                    k: v for k, v in _ext.items()
                    if not k.startswith("_") and v not in (None, "", [], {})
                }
                if _ext_display:
                    with st.expander("📋 Extracted Fields (Gemma4)", expanded=False):
                        st.json(_ext_display)
                else:
                    # Gemma4 ran but all fields came back null/empty
                    st.info(
                        "📋 Gemma4 ran but could not read clear field values from this document "
                        "(image may be low-resolution or text is not machine-readable).",
                        icon="ℹ️",
                    )

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
                    log.info(
                        f"Bulk Saving Batch Complete: Successfully saved {len(reports)} reports to the database.",
                        extra={"entity_ref": batch_entity_ref or "", "report_count": len(reports)},
                    )
                    st.rerun()
    else:
        st.warning("Database is offline — connect PostgreSQL to save results.")
