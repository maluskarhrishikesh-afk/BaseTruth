"""Reusable helpers for forensic document analysis in UI and API flows.

This module centralizes forensic-routing logic so screens can reuse the same
implementation (Single Responsibility and DRY):
- Decide whether a file is image-based or structured
- Route to image or PDF forensic engine
- Build a normalized response payload for UI/API consumers
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict

from basetruth.logger import get_logger

log = get_logger(__name__)

# Supported image file extensions that can be analysed directly.
_IMAGE_EXTS: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# Lightweight filename heuristic used only for display labels.
_FILENAME_DOC_TYPES: list[tuple[str, str]] = [
    ("payslip", "payslip"),
    ("salary", "payslip"),
    ("bank", "bank_statement"),
    ("statement", "bank_statement"),
    ("pan", "pan_card"),
    ("aadhar", "aadhaar"),
    ("aadhaar", "aadhaar"),
    ("passport", "passport"),
    ("form-16", "form16"),
    ("form16", "form16"),
    ("form_16", "form16"),
    ("offer", "offer_letter"),
    ("appointment", "offer_letter"),
    ("experience", "experience_letter"),
    ("relieving", "relieving_letter"),
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


def generate_honest_review(
    verdict: str,
    score: float,
    evidence: list,
    layers: dict,
    document_type: str,
    filename: str,
) -> str:
    """Ask the LLM to write a brutally honest, plain-English review of the forensic findings.

    The review is written for a non-technical end user — someone who does not
    know what ELA or DCT means.  The tone should be clear and direct: tell the
    user what we found, what it means, and what they should do next.

    Why we call the LLM here (rather than building a template):
    - The LLM can combine evidence signals into a coherent paragraph that reads
      naturally, rather than a robotic list of flags.
    - Template-based text would repeat the same boilerplate regardless of *which*
      layers fired, which makes it less useful when the signals are subtle.

    If the LLM is offline or the call fails we fall back to a rule-based
    summary so the screen never shows an empty review section.
    """
    # Collect only the suspicious layer names for a concise prompt.
    suspicious_layers = [
        layer_data.get("name", key)
        for key, layer_data in layers.items()
        if isinstance(layer_data, dict) and layer_data.get("status") == "SUSPICIOUS"
    ]

    # Build the LLM prompt asking for a plain-English honest review.
    system_prompt = (
        "You are a document fraud expert writing a verdict for a non-technical reviewer. "
        "Explain your findings in simple, plain English — as if you are talking to a "
        "bank employee who has never heard of ELA or metadata forensics. "
        "Be brutally honest and direct. Do NOT use technical jargon. "
        "Write 10 sentences maximum. Do not use bullet points."
    )

    # Summarise the evidence list so the prompt is concise.
    evidence_text = "; ".join(evidence) if evidence else "No specific evidence flags."

    user_prompt = (
        f"Document: {filename} (type: {document_type})\n"
        f"Forensic verdict: {verdict}\n"
        f"Forgery score: {score:.1f} out of 100 (0 = clean, 100 = definitely tampered)\n"
        f"Suspicious signals found: {', '.join(suspicious_layers) if suspicious_layers else 'None'}\n"
        f"Evidence: {evidence_text}\n\n"
        "Write a short, honest, plain-English explanation of what these findings mean. "
        "Tell the reviewer whether they should trust this document or be suspicious, "
        "and what the most important red flag is (if any). "
        "Keep it under 10 sentences. No bullet points, no technical terms."
    )

    try:
        from basetruth.integrations.ollama import _route_vlm_chat  # noqa: PLC0415

        content, _engine, _model, _base = _route_vlm_chat(
            system_prompt,
            user_prompt,
            [],  # no image needed for this text-only review call
            # "document_extraction" is the feature key registered in settings.json
            # under providers.feature_models, which maps to Google gemma-4-31b-it.
            feature="document_extraction",
        )
        review = (content or "").strip()
        if review:
            log.debug(
                "forensics_utils: honest_review generated",
                extra={"doc_filename": filename, "review_len": len(review)},
            )
            return review
    except Exception:  # noqa: BLE001
        # LLM unavailable — fall through to the rule-based fallback below.
        log.debug(
            "forensics_utils: LLM honest_review skipped, using rule-based fallback",
            extra={"doc_filename": filename},
        )

    # ── Rule-based fallback: always produce a meaningful review even without the LLM ──
    verdict_upper = verdict.upper()
    if verdict_upper == "ORIGINAL":
        return (
            f"This document ({filename}) passed all forensic checks. "
            "None of the 11 detection layers found any signs of tampering, "
            "editing, or suspicious modification. "
            "You can treat this document as authentic based on technical analysis."
        )
    if verdict_upper == "UNLIKELY TAMPERED":
        return (
            f"This document ({filename}) looks mostly clean. "
            "One or two minor signals were flagged, but they are not strong enough to "
            "conclude the document was altered. "
            "Treat it with normal caution — minor editing software sometimes leaves traces "
            "even on completely authentic documents."
        )
    if verdict_upper == "UNCERTAIN":
        flag_text = f" The most notable issue is: {evidence[0]}." if evidence else ""
        return (
            f"This document ({filename}) raised some concerns but the evidence is not conclusive.{flag_text} "
            "We recommend a manual review by a human expert before accepting this document. "
            "Do not reject it outright, but do not rely on it without additional verification."
        )
    if verdict_upper == "LIKELY TAMPERED":
        flag_text = f" Key red flag: {evidence[0]}." if evidence else ""
        return (
            f"Warning: this document ({filename}) shows strong signs of tampering.{flag_text} "
            f"Forgery score is {score:.0f}/100 — above the threshold for serious concern. "
            "We strongly recommend rejecting this document and asking the person to provide "
            "a fresh, unedited copy directly from the issuing authority."
        )
    if verdict_upper == "TAMPERED":
        flag_text = f" Primary evidence: {evidence[0]}." if evidence else ""
        return (
            f"This document ({filename}) has been digitally altered.{flag_text} "
            f"The forensic score is {score:.0f}/100 — multiple independent detection layers fired. "
            "This document should be rejected immediately. "
            "Report this to your fraud team and request a certified original from the source."
        )
    # Catch-all for UNAVAILABLE or unknown verdicts.
    return (
        f"Forensic analysis for {filename} could not reach a definitive conclusion. "
        "The forensic engine may have been unable to process this file type fully. "
        "Treat this document with caution and perform a manual review."
    )


class ForensicAnalyzer:
    """Facade class that exposes one forensic-analysis entry point.

    The class encapsulates routing and payload shaping so UI/API callers do not
    need to know which forensic engine to call for each file type.
    """

    @staticmethod
    def infer_is_image_based(file_bytes: bytes, filename: str) -> bool:
        """Return True for scanned images or scanned/image-only PDFs.

        Why this check exists:
        - Structured PDFs should go through the dedicated PDF forensic engine.
        - Scanned/image-based documents should go through the image forensic
          engine to catch pixel-level tampering artefacts.
        """
        suffix = Path(filename).suffix.lower()
        if suffix in _IMAGE_EXTS:
            return True
        if suffix == ".pdf" or file_bytes[:4] == b"%PDF":
            try:
                import fitz  # PyMuPDF  # noqa: PLC0415

                doc = fitz.open(stream=file_bytes, filetype="pdf")
                txt = doc[0].get_text().strip() if doc.page_count > 0 else ""
                doc.close()
                # Low embedded-text count implies scan/image PDF.
                return len(txt) < 200
            except Exception:
                # On parser failure we default to structured to avoid false
                # positive "scan" labels.
                return False
        return False

    @staticmethod
    def guess_document_type(filename: str) -> str:
        """Best-effort document type label for the output payload."""
        lower = filename.lower()
        for keyword, doc_type in _FILENAME_DOC_TYPES:
            if keyword in lower:
                return doc_type
        return "document"

    @staticmethod
    def analyze_document(file_bytes: bytes, filename: str) -> Dict[str, Any]:
        """Run forensic analysis and return a normalized response payload.

        Returns keys suitable for both Streamlit UI and REST responses:
        filename, document_type, is_image_based, forensic_verdict,
        forgery_score_0_100, overall_explanation, evidence, layers,
        and layered_analysis_json.

        The function never raises; failures are returned with an `error` field.
        """
        log.info(
            "forensics_utils: analyze_document called",
            extra={"doc_filename": filename, "size_bytes": len(file_bytes)},
        )

        suffix = Path(filename).suffix or ".pdf"
        temp_path: Path | None = None
        try:
            from basetruth.analysis.image_forensics_detect import (  # noqa: PLC0415
                run_forensics,
                run_forensics_on_pdf,
            )
            from basetruth.analysis.pdf_forensics_detect import run_pdf_forensics  # noqa: PLC0415

            is_image_based = ForensicAnalyzer.infer_is_image_based(file_bytes, filename)

            # Start with filename heuristic; upgrade to LLM classification below.
            # This mirrors the exact approach used on the Scan Document screen.
            document_type = ForensicAnalyzer.guess_document_type(filename)

            # Ask the LLM to classify the document type — only the document_type
            # field is used; full extraction fields are discarded to keep this fast.
            try:
                from basetruth.integrations.document_extract import extract_document_fields  # noqa: PLC0415

                meta = extract_document_fields(file_bytes, doc_type="generic", filename=filename)
                # document_type may live at the top level or inside an extracted_fields envelope.
                llm_type = str(meta.get("document_type", "") or "").strip()
                if llm_type and llm_type.lower() not in ("", "unknown", "generic", "document"):
                    document_type = llm_type
                    log.debug(
                        "forensics_utils: LLM document_type accepted",
                        extra={"doc_type": document_type, "doc_filename": filename},
                    )
            except Exception:  # noqa: BLE001
                # LLM unavailable — filename heuristic is already set, continue normally.
                log.debug(
                    "forensics_utils: LLM classification skipped, using filename heuristic",
                    extra={"doc_filename": filename},
                )

            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(file_bytes)
                temp_path = Path(tmp.name)

            # Route to the right forensic engine based on file nature.
            if temp_path.suffix.lower() in _IMAGE_EXTS:
                layered = run_forensics(str(temp_path))
            elif temp_path.suffix.lower() == ".pdf":
                layered = run_forensics_on_pdf(str(temp_path)) if is_image_based else run_pdf_forensics(str(temp_path))
            else:
                layered = run_forensics(str(temp_path))

            summary = layered.get("scan_summary", {}) if isinstance(layered, dict) else {}
            verdict = str(summary.get("forensic_verdict", "UNAVAILABLE") or "UNAVAILABLE")
            score = float(summary.get("forgery_score_0_100", 0.0) or 0.0)
            scoring_method = str(summary.get("scoring_method", "heuristic") or "heuristic")

            log.info(
                "forensics_utils: analyze_document complete",
                extra={
                    "doc_filename": filename,
                    "doc_type": document_type,
                    "is_image_based": is_image_based,
                    "forensic_verdict": verdict,
                    "forgery_score_0_100": score,
                },
            )

            evidence_list = summary.get("evidence", []) or []
            layers_dict = layered.get("layers", {}) if isinstance(layered, dict) else {}

            # Generate the LLM honest review after forensics so it has the full
            # picture (verdict, score, all evidence, which layers fired).
            honest_review = generate_honest_review(
                verdict=verdict,
                score=score,
                evidence=evidence_list,
                layers=layers_dict,
                document_type=document_type,
                filename=filename,
            )

            return {
                "filename": filename,
                "document_type": document_type,
                "is_image_based": is_image_based,
                "forensic_verdict": verdict,
                "forgery_score_0_100": score,
                "scoring_method": scoring_method,
                "overall_explanation": summary.get("overall_explanation", ""),
                "evidence": evidence_list,
                "honest_review": honest_review,
                "layers": layers_dict,
                "layered_analysis_json": layered,
            }
        except Exception as exc:  # noqa: BLE001
            log.error(
                "forensics_utils: analyze_document failed",
                extra={"doc_filename": filename, "error": str(exc)},
                exc_info=True,
            )
            return {
                "filename": filename,
                "document_type": "document",
                "is_image_based": False,
                "forensic_verdict": "UNAVAILABLE",
                "forgery_score_0_100": 0.0,
                "overall_explanation": "",
                "evidence": [],
                "honest_review": (
                    f"Forensic analysis for {filename} could not be completed due to an error. "
                    "Please check the file format and try again."
                ),
                "layers": {},
                "layered_analysis_json": {},
                "error": str(exc),
            }
        finally:
            if temp_path is not None:
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    pass
