"""Reusable helpers for forensic document analysis in UI and API flows.

This module centralizes forensic-routing logic so screens can reuse the same
implementation (Single Responsibility and DRY):
- Decide whether a file is image-based or structured
- Route to image or PDF forensic engine
- Build a normalized response payload for UI/API consumers
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict

from basetruth.logger import get_logger

log = get_logger(__name__)

# Supported image file extensions that can be analysed directly.
_IMAGE_EXTS: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}



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


def _visual_clues_unavailable() -> dict:
    """Return a safe empty payload when the visual detective call cannot run."""
    return {
        "_unavailable": True,
        "document_type": "",
        "findings": [],
        "overall_assessment": "",
        "no_clues_found": True,
    }


def generate_visual_clues(file_bytes: bytes, filename: str) -> dict:
    """Call Gemma4 once to classify the document type AND hunt for visual fraud clues.

    Combining both tasks into a single LLM call avoids a second image-upload
    round-trip and gives the model full context about what kind of document it
    is while it scans for anomalies.

    The model acts as a "Logical Detective": given the document image, it:
      1. Names the document type (payslip, bank statement, PAN card, etc.)
      2. Scans every region for visual anomalies a trained fraud examiner would
         catch — font mismatches, cut-and-paste halos, colour patches, misaligned
         fields, irregular stamps, date/number formatting breaks, etc.

    Returns a dict with:
      document_type       — Gemma4's classification string
      findings            — list of {area, clue, suspicion_level, reason} dicts
      overall_assessment  — 2–3 sentence plain-English detective summary
      no_clues_found      — True when nothing suspicious was spotted
      _unavailable        — True when Ollama is offline or the call failed
    """
    system_prompt = (
        "You are a senior document fraud investigator acting as a Logical Detective. "
        "You examine document images carefully and spot visual anomalies that indicate "
        "tampering, forgery, or manipulation — the kind a skilled human examiner would "
        "notice when holding the document under a good light. "
        "Think step-by-step like Sherlock Holmes examining evidence. "
        "Be precise: only report things you can actually see in the image. "
        "Do not hallucinate clues that are not visible."
    )

    user_prompt = (
        "Examine this document image carefully.\n\n"
        "Task 1 — Identify the document type (e.g. payslip, bank statement, "
        "Aadhaar card, PAN card, marksheet, offer letter, degree certificate, etc.).\n\n"
        "Task 2 — Inspect every part of the document as a fraud investigator. "
        "Look specifically for:\n"
        "  \u2022 Font inconsistencies \u2014 weight, size, or family differs from surrounding text\n"
        "  \u2022 Mis-aligned text or numbers that should line up with their column/row\n"
        "  \u2022 Colour or brightness patches \u2014 areas that look differently lit or shaded "
        "compared to the rest of the page\n"
        "  \u2022 Cut-and-paste artefacts \u2014 hard edges, blurring, or halos around "
        "numbers or text blocks\n"
        "  \u2022 Seal or stamp anomalies \u2014 pixelated, incomplete, or oddly coloured stamps\n"
        "  \u2022 Signature irregularities \u2014 digital-looking, traced, or inconsistent pressure\n"
        "  \u2022 Date or number formatting breaks \u2014 different digit styles in the same field\n"
        "  \u2022 Watermark, letterhead, or template anomalies\n"
        "  \u2022 Any area where the background texture or paper tone suddenly changes\n\n"
        "Return your answer as a JSON object with EXACTLY this structure "
        "(no markdown fences, no prose \u2014 raw JSON only):\n"
        "{\n"
        '  "document_type": "<identified type>",\n'
        '  "findings": [\n'
        "    {\n"
        '      "area": "<where in the document, e.g. Salary row, Date field>",\n'
        '      "clue": "<what you visually observed>",\n'
        '      "suspicion_level": "HIGH",\n'
        '      "reason": "<why this is suspicious in one sentence>"\n'
        "    }\n"
        "  ],\n"
        '  "overall_assessment": "<2-3 sentence plain-English detective summary>",\n'
        '  "no_clues_found": false\n'
        "}\n\n"
        'suspicion_level must be "HIGH", "MEDIUM", or "LOW".\n'
        "If you find nothing suspicious: set findings to [] and no_clues_found to true."
    )

    log.info(
        "forensics_utils: generate_visual_clues — sending document to Gemma4",
        extra={"doc_filename": filename},
    )

    try:
        import io as _io  # noqa: PLC0415
        import re as _re  # noqa: PLC0415
        from PIL import Image as _PIL  # noqa: PLC0415
        from basetruth.integrations.ollama import _route_vlm_chat  # noqa: PLC0415

        # Convert the document to a JPEG so Gemma4 can read it regardless of
        # input format (PNG, TIFF, BMP, WebP, or PDF page 1).
        try:
            img = _PIL.open(_io.BytesIO(file_bytes)).convert("RGB")
        except Exception:  # noqa: BLE001
            # Treat as a PDF and render page 1 at 2× resolution for readability.
            import fitz  # noqa: PLC0415
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            page = doc.load_page(0)
            pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
            img = _PIL.frombytes("RGB", (pix.width, pix.height), pix.samples)
            doc.close()

        # Resize large images so they fit within Gemma4's context window.
        w, h = img.size
        max_dim = 2048
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), _PIL.LANCZOS)

        buf = _io.BytesIO()
        img.save(buf, format="JPEG", quality=92)
        image_bytes = buf.getvalue()

        content, _engine, _model, _base = _route_vlm_chat(
            system_prompt,
            user_prompt,
            [image_bytes],
            feature="document_extraction",
        )
        content = (content or "").strip()
        if not content:
            return _visual_clues_unavailable()

        # Strip markdown code fences that some models wrap around their JSON reply.
        content = _re.sub(r"^```(?:json)?\s*", "", content, flags=_re.MULTILINE)
        content = _re.sub(r"\s*```$", "", content, flags=_re.MULTILINE).strip()

        parsed = json.loads(content)
        log.info(
            "forensics_utils: visual_clues received",
            extra={
                "doc_filename": filename,
                "findings_count": len(parsed.get("findings", [])),
                "no_clues_found": parsed.get("no_clues_found", False),
            },
        )
        return parsed

    except Exception as exc:  # noqa: BLE001
        log.warning(
            "forensics_utils: generate_visual_clues failed — %s",
            exc,
        )
        return _visual_clues_unavailable()


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

            # Gemma4 will provide the document type (and visual fraud clues) in
            # a single combined call below.  Default to "document" for the rare
            # case where Ollama is offline.
            document_type = "document"

            # Combined call: Gemma4 identifies the document type AND scans the
            # image as a "Logical Detective" looking for visual fraud clues in one
            # round-trip.  This avoids paying the image-upload cost twice.
            visual_clues = generate_visual_clues(file_bytes, filename)
            llm_type = str(visual_clues.get("document_type", "") or "").strip()
            if llm_type and llm_type.lower() not in ("", "unknown", "generic", "document"):
                document_type = llm_type
                log.debug(
                    "forensics_utils: Gemma4 document_type accepted",
                    extra={"doc_type": document_type, "doc_filename": filename},
                )

            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(file_bytes)
                temp_path = Path(tmp.name)

            # Route to the right forensic engine based on file nature.
            is_image_file = temp_path.suffix.lower() in _IMAGE_EXTS
            if is_image_file:
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
                # feature_contributions is a {feature_name: shap_value} dict when ML
                # scoring ran, or None when the heuristic was used / SHAP failed.
                "feature_contributions": summary.get("feature_contributions"),
                # Gemma4 visual detective: document_type + fraud clue findings.
                # Kept out of the DB — never persisted, forensic-scan-only.
                "visual_clues": visual_clues,
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
