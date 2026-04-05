"""Helpers for storing consistent upload authenticity evidence."""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from basetruth.analysis.image_forensics import analyse_image

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _status_label(passed: Optional[bool], *, na_label: str = "INFO") -> str:
    if passed is True:
        return "PASS"
    if passed is False:
        return "FAIL"
    return na_label


def build_format_check(message: str, passed: Optional[bool]) -> Dict[str, Any]:
    """Build the shared Layer 1 payload."""
    return {
        "title": "Layer 1 - Format / Structural Check",
        "passed": passed,
        "status": _status_label(passed, na_label="INFO"),
        "message": message,
    }


def analyse_upload_authenticity(
    file_bytes: bytes | None,
    filename: str,
    *,
    format_check: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return Layer 1 and Layer 4 authenticity evidence for an uploaded file."""
    checks = [
        format_check
        or build_format_check(
            f"{Path(filename or 'uploaded_file').name} was uploaded successfully.",
            True if filename else None,
        )
    ]
    payload: Dict[str, Any] = {
        "filename": filename or "",
        "checks": checks,
    }

    suffix = Path(filename or "uploaded_file").suffix.lower() or ".jpg"
    if not file_bytes:
        checks.append(
            {
                "title": "Layer 4 - Image Tampering (ELA)",
                "passed": None,
                "status": "INCONCLUSIVE",
                "message": "No uploaded file bytes were available for image tampering analysis.",
            }
        )
        return payload

    if suffix not in _IMAGE_SUFFIXES:
        checks.append(
            {
                "title": "Layer 4 - Image Tampering (ELA)",
                "passed": None,
                "status": "N/A",
                "message": (
                    "ELA is only available for image uploads. Native PDFs are validated through "
                    "the scan pipeline's structure and tamper signals instead."
                ),
            }
        )
        return payload

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
            handle.write(file_bytes)
            temp_path = Path(handle.name)

        image_forensics = analyse_image(temp_path)
        payload["image_forensics_summary"] = {
            "ela_score": image_forensics.get("ela_score"),
            "high_error_frac": image_forensics.get("high_error_frac"),
            "suspicious_tool": image_forensics.get("suspicious_tool"),
        }

        ela_signal = next(
            (
                signal
                for signal in image_forensics.get("signals", [])
                if signal.get("name") == "image_ela_tampering_score"
            ),
            {},
        )
        ela_score = image_forensics.get("ela_score")
        if ela_signal.get("passed") is True:
            status = f"CLEAN ({ela_score:.1f}/100)" if isinstance(ela_score, (int, float)) else "CLEAN"
        elif ela_signal.get("passed") is False:
            status = f"SUSPECT ({ela_score:.1f}/100)" if isinstance(ela_score, (int, float)) else "SUSPECT"
        else:
            status = "INCONCLUSIVE"

        checks.append(
            {
                "title": "Layer 4 - Image Tampering (ELA)",
                "passed": ela_signal.get("passed"),
                "status": status,
                "message": ela_signal.get(
                    "summary",
                    "Image tampering analysis did not return a usable ELA summary.",
                ),
                "details": ela_signal.get("details") or {},
            }
        )
    except Exception as exc:
        checks.append(
            {
                "title": "Layer 4 - Image Tampering (ELA)",
                "passed": None,
                "status": "INCONCLUSIVE",
                "message": f"Image tampering analysis was unavailable: {exc}",
            }
        )
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

    return payload


def build_scan_authenticity_payload(report: Dict[str, Any]) -> Dict[str, Any]:
    """Build Layer 1 and Layer 4 authenticity evidence from a saved scan report."""
    structured_summary = report.get("structured_summary") or {}
    tamper_assessment = report.get("tamper_assessment") or {}
    parse_method = structured_summary.get("parse_method") or "unknown"
    document_info = structured_summary.get("document") or {}
    document_type = document_info.get("type") or report.get("document_type") or "document"

    if structured_summary.get("parse_fallback"):
        format_message = (
            f"{document_type.title()} parsing required fallback mode ({parse_method}). "
            "File structure was still ingested, but the strongest parser path was not available."
        )
        format_passed: Optional[bool] = None
    else:
        format_message = (
            f"{document_type.title()} structure was parsed successfully using {parse_method}."
        )
        format_passed = True

    checks = [build_format_check(format_message, format_passed)]
    image_summary = tamper_assessment.get("image_forensics_summary") or {}
    ela_score = image_summary.get("ela_score")
    if isinstance(ela_score, (int, float)):
        if ela_score >= 20:
            status = f"SUSPECT ({ela_score:.1f}/100)"
            passed = False
            message = "High ELA residuals indicate potential localized image editing artefacts."
        elif ela_score >= 8:
            status = f"REVIEW ({ela_score:.1f}/100)"
            passed = None
            message = "Moderate ELA residuals suggest the image may have been processed and should be reviewed."
        else:
            status = f"CLEAN ({ela_score:.1f}/100)"
            passed = True
            message = "ELA residuals are consistent with an unedited image capture."
    else:
        status = "N/A"
        passed = None
        message = (
            "No image-specific ELA result was stored for this file type. Review the saved tamper signals "
            "below for the strongest available authenticity evidence."
        )

    checks.append(
        {
            "title": "Layer 4 - Image Tampering (ELA)",
            "passed": passed,
            "status": status,
            "message": message,
        }
    )
    return {
        "checks": checks,
        "truth_score": tamper_assessment.get("truth_score"),
        "risk_level": tamper_assessment.get("risk_level"),
        "image_forensics_summary": image_summary,
    }