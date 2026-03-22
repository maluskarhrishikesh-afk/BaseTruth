from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from basetruth.analysis.validators import validate_document
from basetruth.models import Signal, signals_to_dict


NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}

NUMBER_SCALES = {
    "hundred": 100,
    "thousand": 1_000,
    "million": 1_000_000,
    "billion": 1_000_000_000,
}

SUSPICIOUS_EDITORS = (
    "photoshop",
    "illustrator",
    "canva",
    "gimp",
    "coreldraw",
    "inkscape",
)


def _parse_numeric_value(value: Any) -> Optional[int]:
    digits = re.sub(r"[^0-9-]", "", str(value or ""))
    if not digits or digits == "-":
        return None
    try:
        return int(digits)
    except ValueError:
        return None


def _parse_number_words(text: str) -> Optional[int]:
    tokens = [token for token in re.split(r"[^a-z]+", str(text or "").lower()) if token and token != "and"]
    if not tokens:
        return None
    total = 0
    current = 0
    matched = False
    for token in tokens:
        if token in NUMBER_WORDS:
            current += NUMBER_WORDS[token]
            matched = True
            continue
        if token == "hundred":
            current = max(current, 1) * NUMBER_SCALES[token]
            matched = True
            continue
        if token in NUMBER_SCALES and NUMBER_SCALES[token] >= 1000:
            current = max(current, 1)
            total += current * NUMBER_SCALES[token]
            current = 0
            matched = True
            continue
        return None
    if not matched:
        return None
    return total + current


def _risk_level(score: int) -> tuple[int, str, str]:
    truth_score = max(0, 100 - score)
    if score >= 60:
        return truth_score, "high", "suspicious_inconsistency"
    if score >= 25:
        return truth_score, "medium", "review_recommended"
    return truth_score, "low", "no_obvious_tampering"


def evaluate_tamper_risk(summary: Dict[str, Any], pdf_metadata: Dict[str, Any]) -> Dict[str, Any]:
    signals: List[Signal] = []
    document_type = summary.get("document", {}).get("type", "generic")
    key_fields = summary.get("key_fields", {})

    if document_type == "payslip":
        gross = _parse_numeric_value(key_fields.get("gross_earnings"))
        deductions = _parse_numeric_value(key_fields.get("total_deductions"))
        net_pay = _parse_numeric_value(key_fields.get("net_pay"))
        if gross is not None and deductions is not None and net_pay is not None:
            expected = gross - deductions
            signals.append(
                Signal(
                    name="gross_minus_deductions_equals_net_pay",
                    severity="high" if expected != net_pay else "info",
                    score=65 if expected != net_pay else 0,
                    summary="Verify that gross earnings minus total deductions equals net pay.",
                    passed=expected == net_pay,
                    details={"expected": expected, "actual": net_pay},
                )
            )

        words = key_fields.get("net_pay_in_words")
        if words and net_pay is not None:
            parsed_words = _parse_number_words(words)
            mismatch = parsed_words is None or parsed_words != net_pay
            signals.append(
                Signal(
                    name="amount_in_words_matches_net_pay",
                    severity="medium" if mismatch else "info",
                    score=40 if mismatch else 0,
                    summary="Verify that the amount in words matches the numeric net pay.",
                    passed=not mismatch,
                    details={"expected": net_pay, "actual": parsed_words},
                )
            )

        required_fields = ["employee_id", "employee_name", "gross_earnings", "total_deductions", "net_pay"]
        missing = [field for field in required_fields if not key_fields.get(field)]
        signals.append(
            Signal(
                name="required_payslip_fields_present",
                severity="medium" if missing else "info",
                score=min(25, 5 * len(missing)),
                summary="Ensure the critical payslip fields are present.",
                passed=not missing,
                details={"missing_fields": missing},
            )
        )

    metadata = pdf_metadata.get("metadata", {}) if isinstance(pdf_metadata.get("metadata"), dict) else {}
    producer_blob = " ".join(
        str(metadata.get(key, ""))
        for key in ("Producer", "producer", "Creator", "creator")
    ).lower()
    suspicious_editor_found = any(editor in producer_blob for editor in SUSPICIOUS_EDITORS)
    signals.append(
        Signal(
            name="editor_software_mismatch",
            severity="medium" if suspicious_editor_found else "info",
            score=35 if suspicious_editor_found and document_type in {"payslip", "invoice", "bank_statement"} else 0,
            summary="Look for editing software in PDF metadata that is inconsistent with official document generation.",
            passed=not suspicious_editor_found,
            details={"producer_blob": producer_blob},
        )
    )

    has_signature_markers = bool(pdf_metadata.get("has_digital_signature_markers"))
    signals.append(
        Signal(
            name="digital_signature_markers_present",
            severity="info",
            score=0,
            summary="Record whether the PDF contains digital-signature markers.",
            passed=has_signature_markers,
            details={"signature_markers": pdf_metadata.get("signature_markers", [])},
        )
    )

    conflicting = [
        key
        for key, value in summary.get("generic_candidates", {}).items()
        if isinstance(value, list) and key == "dates" and len(value) > 5
    ]
    signals.append(
        Signal(
            name="unexpected_candidate_density",
            severity="low" if conflicting else "info",
            score=10 if conflicting else 0,
            summary="Surface unusually dense extracted candidates that may indicate OCR or layout confusion.",
            passed=not conflicting,
            details={"flags": conflicting},
        )
    )

    # Metadata date consistency: modification date should not predate creation date.
    metadata = pdf_metadata.get("metadata", {}) if isinstance(pdf_metadata.get("metadata"), dict) else {}
    create_date = str(metadata.get("CreationDate") or metadata.get("creation_date") or "").strip()
    mod_date = str(metadata.get("ModDate") or metadata.get("mod_date") or "").strip()
    if create_date and mod_date and len(create_date) >= 14 and len(mod_date) >= 14:
        try:
            create_prefix = create_date[2:16] if create_date.startswith("D:") else create_date[:14]
            mod_prefix = mod_date[2:16] if mod_date.startswith("D:") else mod_date[:14]
            date_order_ok = mod_prefix >= create_prefix
            signals.append(
                Signal(
                    name="metadata_date_consistency",
                    severity="medium" if not date_order_ok else "info",
                    score=30 if not date_order_ok else 0,
                    summary="Modification date should not predate the creation date in PDF metadata.",
                    passed=date_order_ok,
                    details={"creation_date": create_date, "mod_date": mod_date},
                )
            )
        except (ValueError, IndexError):
            pass

    # Domain-specific validation pack signals.
    for domain_signal in validate_document(summary):
        signals.append(
            Signal(
                name=f"domain::{domain_signal.get('rule', 'unknown')}",
                severity=str(domain_signal.get("severity", "info")),
                score=int(domain_signal.get("score", 0)),
                summary=str(domain_signal.get("message", "")),
                passed=bool(domain_signal.get("passed", True)),
                details=dict(domain_signal.get("details", {})),
            )
        )

    total_risk = sum(signal.score for signal in signals)
    truth_score, risk_level, verdict = _risk_level(total_risk)
    return {
        "truth_score": truth_score,
        "risk_level": risk_level,
        "risk_score": total_risk,
        "verdict": verdict,
        "signals": signals_to_dict(signals),
        "limitations": [
            "This is a forensic heuristic layer, not conclusive proof of authenticity.",
            "Cryptographic validation requires trusted issuer certificates or external signature tools.",
        ],
    }
