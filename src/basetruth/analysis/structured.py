from __future__ import annotations

"""
Build a structured summary from a LiteParse (or fallback) JSON document.

The pipeline has three stages:
  1. Text assembly  -- join the raw page texts from the LiteParse JSON output.
  2. Field extraction -- use regular expressions and heuristics to identify
     document type (payslip, offer letter, bank statement ...) and extract
     key fields (employee name, gross pay, net pay, pay period, issuer, etc.).
  3. Normalisation -- convert currency strings to floats, parse ISO-8601 dates,
     produce a canonical dict consumed by tamper.evaluate_tamper_risk() and the
     reporting layer.

Public API
----------
  build_structured_summary(raw_parse_path, source_path=None) -> Dict[str, Any]
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


DOC_TYPE_PATTERNS = {
    "payslip": ("payslip", "salary slip", "employee pay summary", "gross earnings", "net pay"),
    "invoice": ("invoice", "bill to", "invoice number", "amount due"),
    "receipt": ("receipt", "receipt number", "payment received"),
    "bank_statement": ("bank statement", "closing balance", "statement period"),
    "payment_receipt": ("upi", "neft", "rtgs", "transaction id", "payment receipt", "utr number"),
    "insurance": ("policy number", "insured name", "sum insured", "premium", "insurance certificate"),
    "healthcare": ("patient name", "discharge summary", "diagnosis", "hospital invoice", "uhid"),
    "compliance": ("audit report", "compliance certificate", "board resolution", "kyc", "aml declaration"),
    "generic": (),
}

PAYSLIP_FIELD_MAP = {
    "employee_id": ("employee_id",),
    "employee_name": ("employee_name",),
    "designation": ("designation",),
    "date_of_joining": ("date_of_joining",),
    "pay_date": ("pay_date",),
    "paid_days": ("paid_days",),
    "lop_days": ("lop_days",),
    "uan": ("uan",),
    "basic": ("basic",),
    "house_rent_allowance": ("house_rent_allowance",),
    "fixed_bonus": ("fixed_bonus",),
    "other_allowances": ("other_allowances",),
    "advance_or_arrears": ("advance_or_arrears_or_notice_pay", "advance_or_arrears"),
    "gross_earnings": ("gross_earnings",),
    "net_pay": ("net_pay",),
    "total_deductions": ("total_deductions",),
    "epf_contribution": ("epf_contribution",),
    "income_tax": ("income_tax",),
    "professional_tax": ("professional_tax",),
    "other_deductions": ("other_deductions",),
}


def _normalize_label(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(label or "").strip().lower()).strip("_")


def _parse_numeric_value(value: Any) -> Optional[int]:
    digits = re.sub(r"[^0-9-]", "", str(value or ""))
    if not digits or digits == "-":
        return None
    try:
        return int(digits)
    except ValueError:
        return None


def _extract_lines(parsed_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    lines: List[Dict[str, Any]] = []
    pages = parsed_data.get("pages") if isinstance(parsed_data, dict) else None
    if not isinstance(pages, list):
        return lines
    for page in pages:
        if not isinstance(page, dict):
            continue
        page_number = page.get("page")
        for raw_line in str(page.get("text", "") or "").splitlines():
            if raw_line.strip():
                lines.append({"page": page_number, "line": raw_line.strip()})
    return lines


def _extract_pairs(lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    pairs: List[Dict[str, Any]] = []
    for entry in lines:
        segments = [segment.strip() for segment in re.split(r"\s{2,}", entry["line"]) if segment.strip()]
        if len(segments) < 2:
            continue
        for index in range(0, len(segments) - 1, 2):
            label = segments[index]
            value = segments[index + 1]
            pairs.append(
                {
                    "page": entry["page"],
                    "label": label,
                    "normalized_label": _normalize_label(label),
                    "value": value,
                }
            )
    return pairs


def _build_label_index(pairs: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    index: Dict[str, List[str]] = {}
    for pair in pairs:
        index.setdefault(pair["normalized_label"], [])
        if pair["value"] not in index[pair["normalized_label"]]:
            index[pair["normalized_label"]].append(pair["value"])
    return index


def _detect_document_type(source_name: str, full_text: str) -> Dict[str, Any]:
    haystack = f"{source_name}\n{full_text}".lower()
    best_type = "generic"
    best_score = 0
    for doc_type, markers in DOC_TYPE_PATTERNS.items():
        if doc_type == "generic":
            continue
        score = sum(1 for marker in markers if marker in haystack)
        if doc_type in source_name.lower():
            score += 2
        if score > best_score:
            best_type = doc_type
            best_score = score
    confidence = 0.2 if best_type == "generic" else min(0.98, 0.4 + (best_score * 0.12))
    return {"type": best_type, "confidence": round(confidence, 2)}


def _extract_generic_candidates(lines: List[Dict[str, Any]], pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
    dates: List[str] = []
    identifiers: List[Dict[str, Any]] = []
    monetary_fields: List[Dict[str, Any]] = []

    for entry in lines:
        for match in re.findall(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", entry["line"]):
            if match not in dates:
                dates.append(match)

    for pair in pairs:
        normalized = pair["normalized_label"]
        value = pair["value"]
        if any(token in normalized for token in ("id", "uan", "account", "reference", "number")):
            identifiers.append({"label": pair["label"], "value": value, "page": pair["page"]})
        if "date" in normalized:
            continue
        numeric_value = _parse_numeric_value(value)
        if numeric_value is not None and any(
            token in normalized
            for token in ("amount", "pay", "gross", "net", "bonus", "allowance", "tax", "deduction", "contribution", "salary", "arrears", "earnings")
        ):
            monetary_fields.append(
                {
                    "label": pair["label"],
                    "value": value,
                    "numeric_value": numeric_value,
                    "page": pair["page"],
                }
            )

    named_fields: Dict[str, Any] = {}
    for pair in pairs:
        current = named_fields.get(pair["normalized_label"])
        if current is None:
            named_fields[pair["normalized_label"]] = pair["value"]
        elif isinstance(current, list):
            if pair["value"] not in current:
                current.append(pair["value"])
        elif current != pair["value"]:
            named_fields[pair["normalized_label"]] = [current, pair["value"]]

    return {
        "dates": dates,
        "identifiers": identifiers,
        "monetary_fields": monetary_fields,
        "named_fields": named_fields,
    }


def _extract_title(lines: List[Dict[str, Any]], document_type: str) -> str:
    for entry in lines:
        line = entry["line"]
        lowered = line.lower()
        if document_type == "payslip" and "payslip for the month of" in lowered:
            return line
        if len(line) > 10 and not re.fullmatch(r"[*\-= ]+", line):
            return line
    return ""


def _extract_payslip_fields(full_text: str, label_index: Dict[str, List[str]]) -> Dict[str, Any]:
    fields: Dict[str, Any] = {}
    for field_name, aliases in PAYSLIP_FIELD_MAP.items():
        for alias in aliases:
            if label_index.get(alias):
                fields[field_name] = label_index[alias][0]
                break

    pay_period_match = re.search(r"Payslip for the month of\s+([A-Za-z]+\s+\d{4})", full_text, re.IGNORECASE)
    if pay_period_match:
        fields["pay_period"] = pay_period_match.group(1).strip()

    words_match = re.search(r"Rupees\s+(.+?)\s+Only", full_text, re.IGNORECASE | re.DOTALL)
    if words_match:
        fields["net_pay_in_words"] = re.sub(r"\s+", " ", words_match.group(1)).strip()

    fields["earnings_breakdown"] = {
        key: fields[key]
        for key in ("basic", "house_rent_allowance", "fixed_bonus", "other_allowances", "advance_or_arrears")
        if key in fields
    }
    fields["deductions_breakdown"] = {
        key: fields[key]
        for key in ("epf_contribution", "income_tax", "professional_tax", "other_deductions", "total_deductions")
        if key in fields
    }
    return fields


def build_structured_summary(parse_output_path: Path, source_path: Optional[Path] = None) -> Dict[str, Any]:
    parsed_data = json.loads(parse_output_path.read_text(encoding="utf-8"))
    if not isinstance(parsed_data, dict):
        raise ValueError("LiteParse JSON must contain a top-level object.")

    lines = _extract_lines(parsed_data)
    pairs = _extract_pairs(lines)
    label_index = _build_label_index(pairs)
    full_text = "\n".join(entry["line"] for entry in lines)
    source_name = source_path.name if source_path else parse_output_path.name
    document = _detect_document_type(source_name, full_text)
    generic_candidates = _extract_generic_candidates(lines, pairs)
    key_fields: Dict[str, Any]
    if document["type"] == "payslip":
        key_fields = _extract_payslip_fields(full_text, label_index)
    else:
        key_fields = {
            "title": _extract_title(lines, document["type"]),
            "primary_date": generic_candidates["dates"][0] if generic_candidates["dates"] else None,
            "named_fields": generic_candidates["named_fields"],
        }

    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "path": str(source_path) if source_path else "",
            "name": source_name,
            "extension": source_path.suffix.lower() if source_path else parse_output_path.suffix.lower(),
        },
        "parser": {
            "parse_output_path": str(parse_output_path),
            "page_count": len(parsed_data.get("pages", [])) if isinstance(parsed_data.get("pages"), list) else 0,
            "top_level_keys": list(parsed_data.keys())[:20],
        },
        "document": {
            "type": document["type"],
            "type_confidence": document["confidence"],
            "issuer": lines[0]["line"] if lines else "",
            "title": _extract_title(lines, document["type"]),
        },
        "key_fields": key_fields,
        "generic_candidates": {
            "dates": generic_candidates["dates"],
            "identifiers": generic_candidates["identifiers"],
            "monetary_fields": generic_candidates["monetary_fields"],
        },
        "label_value_pairs": pairs,
    }
