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
    # Mortgage / home-loan document types
    "employment_letter": (
        "to whomsoever it may concern", "employment letter", "date of joining",
        "company identification number", "cin", "hr department", "employment verification",
    ),
    "form16": (
        "form 16", "form-16", "tds certificate", "certificate of tax deducted",
        "assessment year", "tan of employer", "section 203",
    ),
    "utility_bill": (
        "electricity bill", "utility bill", "consumer number", "units consumed",
        "kwh", "electricity duty", "msedcl", "bescom", "tangedco", "best electricity",
    ),
    "gift_letter": (
        "gift declaration", "gift letter", "hereby declare", "amount is a gift",
        "not a loan", "down payment gift",
    ),
    "property_agreement": (
        "agreement for sale", "sale agreement", "sale deed", "sale consideration",
        "vendor", "purchaser", "property address", "rera",
    ),
    "mortgage": (
        "home loan", "mortgage", "loan application", "loan against property",
        "property documents", "loan to value",
    ),
    "generic": (),
}

_EMPLOYMENT_CIN_RE = re.compile(
    r"\bCIN[:\s]+([LU][0-9]{5}[A-Z]{2}[0-9]{4}[A-Z]{2,4}[0-9]{6})\b",
    re.IGNORECASE,
)


def _extract_employment_letter_fields(full_text: str) -> Dict[str, Any]:
    """Extract key fields from employment / offer letter paragraph prose.

    Employment letters are free-form text, so the standard label-value pair
    extraction misses everything.  This function uses targeted regexes to pull
    the fields that matter for income-fraud detection.

    Fields extracted:
      cin               -- Company Identification Number (MCA format)
      annual_ctc        -- Annual Cost-to-Company figure
      gross_monthly_salary -- Gross monthly salary
      date_of_joining   -- Employee join / effective date (the claimed start date)
      letter_issue_date -- The date the letter itself was printed/issued
                          (appears as "Date: 15 March 2026" at the top of letter)
      employee_name     -- Employee name as stated in the letter

    Key fraud signal: if letter_issue_date is recent but date_of_joining is
    many years older, the join date may have been backdated artificially.
    """
    fields: Dict[str, Any] = {}

    # CIN — typically appears as "CIN: L22210MH1995PLC084781"
    cin_match = _EMPLOYMENT_CIN_RE.search(full_text)
    if cin_match:
        fields["cin"] = cin_match.group(1).upper()

    # Annual CTC — "CTC) is n11,10,000 per annum" or "CTC of Rs. 5,40,000"
    # The currency symbol is sometimes captured as 'n' by the PDF parser.
    ctc_match = re.search(
        r"(?:ctc|cost.{0,30}?company)[^0-9\n]{0,40}?(\d[\d,]+)\s+per\s+annum",
        full_text,
        re.IGNORECASE,
    )
    if ctc_match:
        val = _parse_numeric_value(ctc_match.group(1))
        if val and val > 10_000:
            fields["annual_ctc"] = val

    # Gross monthly — "(Gross Monthly: n92,500)" or "Gross Monthly Salary: 92,500"
    gross_match = re.search(
        r"Gross\s+Monthly[^0-9\n]{0,20}?(\d[\d,]+)",
        full_text,
        re.IGNORECASE,
    )
    if gross_match:
        val = _parse_numeric_value(gross_match.group(1))
        if val and val > 1_000:
            fields["gross_monthly_salary"] = val

    # Join / effective date — "with effect from 20 September 2021"
    # This is the DATE THE EMPLOYEE STARTED (can be backdated in tampered docs)
    join_match = re.search(
        r"(?:with\s+effect\s+from|date\s+of\s+joining|joining\s+date|joining\s+from)"
        r"\s+([0-9]{1,2}\s+[A-Za-z]+\s+[0-9]{4}|[0-9]{1,2}[/-][A-Za-z0-9]+[/-][0-9]{4})",
        full_text,
        re.IGNORECASE,
    )
    if join_match:
        fields["date_of_joining"] = re.sub(r"\s+", " ", join_match.group(1)).strip()

    # Letter issue date — "Date: 15 March 2026" or "Dated: 15-03-2026"
    # This is the DATE THE LETTER WAS TYPED/ISSUED by the HR department.
    # A recently-issued letter with a suspiciously old join date is a backdating signal.
    issue_date_match = re.search(
        r"(?:^|\n)\s*[Dd]ated?\s*[:\-]\s*"
        r"([0-9]{1,2}\s+[A-Za-z]+\s+[0-9]{4}|[0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{4})",
        full_text,
    )
    if issue_date_match:
        fields["letter_issue_date"] = re.sub(r"\s+", " ", issue_date_match.group(1)).strip()

    # Employee name — "certify that Widisha Thaker," or "employee name: John Doe"
    name_match = re.search(
        r"(?:certify\s+that|employee\s+name\s*[:\-])\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})[\s,.]",
        full_text,
    )
    if name_match:
        fields["employee_name"] = name_match.group(1).strip()

    return fields


def _parse_bank_date(date_str: str) -> Optional[datetime]:
    """Parse a bank statement date (DD-MM-YYYY or DD-Mon-YYYY) into a datetime."""
    for fmt in ("%d-%m-%Y", "%d-%b-%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    return None


def _extract_bank_statement_fields(full_text: str, label_index: Dict[str, List[str]]) -> Dict[str, Any]:
    """Extract key fields from a bank account statement.

    Bank statements have both structured table content (label-value pairs) and
    free-form text in the transaction narrative.  This function pulls:

      bank_name           -- Name of the issuing bank (first header line)
      ifsc                -- IFSC code of the branch
      account_holder      -- Account holder name
      account_number      -- Masked account number
      branch              -- Branch name / city
      period_from         -- Statement start date (e.g. "01-Oct-2025")
      period_to           -- Statement end date   (e.g. "31-Mar-2026")
      opening_balance     -- Opening balance at period start
      closing_balance     -- Closing balance at period end
      transactions        -- List of parsed transaction rows (date/desc/ref/debit/credit/balance)
      credit_total        -- Sum of all credit amounts across transactions
      debit_total         -- Sum of all debit amounts across transactions
      earliest_txn_date   -- Date of the earliest transaction (DD-MM-YYYY)
      latest_txn_date     -- Date of the latest transaction (DD-MM-YYYY)

    The ``transactions`` list enables:
      - bank_statement_date_range: do transactions actually span the stated period?
      - bank_debit_credit_arithmetic: running balance should be consistent row-by-row
      - circular_funds_detection: same-day large debit+credit pairs
      - salary_credit_regularity: salary credited once per calendar month
    """
    fields: Dict[str, Any] = {}

    # ---- Bank name — first significant line of the statement ----
    # LiteParse preserves the bank letterhead as the first non-empty lines.
    # The bank name (e.g. "HDFC BANK", "STATE BANK OF INDIA") always appears
    # before the branch and IFSC details on line 2.
    for raw_line in full_text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        if len(stripped) < 70 and re.search(r"\bBANK\b", stripped, re.IGNORECASE):
            fields["bank_name"] = stripped
            break

    # ---- Statement period — "01-Oct-2025 to 31-Mar-2026" ----
    period_match = re.search(
        r"([0-9]{1,2}[/-][A-Za-z0-9]+[/-][0-9]{4})\s+to\s+([0-9]{1,2}[/-][A-Za-z0-9]+[/-][0-9]{4})",
        full_text,
        re.IGNORECASE,
    )
    if period_match:
        fields["period_from"] = period_match.group(1).strip()
        fields["period_to"] = period_match.group(2).strip()

    # ---- IFSC code — standard 11-char Indian format ----
    ifsc_match = re.search(r"\b([A-Z]{4}0[A-Z0-9]{6})\b", full_text, re.IGNORECASE)
    if ifsc_match:
        fields["ifsc"] = ifsc_match.group(1).upper()

    # ---- Account number (masked) — "XXXX XXXX 8677" ----
    acc_match = re.search(r"\b((?:X{4}\s?){2}[0-9]{4})\b", full_text)
    if acc_match:
        fields["account_number"] = acc_match.group(1).strip()

    # ---- Opening / closing balance from label-value pairs ----
    for label_key in ("opening_balance", "opening_bal", "balance_bf", "balance_brought_forward"):
        if label_index.get(label_key):
            val = _parse_numeric_value(label_index[label_key][0])
            if val is not None:
                fields["opening_balance"] = val
                break

    for label_key in ("closing_balance", "closing_bal", "balance_cf", "balance_carried_forward"):
        if label_index.get(label_key):
            val = _parse_numeric_value(label_index[label_key][0])
            if val is not None:
                fields["closing_balance"] = val
                break

    # ---- Transaction row parsing ----
    # LiteParse format: each row is one line with 5 columns separated by 2+ spaces.
    #   DD-MM-YYYY  DESCRIPTION  REFNO  amount(debit or credit)  balance
    # Credit rows: Debit column is blank → only 1 amount before balance.
    # Debit rows: Credit column is blank → only 1 amount before balance.
    # Both produce exactly 5 parts after re.split(r"\s{2,}", line).
    # The last element is always the running balance.
    # The second-to-last element is the debit OR credit amount.
    # We distinguish debit vs credit from the description: if it contains
    # 'credit', 'interest', or 'refund' it is a credit; otherwise a debit.
    _DATE_RE = re.compile(r"^\d{2}-\d{2}-\d{4}$")
    transactions: List[Dict[str, Any]] = []

    for raw_line in full_text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        parts = [p.strip() for p in re.split(r"\s{2,}", stripped) if p.strip()]
        # Need at minimum: date, description, ref, amount, balance (5 parts)
        if len(parts) < 5:
            continue
        # First part must match a date
        if not _DATE_RE.match(parts[0]):
            continue
        # Last part must be a plausible balance (≥ 1,000 to filter page-break truncation)
        balance_val = _parse_numeric_value(parts[-1])
        if balance_val is None or balance_val < 1_000:
            continue
        # Second-to-last is the debit or credit amount
        amount_val = _parse_numeric_value(parts[-2])
        if amount_val is None:
            continue
        # Description is everything between date and the last 3 parts (ref, amount, balance)
        desc = " ".join(parts[1:-3]).strip() if len(parts) > 4 else parts[1]
        _desc_lower = desc.lower()
        is_credit = any(kw in _desc_lower for kw in (
            "credit", "interest", "refund", "gift", "inward", "reversal",
            "cashback", "dividend", "proceeds",
        ))
        transactions.append({
            "date":    parts[0],
            "desc":    desc,
            "ref":     parts[-3] if len(parts) > 3 else "",
            "debit":   None if is_credit else amount_val,
            "credit":  amount_val if is_credit else None,
            "balance": balance_val,
        })

    if transactions:
        fields["transactions"] = transactions
        # Sort by date for date-range and arithmetic checks
        sorted_txns = sorted(
            transactions,
            key=lambda t: (_parse_bank_date(t["date"]) or datetime.min),
        )
        fields["credit_total"] = sum(t["credit"] for t in transactions if t.get("credit") is not None)
        fields["debit_total"]  = sum(t["debit"]  for t in transactions if t.get("debit")  is not None)
        fields["earliest_txn_date"] = sorted_txns[0]["date"]
        fields["latest_txn_date"]   = sorted_txns[-1]["date"]

    return fields


def _extract_gift_letter_fields(full_text: str) -> Dict[str, Any]:
    """Extract key fields from a gift declaration letter.

    Fields extracted:
      gift_amount    -- The rupee value of the gift (e.g. 1_800_000)
      property_value -- Target property value if stated
      donor_pan      -- Donor's PAN number
      letter_date    -- Date the letter was issued
    """
    fields: Dict[str, Any] = {}

    # Gift amount — "gifted a sum of [symbol]18,00,000" or "gift amount ... 18,00,000"
    gift_match = re.search(
        r"(?:gifted?\s+a?\s*sum\s+of|gift\s+(?:amount|of))[^\d\n]{0,30}?(\d[\d,]+)",
        full_text, re.IGNORECASE,
    )
    if gift_match:
        val = _parse_numeric_value(gift_match.group(1))
        if val and val > 10_000:
            fields["gift_amount"] = val

    # Property value — "property value is [symbol]20,00,000"
    prop_match = re.search(
        r"property\s+value\s+is[^\d\n]{0,20}?(\d[\d,]+)",
        full_text, re.IGNORECASE,
    )
    if prop_match:
        val = _parse_numeric_value(prop_match.group(1))
        if val and val > 10_000:
            fields["property_value"] = val

    # Donor PAN — "PAN: YLSFF4758H"
    pan_match = re.search(r"\bPAN\s*[:\-]\s*([A-Z]{5}[0-9]{4}[A-Z])\b", full_text)
    if pan_match:
        fields["donor_pan"] = pan_match.group(1)

    # Letter date — "Date: 10 March 2026"
    date_match = re.search(
        r"[Dd]ate\s*[:\-]\s*(\d{1,2}\s+[A-Za-z]+\s+\d{4})",
        full_text,
    )
    if date_match:
        fields["letter_date"] = re.sub(r"\s+", " ", date_match.group(1)).strip()

    return fields


PAYSLIP_FIELD_MAP = {
    "employee_id": ("employee_id",),
    "employee_name": ("employee_name",),
    "designation": ("designation",),
    "date_of_joining": ("date_of_joining",),
    "pay_date": ("pay_date",),
    "paid_days": ("paid_days",),
    "lop_days": ("lop_days",),
    "uan": ("uan",),
    # PAN (for cross-doc PAN mismatch check)
    "pan": ("pan",),
    # Bank name on payslip (for cross-doc IFSC/bank consistency check)
    "bank": ("bank",),
    # Basic salary: 'Basic' normalized to 'basic', but payslips commonly say 'Basic Salary' → 'basic_salary'
    "basic": ("basic", "basic_salary"),
    "house_rent_allowance": ("house_rent_allowance",),
    "fixed_bonus": ("fixed_bonus",),
    "other_allowances": ("other_allowances",),
    "advance_or_arrears": ("advance_or_arrears_or_notice_pay", "advance_or_arrears"),
    "gross_earnings": ("gross_earnings",),
    # Net pay: 'NET PAY (TAKE HOME)' → 'net_pay_take_home'; 'Net Salary' → 'net_salary'
    "net_pay": ("net_pay", "net_pay_take_home", "net_salary"),
    "total_deductions": ("total_deductions",),
    # EPF / PF: payslips use 'Provident Fund (Emp)' → 'provident_fund_emp'
    "epf_contribution": ("epf_contribution", "provident_fund_emp", "provident_fund", "pf_emp"),
    # Income Tax: 'Income Tax (TDS)' → 'income_tax_tds'
    "income_tax": ("income_tax", "income_tax_tds"),
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
    all_scores: Dict[str, int] = {}
    matched_markers: Dict[str, List[str]] = {}
    for doc_type, markers in DOC_TYPE_PATTERNS.items():
        if doc_type == "generic":
            continue
        scored = [m for m in markers if m in haystack]
        score = len(scored)
        filename_boost = 0
        if doc_type in source_name.lower():
            score += 2
            filename_boost = 2
        all_scores[doc_type] = score
        matched_markers[doc_type] = scored + (["(filename_boost)"] if filename_boost else [])
        if score > best_score:
            best_type = doc_type
            best_score = score
    confidence = 0.2 if best_type == "generic" else min(0.98, 0.4 + (best_score * 0.12))
    # Only include types that got at least one signal, plus the winner
    nonzero = {k: v for k, v in all_scores.items() if v > 0}
    return {
        "type": best_type,
        "confidence": round(confidence, 2),
        "classification_scores": nonzero or {"generic": 0},
        "matched_markers": {k: matched_markers[k] for k in nonzero} if nonzero else {},
        "text_length": len(full_text),
        "text_preview": full_text[:300].replace("\n", " ").strip(),
    }


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
    elif document["type"] == "employment_letter":
        key_fields = _extract_employment_letter_fields(full_text)
        # Preserve generic named_fields as a fallback for any fields we missed
        if generic_candidates["named_fields"]:
            key_fields.setdefault("named_fields", generic_candidates["named_fields"])
    elif document["type"] == "bank_statement":
        # Extract structured bank-statement fields for arithmetic and period checks
        key_fields = _extract_bank_statement_fields(full_text, label_index)
        if generic_candidates["named_fields"]:
            key_fields.setdefault("named_fields", generic_candidates["named_fields"])
    elif document["type"] == "gift_letter":
        key_fields = _extract_gift_letter_fields(full_text)
        if generic_candidates["named_fields"]:
            key_fields.setdefault("named_fields", generic_candidates["named_fields"])
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
            # Diagnostics: how much text was extracted and how many label-value pairs
            "text_length": len(full_text),
            "line_count": len(lines),
            "label_value_pair_count": len(pairs),
        },
        "document": {
            "type": document["type"],
            "type_confidence": document["confidence"],
            "issuer": lines[0]["line"] if lines else "",
            "title": _extract_title(lines, document["type"]),
            # Full classification diagnostics — which type won and why
            "classification_scores": document.get("classification_scores", {}),
            "matched_markers": document.get("matched_markers", {}),
            "text_preview": document.get("text_preview", ""),
        },
        "key_fields": key_fields,
        "generic_candidates": {
            "dates": generic_candidates["dates"],
            "identifiers": generic_candidates["identifiers"],
            "monetary_fields": generic_candidates["monetary_fields"],
        },
        "label_value_pairs": pairs,
    }
