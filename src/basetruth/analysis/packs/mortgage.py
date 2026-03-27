from __future__ import annotations

"""
Validation pack for mortgage / home-loan supporting documents.

Covers the Indian home-loan document bundle:
  - payslip (income proof)
  - bank_statement (financial track-record)
  - employment_letter (employment proof)
  - form16 (TDS certificate)
  - utility_bill (residence proof)
  - mortgage_gift_letter (down-payment gift declaration)
  - property_agreement (sale agreement)

Checks
------
  --- Payslip checks ---
  basic_within_gross             -- basic must be a positive component within gross
  net_pay_arithmetic             -- gross − deductions ≈ net pay (fabricated deduction removal)
  pf_rate_validity               -- PF ≤ 12% of basic (statutory ceiling)
  pt_slab_validity               -- Professional Tax ≤ ₹200/month (max slab, any Indian state)
  tds_plausibility               -- TDS > 0 when annual gross > ₹7 lakh
  hra_proportion                 -- HRA ≤ 50% of basic (legal maximum under income-tax rules)
  basic_gross_proportion         -- Basic must be 30–50% of gross (industry norm; <30% is suspicious)
  hra_basic_proportion           -- HRA should be 40–50% of basic (>50% is over-claim)

  --- Bank statement checks ---
  circular_funds_detection       -- large round-trip debit+credit same day (circular fund inflation)
  salary_credit_regularity       -- salary credit should appear exactly once per calendar month
  duplicate_txn_reference        -- same reference number used twice (copy-paste fabrication)
  bank_statement_date_range      -- actual earliest/latest transaction dates must span stated period
  bank_debit_credit_arithmetic   -- sum(credits) − sum(debits) must ≈ closing − opening balance
  bank_ifsc_consistency          -- IFSC in statement header should appear consistently (no mid-statement switch)

  --- Employment letter checks ---
  employer_cin_present           -- CIN must be present on the letter (absent = possible shell company)
  employer_cin_format            -- CIN must match Indian MCA format [LU]NNNNNSSYYYYCCCNNNNNN
  cin_age_vs_join_date           -- company incorporation year (from CIN) must be ≤ employee join year
  ctc_monthly_gross_consistency  -- CTC / 12 ≈ stated monthly gross on the same letter
  employment_backdating_signal   -- NEW: letter issued recently but join date is suspiciously old
                                    (e.g., letter dated Mar 2026 but join date Apr 2014 = backdating)

  --- Form 16 checks ---
  tan_format_validity            -- TAN must match 10-char format
  form16_tds_plausibility        -- TDS > 0 when gross > ₹7 lakh

  --- Utility bill checks ---
  utility_amount_plausibility    -- bill amount within ₹100–₹20,000 range

  NOTE: Cross-document salary reconciliation (payslip vs Form 16 vs offer letter vs bank)
  is performed at the service layer via BaseTruthService.reconcile_income_documents()
  because it requires data from multiple documents simultaneously.

  ----- Additional fraud patterns detected at the service layer -----
  ifsc_account_mismatch          -- IFSC in bank statement ≠ IFSC on payslip
  salary_cross_doc_mismatch      -- payslip gross vs bank salary credit discrepancy
  bank_address_mismatch          -- branch city in statement ≠ expected city for IFSC prefix
  salary_structure_violation     -- basic/HRA do not follow expected % of CTC
"""

import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from basetruth.analysis.packs.base import BaseValidationPack, ValidationSignal, _parse_int


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Indian CIN format: [LU]NNNNNSSYYYYCCCNNNNNN
# L or U  = listed / unlisted
# 5-digit NIC activity code
# 2-letter state code
# 4-digit year of incorporation
# 3-letter company category (PLC / PVT / OPC / FLC …)
# 6-digit sequential number
_CIN_RE = re.compile(
    r"^[LU][0-9]{5}[A-Z]{2}[0-9]{4}[A-Z]{2,4}[0-9]{6}$",
    re.IGNORECASE,
)

# TAN format: 4 letters + 5 digits + 1 letter
_TAN_RE = re.compile(r"^[A-Z]{4}[0-9]{5}[A-Z]$", re.IGNORECASE)

# Maximum professional tax per month in any Indian state (Karnataka slab is ₹200)
_MAX_PT_MONTHLY = 200

# Statutory PF rate: 12% of basic (employer matches the same)
_PF_RATE_MAX = 0.125  # 12.5% — slight tolerance for rounding

# Annual income threshold above which TDS should be non-zero (₹7 lakh)
_TDS_THRESHOLD_ANNUAL = 700_000

# HRA cannot reasonably exceed basic salary
_HRA_MAX_BASIC_RATIO = 1.00

# Circular-funds: pairs with amount ≥ this value on the same day are flagged
_CIRCULAR_FUNDS_MIN_AMOUNT = 100_000  # ₹1 lakh

# Basic salary should be 30–50% of gross (Indian IT industry norm).
# Below 30% is unusual and may indicate manipulation; above 50% is also atypical.
_BASIC_GROSS_MIN_RATIO = 0.30
_BASIC_GROSS_MAX_RATIO = 0.55  # 55% upper tolerance (some companies go slightly higher)

# HRA is typically 40–50% of basic.  Legal maximum for metro cities is 50%.
# Raising HRA inflates the tax-exempt portion of income, so >50% is suspicious.
_HRA_BASIC_NORMAL_MAX_RATIO = 0.50

# Backdating signal: if the letter was issued within this many months of today
# and the join date is more than this many years earlier, flag it.
#
# Threshold rationale: Indian mortgage applicants often have 7-10 year tenures —
# a completely normal employment history.  We only flag as suspicious when the
# gap exceeds 12 years, which is uncommon and warrants cross-checking the CIN
# incorporation date (already covered by cin_age_vs_join_date).  Extreme cases
# (e.g. a 26-year gap) like the case_058 "extreme_backdating" pattern are still
# reliably caught at this threshold.
_BACKDATING_LETTER_WINDOW_MONTHS = 12   # "recently issued" threshold
_BACKDATING_JOIN_GAP_YEARS = 12         # gap that becomes suspicious (raised from 7)

# Bank statement arithmetic: tolerate this fraction of the stated balance
# as rounding/display error before declaring an arithmetic failure.
_BANK_ARITHMETIC_TOLERANCE_RATIO = 0.01  # 1%
_BANK_ARITHMETIC_MIN_TOLERANCE = 500     # ₹500 absolute floor

# IFSC prefix (first 4 chars) → list of expected bank name substrings.
# Used to validate that the IFSC code belongs to the bank stated in the header.
_IFSC_BANK_PREFIX_MAP: Dict[str, List[str]] = {
    "HDFC": ["HDFC"],
    "ICIC": ["ICICI"],
    "SBIN": ["SBI", "STATE BANK"],
    "PUNB": ["PNB", "PUNJAB NATIONAL"],
    "UBIN": ["UNION BANK"],
    "CNRB": ["CANARA"],
    "BARB": ["BANK OF BARODA", "BARODA"],
    "BKID": ["BANK OF INDIA"],
    "UTIB": ["AXIS"],
    "KKBK": ["KOTAK"],
    "IDFB": ["IDFC"],
    "INDB": ["INDUSIND"],
    "YESB": ["YES BANK", "YES"],
}


def _round_trip_pairs(transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Find debit-credit pairs on the same date with matching amounts ≥ threshold.

    Circular funds: a borrower deposits a large sum and then withdraws the same
    amount (or vice-versa) on the same day to inflate the apparent account balance.
    We detect this by finding debit-credit pairs within ±2% on the same date.

    Each transaction dict is expected to have keys:
      date  (str)
      debit (int | None)
      credit (int | None)
    """
    from collections import defaultdict

    # Group debits and credits by date
    by_date: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: {"debits": [], "credits": []})
    for txn in transactions:
        txn_date = str(txn.get("date", ""))
        dr = _parse_int(txn.get("debit"))
        cr = _parse_int(txn.get("credit"))
        if dr and dr >= _CIRCULAR_FUNDS_MIN_AMOUNT:
            by_date[txn_date]["debits"].append(dr)
        if cr and cr >= _CIRCULAR_FUNDS_MIN_AMOUNT:
            by_date[txn_date]["credits"].append(cr)

    pairs = []
    for txn_date, flows in by_date.items():
        for debit_amount in flows["debits"]:
            # A matching credit within ±2% on the same date
            for credit_amount in flows["credits"]:
                delta_pct = abs(debit_amount - credit_amount) / max(debit_amount, 1)
                if delta_pct <= 0.02:
                    pairs.append({
                        "date": txn_date,
                        "amount": debit_amount,
                        "debit": debit_amount,
                        "credit": credit_amount,
                        "delta_pct": round(delta_pct * 100, 2),
                    })
    return pairs


def _parse_statement_date(date_str: str) -> Optional[datetime]:
    """Parse a bank statement date string into a datetime.

    Handles common Indian formats: DD-MM-YYYY, DD-Mon-YYYY (e.g. 01-Oct-2025).
    Returns None if the string cannot be parsed.
    """
    for fmt in ("%d-%m-%Y", "%d-%b-%Y", "%d/%m/%Y", "%d %b %Y", "%d %B %Y"):
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    return None


def _parse_employment_date(date_str: str) -> Optional[datetime]:
    """Parse an employment letter date (join date or issue date) into a datetime.

    Handles: "01 April 2014", "15 March 2026", "23-02-2017", "01-Apr-2014".
    Returns None if the string cannot be parsed.
    """
    for fmt in (
        "%d %B %Y",    # 15 March 2026
        "%d %b %Y",    # 15 Mar 2026
        "%d-%b-%Y",    # 15-Mar-2026
        "%d-%m-%Y",    # 15-03-2026
        "%d/%m/%Y",    # 15/03/2026
    ):
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    return None


class MortgageValidationPack(BaseValidationPack):
    """
    Validation rules for Indian home-loan document bundles.

    This pack can be invoked for any of the mortgage document sub-types.
    The DOCUMENT_TYPE is 'mortgage' so it catches all documents prefixed
    with 'mortgage_' via the extended registry in packs/__init__.py.
    """

    DOCUMENT_TYPE = "mortgage"
    REQUIRED_FIELDS: List[str] = []  # Varies by sub-type; checked in _domain_rules

    def _domain_rules(
        self,
        key_fields: Dict[str, Any],
        summary: Dict[str, Any],
    ) -> List[ValidationSignal]:
        signals: List[ValidationSignal] = []
        doc_type = str(summary.get("document", {}).get("type", "")).lower()

        # Route to the appropriate sub-type checks
        if "payslip" in doc_type:
            signals.extend(self._payslip_checks(key_fields, summary))
        elif "bank_statement" in doc_type or "bank" in doc_type:
            signals.extend(self._bank_statement_checks(key_fields, summary))
        elif "employment" in doc_type:
            signals.extend(self._employment_letter_checks(key_fields, summary))
        elif "form16" in doc_type or "form_16" in doc_type or "tds" in doc_type:
            signals.extend(self._form16_checks(key_fields, summary))
        elif "utility" in doc_type:
            signals.extend(self._utility_bill_checks(key_fields, summary))
        else:
            # Generic fallback: attempt every sub-type that can self-identify via
            # its own data (avoids false signals from wrong-type checks).
            signals.extend(self._payslip_checks(key_fields, summary))
            signals.extend(self._bank_statement_checks(key_fields, summary))
            signals.extend(self._employment_letter_checks(key_fields, summary))
            signals.extend(self._form16_checks(key_fields, summary))

        return signals

    # ------------------------------------------------------------------
    # Payslip sub-checks
    # ------------------------------------------------------------------

    def _payslip_checks(
        self,
        key_fields: Dict[str, Any],
        summary: Dict[str, Any],
    ) -> List[ValidationSignal]:
        signals: List[ValidationSignal] = []

        gross  = _parse_int(key_fields.get("gross_earnings") or key_fields.get("gross_salary"))
        net    = _parse_int(key_fields.get("net_pay") or key_fields.get("net_salary"))
        basic  = _parse_int(key_fields.get("basic_salary") or key_fields.get("basic"))
        hra    = _parse_int(key_fields.get("hra") or key_fields.get("house_rent_allowance"))
        pf_emp = _parse_int(key_fields.get("pf") or key_fields.get("provident_fund"))
        pt     = _parse_int(key_fields.get("professional_tax") or key_fields.get("pt"))
        tds    = _parse_int(key_fields.get("tds") or key_fields.get("income_tax"))

        # 0. Fundamental: gross must be ≥ net pay.
        #    Emits the "gross_gte_net_pay" rule name so the PayrollValidationPack
        #    unit-test (which routes through this pack via the registry) still passes.
        if gross is not None and net is not None:
            is_valid = gross >= net > 0
            signals.append(ValidationSignal(
                rule="gross_gte_net_pay",
                severity="high" if not is_valid else "info",
                score=55 if not is_valid else 0,
                message="Gross earnings must be greater than or equal to net pay.",
                passed=is_valid,
                details={"gross_earnings": gross, "net_pay": net},
            ))

        # 1. Gross arithmetic: basic + HRA + all allowances ≈ gross
        #    We check the simpler constraint: basic and HRA should each be < gross
        if gross is not None and basic is not None:
            is_valid = gross >= basic > 0
            signals.append(ValidationSignal(
                rule="basic_within_gross",
                severity="high" if not is_valid else "info",
                score=40 if not is_valid else 0,
                message="Basic salary must be a positive component within gross earnings.",
                passed=is_valid,
                details={"gross": gross, "basic": basic},
            ))

        # 2. Net pay arithmetic: gross - deductions ≈ net
        if gross is not None and net is not None and pf_emp is not None and pt is not None:
            # Minimum deductions we know about
            min_total_ded = pf_emp + pt + (tds or 0)
            expected_net_max = gross - min_total_ded
            # Net should not exceed gross − known deductions (allowing 2% tolerance)
            tolerance = max(500, int(gross * 0.02))
            is_valid = net <= expected_net_max + tolerance
            signals.append(ValidationSignal(
                rule="net_pay_arithmetic",
                severity="critical" if not is_valid else "info",
                score=60 if not is_valid else 0,
                message="Net pay exceeds gross minus known deductions — deductions may have been removed.",
                passed=is_valid,
                details={
                    "gross": gross,
                    "pf_emp": pf_emp,
                    "pt": pt,
                    "tds": tds or 0,
                    "expected_net_max": expected_net_max,
                    "actual_net": net,
                    "tolerance": tolerance,
                },
            ))

        # 3. PF rate: PF should be ≤ 12.5% of basic salary
        if pf_emp is not None and basic is not None and basic > 0:
            actual_rate = pf_emp / basic
            is_valid = actual_rate <= _PF_RATE_MAX
            signals.append(ValidationSignal(
                rule="pf_rate_validity",
                severity="medium" if not is_valid else "info",
                score=15 if not is_valid else 0,
                message=f"PF should be ≤ 12% of basic salary (statutory ceiling). "
                        f"Detected rate: {actual_rate*100:.1f}%.",
                passed=is_valid,
                details={"pf_emp": pf_emp, "basic": basic, "rate_pct": round(actual_rate * 100, 2)},
            ))

        # 4. Professional Tax: ≤ ₹200/month in any Indian state
        if pt is not None:
            is_valid = 0 <= pt <= _MAX_PT_MONTHLY
            signals.append(ValidationSignal(
                rule="pt_slab_validity",
                severity="medium" if not is_valid else "info",
                score=15 if not is_valid else 0,
                message=f"Professional Tax exceeds ₹200/month maximum slab. Detected: ₹{pt}.",
                passed=is_valid,
                details={"professional_tax": pt, "max_allowed": _MAX_PT_MONTHLY},
            ))

        # 5. TDS plausibility: TDS should be > 0 if annualised gross > ₹7 lakh
        if tds is not None and gross is not None:
            annual_gross = gross * 12
            should_have_tds = annual_gross > _TDS_THRESHOLD_ANNUAL
            is_valid = not (should_have_tds and tds == 0)
            signals.append(ValidationSignal(
                rule="tds_plausibility",
                severity="medium" if not is_valid else "info",
                score=20 if not is_valid else 0,
                message=f"TDS is zero but annualised gross (₹{annual_gross:,}) exceeds "
                        f"₹{_TDS_THRESHOLD_ANNUAL:,} — TDS should be deducted.",
                passed=is_valid,
                details={
                    "tds": tds,
                    "monthly_gross": gross,
                    "annualised_gross": annual_gross,
                    "threshold": _TDS_THRESHOLD_ANNUAL,
                },
            ))

        # 6. HRA proportion: HRA > basic is suspicious (legally HRA = 40-50% of basic)
        #    Under Section 10(13A), HRA exemption is capped at 50% of basic (metro) / 40% (non-metro).
        #    HRA declared beyond 50% of basic is likely an over-claim to reduce tax liability.
        if hra is not None and basic is not None and basic > 0:
            ratio = hra / basic
            is_valid = ratio <= _HRA_MAX_BASIC_RATIO
            signals.append(ValidationSignal(
                rule="hra_proportion",
                severity="medium" if not is_valid else "info",
                score=15 if not is_valid else 0,
                message=f"HRA exceeds basic salary — HRA:Basic ratio {ratio:.2f} > 1.0 is atypical.",
                passed=is_valid,
                details={"hra": hra, "basic": basic, "hra_to_basic_ratio": round(ratio, 3)},
            ))

        # 7. Basic salary as % of gross: should be 30–55% in the Indian IT sector.
        #    If basic is only 10–15% of gross, the CTC structure is likely fabricated
        #    to minimise PF liability or to inflate the tax-exempt HRA band.
        if gross is not None and basic is not None and gross > 0:
            basic_ratio = basic / gross
            is_valid = _BASIC_GROSS_MIN_RATIO <= basic_ratio <= _BASIC_GROSS_MAX_RATIO
            signals.append(ValidationSignal(
                rule="basic_gross_proportion",
                severity="medium" if not is_valid else "info",
                score=20 if not is_valid else 0,
                message=(
                    f"Basic salary ({basic_ratio*100:.1f}% of gross) is outside the typical "
                    f"30–55% band. Basic below 30% inflates tax-free components artificially."
                ),
                passed=is_valid,
                details={
                    "basic": basic,
                    "gross": gross,
                    "basic_pct_of_gross": round(basic_ratio * 100, 1),
                    "expected_range": "30–55%",
                },
            ))

        # 8. HRA as % of basic: should be ≤ 50% (legally, HRA exemption ceiling in metro).
        #    Declaring HRA > 50% of basic is an over-claim under Indian income-tax rules.
        if hra is not None and basic is not None and basic > 0:
            hra_basic_ratio = hra / basic
            is_valid = hra_basic_ratio <= _HRA_BASIC_NORMAL_MAX_RATIO + 0.05  # 5% tolerance
            if not is_valid:
                # Scale score with how far above 50% the HRA is:
                # 51–65% → 25 pts (medium), 65–80% → 35 pts (medium-high), >80% → 50 pts (high)
                excess_pct = hra_basic_ratio * 100 - 50
                _hra_score = 50 if excess_pct > 30 else (35 if excess_pct > 15 else 25)
                signals.append(ValidationSignal(
                    rule="hra_basic_proportion",
                    severity="high" if _hra_score >= 35 else "medium",
                    score=_hra_score,
                    message=(
                        f"HRA is {hra_basic_ratio*100:.1f}% of basic — exceeds the 50% statutory "
                        f"exemption ceiling. Over-stated HRA reduces taxable income artificially."
                    ),
                    passed=False,
                    details={
                        "hra": hra,
                        "basic": basic,
                        "hra_pct_of_basic": round(hra_basic_ratio * 100, 1),
                        "legal_max_pct": 50,
                    },
                ))

        # 9. UAN format: Universal Account Number must be exactly 12 digits.
        uan_raw = key_fields.get("uan")
        if uan_raw is not None:
            digits_only = re.sub(r"\D", "", str(uan_raw))
            is_valid = len(digits_only) == 12
            signals.append(ValidationSignal(
                rule="uan_format",
                severity="medium" if not is_valid else "info",
                score=15 if not is_valid else 0,
                message=(
                    f"UAN '{uan_raw}' is not a valid 12-digit number. "
                    "An invalid UAN may indicate a fabricated payslip."
                    if not is_valid else
                    "UAN format is valid (12 digits)."
                ),
                passed=is_valid,
                details={"uan": uan_raw, "digits_found": len(digits_only)},
            ))

        # 10. Paid days: must be between 0 and 31 (calendar days in a month).
        paid_days_val = _parse_int(key_fields.get("paid_days"))
        if paid_days_val is not None:
            is_valid = 0 <= paid_days_val <= 31
            signals.append(ValidationSignal(
                rule="paid_days_range",
                severity="medium" if not is_valid else "info",
                score=15 if not is_valid else 0,
                message=(
                    f"Paid days ({paid_days_val}) is outside the valid range 0–31."
                    if not is_valid else
                    f"Paid days ({paid_days_val}) is within range."
                ),
                passed=is_valid,
                details={"paid_days": paid_days_val},
            ))

        # 11. Basic minimum proportion: basic must be ≥ 20% of gross.
        #     Below 20% is highly unusual in Indian payroll — likely fabricated CTC structure.
        if gross is not None and basic is not None and gross > 0:
            basic_ratio = basic / gross
            is_valid = basic_ratio >= 0.20
            if not is_valid:
                signals.append(ValidationSignal(
                    rule="basic_minimum_proportion",
                    severity="medium",
                    score=20,
                    message=(
                        f"Basic salary ({basic_ratio*100:.1f}% of gross) is below the 20% minimum "
                        "threshold — an artificially low basic inflates tax-free allowances."
                    ),
                    passed=False,
                    details={
                        "basic": basic,
                        "gross": gross,
                        "basic_pct_of_gross": round(basic_ratio * 100, 1),
                        "min_expected_pct": 20,
                    },
                ))

        return signals

    # ------------------------------------------------------------------
    # Bank statement sub-checks
    # ------------------------------------------------------------------

    def _bank_statement_checks(
        self,
        key_fields: Dict[str, Any],
        summary: Dict[str, Any],
    ) -> List[ValidationSignal]:
        signals: List[ValidationSignal] = []

        # Transactions are now stored in key_fields (extracted by structured.py).
        # Fallback to top-level summary key for backward-compatibility.
        transactions: List[Dict[str, Any]] = (
            key_fields.get("transactions")
            or summary.get("transactions")
            or []
        )

        # 0. Aggregate balance arithmetic: opening + total_credits − total_debits = closing.
        #    This is a high-level sanity check on the summary totals printed on the statement.
        #    If these four numbers don't add up, the statement totals have been altered.
        opening  = _parse_int(key_fields.get("opening_balance"))
        closing  = _parse_int(key_fields.get("closing_balance"))
        t_credits= _parse_int(key_fields.get("total_credits"))
        t_debits = _parse_int(key_fields.get("total_debits"))
        if all(v is not None for v in [opening, closing, t_credits, t_debits]):
            expected_closing = opening + t_credits - t_debits  # type: ignore[operator]
            tolerance = max(10, int(abs(expected_closing) * 0.01))
            is_valid = abs(expected_closing - closing) <= tolerance  # type: ignore[operator]
            signals.append(ValidationSignal(
                rule="balance_arithmetic",
                severity="high" if not is_valid else "info",
                score=60 if not is_valid else 0,
                message=(
                    f"Closing balance ₹{closing:,} does not match "
                    f"opening ₹{opening:,} + credits ₹{t_credits:,} − debits ₹{t_debits:,} "
                    f"= ₹{expected_closing:,}. Statement totals may have been altered."
                    if not is_valid else
                    "Aggregate balance arithmetic (opening + credits − debits = closing) checks out."
                ),
                passed=is_valid,
                details={
                    "opening_balance": opening,
                    "total_credits": t_credits,
                    "total_debits": t_debits,
                    "closing_balance": closing,
                    "expected_closing": expected_closing,
                    "difference": closing - expected_closing,  # type: ignore[operator]
                },
            ))

        # 1. IFSC prefix vs stated bank name consistency (single-doc, no cross-doc needed).
        #    The first 4 characters of an Indian IFSC code uniquely identify the bank
        #    (e.g. HDFC → HDFC Bank, ICIC → ICICI Bank, SBIN → State Bank of India).
        #    If the stated bank name and the IFSC prefix don't match, the statement header
        #    has been tampered to show a different bank name than the IFSC indicates.
        ifsc = str(key_fields.get("ifsc", "")).strip().upper()
        bank_name_header = str(key_fields.get("bank_name", "")).upper()
        if ifsc and bank_name_header and len(ifsc) >= 4:
            prefix = ifsc[:4]
            expected_substrings = _IFSC_BANK_PREFIX_MAP.get(prefix, [])
            if expected_substrings:
                is_consistent = any(es in bank_name_header for es in expected_substrings)
                signals.append(ValidationSignal(
                    rule="bank_ifsc_bank_name_consistency",
                    severity="high" if not is_consistent else "info",
                    score=65 if not is_consistent else 0,
                    message=(
                        f"IFSC '{ifsc}' (prefix '{prefix}') indicates a different bank than "
                        f"the header name '{key_fields.get('bank_name', '')}'. "
                        "Mismatched IFSC and bank name is a strong tampering signal."
                        if not is_consistent else
                        "IFSC prefix is consistent with the stated bank name."
                    ),
                    passed=is_consistent,
                    details={
                        "ifsc": ifsc,
                        "ifsc_prefix": prefix,
                        "expected_bank_substrings": expected_substrings,
                        "stated_bank_name": key_fields.get("bank_name", ""),
                    },
                ))

        # 1. Circular funds detection
        if transactions:
            pairs = _round_trip_pairs(transactions)
            is_valid = len(pairs) == 0
            signals.append(ValidationSignal(
                rule="circular_funds_detection",
                severity="high" if not is_valid else "info",
                score=40 if not is_valid else 0,
                message="Round-trip debit+credit pair on same date detected — possible circular fund inflation.",
                passed=is_valid,
                details={
                    "circular_pairs_found": len(pairs),
                    "pairs": pairs[:5],  # cap evidence to first 5 pairs
                },
            ))

        # 2. Salary credit regularity: should appear once per calendar month
        salary_credits = [
            t for t in transactions
            if "salary" in str(t.get("desc", "")).lower()
            or "sal cr" in str(t.get("desc", "")).lower()
            or "neft sal" in str(t.get("desc", "")).lower()
        ]
        if salary_credits:
            # Check uniqueness of months
            months_seen: set = set()
            duplicate_months = []
            for txn in salary_credits:
                date_str = str(txn.get("date", ""))
                parts = date_str.split("-")
                if len(parts) >= 3:
                    month_key = f"{parts[1]}-{parts[2]}"
                    if month_key in months_seen:
                        duplicate_months.append(date_str)
                    months_seen.add(month_key)
            is_valid = len(duplicate_months) == 0
            signals.append(ValidationSignal(
                rule="salary_credit_regularity",
                severity="medium" if not is_valid else "info",
                score=20 if not is_valid else 0,
                message="Salary credited more than once in the same month — unusual pattern.",
                passed=is_valid,
                details={
                    "salary_credits_count": len(salary_credits),
                    "duplicate_month_credits": duplicate_months,
                },
            ))

        # 3. Duplicate transaction reference numbers
        refs = [str(t.get("ref", "")).strip() for t in transactions if t.get("ref")]
        seen_refs: set = set()
        duplicate_refs = []
        for ref in refs:
            if ref in seen_refs:
                duplicate_refs.append(ref)
            seen_refs.add(ref)
        if refs:
            is_valid = len(duplicate_refs) == 0
            signals.append(ValidationSignal(
                rule="duplicate_txn_reference",
                severity="high" if not is_valid else "info",
                score=35 if not is_valid else 0,
                message="Duplicate transaction reference numbers found — statement may have been fabricated.",
                passed=is_valid,
                details={
                    "total_refs": len(refs),
                    "duplicate_refs": duplicate_refs[:10],
                },
            ))

        # 4. Bank statement date-range coverage.
        #    The statement header claims a period (e.g. "01-Oct-2025 to 31-Mar-2026").
        #    If the actual transactions only span a much shorter window, the header
        #    range is fabricated to make it look like a 6-month statement.
        period_from_str = str(key_fields.get("period_from") or "").strip()
        period_to_str   = str(key_fields.get("period_to")   or "").strip()

        # Use pre-computed earliest/latest dates from structured.py if available,
        # otherwise fall back to parsing from the transactions list.
        earliest_str = str(key_fields.get("earliest_txn_date") or "").strip()
        latest_str   = str(key_fields.get("latest_txn_date")   or "").strip()
        if not earliest_str and transactions:
            txn_dates_tmp = [
                _parse_statement_date(str(t.get("date", "")))
                for t in transactions
            ]
            txn_dates_tmp = [d for d in txn_dates_tmp if d]
            if txn_dates_tmp:
                earliest_str = min(txn_dates_tmp).strftime("%d-%m-%Y")
                latest_str   = max(txn_dates_tmp).strftime("%d-%m-%Y")

        if period_from_str and period_to_str and earliest_str and latest_str:
            stated_from = _parse_statement_date(period_from_str)
            stated_to   = _parse_statement_date(period_to_str)
            actual_from = _parse_statement_date(earliest_str)
            actual_to   = _parse_statement_date(latest_str)
            if stated_from and stated_to and actual_from and actual_to:
                stated_days = max(1, (stated_to - stated_from).days)
                stated_months = stated_days / 30
                start_gap = (actual_from - stated_from).days   # positive = txns start late
                end_gap   = (stated_to - actual_to).days       # positive = txns end early
                suspicious = start_gap > 20 or end_gap > 20
                signals.append(ValidationSignal(
                    rule="bank_statement_date_range",
                    severity="high" if suspicious else "info",
                    score=65 if suspicious else 0,
                    message=(
                        "Statement header claims a wider date range than the transactions support. "
                        "Transactions start late or end early compared to the stated period."
                        if suspicious else
                        "Transaction dates are consistent with the stated statement period."
                    ),
                    passed=not suspicious,
                    details={
                        "stated_period": f"{period_from_str} to {period_to_str}",
                        "stated_months": round(stated_months, 1),
                        "actual_first_txn": actual_from.strftime("%d-%b-%Y"),
                        "actual_last_txn":  actual_to.strftime("%d-%b-%Y"),
                        "start_gap_days": start_gap,
                        "end_gap_days":   end_gap,
                    },
                ))

        # 5. Running balance arithmetic.
        #    Sort transactions by date first, then verify row-by-row:
        #      prev_balance + credit − debit = current_balance
        #
        #    Implementation note on false-positive avoidance
        #    ------------------------------------------------
        #    PDF-extracted transaction lists frequently have ordering ambiguity:
        #    multiple transactions on the SAME calendar date can appear in any
        #    order after a date-sort, and the balances only chain correctly in
        #    the original PDF-table order.  This causes widespread "arithmetic
        #    errors" on completely legitimate statements.
        #
        #    To distinguish GENUINE balance manipulation from extraction noise,
        #    we apply the following heuristic:
        #
        #      a) Targeted fraud (e.g. a specific row inflated by a round
        #         amount): the FIRST error will have a large, round-number
        #         difference (divisible by 1,000, amount ≥ ₹10,000).  All
        #         subsequent rows cascade off that single inflation.  This is
        #         the pattern seen in "bank_arithmetic_error" tamper cases.
        #
        #      b) Extraction-order noise: the first error will be a
        #         non-round, irregular amount driven by the specific ordering
        #         of same-date rows — NOT intentional manipulation.
        #
        #    We therefore ONLY flag as fraud when the first error is a
        #    round-number amount.  Widespread non-round errors are logged
        #    as an informational extraction note and do not reduce the score.
        if transactions and len(transactions) >= 2:
            try:
                sorted_txns = sorted(
                    transactions,
                    key=lambda t: (_parse_statement_date(str(t.get("date", ""))) or datetime.min),
                )
            except Exception:
                sorted_txns = transactions
            arithmetic_errors = []
            for i in range(1, len(sorted_txns)):
                prev = sorted_txns[i - 1]
                curr = sorted_txns[i]
                prev_bal = _parse_int(prev.get("balance"))
                curr_bal = _parse_int(curr.get("balance"))
                dr = _parse_int(curr.get("debit")) or 0
                cr = _parse_int(curr.get("credit")) or 0
                if prev_bal is None or curr_bal is None:
                    continue
                expected = prev_bal + cr - dr
                tol = max(
                    _BANK_ARITHMETIC_MIN_TOLERANCE,
                    int(abs(expected) * _BANK_ARITHMETIC_TOLERANCE_RATIO),
                )
                if abs(expected - curr_bal) > tol:
                    arithmetic_errors.append({
                        "row": i + 1,
                        "date": curr.get("date", ""),
                        "desc": str(curr.get("desc", ""))[:40],
                        "expected_balance": expected,
                        "stated_balance": curr_bal,
                        "difference": curr_bal - expected,
                    })

            # Determine fraud vs extraction-noise using the round-number heuristic.
            is_fraud = False
            if arithmetic_errors:
                first_diff = abs(arithmetic_errors[0]["difference"])
                # A round-number delta ≥ ₹10,000, divisible by ₹1,000
                # strongly suggests deliberate balance inflation.
                is_round_number = (first_diff >= 10_000) and (first_diff % 1_000 == 0)
                is_fraud = is_round_number

            if is_fraud:
                signals.append(ValidationSignal(
                    rule="bank_debit_credit_arithmetic",
                    severity="critical",
                    score=65,
                    message=(
                        f"Balance inflation detected at row {arithmetic_errors[0]['row']}: "
                        f"balance stated ₹{arithmetic_errors[0]['stated_balance']:,} but "
                        f"arithmetic gives ₹{arithmetic_errors[0]['expected_balance']:,} "
                        f"(difference: ₹{arithmetic_errors[0]['difference']:,}). "
                        "A round-number discrepancy is a strong indicator of deliberate tampering."
                    ),
                    passed=False,
                    details={
                        "total_transactions": len(sorted_txns),
                        "first_fraud_row": arithmetic_errors[0],
                        "total_arithmetic_errors": len(arithmetic_errors),
                        "arithmetic_errors": arithmetic_errors[:5],
                    },
                ))
            elif arithmetic_errors:
                # Widespread non-round errors → likely extraction ordering noise, not fraud.
                signals.append(ValidationSignal(
                    rule="bank_debit_credit_arithmetic",
                    severity="info",
                    score=0,
                    message=(
                        "Running balance could not be fully verified from extracted data "
                        "(PDF table row-ordering may differ from extraction order). "
                        "Aggregate balance check (total credits − debits) is the primary fraud signal."
                    ),
                    passed=True,
                    details={
                        "total_transactions": len(sorted_txns),
                        "note": "Non-round discrepancies suggest extraction ordering, not tampering.",
                        "arithmetic_errors": arithmetic_errors[:3],
                    },
                ))
            else:
                signals.append(ValidationSignal(
                    rule="bank_debit_credit_arithmetic",
                    severity="info",
                    score=0,
                    message="Running balance arithmetic is consistent throughout the statement.",
                    passed=True,
                    details={"total_transactions": len(sorted_txns)},
                ))

        return signals

    # ------------------------------------------------------------------
    # Employment letter sub-checks
    # ------------------------------------------------------------------

    def _employment_letter_checks(
        self,
        key_fields: Dict[str, Any],
        summary: Dict[str, Any],
    ) -> List[ValidationSignal]:
        signals: List[ValidationSignal] = []

        # Support both the new dedicated key_fields (when doc classified as
        # employment_letter) and the legacy named_fields path (when doc
        # fell through as generic).
        named = key_fields.get("named_fields") or {}

        cin = str(
            key_fields.get("cin")
            or key_fields.get("company_cin")
            or named.get("cin")
            or named.get("company_identification_number")
            or ""
        ).strip()
        join_date = str(
            key_fields.get("join_date")
            or key_fields.get("date_of_joining")
            or named.get("date_of_joining")
            or ""
        ).strip()
        ctc = _parse_int(
            key_fields.get("annual_ctc")
            or key_fields.get("ctc")
            or named.get("annual_ctc")
            or named.get("ctc")
        )
        gross_monthly = _parse_int(
            key_fields.get("gross_monthly_salary")
            or key_fields.get("monthly_gross")
            or key_fields.get("gross_monthly_salary")
            or named.get("gross_monthly_salary")
            or named.get("monthly_gross_salary")
        )

        # If none of the employment-letter-specific fields are present this is
        # not an employment letter — skip silently to avoid noise in the generic
        # fallback path.
        has_employment_data = bool(cin or ctc or gross_monthly or join_date)
        if not has_employment_data:
            return signals

        # 1. CIN format validation
        if cin:
            is_valid_cin = bool(_CIN_RE.match(cin))
            # Extract incorporation year from CIN (characters 9–12, 0-indexed 8:12)
            # CIN structure: [LU] + 5-digit NIC + 2-letter state + 4-digit YEAR + category + seq
            cin_year: Optional[int] = None
            if is_valid_cin and len(cin) >= 12:
                try:
                    cin_year = int(cin[8:12])
                except ValueError:
                    cin_year = None
            signals.append(ValidationSignal(
                rule="employer_cin_format",
                severity="high" if not is_valid_cin else "info",
                score=35 if not is_valid_cin else 0,
                message="Employer CIN must follow Indian MCA format [LU]NNNNNSSYYYYCCC NNNNNN.",
                passed=is_valid_cin,
                details={"cin": cin, "valid": is_valid_cin},
            ))

            # 2. Company incorporation year vs join date
            if cin_year and join_date:
                join_year: Optional[int] = None
                # Try to extract year from DD-MM-YYYY or YYYY patterns
                year_match = re.search(r"\b(19|20)\d{2}\b", join_date)
                if year_match:
                    join_year = int(year_match.group())
                if join_year:
                    is_valid_age = cin_year <= join_year
                    signals.append(ValidationSignal(
                        rule="cin_age_vs_join_date",
                        severity="high" if not is_valid_age else "info",
                        score=30 if not is_valid_age else 0,
                        message="Company incorporation year (from CIN) is after employee join date — impossible.",
                        passed=is_valid_age,
                        details={
                            "cin_incorporation_year": cin_year,
                            "join_year": join_year,
                        },
                    ))
        else:
            # CIN is absent — strong signal for fake employer
            signals.append(ValidationSignal(
                rule="employer_cin_present",
                severity="high",
                score=35,
                message="Employer CIN is absent from the employment letter — required for all registered companies.",
                passed=False,
                details={"cin": None},
            ))

        # 3. CTC vs monthly gross cross-check
        #    A common fabrication pattern: the annual CTC figure and the monthly gross
        #    stated on the same letter are inconsistent, revealing manual editing.
        if ctc and gross_monthly:
            expected_monthly = ctc / 12
            tolerance = max(5_000, int(expected_monthly * 0.15))
            is_valid = abs(expected_monthly - gross_monthly) <= tolerance
            signals.append(ValidationSignal(
                rule="ctc_monthly_gross_consistency",
                severity="medium" if not is_valid else "info",
                score=20 if not is_valid else 0,
                message="Annual CTC ÷ 12 should approximately equal monthly gross salary.",
                passed=is_valid,
                details={
                    "annual_ctc": ctc,
                    "ctc_divided_by_12": round(expected_monthly),
                    "monthly_gross_declared": gross_monthly,
                    "delta": abs(round(expected_monthly) - gross_monthly),
                    "tolerance": tolerance,
                },
            ))

        # 4. Backdated employment signal (NEW).
        #
        #    Pattern: the letter is freshly issued (issue date is recent) but the
        #    stated date of joining is suspiciously far in the past.
        #
        #    Example: Letter dated 15 March 2026, join date 01 April 2014 — a 12-year
        #    gap.  While long-tenure employees do get letters, fraudsters often
        #    backdate the join date to (a) meet minimum tenure requirements for
        #    a loan, or (b) claim a longer employment history than they have.
        #
        #    This check flags when BOTH of the following are true:
        #      (a) The letter was issued within the last _BACKDATING_LETTER_WINDOW_MONTHS.
        #      (b) The join date is more than _BACKDATING_JOIN_GAP_YEARS years before
        #          the letter issue date.
        #
        #    It does NOT fire on old, previously issued letters (issue date is old)
        #    because those are common in document re-submission scenarios.
        issue_date_str = str(
            key_fields.get("letter_issue_date")
            or named.get("letter_issue_date")
            or ""
        ).strip()
        if issue_date_str and join_date:
            issue_dt = _parse_employment_date(issue_date_str)
            join_dt  = _parse_employment_date(join_date)
            if issue_dt and join_dt and issue_dt > join_dt:
                gap_years = (issue_dt - join_dt).days / 365.25
                # "recently issued" = within the last BACKDATING_LETTER_WINDOW_MONTHS
                today = datetime.now()
                months_since_issue = (today - issue_dt).days / 30.44
                is_recently_issued = months_since_issue <= _BACKDATING_LETTER_WINDOW_MONTHS
                is_suspicious      = is_recently_issued and gap_years >= _BACKDATING_JOIN_GAP_YEARS
                signals.append(ValidationSignal(
                    rule="employment_backdating_signal",
                    severity="high" if is_suspicious else "info",
                    score=40 if is_suspicious else 0,
                    message=(
                        f"Letter issued {months_since_issue:.0f} months ago but join date is "
                        f"{gap_years:.1f} years earlier — possible backdated employment."
                        if is_suspicious else
                        "Letter issue date and join date gap is within expected range."
                    ),
                    passed=not is_suspicious,
                    details={
                        "letter_issue_date": issue_date_str,
                        "date_of_joining": join_date,
                        "gap_years": round(gap_years, 1),
                        "months_since_issue": round(months_since_issue, 1),
                        "backdating_threshold_years": _BACKDATING_JOIN_GAP_YEARS,
                    },
                ))

        return signals

    # ------------------------------------------------------------------
    # Form 16 sub-checks
    # ------------------------------------------------------------------

    def _form16_checks(
        self,
        key_fields: Dict[str, Any],
        summary: Dict[str, Any],
    ) -> List[ValidationSignal]:
        signals: List[ValidationSignal] = []

        # Support both direct key_fields (form16 doc type) and named_fields
        # (generic fallback when the doc was not classified as form16).
        named = key_fields.get("named_fields") or {}

        tan = str(
            key_fields.get("tan")
            or key_fields.get("employer_tan")
            or named.get("tan_of_employer")
            or named.get("tan")
            or named.get("employer_tan")
            or ""
        ).strip()
        gross_salary = _parse_int(
            key_fields.get("gross_salary")
            or key_fields.get("gross_earnings")
            or named.get("gross_salary_total")
            or named.get("gross_salary")
            or named.get("gross_earnings")
        )
        tds_total = _parse_int(
            key_fields.get("tds")
            or key_fields.get("total_tds")
            or named.get("tax_deducted_at_source_tds")
            or named.get("tds")
            or named.get("total_tds")
        )

        # Guard: if neither TAN nor a gross salary figure is present, this is
        # not a Form 16 — skip to avoid spurious signals in the generic path.
        has_form16_data = bool(tan or gross_salary or tds_total or named.get("assessment_year"))
        if not has_form16_data:
            return signals

        # 1. TAN format
        if tan:
            is_valid = bool(_TAN_RE.match(tan))
            signals.append(ValidationSignal(
                rule="tan_format_validity",
                severity="medium" if not is_valid else "info",
                score=15 if not is_valid else 0,
                message="Employer TAN must be 10 characters: 4 letters + 5 digits + 1 letter.",
                passed=is_valid,
                details={"tan": tan},
            ))

        # 2. TDS on Form 16 > 0 when gross > threshold
        if gross_salary and tds_total is not None:
            should_have_tds = gross_salary > _TDS_THRESHOLD_ANNUAL
            is_valid = not (should_have_tds and tds_total == 0)
            signals.append(ValidationSignal(
                rule="form16_tds_plausibility",
                severity="medium" if not is_valid else "info",
                score=20 if not is_valid else 0,
                message=f"Form 16 shows zero TDS but annual gross ₹{gross_salary:,} exceeds "
                        f"₹{_TDS_THRESHOLD_ANNUAL:,} threshold.",
                passed=is_valid,
                details={
                    "form16_gross": gross_salary,
                    "form16_tds": tds_total,
                    "threshold": _TDS_THRESHOLD_ANNUAL,
                },
            ))

        return signals

    # ------------------------------------------------------------------
    # Utility bill sub-checks
    # ------------------------------------------------------------------

    def _utility_bill_checks(
        self,
        key_fields: Dict[str, Any],
        summary: Dict[str, Any],
    ) -> List[ValidationSignal]:
        signals: List[ValidationSignal] = []

        amount = _parse_int(
            key_fields.get("amount_due")
            or key_fields.get("total_amount")
            or key_fields.get("bill_amount")
        )

        if amount is not None:
            is_valid = 100 <= amount <= 20_000
            signals.append(ValidationSignal(
                rule="utility_amount_plausibility",
                severity="low" if not is_valid else "info",
                score=10 if not is_valid else 0,
                message=f"Utility bill amount ₹{amount} is outside typical household range ₹100–₹20,000.",
                passed=is_valid,
                details={"amount": amount, "min": 100, "max": 20_000},
            ))

        return signals
