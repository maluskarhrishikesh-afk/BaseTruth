from __future__ import annotations

"""
Validation pack for banking and lending documents.

Covers: bank account statements, loan statements, credit card statements.

Checks
------
  required_fields_present   -- account_number, statement_period
  balance_arithmetic        -- opening + credits - debits == closing (within 1%)
  ifsc_format               -- IFSC code must be 11 characters (4 alpha + 7 alnum)
  overdraft_flag            -- closing balance below zero is noted for lending contexts
  circular_funds_detection  -- large round-trip debit+credit pair on same date
  duplicate_txn_reference   -- same reference number used more than once
  salary_credit_regularity  -- salary should credit exactly once per calendar month
"""

import re
from typing import Any, Dict, List

from basetruth.analysis.packs.base import BaseValidationPack, ValidationSignal, _parse_int


class BankingValidationPack(BaseValidationPack):
    """Validation rules for banking and lending documents."""

    DOCUMENT_TYPE = "bank_statement"
    REQUIRED_FIELDS = ["account_number", "statement_period"]

    def _domain_rules(
        self,
        key_fields: Dict[str, Any],
        summary: Dict[str, Any],
    ) -> List[ValidationSignal]:
        signals: List[ValidationSignal] = []

        opening = _parse_int(key_fields.get("opening_balance"))
        closing = _parse_int(key_fields.get("closing_balance"))
        credits = _parse_int(key_fields.get("total_credits"))
        debits = _parse_int(key_fields.get("total_debits"))

        # Core balance identity: opening + credits - debits = closing.
        # Tolerance of 1% or Rs 10, whichever is higher, to allow for rounding.
        if all(v is not None for v in [opening, closing, credits, debits]):
            expected = opening + credits - debits  # type: ignore[operator]
            tolerance = max(10, int(abs(expected) * 0.01))
            is_valid = abs(expected - closing) <= tolerance  # type: ignore[operator]
            signals.append(
                ValidationSignal(
                    rule="balance_arithmetic",
                    severity="high" if not is_valid else "info",
                    score=60 if not is_valid else 0,
                    message="Opening + credits - debits should equal closing balance.",
                    passed=is_valid,
                    details={
                        "expected_closing": expected,
                        "actual_closing": closing,
                        "delta": abs(expected - closing),  # type: ignore[operator]
                        "tolerance": tolerance,
                    },
                )
            )

        # IFSC format: 4 letters + 0 + 6 alphanumeric characters (11 chars total).
        ifsc = key_fields.get("ifsc_code")
        if ifsc:
            is_valid_ifsc = bool(re.match(r"^[A-Z]{4}0[A-Z0-9]{6}$", str(ifsc).upper()))
            signals.append(
                ValidationSignal(
                    rule="ifsc_format",
                    severity="low" if not is_valid_ifsc else "info",
                    score=5 if not is_valid_ifsc else 0,
                    message="IFSC code should be 11 characters (4 letters + 0 + 6 alphanumeric).",
                    passed=is_valid_ifsc,
                    details={"ifsc": ifsc},
                )
            )

        # Negative closing balance is not necessarily wrong but is worth flagging.
        if closing is not None and closing < 0:
            signals.append(
                ValidationSignal(
                    rule="overdraft_flag",
                    severity="low",
                    score=5,
                    message="Closing balance is negative (possible overdraft or data error).",
                    passed=False,
                    details={"closing_balance": closing},
                )
            )

        # ── Circular funds detection ──────────────────────────────────────────────
        # A large debit and a matching credit on the same date is a classic
        # round-trip / circular-fund indicator used to inflate the apparent
        # balance or cash-flow on a statement.
        _CIRCULAR_MIN = 100_000  # ₹1 lakh threshold
        transactions = summary.get("transactions", [])
        if transactions:
            from collections import defaultdict

            by_date: dict = defaultdict(lambda: {"debits": [], "credits": []})
            for txn in transactions:
                date_str = str(txn.get("date", ""))
                dr = _parse_int(txn.get("debit"))
                cr = _parse_int(txn.get("credit"))
                if dr and dr >= _CIRCULAR_MIN:
                    by_date[date_str]["debits"].append(dr)
                if cr and cr >= _CIRCULAR_MIN:
                    by_date[date_str]["credits"].append(cr)

            pairs = []
            for date_str, flows in by_date.items():
                for dr_amt in flows["debits"]:
                    for cr_amt in flows["credits"]:
                        delta_pct = abs(dr_amt - cr_amt) / max(dr_amt, 1)
                        if delta_pct <= 0.02:
                            pairs.append({
                                "date": date_str,
                                "debit": dr_amt,
                                "credit": cr_amt,
                                "delta_pct": round(delta_pct * 100, 2),
                            })

            is_clean = len(pairs) == 0
            signals.append(
                ValidationSignal(
                    rule="circular_funds_detection",
                    severity="high" if not is_clean else "info",
                    score=40 if not is_clean else 0,
                    message="Round-trip debit+credit pair on the same date detected — "
                            "possible circular fund inflation.",
                    passed=is_clean,
                    details={
                        "circular_pairs_found": len(pairs),
                        "pairs": pairs[:5],
                    },
                )
            )

            # ── Duplicate transaction reference numbers ───────────────────────────
            refs = [str(t.get("ref", "")).strip() for t in transactions if t.get("ref")]
            seen: set = set()
            dupes: list = []
            for ref in refs:
                if ref in seen:
                    dupes.append(ref)
                seen.add(ref)
            if refs:
                is_valid = len(dupes) == 0
                signals.append(
                    ValidationSignal(
                        rule="duplicate_txn_reference",
                        severity="high" if not is_valid else "info",
                        score=35 if not is_valid else 0,
                        message="Duplicate transaction reference number(s) found — "
                                "statement may have been fabricated.",
                        passed=is_valid,
                        details={"total_refs": len(refs), "duplicate_refs": dupes[:10]},
                    )
                )

            # ── Salary credit regularity ─────────────────────────────────────────
            salary_txns = [
                t for t in transactions
                if any(kw in str(t.get("desc", "")).lower()
                       for kw in ("salary", "sal cr", "neft sal", "salary credit"))
            ]
            if salary_txns:
                months_seen: set = set()
                dupe_months: list = []
                for txn in salary_txns:
                    date_parts = str(txn.get("date", "")).split("-")
                    if len(date_parts) >= 3:
                        month_key = f"{date_parts[1]}-{date_parts[2]}"
                        if month_key in months_seen:
                            dupe_months.append(txn.get("date"))
                        months_seen.add(month_key)
                is_regular = len(dupe_months) == 0
                signals.append(
                    ValidationSignal(
                        rule="salary_credit_regularity",
                        severity="medium" if not is_regular else "info",
                        score=20 if not is_regular else 0,
                        message="Salary credited more than once in the same calendar month.",
                        passed=is_regular,
                        details={
                            "salary_credits_total": len(salary_txns),
                            "duplicate_month_entries": dupe_months,
                        },
                    )
                )

        return signals
