from __future__ import annotations

"""
Validation pack for banking and lending documents.

Covers: bank account statements, loan statements, credit card statements.

Checks
------
  required_fields_present  -- account_number, statement_period
  balance_arithmetic       -- opening + credits - debits == closing (within 1%)
  ifsc_format              -- IFSC code must be 11 characters (4 alpha + 7 alnum)
  overdraft_flag           -- closing balance below zero is noted for lending contexts
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

        return signals
