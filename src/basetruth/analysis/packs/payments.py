from __future__ import annotations

"""
Validation pack for payments and fintech documents.

Covers: payment receipts, UPI transaction records, wallet statements,
        NEFT/RTGS transfer confirmations.

Checks
------
  required_fields_present     -- transaction_id, amount, payment_date
  amount_positive             -- payment amount must be > 0
  upi_id_format               -- UPI ID must follow <handle>@<bank> pattern
  transaction_ref_not_empty   -- UTR / reference number should not be blank
"""

import re
from typing import Any, Dict, List

from basetruth.analysis.packs.base import BaseValidationPack, ValidationSignal, _parse_int


class PaymentsValidationPack(BaseValidationPack):
    """Validation rules for payments and fintech documents."""

    DOCUMENT_TYPE = "payment_receipt"
    REQUIRED_FIELDS = ["transaction_id", "amount", "payment_date"]

    def _domain_rules(
        self,
        key_fields: Dict[str, Any],
        summary: Dict[str, Any],
    ) -> List[ValidationSignal]:
        signals: List[ValidationSignal] = []

        amount = _parse_int(key_fields.get("amount"))

        # Payment amount must be strictly positive.
        if amount is not None:
            is_valid = amount > 0
            signals.append(
                ValidationSignal(
                    rule="amount_positive",
                    severity="high" if not is_valid else "info",
                    score=50 if not is_valid else 0,
                    message="Payment amount must be greater than zero.",
                    passed=is_valid,
                    details={"amount": amount},
                )
            )

        # UPI ID format: anything@anything (basic sanity check).
        upi_id = key_fields.get("upi_id")
        if upi_id:
            is_valid_upi = bool(re.match(r"^[a-zA-Z0-9.\-_+]+@[a-zA-Z0-9]+$", str(upi_id)))
            signals.append(
                ValidationSignal(
                    rule="upi_id_format",
                    severity="low" if not is_valid_upi else "info",
                    score=5 if not is_valid_upi else 0,
                    message="UPI ID should follow the format handle@bank.",
                    passed=is_valid_upi,
                    details={"upi_id": upi_id},
                )
            )

        # UTR / reference number should not be blank (blank may indicate a fake receipt).
        ref = key_fields.get("transaction_id") or key_fields.get("reference_number")
        if ref is not None:
            is_valid_ref = bool(str(ref).strip())
            signals.append(
                ValidationSignal(
                    rule="transaction_ref_not_empty",
                    severity="medium" if not is_valid_ref else "info",
                    score=25 if not is_valid_ref else 0,
                    message="Transaction reference/UTR number should not be empty.",
                    passed=is_valid_ref,
                    details={"reference": ref},
                )
            )

        return signals
