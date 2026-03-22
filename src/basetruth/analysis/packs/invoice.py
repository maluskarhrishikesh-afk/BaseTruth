from __future__ import annotations

"""
Validation pack for commercial invoices.

Covers: GST invoices, proforma invoices, tax invoices, credit notes.

Checks
------
  required_fields_present   -- invoice_number, amount_due
  invoice_amount_arithmetic -- subtotal + tax should equal amount_due
  gstin_format              -- GSTIN must be 15 chars: 2 digits + 10 alnum + 1 alnum + Z + 1 alnum
  invoice_number_not_empty  -- invoice number should not be blank
"""

import re
from typing import Any, Dict, List

from basetruth.analysis.packs.base import BaseValidationPack, ValidationSignal, _parse_int


class InvoiceValidationPack(BaseValidationPack):
    """Validation rules for GST and commercial invoices."""

    DOCUMENT_TYPE = "invoice"
    REQUIRED_FIELDS = ["invoice_number", "amount_due"]

    def _domain_rules(
        self,
        key_fields: Dict[str, Any],
        summary: Dict[str, Any],
    ) -> List[ValidationSignal]:
        signals: List[ValidationSignal] = []

        amount_due = _parse_int(key_fields.get("amount_due"))
        subtotal = _parse_int(key_fields.get("subtotal"))
        tax = _parse_int(key_fields.get("tax_amount") or key_fields.get("gst_amount"))

        # Arithmetic: subtotal + tax == total amount due.
        if amount_due is not None and subtotal is not None and tax is not None:
            expected = subtotal + tax
            is_valid = abs(expected - amount_due) <= 2
            signals.append(
                ValidationSignal(
                    rule="invoice_amount_arithmetic",
                    severity="high" if not is_valid else "info",
                    score=50 if not is_valid else 0,
                    message="Subtotal + tax should equal the total amount due.",
                    passed=is_valid,
                    details={"expected": expected, "actual": amount_due},
                )
            )

        # GSTIN format check.
        gstin = key_fields.get("gstin") or key_fields.get("gst_number")
        if gstin:
            # Standard pattern: 2 state digits + 10 PAN chars + 1 entity + Z + 1 check digit
            is_valid_gstin = bool(re.match(r"^\d{2}[A-Z]{5}\d{4}[A-Z]{1}[A-Z\d]{1}Z[A-Z\d]{1}$", str(gstin).upper()))
            signals.append(
                ValidationSignal(
                    rule="gstin_format",
                    severity="medium" if not is_valid_gstin else "info",
                    score=15 if not is_valid_gstin else 0,
                    message="GSTIN should be a 15-character code following standard format.",
                    passed=is_valid_gstin,
                    details={"gstin": gstin},
                )
            )

        return signals
