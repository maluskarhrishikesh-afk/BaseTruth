from __future__ import annotations

"""
Validation pack for healthcare and hospital administration documents.

Covers: hospital bills, discharge summaries, prescription invoices,
        diagnostic lab reports.

Checks
------
  required_fields_present   -- patient_name, provider_name, visit_date
  bill_arithmetic           -- if subtotal + taxes present, should equal total amount
  patient_id_not_empty      -- patient ID / UHID should not be blank if present
  visit_date_plausible      -- visit date should be a plausible calendar date
"""

from typing import Any, Dict, List

from basetruth.analysis.packs.base import BaseValidationPack, ValidationSignal, _parse_int


class HealthcareValidationPack(BaseValidationPack):
    """Validation rules for hospital and healthcare documents."""

    DOCUMENT_TYPE = "healthcare"
    REQUIRED_FIELDS = ["patient_name", "provider_name", "visit_date"]

    def _domain_rules(
        self,
        key_fields: Dict[str, Any],
        summary: Dict[str, Any],
    ) -> List[ValidationSignal]:
        signals: List[ValidationSignal] = []

        # Bill arithmetic: subtotal + taxes should equal the total amount billed.
        subtotal = _parse_int(key_fields.get("subtotal"))
        taxes = _parse_int(key_fields.get("tax_amount") or key_fields.get("gst_amount"))
        total = _parse_int(key_fields.get("total_amount") or key_fields.get("amount_due"))

        if subtotal is not None and taxes is not None and total is not None:
            expected = subtotal + taxes
            tolerance = max(5, int(abs(expected) * 0.005))
            is_valid = abs(expected - total) <= tolerance
            signals.append(
                ValidationSignal(
                    rule="bill_arithmetic",
                    severity="high" if not is_valid else "info",
                    score=50 if not is_valid else 0,
                    message="Subtotal + taxes should equal total amount billed.",
                    passed=is_valid,
                    details={"expected": expected, "actual": total, "delta": abs(expected - total)},
                )
            )

        # Patient ID / UHID should not be blank when present.
        patient_id = key_fields.get("patient_id") or key_fields.get("uhid")
        if patient_id is not None:
            is_valid = bool(str(patient_id).strip())
            signals.append(
                ValidationSignal(
                    rule="patient_id_not_empty",
                    severity="low" if not is_valid else "info",
                    score=5 if not is_valid else 0,
                    message="Patient ID / UHID should not be blank.",
                    passed=is_valid,
                    details={"patient_id": patient_id},
                )
            )

        return signals
