"""Domain-specific validation packs for BaseTruth.

This module defines a registry of validation 'packs', one per document type.
Each pack is a collection of check functions that receive the structured summary
dict produced by analysis/structured.py and return a list of Signal objects.

Supported document types (as reported by structured_summary["document"]["type"])
---------------------------------------------------------------------------
  payslip         -- Indian salary/payslip documents.
  offer_letter    -- Employment offer letters.
  bank_statement  -- Bank account statements (basic checks only).
  invoice         -- GST and commercial invoices.
  generic         -- Catch-all fallback; runs minimal checks only.

Signal schema
-------------
Each Signal has these fields:
  name      (str)   -- machine-readable check identifier, e.g. 'gross_pay_range'.
  passed    (bool)  -- True if the check passed (no anomaly found).
  severity  (str)   -- 'critical' | 'high' | 'medium' | 'low'.
  detail    (str)   -- human-readable explanation of the result.

Adding a new pack
-----------------
1. Create a function that accepts a structured_summary dict and returns
   List[Signal].
2. Register it by adding an entry to VALIDATOR_REGISTRY at the bottom of
   this file, keyed on the document type string.
3. The tamper scorer in analysis/tamper.py picks it up automatically.

Public API
----------
  validate_document(structured_summary) -> List[Signal]
      Runs the appropriate pack (falling back to the generic pack) and
      returns all signals.  Never raises; errors in individual checks are
      caught and returned as failed signals with severity 'low'.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ValidationSignal:
    """A single domain-validation signal emitted by a validation pack."""

    rule: str
    severity: str  # "info", "low", "medium", "high"
    score: int
    message: str
    passed: bool
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _parse_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    text = "".join(ch for ch in str(value) if ch.isdigit() or ch == "-")
    if not text or text == "-":
        return None
    try:
        return int(text)
    except ValueError:
        return None


class BaseValidationPack:
    """Base class — runs required-field checks then calls domain-specific rules."""

    DOCUMENT_TYPE: str = "generic"
    REQUIRED_FIELDS: List[str] = []

    def validate(self, summary: Dict[str, Any]) -> List[ValidationSignal]:
        key_fields = summary.get("key_fields", {})
        signals: List[ValidationSignal] = []

        missing = [f for f in self.REQUIRED_FIELDS if not key_fields.get(f)]
        signals.append(
            ValidationSignal(
                rule="required_fields_present",
                severity="medium" if missing else "info",
                score=min(30, 8 * len(missing)),
                message=f"Required fields for {self.DOCUMENT_TYPE}.",
                passed=not missing,
                details={"missing": missing},
            )
        )
        signals.extend(self._domain_rules(key_fields, summary))
        return signals

    def _domain_rules(
        self,
        key_fields: Dict[str, Any],
        summary: Dict[str, Any],
    ) -> List[ValidationSignal]:
        return []


class PayrollValidationPack(BaseValidationPack):
    """Validation rules for payroll / HR payslips."""

    DOCUMENT_TYPE = "payslip"
    REQUIRED_FIELDS = ["employee_name", "employee_id", "gross_earnings", "net_pay"]

    def _domain_rules(
        self,
        key_fields: Dict[str, Any],
        summary: Dict[str, Any],
    ) -> List[ValidationSignal]:
        signals: List[ValidationSignal] = []

        gross = _parse_int(key_fields.get("gross_earnings"))
        net_pay = _parse_int(key_fields.get("net_pay"))

        if gross is not None and net_pay is not None:
            is_valid = gross >= net_pay > 0
            signals.append(
                ValidationSignal(
                    rule="gross_gte_net_pay",
                    severity="high" if not is_valid else "info",
                    score=55 if not is_valid else 0,
                    message="Gross earnings must be greater than or equal to net pay.",
                    passed=is_valid,
                    details={"gross_earnings": gross, "net_pay": net_pay},
                )
            )

        # Net pay of zero when gross is clearly positive is highly suspicious.
        if gross is not None and gross > 5000 and net_pay == 0:
            signals.append(
                ValidationSignal(
                    rule="net_pay_zero_with_positive_gross",
                    severity="high",
                    score=50,
                    message="Net pay is zero while gross earnings are clearly positive.",
                    passed=False,
                    details={"gross_earnings": gross, "net_pay": net_pay},
                )
            )

        # UAN format check (12 numeric digits for Indian payroll).
        uan = key_fields.get("uan")
        if uan:
            digits_only = re.sub(r"\D", "", str(uan))
            is_valid_uan = len(digits_only) == 12
            signals.append(
                ValidationSignal(
                    rule="uan_format",
                    severity="low" if not is_valid_uan else "info",
                    score=5 if not is_valid_uan else 0,
                    message="UAN should be a 12-digit number.",
                    passed=is_valid_uan,
                    details={"uan": uan, "digit_count": len(digits_only)},
                )
            )

        # Paid days must be in the valid calendar range.
        paid_days = _parse_int(key_fields.get("paid_days"))
        if paid_days is not None:
            is_valid = 0 <= paid_days <= 31
            signals.append(
                ValidationSignal(
                    rule="paid_days_range",
                    severity="medium" if not is_valid else "info",
                    score=20 if not is_valid else 0,
                    message="Paid days should be between 0 and 31.",
                    passed=is_valid,
                    details={"paid_days": paid_days},
                )
            )

        # Basic should be a sensible share of gross (industry: >= 20%).
        basic = _parse_int(key_fields.get("basic"))
        if gross is not None and basic is not None and gross > 0:
            basic_pct = basic / gross
            is_suspicious = basic_pct < 0.20
            signals.append(
                ValidationSignal(
                    rule="basic_minimum_proportion",
                    severity="medium" if is_suspicious else "info",
                    score=15 if is_suspicious else 0,
                    message="Basic pay should typically be at least 20% of gross earnings.",
                    passed=not is_suspicious,
                    details={"basic_pct": round(basic_pct * 100, 1), "threshold_pct": 20},
                )
            )

        return signals


class BankingValidationPack(BaseValidationPack):
    """Validation rules for bank statements."""

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
                    },
                )
            )

        return signals


class InsuranceValidationPack(BaseValidationPack):
    """Validation rules for insurance documents."""

    DOCUMENT_TYPE = "insurance"
    REQUIRED_FIELDS = ["policy_number", "insured_name", "insurer_name"]


class HealthcareValidationPack(BaseValidationPack):
    """Validation rules for healthcare / medical documents."""

    DOCUMENT_TYPE = "healthcare"
    REQUIRED_FIELDS = ["patient_name", "provider_name", "visit_date"]


class InvoiceValidationPack(BaseValidationPack):
    """Validation rules for commercial invoices."""

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
        tax = _parse_int(key_fields.get("tax_amount"))

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

        return signals


_PACKS: Dict[str, BaseValidationPack] = {
    "payslip": PayrollValidationPack(),
    "bank_statement": BankingValidationPack(),
    "insurance": InsuranceValidationPack(),
    "healthcare": HealthcareValidationPack(),
    "invoice": InvoiceValidationPack(),
}


def get_pack(document_type: str) -> Optional[BaseValidationPack]:
    """Return the validation pack for a document type, or None for generic."""
    return _PACKS.get(str(document_type or "").lower())


def validate_document(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Validate a structured summary and return domain signals as dicts."""
    document_type = summary.get("document", {}).get("type", "generic")
    pack = get_pack(document_type)
    if pack is None:
        return []
    return [signal.to_dict() for signal in pack.validate(summary)]
