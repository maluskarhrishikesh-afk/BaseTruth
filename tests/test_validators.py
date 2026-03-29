from __future__ import annotations

import pytest

from basetruth.analysis.validators import (
    get_pack,
    validate_document,
    PayrollValidationPack,
    BankingValidationPack,
    InvoiceValidationPack,
)


# ---------------------------------------------------------------------------
# PayrollValidationPack
# ---------------------------------------------------------------------------


def _payslip_summary(overrides: dict | None = None) -> dict:
    fields = {
        "employee_name": "John Doe",
        "employee_id": "E001",
        "gross_earnings": "50000",
        "net_pay": "42000",
        "basic": "25000",
        "paid_days": "22",
        "uan": "100123456789",
    }
    if overrides:
        fields.update(overrides)
    return {"document": {"type": "payslip"}, "key_fields": fields}


def test_payroll_pack_passes_valid_document() -> None:
    signals = validate_document(_payslip_summary())
    # All domain rules should pass (score == 0) for a clean payslip
    failing = [s for s in signals if not s["passed"]]
    assert failing == [], f"Unexpected failures: {failing}"


def test_payroll_pack_flags_net_exceeds_gross() -> None:
    signals = validate_document(_payslip_summary({"net_pay": "60000"}))
    rule_signals = {s["rule"]: s for s in signals}
    sig = rule_signals.get("gross_gte_net_pay")
    assert sig is not None, "Expected gross_gte_net_pay signal"
    assert not sig["passed"]
    assert sig["score"] > 0


def test_payroll_pack_flags_invalid_uan_format() -> None:
    signals = validate_document(_payslip_summary({"uan": "12345"}))  # < 12 digits
    rule_signals = {s["rule"]: s for s in signals}
    sig = rule_signals.get("uan_format")
    assert sig is not None
    assert not sig["passed"]


def test_payroll_pack_flags_paid_days_out_of_range() -> None:
    signals = validate_document(_payslip_summary({"paid_days": "45"}))
    rule_signals = {s["rule"]: s for s in signals}
    sig = rule_signals.get("paid_days_range")
    assert sig is not None
    assert not sig["passed"]


def test_payroll_pack_flags_low_basic_proportion() -> None:
    # Basic < 20% of gross
    signals = validate_document(_payslip_summary({"gross_earnings": "100000", "net_pay": "80000", "basic": "5000"}))
    rule_signals = {s["rule"]: s for s in signals}
    sig = rule_signals.get("basic_minimum_proportion")
    assert sig is not None
    assert not sig["passed"]


# ---------------------------------------------------------------------------
# BankingValidationPack
# ---------------------------------------------------------------------------


def _bank_summary(overrides: dict | None = None) -> dict:
    fields = {
        "account_number": "1234567890",
        "statement_period": "Jan 2025",
        "opening_balance": "10000",
        "total_credits": "5000",
        "total_debits": "3000",
        "closing_balance": "12000",  # correct: 10000 + 5000 - 3000 = 12000
    }
    if overrides:
        fields.update(overrides)
    return {"document": {"type": "bank_statement"}, "key_fields": fields}


def test_bank_pack_passes_correct_arithmetic() -> None:
    signals = validate_document(_bank_summary())
    failing = [s for s in signals if not s["passed"] and s["rule"] == "balance_arithmetic"]
    assert failing == []


def test_bank_pack_flags_balance_mismatch() -> None:
    signals = validate_document(_bank_summary({"closing_balance": "15000"}))  # wrong
    rule_signals = {s["rule"]: s for s in signals}
    sig = rule_signals.get("balance_arithmetic")
    assert sig is not None
    assert not sig["passed"]
    assert sig["score"] > 0


# ---------------------------------------------------------------------------
# InvoiceValidationPack
# ---------------------------------------------------------------------------


def _invoice_summary(overrides: dict | None = None) -> dict:
    fields = {
        "invoice_number": "INV-001",
        "amount_due": "11800",
        "subtotal": "10000",
        "tax_amount": "1800",
    }
    if overrides:
        fields.update(overrides)
    return {"document": {"type": "invoice"}, "key_fields": fields}


def test_invoice_pack_passes_correct_arithmetic() -> None:
    signals = validate_document(_invoice_summary())
    failing = [s for s in signals if not s["passed"] and s["rule"] == "invoice_amount_arithmetic"]
    assert failing == []


def test_invoice_pack_flags_amount_mismatch() -> None:
    signals = validate_document(_invoice_summary({"amount_due": "99999"}))
    rule_signals = {s["rule"]: s for s in signals}
    sig = rule_signals.get("invoice_amount_arithmetic")
    assert sig is not None
    assert not sig["passed"]


# ---------------------------------------------------------------------------
# get_pack and unknown types
# ---------------------------------------------------------------------------


def test_get_pack_returns_none_for_unknown_type() -> None:
    assert get_pack("alien_document") is None


def test_validate_document_returns_empty_for_generic() -> None:
    summary = {"document": {"type": "generic"}, "key_fields": {}}
    assert validate_document(summary) == []


def test_validate_document_returns_empty_when_no_type() -> None:
    summary = {"document": {}, "key_fields": {}}
    assert validate_document(summary) == []
