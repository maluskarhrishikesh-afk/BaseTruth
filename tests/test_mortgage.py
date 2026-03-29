from __future__ import annotations

"""
Tests for the MortgageValidationPack and the mortgage-specific extensions
to BankingValidationPack (circular funds, duplicate refs, salary regularity).
"""

import pytest

from basetruth.analysis.packs.mortgage import MortgageValidationPack
from basetruth.analysis.packs.banking import BankingValidationPack
from basetruth.analysis.packs import get_pack, REGISTRY


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

def test_mortgage_pack_registered() -> None:
    assert get_pack("mortgage") is not None
    assert isinstance(get_pack("mortgage"), MortgageValidationPack)


def test_mortgage_subtypes_registered() -> None:
    for sub in (
        "mortgage_payslip",
        "mortgage_bank_statement",
        "mortgage_employment_letter",
        "mortgage_form16",
        "mortgage_utility_bill",
        "employment_letter",
        "form16",
        "utility_bill",
    ):
        pack = get_pack(sub)
        assert pack is not None, f"Sub-type '{sub}' not registered"
        assert isinstance(pack, MortgageValidationPack)


# ---------------------------------------------------------------------------
# MortgageValidationPack — payslip checks
# ---------------------------------------------------------------------------

def _payslip_summary(overrides: dict | None = None) -> dict:
    base = {
        "document": {"type": "mortgage_payslip"},
        "key_fields": {
            "gross_earnings": "80000",
            "net_pay": "65000",
            "basic_salary": "32000",
            "hra": "12800",
            "pf": "3840",       # 12% of 32000 = 3840 ✅
            "professional_tax": "200",
            "tds": "4000",
        },
        "transactions": [],
    }
    if overrides:
        base["key_fields"].update(overrides)
    return base


def test_payslip_clean_passes() -> None:
    pack = MortgageValidationPack()
    signals = pack.validate(_payslip_summary())
    failed = [s for s in signals if not s.passed]
    assert len(failed) == 0, f"Unexpected failures: {[s.rule for s in failed]}"


def test_payslip_pf_over_limit_flagged() -> None:
    """PF > 12.5% of basic should raise pf_rate_validity."""
    pack = MortgageValidationPack()
    signals = pack.validate(_payslip_summary({"pf": "5000"}))  # 5000/32000 = 15.6%
    rules = {s.rule for s in signals if not s.passed}
    assert "pf_rate_validity" in rules


def test_payslip_pt_over_limit_flagged() -> None:
    """PT > ₹200 should raise pt_slab_validity."""
    pack = MortgageValidationPack()
    signals = pack.validate(_payslip_summary({"professional_tax": "500"}))
    rules = {s.rule for s in signals if not s.passed}
    assert "pt_slab_validity" in rules


def test_payslip_zero_tds_high_income_flagged() -> None:
    """TDS = 0 with gross that annualises above ₹7 lakh should raise tds_plausibility."""
    pack = MortgageValidationPack()
    # gross = ₹70,000/month → annual = ₹8,40,000 > ₹7,00,000 → TDS must be non-zero
    signals = pack.validate(_payslip_summary({"gross_earnings": "70000", "tds": "0"}))
    rules = {s.rule for s in signals if not s.passed}
    assert "tds_plausibility" in rules


def test_payslip_hra_overclaim_flagged() -> None:
    """HRA > basic should raise hra_proportion."""
    pack = MortgageValidationPack()
    signals = pack.validate(_payslip_summary({"hra": "40000"}))  # 40000 > basic 32000
    rules = {s.rule for s in signals if not s.passed}
    assert "hra_proportion" in rules


# ---------------------------------------------------------------------------
# MortgageValidationPack — employment letter checks
# ---------------------------------------------------------------------------

def _employment_summary(overrides: dict | None = None) -> dict:
    base = {
        "document": {"type": "employment_letter"},
        "key_fields": {
            "cin": "L85110KA1981PLC013341",  # Infosys CIN
            "join_date": "15-06-2022",
            "annual_ctc": "1200000",
            "monthly_gross": "100000",
        },
        "transactions": [],
    }
    if overrides:
        base["key_fields"].update(overrides)
    return base


def test_employment_clean_passes() -> None:
    pack = MortgageValidationPack()
    signals = pack.validate(_employment_summary())
    failed = [s for s in signals if not s.passed]
    assert len(failed) == 0, f"Unexpected failures: {[s.rule for s in failed]}"


def test_employment_invalid_cin_flagged() -> None:
    """A malformed CIN must raise employer_cin_format."""
    pack = MortgageValidationPack()
    signals = pack.validate(_employment_summary({"cin": "INVALID123"}))
    rules = {s.rule for s in signals if not s.passed}
    assert "employer_cin_format" in rules


def test_employment_missing_cin_flagged() -> None:
    """Absent CIN must raise employer_cin_present."""
    pack = MortgageValidationPack()
    signals = pack.validate(_employment_summary({"cin": ""}))
    rules = {s.rule for s in signals if not s.passed}
    assert "employer_cin_present" in rules


def test_employment_cin_age_before_join_flagged() -> None:
    """CIN year 2023 and join date 2022 should raise cin_age_vs_join_date."""
    pack = MortgageValidationPack()
    # CIN with 2023 incorporation year
    signals = pack.validate(_employment_summary({
        "cin": "L85110KA2023PLC013341",
        "join_date": "15-06-2022",
    }))
    rules = {s.rule for s in signals if not s.passed}
    assert "cin_age_vs_join_date" in rules


# ---------------------------------------------------------------------------
# BankingValidationPack — circular funds detection
# ---------------------------------------------------------------------------

def _bank_summary_with_transactions(transactions: list) -> dict:
    return {
        "document": {"type": "bank_statement"},
        "key_fields": {
            "account_number": "XXXX1234",
            "statement_period": "Oct 2025 - Mar 2026",
            "opening_balance": "100000",
            "closing_balance": "120000",
            "total_credits": "720000",
            "total_debits": "700000",
        },
        "transactions": transactions,
    }


def _clean_transactions() -> list:
    return [
        {"date": "01-01-2026", "desc": "SALARY CREDIT", "ref": "NEFT001", "debit": 0, "credit": 85000},
        {"date": "05-01-2026", "desc": "NACH/EMI", "ref": "NACH001", "debit": 20000, "credit": 0},
        {"date": "10-01-2026", "desc": "UPI GROCERY", "ref": "UPI001", "debit": 5000, "credit": 0},
    ]


def test_circular_funds_clean_passes() -> None:
    pack = BankingValidationPack()
    signals = pack.validate(_bank_summary_with_transactions(_clean_transactions()))
    rules_failed = {s.rule for s in signals if not s.passed}
    assert "circular_funds_detection" not in rules_failed


def test_circular_funds_detected() -> None:
    """Round-trip debit+credit on same date should raise circular_funds_detection."""
    pack = BankingValidationPack()
    txns = _clean_transactions() + [
        {"date": "15-01-2026", "desc": "NEFT/SELF OUT", "ref": "ST001", "debit": 500000, "credit": 0},
        {"date": "15-01-2026", "desc": "NEFT/SELF IN",  "ref": "ST002", "debit": 0, "credit": 500000},
    ]
    signals = pack.validate(_bank_summary_with_transactions(txns))
    rules_failed = {s.rule for s in signals if not s.passed}
    assert "circular_funds_detection" in rules_failed


def test_duplicate_txn_reference_detected() -> None:
    """Reused transaction reference number must raise duplicate_txn_reference."""
    pack = BankingValidationPack()
    txns = [
        {"date": "01-01-2026", "desc": "SALARY",    "ref": "NEFT001", "debit": 0,     "credit": 85000},
        {"date": "05-01-2026", "desc": "EMI DEBIT", "ref": "NEFT001", "debit": 20000, "credit": 0},  # same ref
    ]
    signals = pack.validate(_bank_summary_with_transactions(txns))
    rules_failed = {s.rule for s in signals if not s.passed}
    assert "duplicate_txn_reference" in rules_failed


def test_salary_credited_twice_in_month_flagged() -> None:
    """Two salary credits in the same calendar month must raise salary_credit_regularity."""
    pack = BankingValidationPack()
    txns = [
        {"date": "01-01-2026", "desc": "SALARY CREDIT payroll", "ref": "NEFT001", "debit": 0, "credit": 85000},
        {"date": "15-01-2026", "desc": "SALARY CREDIT payroll", "ref": "NEFT002", "debit": 0, "credit": 85000},
    ]
    signals = pack.validate(_bank_summary_with_transactions(txns))
    rules_failed = {s.rule for s in signals if not s.passed}
    assert "salary_credit_regularity" in rules_failed


# ---------------------------------------------------------------------------
# MortgageValidationPack — Form 16 checks
# ---------------------------------------------------------------------------

def test_form16_zero_tds_high_income_flagged() -> None:
    pack = MortgageValidationPack()
    summary = {
        "document": {"type": "form16"},
        "key_fields": {
            "tan": "PUNE12345E",
            "gross_salary": "900000",  # ₹9 lakh > ₹7 lakh threshold
            "tds": "0",
        },
        "transactions": [],
    }
    signals = pack.validate(summary)
    rules_failed = {s.rule for s in signals if not s.passed}
    assert "form16_tds_plausibility" in rules_failed


def test_form16_invalid_tan_flagged() -> None:
    pack = MortgageValidationPack()
    summary = {
        "document": {"type": "form16"},
        "key_fields": {
            "tan": "BADTAN",
            "gross_salary": "600000",
            "tds": "10000",
        },
        "transactions": [],
    }
    signals = pack.validate(summary)
    rules_failed = {s.rule for s in signals if not s.passed}
    assert "tan_format_validity" in rules_failed


# ---------------------------------------------------------------------------
# Structured.py — document type detection for new types
# ---------------------------------------------------------------------------

def test_structured_detects_employment_letter() -> None:
    from basetruth.analysis.structured import _detect_document_type
    result = _detect_document_type(
        "employment_letter.pdf",
        "To Whomsoever It May Concern This is to certify CIN L85110KA1981PLC013341 "
        "Date of Joining 15 June 2022 HR Department",
    )
    assert result["type"] == "employment_letter"


def test_structured_detects_form16() -> None:
    from basetruth.analysis.structured import _detect_document_type
    result = _detect_document_type(
        "form16.pdf",
        "FORM 16 Certificate of Tax Deducted at Source Section 203 Assessment Year 2025-26 TAN of Employer",
    )
    assert result["type"] == "form16"


def test_structured_detects_utility_bill() -> None:
    from basetruth.analysis.structured import _detect_document_type
    result = _detect_document_type(
        "utility_bill.pdf",
        "Electricity Bill Consumer Number 1234567890 Units Consumed 350 kWh Electricity Duty",
    )
    assert result["type"] == "utility_bill"


def test_structured_detects_property_agreement() -> None:
    from basetruth.analysis.structured import _detect_document_type
    result = _detect_document_type(
        "property_agreement.pdf",
        "Agreement for Sale of Immovable Property Vendor Purchaser Sale Consideration RERA",
    )
    assert result["type"] == "property_agreement"
