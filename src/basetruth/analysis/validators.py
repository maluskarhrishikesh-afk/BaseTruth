"""Domain-specific validation packs for BaseTruth.

This module is now a backwards-compatible shim.  All logic has moved to
the industry-specific files in analysis/packs/:

  analysis/packs/base.py        -- BaseValidationPack, ValidationSignal
  analysis/packs/payroll.py     -- PayrollValidationPack (payslip / HR)
  analysis/packs/banking.py     -- BankingValidationPack (bank statements)
  analysis/packs/payments.py    -- PaymentsValidationPack (UPI / fintech)
  analysis/packs/insurance.py   -- InsuranceValidationPack
  analysis/packs/healthcare.py  -- HealthcareValidationPack (hospital bills)
  analysis/packs/invoice.py     -- InvoiceValidationPack (GST invoices)
  analysis/packs/compliance.py  -- ComplianceValidationPack (audit / KYC)
  analysis/packs/__init__.py    -- REGISTRY, get_pack(), validate_document()

Extending BaseTruth with a new industry
----------------------------------------
Create a pack file in analysis/packs/, subclass BaseValidationPack,
and add one line to REGISTRY in analysis/packs/__init__.py.
No changes needed here.

Supported document types (as reported by structured_summary["document"]["type"])
---------------------------------------------------------------------------
  payslip          -- Indian salary/payslip documents (payroll.py)
  bank_statement   -- Bank account statements (banking.py)
  payment_receipt  -- UPI / NEFT / payment receipts (payments.py)
  insurance        -- Insurance policies and claim letters (insurance.py)
  healthcare       -- Hospital bills and medical invoices (healthcare.py)
  invoice          -- GST and commercial invoices (invoice.py)
  compliance       -- Audit reports and compliance certificates (compliance.py)
  generic          -- Catch-all fallback; no domain checks run.

Signal schema
-------------
Each signal dict has:
  rule      (str)   -- machine-readable check identifier
  severity  (str)   -- 'info' | 'low' | 'medium' | 'high'
  score     (int)   -- tamper-risk contribution (0 when passed)
  message   (str)   -- human-readable description
  passed    (bool)  -- True if no anomaly was found
  details   (dict)  -- supporting evidence

Public API (unchanged for existing callers)
-------------------------------------------
  validate_document(structured_summary) -> List[Dict]
  get_pack(document_type) -> Optional[BaseValidationPack]
"""
from __future__ import annotations

# Re-export everything from the packs package so existing imports keep working.
from basetruth.analysis.packs import (  # noqa: F401
    REGISTRY,
    BaseValidationPack,
    BankingValidationPack,
    ComplianceValidationPack,
    HealthcareValidationPack,
    InsuranceValidationPack,
    InvoiceValidationPack,
    PaymentsValidationPack,
    PayrollValidationPack,
    ValidationSignal,
    _parse_int,
    get_pack,
    validate_document,
)

# Legacy name alias kept for any code that still refers to the old class name.
_PACKS = REGISTRY
