from __future__ import annotations

"""
Industry validation packs for BaseTruth.

This package uses a simple registry pattern (a plain dict) so new packs can be
added without touching any existing code (Open/Closed Principle).

How to add a new industry pack
-------------------------------
1. Create a new .py file in this directory, e.g. ``real_estate.py``.
2. Subclass BaseValidationPack, declare DOCUMENT_TYPE and REQUIRED_FIELDS, and
   override _domain_rules().
3. Import the class here and add one line to REGISTRY.
Done -- the tamper scorer, CLI, REST API, and UI all pick it up automatically.

Public API
----------
  validate_document(summary)     -- run the correct pack and return signal dicts
  get_pack(document_type)        -- return the BaseValidationPack instance or None
  REGISTRY                       -- dict[str, BaseValidationPack]: full pack map
"""

from basetruth.analysis.packs.banking import BankingValidationPack
from basetruth.analysis.packs.base import BaseValidationPack, ValidationSignal, _parse_int
from basetruth.analysis.packs.compliance import ComplianceValidationPack
from basetruth.analysis.packs.healthcare import HealthcareValidationPack
from basetruth.analysis.packs.insurance import InsuranceValidationPack
from basetruth.analysis.packs.invoice import InvoiceValidationPack
from basetruth.analysis.packs.payments import PaymentsValidationPack
from basetruth.analysis.packs.payroll import PayrollValidationPack
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Central registry -- map document_type string -> pack instance
# ---------------------------------------------------------------------------
REGISTRY: Dict[str, BaseValidationPack] = {
    PayrollValidationPack.DOCUMENT_TYPE: PayrollValidationPack(),
    BankingValidationPack.DOCUMENT_TYPE: BankingValidationPack(),
    PaymentsValidationPack.DOCUMENT_TYPE: PaymentsValidationPack(),
    InsuranceValidationPack.DOCUMENT_TYPE: InsuranceValidationPack(),
    HealthcareValidationPack.DOCUMENT_TYPE: HealthcareValidationPack(),
    InvoiceValidationPack.DOCUMENT_TYPE: InvoiceValidationPack(),
    ComplianceValidationPack.DOCUMENT_TYPE: ComplianceValidationPack(),
}


def get_pack(document_type: str) -> Optional[BaseValidationPack]:
    """Return the validation pack for a document_type string, or None if not registered."""
    return REGISTRY.get(str(document_type or "").lower())


def validate_document(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Validate a structured summary and return all domain signals as plain dicts.

    Looks up the document type from summary["document"]["type"], runs the
    matching pack (or returns an empty list for unknown/generic documents),
    and serialises each ValidationSignal to a dict for JSON portability.
    """
    document_type = summary.get("document", {}).get("type", "generic")
    pack = get_pack(document_type)
    if pack is None:
        return []
    return [signal.to_dict() for signal in pack.validate(summary)]


__all__ = [
    "REGISTRY",
    "BaseValidationPack",
    "ValidationSignal",
    "_parse_int",
    "PayrollValidationPack",
    "BankingValidationPack",
    "PaymentsValidationPack",
    "InsuranceValidationPack",
    "HealthcareValidationPack",
    "InvoiceValidationPack",
    "ComplianceValidationPack",
    "get_pack",
    "validate_document",
]
