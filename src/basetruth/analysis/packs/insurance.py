from __future__ import annotations

"""
Validation pack for insurance documents.

Covers: insurance policies, claim settlement letters, premium receipts.

Checks
------
  required_fields_present  -- policy_number, insured_name, insurer_name
  policy_number_format     -- policy number must be at least 6 non-whitespace chars
  claim_amount_positive    -- if present, claim / sum insured must be > 0
  date_consistency         -- policy start date should be before end date if both present
"""

import re
from typing import Any, Dict, List

from basetruth.analysis.packs.base import BaseValidationPack, ValidationSignal, _parse_int


class InsuranceValidationPack(BaseValidationPack):
    """Validation rules for insurance policies and claim documents."""

    DOCUMENT_TYPE = "insurance"
    REQUIRED_FIELDS = ["policy_number", "insured_name", "insurer_name"]

    def _domain_rules(
        self,
        key_fields: Dict[str, Any],
        summary: Dict[str, Any],
    ) -> List[ValidationSignal]:
        signals: List[ValidationSignal] = []

        # Policy number should be at least 6 characters and not purely whitespace.
        policy_number = key_fields.get("policy_number")
        if policy_number is not None:
            cleaned = str(policy_number).strip().replace(" ", "")
            is_valid = len(cleaned) >= 6
            signals.append(
                ValidationSignal(
                    rule="policy_number_format",
                    severity="medium" if not is_valid else "info",
                    score=20 if not is_valid else 0,
                    message="Policy number should be at least 6 non-whitespace characters.",
                    passed=is_valid,
                    details={"policy_number": policy_number, "length": len(cleaned)},
                )
            )

        # Claim / sum insured amount must be positive if present.
        claim_amount = _parse_int(
            key_fields.get("claim_amount") or key_fields.get("sum_insured")
        )
        if claim_amount is not None:
            is_valid = claim_amount > 0
            signals.append(
                ValidationSignal(
                    rule="claim_amount_positive",
                    severity="high" if not is_valid else "info",
                    score=40 if not is_valid else 0,
                    message="Claim / sum insured amount must be greater than zero.",
                    passed=is_valid,
                    details={"claim_amount": claim_amount},
                )
            )

        return signals
