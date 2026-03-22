from __future__ import annotations

"""
Base classes and shared utilities for BaseTruth validation packs.

Every industry pack inherits from BaseValidationPack.  The base class:
  - Declares DOCUMENT_TYPE (str) and REQUIRED_FIELDS (list) as class-level
    constants so adding a new pack requires only subclassing and filling them in.
  - Runs the required-fields check automatically in validate().
  - Delegates industry-specific logic to _domain_rules(), which subclasses
    override.  If they don't, validate() just returns the required-fields signal.

Open/Closed Principle
---------------------
To add a new industry:
  1. Create a new file in analysis/packs/ (e.g. healthcare.py).
  2. Subclass BaseValidationPack, declare DOCUMENT_TYPE and REQUIRED_FIELDS.
  3. Override _domain_rules() with industry checks.
  4. Register the instance in analysis/packs/__init__.py REGISTRY dict.
  No changes needed in any other file.
"""

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ValidationSignal:
    """A single rule result emitted by a validation pack.

    Fields
    ------
    rule     : machine-readable identifier, e.g. 'gross_gte_net_pay'
    severity : 'info' | 'low' | 'medium' | 'high'
    score    : tamper-risk contribution (0-100); 0 means the check passed
    message  : human-readable description of the rule
    passed   : True when the check passed (no anomaly found)
    details  : arbitrary key/value evidence dict for the report
    """

    rule: str
    severity: str
    score: int
    message: str
    passed: bool
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _parse_int(value: Any) -> Optional[int]:
    """Extract an integer from a value that may be a string with currency symbols."""
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
    """Abstract base class for a domain-specific validation pack.

    Subclasses *must* declare:
      DOCUMENT_TYPE  (str)   -- the document type string that routes to this pack
      REQUIRED_FIELDS (list) -- field names that must be present in key_fields
    """

    DOCUMENT_TYPE: str = "generic"
    REQUIRED_FIELDS: List[str] = []

    def validate(self, summary: Dict[str, Any]) -> List[ValidationSignal]:
        """Run required-field checks and domain rules; return all signals."""
        key_fields = summary.get("key_fields", {})
        signals: List[ValidationSignal] = []

        missing = [f for f in self.REQUIRED_FIELDS if not key_fields.get(f)]
        signals.append(
            ValidationSignal(
                rule="required_fields_present",
                severity="medium" if missing else "info",
                score=min(30, 8 * len(missing)),
                message=f"Required fields check for {self.DOCUMENT_TYPE}.",
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
        """Override in subclasses to add industry-specific checks."""
        return []
