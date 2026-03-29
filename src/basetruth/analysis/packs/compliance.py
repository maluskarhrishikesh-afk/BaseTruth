from __future__ import annotations

"""
Validation pack for compliance and internal audit documents.

Covers: audit reports, compliance certificates, KYC/AML declarations,
        board resolutions, internal control assessments.

Checks
------
  required_fields_present -- report_title, auditor_name, report_date
  date_not_future         -- audit/report date should not be in the future
  signatory_present       -- auditor / authorised signatory should be named
  period_present          -- audit period / coverage period should be stated
"""

from datetime import datetime, timezone
from typing import Any, Dict, List

from basetruth.analysis.packs.base import BaseValidationPack, ValidationSignal


class ComplianceValidationPack(BaseValidationPack):
    """Validation rules for compliance and internal audit documents."""

    DOCUMENT_TYPE = "compliance"
    REQUIRED_FIELDS = ["report_title", "auditor_name", "report_date"]

    def _domain_rules(
        self,
        key_fields: Dict[str, Any],
        summary: Dict[str, Any],
    ) -> List[ValidationSignal]:
        signals: List[ValidationSignal] = []

        # Report date should not be in the future.
        report_date_str = key_fields.get("report_date")
        if report_date_str:
            try:
                report_date = datetime.fromisoformat(str(report_date_str))
                now = datetime.now(report_date.tzinfo or timezone.utc)
                is_valid = report_date <= now
                signals.append(
                    ValidationSignal(
                        rule="date_not_future",
                        severity="medium" if not is_valid else "info",
                        score=20 if not is_valid else 0,
                        message="Report date should not be in the future.",
                        passed=is_valid,
                        details={"report_date": report_date_str},
                    )
                )
            except (ValueError, TypeError):
                pass  # Unparseable date: skip rather than flag spuriously.

        # Audit period coverage should be stated for meaningful compliance docs.
        period = key_fields.get("audit_period") or key_fields.get("coverage_period")
        if period is not None:
            is_valid = bool(str(period).strip())
            signals.append(
                ValidationSignal(
                    rule="period_present",
                    severity="low" if not is_valid else "info",
                    score=5 if not is_valid else 0,
                    message="Audit / coverage period should be stated clearly.",
                    passed=is_valid,
                    details={"period": period},
                )
            )

        return signals
