from __future__ import annotations

"""
Validation pack for payroll and HR documents.

Covers: pay slips, salary slips, employee pay summaries.

Checks
------
  required_fields_present    -- employee_name, employee_id, gross_earnings, net_pay
  gross_gte_net_pay          -- gross earnings must be >= net pay (arithmetic check)
  net_pay_zero_with_positive_gross -- net == 0 while gross > 5000 is suspicious
  uan_format                 -- Indian UAN must be exactly 12 digits
  paid_days_range            -- paid days must be between 0 and 31
  basic_minimum_proportion   -- basic should be at least 20% of gross
"""

import re
from typing import Any, Dict, List

from basetruth.analysis.packs.base import BaseValidationPack, ValidationSignal, _parse_int


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

        # Zero net pay alongside clearly positive gross is a strong tamper signal.
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

        # UAN format: exactly 12 numeric digits for Indian payroll.
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

        # Paid days must fall within the valid calendar range.
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

        # Basic pay should be a sensible proportion of gross (industry norm: >= 20%).
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
