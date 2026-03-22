from __future__ import annotations

from basetruth.analysis.payslip import compare_payslip_summaries


def test_compare_payslip_summaries_flags_variance() -> None:
    comparison = compare_payslip_summaries(
        [
            {
                "document": {"type": "payslip"},
                "source": {"name": "jan.pdf"},
                "key_fields": {
                    "pay_period": "January 2026",
                    "gross_earnings": "100000",
                    "net_pay": "80000",
                    "total_deductions": "20000",
                    "paid_days": "31",
                    "lop_days": "0",
                },
            },
            {
                "document": {"type": "payslip"},
                "source": {"name": "feb.pdf"},
                "key_fields": {
                    "pay_period": "February 2026",
                    "gross_earnings": "140000",
                    "net_pay": "95000",
                    "total_deductions": "45000",
                    "paid_days": "28",
                    "lop_days": "2",
                },
            },
        ]
    )

    types = {anomaly["type"] for anomaly in comparison["anomalies"]}
    assert "gross_earnings_variance" in types
    assert "deduction_spike" in types
