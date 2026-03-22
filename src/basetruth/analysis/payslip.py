from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional


def _parse_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    text = "".join(ch for ch in str(value) if ch.isdigit() or ch == "-")
    if not text or text == "-":
        return None
    return int(text)


def _parse_period(summary: Dict[str, Any]) -> Optional[datetime]:
    key_fields = summary.get("key_fields", {})
    pay_period = key_fields.get("pay_period")
    if pay_period:
        try:
            return datetime.strptime(str(pay_period), "%B %Y")
        except ValueError:
            pass
    pay_date = key_fields.get("pay_date")
    if pay_date:
        for fmt in ("%m/%d/%Y", "%m/%d/%y"):
            try:
                return datetime.strptime(str(pay_date), fmt)
            except ValueError:
                continue
    return None


def compare_payslip_summaries(summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    rows = []
    for summary in summaries:
        if summary.get("document", {}).get("type") != "payslip":
            continue
        rows.append({
            "summary": summary,
            "period": _parse_period(summary),
        })

    rows.sort(key=lambda item: item["period"] or datetime.max)
    anomalies: List[Dict[str, Any]] = []
    comparisons: List[Dict[str, Any]] = []

    previous = None
    for item in rows:
        summary = item["summary"]
        key_fields = summary.get("key_fields", {})
        current = {
            "period": key_fields.get("pay_period") or key_fields.get("pay_date") or summary.get("source", {}).get("name"),
            "gross_earnings": _parse_int(key_fields.get("gross_earnings")),
            "net_pay": _parse_int(key_fields.get("net_pay")),
            "total_deductions": _parse_int(key_fields.get("total_deductions")),
            "paid_days": _parse_int(key_fields.get("paid_days")),
            "lop_days": _parse_int(key_fields.get("lop_days")),
            "source_name": summary.get("source", {}).get("name", ""),
        }
        if previous is not None:
            delta = {
                "from_period": previous["period"],
                "to_period": current["period"],
                "gross_change": None if previous["gross_earnings"] is None or current["gross_earnings"] is None else current["gross_earnings"] - previous["gross_earnings"],
                "net_pay_change": None if previous["net_pay"] is None or current["net_pay"] is None else current["net_pay"] - previous["net_pay"],
                "deduction_change": None if previous["total_deductions"] is None or current["total_deductions"] is None else current["total_deductions"] - previous["total_deductions"],
                "lop_day_change": None if previous["lop_days"] is None or current["lop_days"] is None else current["lop_days"] - previous["lop_days"],
            }
            comparisons.append(delta)

            if previous["gross_earnings"] and current["gross_earnings"]:
                gross_pct = abs(current["gross_earnings"] - previous["gross_earnings"]) / max(previous["gross_earnings"], 1)
                if gross_pct >= 0.25:
                    anomalies.append(
                        {
                            "type": "gross_earnings_variance",
                            "severity": "medium",
                            "from_period": previous["period"],
                            "to_period": current["period"],
                            "details": {
                                "from_value": previous["gross_earnings"],
                                "to_value": current["gross_earnings"],
                                "percent_change": round(gross_pct * 100, 2),
                            },
                        }
                    )

            if previous["total_deductions"] is not None and current["total_deductions"] is not None:
                if current["total_deductions"] > max(previous["total_deductions"] * 1.5, previous["total_deductions"] + 5000):
                    anomalies.append(
                        {
                            "type": "deduction_spike",
                            "severity": "medium",
                            "from_period": previous["period"],
                            "to_period": current["period"],
                            "details": {
                                "from_value": previous["total_deductions"],
                                "to_value": current["total_deductions"],
                            },
                        }
                    )

            if current["lop_days"] and current["lop_days"] > 0 and (previous["lop_days"] or 0) == 0:
                anomalies.append(
                    {
                        "type": "lop_days_increase",
                        "severity": "low",
                        "from_period": previous["period"],
                        "to_period": current["period"],
                        "details": {
                            "from_value": previous["lop_days"],
                            "to_value": current["lop_days"],
                        },
                    }
                )
        previous = current

    return {
        "document_type": "payslip",
        "summary_count": len(rows),
        "comparisons": comparisons,
        "anomalies": anomalies,
    }
