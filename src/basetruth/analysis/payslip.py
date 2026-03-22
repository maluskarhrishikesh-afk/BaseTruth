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

    # --- Identity drift: employee ID or name changed across the series. ---
    employee_ids: List[str] = []
    employee_names: List[str] = []
    for item in rows:
        kf = item["summary"].get("key_fields", {})
        eid = str(kf.get("employee_id") or "").strip()
        ename = str(kf.get("employee_name") or "").strip()
        if eid and eid not in employee_ids:
            employee_ids.append(eid)
        if ename and ename not in employee_names:
            employee_names.append(ename)

    if len(employee_ids) > 1:
        anomalies.append(
            {
                "type": "employee_id_drift",
                "severity": "high",
                "from_period": rows[0]["summary"].get("key_fields", {}).get("pay_period", ""),
                "to_period": rows[-1]["summary"].get("key_fields", {}).get("pay_period", ""),
                "details": {"observed_ids": employee_ids},
            }
        )

    if len(employee_names) > 1:
        anomalies.append(
            {
                "type": "employee_name_drift",
                "severity": "medium",
                "from_period": rows[0]["summary"].get("key_fields", {}).get("pay_period", ""),
                "to_period": rows[-1]["summary"].get("key_fields", {}).get("pay_period", ""),
                "details": {"observed_names": employee_names},
            }
        )

    # --- Net pay drop spike: sudden large downward revision. ---
    for delta in comparisons:
        change = delta.get("net_pay_change")
        if change is None:
            continue
        # Find the prior net pay to compute percentage.
        prev_net: Optional[int] = None
        for item in rows:
            period_label = item["summary"].get("key_fields", {}).get("pay_period") or item["summary"].get("key_fields", {}).get("pay_date")
            if str(period_label) == str(delta.get("from_period")):
                prev_net = _parse_int(item["summary"].get("key_fields", {}).get("net_pay"))
                break
        if prev_net and prev_net > 0 and change < 0:
            drop_pct = abs(change) / prev_net
            if drop_pct >= 0.30:
                anomalies.append(
                    {
                        "type": "net_pay_drop_spike",
                        "severity": "high" if drop_pct >= 0.50 else "medium",
                        "from_period": delta["from_period"],
                        "to_period": delta["to_period"],
                        "details": {
                            "from_net_pay": prev_net,
                            "change": change,
                            "percent_drop": round(drop_pct * 100, 2),
                        },
                    }
                )

    # --- Period gap: months missing from the series. ---
    known_periods = []
    for item in rows:
        period = item.get("period")
        if period is not None:
            known_periods.append(period)

    if len(known_periods) >= 2:
        from datetime import timedelta

        known_periods.sort()
        for idx in range(len(known_periods) - 1):
            p1, p2 = known_periods[idx], known_periods[idx + 1]
            # Estimate expected months between p1 and p2.
            months_diff = (p2.year - p1.year) * 12 + (p2.month - p1.month)
            if months_diff > 1:
                anomalies.append(
                    {
                        "type": "period_gap",
                        "severity": "low",
                        "from_period": p1.strftime("%B %Y"),
                        "to_period": p2.strftime("%B %Y"),
                        "details": {"missing_months": months_diff - 1},
                    }
                )

    return {
        "document_type": "payslip",
        "summary_count": len(rows),
        "comparisons": comparisons,
        "anomalies": anomalies,
    }
