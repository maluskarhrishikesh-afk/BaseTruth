from __future__ import annotations

from typing import Any, Dict


_RISK_ICONS = {"high": "🚨", "medium": "⚠️", "low": "✅", "review": "🔷"}
_SIGNAL_ICONS = {True: "✅", False: "🚨"}


def _signal_rows(signals: list) -> str:
    rows = []
    for sig in signals:
        passed = sig.get("passed")
        icon = _SIGNAL_ICONS.get(passed, "ℹ️")
        name = str(sig.get("name", "")).replace("::", " › ").replace("_", " ").title()
        score_part = f" · score {sig.get('score', 0)}" if sig.get("score", 0) else ""
        rows.append(f"| {icon} | {name} | {sig.get('severity', '')} | {sig.get('score', 0)}{score_part.replace(score_part, '')} | {sig.get('summary', '')} |")
    return "\n".join(rows)


def render_scan_report(report: Dict[str, Any]) -> str:
    source = report.get("source", {})
    summary = report.get("structured_summary", {})
    tamper = report.get("tamper_assessment", {})
    key_fields = summary.get("key_fields", {})
    risk_level = str(tamper.get("risk_level", "low"))
    risk_icon = _RISK_ICONS.get(risk_level, "ℹ️")
    score = tamper.get("truth_score", "—")
    doc = summary.get("document", {})

    lines = [
        "# BaseTruth Verification Report",
        "",
        f"> {risk_icon} **Risk Level: {risk_level.upper()}** · Truth Score: **{score} / 100**",
        "",
        "## Document",
        "",
        f"| Field | Value |",
        f"|---|---|",
        f"| Source | {source.get('name', '')} |",
        f"| Path | `{source.get('path', '')}` |",
        f"| SHA-256 | `{source.get('sha256', '')}` |",
        f"| Document type | {doc.get('type', '').replace('_', ' ').title()} (confidence {doc.get('type_confidence', '')}) |",
        f"| Verdict | {tamper.get('verdict', '')} |",
        "",
        "## Key Fields",
        "",
        "| Field | Value |",
        "|---|---|",
    ]
    for key, value in key_fields.items():
        if isinstance(value, (dict, list)):
            continue
        if value is not None and str(value).strip():
            lines.append(f"| {key.replace('_', ' ').title()} | {value} |")

    signals = tamper.get("signals", [])
    lines += [
        "",
        "## Forensic Signals",
        "",
        f"| | Signal | Severity | Score | Summary |",
        "|---|---|---|---|---|",
    ]
    for sig in signals:
        passed = sig.get("passed")
        icon = _SIGNAL_ICONS.get(passed, "ℹ️")
        name = str(sig.get("name", "")).replace("::", " › ").replace("_", " ").title()
        lines.append(
            f"| {icon} | {name} | {sig.get('severity', '')} | {sig.get('score', 0)} | {sig.get('summary', '')} |"
        )

    metadata = report.get("pdf_metadata", {})
    lines += [
        "",
        "## PDF Metadata",
        "",
        f"| Field | Value |",
        "|---|---|",
        f"| Has digital signature markers | {metadata.get('has_digital_signature_markers', False)} |",
        f"| Signature markers | {', '.join(metadata.get('signature_markers', []))} |",
        f"| PDF header | `{metadata.get('pdf_header', '')}` |",
    ]
    meta_detail = metadata.get("metadata", {})
    for k, v in (meta_detail.items() if isinstance(meta_detail, dict) else {}.items()):
        lines.append(f"| {k} | {v} |")

    lines += [
        "",
        "## Limitations",
        "",
    ]
    for limitation in tamper.get("limitations", []):
        lines.append(f"- {limitation}")

    lines.append("")
    return "\n".join(lines)


def render_comparison_report(comparison: Dict[str, Any]) -> str:
    anomalies = comparison.get("anomalies", [])
    comparisons = comparison.get("comparisons", [])
    lines = [
        "# BaseTruth Payslip Comparison Report",
        "",
        f"- Summaries analysed: **{comparison.get('summary_count', 0)}**",
        f"- Anomalies detected: **{len(anomalies)}**",
        "",
    ]

    if comparisons:
        lines += [
            "## Month-on-month summary",
            "",
            "| From | To | Gross Δ | Net Pay Δ | Deduction Δ |",
            "|---|---|---|---|---|",
        ]
        for c in comparisons:
            lines.append(
                f"| {c.get('from_period', '')} | {c.get('to_period', '')} "
                f"| {c.get('gross_change', '')} | {c.get('net_pay_change', '')} "
                f"| {c.get('deduction_change', '')} |"
            )
        lines.append("")

    if not anomalies:
        lines += ["## Anomalies", "", "No anomalies detected.", ""]
    else:
        lines += [
            "## Anomalies",
            "",
            "| Severity | Type | From | To | Detail |",
            "|---|---|---|---|---|",
        ]
        for anomaly in anomalies:
            sev = str(anomaly.get("severity", "low"))
            icon = {"high": "🚨", "medium": "⚠️", "low": "🔷"}.get(sev, "ℹ️")
            lines.append(
                f"| {icon} {sev.upper()} | {anomaly.get('type', '').replace('_', ' ').title()} "
                f"| {anomaly.get('from_period', '')} | {anomaly.get('to_period', '')} "
                f"| {anomaly.get('details', {})} |"
            )
        lines.append("")

    return "\n".join(lines)
