from __future__ import annotations

from typing import Any, Dict


def render_scan_report(report: Dict[str, Any]) -> str:
    source = report["source"]
    summary = report["structured_summary"]
    tamper = report["tamper_assessment"]
    lines = [
        "# BaseTruth Verification Report",
        "",
        f"- Source: {source['name']}",
        f"- Path: {source['path']}",
        f"- SHA256: {source['sha256']}",
        f"- Document type: {summary['document']['type']}",
        f"- Truth score: {tamper['truth_score']}",
        f"- Risk level: {tamper['risk_level']}",
        f"- Verdict: {tamper['verdict']}",
        "",
        "## Key Fields",
        "",
    ]
    for key, value in summary.get("key_fields", {}).items():
        if isinstance(value, dict):
            continue
        lines.append(f"- {key}: {value}")

    lines.extend(["", "## Signals", ""])
    for signal in tamper.get("signals", []):
        lines.append(
            f"- {signal['name']}: severity={signal['severity']}, score={signal['score']}, passed={signal.get('passed')}"
        )
        if signal.get("details"):
            lines.append(f"  details: {signal['details']}")

    lines.extend(["", "## PDF Metadata", ""])
    metadata = report.get("pdf_metadata", {})
    lines.append(f"- has_digital_signature_markers: {metadata.get('has_digital_signature_markers')}")
    lines.append(f"- signature_markers: {metadata.get('signature_markers', [])}")
    lines.append(f"- metadata: {metadata.get('metadata', {})}")
    return "\n".join(lines) + "\n"


def render_comparison_report(comparison: Dict[str, Any]) -> str:
    lines = [
        "# BaseTruth Cross-Month Payslip Comparison",
        "",
        f"- summaries analyzed: {comparison['summary_count']}",
        f"- anomalies detected: {len(comparison.get('anomalies', []))}",
        "",
        "## Anomalies",
        "",
    ]
    if not comparison.get("anomalies"):
        lines.append("- none")
    else:
        for anomaly in comparison["anomalies"]:
            lines.append(
                f"- {anomaly['type']}: {anomaly['from_period']} -> {anomaly['to_period']} | {anomaly['details']}"
            )
    return "\n".join(lines) + "\n"
