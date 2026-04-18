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


# ---------------------------------------------------------------------------
# Entity final-verification report renderer
# ---------------------------------------------------------------------------

# Status icons for cross-document consistency checks.
# PASS → green tick, MISMATCH/TAMPERED/FAIL → red cross, anything else → neutral.
_CHECK_ICONS = {
    "PASS":     "✅",
    "MISMATCH": "❌",
    "TAMPERED": "❌",
    "FAIL":     "❌",
    "CLEAR":    "✅",
    "SKIP":     "➖",
}

# Forensic verdict → traffic-light icon for the document inventory table.
_FORENSIC_ICONS = {
    "ORIGINAL":        "🟢",
    "UNCERTAIN":       "🟡",
    "LIKELY TAMPERED": "🟠",
    "TAMPERED":        "🔴",
}


def render_entity_report_markdown(
    report_json: Dict[str, Any],
    report_ref: str = "",
) -> str:
    """Render the full final verification report as a Markdown string.

    Layout:
      - Cover header (refs, name, verdict, date).
      - The Gemma4-written 10-section narrative (sections 1-10) as the main body.
      - Appendix A: Cross-document consistency check details.
      - Appendix B: Document inventory (forensic verdict + score per doc).
      - Appendix C: Per-document evidence (extracted identity fields).
      - Appendix D: Full document extraction data (raw JSON, for audit trail).
      - Appendix E: Full forensic analysis data (raw JSON, for audit trail).

    The AI narrative is the centrepiece — the appendix sections provide the raw
    evidence for reviewers who want to verify the AI's findings.
    """
    from datetime import datetime, timezone
    import json as _json

    entity_ref     = report_json.get("entity_ref", "—")
    entity_name    = report_json.get("entity_name", "—")
    overall        = report_json.get("overall_verdict", "—")
    docs_analysed  = report_json.get("documents_analysed", 0)
    scans_reviewed = report_json.get("scans_reviewed", 0)
    checks         = report_json.get("checks", {})
    evidence       = report_json.get("per_document_evidence", [])
    generated_at   = datetime.now(timezone.utc).strftime("%d %b %Y %H:%M UTC")

    entity_summary      = report_json.get("entity_summary") or {}
    gemma_narrative     = report_json.get("gemma_narrative") or ""
    gemma_source        = report_json.get("gemma_narrative_source") or "—"
    doc_extractions_raw = report_json.get("document_extractions_raw") or []
    scan_analysis_raw   = report_json.get("scan_layered_analysis_raw") or []

    ov_icon = "✅" if overall == "PASS" else "❌" if overall == "FAIL" else "❓"

    # ── Cover header ─────────────────────────────────────────────────────────
    lines: list[str] = [
        "# BaseTruth — Final Verification Report",
        "",
        f"| Field | Value |",
        f"|---|---|",
        f"| Report Ref | {report_ref or '—'} |",
        f"| Entity Ref | {entity_ref} |",
        f"| Subject Name | {entity_summary.get('full_name') or entity_name or '—'} |",
        f"| Date of Issue | {generated_at} |",
        f"| Verification Type | Comprehensive Identity, Education, Employment & Document Forensics Review |",
        f"| Final Outcome | {ov_icon} **{overall}** |",
        "",
        "---",
        "",
    ]

    # ── Main body: Gemma4 narrative ──────────────────────────────────────────
    # The narrative already contains sections 1-10 if Gemma4 followed the prompt.
    # We present it as-is, with a byline.
    if gemma_narrative:
        lines += [
            f"*Report generated by: {gemma_source}*",
            "",
            gemma_narrative,
            "",
        ]
    else:
        lines += [
            "*AI narrative not available — please review the Appendix sections below.*",
            "",
        ]

    # ── Appendix A: Cross-document consistency ────────────────────────────────
    lines += [
        "---",
        "",
        "## Appendix A — Cross-Document Consistency Checks",
        "",
        "| Check | Status | Finding |",
        "|---|---|---|",
    ]
    for check_name, chk in checks.items():
        status = (chk or {}).get("status", "—")
        detail = (chk or {}).get("detail", "—")
        icon   = _CHECK_ICONS.get(status, "❓")
        lines.append(f"| **{check_name.capitalize()}** | {icon} {status} | {detail} |")

    # ── Appendix B: Document inventory ───────────────────────────────────────
    lines += [
        "",
        "---",
        "",
        "## Appendix B — Document Inventory",
        "",
        "| # | File Name | Document Type | Forensic Verdict | Forgery Score |",
        "|---|---|---|---|---|",
    ]
    for idx, ev in enumerate(evidence, start=1):
        fname     = ev.get("file_name", "—")
        doc_type  = (ev.get("document_type") or "unknown").replace("_", " ").title()
        fv        = ev.get("forensic_verdict", "")
        score     = ev.get("forgery_score", "")
        fv_icon   = _FORENSIC_ICONS.get(fv, "—")
        fv_label  = f"{fv_icon} {fv}" if fv else "—"
        score_str = f"{score:.1f}" if isinstance(score, (int, float)) else (score or "—")
        lines.append(f"| {idx} | {fname} | {doc_type} | {fv_label} | {score_str} |")

    # ── Appendix C: Per-document evidence ────────────────────────────────────
    lines += [
        "",
        "---",
        "",
        "## Appendix C — Per-Document Evidence",
        "",
        "Identity fields extracted from each document and used in the consistency checks.",
        "",
    ]
    for idx, ev in enumerate(evidence, start=1):
        fname    = ev.get("file_name", "—")
        doc_type = (ev.get("document_type") or "unknown").replace("_", " ").title()
        fv       = ev.get("forensic_verdict") or ""
        score    = ev.get("forgery_score")
        fv_label = f"  |  Forensic: {_FORENSIC_ICONS.get(fv, '')} {fv} (score: {score:.1f})" if fv and isinstance(score, (int, float)) else (f"  |  Forensic: {fv}" if fv else "")
        lines += [f"### {idx}. {fname}", f"*{doc_type}{fv_label}*", ""]

        field_map = [
            ("Name",           ev.get("name")),
            ("PAN",            ev.get("pan")),
            ("Aadhaar",        ev.get("aadhaar")),
            ("Address",        ev.get("address")),
            ("Salary (slip)",  ev.get("salary_payslip")),
            ("Salary (offer)", ev.get("salary_offer")),
        ]
        has_any = any(v for _, v in field_map)
        if has_any:
            lines += ["| Field | Extracted Value |", "|---|---|"]
            for field_label, val in field_map:
                if val:
                    lines.append(f"| {field_label} | {val} |")
        else:
            lines.append("*No identity fields extracted from this document.*")
        lines.append("")

    # ── Appendix D: Full extraction data (raw JSON) ───────────────────────────
    if doc_extractions_raw:
        lines += [
            "---",
            "",
            "## Appendix D — Full Document Extraction Data",
            "",
            "Complete field data extracted by the AI extraction engine from each document.",
            "",
        ]
        for idx, ext in enumerate(doc_extractions_raw, start=1):
            fname    = ext.get("file_name") or f"Document {idx}"
            doc_type = (ext.get("document_type") or "unknown").replace("_", " ").title()
            data     = ext.get("extracted_data") or {}
            lines += [
                f"### {idx}. {fname}",
                f"*{doc_type}*",
                "",
                "```json",
                _json.dumps(data, indent=2, ensure_ascii=False, default=str),
                "```",
                "",
            ]

    # ── Appendix E: Full forensic analysis data (raw JSON) ───────────────────
    if scan_analysis_raw:
        lines += [
            "---",
            "",
            "## Appendix E — Full Forensic Analysis Data",
            "",
            "Complete 11-layer forensic analysis output for each scanned document.",
            "",
        ]
        for idx, scan in enumerate(scan_analysis_raw, start=1):
            src       = scan.get("source_name") or f"Scan {idx}"
            doc_type  = (scan.get("document_type") or "unknown").replace("_", " ").title()
            verdict   = scan.get("forensic_verdict") or "—"
            score     = scan.get("forgery_score")
            la        = scan.get("layered_analysis_json") or {}
            score_str = f"{score:.1f}" if isinstance(score, (int, float)) else "—"
            lines += [
                f"### {idx}. {src}",
                f"*{doc_type}  |  Verdict: {verdict}  |  Score: {score_str}*",
                "",
                "```json",
                _json.dumps(la, indent=2, ensure_ascii=False, default=str),
                "```",
                "",
            ]

    lines += [
        "---",
        "",
        "*This report is generated automatically by BaseTruth.  "
        "It supports — but does not replace — a human review and sign-off "
        "by an authorised senior reviewer.*",
    ]
    return "\n".join(lines)
