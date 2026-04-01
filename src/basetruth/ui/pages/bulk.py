"""Bulk scan page."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

from basetruth.service import BaseTruthService
from basetruth.ui.components import (
    _DB_IMPORTS_OK,
    _display_truth_score,
    _render_entity_link_widget,
    _render_report_summary,
    _save_uploaded_files,
    db_available,
    minio_upload,
)


def _page_bulk(service: BaseTruthService) -> None:
    st.markdown("# 📦 Bulk Scan")

    with st.expander("ℹ️ How to use this screen", expanded=False):
        st.markdown(
            """
**Bulk Scan** lets you process an entire mortgage / loan application folder in one go.

1. Upload all the applicant's documents at once (payslips, bank statements, PAN card, Aadhaar, etc.).
2. Expand **"Associate documents with a person"** — enter the applicant's PAN, email, or phone so
   every document in this batch links to the same profile in Records.  Without this, documents
   that don't embed PAN / Aadhaar may create separate "unknown" entity entries.
3. Tick **"Run cross-month payslip comparison"** if the batch includes multi-month payslips.
4. Hit **Run bulk scan**.  A case-report PDF is auto-generated and saved to `artifacts/case_reports/`.
"""
        )

    uploads = st.file_uploader(
        "Upload multiple documents",
        type=None,
        accept_multiple_files=True,
    )
    folder_input = st.text_input("Or scan all supported files from a folder on disk")
    compare_payslips = st.checkbox(
        "Run cross-month payslip comparison after scan", value=True
    )

    bulk_forced_ref, bulk_extra_identity = _render_entity_link_widget("bulk")

    if st.button("Run bulk scan →", type="primary"):
        with st.spinner("Scanning…"):
            paths: List[Path] = []
            if uploads:
                temp_dir = Path(tempfile.mkdtemp(prefix="bt_bulk_"))
                paths.extend(_save_uploaded_files(list(uploads), temp_dir))
            if folder_input.strip():
                paths.extend(service.collect_supported_files(folder_input.strip()))
            if not paths:
                st.warning("Provide uploaded files or a folder path.")
                st.stop()

            _new_reports: List[Dict[str, Any]] = []
            _new_errors: List[str] = []
            _batch_entity_ref: str | None = bulk_forced_ref or None
            prog = st.progress(0)
            for i, p in enumerate(paths):
                try:
                    r = service.scan_document(
                        p,
                        forced_entity_ref=_batch_entity_ref or None,
                        extra_identity=bulk_extra_identity or None,
                    )
                    _new_reports.append(r)
                    if _batch_entity_ref is None and r.get("_entity_ref"):
                        _batch_entity_ref = r["_entity_ref"]
                except Exception as exc:  # noqa: BLE001
                    _new_errors.append(f"{p.name}: {exc}")
                prog.progress((i + 1) / len(paths))

        st.session_state["bt_bulk_reports"] = _new_reports
        st.session_state["bt_bulk_errors"] = _new_errors
        st.session_state["bt_bulk_compare"] = compare_payslips
        st.session_state["bt_bulk_entity_ref"] = _batch_entity_ref
        st.session_state.pop("bt_bundle_pdf_bytes", None)
        st.session_state.pop("bt_bundle_pdf_path", None)

        if _new_reports:
            try:
                import datetime  # noqa: PLC0415

                from basetruth.reporting.pdf import render_case_bundle_pdf  # noqa: PLC0415

                with st.spinner("Generating case report PDF…"):
                    _auto_reconciliation = service.reconcile_income_documents(_new_reports)
                    _ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    _auto_title = f"Case Report — {_ts}"
                    _pdf_bytes = render_case_bundle_pdf(
                        reports=_new_reports,
                        reconciliation=_auto_reconciliation,
                        case_title=_auto_title,
                    )

                if _auto_reconciliation.get("anomalies"):
                    for _err in _auto_reconciliation["anomalies"]:
                        st.session_state["bt_bulk_errors"].append(
                            f"Anomaly Detected: {_err.get('details', {}).get('explanation', _err.get('type'))}"
                        )
                    for _rep in _new_reports:
                        _ent_ref = _rep.get("_entity_ref")
                        _doc_type = _rep.get("structured_summary", {}).get(
                            "document", {}
                        ).get("type", "generic")
                        _c_key = (
                            f"{_doc_type}::{_ent_ref}"
                            if _ent_ref
                            else service._case_key_for_report(_rep)
                        )
                        service.update_case(
                            _c_key,
                            status="triage",
                            priority="high",
                            disposition="open",
                            note_text=(
                                "Cross-document reconciliation uncovered a discrepancy "
                                "flagged in the bulk case report."
                            ),
                            note_author="system",
                        )

                _reports_dir = service.artifact_root / "case_reports"
                _reports_dir.mkdir(parents=True, exist_ok=True)
                _pdf_path = _reports_dir / f"{_ts}_case_report.pdf"
                _pdf_path.write_bytes(_pdf_bytes)
                st.session_state["bt_bundle_pdf_bytes"] = _pdf_bytes
                st.session_state["bt_bundle_pdf_title"] = f"{_ts}_case_report"
                st.session_state["bt_bundle_pdf_path"] = str(_pdf_path)
                if _DB_IMPORTS_OK and _batch_entity_ref:
                    try:
                        minio_upload(
                            f"{_batch_entity_ref}/case_reports/{_ts}_case_report.pdf",
                            _pdf_bytes,
                            "application/pdf",
                        )
                    except Exception:  # noqa: BLE001
                        pass
            except Exception as _pdf_err:  # noqa: BLE001
                st.session_state["bt_bundle_pdf_bytes"] = None
                st.warning(f"Scan complete — PDF generation failed: {_pdf_err}")

    if "bt_bulk_reports" not in st.session_state:
        return

    reports: List[Dict[str, Any]] = st.session_state["bt_bulk_reports"]
    errors: List[str] = st.session_state["bt_bulk_errors"]
    compare_payslips = st.session_state.get("bt_bulk_compare", compare_payslips)

    st.success(f"Scanned {len(reports)} document(s).")
    if errors:
        with st.expander(
            f"{len(errors)} document(s) had errors -- click to expand"
        ):
            for err in errors:
                st.error(err)

    try:
        import pandas as pd  # noqa: PLC0415

        summary_rows = []
        for r in reports:
            ss = r.get("structured_summary", {})
            doc = ss.get("document", {})
            kf = ss.get("key_fields", {})
            principal = (
                kf.get("employee_name")
                or kf.get("account_holder")
                or kf.get("employer_name")
                or kf.get("company_name")
                or ""
            )
            key_amount = (
                kf.get("gross_earnings")
                or kf.get("gross_monthly_salary")
                or kf.get("annual_ctc")
                or kf.get("opening_balance")
                or ""
            )
            if key_amount:
                try:
                    key_amount = f"Rs {int(str(key_amount).replace(',', '')):,}"
                except (ValueError, TypeError):
                    key_amount = str(key_amount)
            summary_rows.append(
                {
                    "File": r.get("source", {}).get("name", ""),
                    "Type": doc.get("type", ""),
                    "Principal": str(principal)[:25] if principal else "—",
                    "Key Amount": str(key_amount) if key_amount else "—",
                    "Score": _display_truth_score(
                        r.get("tamper_assessment", {}).get("truth_score")
                    ),
                    "Risk": str(
                        r.get("tamper_assessment", {}).get("risk_level", "")
                    ).title(),
                    "Confidence": f"{int(doc.get('type_confidence', 0) * 100)}%",
                    "Parse Method": ss.get("parse_method", "?"),
                    "Fallback": "⚠️ Yes" if ss.get("parse_fallback") else "No",
                }
            )
        st.dataframe(
            pd.DataFrame(summary_rows), hide_index=True, use_container_width=True
        )
    except ImportError:
        st.json([r.get("source", {}).get("name", "") for r in reports])

    st.subheader("Document Results")
    for r in reports:
        fname = r.get("source", {}).get("name", "unknown")
        stem = Path(fname).stem
        risk = str(r.get("tamper_assessment", {}).get("risk_level", "low")).lower()
        score = r.get("tamper_assessment", {}).get("truth_score", 100)
        risk_icon = {"high": "🚨", "critical": "🚨", "medium": "⚠️", "review": "🔷"}.get(
            risk, "✅"
        )
        with st.expander(
            f"{risk_icon} {fname}  —  Score: {score}/100  |  Risk: {risk.title()}"
        ):
            _render_report_summary(r)
            r_artifacts = r.get("artifacts", {})
            dl1, dl2 = st.columns(2)
            with dl1:
                st.download_button(
                    "⬇ Download JSON report",
                    data=json.dumps(r, indent=2, ensure_ascii=False),
                    file_name=f"{stem}_verification.json",
                    mime="application/json",
                    key=f"json_{stem}",
                    use_container_width=True,
                )
            with dl2:
                pdf_path_str = r_artifacts.get("pdf_report_path", "")
                if pdf_path_str and Path(pdf_path_str).exists():
                    st.download_button(
                        "⬇ Download PDF report",
                        data=Path(pdf_path_str).read_bytes(),
                        file_name=f"{stem}_report.pdf",
                        mime="application/pdf",
                        key=f"pdf_{stem}",
                        use_container_width=True,
                    )
                else:
                    st.button(
                        "PDF (not available)",
                        disabled=True,
                        key=f"pdf_na_{stem}",
                        use_container_width=True,
                    )

    with st.expander(
        "Classifier diagnostics (click to inspect per-file classification)"
    ):
        for r in reports:
            ss = r.get("structured_summary", {})
            doc = ss.get("document", {})
            parser = ss.get("parser", {})
            fname = r.get("source", {}).get("name", "?")
            doc_type = doc.get("type", "generic")
            parse_method = ss.get("parse_method", "?")
            fallback = ss.get("parse_fallback", False)
            fallback_reason = ss.get("parse_fallback_reason", "")
            scores = doc.get("classification_scores", {})
            markers = doc.get("matched_markers", {})
            text_len = parser.get("text_length", 0)
            preview = doc.get("text_preview", "")

            st.markdown(
                f"**{fname}** → `{doc_type}` "
                f"({int(doc.get('type_confidence', 0)*100)}% confidence)"
            )
            col1, col2 = st.columns(2)
            col1.metric("Parse method", parse_method)
            col1.metric("Text extracted (chars)", text_len)
            col2.metric("Classification winner", doc_type)
            if scores:
                col2.write("All type scores:")
                col2.json(scores)
            if markers.get(doc_type):
                st.caption("Matched markers: " + ", ".join(markers[doc_type]))
            if fallback:
                st.warning(f"Parse fallback used: {fallback_reason}")
            if text_len == 0:
                st.error(
                    "No text was extracted from this document — classification will be generic. "
                    "Check that LiteParse (Node.js) is installed or Tesseract OCR is available."
                )
            elif text_len < 100:
                st.warning(
                    f"Very little text extracted ({text_len} chars). "
                    "Classification may be unreliable."
                )
            if preview:
                st.caption("Text preview: " + preview[:200])
            st.divider()

    if compare_payslips:
        comparison = service.compare_payslip_summaries_from_reports(reports)
        if comparison.get("anomalies"):
            st.subheader(
                f"Payslip anomalies — {len(comparison['anomalies'])} detected"
            )
            for anomaly in comparison["anomalies"]:
                sev = str(anomaly.get("severity", "low"))
                icon = (
                    "🚨" if sev == "high" else "⚠️" if sev == "medium" else "🔷"
                )
                with st.expander(
                    f"{icon} {anomaly.get('type', '').replace('_', ' ').title()}  "
                    f"— {anomaly.get('from_period', '')} → {anomaly.get('to_period', '')}"
                ):
                    st.json(anomaly.get("details", {}))
        else:
            st.success(
                "No payslip anomalies detected across this document set."
            )

    st.divider()
    reconciliation = service.reconcile_income_documents(reports)
    income_anomalies = reconciliation.get("anomalies", [])
    evidence = reconciliation.get("evidence", {})

    st.subheader("Cross-Document Income Reconciliation")
    evidence_rows = []
    if evidence.get("payslip_avg_monthly_gross"):
        evidence_rows.append(
            {
                "Source": f"Payslips ({evidence.get('payslip_count', 0)} docs)",
                "Monthly Gross": f"₹{evidence['payslip_avg_monthly_gross']:,}",
                "Annual (×12)": f"₹{evidence.get('payslip_annualised_gross', 0):,}",
            }
        )
    if evidence.get("letter_annual_ctc"):
        monthly = evidence.get("letter_gross_monthly")
        evidence_rows.append(
            {
                "Source": evidence.get("letter_source", "Offer letter"),
                "Monthly Gross": f"₹{monthly:,}" if monthly else "—",
                "Annual (×12)": f"₹{evidence['letter_annual_ctc']:,}",
            }
        )
    if evidence.get("form16_annual_gross"):
        evidence_rows.append(
            {
                "Source": evidence.get("form16_source", "Form 16"),
                "Monthly Gross": "—",
                "Annual (×12)": f"₹{evidence['form16_annual_gross']:,}",
            }
        )
    if evidence.get("bank_avg_salary_credit"):
        evidence_rows.append(
            {
                "Source": f"Bank statement ({evidence.get('bank_salary_credit_count', 0)} credits)",
                "Monthly Gross": f"₹{evidence['bank_avg_salary_credit']:,} (net credit)",
                "Annual (×12)": f"₹{evidence['bank_avg_salary_credit'] * 12:,}",
            }
        )
    if evidence_rows:
        try:
            import pandas as pd  # noqa: PLC0415
            st.dataframe(
                pd.DataFrame(evidence_rows),
                hide_index=True,
                use_container_width=True,
            )
        except ImportError:
            for row in evidence_rows:
                st.write(row)

    if income_anomalies:
        st.error(
            f"⚠️ {len(income_anomalies)} income inconsistenc"
            f"{'y' if len(income_anomalies) == 1 else 'ies'} detected "
            "— possible income inflation fraud"
        )
        for anomaly in income_anomalies:
            sev = str(anomaly.get("severity", "low"))
            icon = "🚨" if sev == "high" else "⚠️"
            with st.expander(
                f"{icon} {anomaly.get('type', '').replace('_', ' ').title()}  "
                f"— {anomaly.get('from_period', '')} → {anomaly.get('to_period', '')}"
            ):
                details = anomaly.get("details", {})
                explanation = details.get("explanation", "")
                if explanation:
                    st.warning(explanation)
                st.json({k: v for k, v in details.items() if k != "explanation"})

        _batch_entity_ref = st.session_state.get("bt_bulk_entity_ref")
        if _batch_entity_ref:
            _anomaly_types = [a.get("type", "") for a in income_anomalies]
            _anomaly_count = len(income_anomalies)
            _reopened = service.flag_entity_cases_for_review(
                _batch_entity_ref,
                reason=(
                    f"{_anomaly_count} cross-document income inconsistenc"
                    f"{'y' if _anomaly_count == 1 else 'ies'} detected"
                ),
                anomaly_types=_anomaly_types,
            )
            if _reopened > 0:
                st.warning(
                    f"🔓 **{_reopened} previously auto-approved case(s) have been reopened** "
                    f"for entity {_batch_entity_ref} due to cross-document anomalies. "
                    "Go to **Cases** to review them."
                )
    elif evidence_rows:
        st.success("✅ Income figures are consistent across all documents.")

    st.divider()
    _pdf_bytes_ready = st.session_state.get("bt_bundle_pdf_bytes")
    _pdf_path_saved = st.session_state.get("bt_bundle_pdf_path", "")
    if _pdf_bytes_ready:
        _pdf_title = st.session_state.get("bt_bundle_pdf_title", "case_report")
        _safe_fname = (
            "".join(c if c.isalnum() or c in "-_ " else "_" for c in _pdf_title).strip()
            + ".pdf"
        )
        st.markdown(
            f'<div class="bt-pdf-banner">'
            f'<span style="font-size:1.5rem;">📄</span>'
            f'<div><div style="font-weight:700;font-size:0.95rem;color:var(--text-color,#1e293b);">'
            f"Case Report PDF generated automatically</div>"
            f'<div style="font-size:0.8rem;color:#94a3b8;margin-top:2px;">'
            f"Saved to: {_pdf_path_saved or 'artifacts/case_reports/'}</div></div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.download_button(
            label="⬇  Download Case Report PDF",
            data=_pdf_bytes_ready,
            file_name=_safe_fname,
            mime="application/pdf",
            key="dl_bundle_pdf",
            use_container_width=True,
            type="primary",
        )
    else:
        st.info(
            "📄 **Case Report PDF** — A PDF report will be generated automatically "
            "after scanning documents. It covers all documents, income reconciliation, "
            "and an overall verdict suitable for loan officer or regulator review."
        )
