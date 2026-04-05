"""Dashboard page."""
from __future__ import annotations

from typing import Any, Dict

import streamlit as st

from basetruth.service import BaseTruthService
from basetruth.ui.components import (
    _DB_IMPORTS_OK,
    _badge,
    _display_truth_score,
    _page_title,
    _db_available_cached,
    db_dashboard_stats,
    search_entities,
)


def _page_dashboard(service: BaseTruthService) -> None:
    st.markdown(_page_title("🏠", "Dashboard"), unsafe_allow_html=True)

    with st.expander("ℹ️ How to use this screen", expanded=False):
        st.markdown(
            """
The Dashboard gives you an **at-a-glance health check** of all document processing.

- **Pending Review** — cases with high or medium risk that need your decision. Go to **Cases** to approve or reject them.
- **Documents Scanned** — total document verifications stored in the database.
- **High Risk** — documents where the Truth Score is critically low.
- **Avg Truth Score** — average score across all scans (100 = perfect integrity).

Use **Scan** (single file) or **Bulk Scan** (entire loan folder) to add new documents.
"""
        )

    if _DB_IMPORTS_OK and _db_available_cached():
        stats = db_dashboard_stats()
        if not stats:
            st.warning("Could not load dashboard statistics from the database.")
            return

        avg_s = stats.get("avg_score")
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        with m1:
            st.metric("Entities", stats.get("entities", 0),
                      help="Unique applicants stored in the database.")
            if st.button("View →", key="dash_goto_records", use_container_width=True):
                st.session_state["page"] = "records"
                st.rerun()
        with m2:
            st.metric("Docs Scanned", stats.get("total_scans", 0),
                      help="Total document scans in the database.")
            if st.button("View →", key="dash_goto_reports", use_container_width=True):
                st.session_state["page"] = "reports"
                st.rerun()
        with m3:
            st.metric("Pending Review", stats.get("pending_review", 0),
                      help="Open cases requiring Approve / Reject.")
            if st.button(
                "Review →",
                key="dash_goto_cases_pending",
                use_container_width=True,
                type="primary" if stats.get("pending_review", 0) > 0 else "secondary",
            ):
                st.session_state["page"] = "cases"
                st.rerun()
        with m4:
            st.metric("High Risk", stats.get("high_risk", 0),
                      help="Documents with high tamper risk.")
            if st.button(
                "View →",
                key="dash_goto_cases_risk",
                use_container_width=True,
                type="primary" if stats.get("high_risk", 0) > 0 else "secondary",
            ):
                st.session_state["page"] = "cases"
                st.rerun()
        with m5:
            st.metric("Auto Approved", stats.get("auto_approved", 0),
                      help="Low-risk documents automatically cleared.")
            if st.button("View →", key="dash_goto_cases_auto", use_container_width=True):
                st.session_state["page"] = "cases"
                st.rerun()
        with m6:
            st.metric(
                "Avg Score",
                f"{avg_s}/100" if avg_s is not None else "—",
                help="Average Truth Score across all scans (100 = perfect).",
            )
            if st.button("View →", key="dash_goto_records_score", use_container_width=True):
                st.session_state["page"] = "records"
                st.rerun()

        st.divider()

        if stats.get("total_scans", 0) == 0:
            st.info(
                "No documents scanned yet. Use **Scan** or **Bulk Scan** to get started."
            )
            return

        chart_col, info_col = st.columns([1, 2])
        with chart_col:
            st.subheader("Risk Distribution")
            risk_counts = {
                "High Risk": stats.get("high_risk", 0),
                "Medium Risk": stats.get("medium_risk", 0),
                "Low Risk": stats.get("low_risk", 0),
            }
            try:
                import pandas as pd  # noqa: PLC0415
                st.bar_chart(
                    pd.DataFrame(
                        {"Count": list(risk_counts.values())},
                        index=list(risk_counts.keys()),
                    )
                )
            except ImportError:
                st.json(risk_counts)

        with info_col:
            st.subheader(f"Applicants ({stats.get('entities', 0)})")
            entities_list = stats.get("risk_by_entity", [])
            if entities_list:
                try:
                    import pandas as pd  # noqa: PLC0415
                    df = pd.DataFrame(entities_list)[["entity_ref", "name", "scans"]]
                    df.columns = ["Reference", "Name", "Documents"]
                    st.dataframe(
                        df, hide_index=True, use_container_width=True, height=280
                    )
                except ImportError:
                    for e in entities_list:
                        st.write(
                            f"{e['entity_ref']} — {e['name']} ({e['scans']} docs)"
                        )
            else:
                st.info("No entities yet.")

        if stats.get("pending_review", 0) > 0:
            st.divider()
            st.subheader(
                f"⛔ Cases Requiring Your Review ({stats['pending_review']})"
            )
            st.caption("Go to **Cases** to Approve or Reject.")
            cases = service.list_cases()
            needs_review_cases = [c for c in cases if c.get("needs_review")]
            if needs_review_cases:
                try:
                    import pandas as pd  # noqa: PLC0415
                    rows = [
                        {
                            "Case": c.get("case_key", ""),
                            "Type": c.get("document_type", "").replace("_", " ").title(),
                            "Docs": str(c.get("document_count", 0)),
                            "Risk": str(c.get("max_risk_level", "low")).title(),
                            "Status": str(c.get("status", "new")).replace(
                                "_", " "
                            ).title(),
                        }
                        for c in needs_review_cases
                    ]
                    st.dataframe(
                        pd.DataFrame(rows),
                        hide_index=True,
                        use_container_width=True,
                    )
                except ImportError:
                    for c in needs_review_cases:
                        st.write(c.get("case_key", ""))
        else:
            st.divider()
            st.success("✅ All cases resolved — nothing pending review.")

    else:
        st.info(
            "📴 **Database offline** — showing file-based stats. "
            "Connect PostgreSQL for accurate counts.",
            icon=None,
        )
        reports = service.list_reports()
        ver_reports = [r for r in reports if r.get("kind") == "verification"]
        scores = [
            r.get("truth_score")
            for r in ver_reports
            if isinstance(r.get("truth_score"), int)
        ]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Documents Scanned", len(ver_reports))
        col2.metric(
            "High Risk",
            sum(1 for r in ver_reports if r.get("risk_level") == "high"),
        )
        col3.metric(
            "Avg Truth Score",
            f"{round(sum(scores)/len(scores),1)}/100" if scores else "—",
        )
        col4.metric("Reports on disk", len(reports))
