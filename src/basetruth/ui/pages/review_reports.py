"""Review Reports page — 2-level human-in-the-loop approval of entity verification reports."""
from __future__ import annotations

import streamlit as st

from basetruth.ui.components import (
    _DB_IMPORTS_OK,
    _db_available_cached,
    _minio_available_cached,
    _page_title,
    first_level_approve_entity_report,
    first_level_reject_entity_report,
    list_all_entity_reports,
    minio_get_object,
    second_level_approve_entity_report,
    second_level_reject_entity_report,
)
from basetruth.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Status helpers
# ---------------------------------------------------------------------------

def _overall_verdict_badge(rpt: dict) -> str:
    overall = (rpt.get("report_json") or {}).get("overall_verdict", "")
    return {"PASS": "✅ PASS", "FAIL": "❌ FAIL"}.get(overall, f"❓ {overall}" if overall else "—")


def _approval_summary(rpt: dict) -> str:
    l1 = rpt.get("first_level_approval")
    l2 = rpt.get("second_level_approval")
    if l1 == "Y" and l2 == "Y":
        return "✅✅ Fully Approved"
    if l1 == "N":
        return "❌ Rejected at 1st level"
    if l2 == "N":
        return "❌ Rejected at 2nd level"
    if l1 == "Y":
        return "🔄 Awaiting 2nd Review"
    return "⏳ Pending 1st Review"


# ---------------------------------------------------------------------------
# Card renderer
# ---------------------------------------------------------------------------

def _render_report_card(rpt: dict, show_actions: bool, key_prefix: str = "") -> None:
    report_ref   = rpt.get("report_ref", "?")
    entity_ref   = rpt.get("entity_ref", "—")
    entity_name  = rpt.get("entity_name", "") or entity_ref
    gen_at       = str(rpt.get("generated_at", ""))[:19].replace("T", " ")
    minio_key    = rpt.get("report_minio_key", "")
    l1           = rpt.get("first_level_approval")
    l2           = rpt.get("second_level_approval")
    verdict_badge = _overall_verdict_badge(rpt)
    approval_str  = _approval_summary(rpt)

    with st.container(border=True):
        h1, h2, h3, h4 = st.columns([3, 2, 2, 2])
        h1.markdown(f"**{report_ref}**  \n`{entity_ref}`")
        h2.markdown(f"**Applicant:** {entity_name}")
        h3.markdown(verdict_badge)
        h4.markdown(f"{approval_str}  \n🕐 {gen_at}")

        # ── PDF Preview ──────────────────────────────────────────────────────
        if minio_key and _minio_available_cached():
            with st.expander("📄 View Report PDF", expanded=False):
                pdf_bytes = minio_get_object(minio_key)
                if pdf_bytes:
                    # Render page-1 as image preview via PyMuPDF
                    try:
                        import fitz
                        import io as _io
                        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                        page = doc[0]
                        mat = fitz.Matrix(150 / 72, 150 / 72)
                        pix = page.get_pixmap(matrix=mat)
                        png_bytes = _io.BytesIO(pix.tobytes("png"))
                        doc.close()
                        st.image(png_bytes, caption=f"{report_ref} — page 1 preview",
                                 use_container_width=True)
                        st.caption("📄 Full PDF stored in MinIO — page 1 shown above.")
                    except Exception as _e:
                        log.warning("review_reports: PDF preview failed for %s: %s", report_ref, _e)
                        st.caption(f"📄 PDF stored in MinIO — preview unavailable ({_e})")

                    st.download_button(
                        "⬇ Download Full Report (PDF)",
                        data=pdf_bytes,
                        file_name=f"{report_ref}.pdf",
                        mime="application/pdf",
                        key=f"{key_prefix}dl_{report_ref}",
                        use_container_width=True,
                    )
                else:
                    st.caption("⚠️ PDF not found in MinIO for this report key.")
        elif minio_key:
            st.caption(f"📄 PDF key: `{minio_key}` — MinIO unavailable right now.")

        # ── Report JSON summary ──────────────────────────────────────────────
        rj = rpt.get("report_json") or {}
        if rj:
            with st.expander("📋 Report Details", expanded=False):
                summary_fields = {
                    k: v for k, v in rj.items()
                    if k not in ("photo_minio_key",) and not isinstance(v, (dict, list))
                }
                if summary_fields:
                    for k, v in summary_fields.items():
                        st.markdown(f"**{k.replace('_', ' ').title()}:** {v}")
                st.json(rj)

        # ── Approval controls ────────────────────────────────────────────────
        if show_actions:
            if l1 is None:
                # 1st-level review
                st.caption("**1st Level Review** — Initial reviewer decision")
                c_key = f"{key_prefix}cmt1_{report_ref}"
                comment = st.text_input(
                    "Comment (optional)",
                    key=c_key,
                    label_visibility="collapsed",
                    placeholder="Optional 1st-level comment…",
                )
                bc1, bc2, _ = st.columns([1.5, 1.5, 5])
                with bc1:
                    if st.button("✅ 1st Approve", key=f"{key_prefix}app1_{report_ref}",
                                 use_container_width=True):
                        res = first_level_approve_entity_report(
                            report_ref, approved_by="reviewer-l1", comment=comment
                        )
                        if res:
                            st.success(f"{report_ref} — 1st level approved.")
                            log.info("Review Reports: %s 1st-level APPROVED", report_ref)
                            st.rerun()
                        else:
                            st.error("Failed to approve. Check the logs.")
                with bc2:
                    if st.button("❌ 1st Reject", key=f"{key_prefix}rej1_{report_ref}",
                                 use_container_width=True):
                        res = first_level_reject_entity_report(
                            report_ref, approved_by="reviewer-l1", comment=comment
                        )
                        if res:
                            st.warning(f"{report_ref} — 1st level rejected.")
                            log.info("Review Reports: %s 1st-level REJECTED", report_ref)
                            st.rerun()
                        else:
                            st.error("Failed to reject. Check the logs.")

            elif l1 == "Y" and l2 is None:
                # 2nd-level review
                st.caption("✅ 1st level approved  |  **2nd Level Review** — Senior reviewer decision")
                c_key2 = f"{key_prefix}cmt2_{report_ref}"
                comment2 = st.text_input(
                    "2nd-level comment (optional)",
                    key=c_key2,
                    label_visibility="collapsed",
                    placeholder="Optional 2nd-level comment…",
                )
                bc1, bc2, _ = st.columns([1.5, 1.5, 5])
                with bc1:
                    if st.button("✅ 2nd Approve", key=f"{key_prefix}app2_{report_ref}",
                                 use_container_width=True):
                        res = second_level_approve_entity_report(
                            report_ref, approved_by="reviewer-l2", comment=comment2
                        )
                        if res:
                            st.success(f"{report_ref} — fully approved (2nd level).")
                            log.info("Review Reports: %s 2nd-level APPROVED", report_ref)
                            st.rerun()
                        else:
                            st.error("Failed to approve at 2nd level. Check the logs.")
                with bc2:
                    if st.button("❌ 2nd Reject", key=f"{key_prefix}rej2_{report_ref}",
                                 use_container_width=True):
                        res = second_level_reject_entity_report(
                            report_ref, approved_by="reviewer-l2", comment=comment2
                        )
                        if res:
                            st.warning(f"{report_ref} — rejected at 2nd level.")
                            log.info("Review Reports: %s 2nd-level REJECTED", report_ref)
                            st.rerun()
                        else:
                            st.error("Failed to reject at 2nd level. Check the logs.")
            else:
                # Both levels decided — show final badge only
                st.caption(_approval_summary(rpt))
        else:
            st.caption(_approval_summary(rpt))


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------

def _page_review_reports() -> None:
    st.markdown(_page_title("📋", "Review Reports"), unsafe_allow_html=True)

    st.markdown(
        "Review each verification report generated by BaseTruth. "
        "**1st-level review** is the initial check; "
        "**2nd-level review** is senior sign-off. "
        "Only reports that pass both levels are fully approved."
    )

    if not _DB_IMPORTS_OK or not _db_available_cached():
        st.warning(
            "PostgreSQL is not connected. Review Reports requires a live database.  \n"
            "Ensure `DATABASE_URL` is set and the `db` Docker service is healthy."
        )
        return

    # ── Load all reports from entity_reports table ───────────────────────────
    all_reports = list_all_entity_reports(limit=500)

    # Categorise by approval state (mirrors Review Scans logic)
    pending = [
        r for r in all_reports
        if r.get("first_level_approval") is None
    ]
    awaiting_2nd = [
        r for r in all_reports
        if r.get("first_level_approval") == "Y" and r.get("second_level_approval") is None
    ]
    fully_approved = [
        r for r in all_reports
        if r.get("first_level_approval") == "Y" and r.get("second_level_approval") == "Y"
    ]
    rejected = [
        r for r in all_reports
        if r.get("first_level_approval") == "N" or r.get("second_level_approval") == "N"
    ]

    tab_pending, tab_awaiting, tab_approved, tab_rejected, tab_all = st.tabs([
        f"⏳ Pending ({len(pending)})",
        f"🔄 Awaiting 2nd Review ({len(awaiting_2nd)})",
        f"✅ Fully Approved ({len(fully_approved)})",
        f"❌ Rejected ({len(rejected)})",
        f"📋 All ({len(all_reports)})",
    ])

    with tab_pending:
        st.caption(f"{len(pending)} report(s) awaiting 1st-level review")
        if not pending:
            st.info("No reports pending 1st-level review. Generate a report from Document Intelligence first.", icon="ℹ️")
        else:
            for rpt in pending:
                _render_report_card(rpt, show_actions=True, key_prefix="p_")

    with tab_awaiting:
        st.caption(f"{len(awaiting_2nd)} report(s) awaiting 2nd-level review")
        if not awaiting_2nd:
            st.info("No reports waiting for 2nd-level review.", icon="ℹ️")
        else:
            for rpt in awaiting_2nd:
                _render_report_card(rpt, show_actions=True, key_prefix="a2_")

    with tab_approved:
        st.caption(f"{len(fully_approved)} fully approved report(s)")
        if not fully_approved:
            st.info("No fully approved reports yet.", icon="ℹ️")
        else:
            for rpt in fully_approved:
                _render_report_card(rpt, show_actions=False, key_prefix="fa_")

    with tab_rejected:
        st.caption(f"{len(rejected)} rejected report(s)")
        if not rejected:
            st.info("No rejected reports.", icon="ℹ️")
        else:
            for rpt in rejected:
                _render_report_card(rpt, show_actions=False, key_prefix="rj_")

    with tab_all:
        st.caption(f"{len(all_reports)} total report(s)")
        if not all_reports:
            st.info("No reports found in the database.", icon="ℹ️")
        else:
            for rpt in all_reports:
                _needs_action = (
                    rpt.get("first_level_approval") is None
                    or (rpt.get("first_level_approval") == "Y" and rpt.get("second_level_approval") is None)
                )
                _render_report_card(rpt, show_actions=_needs_action, key_prefix="all_")
