"""Scans approval screen — 1st-level and 2nd-level human-in-the-loop review of forensic results."""
from __future__ import annotations

import json

import streamlit as st

from basetruth.ui.components import (
    _DB_IMPORTS_OK,
    _db_available_cached,
    _minio_available_cached,
    _page_title,
    approve_scan,
    get_scan_with_forensics,
    list_all_scans_with_status,
    list_pending_scans,
    minio_get_object,
    reject_scan,
    second_level_approve_scan,
    second_level_reject_scan,
)
from basetruth.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Verdict colour maps
# ---------------------------------------------------------------------------
_FORENSIC_VERDICT_BADGE: dict[str, str] = {
    "ORIGINAL": "🟢 ORIGINAL",
    "UNCERTAIN": "🟡 UNCERTAIN",
    "LIKELY TAMPERED": "🟠 LIKELY TAMPERED",
    "TAMPERED": "🔴 TAMPERED",
}

_APPROVAL_BADGE: dict[str | None, str] = {
    None: "⏳ Pending",
    "approved": "✅ Fully Approved",
    "rejected": "❌ Rejected",
}

_LAYER_STATUS_ICON: dict[str, str] = {
    "CLEAN": "✅",
    "SUSPICIOUS": "⚠️",
    "N/A": "➖",
    "ERROR": "❓",
}


def _show_approval_badge(scan: dict) -> None:
    """Render a read-only approval status line showing both review levels."""
    first_level = scan.get("first_level_approval")
    second_level = scan.get("second_level_approval")
    approval_status = scan.get("approved")

    # Build a human-readable two-level status line
    if first_level == "Y" and second_level == "Y":
        line = "✅✅ **Fully Approved** (both levels)"
        l2_by = scan.get("second_level_approved_by", "")
        l2_at = (scan.get("second_level_approved_at") or "")[:19].replace("T", " ")
        if l2_by:
            line += f" — 2nd by **{l2_by}**"
        if l2_at:
            line += f" on {l2_at}"
    elif first_level == "N":
        line = "❌ **Rejected** at 1st level"
        l1_by = scan.get("first_level_approved_by", "")
        l1_at = (scan.get("first_level_approved_at") or "")[:19].replace("T", " ")
        if l1_by:
            line += f" by **{l1_by}**"
        if l1_at:
            line += f" on {l1_at}"
        cmt = scan.get("first_level_approval_comment", "")
        if cmt:
            line += f' — "{cmt}"'
    elif second_level == "N":
        line = "❌ **Rejected** at 2nd level"
        l2_by = scan.get("second_level_approved_by", "")
        if l2_by:
            line += f" by **{l2_by}**"
        cmt = scan.get("second_level_approval_comment", "")
        if cmt:
            line += f' — "{cmt}"'
    elif approval_status == "approved":
        # Legacy single-level approval (old records before 2-level system)
        reviewer = scan.get("approved_by", "")
        at_str = (scan.get("approved_at") or "")[:19].replace("T", " ")
        line = "✅ **Approved**"
        if reviewer:
            line += f" by **{reviewer}**"
        if at_str:
            line += f" on {at_str}"
    elif approval_status == "rejected":
        reviewer = scan.get("approved_by", "")
        line = "❌ **Rejected**"
        if reviewer:
            line += f" by **{reviewer}**"
        cmt = scan.get("approval_comment", "")
        if cmt:
            line += f' — "{cmt}"'
    else:
        line = "⏳ Pending review"

    st.caption(line)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _render_forensics_card(scan: dict) -> None:
    """Render an expandable forensics detail card for one scan row."""
    la = scan.get("layered_analysis_json") or {}
    summary = la.get("scan_summary", {})
    layers = la.get("layers", {})

    if not la:
        st.caption("No forensics data available — re-save the document to trigger analysis.")
        return

    verdict = summary.get("forensic_verdict", "—")
    score = summary.get("forgery_score_0_100", 0)
    explanation = summary.get("overall_explanation", "")
    evidence = summary.get("evidence", [])

    col1, col2, col3 = st.columns(3)
    col1.metric("Forensic Verdict", _FORENSIC_VERDICT_BADGE.get(verdict, verdict))
    col2.metric("Forgery Score", f"{score:.1f} / 100")
    col3.metric("File", summary.get("format", "—"))

    if explanation:
        st.info(explanation)

    if evidence:
        with st.expander("🔍 Evidence", expanded=False):
            for ev in evidence:
                st.markdown(f"- {ev}")

    if layers:
        with st.expander("🧪 All 11 Forensic Layers", expanded=False):
            for layer_key in sorted(layers.keys()):
                layer = layers[layer_key]
                icon = _LAYER_STATUS_ICON.get(layer.get("status", ""), "➖")
                name = layer.get("name", layer_key)
                plain = layer.get("plain_english", "")
                status = layer.get("status", "N/A")
                col_a, col_b = st.columns([1, 4])
                col_a.markdown(f"**{icon} {name}**  \n`{status}`")
                col_b.caption(plain or "—")

            st.divider()
            with st.expander("📋 Raw JSON", expanded=False):
                st.json(la)


def _render_scan_row(scan: dict, show_approve_buttons: bool = True, key_prefix: str = "") -> None:
    """Render a single scan as a card with optional approve/reject controls.

    key_prefix must be unique per tab so that the same scan_id can appear in
    multiple tabs (e.g. Pending and All) without duplicating Streamlit widget keys.
    """
    scan_id: int = scan["id"]
    source_name: str = scan["source_name"] or "—"
    doc_type: str = scan["document_type"] or "generic"
    entity_ref: str = scan["entity_ref"]
    entity_name: str = scan["entity_name"]
    generated_at: str = scan.get("generated_at", "")[:19].replace("T", " ")
    # Overall approval status derived from two-level system (see store._scan_to_summary_dict)
    approval_status = scan.get("approved")
    first_level = scan.get("first_level_approval")   # 'Y' | 'N' | None
    second_level = scan.get("second_level_approval")  # 'Y' | 'N' | None

    la = scan.get("layered_analysis_json") or {}
    summary = la.get("scan_summary", {})
    # Forensic verdict and score come from layered_analysis_json — no DB columns for these
    forensic_verdict = summary.get("forensic_verdict", "")
    score = summary.get("forgery_score_0_100")

    with st.container(border=True):
        h1, h2, h3, h4 = st.columns([3, 2, 2, 2])
        h1.markdown(f"**{source_name}**  \n`{doc_type}`")
        h2.markdown(f"**Entity:** {entity_ref}  \n{entity_name}")
        h3.markdown(_FORENSIC_VERDICT_BADGE.get(forensic_verdict, forensic_verdict or "—"))
        score_str = f"{score:.1f}/100" if score is not None else "—"
        h4.markdown(f"**Score:** {score_str}  \n🕐 {generated_at}")

        # ── Original image preview ──────────────────────────────────────────
        # Try to load the source document image from MinIO (key = entity_ref/source_name).
        # We only attempt this when MinIO is available to avoid freezing the UI.
        _image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
        _src_ext = ("." + source_name.rsplit(".", 1)[-1]).lower() if "." in source_name else ""
        if entity_ref and entity_ref != "—" and _src_ext in _image_exts and _minio_available_cached():
            _minio_key = f"{entity_ref}/{source_name}"
            _img_bytes = minio_get_object(_minio_key)
            if _img_bytes:
                with st.expander("🖼️ Original Document", expanded=False):
                    st.image(_img_bytes, caption=source_name, use_container_width=True)

        with st.expander("🔬 Forensic Details", expanded=False):
            _render_forensics_card(scan)

        # ── Approval controls ───────────────────────────────────────────────
        if show_approve_buttons:
            if first_level is None and approval_status is None:
                # ── 1st-level review has not happened yet ──
                st.caption("**1st Level Review** — Initial reviewer decision")
                comment_key = f"{key_prefix}scan_comment_{scan_id}"
                comment = st.text_input(
                    "Comment (optional)",
                    key=comment_key,
                    label_visibility="collapsed",
                    placeholder="Optional 1st-level reviewer comment…",
                )
                btn_col1, btn_col2, _ = st.columns([1.5, 1.5, 5])
                with btn_col1:
                    if st.button("✅ 1st Approve", key=f"{key_prefix}approve_{scan_id}", use_container_width=True):
                        result = approve_scan(scan_id, approved_by="reviewer-l1", comment=comment)
                        if result:
                            st.success(f"Scan #{scan_id} — 1st level approved.")
                            log.info("Scan 1st-level approved via UI", extra={"scan_id": scan_id})
                            st.rerun()
                        else:
                            st.error(f"Failed to approve scan #{scan_id}. Check the logs.")
                with btn_col2:
                    if st.button("❌ 1st Reject", key=f"{key_prefix}reject_{scan_id}", use_container_width=True):
                        result = reject_scan(scan_id, approved_by="reviewer-l1", comment=comment)
                        if result:
                            st.warning(f"Scan #{scan_id} — 1st level rejected.")
                            log.info("Scan 1st-level rejected via UI", extra={"scan_id": scan_id})
                            st.rerun()
                        else:
                            st.error(f"Failed to reject scan #{scan_id}. Check the logs.")

            elif first_level == "Y" and second_level is None:
                # ── 1st level approved, waiting for 2nd level ──
                st.caption("✅ 1st level approved  |  **2nd Level Review** — Senior reviewer decision")
                comment_key = f"{key_prefix}scan_comment2_{scan_id}"
                comment2 = st.text_input(
                    "2nd-level comment (optional)",
                    key=comment_key,
                    label_visibility="collapsed",
                    placeholder="Optional 2nd-level reviewer comment…",
                )
                btn_col1, btn_col2, _ = st.columns([1.5, 1.5, 5])
                with btn_col1:
                    if st.button("✅ 2nd Approve", key=f"{key_prefix}approve2_{scan_id}", use_container_width=True):
                        result = second_level_approve_scan(scan_id, approved_by="reviewer-l2", comment=comment2)
                        if result:
                            st.success(f"Scan #{scan_id} — fully approved (2nd level).")
                            log.info("Scan 2nd-level approved via UI", extra={"scan_id": scan_id})
                            st.rerun()
                        else:
                            st.error(f"Failed to 2nd-level approve scan #{scan_id}. Check the logs.")
                with btn_col2:
                    if st.button("❌ 2nd Reject", key=f"{key_prefix}reject2_{scan_id}", use_container_width=True):
                        result = second_level_reject_scan(scan_id, approved_by="reviewer-l2", comment=comment2)
                        if result:
                            st.warning(f"Scan #{scan_id} — rejected at 2nd level.")
                            log.info("Scan 2nd-level rejected via UI", extra={"scan_id": scan_id})
                            st.rerun()
                        else:
                            st.error(f"Failed to 2nd-level reject scan #{scan_id}. Check the logs.")
            else:
                # ── Both levels decided — show final badge ──
                _show_approval_badge(scan)
        else:
            _show_approval_badge(scan)


# ---------------------------------------------------------------------------
# Main page function
# ---------------------------------------------------------------------------

def _page_scans() -> None:
    st.markdown(_page_title("🔬", "Scans"), unsafe_allow_html=True)

    st.markdown(
        "Review each scan's 11-layer forensic analysis. "
        "**1st-level review** is the initial check; "
        "**2nd-level review** is senior sign-off. "
        "Only scans that pass both levels are fully approved."
    )

    if not _DB_IMPORTS_OK or not _db_available_cached():
        st.warning(
            "PostgreSQL is not connected. Scans require a live database.  \n"
            "Ensure `DATABASE_URL` is set and the `db` Docker service is healthy."
        )
        return

    # ── Build categorised lists from a single DB query ──────────────────────
    all_scans = list_all_scans_with_status(limit=500)

    # Pending: no first-level decision yet, and no legacy approved value
    pending = [
        s for s in all_scans
        if s.get("first_level_approval") is None and s.get("approved") is None
    ]
    # Awaiting 2nd: 1st level approved, 2nd level not yet decided
    awaiting_2nd = [
        s for s in all_scans
        if s.get("first_level_approval") == "Y" and s.get("second_level_approval") is None
    ]
    # Fully approved: both levels said Y (or legacy single-level approved)
    fully_approved = [
        s for s in all_scans
        if (s.get("first_level_approval") == "Y" and s.get("second_level_approval") == "Y")
        or (s.get("first_level_approval") is None and s.get("approved") == "approved")
    ]
    # Rejected: any level said N (or legacy rejected)
    rejected = [
        s for s in all_scans
        if s.get("first_level_approval") == "N"
        or s.get("second_level_approval") == "N"
        or (s.get("first_level_approval") is None and s.get("approved") == "rejected")
    ]

    tab_pending, tab_awaiting, tab_approved, tab_rejected, tab_all = st.tabs(
        ["⏳ Pending", "🔄 Awaiting 2nd Review", "✅ Fully Approved", "❌ Rejected", "📋 All"]
    )

    with tab_pending:
        st.caption(f"{len(pending)} scan(s) awaiting 1st-level review")
        if not pending:
            st.info("No scans pending 1st-level review. Run a Bulk Scan to populate this list.", icon="ℹ️")
        else:
            for scan in pending:
                # key_prefix "p_" makes widget keys unique in this tab
                _render_scan_row(scan, show_approve_buttons=True, key_prefix="p_")

    with tab_awaiting:
        st.caption(f"{len(awaiting_2nd)} scan(s) awaiting 2nd-level review")
        if not awaiting_2nd:
            st.info("No scans waiting for 2nd-level review.", icon="ℹ️")
        else:
            for scan in awaiting_2nd:
                # key_prefix "a2_" ensures unique keys in this tab
                _render_scan_row(scan, show_approve_buttons=True, key_prefix="a2_")

    with tab_approved:
        st.caption(f"{len(fully_approved)} fully approved scan(s)")
        if not fully_approved:
            st.info("No fully approved scans yet.", icon="ℹ️")
        else:
            for scan in fully_approved:
                _render_scan_row(scan, show_approve_buttons=False, key_prefix="fa_")

    with tab_rejected:
        st.caption(f"{len(rejected)} rejected scan(s)")
        if not rejected:
            st.info("No rejected scans.", icon="ℹ️")
        else:
            for scan in rejected:
                _render_scan_row(scan, show_approve_buttons=False, key_prefix="rj_")

    with tab_all:
        st.caption(f"{len(all_scans)} total scan(s)")
        if not all_scans:
            st.info("No scans found in the database.", icon="ℹ️")
        else:
            for scan in all_scans:
                # Determine whether this scan still needs action
                _needs_action = (
                    (scan.get("first_level_approval") is None and scan.get("approved") is None)
                    or (scan.get("first_level_approval") == "Y" and scan.get("second_level_approval") is None)
                )
                # key_prefix "all_" keeps this tab's keys distinct from all other tabs
                _render_scan_row(scan, show_approve_buttons=_needs_action, key_prefix="all_")

