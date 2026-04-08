"""Document Intelligence page — forensic scan results, per-entity, per-document."""
from __future__ import annotations

import streamlit as st

from basetruth.logger import get_logger
from basetruth.ui.components import (
    _DB_IMPORTS_OK,
    _db_available_cached,
    _minio_available_cached,
    _page_title,
    get_entity_scans,
    minio_get_object,
    search_entities,
)

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Forensic verdict display helpers (mirror scans.py for visual consistency)
# ---------------------------------------------------------------------------

_FORENSIC_VERDICT_BADGE: dict[str, str] = {
    "ORIGINAL": "🟢 ORIGINAL",
    "UNCERTAIN": "🟡 UNCERTAIN",
    "LIKELY TAMPERED": "🟠 LIKELY TAMPERED",
    "TAMPERED": "🔴 TAMPERED",
}

_LAYER_STATUS_ICON: dict[str, str] = {
    "CLEAN": "✅",
    "SUSPICIOUS": "⚠️",
    "N/A": "➖",
    "ERROR": "❓",
}


def _render_forensics_card(scan: dict) -> None:
    """Render a full forensics breakdown card for one scan (identical layout to Scans screen).

    Reads from 'layered_analysis_json' which stores the full output of the 11-layer
    forensic engine.  Shows the overall verdict and forgery score at the top,
    followed by an evidence list and an expandable per-layer breakdown.
    If layered_analysis_json is empty (e.g. a very old scan), shows a prompt to re-scan.
    """
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


def _approval_label(scan: dict) -> str:
    """Return a short human-readable approval status string for a scan.

    Checks the two-level approval system columns first (first_level_approval,
    second_level_approval).  A scan is 'Fully Approved' only when BOTH levels
    have 'Y'.  Any 'N' at either level means rejected.  NULL means the reviewer
    hasn't made a decision yet.
    """
    fl = scan.get("first_level_approval")
    sl = scan.get("second_level_approval")
    if fl == "Y" and sl == "Y":
        return "✅✅ Fully Approved"
    if fl == "N":
        return "❌ Rejected (1st level)"
    if sl == "N":
        return "❌ Rejected (2nd level)"
    if fl == "Y":
        return "✅ 1st Approved, pending 2nd"
    return "⏳ Pending review"


def _render_scan_card(scan: dict) -> None:
    """Render one scan row as an expandable card showing verdict + image + forensics layers."""
    source_name: str = scan.get("source_name") or "—"
    doc_type: str = (scan.get("document_type") or "document").replace("_", " ").title()
    entity_ref: str = scan.get("entity_ref") or "—"
    generated_at: str = scan.get("generated_at", "")[:19].replace("T", " ")

    la = scan.get("layered_analysis_json") or {}
    summary = la.get("scan_summary", {})
    forensic_verdict = summary.get("forensic_verdict", "")
    score = summary.get("forgery_score_0_100")
    score_str = f"{score:.1f}/100" if score is not None else "—"
    verdict_label = _FORENSIC_VERDICT_BADGE.get(forensic_verdict, forensic_verdict or "—")
    approval = _approval_label(scan)

    with st.expander(
        f"{verdict_label}  ·  {source_name}  ·  {doc_type}  ·  Score: {score_str}  ·  {approval}",
        expanded=False,
    ):
        # Show document image from MinIO if available
        _image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
        _src_ext = ("." + source_name.rsplit(".", 1)[-1]).lower() if "." in source_name else ""
        if entity_ref and entity_ref != "—" and _src_ext in _image_exts and _minio_available_cached():
            _minio_key = f"{entity_ref}/{source_name}"
            _img_bytes = minio_get_object(_minio_key)
            if _img_bytes:
                with st.expander("🖼️ Original Document", expanded=False):
                    st.image(_img_bytes, caption=source_name, use_container_width=True)

        st.caption(f"Scanned: {generated_at}  |  Approval: {_approval_label(scan)}")

        with st.expander("🔬 Forensic Details", expanded=True):
            _render_forensics_card(scan)


def _page_document_intelligence() -> None:
    """Render the Document Intelligence screen.

    Shows the same forensic signals and layers as the Scans approval screen, but
    organised per-applicant. Only scans that have passed both approval levels are shown
    by default; pending and rejected scans are mentioned with a count.
    """
    st.markdown(_page_title("🧠", "Document Intelligence"), unsafe_allow_html=True)

    with st.expander("ℹ️ How to use this screen", expanded=False):
        st.markdown(
            """
**Document Intelligence** shows the forensic analysis for every scanned document,
grouped by applicant.

- **Search** an applicant by name, PAN, Aadhaar, email, phone, or BaseTruth reference (BT-XXXXXX).
- Select the entity to see all their scanned documents and the 11-layer forensic result for each.
- Click **🔬 Forensic Details** inside a card to see the full layer breakdown.
- Documents must pass the approval workflow on the **🔬 Scans** screen before they
  appear here under "Approved Documents".
"""
        )

    if not _DB_IMPORTS_OK or not _db_available_cached():
        st.warning(
            "PostgreSQL is not available. Connect the database to use Document Intelligence.\n\n"
            "Ensure `DATABASE_URL` is set and the `db` Docker service is healthy."
        )
        return

    # ── Search bar ----------------------------------------------------------
    sc1, sc2, sc3 = st.columns([4, 1.5, 1])
    with sc1:
        search_query = st.text_input(
            "Search",
            placeholder="Name, PAN, Aadhaar, email, phone, BT-XXXXXX…",
            label_visibility="collapsed",
            key="di_search_query",
        )
    field_opts = {
        "All fields": "all",
        "Name": "name",
        "PAN": "pan",
        "Aadhaar": "aadhar",
        "Email": "email",
        "Phone": "phone",
    }
    with sc2:
        search_field_label = st.selectbox(
            "Field",
            list(field_opts.keys()),
            label_visibility="collapsed",
            key="di_search_field",
        )
    search_field = field_opts[search_field_label]
    with sc3:
        do_search = st.button(
            "Search →", use_container_width=True, type="primary", key="di_do_search"
        )

    if do_search or search_query:
        results = search_entities(search_query, search_field, limit=100)
    else:
        results = search_entities("", "all", limit=50)

    if not results:
        st.info(
            "No entities found. Scan some documents first — they will appear here automatically."
        )
        return

    st.caption(f"{len(results)} applicant{'s' if len(results) != 1 else ''} found")

    # ── Entity selector -----------------------------------------------------
    ref_options = [r["entity_ref"] for r in results]
    selected_ref = st.selectbox(
        "Select applicant",
        options=ref_options,
        format_func=lambda ref: next(
            (
                f"{ref}  •  {r['first_name']} {r['last_name']}"
                for r in results
                if r["entity_ref"] == ref
            ),
            ref,
        ),
        key="di_selected_ref",
    )

    selected_entity = next((r for r in results if r["entity_ref"] == selected_ref), None)
    if not selected_entity:
        return

    # ── Entity identity strip (compact) ------------------------------------
    _fname = selected_entity.get("first_name", "")
    _lname = selected_entity.get("last_name", "")
    _pan   = selected_entity.get("pan_number") or "—"
    _aadh  = selected_entity.get("aadhar_number") or "—"
    st.markdown(
        f"""
        <div style="
          background:var(--secondary-background-color,#ffffff);
          border:1px solid rgba(99,102,241,0.20);
          border-left:4px solid #6366f1;
          border-radius:14px;
          padding:1rem 1.25rem;
          display:flex;align-items:center;gap:14px;flex-wrap:wrap;
          box-shadow:0 2px 8px rgba(99,102,241,0.08);
          margin-bottom:1rem;">
          <div style="width:40px;height:40px;border-radius:10px;
            background:linear-gradient(135deg,#6366f1,#8b5cf6);
            display:flex;align-items:center;justify-content:center;
            font-size:16px;font-weight:800;color:#fff;flex-shrink:0;">
            {(_fname[0].upper() if _fname else "?")}
          </div>
          <div style="flex:1;min-width:0;">
            <div style="font-size:1.1rem;font-weight:700;color:var(--text-color,#0f172a);">
              {_fname} {_lname}
            </div>
            <div style="font-size:11px;color:#94a3b8;margin-top:2px;">
              {selected_ref} &nbsp;·&nbsp; PAN: {_pan} &nbsp;·&nbsp; Aadhaar: {_aadh}
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Load all scans for this entity -------------------------------------
    all_scans = get_entity_scans(selected_ref)
    log.debug("Document Intelligence: loaded scans", extra={"entity_ref": selected_ref, "count": len(all_scans)})

    if not all_scans:
        st.info(
            "No scans found for this applicant. "
            "Run a Bulk Scan to capture document forensics."
        )
        return

    # Split into approved, pending, rejected for display
    approved_scans = [s for s in all_scans if s.get("first_level_approval") == "Y" and s.get("second_level_approval") == "Y"]
    pending_scans  = [s for s in all_scans if s.get("first_level_approval") is None and s.get("approved") is None]
    rejected_scans = [s for s in all_scans if s.get("first_level_approval") == "N" or s.get("second_level_approval") == "N"]

    pending_count  = len(pending_scans)
    rejected_count = len(rejected_scans)

    # Warn the user about scans awaiting review
    if pending_count or rejected_count:
        parts = []
        if pending_count:
            parts.append(f"**{pending_count}** pending review")
        if rejected_count:
            parts.append(f"**{rejected_count}** rejected")
        st.warning(
            f"⚠️ {' and '.join(parts)} scan(s) are not shown in the approved section. "
            "Go to **🔬 Scans** to approve them.",
            icon="⚠️",
        )

    total = len(all_scans)
    st.markdown(
        f"**{total} scan{'s' if total != 1 else ''}** — "
        f"{len(approved_scans)} fully approved, {pending_count} pending, {rejected_count} rejected."
    )

    # ── Tabs: Approved | All -----------------------------------------------
    tab_approved, tab_all = st.tabs([
        f"✅ Approved ({len(approved_scans)})",
        f"📋 All Scans ({total})",
    ])

    with tab_approved:
        if not approved_scans:
            st.info(
                "No fully approved scans yet. "
                "Complete the 2-level approval workflow on the **🔬 Scans** screen."
            )
        else:
            for scan in approved_scans:
                _render_scan_card(scan)

    with tab_all:
        for scan in all_scans:
            _render_scan_card(scan)
