"""Document Intelligence page — forensic scan results, per-entity, per-document."""
from __future__ import annotations

import json
import streamlit as st

from basetruth.logger import get_logger
from basetruth.ui.components import (
    _DB_IMPORTS_OK,
    _db_available_cached,
    _minio_available_cached,
    _page_title,
    get_entity_document_information,
    get_entity_reports,
    get_entity_scans,
    minio_get_object,
    save_entity_report,
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


# ---------------------------------------------------------------------------
# Cross-document analysis helper
# ---------------------------------------------------------------------------

# Field name aliases used in different document types for the same concept.
# Each tuple lists all known field names that could carry that piece of information.
_NAME_FIELDS    = ("candidate_name", "name", "employee_name", "account_holder_name",
                   "applicant_name", "holder_name")
_ADDRESS_FIELDS = ("address", "permanent_address", "residential_address", "current_address")
_PAN_FIELDS     = ("pan_number", "pan")
_AADHAR_FIELDS  = ("aadhaar_number", "aadhar_number", "uid", "aadhaar")
_SALARY_FIELDS_PAYSLIP = ("net_salary", "gross_salary", "net_pay", "ctc",
                           "total_compensation", "net_amount")
_SALARY_FIELDS_OFFER   = ("offered_salary", "ctc", "annual_ctc", "gross_salary",
                           "salary", "annual_salary", "package")


def _first(d: dict, keys: tuple) -> str | None:
    """Return the first non-empty value found in d for any of the given keys."""
    for k in keys:
        v = d.get(k)
        if v and str(v).strip():
            return str(v).strip()
    return None


def _normalise_name(name: str | None) -> str:
    """Lower-case and strip extra whitespace for a fuzzy name comparison."""
    if not name:
        return ""
    return " ".join(name.lower().split())


def _normalise_number(val: str | None) -> str:
    """Strip spaces, dashes, and upper-case an ID number for comparison."""
    if not val:
        return ""
    return val.replace(" ", "").replace("-", "").upper()


def _to_float(val: str | None) -> float | None:
    """Convert a possibly formatted salary string ('₹ 1,23,456') to a float."""
    if not val:
        return None
    import re
    digits = re.sub(r"[^\d.]", "", str(val))
    try:
        return float(digits)
    except ValueError:
        return None


def _run_cross_doc_analysis(entity: dict, extractions: list, scans: list) -> dict:
    """Perform a cross-document consistency analysis for one entity.

    Compares names, addresses, PAN, Aadhaar, salary, and forensic verdicts
    across every scanned document's extracted fields. Returns a structured
    payload suitable for storage in EntityReport.report_json.

    The approach is to collect all values seen for each field across all documents
    and then flag mismatches where two or more non-identical values are found.
    Salary comparison is done with a 30% tolerance to accommodate deductions and
    increments between a payslip and an offer/increment letter.
    """
    # Collect per-document evidence rows.
    evidence: list[dict] = []
    for ext in extractions:
        fields = ext.get("fields") or {}
        doc_type = ext.get("document_type") or "unknown"
        file_name = ext.get("file_name") or ""
        evidence.append({
            "file_name": file_name,
            "document_type": doc_type,
            "name":    _first(fields, _NAME_FIELDS),
            "address": _first(fields, _ADDRESS_FIELDS),
            "pan":     _first(fields, _PAN_FIELDS),
            "aadhaar": _first(fields, _AADHAR_FIELDS),
            "salary_payslip": _first(fields, _SALARY_FIELDS_PAYSLIP)
                               if "payslip" in doc_type.lower() else None,
            "salary_offer":   _first(fields, _SALARY_FIELDS_OFFER)
                               if any(kw in doc_type.lower()
                                      for kw in ("offer", "increment", "appointment")) else None,
        })

    # ── Name consistency ─────────────────────────────────────────────────────
    names = [_normalise_name(r["name"]) for r in evidence if r["name"]]
    unique_names = list(dict.fromkeys(names))  # preserve order, deduplicate
    name_status = "PASS" if len(unique_names) <= 1 else "MISMATCH"
    name_detail = f"Values seen: {unique_names}" if len(unique_names) > 1 else (unique_names[0] if unique_names else "No name found")

    # ── Address consistency ──────────────────────────────────────────────────
    addresses = [r["address"] for r in evidence if r["address"]]
    unique_addresses = list(dict.fromkeys(addresses))
    address_status = "PASS" if len(unique_addresses) <= 1 else "MISMATCH"
    address_detail = f"{len(unique_addresses)} distinct addresses found" if len(unique_addresses) > 1 else (unique_addresses[0] if unique_addresses else "No address found")

    # ── PAN consistency ──────────────────────────────────────────────────────
    pans = [_normalise_number(r["pan"]) for r in evidence if r["pan"]]
    unique_pans = list(dict.fromkeys(pans))
    pan_status = "PASS" if len(unique_pans) <= 1 else "MISMATCH"
    pan_detail = f"Values seen: {unique_pans}" if len(unique_pans) > 1 else (unique_pans[0] if unique_pans else "No PAN found")

    # ── Aadhaar consistency ──────────────────────────────────────────────────
    aadhaars = [_normalise_number(r["aadhaar"]) for r in evidence if r["aadhaar"]]
    unique_aadhaars = list(dict.fromkeys(aadhaars))
    aadhaar_status = "PASS" if len(unique_aadhaars) <= 1 else "MISMATCH"
    aadhaar_detail = f"Values seen: {unique_aadhaars}" if len(unique_aadhaars) > 1 else (unique_aadhaars[0] if unique_aadhaars else "No Aadhaar found")

    # ── Salary cross-check ───────────────────────────────────────────────────
    payslip_salaries = [_to_float(r["salary_payslip"]) for r in evidence if r["salary_payslip"]]
    offer_salaries   = [_to_float(r["salary_offer"])   for r in evidence if r["salary_offer"]]
    salary_status = "SKIP"
    salary_detail = "No payslip and/or offer salary data available."
    if payslip_salaries and offer_salaries:
        # Compare the average payslip net salary to the average offered CTC.
        # Allow 30% tolerance — deductions and date-of-joining mid-month can
        # cause legitimate differences between the two numbers.
        avg_pay = sum(payslip_salaries) / len(payslip_salaries)
        avg_off = sum(offer_salaries)   / len(offer_salaries)
        if avg_off > 0:
            ratio = abs(avg_pay - avg_off) / avg_off
            salary_status = "PASS" if ratio <= 0.30 else "MISMATCH"
            salary_detail = (
                f"Payslip avg ₹{avg_pay:,.0f} vs offer avg ₹{avg_off:,.0f} "
                f"({ratio * 100:.1f}% difference)"
            )

    # ── Forensic verdict summary ──────────────────────────────────────────────
    verdicts = [s.get("verdict") or "" for s in scans]
    tampered_docs = [
        s.get("source_name", "?") for s in scans
        if "TAMPERED" in (s.get("verdict") or "").upper()
    ]
    forensic_status = "CLEAR" if not tampered_docs else "TAMPERED"
    forensic_detail = (
        f"{len(tampered_docs)} document(s) flagged as TAMPERED: {tampered_docs}"
        if tampered_docs else
        f"All {len(scans)} document(s) are forensically clean."
    )

    # ── Overall verdict ───────────────────────────────────────────────────────
    all_checks = [name_status, pan_status, aadhaar_status, salary_status, forensic_status]
    # Any MISMATCH or TAMPERED causes an overall FAIL; SKIP checks are neutral.
    if any(c in ("MISMATCH", "TAMPERED") for c in all_checks):
        overall = "FAIL"
    else:
        overall = "PASS"

    # ── Assemble report payload ───────────────────────────────────────────────
    return {
        "entity_ref":        entity.get("entity_ref", ""),
        "entity_name":       f"{entity.get('first_name', '')} {entity.get('last_name', '')}".strip(),
        "overall_verdict":   overall,
        "documents_analysed": len(extractions),
        "scans_reviewed":     len(scans),
        "checks": {
            "name":     {"status": name_status,     "detail": name_detail},
            "address":  {"status": address_status,  "detail": address_detail},
            "pan":      {"status": pan_status,       "detail": pan_detail},
            "aadhaar":  {"status": aadhaar_status,   "detail": aadhaar_detail},
            "salary":   {"status": salary_status,    "detail": salary_detail},
            "forensics":{"status": forensic_status,  "detail": forensic_detail},
        },
        "per_document_evidence": evidence,
    }


def _render_entity_reports_section(entity_ref: str) -> None:
    """Display existing EntityReport records for this entity and allow regeneration.

    Shows a summary card for each saved report with its approval status and a
    collapsible JSON payload so analysts can inspect the full findings.
    """
    reports = get_entity_reports(entity_ref)
    if not reports:
        st.info("No final report generated yet. Click **🎯 Generate Final Report** below.")
        return

    for rpt in reports:
        first = rpt.get("first_level_approval")
        second = rpt.get("second_level_approval")
        if second == "Y":
            badge = "🟢 Fully Approved"
        elif second == "N" or first == "N":
            badge = "🔴 Rejected"
        elif first == "Y":
            badge = "🟡 Pending 2nd-Level"
        else:
            badge = "⏳ Pending Review"

        overall = (rpt.get("report_json") or {}).get("overall_verdict", "?")
        ov_icon = "✅" if overall == "PASS" else "❌" if overall == "FAIL" else "❓"

        with st.expander(
            f"{rpt['report_ref']}  ·  {ov_icon} {overall}  ·  {badge}  "
            f"·  Generated {rpt.get('generated_at', '')[:10]}",
            expanded=(second is None and first is None),
        ):
            # Summarise each check as a pass/fail row.
            checks = (rpt.get("report_json") or {}).get("checks", {})
            for check_name, chk in checks.items():
                icon = "✅" if chk["status"] == "PASS" else (
                       "❌" if chk["status"] in ("MISMATCH", "TAMPERED", "FAIL") else "➖"
                )
                st.markdown(f"**{icon} {check_name.capitalize()}** — {chk['detail']}")

            # Full JSON payload is available but collapsed to avoid clutter.
            with st.expander("Full report JSON", expanded=False):
                st.json(rpt.get("report_json") or {})

            # Show approval trail.
            if first:
                st.caption(
                    f"1st-level: {'✅ Approved' if first == 'Y' else '❌ Rejected'} "
                    f"by {rpt.get('first_level_approved_by') or 'unknown'}"
                    f"  {rpt.get('first_level_approval_comment') or ''}"
                )
            if second:
                st.caption(
                    f"2nd-level: {'✅ Approved' if second == 'Y' else '❌ Rejected'} "
                    f"by {rpt.get('second_level_approved_by') or 'unknown'}"
                    f"  {rpt.get('second_level_approval_comment') or ''}"
                )


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
    log.debug(f"Document Intelligence Engine: Successfully retrieved {len(all_scans)} scans associated with applicant '{selected_ref}'.", extra={"entity_ref": selected_ref, "count": len(all_scans)})

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

    # ── Generate Final Report ─────────────────────────────────────────────────
    st.divider()
    st.markdown("#### 🎯 Final Verification Report")
    st.caption(
        "Generates a cross-document consistency report comparing names, PAN, Aadhaar, "
        "address, salary, and forensic verdicts across all documents for this applicant."
    )

    _existing_reports = get_entity_reports(selected_ref)
    _has_pending = any(
        r.get("first_level_approval") is None for r in _existing_reports
    )

    if _has_pending:
        st.info(
            "ℹ️ A pending report already exists. Generating again will refresh it "
            "with the latest document data (it can be regenerated until approved)."
        )

    if st.button("🎯 Generate Final Report", type="primary", key="di_gen_report"):
        with st.spinner("Running cross-document analysis…"):
            # Fetch all document extractions for this entity.
            extractions = get_entity_document_information(selected_ref)
            report_payload = _run_cross_doc_analysis(selected_entity, extractions, all_scans)
            result = save_entity_report(selected_ref, report_payload)

        if result:
            st.success(
                f"✅ Report **{result['report_ref']}** saved. "
                "Go to **📂 Cases** to approve it, or download it from **📊 Reports**."
            )
            log.info(
                "Final report generated from Document Intelligence",
                extra={"entity_ref": selected_ref, "report_ref": result["report_ref"]},
            )
            st.rerun()
        else:
            st.error(
                "❌ Could not save the report. Check the database connection and try again."
            )

    # ── Existing reports section ──────────────────────────────────────────────
    if _existing_reports:
        st.markdown("##### Saved Reports")
        _render_entity_reports_section(selected_ref)
