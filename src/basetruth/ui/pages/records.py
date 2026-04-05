"""Records page — entity search, identity card, scan history, edit panel."""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from basetruth.ui.components import (
    _DB_IMPORTS_OK,
    _badge,
    _page_title,
    _render_report_summary,
    _db_available_cached,
    get_entity_identity_checks,
    get_entity_latest_pdf,
    get_entity_scans,
    search_entities,
    update_entity,
)


def _page_records() -> None:
    st.markdown(_page_title("🗂️", "Records"), unsafe_allow_html=True)

    with st.expander("ℹ️ How to use this screen", expanded=False):
        st.markdown(
            """
Records shows every **applicant (entity)** in the database and all the documents linked to them.

- **Search** — type a name, PAN, Aadhaar, email, phone number, or BaseTruth reference
  (BT-XXXXXX) to find the right person.
- **Entity card** — click a result to expand the full entity card: all identity fields,
  risk summary, and individual scan results.
- **Download PDF** — each entity card has a "Download entity report" button that produces
  a single PDF covering every document scanned for that person.
- **Linking tip** — if the same person appears multiple times, use the Scan or Bulk Scan
  page with the entity-linking widget to attach future documents to the correct existing record.
"""
        )

    if not _DB_IMPORTS_OK or not _db_available_cached():
        st.warning(
            "PostgreSQL is not available. Connect the database to use the Records feature.\n\n"
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
            key="rec_search_query",
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
            key="rec_search_field",
        )
    search_field = field_opts[search_field_label]
    with sc3:
        do_search = st.button(
            "Search →", use_container_width=True, type="primary", key="rec_do_search"
        )

    if do_search or search_query:
        results = search_entities(search_query, search_field, limit=100)
    else:
        results = search_entities("", "all", limit=50)

    # ── Entity table --------------------------------------------------------
    if not results:
        st.info(
            "No records found. Scan some documents first — they will be stored automatically."
        )
        return

    st.subheader(f"{len(results)} record{'s' if len(results) != 1 else ''} found")
    try:
        import pandas as pd  # noqa: PLC0415

        tbl_rows = [
            {
                "Ref #": r["entity_ref"],
                "First Name": r["first_name"],
                "Last Name": r["last_name"],
                "PAN": r["pan_number"],
                "Aadhaar": r["aadhar_number"],
                "Email": r["email"],
                "Phone": r["phone"],
                "Scans": r["scan_count"],
                "Latest Risk": str(r["latest_risk"]).title() if r["latest_risk"] else "—",
                "Score": r["latest_score"] if r["latest_score"] is not None else "—",
            }
            for r in results
        ]
        st.dataframe(pd.DataFrame(tbl_rows), hide_index=True, use_container_width=True)
    except ImportError:
        for r in results:
            st.write(f"{r['entity_ref']} — {r['first_name']} {r['last_name']}")

    st.divider()

    # ── Entity detail panel -------------------------------------------------
    ref_options = [r["entity_ref"] for r in results]
    selected_ref = st.selectbox(
        "Open entity record",
        options=ref_options,
        format_func=lambda ref: next(
            (
                f"{ref}  •  {r['first_name']} {r['last_name']}"
                for r in results
                if r["entity_ref"] == ref
            ),
            ref,
        ),
        key="rec_selected_ref",
    )

    selected_entity = next(
        (r for r in results if r["entity_ref"] == selected_ref), None
    )
    if not selected_entity:
        return

    # ---- Identity card -----------------------------------------------------
    _pan = selected_entity["pan_number"] or "—"
    _aadh = selected_entity["aadhar_number"] or "—"
    _email = selected_entity["email"] or "—"
    _phone = selected_entity["phone"] or "—"
    _scans = selected_entity["scan_count"]
    _since = str(selected_entity["created_at"])[:10]
    _fname = selected_entity["first_name"]
    _lname = selected_entity["last_name"]
    st.markdown(
        f"""
        <div class="bt-entity-card" style="
          background:var(--secondary-background-color,#ffffff);
          border:1px solid rgba(99,102,241,0.20);
          border-left:4px solid #6366f1;
          box-shadow:0 2px 12px rgba(99,102,241,0.08);">
          <div style="display:flex;align-items:center;gap:14px;margin-bottom:16px;flex-wrap:wrap;">
            <div style="width:44px;height:44px;border-radius:12px;
              background:linear-gradient(135deg,#6366f1,#8b5cf6);
              display:flex;align-items:center;justify-content:center;
              font-size:18px;font-weight:800;color:#fff;flex-shrink:0;">
              {_fname[0].upper() if _fname else '?'}
            </div>
            <div>
              <div class="bt-entity-name">{_fname} {_lname}</div>
              <div style="margin-top:4px;">
                <span style="font-size:11px;font-weight:700;
                  background:rgba(99,102,241,0.12);color:#6366f1;
                  border:1px solid rgba(99,102,241,0.28);
                  padding:2px 10px;border-radius:6px;">{selected_ref}</span>
              </div>
            </div>
          </div>
          <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:14px 24px;">
            <div>
              <div style="font-size:10px;font-weight:700;color:#94a3b8;
                text-transform:uppercase;letter-spacing:0.08em;margin-bottom:3px;">PAN</div>
              <div class="bt-entity-field-value">{_pan}</div>
            </div>
            <div>
              <div style="font-size:10px;font-weight:700;color:#94a3b8;
                text-transform:uppercase;letter-spacing:0.08em;margin-bottom:3px;">Aadhaar</div>
              <div class="bt-entity-field-value">{_aadh}</div>
            </div>
            <div>
              <div style="font-size:10px;font-weight:700;color:#94a3b8;
                text-transform:uppercase;letter-spacing:0.08em;margin-bottom:3px;">Email</div>
              <div class="bt-entity-field-value">{_email}</div>
            </div>
            <div>
              <div style="font-size:10px;font-weight:700;color:#94a3b8;
                text-transform:uppercase;letter-spacing:0.08em;margin-bottom:3px;">Phone</div>
              <div class="bt-entity-field-value">{_phone}</div>
            </div>
            <div>
              <div style="font-size:10px;font-weight:700;color:#94a3b8;
                text-transform:uppercase;letter-spacing:0.08em;margin-bottom:3px;">Documents Scanned</div>
              <div class="bt-entity-field-value">{_scans}</div>
            </div>
            <div>
              <div style="font-size:10px;font-weight:700;color:#94a3b8;
                text-transform:uppercase;letter-spacing:0.08em;margin-bottom:3px;">Member Since</div>
              <div class="bt-entity-field-value">{_since}</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- Entity-level PDF download ----------------------------------------
    _entity_pdf, _entity_pdf_src = get_entity_latest_pdf(selected_ref)
    if _entity_pdf:
        _pdf_label = (
            f"{Path(_entity_pdf_src).stem}_report.pdf"
            if _entity_pdf_src
            else f"{selected_ref}_report.pdf"
        )
        st.download_button(
            "📥  Download Latest PDF Report",
            data=_entity_pdf,
            file_name=_pdf_label,
            mime="application/pdf",
            key=f"entity_pdf_{selected_ref}",
            type="primary",
        )
    else:
        st.caption("No PDF report available yet for this entity.")

    # ---- Identity Checks history -------------------------------------------
    id_checks = get_entity_identity_checks(selected_ref)
    if id_checks:
        st.divider()
        st.subheader("Identity Verification History")
        try:
            import pandas as pd  # noqa: PLC0415

            id_df = pd.DataFrame(
                [
                    {
                        "Date": c["created_at"][:19] if c.get("created_at") else "-",
                        "Type": c["check_type"].replace("_", " ").title(),
                        "Verdict": c["verdict"],
                        "Score": (
                            f"{c['display_score']:.1f}%"
                            if c.get("display_score")
                            else "-"
                        ),
                        "Match": "Yes" if c.get("is_match") else "No",
                        "Liveness": (
                            ("Pass" if c.get("liveness_passed") else "Fail")
                            if c["check_type"] == "video_kyc"
                            else "-"
                        ),
                        "Document": c.get("doc_filename", ""),
                    }
                    for c in id_checks
                ]
            )
            st.dataframe(id_df, hide_index=True, use_container_width=True)
        except ImportError:
            for c in id_checks:
                st.write(f"{c['check_type']} — {c['verdict']}")

    # ---- Edit identity fields inline -------------------------------------
    with st.expander("✏️  Edit identity details", expanded=False):
        with st.form(f"edit_entity_{selected_ref}"):
            e1, e2 = st.columns(2)
            f_first = e1.text_input("First name", value=selected_entity["first_name"])
            f_last = e2.text_input("Last name", value=selected_entity["last_name"])
            e3, e4 = st.columns(2)
            f_email = e3.text_input("Email", value=selected_entity["email"])
            f_phone = e4.text_input("Phone", value=selected_entity["phone"])
            e5, e6 = st.columns(2)
            f_pan = e5.text_input("PAN number", value=selected_entity["pan_number"])
            f_aadhar = e6.text_input("Aadhaar number", value=selected_entity["aadhar_number"])
            if st.form_submit_button("Save changes", type="primary"):
                result = update_entity(
                    selected_ref,
                    {
                        "first_name": f_first,
                        "last_name": f_last,
                        "email": f_email,
                        "phone": f_phone,
                        "pan_number": f_pan,
                        "aadhar_number": f_aadhar,
                    },
                )
                if result:
                    st.success("Record updated.")
                    st.rerun()
                else:
                    st.error("Update failed — check DB connection.")

    # ---- Scan history for this entity ------------------------------------
    scans = get_entity_scans(selected_ref)
    st.subheader(
        f"Document history  ({len(scans)} scan{'s' if len(scans) != 1 else ''})"
    )

    if not scans:
        st.info("No documents scanned for this entity yet.")
        return

    st.caption(
        "Expand a row to view forensic details. Download the entity PDF report above."
    )
    for sc in scans:
        risk = str(sc.get("risk_level", "low")).lower()
        score = sc.get("truth_score", "")
        doc_type = str(sc.get("document_type", "generic")).replace("_", " ").title()
        fname = sc.get("source_name", "unknown")
        ts = str(sc.get("generated_at", ""))[:19].replace("T", " ")
        risk_icon = {"high": "🚨", "medium": "⚠️", "review": "🔷"}.get(risk, "✅")
        score_str = f"{score}/100" if isinstance(score, int) else "—"

        row_c1, row_c2, row_c3, row_c4, row_c5, row_c6 = st.columns(
            [0.4, 3.2, 2, 1, 2, 1.4]
        )
        row_c1.markdown(risk_icon)
        row_c2.markdown(f"**{fname}**")
        row_c3.markdown(doc_type)
        row_c4.markdown(f"**{score_str}**")
        row_c4.markdown(_badge(risk), unsafe_allow_html=True)
        row_c5.caption(ts)
        row_c6.download_button(
            "⬇ JSON",
            data=json.dumps(sc["report_json"], indent=2, ensure_ascii=False),
            file_name=f"{Path(fname).stem}_verification.json",
            mime="application/json",
            key=f"rec_json_{sc['id']}",
            use_container_width=True,
        )

        with st.expander(f"🔬 Forensic details — {fname}", expanded=False):
            _render_report_summary(sc["report_json"])
