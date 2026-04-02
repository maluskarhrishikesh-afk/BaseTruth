"""Reports page."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import streamlit as st

from basetruth.service import BaseTruthService
from basetruth.ui.components import (
    _DB_IMPORTS_OK,
    _display_truth_score,
    _page_title,
    db_available,
    get_all_entities_with_scans,
    get_scan_pdf,
    minio_get_object,
    minio_list_objects,
)


def _page_reports(service: BaseTruthService) -> None:
    st.markdown(_page_title("📈", "Reports"), unsafe_allow_html=True)

    with st.expander("ℹ️ How to use this screen", expanded=False):
        st.markdown(
            """
Reports are grouped **by applicant** — one section per person, listing every document
scanned for them with individual download buttons.

- Each applicant card shows their name / reference, a summary table of all their documents,
  and a **Download PDF Report** button for each scan.
- To view raw JSON or run a new scan, use the **Scan** page.
- To search across all applicants, use the **Records** page.
"""
        )

    if _DB_IMPORTS_OK and db_available():
        entities = get_all_entities_with_scans()
        if not entities:
            st.info(
                "No scans in the database yet. "
                "Use **Scan** or **Bulk Scan** to process documents."
            )
            return

        _case_report_tab, _scan_report_tab = st.tabs(
            ["📄 Case Reports", "📋 Individual Scan Reports"]
        )

        with _case_report_tab:
            st.caption(
                "Case reports are generated automatically when you run a **Bulk Scan**. "
                "Each report covers all documents in the batch, income reconciliation, "
                "and an overall verdict."
            )
            _case_reports_found: list = []

            try:
                _all_minio_objs = minio_list_objects(limit=500)
                for obj in _all_minio_objs:
                    if "case_reports/" in obj["key"] and obj["key"].endswith(".pdf"):
                        try:
                            _pdf_data = minio_get_object(obj["key"])
                            if _pdf_data:
                                _parts = obj["key"].split("/")
                                _entity_ref = _parts[0] if len(_parts) > 2 else "—"
                                _fname = _parts[-1]
                                _size_kb = round(obj.get("size_bytes", 0) / 1024, 1)
                                _modified = (
                                    obj.get("last_modified", "")[:19].replace("T", " ")
                                )
                                _case_reports_found.append((
                                    _entity_ref,
                                    f"📄 **{_fname}**  ·  {_size_kb} KB  ·  {_modified}",
                                    _pdf_data,
                                    _fname,
                                    f"cr_minio_{_fname}",
                                ))
                        except Exception:  # noqa: BLE001
                            pass
            except Exception:  # noqa: BLE001
                pass

            if not _case_reports_found:
                _cr_dir = service.artifact_root / "case_reports"
                if _cr_dir.exists():
                    import datetime as _dt  # noqa: PLC0415

                    for _pdf_path in sorted(_cr_dir.glob("*.pdf"), reverse=True):
                        _size_kb = round(_pdf_path.stat().st_size / 1024, 1)
                        _ts_str = _dt.datetime.fromtimestamp(
                            _pdf_path.stat().st_mtime
                        ).strftime("%Y-%m-%d %H:%M")
                        _case_reports_found.append((
                            "System Reports (Disk)",
                            f"📄 **{_pdf_path.name}**  ·  {_size_kb} KB  ·  {_ts_str}",
                            _pdf_path.read_bytes(),
                            _pdf_path.name,
                            f"cr_disk_{_pdf_path.stem}",
                        ))

            if _case_reports_found:
                search_cr = st.text_input(
                    "🔍 Filter case reports",
                    placeholder="Search by BT-reference or filename...",
                    key="cr_search",
                ).strip().lower()
                filtered_cr = [
                    rpt for rpt in _case_reports_found
                    if not search_cr
                    or search_cr in rpt[0].lower()
                    or search_cr in rpt[1].lower()
                    or search_cr in rpt[3].lower()
                ]
                if filtered_cr:
                    from collections import defaultdict  # noqa: PLC0415
                    grouped: dict = defaultdict(list)
                    for _eref, _label, _data, _fname, _key in filtered_cr:
                        grouped[_eref].append((_label, _data, _fname, _key))
                    for _eref, items in grouped.items():
                        _ent_name = ""
                        if _eref != "System Reports (Disk)":
                            _ent = next(
                                (e for e in entities if e.get("entity_ref") == _eref),
                                None,
                            )
                            if _ent:
                                _ent_name = (
                                    f" — {_ent.get('first_name','')} "
                                    f"{_ent.get('last_name','')}".strip()
                                )
                        with st.expander(
                            f"👤 {_eref}{_ent_name}  ({len(items)} docs)",
                            expanded=True,
                        ):
                            for _label, _data, _fname, _key in items:
                                c1, c2 = st.columns([5, 1])
                                c1.markdown(_label, unsafe_allow_html=True)
                                c2.download_button(
                                    "⬇ Download",
                                    data=_data,
                                    file_name=_fname,
                                    mime="application/pdf",
                                    key=_key,
                                    use_container_width=True,
                                )
                else:
                    st.info("No matching case reports found.")
            else:
                st.info(
                    "No case reports yet. Run a **Bulk Scan** to generate one."
                )

        with _scan_report_tab:
            search_rpt = st.text_input(
                "🔍 Filter applicants",
                placeholder="Name, PAN, email, or BT-reference…",
                key="rpt_search",
            ).strip().lower()
            if search_rpt:
                filtered = [
                    e for e in entities
                    if search_rpt in (e.get("name") or "").lower()
                    or search_rpt in (e.get("pan_number") or "").lower()
                    or search_rpt in (e.get("email") or "").lower()
                    or search_rpt in (e.get("entity_ref") or "").lower()
                ]
            else:
                filtered = entities

            st.caption(
                f"{len(filtered)} applicant(s) shown — "
                f"{sum(len(e['scans']) for e in filtered)} document(s)"
            )

            import pandas as pd  # noqa: PLC0415

            for ent in filtered:
                ref = ent["entity_ref"]
                name = ent["name"] or ref
                scans = ent["scans"]
                pan = ent["pan_number"]
                email = ent["email"]
                sub = "  ·  ".join(filter(None, [pan, email]))
                hdr = f"👤 **{name}** — {ref}" + (f"  ·  {sub}" if sub else "")
                with st.expander(
                    hdr + f"  ({len(scans)} doc{'s' if len(scans) != 1 else ''})",
                    expanded=False,
                ):
                    if not scans:
                        st.info("No scans linked to this entity.")
                        continue

                    rows = [
                        {
                            "Document": s["source_name"],
                            "Type": s["document_type"].replace("_", " ").title(),
                            "Risk": s["risk_level"].title(),
                            "Score": _display_truth_score(s["truth_score"]),
                            "Scanned": (
                                s["generated_at"][:19].replace("T", " ")
                                if s["generated_at"] else "—"
                            ),
                            "PDF": "✅" if s["has_pdf"] else "—",
                        }
                        for s in scans
                    ]
                    st.dataframe(
                        pd.DataFrame(rows),
                        hide_index=True,
                        use_container_width=True,
                    )

                    st.markdown("**Download individual scan reports:**")
                    pdf_buttons: list = []
                    for s in scans:
                        stem = Path(s["source_name"]).stem
                        pdf_data: bytes | None = None
                        if s["has_pdf"]:
                            pdf_data = get_scan_pdf(s["id"])
                        if not pdf_data:
                            try:
                                pdf_data = minio_get_object(
                                    f"{ref}/{stem}_report.pdf"
                                )
                            except Exception:  # noqa: BLE001
                                pass
                        if pdf_data:
                            pdf_buttons.append((
                                f"⬇ {stem[:20]}",
                                pdf_data,
                                f"{ref}_{stem}_report.pdf",
                                f"rpt_pdf_{s['id']}",
                            ))
                    if pdf_buttons:
                        btn_cols = st.columns(min(len(pdf_buttons), 3))
                        for idx, (label, data, fname, key) in enumerate(
                            pdf_buttons
                        ):
                            btn_cols[idx % 3].download_button(
                                label,
                                data=data,
                                file_name=fname,
                                mime="application/pdf",
                                key=key,
                                use_container_width=True,
                            )
                    else:
                        st.caption(
                            "No PDF reports available for this applicant. "
                            "Re-scan to generate them."
                        )
    else:
        st.info(
            "📴 Database offline — showing file-based reports. "
            "Connect PostgreSQL for entity-grouped view."
        )
        artifact_root = service.artifact_root
        bundle_dir = artifact_root / "case_reports"
        bundle_pdfs = (
            sorted(bundle_dir.glob("*.pdf"), reverse=True)
            if bundle_dir.exists()
            else []
        )
        if bundle_pdfs:
            st.subheader("📄 Case Bundle PDFs")
            import datetime as _dt  # noqa: PLC0415

            for pdf_path in bundle_pdfs:
                size_kb = round(pdf_path.stat().st_size / 1024, 1)
                ts_str = _dt.datetime.fromtimestamp(
                    pdf_path.stat().st_mtime
                ).strftime("%Y-%m-%d %H:%M")
                c1, c2 = st.columns([5, 1])
                c1.markdown(
                    f"**{pdf_path.name}**  ·  "
                    f'<span style="color:#94a3b8;font-size:11px;">'
                    f"{ts_str} · {size_kb} KB</span>",
                    unsafe_allow_html=True,
                )
                c2.download_button(
                    "⬇",
                    data=pdf_path.read_bytes(),
                    file_name=pdf_path.name,
                    mime="application/pdf",
                    key=f"bundle_{pdf_path.stem}",
                    use_container_width=True,
                )
        else:
            st.info("No reports found. Run a Bulk Scan to generate them.")
