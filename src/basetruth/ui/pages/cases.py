"""Cases page."""
from __future__ import annotations

from typing import Any, Dict

import streamlit as st

from basetruth.service import BaseTruthService
from basetruth.ui.components import (
    _DB_IMPORTS_OK,
    _DISPOSITION_ICONS,
    _badge,
    db_available,
    list_cases_from_db,
)


def _render_case_card(
    service: BaseTruthService,
    case: Dict[str, Any],
    *,
    show_actions: bool,
    use_db: bool = False,
) -> None:
    case_key = case.get("case_key", "")
    risk = case.get("max_risk_level", "low")
    disposition = case.get("disposition", "open")
    doc_type = case.get("document_type", "").replace("_", " ").title()
    doc_count = case.get("document_count", 0)
    entity_ref = case.get("entity_ref", "")
    entity_name = case.get("entity_name", "")
    risk_icon = {"high": "🚨", "medium": "⚠️", "low": "✅"}.get(risk, "🔷")
    disp_icon = _DISPOSITION_ICONS.get(disposition, "")

    name_part = f"  —  {entity_name}" if entity_name else ""
    ref_part = f"  ({entity_ref})" if entity_ref and entity_ref != "unlinked" else ""
    header = (
        f"{risk_icon} {doc_type}{name_part}{ref_part}"
        f"  |  {doc_count} doc(s)  |  {disp_icon} {disposition.replace('_', ' ').title()}"
    )

    with st.expander(header, expanded=show_actions and risk == "high"):
        if show_actions:
            btn_c1, btn_c2, _ = st.columns([1, 1, 3])
            if btn_c1.button(
                "✅  Approve",
                key=f"approve_{case_key}",
                use_container_width=True,
                type="primary",
            ):
                service.update_case(
                    case_key,
                    status="closed",
                    disposition="cleared",
                    note_text="Manually approved by analyst.",
                    note_author="analyst",
                )
                st.toast("✅ Case approved.", icon="✅")
                st.rerun()
            if btn_c2.button(
                "❌  Reject",
                key=f"reject_{case_key}",
                use_container_width=True,
            ):
                service.update_case(
                    case_key,
                    status="closed",
                    disposition="fraud_confirmed",
                    note_text="Rejected by analyst — fraud confirmed.",
                    note_author="analyst",
                )
                st.toast("❌ Case rejected.", icon="❌")
                st.rerun()
            st.divider()
        else:
            verdict_color = (
                "#16a34a"
                if disposition == "cleared"
                else "#dc2626"
                if disposition == "fraud_confirmed"
                else "#6366f1"
            )
            verdict_label = {
                "cleared": "Approved ✅",
                "fraud_confirmed": "Rejected ❌",
            }.get(disposition, disposition.replace("_", " ").title())
            st.markdown(
                f'<div style="font-size:1rem;font-weight:700;color:{verdict_color};'
                f'margin-bottom:8px;">{verdict_label}</div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            f"**Risk:** {_badge(risk)}  &nbsp;&nbsp;  "
            f"**Priority:** {case.get('priority', 'normal').title()}  &nbsp;&nbsp;  "
            f"**Assignee:** {case.get('assignee') or '—'}",
            unsafe_allow_html=True,
        )

        docs = case.get("documents", [])
        if docs:
            st.markdown("**Documents:**")
            for doc in docs:
                src = doc.get("source_name", "unknown")
                dlvl = str(doc.get("risk_level", "low"))
                dscore = doc.get("truth_score", "")
                st.markdown(
                    f"&nbsp;&nbsp;{_badge(dlvl)} {src}  —  "
                    f"Score: **{dscore if isinstance(dscore, int) else '—'}**",
                    unsafe_allow_html=True,
                )

        adv_key = f"_adv_open_{case_key}"
        if not st.session_state.get(adv_key):
            if st.button(
                "⚙️ Advanced options",
                key=f"adv_btn_{case_key}",
                use_container_width=False,
            ):
                st.session_state[adv_key] = True
                st.rerun()
        else:
            if st.button(
                "▲ Hide advanced",
                key=f"adv_hide_{case_key}",
                use_container_width=False,
            ):
                st.session_state[adv_key] = False
                st.rerun()

            if use_db:
                workflow = {
                    "status": case.get("status", "new"),
                    "disposition": case.get("disposition", "open"),
                    "priority": case.get("priority", "normal"),
                    "assignee": case.get("assignee", ""),
                    "labels": case.get("labels", []),
                    "notes": case.get("notes", []),
                }
            else:
                try:
                    case_detail = service.get_case_detail(case_key)
                    workflow = case_detail["workflow"]
                except KeyError:
                    st.warning("Case detail not found.")
                    return

            statuses = [
                "new", "triage", "investigating", "pending_client", "closed"
            ]
            dispositions = [
                "open", "monitor", "escalate", "cleared", "fraud_confirmed"
            ]
            priorities = ["low", "normal", "high", "critical"]
            with st.form(f"adv_form_{case_key}"):
                wf1, wf2, wf3 = st.columns(3)
                cur_s = str(workflow.get("status", "new"))
                cur_d = str(workflow.get("disposition", "open"))
                cur_p = str(workflow.get("priority", "normal"))
                status_sel = wf1.selectbox(
                    "Status",
                    statuses,
                    index=statuses.index(cur_s) if cur_s in statuses else 0,
                    key=f"s_{case_key}",
                )
                disp_sel = wf2.selectbox(
                    "Disposition",
                    dispositions,
                    index=dispositions.index(cur_d) if cur_d in dispositions else 0,
                    key=f"d_{case_key}",
                )
                prio_sel = wf3.selectbox(
                    "Priority",
                    priorities,
                    index=priorities.index(cur_p) if cur_p in priorities else 1,
                    key=f"p_{case_key}",
                )
                assignee_val = st.text_input(
                    "Assignee",
                    value=str(workflow.get("assignee", "")),
                    key=f"a_{case_key}",
                )
                labels_val = st.text_input(
                    "Labels (comma-separated)",
                    value=", ".join(workflow.get("labels", [])),
                    key=f"l_{case_key}",
                )
                note_author = st.text_input(
                    "Note author", value="analyst", key=f"na_{case_key}"
                )
                note_text = st.text_area(
                    "Add a note",
                    placeholder="Observations, evidence, next steps…",
                    key=f"nt_{case_key}",
                )
                if st.form_submit_button("Save", type="primary"):
                    service.update_case(
                        case_key,
                        status=status_sel,
                        disposition=disp_sel,
                        priority=prio_sel,
                        assignee=assignee_val,
                        labels=[
                            i.strip()
                            for i in labels_val.split(",")
                            if i.strip()
                        ],
                        note_text=note_text,
                        note_author=note_author,
                    )
                    st.success("Updated.")
                    st.rerun()

            notes = workflow.get("notes", [])
            if notes:
                st.markdown(f"**Notes ({len(notes)}):**")
                for note in reversed(notes):
                    ts = str(note.get("created_at", ""))[:19].replace("T", " ")
                    author = note.get("author", "")
                    st.markdown(
                        f'<div style="background:var(--bt-note-bg);border-left:3px solid '
                        f'var(--bt-note-accent);padding:8px 12px;border-radius:0 8px 8px 0;'
                        f'margin-bottom:8px;">'
                        f'<span style="font-size:11px;color:var(--bt-text-muted);">'
                        f"{ts} · {author}</span><br>"
                        f'{note.get("text", "")}</div>',
                        unsafe_allow_html=True,
                    )


def _page_cases(service: BaseTruthService) -> None:
    st.markdown("# 📁 Cases")

    with st.expander("ℹ️ How to use this screen", expanded=False):
        st.markdown(
            """
A **case** is created automatically whenever a document is scanned.

- **Needs Review** — your action queue for high / medium risk documents.
  Press **✅ Approve** or **❌ Reject** directly on the card.
- **Resolved** — cases you have already decided on (Approved or Rejected).
- **Auto-Approved** — low-risk documents cleared automatically; no action needed.

When PostgreSQL is connected, cases are read from the database (accurate, reset-safe).
Falling back to local files when the database is offline.
"""
        )

    use_db = _DB_IMPORTS_OK and db_available()
    if use_db:
        cases = list(list_cases_from_db())
    else:
        cases = service.list_cases()

    if not cases:
        st.info(
            "No cases yet. Scan documents first and cases will appear here automatically."
        )
        return

    cases_filter = st.text_input(
        "🔍 Filter cases",
        placeholder="Entity name, BT-reference, case key, or document type…",
        key="cases_filter",
    ).strip().lower()
    if cases_filter:
        cases = [
            c for c in cases
            if cases_filter in (c.get("entity_name") or "").lower()
            or cases_filter in (c.get("entity_ref") or "").lower()
            or cases_filter in (c.get("document_type") or "").lower()
            or cases_filter in (c.get("case_key") or "").lower()
        ]
        if not cases:
            st.info("No cases match your filter.")
            return

    needs_review = [c for c in cases if c.get("needs_review")]
    resolved = [
        c for c in cases
        if c.get("disposition") in ("cleared", "fraud_confirmed")
    ]
    auto_ok = [
        c for c in cases
        if not c.get("needs_review")
        and c.get("disposition") not in ("cleared", "fraud_confirmed")
    ]

    tab_labels = [
        f"⛔ Needs Review ({len(needs_review)})",
        f"✅ Resolved ({len(resolved)})",
    ]
    if auto_ok:
        tab_labels.append(f"🔵 Auto-Approved ({len(auto_ok)})")

    tabs = st.tabs(tab_labels)

    def _render_grouped(case_list: list, show_actions: bool) -> None:
        from collections import defaultdict  # noqa: PLC0415
        by_entity: dict = defaultdict(list)
        for c in case_list:
            by_entity[c.get("entity_ref") or "unlinked"].append(c)
        for ref in sorted(by_entity.keys()):
            entity_cases = by_entity[ref]
            name = entity_cases[0].get("entity_name", "") or ref
            header = (
                f"👤 **{name}** &nbsp; `{ref}` &nbsp;—&nbsp; "
                f"{len(entity_cases)} case(s)"
            )
            with st.expander(header, expanded=show_actions):
                for case in entity_cases:
                    _render_case_card(
                        service, case, show_actions=show_actions, use_db=use_db
                    )

    with tabs[0]:
        if not needs_review:
            st.success(
                "🎉 No cases pending review — all documents have been assessed."
            )
        else:
            _render_grouped(needs_review, show_actions=True)

    with tabs[1]:
        if not resolved:
            st.info("No resolved cases yet.")
        else:
            _render_grouped(resolved, show_actions=False)

    if auto_ok:
        with tabs[2]:
            _render_grouped(auto_ok, show_actions=False)
