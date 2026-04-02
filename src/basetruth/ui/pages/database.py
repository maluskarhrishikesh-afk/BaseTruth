"""Database Viewer page — PostgreSQL tables, MinIO storage, danger zone."""
from __future__ import annotations

import streamlit as st

from basetruth.ui.components import (
    _DB_IMPORTS_OK,
    _page_title,
    db_available,
    db_table_counts,
    db_table_rows,
    minio_available,
    minio_bucket_stats,
    minio_list_objects,
    minio_truncate_bucket,
    reset_db,
)

_DB_TABLE_LABELS: dict[str, str] = {
    "entities": "Entities",
    "scans": "Scans",
    "document_information": "Document Extractions",
    "cases": "Cases",
    "case_notes": "Case Notes",
}


def _page_database() -> None:
    st.markdown(_page_title("💾", "Database Viewer"), unsafe_allow_html=True)

    with st.expander("ℹ️ How to use this screen", expanded=False):
        st.markdown(
            """
This screen gives you direct visibility into what is stored in the system.

- **PostgreSQL tables** — browse entities, scans, cases, and notes row-by-row.
- **MinIO object storage** — list PDF reports and source documents stored in the
  S3-compatible bucket. Files are automatically uploaded here after each scan,
  organised by applicant reference (e.g. `BT-000001/payslip_report.pdf`).
- **Danger Zone** — reset (empty) both stores; useful during testing.
  Type `RESET` to confirm before anything is deleted.
"""
        )

    pg_tab, minio_tab, danger_tab = st.tabs(
        ["🐘  PostgreSQL", "🪣  MinIO Storage", "⚠️  Danger Zone"]
    )

    # ── PostgreSQL tab ───────────────────────────────────────────────────────
    with pg_tab:
        if not _DB_IMPORTS_OK or not db_available():
            st.warning(
                "PostgreSQL is not available.  Start the `db` Docker service and ensure "
                "`DATABASE_URL` is set correctly."
            )
        else:
            counts = db_table_counts()
            cc = st.columns(len(_DB_TABLE_LABELS))
            for i, (tbl, lbl) in enumerate(_DB_TABLE_LABELS.items()):
                cc[i].metric(lbl, f"{counts.get(tbl, 0):,}")

            st.divider()

            sel_col, _ = st.columns([2, 6])
            with sel_col:
                chosen_table = st.selectbox(
                    "Browse table",
                    list(_DB_TABLE_LABELS.keys()),
                    format_func=lambda t: _DB_TABLE_LABELS.get(t) or t,
                    key="db_viewer_table",
                )

            rows, total = db_table_rows(chosen_table, limit=500)
            cap = f"**{_DB_TABLE_LABELS[chosen_table]}** — {total:,} rows total"
            if total > 500:
                cap += "  ·  showing most-recent 500"
            st.subheader(cap)

            if rows:
                import pandas as pd  # noqa: PLC0415

                df = pd.DataFrame(rows)
                for col in df.select_dtypes(
                    include=["datetimetz", "datetime64[ns, UTC]", "object"]
                ).columns:
                    try:
                        df[col] = df[col].astype(str)
                    except Exception:  # noqa: BLE001
                        pass
                st.dataframe(df, hide_index=True, use_container_width=True, height=480)
            else:
                st.info(f"No rows in **{_DB_TABLE_LABELS[chosen_table]}** yet.")

    # ── MinIO tab ────────────────────────────────────────────────────────────
    with minio_tab:
        if not _DB_IMPORTS_OK:
            st.warning("Store module not loaded.")
        else:
            minio_up = minio_available()
            if not minio_up:
                st.warning(
                    "MinIO is not reachable. Check that the `minio` Docker service is running "
                    "and that `MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY` are set."
                )
            else:
                stats = minio_bucket_stats()
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Bucket", stats.get("bucket", "—"))
                mc2.metric("Objects", f"{stats.get('object_count', 0):,}")
                mc3.metric("Total size", f"{stats.get('total_mb', 0):.1f} MB")

                st.divider()
                objs = minio_list_objects(limit=500)
                if objs:
                    import pandas as pd  # noqa: PLC0415

                    st.subheader(f"{len(objs)} objects (most-recent first)")
                    obj_df = pd.DataFrame(
                        [
                            {
                                "Key": o["key"],
                                "Size (KB)": o["size_kb"],
                                "Last Modified": o["last_modified"][:19].replace("T", " "),
                            }
                            for o in objs
                        ]
                    )
                    st.dataframe(obj_df, hide_index=True, use_container_width=True, height=400)
                else:
                    st.info("The bucket is empty.")

    # ── Danger Zone tab ──────────────────────────────────────────────────────
    with danger_tab:
        st.markdown("### ⚠️ Irreversible Operations")
        st.error(
            "Actions below **permanently delete data** with no undo. "
            "Type the exact confirmation word shown before pressing the button."
        )

        dc1, dc2 = st.columns(2)

        with dc1:
            st.markdown("#### 🗄️ Reset PostgreSQL")
            st.caption("Deletes all entities, scans, cases, and notes.")

            if "db_reset_success" in st.session_state:
                st.success("✅ Database reset — all tables cleared.")
                del st.session_state["db_reset_success"]

            db_confirm = st.text_input(
                "Type RESET to confirm",
                key="db_reset_confirm_input",
                placeholder="RESET",
            )
            if st.button("💀 Empty Database", type="primary", key="db_reset_execute_btn"):
                if db_confirm.strip() == "RESET":
                    ok = reset_db()
                    if ok:
                        st.session_state["db_reset_success"] = True
                        st.rerun()
                    else:
                        st.error("Reset failed — check the Logs page for details.")
                else:
                    st.error("Type exactly `RESET` (all caps) to confirm.")

        with dc2:
            st.markdown("#### 🪣 Reset MinIO Bucket")
            st.caption("Deletes all PDF/image objects in the storage bucket.")

            if "minio_reset_success" in st.session_state:
                st.success("✅ MinIO bucket cleared — all objects deleted.")
                del st.session_state["minio_reset_success"]

            minio_confirm = st.text_input(
                "Type RESET to confirm",
                key="minio_truncate_confirm",
                placeholder="RESET",
            )
            if st.button("🗑️ Empty MinIO Bucket", type="primary", key="minio_truncate_btn"):
                if minio_confirm.strip() == "RESET":
                    ok = minio_truncate_bucket()
                    if ok:
                        st.session_state["minio_reset_success"] = True
                        st.rerun()
                    else:
                        st.error(
                            "Reset failed — MinIO may be offline or misconfigured. "
                            "Check the Logs page."
                        )
                else:
                    st.error("Type exactly `RESET` (all caps) to confirm.")
