"""Database Viewer page — PostgreSQL tables, MinIO storage, danger zone."""
from __future__ import annotations

import json

import streamlit as st

from basetruth.ui.components import (
    _DB_IMPORTS_OK,
    _db_available_cached,
    _minio_available_cached,
    _page_title,
    db_table_counts,
    db_table_rows,
    minio_bucket_stats,
    minio_list_objects,
    minio_truncate_bucket,
    reset_db,
)

_DB_TABLE_LABELS: dict[str, str] = {
    "entities": "Entities",
    "scans": "Scans",
    "document_information": "Document Extractions",
    "identity_checks": "Identity Checks",
    "layered_analysis_entries": "Layered Analysis Entries",
    "cases": "Cases",
    "case_notes": "Case Notes",
}

# Schema reference — (column_name, type, description)
_TABLE_SCHEMA: dict[str, list[tuple[str, str, str]]] = {
    "entities": [
        ("id", "SERIAL", "Primary key"),
        ("entity_ref", "VARCHAR(50)", "System reference e.g. BT-000001"),
        ("name", "VARCHAR(255)", "Applicant full name"),
        ("pan_number", "VARCHAR(20)", "PAN card number"),
        ("aadhaar_uid", "VARCHAR(20)", "Aadhaar UID (masked)"),
        ("email", "VARCHAR(255)", "Contact email"),
        ("phone", "VARCHAR(50)", "Contact phone"),
        ("created_at", "TIMESTAMPTZ", "Record creation timestamp"),
    ],
    "scans": [
        ("id", "SERIAL", "Primary key"),
        ("entity_id", "FK → entities.id", "Linked entity"),
        ("source_name", "VARCHAR(500)", "Original filename"),
        ("source_sha256", "VARCHAR(64)", "SHA-256 hash of source file"),
        ("document_type", "VARCHAR(100)", "Detected doc type e.g. payslip, bank_statement"),
        ("truth_score", "INTEGER", "0-100 authenticity score (higher = more genuine)"),
        ("risk_level", "VARCHAR(20)", "low / medium / high / review"),
        ("verdict", "VARCHAR(50)", "Pass / Fail / Review verdict"),
        ("parse_method", "VARCHAR(50)", "Extraction method used"),
        ("report_json", "JSONB", "Full structured scan report payload"),
        ("pdf_report", "BYTEA", "PDF binary — excluded from display to save memory"),
        ("generated_at", "TIMESTAMPTZ", "Scan completion timestamp"),
    ],
    "document_information": [
        ("id", "SERIAL", "Primary key"),
        ("entity_id", "FK → entities.id", "Linked entity"),
        ("scan_id", "FK → scans.id", "Linked scan"),
        ("doc_type", "VARCHAR(100)", "Document type"),
        ("extracted_data", "JSONB", "Structured extracted fields"),
        ("created_at", "TIMESTAMPTZ", "Insert timestamp"),
    ],
    "identity_checks": [
        ("id", "SERIAL", "Primary key"),
        ("entity_id", "FK → entities.id", "Linked entity"),
        ("check_type", "VARCHAR(50)", "identity_verification / video_kyc"),
        ("cosine_similarity", "FLOAT", "Face match cosine similarity score"),
        ("threshold", "FLOAT", "Match acceptance threshold"),
        ("liveness_state", "VARCHAR(50)", "Liveness challenge state"),
        ("verdict", "VARCHAR(50)", "VERIFIED / REJECTED / REVIEW"),
        ("details_json", "JSONB", "Full check payload"),
        ("created_at", "TIMESTAMPTZ", "Check timestamp"),
    ],
    "layered_analysis_entries": [
        ("id", "SERIAL", "Primary key"),
        ("entity_id", "FK → entities.id", "Required — links to applicant"),
        ("screen_name", "VARCHAR(100)", "Source screen e.g. Identity Verification, Video KYC, Scan Document, Bulk Scan"),
        ("section_name", "VARCHAR(255)", "Section e.g. Aadhaar, PAN Card, Run Verification, or source filename"),
        ("details_captured_json", "JSONB", "Structured section payload — all extracted fields and check results"),
        ("created_at", "TIMESTAMPTZ", "First insert timestamp"),
        ("updated_at", "TIMESTAMPTZ", "Latest UPSERT timestamp"),
    ],
    "cases": [
        ("id", "SERIAL", "Primary key"),
        ("case_key", "VARCHAR(50)", "Human-readable case ID e.g. CASE-00042"),
        ("entity_id", "FK → entities.id", "Linked entity"),
        ("status", "VARCHAR(30)", "open / in_review / closed"),
        ("disposition", "VARCHAR(30)", "approved / rejected / pending"),
        ("priority", "VARCHAR(20)", "low / medium / high / critical"),
        ("assignee", "VARCHAR(255)", "Assigned reviewer"),
        ("labels", "TEXT[]", "Tag array for filtering"),
        ("created_at", "TIMESTAMPTZ", "Case creation timestamp"),
        ("updated_at", "TIMESTAMPTZ", "Last update timestamp"),
    ],
    "case_notes": [
        ("id", "SERIAL", "Primary key"),
        ("case_id", "FK → cases.id", "Linked case"),
        ("author", "VARCHAR(255)", "Note author"),
        ("text", "TEXT", "Note content"),
        ("created_at", "TIMESTAMPTZ", "Note timestamp"),
    ],
}


@st.cache_data(ttl=60, show_spinner=False)
def _cached_db_table_counts() -> dict:
    return db_table_counts()


@st.cache_data(ttl=60, show_spinner=False)
def _cached_db_table_rows(table: str, limit: int = 500) -> tuple:
    return db_table_rows(table, limit=limit)


@st.cache_data(ttl=60, show_spinner=False)
def _cached_minio_bucket_stats() -> dict:
    return minio_bucket_stats()


@st.cache_data(ttl=60, show_spinner=False)
def _cached_minio_list_objects(limit: int = 500) -> list:
    return minio_list_objects(limit=limit)


def _page_database() -> None:
    st.markdown(_page_title("🗄️", "Database Viewer"), unsafe_allow_html=True)
    if st.button("🔄 Refresh", key="db_viewer_refresh"):
        _cached_db_table_counts.clear()
        _cached_db_table_rows.clear()
        _cached_minio_bucket_stats.clear()
        _cached_minio_list_objects.clear()
        st.rerun()
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
        if not _DB_IMPORTS_OK or not _db_available_cached():
            st.warning(
                "PostgreSQL is not available.  Start the `db` Docker service and ensure "
                "`DATABASE_URL` is set correctly."
            )
        else:
            counts = _cached_db_table_counts()
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

            limit = st.select_slider(
                "Rows to load",
                options=[50, 100, 250, 500],
                value=250,
                key="db_viewer_limit",
            )

            rows, total = _cached_db_table_rows(chosen_table, limit=limit)
            cap = f"**{_DB_TABLE_LABELS[chosen_table]}** — {total:,} rows total"
            if total > limit:
                cap += f"  ·  showing most-recent {limit}"
            st.subheader(cap)

            # ── Schema reference ─────────────────────────────────────────
            schema = _TABLE_SCHEMA.get(chosen_table, [])
            if schema:
                with st.expander("📋 Table Schema", expanded=False):
                    import pandas as pd  # noqa: PLC0415
                    schema_df = pd.DataFrame(
                        schema, columns=["Column", "Type", "Description"]
                    )
                    st.dataframe(
                        schema_df,
                        hide_index=True,
                        use_container_width=True,
                        column_config={
                            "Column": st.column_config.TextColumn("Column", width="small"),
                            "Type": st.column_config.TextColumn("Type", width="medium"),
                            "Description": st.column_config.TextColumn("Description", width="large"),
                        },
                    )

            if rows:
                import pandas as pd  # noqa: PLC0415

                def _display_value(value: object) -> object:
                    if isinstance(value, bytes):
                        return f"<{len(value)} bytes binary>"
                    if isinstance(value, (dict, list)):
                        try:
                            text = json.dumps(value, ensure_ascii=False)
                            return text[:300] + "…" if len(text) > 300 else text
                        except Exception:  # noqa: BLE001
                            return str(value)
                    return value

                df = pd.DataFrame([
                    {column: _display_value(value) for column, value in row.items()}
                    for row in rows
                ])

                # Build column_config: use TextColumn for long-text fields
                jsonb_cols = {
                    col for col in df.columns
                    if col.endswith("_json") or col.endswith("_data") or col in ("labels",)
                }
                col_cfg: dict = {}
                for col in df.columns:
                    if col in jsonb_cols:
                        col_cfg[col] = st.column_config.TextColumn(col, width="large")
                    elif col in ("id", "entity_id", "scan_id", "case_id"):
                        col_cfg[col] = st.column_config.NumberColumn(col, width="small")
                    elif "score" in col or "similarity" in col or "threshold" in col:
                        col_cfg[col] = st.column_config.NumberColumn(col, width="small", format="%.4f")
                    elif col.endswith("_at") or col in ("created_at", "updated_at", "generated_at"):
                        col_cfg[col] = st.column_config.TextColumn(col, width="medium")

                st.dataframe(
                    df,
                    hide_index=True,
                    use_container_width=True,
                    height=480,
                    column_config=col_cfg,
                )

                st.caption(f"Columns ({len(df.columns)}): " + " · ".join(df.columns.tolist()))

                # ── Full row JSON inspector ──────────────────────────────
                st.markdown("**🔍 Inspect Full Row**")
                row_options = {
                    f"Row {index + 1}  ·  {next(iter(row.values()), '')}": index
                    for index, row in enumerate(rows)
                }
                selected_label = st.selectbox(
                    "Select row to inspect",
                    list(row_options.keys()),
                    key=f"db_viewer_row_{chosen_table}",
                )
                selected_row = rows[row_options[selected_label]]
                # Full payload — parse JSONB string back to dict for pretty display
                display_row: dict = {}
                for column, value in selected_row.items():
                    if isinstance(value, bytes):
                        display_row[column] = f"<{len(value)} bytes binary — excluded>"
                    elif isinstance(value, str) and len(value) > 2 and value[0] in "{[":
                        try:
                            display_row[column] = json.loads(value)
                        except Exception:  # noqa: BLE001
                            display_row[column] = value
                    else:
                        display_row[column] = value
                st.json(display_row, expanded=False)
            else:
                st.info(f"No rows in **{_DB_TABLE_LABELS[chosen_table]}** yet.")

    # ── MinIO tab ────────────────────────────────────────────────────────────
    with minio_tab:
        if not _DB_IMPORTS_OK:
            st.warning("Store module not loaded.")
        else:
            minio_up = _minio_available_cached()
            if not minio_up:
                st.warning(
                    "MinIO is not reachable. Check that the `minio` Docker service is running "
                    "and that `MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY` are set."
                )
            else:
                stats = _cached_minio_bucket_stats()
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Bucket", stats.get("bucket", "—"))
                mc2.metric("Objects", f"{stats.get('object_count', 0):,}")
                mc3.metric("Total size", f"{stats.get('total_mb', 0):.1f} MB")

                st.divider()
                objs = _cached_minio_list_objects(limit=500)
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
                    with st.spinner("Truncating all tables…"):
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
                    with st.spinner("Deleting all objects from the bucket…"):
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
