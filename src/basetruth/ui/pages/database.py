"""Database Viewer page — PostgreSQL tables, MinIO storage, danger zone."""
from __future__ import annotations

import json
import os

import streamlit as st

from basetruth.ui.components import (
    _DB_IMPORTS_OK,
    _db_available_cached,
    _minio_available_cached,
    _page_title,
    db_table_counts,
    db_table_rows,
    minio_bucket_stats,
    minio_delete_object,
    minio_docs_bucket_stats,
    minio_docs_delete,
    minio_docs_get,
    minio_docs_put,
    minio_get_object,
    minio_list_docs_objects,
    minio_list_objects,
    minio_truncate_bucket,
    minio_upload,
    reset_db,
    truncate_table,
    db_viewer_get_row,
    db_viewer_fk_options,
    db_viewer_create_row,
    db_viewer_update_row,
    db_viewer_delete_row,
)
from basetruth.logger import get_logger

log = get_logger(__name__)

# Table metadata that drives form generation — imported from store so there
# is exactly one source of truth.  Falls back to empty dict when the DB
# module is unavailable (prevents NameError in the render path).
try:
    from basetruth.store import _DB_VIEWER_TABLE_META  # type: ignore[import]
except Exception:  # noqa: BLE001
    _DB_VIEWER_TABLE_META: dict = {}

_DB_TABLE_LABELS: dict[str, str] = {
    "entities": "Entities",
    "scans": "Scans",
    "document_extractions": "Document Extractions",
    "identity_checks": "Identity Checks",
    "entity_reports": "Entity Reports",
    "video_kyc_checks": "Video KYC Checks",
    "face_scan_live_results": "Face Scan Live Results",
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
        ("entity_id", "FK → entities.id", "Linked entity — nullable"),
        ("source_name", "VARCHAR(500)", "Original filename"),
        ("source_sha256", "VARCHAR(64)", "SHA-256 hash of source file"),
        ("document_type", "VARCHAR(100)", "Document type e.g. payslip, pan_card, aadhaar"),
        ("layered_analysis_json", "JSONB", "Full 11-layer forensic analysis result (ELA, noise, clone, etc.)"),
        ("first_level_approval", "VARCHAR(1)", "Y = approved, N = rejected, NULL = pending (initial reviewer)"),
        ("first_level_approved_by", "VARCHAR(255)", "Initial reviewer name or ID"),
        ("first_level_approved_at", "TIMESTAMPTZ", "Timestamp of initial reviewer decision"),
        ("first_level_approval_comment", "TEXT", "Optional comment from initial reviewer"),
        ("second_level_approval", "VARCHAR(1)", "Y = approved, N = rejected, NULL = pending (senior reviewer)"),
        ("second_level_approved_by", "VARCHAR(255)", "Senior reviewer name or ID"),
        ("second_level_approved_at", "TIMESTAMPTZ", "Timestamp of senior reviewer decision"),
        ("second_level_approval_comment", "TEXT", "Optional comment from senior reviewer"),
        ("generated_at", "TIMESTAMPTZ", "Scan completion timestamp"),
        ("updated_at", "TIMESTAMPTZ", "Last update timestamp"),
    ],
    "document_extractions": [
        ("id", "SERIAL", "Primary key"),
        ("entity_id", "FK → entities.id", "Linked entity"),
        ("scan_id", "FK → scans.id", "Linked scan (NULL for identity-verification extractions)"),
        ("file_name", "VARCHAR(500)", "Uploaded filename used as the per-entity UPSERT key"),
        ("document_type", "VARCHAR(100)", "Document type e.g. payslip, marksheet, pan_card, aadhaar"),
        ("extracted_data", "JSONB", "Structured extracted fields from the document"),
        ("source_screen", "VARCHAR(100)", "Screen that triggered extraction e.g. bulk_scan, identity_verification"),
        ("created_at", "TIMESTAMPTZ", "Insert timestamp"),
    ],
    "identity_checks": [
        ("id", "SERIAL", "Primary key"),
        ("entity_id", "FK → entities.id", "Linked entity — nullable"),
        ("status", "VARCHAR(20)", "pass | fail | inconclusive"),
        ("cosine_similarity", "FLOAT", "ArcFace face-match cosine similarity score"),
        ("display_score", "FLOAT", "0–100 display score derived from cosine_similarity"),
        ("threshold", "FLOAT", "Match acceptance threshold (default 0.40)"),
        ("is_match", "BOOLEAN", "True when cosine_similarity ≥ threshold"),
        ("verdict", "VARCHAR(20)", "PASS | FAIL"),
        ("selfie_pic", "VARCHAR(500)", "MinIO key for selfie image"),
        ("aadhaar_pic", "VARCHAR(500)", "MinIO key for Aadhaar card image"),
        ("pan_pic", "VARCHAR(500)", "MinIO key for PAN card image"),
        ("signature_pic", "VARCHAR(500)", "MinIO key for cropped PAN signature"),
        ("pdf_report", "VARCHAR(500)", "MinIO key for the generated PDF report"),
        ("aadhar_dtls", "JSONB", "Aadhaar QR extracted fields (name, uid, dob, address, etc.)"),
        ("pan_dtls", "JSONB", "PAN card extracted fields (pan_number, full_name, father_name, etc.)"),
        ("report_json", "JSONB", "Full face-match result payload"),
        ("created_at", "TIMESTAMPTZ", "Row creation timestamp"),
        ("updated_at", "TIMESTAMPTZ", "Last update timestamp"),
    ],
    "video_kyc_checks": [
        ("id", "SERIAL", "Primary key"),
        ("entity_id", "FK → entities.id", "Linked entity — nullable"),
        ("status", "VARCHAR(20)", "pass | fail | inconclusive"),
        ("cosine_similarity", "FLOAT", "ArcFace cosine similarity (reference doc vs live frame)"),
        ("display_score", "FLOAT", "0–100 display score"),
        ("threshold", "FLOAT", "Match threshold (default 0.40)"),
        ("is_match", "BOOLEAN", "True when face matched the reference document"),
        ("liveness_state", "VARCHAR(30)", "Current liveness challenge state"),
        ("liveness_passed", "BOOLEAN", "True when all liveness challenges passed"),
        ("verdict", "VARCHAR(20)", "PASS | FAIL"),
        ("aadhar_dtls", "JSONB", "Aadhaar QR decoded payload (name, dob, gender, uid, state, etc.)"),
        ("pan_dtls", "JSONB", "PAN card extracted fields (pan_number, full_name, father_name, dob)"),
        ("video_kyc_pic", "VARCHAR(500)", "MinIO key for best live frame captured during KYC"),
        ("address_proof_pic", "VARCHAR(500)", "MinIO key for address proof document image"),
        ("aadhaar_pic", "VARCHAR(500)", "MinIO key for Aadhaar card image"),
        ("pan_pic", "VARCHAR(500)", "MinIO key for PAN card image"),
        ("signature_pic", "VARCHAR(500)", "MinIO key for PAN signature crop"),
        ("isAddressMatch", "VARCHAR(20)", "match | mismatch | partial | skipped"),
        ("kyc_comments", "VARCHAR(500)", "Free-text comments from KYC officer or system"),
        ("current_location", "TEXT", "Human-readable address from reverse-geocoding GPS"),
        ("address_distance_meters", "FLOAT", "Distance in metres between address-proof and live GPS"),
        ("pdf_report", "VARCHAR(500)", "MinIO key for the generated PDF report"),
        ("address_dtls", "JSONB", "Extracted fields from the address proof document (address, pincode, etc.)"),
        ("challenge_snapshots_json", "JSONB", "Per-challenge results (nod head, look left, etc.) with frame snapshots and EAR values"),
        ("report_json", "JSONB", "Full KYC session result payload"),
        ("created_at", "TIMESTAMPTZ", "Row creation timestamp"),
        ("updated_at", "TIMESTAMPTZ", "Last update timestamp"),
    ],
    "entity_reports": [
        ("id", "SERIAL", "Primary key"),
        ("entity_id", "FK → entities.id (CASCADE)", "Which applicant this report belongs to"),
        ("report_ref", "VARCHAR(20), UNIQUE", "Human-readable reference e.g. BTR-000001"),
        ("report_json", "JSONB", "Full cross-document analysis payload"),
        ("report_minio_key", "VARCHAR(500)", "MinIO object key for the generated PDF"),
        ("first_level_approval", "VARCHAR(1)", "Y = approved, N = rejected, NULL = pending (initial reviewer)"),
        ("first_level_approved_by", "VARCHAR(255)", "Who made the 1st-level decision"),
        ("first_level_approved_at", "TIMESTAMPTZ", "When 1st-level decision was made"),
        ("first_level_approval_comment", "TEXT", "Optional 1st-level reviewer note"),
        ("second_level_approval", "VARCHAR(1)", "Y = approved, N = rejected, NULL = pending (senior reviewer)"),
        ("second_level_approved_by", "VARCHAR(255)", "Who made the 2nd-level decision"),
        ("second_level_approved_at", "TIMESTAMPTZ", "When 2nd-level decision was made"),
        ("second_level_approval_comment", "TEXT", "Optional 2nd-level reviewer note"),
        ("generated_at", "TIMESTAMPTZ", "When the report was first generated"),
        ("updated_at", "TIMESTAMPTZ", "When the report was last updated"),
    ],
    "face_scan_live_results": [
        ("id", "SERIAL", "Primary key"),
        ("session_id", "VARCHAR(50), UNIQUE", "URL-safe token matching the in-memory FaceScanLiveSession"),
        ("verdict", "VARCHAR(20)", "GENUINE | SUSPICIOUS | DEEPFAKE | INCONCLUSIVE | LIVENESS_FAILED"),
        ("risk_score", "FLOAT", "0–100 risk score derived from ML model or heuristic fallback"),
        ("confidence", "FLOAT", "0–1 model confidence value"),
        ("best_frame_key", "VARCHAR(500)", "MinIO object key for the best captured still frame"),
        ("video_key", "VARCHAR(500)", "MinIO object key for the recorded MP4 video (NULL when not recorded)"),
        ("report_json", "JSONB", "Full live face scan result payload for audit and re-display"),
        ("created_at", "TIMESTAMPTZ", "Row creation timestamp"),
        ("updated_at", "TIMESTAMPTZ", "Last update timestamp"),
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


@st.cache_data(ttl=60, show_spinner=False)
def _cached_minio_docs_bucket_stats() -> dict:
    return minio_docs_bucket_stats()


@st.cache_data(ttl=60, show_spinner=False)
def _cached_minio_list_docs_objects(limit: int = 200) -> list:
    return minio_list_docs_objects(limit=limit)


# ---------------------------------------------------------------------------
# CRUD panel helper
# ---------------------------------------------------------------------------

def _crud_form_fields(
    table: str,
    mode: str,
    prefill: dict,
    form_key: str,
) -> "dict | None":
    """Render create/edit form fields inside a Streamlit form and return the
    submitted payload, or None if the form was not submitted.

    Parameters
    ----------
    table     : allowlisted table name
    mode      : "create" | "edit" | "duplicate"
    prefill   : dict of current column values (empty for create)
    form_key  : unique key for the st.form widget
    """
    meta = _DB_VIEWER_TABLE_META.get(table)
    if meta is None:
        st.error(f"No metadata found for table '{table}'.")
        return None

    payload: dict = {}

    with st.form(key=form_key, clear_on_submit=False):
        for col_def in meta["editable"]:
            col_name = col_def["name"]
            label = col_def["label"]
            ui = col_def["ui"]
            nullable = col_def.get("nullable", False)
            # In duplicate mode we clear system-managed fields so the new row
            # gets fresh values instead of carrying over old IDs / timestamps.
            is_system_key = col_name in {"entity_ref", "report_ref"} and mode == "duplicate"
            raw_val = prefill.get(col_name)

            if ui == "text":
                default_str = "" if is_system_key else (_safe_str(raw_val) if raw_val is not None else "")
                payload[col_name] = st.text_input(label, value=default_str, key=f"{form_key}_{col_name}")

            elif ui == "textarea":
                default_str = _safe_str(raw_val) if raw_val is not None else ""
                payload[col_name] = st.text_area(label, value=default_str, height=80, key=f"{form_key}_{col_name}")

            elif ui == "int":
                default_int = int(raw_val) if raw_val not in (None, "") else 0
                payload[col_name] = st.number_input(label, value=default_int, step=1, key=f"{form_key}_{col_name}")

            elif ui == "float":
                # Use a text input so the user can type an empty string for NULL
                hint = " (leave blank for NULL)" if nullable else ""
                default_float = str(raw_val) if raw_val not in (None, "") else ""
                payload[col_name] = st.text_input(f"{label}{hint}", value=default_float, key=f"{form_key}_{col_name}")

            elif ui == "bool":
                hint = " (unchecked = NULL/false)" if nullable else ""
                default_bool = bool(raw_val) if raw_val is not None else False
                payload[col_name] = st.checkbox(f"{label}{hint}", value=default_bool, key=f"{form_key}_{col_name}")

            elif ui == "select":
                choices = col_def.get("choices", [])
                # Find current index — default to first choice if not found
                current = _safe_str(raw_val) if raw_val is not None else ""
                idx = choices.index(current) if current in choices else 0
                payload[col_name] = st.selectbox(label, choices, index=idx, key=f"{form_key}_{col_name}")

            elif ui == "json":
                # Pretty-print existing JSON or start with a minimal empty object
                if isinstance(raw_val, (dict, list)):
                    default_json = json.dumps(raw_val, indent=2, ensure_ascii=False)
                elif isinstance(raw_val, str) and raw_val.strip():
                    try:
                        default_json = json.dumps(json.loads(raw_val), indent=2)
                    except Exception:  # noqa: BLE001
                        default_json = raw_val
                else:
                    default_json = "{}"
                col1, col2 = st.columns([5, 1])
                with col1:
                    entered = st.text_area(label, value=default_json, height=220, key=f"{form_key}_{col_name}")
                with col2:
                    st.markdown("&nbsp;", unsafe_allow_html=True)
                    if st.form_submit_button("Format", help="Pretty-print the JSON text above"):
                        try:
                            entered = json.dumps(json.loads(entered), indent=2)
                        except Exception:  # noqa: BLE001
                            pass
                payload[col_name] = entered

            elif ui == "fk":
                # Live query to build the selectbox options for the parent table
                fk_tbl = col_def["fk_table"]
                options = db_viewer_fk_options(fk_tbl)
                hint = " (optional)" if nullable else ""
                if options:
                    labels = [o["label"] for o in options]
                    ids = [o["id"] for o in options]
                    # Find the index that matches the current prefill value
                    current_id = raw_val
                    try:
                        current_id = int(current_id) if current_id not in (None, "") else None
                    except (TypeError, ValueError):
                        current_id = None
                    default_idx = ids.index(current_id) if current_id in ids else 0
                    if nullable:
                        # Prepend a blank "—" option so user can set FK to NULL
                        labels = ["— (none)"] + labels
                        ids = [None] + ids
                        default_idx = default_idx + 1 if current_id in ids else 0
                    selected_label = st.selectbox(f"{label}{hint}", labels, index=default_idx, key=f"{form_key}_{col_name}")
                    selected_idx = labels.index(selected_label)
                    payload[col_name] = ids[selected_idx]
                else:
                    # Table is empty — fall back to a plain number input
                    default_id = int(raw_val) if raw_val not in (None, "") else 0
                    payload[col_name] = st.number_input(
                        f"{label}{hint} (no rows in {fk_tbl} yet — enter id manually)",
                        value=default_id, step=1, key=f"{form_key}_{col_name}",
                    )

        # Action buttons at the bottom of the form
        save_label = "💾 Save new row" if mode in ("create", "duplicate") else "💾 Save changes"
        col_save, col_cancel = st.columns([2, 1])
        with col_save:
            submitted = st.form_submit_button(save_label, type="primary")
        with col_cancel:
            cancelled = st.form_submit_button("✖ Cancel")

    if cancelled:
        return None  # signal: user bailed out
    if submitted:
        return payload  # signal: user submitted the form
    return ...  # Ellipsis = form shown but no submit yet (don't clear mode)


def _safe_str(val: object) -> str:
    """Convert any value to a display string for text inputs."""
    if isinstance(val, (dict, list)):
        try:
            return json.dumps(val, ensure_ascii=False)
        except Exception:  # noqa: BLE001
            return str(val)
    return str(val) if val is not None else ""


def _render_crud_panel(table: str, rows: list, selected_row: "dict | None") -> None:
    """Render the row-operations panel inside the PostgreSQL tab.

    This function is only called when BASETRUTH_ENABLE_DB_VIEWER_CRUD=true.
    It shows 4 action buttons (Create / Edit / Duplicate / Delete) and renders
    the appropriate form or confirmation widget based on the current mode.

    Session-state keys used:
      db_crud_mode         – "create" | "edit" | "duplicate" | "delete" | None
      db_crud_target_table – table that was active when mode was set (used to
                             reset mode automatically when the user switches table)
    """
    # ------------------------------------------------------------------
    # Reset the mode if the user switched to a different table
    # ------------------------------------------------------------------
    if st.session_state.get("db_crud_target_table") != table:
        st.session_state["db_crud_mode"] = None
        st.session_state["db_crud_target_table"] = table

    current_mode: "str | None" = st.session_state.get("db_crud_mode")
    row_id: "int | None" = int(selected_row["id"]) if selected_row and "id" in selected_row else None

    st.divider()
    st.warning(
        "⚠️ **Development Mode** — Row editing is enabled. "
        "These tools can create inconsistent data if used carelessly.",
        icon="🔧",
    )
    st.markdown("### 🛠️ Row Operations")

    # ------------------------------------------------------------------
    # Show which row is currently targeted (for edit/duplicate/delete)
    # ------------------------------------------------------------------
    if selected_row is not None:
        # Build a compact one-line summary of the selected row
        _id = selected_row.get("id", "?")
        _hint_keys = ["entity_ref", "source_name", "file_name", "report_ref"]
        _hint = next(
            (f"{k}={selected_row[k]!r}" for k in _hint_keys if selected_row.get(k)),
            "",
        )
        st.caption(f"Selected row: **id={_id}** in `{table}`" + (f"  ·  {_hint}" if _hint else ""))
    else:
        st.caption("No row selected — browse and select a row above to edit, duplicate, or delete it.")

    # ------------------------------------------------------------------
    # Action buttons
    # ------------------------------------------------------------------
    btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)
    with btn_col1:
        if st.button("➕ Create", key="db_crud_btn_create", use_container_width=True):
            st.session_state["db_crud_mode"] = "create"
            st.rerun()
    with btn_col2:
        if st.button(
            "✏️ Edit Row",
            key="db_crud_btn_edit",
            disabled=row_id is None,
            use_container_width=True,
        ):
            st.session_state["db_crud_mode"] = "edit"
            st.rerun()
    with btn_col3:
        if st.button(
            "🧬 Duplicate",
            key="db_crud_btn_dup",
            disabled=row_id is None,
            use_container_width=True,
        ):
            st.session_state["db_crud_mode"] = "duplicate"
            st.rerun()
    with btn_col4:
        if st.button(
            "🗑️ Delete Row",
            key="db_crud_btn_delete",
            disabled=row_id is None,
            use_container_width=True,
        ):
            st.session_state["db_crud_mode"] = "delete"
            st.rerun()

    if current_mode is None:
        return  # nothing further to render until user picks an action

    # ------------------------------------------------------------------
    # Create / Edit / Duplicate — render the form
    # ------------------------------------------------------------------
    if current_mode in ("create", "edit", "duplicate"):
        # Decide what to prefill
        if current_mode == "create":
            prefill: dict = {}
            caption = f"Creating a new row in `{table}`"
        elif current_mode == "edit" and row_id is not None:
            # Fetch the full row (the dataframe view may truncate JSONB columns)
            full_row = db_viewer_get_row(table, row_id)
            prefill = full_row or selected_row or {}
            caption = f"Editing row **id={row_id}** in `{table}`"
        else:  # duplicate
            full_row = db_viewer_get_row(table, row_id) if row_id else None
            prefill = dict(full_row or selected_row or {})
            # Clear system-managed fields so the new row gets fresh defaults
            for _sys_col in ("id", "created_at", "updated_at", "generated_at"):
                prefill.pop(_sys_col, None)
            caption = f"Duplicating row **id={row_id}** in `{table}` — save will create a new row"

        st.markdown(f"**{caption}**")
        # Include row_id in the form key so Streamlit creates a fresh form
        # (with new prefill values) whenever the selected row changes.
        # Without this, Streamlit reuses cached widget values from the old row.
        form_key = f"db_crud_form_{current_mode}_{table}_{row_id}"
        result = _crud_form_fields(table, current_mode, prefill, form_key)

        if result is None:
            # User hit Cancel
            log.debug("Database Viewer CRUD: user cancelled %s on %s", current_mode, table)
            st.session_state["db_crud_mode"] = None
            st.rerun()

        elif result is ...:
            pass  # Form shown, waiting for user input — do nothing

        else:
            # User submitted the form — call the appropriate store helper
            if current_mode in ("create", "duplicate"):
                ok, msg, _new_row = db_viewer_create_row(table, result)
            else:
                ok, msg, _new_row = db_viewer_update_row(table, row_id, result)  # type: ignore[arg-type]

            if ok:
                log.info("Database Viewer CRUD [%s]: %s", current_mode, msg)
                st.success(f"✅ {msg}")
                st.session_state["db_crud_mode"] = None
                # Invalidate the row cache so the table refreshes
                _cached_db_table_counts.clear()
                _cached_db_table_rows.clear()
                st.rerun()
            else:
                log.error("Database Viewer CRUD [%s] failed: %s", current_mode, msg)
                st.error(f"Save failed: {msg}")

    # ------------------------------------------------------------------
    # Delete — confirmation flow
    # ------------------------------------------------------------------
    elif current_mode == "delete":
        if row_id is None:
            st.error("No row selected. Select a row from the table above.")
            st.session_state["db_crud_mode"] = None
            return

        _id = selected_row.get("id", "?")   # type: ignore[union-attr]
        st.error(
            f"⚠️ You are about to permanently delete row **id={_id}** from `{table}`. "
            "This cannot be undone. Type **DELETE** below to confirm."
        )
        del_col1, del_col2, del_col3 = st.columns([3, 2, 2])
        with del_col1:
            del_confirm = st.text_input(
                "Type DELETE to confirm",
                key="db_crud_delete_confirm",
                placeholder="DELETE",
                label_visibility="collapsed",
            )
        with del_col2:
            if st.button("💀 Confirm Delete", type="primary", key="db_crud_delete_exec"):
                if del_confirm.strip() == "DELETE":
                    ok, msg = db_viewer_delete_row(table, row_id)
                    if ok:
                        log.info("Database Viewer CRUD [delete]: %s", msg)
                        st.success(f"✅ {msg}")
                        st.session_state["db_crud_mode"] = None
                        _cached_db_table_counts.clear()
                        _cached_db_table_rows.clear()
                        st.rerun()
                    else:
                        log.error("Database Viewer CRUD [delete] failed: %s", msg)
                        st.error(f"Delete failed: {msg}")
                else:
                    st.error("Type exactly DELETE (all caps) to confirm.")
        with del_col3:
            if st.button("✖ Cancel", key="db_crud_delete_cancel"):
                st.session_state["db_crud_mode"] = None
                st.rerun()


def _page_database() -> None:
    st.markdown(_page_title("🗄️", "Database Viewer"), unsafe_allow_html=True)
    if st.button("🔄 Refresh", key="db_viewer_refresh"):
        log.info("Database Viewer: Refreshing cached table counts, rows, and MinIO stats.")
        _cached_db_table_counts.clear()
        _cached_db_table_rows.clear()
        _cached_minio_bucket_stats.clear()
        _cached_minio_list_objects.clear()
        _cached_minio_docs_bucket_stats.clear()
        _cached_minio_list_docs_objects.clear()
        st.rerun()
    with st.expander("ℹ️ How to use this screen", expanded=False):
        st.markdown(
            """
This screen gives you direct visibility into what is stored in the system.

- **PostgreSQL tables** — browse entities, scans, document extractions, identity checks, and final reports row-by-row.
- **MinIO object storage** — list PDF reports and source documents stored in the
    main S3-compatible bucket, plus technical reference files kept in the separate
    `basetruth-docs` bucket (including `DATABASE.md`).
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
                    import datetime as _dt  # noqa: PLC0415
                    if isinstance(value, bytes):
                        return f"<{len(value)} bytes binary>"
                    if isinstance(value, _dt.datetime):
                        # Format as DD/MM/YYYY and HH:MM:SS on separate implicit columns
                        return value.strftime("%d/%m/%Y %H:%M:%S")
                    if isinstance(value, (dict, list)):
                        try:
                            text = json.dumps(value, ensure_ascii=False)
                            return text[:300] + "\u2026" if len(text) > 300 else text
                        except Exception:  # noqa: BLE001
                            return str(value)
                    # ISO-format timestamp strings (from SQL raw queries) → reformat
                    if isinstance(value, str) and len(value) >= 19 and "T" in value:
                        try:
                            dt = _dt.datetime.fromisoformat(value[:19])
                            return dt.strftime("%d/%m/%Y %H:%M:%S")
                        except ValueError:
                            pass
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

                # Session state key that persists the selected row index for this table.
                # Using a per-table key means switching tables resets the selection.
                sel_idx_key = f"db_viewer_selected_idx_{chosen_table}"

                # Render the dataframe with single-row click-to-select enabled.
                # on_select="rerun" means Streamlit reruns the page immediately
                # when the user clicks a row, and the return value carries the
                # selected row indices.
                event = st.dataframe(
                    df,
                    hide_index=True,
                    use_container_width=True,
                    height=480,
                    column_config=col_cfg,
                    on_select="rerun",
                    selection_mode="single-row",
                )

                st.caption(
                    f"Columns ({len(df.columns)}): " + " · ".join(df.columns.tolist())
                    + "  ·  *Click any row to select it.*"
                )

                # If the user just clicked a row, persist that index.
                if event.selection.rows:
                    st.session_state[sel_idx_key] = event.selection.rows[0]

                # Read back the persisted index (it survives CRUD action reruns).
                active_idx: "int | None" = st.session_state.get(sel_idx_key)

                # Discard stale index when the row count changed (e.g. after a delete).
                if active_idx is not None and (active_idx < 0 or active_idx >= len(rows)):
                    active_idx = None
                    st.session_state.pop(sel_idx_key, None)

                # The active row object (or None if nothing selected yet).
                selected_row: "dict | None" = rows[active_idx] if active_idx is not None else None

                # ── Full row JSON inspector ──────────────────────────────
                st.markdown("**🔍 Inspect Full Row**")
                if selected_row is not None:
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
                    st.info("Click any row in the table above to inspect it and enable row operations.")

                # Keep a reference accessible outside this block for the CRUD panel.
                _current_selected_row: "dict | None" = selected_row
            else:
                st.info(f"No rows in **{_DB_TABLE_LABELS[chosen_table]}** yet.")
                _current_selected_row = None

            # ── CRUD Row Operations panel (enabled via env-var) ──────────
            _crud_enabled = os.environ.get(
                "BASETRUTH_ENABLE_DB_VIEWER_CRUD", ""
            ).lower() in ("1", "true", "yes")
            if _crud_enabled:
                _render_crud_panel(chosen_table, rows, _current_selected_row)

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
                # ── Main reports bucket ───────────────────────────────────
                stats = _cached_minio_bucket_stats()
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Bucket", stats.get("bucket", "—"))
                mc2.metric("Objects", f"{stats.get('object_count', 0):,}")
                mc3.metric("Total size", f"{stats.get('total_mb', 0):.1f} MB")

                st.divider()
                objs = _cached_minio_list_objects(limit=500)
                main_keys: list[str] = [o["key"] for o in objs]

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
                    st.dataframe(obj_df, hide_index=True, use_container_width=True, height=300)
                else:
                    st.info("The bucket is empty.")

                # ── Main bucket operations ────────────────────────────────
                with st.expander("⬇️  Download object", expanded=False):
                    if main_keys:
                        dl_key = st.selectbox(
                            "Select object key",
                            options=main_keys,
                            key="main_dl_key_select",
                        )
                        if dl_key:
                            raw = minio_get_object(dl_key)
                            if raw is not None:
                                st.download_button(
                                    label="⬇️  Download",
                                    data=raw,
                                    file_name=dl_key.split("/")[-1],
                                    key="main_dl_btn",
                                )
                            else:
                                st.error("Could not retrieve the object — it may have been deleted.")
                    else:
                        st.info("No objects in the bucket yet.")

                with st.expander("⬆️  Upload object", expanded=False):
                    uploaded_file = st.file_uploader(
                        "Choose a file to upload",
                        key="main_upload_file",
                    )
                    custom_key = st.text_input(
                        "Object key (leave blank to use original filename)",
                        key="main_upload_key_input",
                        placeholder="e.g. reports/my-doc.pdf",
                    )
                    if st.button("⬆️  Upload", key="main_upload_btn"):
                        if uploaded_file is None:
                            st.warning("Please choose a file first.")
                        else:
                            # Use custom key if provided, otherwise use the original filename
                            target_key = custom_key.strip() if custom_key.strip() else uploaded_file.name
                            data = uploaded_file.getvalue()
                            content_type = uploaded_file.type or "application/octet-stream"
                            ok = minio_upload(target_key, data, content_type)
                            if ok:
                                log.info("Database Viewer [MinIO]: Uploaded '%s' (%d bytes)", target_key, len(data))
                                st.success(f"✅ Uploaded `{target_key}` ({len(data):,} bytes).")
                                _cached_minio_bucket_stats.clear()
                                _cached_minio_list_objects.clear()
                                st.rerun()
                            else:
                                st.error("Upload failed — check Logs for details.")

                with st.expander("🗑️  Delete object", expanded=False):
                    if main_keys:
                        del_key = st.selectbox(
                            "Select object to delete",
                            options=main_keys,
                            key="main_del_key_select",
                        )
                        del_confirm = st.text_input(
                            "Type DELETE to confirm",
                            key="main_del_confirm_input",
                            placeholder="DELETE",
                        )
                        if st.button("🗑️  Delete object", type="primary", key="main_del_btn"):
                            if del_confirm.strip() != "DELETE":
                                st.error("Type exactly `DELETE` (all caps) to confirm.")
                            else:
                                ok = minio_delete_object(del_key)
                                if ok:
                                    log.warning("Database Viewer [MinIO]: Deleted object '%s'", del_key)
                                    st.success(f"✅ Deleted `{del_key}`.")
                                    _cached_minio_bucket_stats.clear()
                                    _cached_minio_list_objects.clear()
                                    st.rerun()
                                else:
                                    st.error("Delete failed — MinIO may be offline or the key no longer exists.")
                    else:
                        st.info("No objects in the bucket yet.")

                st.divider()

                # ── Docs bucket ───────────────────────────────────────────
                docs_stats = _cached_minio_docs_bucket_stats()
                dc1, dc2, dc3 = st.columns(3)
                dc1.metric("Docs Bucket", docs_stats.get("bucket", "—"))
                dc2.metric("Docs Objects", f"{docs_stats.get('object_count', 0):,}")
                dc3.metric("Docs Total Size", f"{docs_stats.get('total_mb', 0):.3f} MB")

                docs_objs = _cached_minio_list_docs_objects(limit=200)
                docs_keys: list[str] = [o["key"] for o in docs_objs]

                if docs_objs:
                    import pandas as pd  # noqa: PLC0415

                    st.subheader("Docs bucket objects")
                    docs_df = pd.DataFrame(
                        [
                            {
                                "Key": o["key"],
                                "Size (KB)": o["size_kb"],
                                "Last Modified": o["last_modified"][:19].replace("T", " "),
                            }
                            for o in docs_objs
                        ]
                    )
                    st.dataframe(docs_df, hide_index=True, use_container_width=True, height=220)
                else:
                    st.info("The docs bucket is empty.")

                # ── Docs bucket operations ────────────────────────────────
                with st.expander("⬇️  Download docs object", expanded=False):
                    if docs_keys:
                        docs_dl_key = st.selectbox(
                            "Select docs object key",
                            options=docs_keys,
                            key="docs_dl_key_select",
                        )
                        if docs_dl_key:
                            docs_raw = minio_docs_get(docs_dl_key)
                            if docs_raw is not None:
                                # Show inline preview for markdown / text files
                                if docs_dl_key.lower().endswith((".md", ".txt")):
                                    st.text_area(
                                        "Preview",
                                        value=docs_raw.decode("utf-8", errors="replace"),
                                        height=250,
                                        key="docs_dl_preview",
                                    )
                                st.download_button(
                                    label="⬇️  Download",
                                    data=docs_raw,
                                    file_name=docs_dl_key.split("/")[-1],
                                    key="docs_dl_btn",
                                )
                            else:
                                st.error("Could not retrieve the object — it may have been deleted.")
                    else:
                        st.info("No objects in the docs bucket yet.")

                with st.expander("⬆️  Upload docs object", expanded=False):
                    docs_uploaded_file = st.file_uploader(
                        "Choose a file to upload to docs bucket",
                        key="docs_upload_file",
                    )
                    docs_custom_key = st.text_input(
                        "Object key (leave blank to use original filename)",
                        key="docs_upload_key_input",
                        placeholder="e.g. DATABASE.md",
                    )
                    if st.button("⬆️  Upload to Docs", key="docs_upload_btn"):
                        if docs_uploaded_file is None:
                            st.warning("Please choose a file first.")
                        else:
                            # Use custom key if provided, otherwise use the original filename
                            target_key = docs_custom_key.strip() if docs_custom_key.strip() else docs_uploaded_file.name
                            data = docs_uploaded_file.getvalue()
                            content_type = docs_uploaded_file.type or "application/octet-stream"
                            ok = minio_docs_put(target_key, data, content_type)
                            if ok:
                                log.info("Database Viewer [MinIO Docs]: Uploaded '%s' (%d bytes)", target_key, len(data))
                                st.success(f"✅ Uploaded `{target_key}` to docs bucket ({len(data):,} bytes).")
                                _cached_minio_docs_bucket_stats.clear()
                                _cached_minio_list_docs_objects.clear()
                                st.rerun()
                            else:
                                st.error("Upload failed — check Logs for details.")

                with st.expander("🗑️  Delete docs object", expanded=False):
                    if docs_keys:
                        docs_del_key = st.selectbox(
                            "Select docs object to delete",
                            options=docs_keys,
                            key="docs_del_key_select",
                        )
                        docs_del_confirm = st.text_input(
                            "Type DELETE to confirm",
                            key="docs_del_confirm_input",
                            placeholder="DELETE",
                        )
                        if st.button("🗑️  Delete docs object", type="primary", key="docs_del_btn"):
                            if docs_del_confirm.strip() != "DELETE":
                                st.error("Type exactly `DELETE` (all caps) to confirm.")
                            else:
                                ok = minio_docs_delete(docs_del_key)
                                if ok:
                                    log.warning("Database Viewer [MinIO Docs]: Deleted docs object '%s'", docs_del_key)
                                    st.success(f"✅ Deleted `{docs_del_key}` from docs bucket.")
                                    _cached_minio_docs_bucket_stats.clear()
                                    _cached_minio_list_docs_objects.clear()
                                    st.rerun()
                                else:
                                    st.error("Delete failed — MinIO may be offline or the key no longer exists.")
                    else:
                        st.info("No objects in the docs bucket yet.")

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
            st.caption("Deletes all entities, scans, document extractions, and identity checks.")

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
                    log.warning("Database Viewer [Danger Zone]: Admin user initiated FULL database reset. Truncating all entities, scans, and cases.")
                    with st.spinner("Truncating all tables…"):
                        ok = reset_db()
                    if ok:
                        log.info("Database Viewer [Danger Zone]: Successfully reset PostgreSQL database.")
                        st.session_state["db_reset_success"] = True
                        st.rerun()
                    else:
                        log.error("Database Viewer [Danger Zone]: Failed to reset PostgreSQL database. Reverting operation.")
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
                    log.warning("Database Viewer [Danger Zone]: Admin user initiated FULL MinIO bucket deletion.")
                    with st.spinner("Deleting all objects from the bucket…"):
                        ok = minio_truncate_bucket()
                    if ok:
                        log.info("Database Viewer [Danger Zone]: Successfully emptied MinIO bucket objects.")
                        st.session_state["minio_reset_success"] = True
                        st.rerun()
                    else:
                        log.error("Database Viewer [Danger Zone]: MinIO bucket deletion process failed.")
                        st.error(
                            "Reset failed — MinIO may be offline or misconfigured. "
                            "Check the Logs page."
                        )
                else:
                    st.error("Type exactly `RESET` (all caps) to confirm.")

        # ── Per-table individual reset ─────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 🗑️ Reset Individual Tables")
        st.caption(
            "Truncate a single table without touching others.  "
            "Useful when you only want to clear scans but keep entity records, etc."
        )

        # Labels shown in the UI and the actual table name they map to.
        _INDIVIDUAL_TABLES: list[tuple[str, str]] = [
            ("Entities", "entities"),
            ("Scans", "scans"),
            ("Document Extractions", "document_extractions"),
            ("Identity Checks", "identity_checks"),
            ("Video KYC Checks", "video_kyc_checks"),
            ("Entity Reports", "entity_reports"),
            ("Face Scan Live Results", "face_scan_live_results"),
        ]

        # Display one row per table: label | description | confirm input | button
        for _label, _tname in _INDIVIDUAL_TABLES:
            _success_key = f"table_reset_success_{_tname}"
            _confirm_key = f"table_reset_confirm_{_tname}"
            _btn_key = f"table_reset_btn_{_tname}"

            tcol1, tcol2, tcol3 = st.columns([3, 3, 2])
            with tcol1:
                st.markdown(f"**{_label}** (`{_tname}`)")
            with tcol2:
                _confirm_val = st.text_input(
                    "Type RESET",
                    key=_confirm_key,
                    placeholder="RESET",
                    label_visibility="collapsed",
                )
            with tcol3:
                if st.button(f"🗑️ Clear {_label}", key=_btn_key):
                    if _confirm_val.strip() == "RESET":
                        log.warning(f"Database Viewer [Danger Zone]: Truncating specific table `{_tname}`.")
                        with st.spinner(f"Truncating `{_tname}`…"):
                            _ok = truncate_table(_tname)
                        if _ok:
                            log.info(f"Database Viewer [Danger Zone]: Safely truncated table `{_tname}`.")
                            st.session_state[_success_key] = True
                            st.rerun()
                        else:
                            log.error(f"Database Viewer [Danger Zone]: Error occurred truncating table `{_tname}`.")
                            st.error(f"Failed to clear `{_tname}` — see Logs for details.")
                    else:
                        st.error("Type exactly `RESET` (all caps) to confirm.")

            # Show success flash message on the row after a successful reset
            if _success_key in st.session_state:
                st.success(f"✅ `{_tname}` cleared successfully.")
                del st.session_state[_success_key]

