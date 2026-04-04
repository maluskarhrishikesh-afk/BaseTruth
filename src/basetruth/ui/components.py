"""BaseTruth UI — shared components: DB imports, service factory, shared widgets.

All page modules import from here so DB availability logic lives in one place.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

from basetruth.datasources import DatasourceConfig, DatasourceRegistry  # noqa: F401 — re-exported
from basetruth.service import BaseTruthService
from basetruth.ui.theme import (  # noqa: F401 — re-exported for page modules
    _DISPOSITION_ICONS,
    _RISK_COLORS,
    _badge,
    _score_card,
    _signal_icon,
    _status_badge,
)

# ---------------------------------------------------------------------------
# Database layer (optional — app boots even without PostgreSQL)
# ---------------------------------------------------------------------------

try:
    from basetruth.db import db_available, init_db
    from basetruth.store import (
        db_dashboard_stats,
        db_stats,
        db_table_counts,
        db_table_rows,
        get_all_entities_with_scans,
        get_entity_identity_checks,
        get_entity_latest_pdf,
        get_entity_scans,
        get_scan_pdf,
        list_cases_from_db,
        list_recent_scans,
        minio_available,
        minio_bucket_stats,
        minio_get_object,
        minio_delete_object,
        minio_list_entity_objects,
        minio_list_objects,
        minio_truncate_bucket,
        minio_upload,
        reset_db,
        save_identity_check,
        search_entities,
        update_case_in_db,
        update_entity,
    )
    _DB_IMPORTS_OK = True
except Exception:  # noqa: BLE001
    _DB_IMPORTS_OK = False

    def db_available() -> bool:  # type: ignore[misc]
        return False

    def init_db() -> bool:  # type: ignore[misc]
        return False

    def db_dashboard_stats() -> dict:  # type: ignore[misc]
        return {}

    def db_stats() -> dict:  # type: ignore[misc]
        return {}

    def db_table_counts() -> dict:  # type: ignore[misc]
        return {}

    def db_table_rows(table: str, limit: int = 500) -> tuple:  # type: ignore[misc]
        return [], 0

    def get_all_entities_with_scans() -> list:  # type: ignore[misc]
        return []

    def get_entity_identity_checks(ref: str) -> list:  # type: ignore[misc]
        return []

    def get_entity_latest_pdf(ref: str) -> tuple:  # type: ignore[misc]
        return None, None

    def get_entity_scans(ref: str) -> list:  # type: ignore[misc]
        return []

    def get_scan_pdf(scan_id: int) -> Optional[bytes]:  # type: ignore[misc]
        return None

    def list_cases_from_db() -> list:  # type: ignore[misc]
        return []

    def list_recent_scans(limit: int = 50) -> list:  # type: ignore[misc]
        return []

    def minio_available() -> bool:  # type: ignore[misc]
        return False

    def minio_bucket_stats() -> dict:  # type: ignore[misc]
        return {}

    def minio_get_object(key: str) -> Optional[bytes]:  # type: ignore[misc]
        return None

    def minio_delete_object(key: str) -> bool:  # type: ignore[misc]
        return False

    def minio_list_entity_objects(entity_ref: str) -> list:  # type: ignore[misc]
        return []

    def minio_list_objects(limit: int = 500) -> list:  # type: ignore[misc]
        return []

    def minio_truncate_bucket() -> bool:  # type: ignore[misc]
        return False

    def minio_upload(key: str, data: bytes, content_type: str = "") -> None:  # type: ignore[misc]
        pass

    def reset_db() -> bool:  # type: ignore[misc]
        return False

    def save_identity_check(**kwargs) -> Optional[dict]:  # type: ignore[misc]
        return None

    def search_entities(query: str, field: str = "all", limit: int = 50) -> list:  # type: ignore[misc]
        return []

    def update_case_in_db(case_key: str, **kwargs) -> bool:  # type: ignore[misc]
        return False

    def update_entity(ref: str, fields: dict) -> Optional[dict]:  # type: ignore[misc]
        return None


# ---------------------------------------------------------------------------
# Logging (optional)
# ---------------------------------------------------------------------------

try:
    from basetruth.logger import log_path as _log_path  # type: ignore[import]
    _LOGGER_OK = True
except Exception:  # noqa: BLE001
    _LOGGER_OK = False

    def _log_path():  # type: ignore[misc]
        return None


# ---------------------------------------------------------------------------
# Cached availability helpers — avoids a live DB/MinIO round-trip on every
# Streamlit re-render (tab click, widget change, etc.).  TTL = 30 s so the
# status pill refreshes briefly after a service comes online.
# ---------------------------------------------------------------------------

@st.cache_data(ttl=30, show_spinner=False)
def _db_available_cached() -> bool:
    """Cached wrapper around db_available() — 30-second TTL."""
    return db_available()


@st.cache_data(ttl=30, show_spinner=False)
def _minio_available_cached() -> bool:
    """Cached wrapper around minio_available() — 30-second TTL."""
    return minio_available()


# ---------------------------------------------------------------------------
# Page title helper — renders emoji in its native colour + text with gradient
# ---------------------------------------------------------------------------

def _page_title(emoji: str, title_text: str) -> str:
    """Return an HTML string for a page <h1> where the emoji keeps its native
    colour and the title text gets the standard BaseTruth indigo gradient.

    Usage::

        st.markdown(_page_title("�️", "Database Viewer"), unsafe_allow_html=True)
    """
    return (
        '<h1 style="letter-spacing:-0.03em;font-weight:800;font-size:2.1rem;'
        'line-height:1.15;margin-bottom:0.15rem;">'
        f'<span style="-webkit-text-fill-color:initial !important;background:none !important;'
        f'color:inherit;">{emoji}</span>'
        '<span style="background:linear-gradient(135deg,#6366f1 0%,#8b5cf6 60%,#06b6d4 100%);'
        '-webkit-background-clip:text;-webkit-text-fill-color:transparent;'
        f'background-clip:text;"> {title_text}</span>'
        "</h1>"
    )


# ---------------------------------------------------------------------------
# Service helpers
# ---------------------------------------------------------------------------

def _default_artifact_root() -> Path:
    return Path.cwd() / "artifacts"


def _get_service() -> BaseTruthService:
    artifact_root = Path(str(st.session_state.get("artifact_root", _default_artifact_root())))
    return BaseTruthService(artifact_root)


# ---------------------------------------------------------------------------
# File upload helper
# ---------------------------------------------------------------------------

def _save_uploaded_files(files: List[Any], temp_dir: Path) -> List[Path]:
    saved: List[Path] = []
    temp_dir.mkdir(parents=True, exist_ok=True)
    for file in files:
        target = temp_dir / file.name
        target.write_bytes(file.getbuffer())
        saved.append(target)
    return saved


def _display_truth_score(value: Any) -> str:
    return "" if value in {None, ""} else str(value)


# ---------------------------------------------------------------------------
# Shared Streamlit widgets
# ---------------------------------------------------------------------------

def _render_report_summary(report: Dict[str, Any]) -> None:
    """Render a scan-report summary (score card + fields + forensic signals)."""
    tamper = report.get("tamper_assessment", {})
    structured = report.get("structured_summary", {})
    key_fields = structured.get("key_fields", {})
    risk_level = str(tamper.get("risk_level", "low"))
    score = tamper.get("truth_score", 0)

    col_score, col_detail = st.columns([1, 2])
    with col_score:
        st.markdown(_score_card(score, risk_level), unsafe_allow_html=True)

    with col_detail:
        doc_type = structured.get("document", {}).get("type", "generic")
        st.markdown(
            f"**Document type:** {doc_type.replace('_', ' ').title()}  \n"
            f"**Source:** {report.get('source', {}).get('name', '')}  \n"
            f"**Verdict:** {tamper.get('verdict', '')}",
        )
        flat_fields = {
            k: v for k, v in key_fields.items()
            if not isinstance(v, (dict, list)) and v is not None
        }
        if flat_fields:
            field_rows = [
                {"Field": k.replace("_", " ").title(), "Value": str(v)}
                for k, v in flat_fields.items()
            ]
            try:
                import pandas as pd  # noqa: PLC0415
                st.dataframe(pd.DataFrame(field_rows), hide_index=True, width="stretch")
            except Exception:  # noqa: BLE001
                st.json(flat_fields)

    signals = tamper.get("signals", [])
    if signals:
        with st.expander(f"Forensic signals  ({len(signals)} total)", expanded=False):
            for sig in signals:
                icon = _signal_icon(sig)
                name = str(sig.get("name", "")).replace("_", " ").replace("::", " › ")
                score_part = f"  — score {sig.get('score', 0)}" if sig.get("score", 0) else ""
                st.markdown(f"{icon} **{name}**{score_part}")
                st.caption(sig.get("summary", ""))
                if sig.get("details"):
                    st.json(sig["details"])


def _render_entity_link_widget(
    key_prefix: str,
    mandatory: bool = False,
) -> Tuple[Optional[str], Optional[dict]]:
    """Render the 'Associate with a person' UI panel.

    Reads ``st.session_state["active_entity_ref"]`` on first render so that
    once a person is selected on any screen they are automatically pre-selected
    on every other screen — the user never has to re-enter the same details.

    When the user picks or confirms an entity, the selection is also written back
    to ``st.session_state["active_entity_ref"]`` so it persists across pages.

    Returns
    -------
    forced_ref : str | None
        entity_ref of an existing entity chosen by the user, or None.
    extra_identity : dict | None
        Identity fields typed by the user (used as hints when forced_ref is None).
    """
    # ── Retrieve / initialise session-level active entity ──────────────────
    _active_ref: Optional[str] = st.session_state.get("active_entity_ref")
    _active_label: str = st.session_state.get("active_entity_label", "")

    _widget_expanded = True if (key_prefix == "bulk" or mandatory) else False
    title_suffix = "(mandatory)" if mandatory else "(recommended)"

    # Show a persistent "current applicant" banner when an entity is active
    if _active_ref and _active_label:
        st.info(
            f"**Current applicant:** {_active_label} — all documents on this screen will be "
            f"linked to this person automatically.  [Change below]",
            icon="👤",
        )

    with st.expander(f"👤 Associate documents with a person {title_suffix}", expanded=_widget_expanded):
        st.markdown(
            """
Linking documents to an applicant **prevents duplicate entity records** and keeps all
their documents grouped under one profile in the Records screen.

Once you select a person here, they remain the **active applicant** across all screens
until you change them — you will never need to re-enter the same details.

**How it works:**
- *Search an existing person* — type their name, PAN, Aadhaar, email, or reference number.  
  All documents you scan will be linked to that profile.
- *Or enter identifying details* — if this is a new applicant, type their PAN / email / phone
  below. BaseTruth will auto-match future documents to the same person using these fields.
""",
        )

        link_mode = st.radio(
            "Link mode",
            ["Search existing person", "Enter applicant details manually"],
            horizontal=True,
            key=f"{key_prefix}_link_mode",
        )

        forced_ref: Optional[str] = None
        extra_identity: Optional[dict] = None

        if link_mode == "Search existing person":
            if _DB_IMPORTS_OK and db_available():
                # Pre-fill search box with the active entity ref if one is set
                default_search = _active_ref or ""
                search_q = st.text_input(
                    "Search by name / PAN / Aadhaar / email / phone / BT-ref",
                    key=f"{key_prefix}_entity_search",
                    placeholder="e.g. Aarini Parekh, MVWNV2212G, BT-000003…",
                    value=default_search,
                )
                if search_q.strip():
                    matches = search_entities(search_q.strip(), "all", limit=10)
                    if matches:
                        opts = {
                            f"{m['entity_ref']}  —  {m['first_name']} {m['last_name']}  "
                            f"({m.get('pan_number') or m.get('email') or 'no id'})": m["entity_ref"]
                            for m in matches
                        }
                        # Pre-select the active entity if present in results
                        opt_keys = list(opts.keys())
                        default_idx = 0
                        if _active_ref:
                            for idx, ref in enumerate(opts.values()):
                                if ref == _active_ref:
                                    default_idx = idx
                                    break
                        chosen_label = st.selectbox(
                            "Select person",
                            opt_keys,
                            index=default_idx,
                            key=f"{key_prefix}_entity_select",
                        )
                        forced_ref = opts[chosen_label]
                        display_name = chosen_label.split("—")[0].strip()
                        st.success(f"✅ Scans will be linked to **{display_name}**")

                        # Persist active entity to session state
                        st.session_state["active_entity_ref"] = forced_ref
                        st.session_state["active_entity_label"] = display_name
                    else:
                        st.info(
                            "No matching person found. "
                            "Switch to 'Enter details manually' to create one."
                        )
                elif _active_ref:
                    # No new search typed — use the existing active entity
                    forced_ref = _active_ref
            else:
                st.warning("Database is offline — entity search unavailable.")

        else:  # Manual entry
            if mandatory:
                st.info(
                    "Required: Please provide all the details to link the document securely.",
                    icon="ℹ️",
                )
            mc1, mc2 = st.columns(2)
            e_fn = mc1.text_input("First name", key=f"{key_prefix}_ei_fn", placeholder="Aarini")
            e_ln = mc2.text_input("Last name", key=f"{key_prefix}_ei_ln", placeholder="Parekh")
            mc3, mc4 = st.columns(2)
            e_pan = mc3.text_input(
                "PAN number", key=f"{key_prefix}_ei_pan", placeholder="MVWNV2212G"
            )
            e_aadh = mc4.text_input(
                "Aadhaar number", key=f"{key_prefix}_ei_aadh", placeholder="1234 5678 9012"
            )
            mc5, mc6 = st.columns(2)
            e_email = mc5.text_input("Email", key=f"{key_prefix}_ei_email")
            e_phone = mc6.text_input("Phone", key=f"{key_prefix}_ei_phone")
            if any([e_fn, e_ln, e_pan, e_aadh, e_email, e_phone]):
                extra_identity = {
                    "first_name": e_fn.strip(),
                    "last_name": e_ln.strip(),
                    "pan_number": e_pan.strip().upper(),
                    "aadhar_number": e_aadh.replace(" ", "").strip(),
                    "email": e_email.strip().lower(),
                    "phone": e_phone.strip(),
                }
                # Store hints in session for propagation across screens
                st.session_state["active_entity_identity"] = extra_identity
                st.success(
                    "✅ Identity hints will be used to group documents under the right person."
                )

    return forced_ref, extra_identity


def set_active_entity(entity_ref: str, display_label: str = "") -> None:
    """Programmatically set the active entity in session state.

    Call this after a successful scan result that auto-detected an entity,
    so subsequent screens pick it up without user interaction.
    """
    st.session_state["active_entity_ref"] = entity_ref
    if display_label:
        st.session_state["active_entity_label"] = display_label


def clear_active_entity() -> None:
    """Clear the active entity from session state (start fresh for new applicant)."""
    st.session_state.pop("active_entity_ref", None)
    st.session_state.pop("active_entity_label", None)
    st.session_state.pop("active_entity_identity", None)


# ---------------------------------------------------------------------------
# Index metrics strip (used on Records / optional utility)
# ---------------------------------------------------------------------------

def _render_index_metrics() -> None:
    """Render the slim top-of-page stats bar — uses DB when available."""
    if _DB_IMPORTS_OK and db_available():
        stats = db_stats()
        cols = st.columns(4)
        cols[0].metric(
            "Entities in DB",
            stats.get("entities", 0),
            help="Unique individuals / organisations stored in PostgreSQL.",
        )
        cols[1].metric(
            "Scans in DB",
            stats.get("scans", 0),
            help="Total document scans persisted to the database.",
        )
        cols[2].metric(
            "High-Risk Scans",
            stats.get("high_risk", 0),
            help="Scans flagged high-risk (truth score < 60).",
        )
        try:
            from basetruth.db import Case as _Case  # noqa: PLC0415
            from basetruth.db import db_session  # noqa: PLC0415
            from sqlalchemy import func as _func  # noqa: PLC0415

            with db_session() as _s:
                pending = (
                    _s.query(_func.count(_Case.id))
                    .filter(_Case.disposition == "open")
                    .scalar() or 0
                )
            cols[3].metric(
                "Pending Review",
                pending,
                help="Cases still open — go to Cases to approve or reject.",
            )
        except Exception:  # noqa: BLE001
            cols[3].metric("Pending Review", "—")
    else:
        st.info(
            "📴 **Database offline** — connect PostgreSQL to see live statistics. "
            "Document scans still work and are saved to disk.",
            icon=None,
        )
