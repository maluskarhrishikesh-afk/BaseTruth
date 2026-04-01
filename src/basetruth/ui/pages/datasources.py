"""Datasources page."""
from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st

from basetruth.datasources import DatasourceConfig, DatasourceRegistry
from basetruth.service import BaseTruthService


def _connector_settings_fields(kind: str, existing: Dict[str, Any]) -> Dict[str, Any]:
    settings: Dict[str, Any] = {}
    if kind == "s3":
        col1, col2 = st.columns(2)
        settings["bucket"] = col1.text_input(
            "S3 bucket *", value=str(existing.get("bucket", ""))
        )
        settings["prefix"] = col2.text_input(
            "Prefix", value=str(existing.get("prefix", ""))
        )
        col3, col4 = st.columns(2)
        settings["region_name"] = col3.text_input(
            "AWS region", value=str(existing.get("region_name", ""))
        )
        settings["profile_name"] = col4.text_input(
            "AWS profile", value=str(existing.get("profile_name", ""))
        )
        st.caption(
            "Auth: uses the named AWS profile, or the standard environment variables "
            "(AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY)."
        )
    elif kind == "google_drive":
        settings["folder_id"] = st.text_input(
            "Drive folder ID *", value=str(existing.get("folder_id", ""))
        )
        settings["service_account_file"] = st.text_input(
            "Service account JSON path (leave empty for ADC)",
            value=str(existing.get("service_account_file", "")),
        )
        st.caption(
            "Auth: service-account JSON for server environments, "
            "or application-default credentials for local use."
        )
    elif kind == "sharepoint":
        col1, col2 = st.columns(2)
        settings["site_id"] = col1.text_input(
            "SharePoint site ID *", value=str(existing.get("site_id", ""))
        )
        settings["drive_id"] = col2.text_input(
            "Drive ID *", value=str(existing.get("drive_id", ""))
        )
        settings["folder_path"] = st.text_input(
            "Folder path", value=str(existing.get("folder_path", ""))
        )
        settings["token_env_var"] = st.text_input(
            "Environment variable holding the Microsoft Graph bearer token",
            value=str(
                existing.get("token_env_var", "BASETRUTH_SHAREPOINT_TOKEN")
            ),
        )
        st.caption(
            "Auth: set the named env var to a valid Microsoft Graph bearer token "
            "before syncing."
        )
    return settings


def _page_datasources(service: BaseTruthService) -> None:
    st.markdown("# 🔗 Datasources")
    registry = DatasourceRegistry(service.artifact_root)
    sources = registry.list_sources()

    if sources:
        st.subheader("Registered datasources")
        try:
            import pandas as pd  # noqa: PLC0415

            st.dataframe(
                pd.DataFrame([s.to_dict() for s in sources]).drop(
                    columns=["settings"], errors="ignore"
                ),
                hide_index=True,
                width="stretch",
            )
        except ImportError:
            st.json([s.to_dict() for s in sources])

        selected_name = st.selectbox(
            "Select datasource", options=[s.name for s in sources]
        )
        selected_cfg = registry.get_source(selected_name)

        with st.expander("Connector auth and config", expanded=False):
            st.json(
                {"path": selected_cfg.path, "settings": selected_cfg.settings or {}}
            )

        col_sync, col_scan, _ = st.columns([1, 1, 4])
        if col_sync.button("Sync"):
            with st.spinner("Syncing…"):
                result = registry.sync_source(selected_name)
            if result.get("status") == "success":
                st.success(result.get("message", "Sync complete."))
            else:
                st.warning(str(result.get("message", result)))
            st.json(result)

        if col_scan.button("Sync + Scan"):
            with st.spinner("Syncing then scanning…"):
                result = registry.sync_source(selected_name)
                scan_reports: List[Dict[str, Any]] = []
                if result.get("status") == "success":
                    for fpath in result.get("copied_files", []):
                        scan_reports.append(service.scan_document(fpath))
            st.success(
                f"Synced {result.get('copied_count', 0)} file(s), "
                f"scanned {len(scan_reports)} document(s)."
            )

        st.divider()

    st.subheader("Register a new datasource")
    with st.form("datasource_form"):
        col_name, col_kind = st.columns(2)
        name = col_name.text_input("Datasource name *")
        kind = col_kind.selectbox(
            "Type",
            options=["folder", "manifest", "s3", "google_drive", "sharepoint"],
        )
        path = st.text_input(
            "Source path (folder / manifest / leave empty to derive from fields below)"
        )
        col_rec, col_ext = st.columns(2)
        recursive = col_rec.checkbox("Recursive", value=True)
        extensions = col_ext.text_input(
            "Extensions", value=".pdf,.json,.png,.jpg,.jpeg"
        )
        description = st.text_area("Description", height=60)

        settings: Dict[str, Any] = {}
        if kind in {"s3", "google_drive", "sharepoint"}:
            st.markdown("**Connector settings**")
            existing_cfg = (
                registry.get_source(name)
                if name and name in [s.name for s in sources]
                else None
            )
            settings = _connector_settings_fields(
                kind, dict(existing_cfg.settings or {}) if existing_cfg else {}
            )

        if st.form_submit_button("Save datasource", type="primary"):
            if not name.strip():
                st.error("Datasource name is required.")
            else:
                resolved_path = registry.build_path_from_settings(
                    kind, settings, path
                )
                registry.upsert_source(
                    DatasourceConfig(
                        name=name.strip(),
                        kind=kind,
                        path=resolved_path,
                        recursive=recursive,
                        extensions=[
                            e.strip() for e in extensions.split(",") if e.strip()
                        ],
                        description=description.strip(),
                        settings=settings or None,
                    )
                )
                st.success(f"Datasource '{name}' saved.")
                st.rerun()

    st.markdown(
        """
        > **Operating model** — BaseTruth syncs documents from client sources into a read-only snapshot
        > under its managed workspace. This preserves chain-of-custody, leaves the client system untouched,
        > and produces deterministic evidence trails for every scanned document.
        """
    )
