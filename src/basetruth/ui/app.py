from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

from basetruth.datasources import DatasourceConfig, DatasourceRegistry
from basetruth.service import BaseTruthService


def _default_artifact_root() -> Path:
    return Path.cwd() / "artifacts"


def _get_service() -> BaseTruthService:
    artifact_root = Path(str(st.session_state.get("artifact_root", _default_artifact_root())))
    return BaseTruthService(artifact_root)


def _display_truth_score(value: Any) -> str:
    return "" if value in {None, ""} else str(value)


def _connector_settings_fields(kind: str, existing: Dict[str, Any]) -> Dict[str, Any]:
    settings: Dict[str, Any] = {}
    if kind == "s3":
        settings["bucket"] = st.text_input("S3 bucket", value=str(existing.get("bucket", "")))
        settings["prefix"] = st.text_input("S3 prefix", value=str(existing.get("prefix", "")))
        settings["region_name"] = st.text_input("AWS region", value=str(existing.get("region_name", "")))
        settings["profile_name"] = st.text_input("AWS profile name", value=str(existing.get("profile_name", "")))
    elif kind == "google_drive":
        settings["folder_id"] = st.text_input("Drive folder id", value=str(existing.get("folder_id", "")))
        settings["service_account_file"] = st.text_input(
            "Service account JSON path",
            value=str(existing.get("service_account_file", "")),
        )
    elif kind == "sharepoint":
        settings["site_id"] = st.text_input("SharePoint site id", value=str(existing.get("site_id", "")))
        settings["drive_id"] = st.text_input("Drive id", value=str(existing.get("drive_id", "")))
        settings["folder_path"] = st.text_input("Folder path", value=str(existing.get("folder_path", "")))
        settings["token_env_var"] = st.text_input(
            "Access token environment variable",
            value=str(existing.get("token_env_var", "BASETRUTH_SHAREPOINT_TOKEN")),
        )
    return settings


def _render_connector_guidance(kind: str, settings: Dict[str, Any]) -> None:
    if kind == "s3":
        st.caption("Authentication uses the selected AWS profile or the standard AWS credential environment variables.")
    elif kind == "google_drive":
        st.caption("Use a service-account JSON path for unattended sync, or rely on local Google application-default credentials.")
    elif kind == "sharepoint":
        token_env_var = str(settings.get("token_env_var", "BASETRUTH_SHAREPOINT_TOKEN"))
        st.caption(f"Sync reads a Microsoft Graph bearer token from {token_env_var}.")


def _save_uploaded_files(files: List[Any], temp_dir: Path) -> List[Path]:
    saved = []
    temp_dir.mkdir(parents=True, exist_ok=True)
    for file in files:
        target = temp_dir / file.name
        target.write_bytes(file.getbuffer())
        saved.append(target)
    return saved


def _render_report_summary(report: Dict[str, Any]) -> None:
    tamper = report.get("tamper_assessment", {})
    structured = report.get("structured_summary", {})
    key_fields = structured.get("key_fields", {})
    cols = st.columns(4)
    cols[0].metric("Truth Score", tamper.get("truth_score", "-"))
    cols[1].metric("Risk Level", tamper.get("risk_level", "-"))
    cols[2].metric("Document Type", structured.get("document", {}).get("type", "-"))
    cols[3].metric("Signals", len(tamper.get("signals", [])))
    st.json(
        {
            "source": report.get("source", {}),
            "key_fields": key_fields,
            "artifacts": report.get("artifacts", {}),
        }
    )
    with st.expander("Signals"):
        st.json(tamper.get("signals", []))


def _scan_many(service: BaseTruthService, paths: List[Path]) -> List[Dict[str, Any]]:
    results = []
    for path in paths:
        results.append(service.scan_document(path))
    return results


def _render_single_scan_tab() -> None:
    st.subheader("Single document scan")
    service = _get_service()
    upload = st.file_uploader("Upload one document or structured JSON", type=None, accept_multiple_files=False)
    path_input = st.text_input("Or scan a file that already exists on disk")
    if st.button("Scan document", type="primary"):
        with st.spinner("Running BaseTruth scan..."):
            if upload is not None:
                temp_dir = Path(tempfile.mkdtemp(prefix="basetruth_upload_"))
                saved_path = _save_uploaded_files([upload], temp_dir)[0]
                report = service.scan_document(saved_path)
            elif path_input.strip():
                report = service.scan_document(path_input.strip())
            else:
                st.warning("Provide an uploaded file or a local file path.")
                return
        _render_report_summary(report)


def _render_bulk_scan_tab() -> None:
    st.subheader("Bulk scan")
    service = _get_service()
    uploads = st.file_uploader("Upload multiple documents", type=None, accept_multiple_files=True)
    folder_input = st.text_input("Or scan all supported files from an existing folder")
    compare_payslips = st.checkbox("Run cross-month payslip comparison after scan", value=True)
    if st.button("Run bulk scan"):
        with st.spinner("Scanning bulk inputs..."):
            paths: List[Path] = []
            if uploads:
                temp_dir = Path(tempfile.mkdtemp(prefix="basetruth_bulk_"))
                paths.extend(_save_uploaded_files(list(uploads), temp_dir))
            if folder_input.strip():
                paths.extend(service.collect_supported_files(folder_input.strip()))
            if not paths:
                st.warning("Provide uploaded files or a folder path.")
                return
            reports = _scan_many(service, paths)
            summary_rows = [
                {
                    "source": report.get("source", {}).get("name", ""),
                    "document_type": report.get("structured_summary", {}).get("document", {}).get("type", ""),
                    "truth_score": report.get("tamper_assessment", {}).get("truth_score", ""),
                    "risk_level": report.get("tamper_assessment", {}).get("risk_level", ""),
                }
                for report in reports
            ]
            st.dataframe(summary_rows, width="stretch")
            if compare_payslips:
                comparison = service.compare_payslip_summaries_from_reports(reports)
                st.markdown("### Payslip anomalies")
                st.json(comparison)


def _render_datasource_tab() -> None:
    st.subheader("Datasource ingestion")
    service = _get_service()
    registry = DatasourceRegistry(service.artifact_root)
    with st.form("datasource_form"):
        name = st.text_input("Datasource name")
        kind = st.selectbox("Datasource type", options=["folder", "manifest", "s3", "google_drive", "sharepoint"])
        path = st.text_input("Source path")
        recursive = st.checkbox("Recursive", value=True)
        extensions = st.text_input("Extensions (comma separated)", value=".pdf,.json,.png,.jpg,.jpeg")
        description = st.text_area("Description")
        settings = _connector_settings_fields(kind, {})
        _render_connector_guidance(kind, settings)
        submitted = st.form_submit_button("Save datasource")
        if submitted:
            resolved_path = registry.build_path_from_settings(kind, settings, path)
            config = DatasourceConfig(
                name=name,
                kind=kind,
                path=resolved_path,
                recursive=recursive,
                extensions=[item.strip() for item in extensions.split(",") if item.strip()],
                description=description,
                settings=settings,
            )
            registry.upsert_source(config)
            st.success(f"Saved datasource '{name}'.")

    st.caption(
        "Path format hints: folder -> local path, manifest -> path to json/csv/txt manifest, "
        "s3 -> s3://bucket/prefix, google_drive -> folder_id or drive:folder_id, "
        "sharepoint -> site_id|drive_id|folder_path"
    )

    sources = registry.list_sources()
    if sources:
        st.markdown("### Registered datasources")
        st.dataframe([source.to_dict() for source in sources], width="stretch")
        selected_source = st.selectbox("Select datasource to sync", options=[source.name for source in sources])
        selected_config = registry.get_source(selected_source)
        with st.expander("Connector auth and config", expanded=False):
            st.json({
                "path": selected_config.path,
                "settings": selected_config.settings or {},
            })
            _render_connector_guidance(selected_config.kind, dict(selected_config.settings or {}))
        col1, col2 = st.columns(2)
        if col1.button("Sync datasource"):
            result = registry.sync_source(selected_source)
            st.json(result)
        if col2.button("Sync and scan"):
            result = registry.sync_source(selected_source)
            if result.get("status") == "success":
                reports = service.scan_many(result.get("copied_files", []))
                st.json({
                    "sync": result,
                    "scan_count": len(reports),
                })
            else:
                st.json(result)
    else:
        st.info("No datasources registered yet.")

    st.markdown("### Recommended operating model")
    st.write(
        "BaseTruth should sync documents from client systems into read-only snapshots under its managed workspace. "
        "That preserves chain-of-custody, keeps the client's live folders untouched, and gives you deterministic evidence trails."
    )


def _render_reports_tab() -> None:
    st.subheader("Reports")
    service = _get_service()
    reports = service.list_reports()
    if not reports:
        st.info("No reports found under the current artifact root.")
        return
    rows = [
        {
            "source": item.get("source_name", ""),
            "kind": item.get("kind", ""),
            "case_key": item.get("case_key", ""),
            "risk_level": item.get("risk_level", ""),
            "truth_score": _display_truth_score(item.get("truth_score")),
            "path": item.get("path", ""),
        }
        for item in reports
    ]
    st.dataframe(rows, width="stretch")
    selection = st.selectbox("Open report", options=[item["path"] for item in rows])
    if selection:
        payload = json.loads(Path(selection).read_text(encoding="utf-8"))
        st.json(payload)


def _render_cases_tab() -> None:
    st.subheader("Cases")
    service = _get_service()
    cases = service.list_cases()
    if not cases:
        st.info("No cases found yet. Run scans first.")
        return
    rows = [
        {
            "case_key": item.get("case_key", ""),
            "document_type": item.get("document_type", ""),
            "document_count": item.get("document_count", 0),
            "max_risk_level": item.get("max_risk_level", ""),
            "min_truth_score": _display_truth_score(item.get("min_truth_score")),
        }
        for item in cases
    ]
    st.dataframe(rows, width="stretch")
    selected_case_key = st.selectbox("Open case", options=[item["case_key"] for item in rows])
    case_detail = service.get_case_detail(selected_case_key)
    workflow = case_detail["workflow"]
    st.markdown("### Case summary")
    st.json(case_detail["case"])
    with st.form("case_workflow_form"):
        status = st.selectbox("Status", options=["new", "triage", "investigating", "pending_client", "closed"], index=["new", "triage", "investigating", "pending_client", "closed"].index(str(workflow.get("status", "new"))))
        disposition = st.selectbox("Disposition", options=["open", "monitor", "escalate", "cleared", "fraud_confirmed"], index=["open", "monitor", "escalate", "cleared", "fraud_confirmed"].index(str(workflow.get("disposition", "open"))))
        priority = st.selectbox("Priority", options=["low", "normal", "high", "critical"], index=["low", "normal", "high", "critical"].index(str(workflow.get("priority", "normal"))))
        assignee = st.text_input("Investigator", value=str(workflow.get("assignee", "")))
        labels_text = st.text_input("Labels", value=", ".join(workflow.get("labels", [])))
        note_author = st.text_input("Note author", value="analyst")
        note_text = st.text_area("Add note")
        workflow_submitted = st.form_submit_button("Update case")
        if workflow_submitted:
            service.update_case(
                selected_case_key,
                status=status,
                disposition=disposition,
                priority=priority,
                assignee=assignee,
                labels=[item.strip() for item in labels_text.split(",") if item.strip()],
                note_text=note_text,
                note_author=note_author,
            )
            st.success("Case workflow updated.")
            case_detail = service.get_case_detail(selected_case_key)
            workflow = case_detail["workflow"]

    st.markdown("### Workflow")
    st.json(workflow)
    st.markdown("### Investigator notes")
    if workflow.get("notes"):
        for note in reversed(workflow["notes"]):
            st.write(f"{note.get('created_at', '')} | {note.get('author', '')}")
            st.write(note.get("text", ""))
    else:
        st.info("No case notes yet.")
    st.markdown("### Case reports")
    for report in case_detail["reports"]:
        with st.expander(report.get("source", {}).get("name", "report")):
            _render_report_summary(report)


def _render_index_metrics() -> None:
    service = _get_service()
    reports = service.list_reports()
    cases = service.list_cases()
    verification_reports = [item for item in reports if item.get("kind") == "verification"]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Cases", len(cases))
    col2.metric("Verification Reports", len(verification_reports))
    col3.metric("High Risk Reports", sum(1 for item in verification_reports if item.get("risk_level") == "high"))
    col4.metric("Comparisons", sum(1 for item in reports if item.get("kind") == "comparison"))


def main() -> None:
    st.set_page_config(page_title="BaseTruth", page_icon="BT", layout="wide")
    if "artifact_root" not in st.session_state:
        st.session_state["artifact_root"] = str(_default_artifact_root())
    st.title("BaseTruth")
    st.caption("Explainable document integrity and fraud detection")
    st.text_input("Artifact root", key="artifact_root")
    _render_index_metrics()

    tab_single, tab_bulk, tab_datasources, tab_reports, tab_cases = st.tabs(
        ["Single Scan", "Bulk Scan", "Datasources", "Reports", "Cases"]
    )
    with tab_single:
        _render_single_scan_tab()
    with tab_bulk:
        _render_bulk_scan_tab()
    with tab_datasources:
        _render_datasource_tab()
    with tab_reports:
        _render_reports_tab()
    with tab_cases:
        _render_cases_tab()


if __name__ == "__main__":
    main()