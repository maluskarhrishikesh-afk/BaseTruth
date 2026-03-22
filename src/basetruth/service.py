from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from basetruth.analysis.payslip import compare_payslip_summaries
from basetruth.analysis.structured import build_structured_summary
from basetruth.analysis.tamper import evaluate_tamper_risk
from basetruth.integrations.liteparse import check_liteparse_available, parse_document_to_json
from basetruth.integrations.pdf import extract_pdf_metadata
from basetruth.models import CaseNote, CaseRecord, VerificationReport
from basetruth.reporting.markdown import render_comparison_report, render_scan_report


class BaseTruthService:
    def __init__(self, artifact_root: Path | str = "artifacts") -> None:
        self.artifact_root = Path(artifact_root)
        self.artifact_root.mkdir(parents=True, exist_ok=True)

    def _document_artifact_dir(self, source_name: str) -> Path:
        stem = Path(source_name).stem.replace(" ", "_")
        target = self.artifact_root / stem
        target.mkdir(parents=True, exist_ok=True)
        return target

    def _config_dir(self) -> Path:
        target = self.artifact_root / "config"
        target.mkdir(parents=True, exist_ok=True)
        return target

    def _case_records_path(self) -> Path:
        return self._config_dir() / "case_records.json"

    def _load_case_records(self) -> Dict[str, CaseRecord]:
        path = self._case_records_path()
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return {}
        items = payload.get("cases", {}) if isinstance(payload, dict) else {}
        records: Dict[str, CaseRecord] = {}
        for case_key, item in items.items():
            if not isinstance(item, dict):
                continue
            notes = [CaseNote(**note) for note in item.get("notes", []) if isinstance(note, dict)]
            records[str(case_key)] = CaseRecord(
                case_key=str(case_key),
                status=str(item.get("status", "new")),
                disposition=str(item.get("disposition", "open")),
                priority=str(item.get("priority", "normal")),
                assignee=str(item.get("assignee", "")),
                labels=[str(label) for label in item.get("labels", [])],
                notes=notes,
                created_at=str(item.get("created_at", "")),
                updated_at=str(item.get("updated_at", "")),
            )
        return records

    def _save_case_records(self, records: Dict[str, CaseRecord]) -> None:
        payload = {
            "schema_version": 1,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "cases": {case_key: record for case_key, record in records.items()},
        }
        self._case_records_path().write_text(json.dumps(payload, default=lambda value: value.__dict__, indent=2, ensure_ascii=False), encoding="utf-8")

    def update_case(
        self,
        case_key: str,
        *,
        status: str | None = None,
        disposition: str | None = None,
        priority: str | None = None,
        assignee: str | None = None,
        labels: List[str] | None = None,
        note_text: str = "",
        note_author: str = "",
    ) -> Dict[str, Any]:
        timestamp = datetime.now(timezone.utc).isoformat()
        records = self._load_case_records()
        record = records.get(case_key) or CaseRecord(case_key=case_key, created_at=timestamp, updated_at=timestamp)
        if status is not None:
            record.status = str(status)
        if disposition is not None:
            record.disposition = str(disposition)
        if priority is not None:
            record.priority = str(priority)
        if assignee is not None:
            record.assignee = str(assignee)
        if labels is not None:
            record.labels = sorted({label.strip() for label in labels if str(label).strip()})
        if note_text.strip():
            record.notes.append(CaseNote(created_at=timestamp, author=str(note_author).strip() or "analyst", text=note_text.strip()))
        record.updated_at = timestamp
        if not record.created_at:
            record.created_at = timestamp
        records[case_key] = record
        self._save_case_records(records)
        return record.__dict__ | {"notes": [note.__dict__ for note in record.notes]}

    def collect_supported_files(self, input_dir: str | Path) -> List[Path]:
        directory = Path(input_dir)
        if not directory.exists() or not directory.is_dir():
            raise NotADirectoryError(directory)
        supported_extensions = {".pdf", ".json", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}
        paths: List[Path] = []
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix.lower() in supported_extensions:
                paths.append(path)
        return paths

    def scan_many(self, input_paths: List[str | Path]) -> List[Dict[str, Any]]:
        return [self.scan_document(path) for path in input_paths]

    def _case_key_for_report(self, report: Dict[str, Any]) -> str:
        summary = report.get("structured_summary", {})
        document_type = summary.get("document", {}).get("type", "generic")
        key_fields = summary.get("key_fields", {})
        parts = [document_type]
        for field_name in ("employee_id", "employee_name", "issuer_name", "account_number"):
            value = key_fields.get(field_name) or summary.get("document", {}).get(field_name)
            if value:
                parts.append(str(value).strip().lower().replace(" ", "_"))
                break
        else:
            parts.append(Path(report.get("source", {}).get("name", "unknown")).stem.lower())
        return "::".join(parts)

    def scan_document(self, input_path: str | Path) -> Dict[str, Any]:
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(path)

        artifact_dir = self._document_artifact_dir(path.name)
        raw_parse_path = artifact_dir / f"{path.stem}_liteparse.json"
        structured_path = artifact_dir / f"{path.stem}_structured.json"
        verification_json_path = artifact_dir / f"{path.stem}_verification.json"
        verification_markdown_path = artifact_dir / f"{path.stem}_verification.md"

        source_path = path
        if path.suffix.lower() == ".json" and path.name.endswith("_structured.json"):
            structured_summary = json.loads(path.read_text(encoding="utf-8"))
            pdf_metadata = {}
            raw_parse_written = ""
        else:
            if path.suffix.lower() == ".json":
                raw_parse_path = path
            else:
                liteparse_status = check_liteparse_available()
                if liteparse_status["available"]:
                    result = parse_document_to_json(path, raw_parse_path)
                    if result["status"] != "success":
                        raise RuntimeError(result["message"])
                else:
                    raise RuntimeError(liteparse_status["message"])
            structured_summary = build_structured_summary(raw_parse_path, source_path=source_path)
            structured_path.write_text(json.dumps(structured_summary, indent=2, ensure_ascii=False), encoding="utf-8")
            pdf_metadata = extract_pdf_metadata(source_path) if source_path.suffix.lower() == ".pdf" else {}
            raw_parse_written = str(raw_parse_path)

        if path.suffix.lower() == ".json" and path.name.endswith("_structured.json"):
            structured_path = path
        tamper_assessment = evaluate_tamper_risk(structured_summary, pdf_metadata)

        source = {
            "path": str(source_path),
            "name": source_path.name,
            "sha256": pdf_metadata.get("sha256", ""),
            "size_bytes": source_path.stat().st_size if source_path.exists() else 0,
        }
        report = VerificationReport(
            schema_version=1,
            generated_at=datetime.now(timezone.utc).isoformat(),
            source=source,
            pdf_metadata=pdf_metadata,
            structured_summary=structured_summary,
            tamper_assessment=tamper_assessment,
            artifacts={
                "raw_parse_path": raw_parse_written,
                "structured_summary_path": str(structured_path),
                "verification_json_path": str(verification_json_path),
                "verification_markdown_path": str(verification_markdown_path),
            },
        )
        verification_json_path.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        verification_markdown_path.write_text(render_scan_report(report.to_dict()), encoding="utf-8")
        return report.to_dict()

    def compare_payslip_folder(self, input_dir: str | Path) -> Dict[str, Any]:
        directory = Path(input_dir)
        if not directory.exists() or not directory.is_dir():
            raise NotADirectoryError(directory)

        summaries: List[Dict[str, Any]] = []
        for path in sorted(directory.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() == ".json" and path.name.endswith("_structured.json"):
                summaries.append(json.loads(path.read_text(encoding="utf-8")))
            elif path.suffix.lower() == ".pdf":
                scan_report = self.scan_document(path)
                summaries.append(scan_report["structured_summary"])

        comparison = compare_payslip_summaries(summaries)
        comparison_dir = self.artifact_root / "comparisons"
        comparison_dir.mkdir(parents=True, exist_ok=True)
        json_path = comparison_dir / "payslip_comparison.json"
        markdown_path = comparison_dir / "payslip_comparison.md"
        comparison["artifacts"] = {
            "comparison_json_path": str(json_path),
            "comparison_markdown_path": str(markdown_path),
        }
        json_path.write_text(json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8")
        markdown_path.write_text(render_comparison_report(comparison), encoding="utf-8")
        return comparison

    def compare_payslip_summaries_from_reports(self, reports: List[Dict[str, Any]]) -> Dict[str, Any]:
        summaries = [report.get("structured_summary", {}) for report in reports if report.get("structured_summary")]
        return compare_payslip_summaries(summaries)

    def list_reports(self) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for path in sorted(self.artifact_root.rglob("*_verification.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                continue
            results.append(
                {
                    "kind": "verification",
                    "path": str(path),
                    "source_name": payload.get("source", {}).get("name", ""),
                    "risk_level": payload.get("tamper_assessment", {}).get("risk_level", ""),
                    "truth_score": payload.get("tamper_assessment", {}).get("truth_score", ""),
                    "case_key": self._case_key_for_report(payload),
                    "document_type": payload.get("structured_summary", {}).get("document", {}).get("type", ""),
                    "generated_at": payload.get("generated_at", ""),
                }
            )
        for path in sorted(self.artifact_root.rglob("*_comparison.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                continue
            results.append(
                {
                    "kind": "comparison",
                    "path": str(path),
                    "source_name": payload.get("document_type", "comparison"),
                    "risk_level": "review" if payload.get("anomalies") else "low",
                    "truth_score": "",
                    "case_key": f"comparison::{payload.get('document_type', 'generic')}",
                    "document_type": payload.get("document_type", ""),
                    "generated_at": payload.get("generated_at", ""),
                }
            )
        return results

    def list_cases(self) -> List[Dict[str, Any]]:
        case_records = self._load_case_records()
        cases: Dict[str, Dict[str, Any]] = {}
        for item in self.list_reports():
            if item.get("kind") != "verification":
                continue
            case_key = item.get("case_key", "uncategorized")
            case = cases.setdefault(
                case_key,
                {
                    "case_key": case_key,
                    "document_type": item.get("document_type", "generic"),
                    "documents": [],
                    "max_risk_level": "low",
                    "min_truth_score": 100,
                },
            )
            case["documents"].append(item)
            truth_score = item.get("truth_score")
            if isinstance(truth_score, int):
                case["min_truth_score"] = min(case["min_truth_score"], truth_score)
            risk_level = str(item.get("risk_level", "low"))
            if risk_level == "high" or (risk_level == "medium" and case["max_risk_level"] == "low"):
                case["max_risk_level"] = risk_level

        for case in cases.values():
            record = case_records.get(case["case_key"])
            case["document_count"] = len(case["documents"])
            case["documents"] = sorted(case["documents"], key=lambda item: str(item.get("generated_at", "")), reverse=True)
            if case["min_truth_score"] == 100 and not case["documents"]:
                case["min_truth_score"] = ""
            case["status"] = record.status if record else "new"
            case["disposition"] = record.disposition if record else "open"
            case["priority"] = record.priority if record else "normal"
            case["assignee"] = record.assignee if record else ""
            case["labels"] = list(record.labels) if record else []
            case["note_count"] = len(record.notes) if record else 0
            case["updated_at"] = record.updated_at if record else case["documents"][0].get("generated_at", "")
        return sorted(cases.values(), key=lambda item: (str(item.get("max_risk_level")), -int(item.get("document_count", 0))), reverse=True)

    def get_case_detail(self, case_key: str) -> Dict[str, Any]:
        case_records = self._load_case_records()
        for case in self.list_cases():
            if case.get("case_key") == case_key:
                details: List[Dict[str, Any]] = []
                for document in case.get("documents", []):
                    payload = json.loads(Path(document["path"]).read_text(encoding="utf-8"))
                    details.append(payload)
                record = case_records.get(case_key)
                return {
                    "case": case,
                    "workflow": (
                        {
                            "case_key": record.case_key,
                            "status": record.status,
                            "disposition": record.disposition,
                            "priority": record.priority,
                            "assignee": record.assignee,
                            "labels": list(record.labels),
                            "notes": [note.__dict__ for note in record.notes],
                            "created_at": record.created_at,
                            "updated_at": record.updated_at,
                        }
                        if record
                        else {
                            "case_key": case_key,
                            "status": "new",
                            "disposition": "open",
                            "priority": "normal",
                            "assignee": "",
                            "labels": [],
                            "notes": [],
                            "created_at": "",
                            "updated_at": "",
                        }
                    ),
                    "reports": details,
                }
        raise KeyError(case_key)
