from __future__ import annotations

import json
from pathlib import Path

from basetruth.service import BaseTruthService


def test_list_cases_groups_related_reports(tmp_path: Path) -> None:
    report_dir = tmp_path / "artifacts" / "jan"
    report_dir.mkdir(parents=True)
    report_path = report_dir / "jan_verification.json"
    payload = {
        "generated_at": "2026-03-22T00:00:00+00:00",
        "source": {"name": "jan.pdf"},
        "structured_summary": {
            "document": {"type": "payslip"},
            "key_fields": {"employee_id": "NZ66", "employee_name": "Hrishikesh Maluskar"},
        },
        "tamper_assessment": {"risk_level": "low", "truth_score": 95},
    }
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    service = BaseTruthService(tmp_path / "artifacts")
    cases = service.list_cases()

    assert len(cases) == 1
    assert cases[0]["document_count"] == 1
    detail = service.get_case_detail(cases[0]["case_key"])
    assert len(detail["reports"]) == 1


def test_update_case_persists_workflow_metadata(tmp_path: Path) -> None:
    report_dir = tmp_path / "artifacts" / "jan"
    report_dir.mkdir(parents=True)
    report_path = report_dir / "jan_verification.json"
    payload = {
        "generated_at": "2026-03-22T00:00:00+00:00",
        "source": {"name": "jan.pdf"},
        "structured_summary": {
            "document": {"type": "payslip"},
            "key_fields": {"employee_id": "NZ66"},
        },
        "tamper_assessment": {"risk_level": "medium", "truth_score": 81},
    }
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    service = BaseTruthService(tmp_path / "artifacts")
    case_key = service.list_cases()[0]["case_key"]

    service.update_case(
        case_key,
        status="investigating",
        disposition="escalate",
        priority="high",
        assignee="hrishi",
        labels=["payroll", "suspected_tamper"],
        note_text="Mismatch between net pay and prior month trend.",
        note_author="hrishi",
    )

    detail = service.get_case_detail(case_key)

    assert detail["workflow"]["status"] == "investigating"
    assert detail["workflow"]["disposition"] == "escalate"
    assert detail["workflow"]["assignee"] == "hrishi"
    assert detail["workflow"]["labels"] == ["payroll", "suspected_tamper"]
    assert len(detail["workflow"]["notes"]) == 1