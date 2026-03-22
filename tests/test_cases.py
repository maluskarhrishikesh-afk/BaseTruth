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