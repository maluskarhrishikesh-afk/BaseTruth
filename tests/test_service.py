from __future__ import annotations

import json
from pathlib import Path

from basetruth.service import BaseTruthService


def test_scan_document_from_existing_raw_parse(tmp_path: Path) -> None:
    raw_parse = tmp_path / "Payslip_2026_Jan_liteparse.json"
    raw_parse.write_text(
        json.dumps(
            {
                "pages": [
                    {
                        "page": 1,
                        "text": (
                            "NeoZoom Technologies Pvt Ltd\n"
                            "Payslip for the month of January 2026\n"
                            "Employee ID   NZ66                   Employee Name     Hrishikesh Maluskar\n"
                            "Gross Earnings                                   349712\n"
                            "Net Pay                                          273082   Total Deductions    76630\n"
                            "Rupees Two Hundred Seventy Three Thousand Eighty Two Only\n"
                        ),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    service = BaseTruthService(tmp_path / "artifacts")
    report = service.scan_document(raw_parse)

    assert report["structured_summary"]["document"]["type"] == "payslip"
    assert Path(report["artifacts"]["verification_json_path"]).exists()
    assert Path(report["artifacts"]["verification_markdown_path"]).exists()
