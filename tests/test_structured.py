from __future__ import annotations

import json
from pathlib import Path

from basetruth.analysis.structured import build_structured_summary


def test_build_structured_summary_for_payslip(tmp_path: Path) -> None:
    parse_output = tmp_path / "Payslip_2026_Jan_liteparse.json"
    parse_output.write_text(
        json.dumps(
            {
                "pages": [
                    {
                        "page": 1,
                        "text": (
                            "NeoZoom Technologies Pvt Ltd\n"
                            "Payslip for the month of January 2026\n"
                            "Employee ID   NZ66                   Employee Name     Hrishikesh Maluskar\n"
                            "Designation   Sr. Java Developer     Date of Joining   9/25/2023\n"
                            "Pay Date      2/3/2026               Paid Days         31\n"
                            "LOP Days      0                      UAN               100165351971\n"
                            "Basic                                             53500   EPF Contribution     6420\n"
                            "House Rent Allowance                              26750   Income Tax          70010\n"
                            "Fixed Bonus                                       32100   Professional Tax      200\n"
                            "Other Allowances                                 237362   Other Deductions        0\n"
                            "Advance or Arrears or Notice Pay                      0\n"
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

    summary = build_structured_summary(parse_output, source_path=Path("Payslip_2026_Jan.pdf"))

    assert summary["document"]["type"] == "payslip"
    assert summary["key_fields"]["employee_id"] == "NZ66"
    assert summary["key_fields"]["net_pay"] == "273082"
    labels = {item["label"] for item in summary["generic_candidates"]["monetary_fields"]}
    assert "Pay Date" not in labels
