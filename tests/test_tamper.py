from __future__ import annotations

from basetruth.analysis.tamper import evaluate_tamper_risk


def test_evaluate_tamper_risk_flags_editor_mismatch() -> None:
    summary = {
        "document": {"type": "payslip"},
        "key_fields": {
            "gross_earnings": "10000",
            "total_deductions": "1000",
            "net_pay": "9000",
            "net_pay_in_words": "Nine Thousand",
            "employee_id": "X1",
            "employee_name": "Test User",
        },
        "generic_candidates": {},
    }
    pdf_metadata = {
        "metadata": {
            "Producer": "Adobe Photoshop 2024",
        },
        "has_digital_signature_markers": False,
        "signature_markers": [],
    }

    assessment = evaluate_tamper_risk(summary, pdf_metadata)

    assert assessment["risk_level"] in {"medium", "high"}
    assert any(signal["name"] == "editor_software_mismatch" and signal["score"] > 0 for signal in assessment["signals"])
