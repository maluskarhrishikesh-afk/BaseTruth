from __future__ import annotations

import base64
import io
from pathlib import Path

from PIL import Image

from basetruth.integrations import document_extract as de


def test_prompt_markdown_loader_contains_required_sections() -> None:
    assert "strict JSON" in de._get_prompt("system")
    assert "document_type\": \"Marksheet\"" in de._get_prompt("marksheet")
    assert "Offer Letter" in de._get_prompt("employment")


def test_classify_marksheet_layout_family_hsc() -> None:
    ocr_text = """
    MAHARASHTRA STATE BOARD
    SUBJECT CODE 01 40 54 55 AZ TOTAL
    MAXIMUM MARKS 100 100 100 100 200 600
    MARKS OBTAINED 070 082 090 091 171 504
    """

    layout = de._classify_marksheet_layout_family(ocr_text)

    assert layout["family"] == de._LAYOUT_HSC_TRANSPOSED
    assert "MAXIMUM MARKS" in layout["markers"]


def test_parse_be_ocr_directly_extracts_components_and_totals() -> None:
    ocr_text = """
    UNIVERSITY OF PUNE
    B.E. EXAMINATION MAY 2006
    MALUSKAR HRISHIKESH NAMDEO
    SEAT NO B2084275
    COURSE NAME MAX MIN OBT
    010. COMPUTER NETWORKS PP 100 40 34
    010. COMPUTER NETWORKS TW 25 10 15
    010. COMPUTER NETWORKS OR 50 20 39
    011. DATABASE MANAGEMENT SYSTEMS PP 100 40 48
    011. DATABASE MANAGEMENT SYSTEMS TW 25 10 18
    011. DATABASE MANAGEMENT SYSTEMS PR 50 20 44
    1500 TOTAL EARNED 928/ RESULT : FIRST CLASS
    """

    parsed = de._parse_be_ocr_directly(ocr_text)

    assert parsed is not None
    assert parsed["board_or_university_name"] == "UNIVERSITY OF PUNE"
    assert parsed["candidate_name"] == "MALUSKAR HRISHIKESH NAMDEO"
    assert parsed["enrollment_or_seat_number"] == "B2084275"
    assert parsed["printed_grand_total"] == 928
    assert parsed["total_max_marks"] == 1500
    assert parsed["result"] == "First Class"
    assert len(parsed["subjects"]) == 6
    assert parsed["subjects"][0] == {
        "subject_name": "Computer Networks (PP)",
        "marks_obtained": 34,
        "max_marks": 100,
    }


def test_select_best_marksheet_ocr_candidate_prefers_stronger_be_parse() -> None:
    weak_text = """
    UNIVERSITY OF PUNE
    COURSE NAME MAX MIN OBT
    010. COMPUTER NETWORKS PP 100 40
    RESULT FIRST CLASS
    """
    strong_text = """
    UNIVERSITY OF PUNE
    B.E. EXAMINATION MAY 2006
    MALUSKAR HRISHIKESH NAMDEO
    SEAT NO B2084275
    COURSE NAME MAX MIN OBT
    010. COMPUTER NETWORKS PP 100 40 34
    010. COMPUTER NETWORKS TW 25 10 15
    010. COMPUTER NETWORKS OR 50 20 39
    011. DATABASE MANAGEMENT SYSTEMS PP 100 40 48
    011. DATABASE MANAGEMENT SYSTEMS TW 25 10 18
    011. DATABASE MANAGEMENT SYSTEMS PR 50 20 44
    1500 TOTAL EARNED 928/ RESULT : FIRST CLASS
    """

    best, comparison = de._select_best_marksheet_ocr_candidate(
        [
            ("embedded_pdf_text", weak_text),
            ("paddleocr", strong_text),
        ]
    )

    assert best is not None
    assert best["engine"] == "paddleocr"
    assert best["layout_family"] == de._LAYOUT_BE_MAX_MIN_OBT
    assert comparison[1]["score"] > comparison[0]["score"]


def test_classify_marksheet_layout_family_ssc_two_row() -> None:
    ocr_text = """
    Maharashtra State Board and of Secondary Higher Secondary Education
    MALUSKAR HRISHIKESH NAMDEO
    AI LANGUAGES
    SUBUECTS MATHS SCIENCE RESULT
    FIAST SECOND /Third SOCIAL GRAND
    SCEINCES TOTAL
    EnG MAR HIN
    100 150 150 750 MAXMUM MAAKS
    073 073 077 134 627 145 125 83 .60
    """

    layout = de._classify_marksheet_layout_family(ocr_text)

    assert layout["family"] == de._LAYOUT_TWO_ROW_SUBJECT_TABLE


def test_parse_two_row_marksheet_ocr_directly_extracts_ssc_subjects() -> None:
    ocr_text = """
    Maharashtra State Board and of Secondary Higher Secondary Education
    Pune C028223 1033 11.356 M4 Rch-1999 05 8 413
    MALUSKAR HRISHIKESH NAMDEO
    AI LANGUAGES
    SUBUECTS MATHS SCIENCE RESULT
    FIAST SECOND /Third SOCIAL GRAND
    SCEINCES TOTAL
    EnG MAR HIN
    100 150 150 750 MAXMUM MAAKS
    PERCENTAGE
    073 073 077 134 627 145 125 83 .60
    """

    parsed = de._parse_two_row_marksheet_ocr_directly(ocr_text)

    assert parsed is not None
    assert parsed["candidate_name"] == "MALUSKAR HRISHIKESH NAMDEO"
    assert parsed["month_year_of_passing"] == "March-1999"
    assert parsed["printed_grand_total"] == 627
    assert parsed["total_max_marks"] == 750
    assert parsed["percentage_or_cgpa"] == "83.60"
    assert parsed["subjects"] == [
        {"subject_name": "English", "marks_obtained": 73, "max_marks": 100},
        {"subject_name": "Marathi", "marks_obtained": 73, "max_marks": 100},
        {"subject_name": "Hindi", "marks_obtained": 77, "max_marks": 100},
        {"subject_name": "Mathematics", "marks_obtained": 134, "max_marks": 150},
        {"subject_name": "Science", "marks_obtained": 145, "max_marks": 150},
        {"subject_name": "Social Sciences", "marks_obtained": 125, "max_marks": 150},
    ]


def test_validate_educational_marksheet_keeps_generic_printed_total_gap() -> None:
    data = {
        "document_type": "Marksheet",
        "candidate_name": "MALUSKAR HRISHIKESH NAMDEO",
        "printed_grand_total": 504,
        "subjects": [
            {"subject_name": "English", "marks_obtained": 70, "max_marks": 100},
            {"subject_name": "Physics", "marks_obtained": 82, "max_marks": 100},
            {"subject_name": "Chemistry", "marks_obtained": 90, "max_marks": 100},
            {"subject_name": "Biology", "marks_obtained": 91, "max_marks": 100},
            {"subject_name": "Mathematics", "marks_obtained": 111, "max_marks": 200},
        ],
    }

    errors = de._validate_educational(data)

    assert errors == []
    assert data["computed_total"] == 444
    assert data["total_max_marks"] == 600
    assert data["extraction_confidence"] == "MEDIUM"
    assert any("Printed total note:" in note for note in data["data_quality_notes"])


def test_validate_educational_marksheet_retries_small_overread() -> None:
    data = {
        "document_type": "Marksheet",
        "candidate_name": "MALUSKAR HRISHIKESH NAMDEO",
        "printed_grand_total": 198,
        "subjects": [
            {"subject_name": "English", "marks_obtained": 100, "max_marks": 100},
            {"subject_name": "Mathematics", "marks_obtained": 100, "max_marks": 100},
        ],
    }

    errors = de._validate_educational(data)

    assert len(errors) == 1
    assert "Marks total mismatch" in errors[0]
    assert data["computed_total"] == 200
    assert data["total_max_marks"] == 200


def test_build_marksheet_structure_hint_for_two_row_layout() -> None:
    ocr_text = """
    MALUSKAR HRISHIKESH NAMDEO
    SUBUECTS MATHS SCIENCE RESULT
    FIAST SECOND /Third SOCIAL GRAND
    EnG MAR HIN
    100 150 150 750 MAXMUM MAAKS
    073 073 077 134 627 145 125 83 .60
    """

    hint = de._build_marksheet_structure_hint(ocr_text, de._LAYOUT_TWO_ROW_SUBJECT_TABLE)

    assert "layout_family: two_row_subject_table" in hint
    assert "subject_candidates" in hint
    assert "detected_printed_grand_total: 627" in hint


def test_extract_uppercase_name_skips_metadata_headers() -> None:
    lines = [
        "Statement showing the marks in each subject obtained at S.S.C. Examination",
        "NO NO SCHOOLNO QEEXAMNAION SIAEMENI",
        "Pune C028223 1033 11.356 M4 Rch-1999 05 8 413",
        "MALUSKAR HRISHIKESH NaMDEO",
    ]

    assert de._extract_uppercase_name(lines) == "MALUSKAR HRISHIKESH NAMDEO"


def test_write_marksheet_ocr_markdown_preserves_row_order(tmp_path) -> None:
    markdown_path = de._write_marksheet_ocr_markdown(
        "tests/sample/SSC-Marksheet.pdf",
        filename="SSC-Marksheet.pdf",
        markdown_text="\nEnglish Marathi Hindi\n\n100 100 100\n073 073 077\n",
        artifact_root=tmp_path,
    )

    saved_path = Path(markdown_path)

    assert saved_path.name == "SSC-Marksheet_ocr_scan.md"
    assert saved_path.read_text(encoding="utf-8") == "English Marathi Hindi\n\n100 100 100\n073 073 077\n"


def test_layout_text_from_ocr_results_preserves_relative_column_spacing() -> None:
    results = [
        ([[0, 0], [40, 0], [40, 20], [0, 20]], "SEAT", 0.99),
        ([[120, 0], [180, 0], [180, 20], [120, 20]], "NUMBER", 0.99),
        ([[0, 50], [30, 50], [30, 70], [0, 70]], "123", 0.99),
        ([[120, 50], [160, 50], [160, 70], [120, 70]], "456", 0.99),
    ]

    plain_text, markdown_text = de._layout_text_from_ocr_results(results, "paddleocr")

    assert plain_text == "SEAT NUMBER\n123 456"
    assert "SEAT        NUMBER" in markdown_text
    assert "123         456" in markdown_text


def test_extract_document_fields_marksheet_requires_paddle(monkeypatch, tmp_path) -> None:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    monkeypatch.setattr(de, "_file_to_jpeg_b64", lambda *args, **kwargs: img_b64)
    monkeypatch.setattr(de, "_paddleocr_to_text", lambda *args, **kwargs: ("", "", ""))
    monkeypatch.setattr(de, "_extract_pdf_text", lambda *args, **kwargs: "THIS MUST NOT BE USED")

    result = de.extract_document_fields(
        b"fake-image-bytes",
        doc_type="marksheet",
        filename="marksheet.png",
        artifact_root=tmp_path,
    )

    assert result["document_type"] == "marksheet"
    assert "PaddleOCR" in result["error"]
    assert result["_ocr_engine_used"] == "paddleocr"