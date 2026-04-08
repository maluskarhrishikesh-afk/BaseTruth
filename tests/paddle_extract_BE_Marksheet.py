"""
PaddleOCR extraction of BE-Marksheet.pdf â€” runs inside Docker.

Engine: RapidOCR (rapidocr-onnxruntime)
-------
PaddleOCR 3.4.0 + PaddlePaddle 3.3.1 have a hard upstream C++ bug:
  oneDNN PIR executor ConvertPirAttribute2RuntimeAttribute does not support
  pir::ArrayAttribute<pir::DoubleAttribute>.

  oneDNN is NOT a separate package â€” it is compiled INTO PaddlePaddle. It
  cannot be uninstalled independently. The bug is at the C++ layer and
  cannot be bypassed from Python (env flags, source patches and monkey-patches
  were all tried and failed because the model files load PIR ops at the
  C++ level before any Python code can intercept them).

  RapidOCR uses the exact same PP-OCRv4 models as PaddleOCR but runs inference
  via ONNX Runtime, completely bypassing PaddlePaddle. Results are identical
  in terms of model architecture and accuracy.

Usage (inside Docker):
    python /tmp/paddle_extract_BE_Marksheet.py

Output:
    /tmp/output_paddleocr_BE-Marksheet.json
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

PDF_PATH = Path("/tmp/BE-Marksheet.pdf")
OUT_PATH = Path("/tmp/output_paddleocr_BE-Marksheet.json")

# â”€â”€ Step 1: Rasterise first page of PDF â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("Rasterising PDF page 1 ...")
import fitz
from PIL import Image as PIL_Image
import numpy as np

doc = fitz.open(str(PDF_PATH))
page = doc.load_page(0)
mat = fitz.Matrix(3.0, 3.0)
pix = page.get_pixmap(matrix=mat)
pil_img = PIL_Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
doc.close()

img_path = "/tmp/be_marksheet_page1.jpg"
pil_img.save(img_path, format="JPEG", quality=95)
print(f"  Page saved: {pil_img.width}x{pil_img.height}px")

# â”€â”€ Step 2: Run RapidOCR (PP-OCRv4 models via ONNX Runtime) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\nRunning RapidOCR (PP-OCRv4 via ONNX Runtime â€” same models as PaddleOCR) ...")
from rapidocr_onnxruntime import RapidOCR

ocr = RapidOCR()
result, elapse = ocr(img_path)
total_elapse = sum(elapse) if isinstance(elapse, (list, tuple)) else float(elapse)
print(f"  Inference time: {total_elapse:.2f}s")

# result is a list of [bounding_box, text, confidence] or None
if result is None:
    result = []

print(f"  Extracted {len(result)} text regions")

# â”€â”€ Step 3: Build line list â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
lines: list[dict] = []
for item in result:
    if len(item) >= 2:
        box = item[0].tolist() if hasattr(item[0], "tolist") else item[0]
        text = str(item[1]).strip()
        conf = float(item[2]) if len(item) > 2 else 0.0
        if text:
            lines.append({"text": text, "confidence": round(conf, 4), "box": box})

full_text = "\n".join(l["text"] for l in lines)
mean_conf = round(sum(l["confidence"] for l in lines) / len(lines), 4) if lines else 0.0

print(f"\nâ”€â”€ All OCR lines (conf / text) â”€â”€")
for l in lines:
    print(f"  [{l['confidence']:.2f}] {l['text']}")

# â”€â”€ Step 4: Parse structured fields â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def find(pattern: str, text: str, default: str = "") -> str:
    m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else default


line_texts = [l["text"] for l in lines]

university = ""
for t in line_texts[:12]:
    if re.search(r"university|board|institute|sabha|vidyapeeth", t, re.I):
        university = t
        break

candidate_name = find(
    r"(?:name|student)\s*[:\-]\s*([A-Z][A-Z\s\.]{5,}?)(?:\n|seat|roll|$)", full_text
)
if not candidate_name:
    _skip_name = re.compile(
        r"UNIVERSITY|BOARD|EXAMINATION|RESULT|CLASS|TOTAL|INSTITUTE|COLLEGE|SCHOOL|STATEMENT|CERTIFICATE",
        re.I,
    )
    for t in line_texts:
        # Require ≥2 words, each ≥3 uppercase letters only (no digits/punctuation)
        parts = t.split()
        if (
            len(parts) >= 2
            and all(len(p) >= 3 and re.match(r"^[A-Z]+$", p) for p in parts)
            and not _skip_name.search(t)
        ):
            candidate_name = t
            break

seat_no = ""
for t in line_texts:
    if re.match(r"^[A-Z]\d{4,}$", t):  # e.g. B2084275 — letter followed by digits
        seat_no = t
        break
if not seat_no:
    # Fallback: label and value on same text segment
    seat_no = find(r"(?:seat\s*no|roll\s*no|seat\.?\s*number)[:\s\.\-]*([A-Z0-9]{5,})", full_text)
year    = find(r"((?:MAY|OCTOBER|NOVEMBER|MARCH|JUNE|JULY|APRIL|AUGUST)\s+\d{4})", full_text)
if not year:
    # Fallback: OCR may merge month+year with no space or garble digits (e.g. "MAYZOO6")
    for _t in line_texts:
        _m = re.search(r"(MAY|OCTOBER|NOVEMBER|MARCH|JUNE|JULY|APRIL|AUGUST)\s*(\w+)", _t, re.I)
        if _m:
            year = f"{_m.group(1).upper()} {_m.group(2)}"
            break
exam    = find(r"(B\.\s*[A-Z][\w\s\(\)\.]+EXAMINATION)", full_text)
result_val = find(r"\b(FIRST\s+CLASS\s+WITH\s+DISTINCTION|FIRST\s+CLASS|SECOND\s+CLASS|DISTINCTION|PASS|FAIL)\b", full_text)
if result_val:
    result_val = " ".join(result_val.split())  # normalise any embedded newlines
grand   = find(r"(?:grand\s*total|total)[:\s=]*(\d{3,4}(?:/\d{3,4})?)", full_text)
if not grand:
    # Fallback: OCR may merge grand total with adjacent text, e.g. "925/1500RESULT:"
    grand = find(r"(\d{3,4}/\d{3,4})", full_text)

def parse_marks_table(text_lines: list[str]) -> list[dict]:
    subjects = []
    num_re   = re.compile(r"^\d{2,3}$")
    frac_re  = re.compile(r"^(\d{2,3})/(\d{2,3})$")
    skip_words = re.compile(
        r"UNIVERSITY|BOARD|EXAMINATION|RESULT|CLASS|TOTAL|INSTITUTE|COLLEGE|SCHOOL|MAHAR|PUNE",
        re.I,
    )
    subject_re = re.compile(r"^[A-Z][A-Z\s\.\(\)\-\&\/\:]{5,}$")

    i = 0
    while i < len(text_lines):
        t = text_lines[i]
        if subject_re.match(t) and not skip_words.search(t):
            marks, max_m = [], None
            for j in range(i + 1, min(i + 6, len(text_lines))):
                c = text_lines[j].strip()
                if num_re.match(c):
                    marks.append(int(c))
                elif frac_re.match(c):
                    p = c.split("/")
                    marks.append(int(p[0]))
                    max_m = int(p[1])
            if marks:
                subjects.append({
                    "subject_name": t,
                    "marks_obtained": marks[0],
                    "max_marks": max_m if max_m else (marks[1] if len(marks) > 1 else None),
                    "grade": None,
                })
        i += 1
    return subjects


parsed_subjects = parse_marks_table(line_texts)

# â”€â”€ Step 5: Build output JSON â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
output = {
    "method": "rapidocr_onnxruntime_pp-ocrv4",
    "engine_note": "PP-OCRv4 models via ONNX Runtime (same models as PaddleOCR, bypasses PaddlePaddle oneDNN PIR bug)",
    "document_type": "Marksheet",
    "candidate_name": candidate_name or None,
    "board_or_university_name": university or None,
    "examination_name": exam or None,
    "enrollment_or_seat_number": seat_no or None,
    "year_or_date_of_passing": year or None,
    "subjects": parsed_subjects,
    "computed_total": sum(
        s["marks_obtained"] for s in parsed_subjects
        if isinstance(s.get("marks_obtained"), int)
    ),
    "printed_grand_total": grand or None,
    "result": result_val or None,
    "ocr_stats": {
        "total_lines": len(lines),
        "mean_confidence": mean_conf,
        "chars_extracted": len(full_text),
        "inference_time_s": round(total_elapse, 3),
    },
    "raw_text": full_text,
    "raw_lines": lines,
}

OUT_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\nâœ“ Written â†’ {OUT_PATH}")

print("\nâ”€â”€ Quick Summary â”€â”€")
print(f"  Method    : RapidOCR (PP-OCRv4 via ONNX Runtime)")
print(f"  Candidate : {output['candidate_name']}")
print(f"  Seat No   : {output['enrollment_or_seat_number']}")
print(f"  University: {output['board_or_university_name']}")
print(f"  Year      : {output['year_or_date_of_passing']}")
print(f"  Exam      : {output['examination_name']}")
print(f"  Result    : {output['result']}")
print(f"  GrandTotal: {output['printed_grand_total']}")
print(f"  Subjects  : {len(parsed_subjects)} parsed")
print(f"  OCR lines : {len(lines)}, mean conf: {mean_conf}")
