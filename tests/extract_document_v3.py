"""
Document extraction v3 — Supports Marksheets and Degrees with a Self-Correction (Retry) Loop.

Features:
- Extracts fields for Aadhar, PAN, Marksheets, Degrees, etc. (focusing on Marksheets & Degrees here).
- If validation fails (e.g., total marks mismatch, missing candidate name), it prompts Gemma again
  in the same chat thread to look closer at the image and correct the errors.
"""
from __future__ import annotations

import base64
import io
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional, List

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

TESTS_DIR = Path(__file__).resolve().parent
FILES = {
    "Psychology_Marksheet.jpg": TESTS_DIR / "Psychology_Marksheet.jpg",
    "SSC-Marksheet.pdf": TESTS_DIR / "SSC-Marksheet.pdf",
    "Psychology_Degree.pdf": TESTS_DIR / "Psychology_Degree.pdf",
    "BE-Degree.pdf": TESTS_DIR / "BE-Degree.pdf",
    "BE-Marksheet.pdf": TESTS_DIR / "BE-Marksheet.pdf",
}

OLLAMA_BASE = "http://localhost:11434"
OLLAMA_MODEL = "gemma4:latest"
CONNECT_TIMEOUT = 5
READ_TIMEOUT = 600
MAX_RETRIES = 1  # 2 total attempts (initial + 1 retry)

def _ollama_chat(messages: list, *, temperature: float = 0) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    resp = requests.post(f"{OLLAMA_BASE}/api/chat", json=payload, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
    resp.raise_for_status()
    return str(resp.json().get("message", {}).get("content", "")).strip()

def _extract_json(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1:
        return ""
    return stripped[start : end + 1]

def get_image_b64(path: Path, max_dim: int = 2048) -> Optional[str]:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp"}:
        from PIL import Image as PIL
        with PIL.open(str(path)) as img:
            img = img.convert("RGB")
            w, h = img.size
            if max(w, h) > max_dim:
                scale = max_dim / max(w, h)
                img = img.resize((int(w * scale), int(h * scale)), PIL.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=92)
            return base64.b64encode(buf.getvalue()).decode("ascii")
    elif suffix == ".pdf":
        import fitz
        doc = fitz.open(str(path))
        page = doc.load_page(0)
        mat = fitz.Matrix(3.0, 3.0)
        pix = page.get_pixmap(matrix=mat)
        from PIL import Image as PIL
        img = PIL.frombytes("RGB", (pix.width, pix.height), pix.samples)
        doc.close()
        w, h = img.size
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), PIL.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=92)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    return None

DOCUMENT_EXTRACTION_SYSTEM_PROMPT = (
    "You are an expert OCR AI specialized in Indian educational documents, "
    "like Board Marksheets and University Degrees. You extract data precisely."
)

DOCUMENT_EXTRACTION_PROMPT = """
Analyze the provided educational document image and extract its contents into a strict JSON object.

IMAGE QUALITY CHECK (do this FIRST):
Before extracting any data, assess whether the image is clear enough to read.
If the image is too blurry, dark, low-resolution, heavily rotated, or otherwise unreadable such that you cannot reliably identify the text, output ONLY this JSON and nothing else:
{
  "document_type": "Unreadable",
  "message": "The document image is not clear enough to extract data reliably. Please upload a higher-quality scan or photograph with better lighting, focus, and orientation."
}

DETERMINE THE DOCUMENT TYPE FIRST: 
Choose one of: "Marksheet", "Degree Certificate", or "Unknown".

To support robust data extraction, we need general identity fields and document-specific ones.

For Marksheets:
- Identity & Institution: candidate_name, board_or_university_name, school_or_college_name, examination_name
- Identifiers: enrollment_or_seat_number, year_or_date_of_passing
- Scores Table (List): subject_name, marks_obtained, max_marks, grade (if applicable)
- Score Aggregates: printed_grand_total, computed_total (your internal check digit), total_max_marks, percentage_or_cgpa, result (Pass/Fail)

For Degrees:
- Identity & Institution: candidate_name, university_name
- Core Details: degree_name (e.g. B.Tech), specialization_or_major (e.g. Psychology), division_or_class (e.g. First Class), year_or_date_of_passing
- Identifiers: enrollment_number / seat_number, certificate_number, date_of_issue

CRITICAL RULES FOR MARKSHEETS:
1. Marks are ALWAYS whole integers (e.g. 73, 100), NEVER decimals like 13.4. If you see spaced digits like "13" and "4", it is "134".
2. You must read each subject's mark independently. 
3. Verify totals. Add all marks_obtained across subjects. If it doesn't match the printed_grand_total, you probably misread a digit. Check carefully.

EXPECTED JSON STRUCTURE:
If the document is a "Marksheet", output THIS exact JSON structure:
{
  "document_type": "Marksheet",
  "candidate_name": "",
  "board_or_university_name": "",
  "school_or_college_name": "",
  "examination_name": "",
  "enrollment_or_seat_number": "",
  "year_or_date_of_passing": "",
  "subjects": [
    {
      "subject_name": "",
      "max_marks": null,
      "marks_obtained": null,
      "grade": ""
    }
  ],
  "computed_total": null,
  "printed_grand_total": null,
  "percentage_or_cgpa": "",
  "result": "",
  "data_quality_notes": [],
  "reasoning_steps": []
}

If the document is a "Degree Certificate", output THIS exact JSON structure:
{
  "document_type": "Degree Certificate",
  "candidate_name": "",
  "university_name": "",
  "degree_name": "",
  "specialization_or_major": "",
  "division_or_class": "",
  "year_or_date_of_passing": "",
  "enrollment_number": "",
  "certificate_number": "",
  "date_of_issue": "",
  "data_quality_notes": [],
  "reasoning_steps": []
}

Output ONLY strict JSON.
"""

def validate_extraction(data: Dict[str, Any]) -> List[str]:
    """Return a list of validation errors/warnings."""
    errors = []
    
    doc_type = data.get("document_type", "")

    # Short-circuit: model flagged the image as unreadable — no further validation needed.
    if doc_type == "Unreadable":
        return errors

    if not data.get("candidate_name"):
        errors.append("Candidate name is missing. Look carefully, it is usually prominent.")
    
    if doc_type == "Marksheet":
        if not data.get("board_or_university_name"):
            errors.append("Board or university name is missing. It is usually at the top.")
            
        subjects = data.get("subjects", [])
        if not subjects:
            errors.append("No subjects found in a marksheet. Look for the marks table.")
        else:
            computed = 0
            for s in subjects:
                mo = s.get("marks_obtained")
                if isinstance(mo, (int, float)):
                    computed += int(mo)
                else:
                    try:
                        computed += int(float(mo))
                        s["marks_obtained"] = int(float(mo)) # auto-fix to int
                    except (ValueError, TypeError):
                        pass
            
            # Populate our own computed total for the check
            data["computed_total"] = computed
            printed = data.get("printed_grand_total")
            
            if printed is not None:
                try:
                    printed_int = int(float(printed))
                    data["printed_grand_total"] = printed_int
                    if computed != printed_int:
                        errors.append(f"Total mismatch: your sum of subject marks is {computed}, but printed_grand_total is {printed_int}. You misread one or more subject marks. Please zoom in on the marks table and correct the values.")
                except (ValueError, TypeError):
                    errors.append("printed_grand_total is not a valid number.")
                    
    elif doc_type == "Degree Certificate":
        if not data.get("university_name"):
            errors.append("University name is missing. It is usually at the top.")
            
        if not data.get("specialization_or_major"):
            errors.append("Degree specialization/branch (e.g. Computer Science, Psychology) seems missing. Please re-read the certificate text.")
            
    return errors

def extract_with_retry(img_b64: str, filename: str) -> Dict[str, Any]:
    print(f"    Extracting data for: {filename}")
    
    messages = [
        {"role": "system", "content": DOCUMENT_EXTRACTION_SYSTEM_PROMPT},
        {"role": "user", "content": DOCUMENT_EXTRACTION_PROMPT, "images": [img_b64]}
    ]
    
    for attempt in range(MAX_RETRIES + 1):
        print(f"      Attempt {attempt + 1}/{MAX_RETRIES + 1}...")
        try:
            content = _ollama_chat(messages, temperature=0.1)
            json_text = _extract_json(content)
            
            if not json_text:
                raise ValueError("No JSON found in response.")
                
            data = json.loads(json_text)
            
            # Validate
            errors = validate_extraction(data)
            
            if not errors:
                print("      ✓ Validation passed!")
                data["_extraction_attempts"] = attempt + 1
                return data
                
            if attempt < MAX_RETRIES:
                print(f"      ⚠ Validation failed: {errors}. Retrying...")
                # Append assistant's response and validation feedback
                messages.append({"role": "assistant", "content": content})
                messages.append({
                    "role": "user",
                    "content": f"Your extraction had the following issues:\n{json.dumps(errors, indent=2)}\n\nPlease re-examine the image carefully and correct these mistakes. Return the FULL updated JSON."
                })
            else:
                print(f"      ✗ Validation failed on final attempt: {errors}")
                data["_validation_errors"] = errors
                data["_extraction_attempts"] = attempt + 1
                return data
                
        except Exception as exc:
            print(f"      ⚠ Error: {exc}")
            if attempt == MAX_RETRIES:
                return {"error": str(exc), "_extraction_attempts": attempt + 1}
            messages.append({"role": "user", "content": f"Error parsing JSON: {exc}. Provide ONLY valid JSON."})

    return {"error": "Failed after max retries"}

def process_file(label: str, path: Path) -> Dict[str, Any]:
    print(f"\n  ── {label} ──")
    if not path.exists():
        return {"error": f"File not found: {path}"}

    img_b64 = get_image_b64(path)
    if not img_b64:
        return {"error": "Could not produce image for Gemma4"}

    try:
        requests.get(f"{OLLAMA_BASE}/api/tags", timeout=CONNECT_TIMEOUT)
    except Exception:
        return {"error": f"Ollama not reachable at {OLLAMA_BASE}."}

    extraction = extract_with_retry(img_b64, label)
    return extraction

if __name__ == "__main__":
    # ── Run only for BE-Marksheet.pdf ──
    TARGET = "BE-Marksheet.pdf"
    target_path = FILES[TARGET]

    result = process_file(TARGET, target_path)

    # Write output file
    stem = Path(TARGET).stem
    out_path = TESTS_DIR / f"output_v3_{stem}.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  ✓  Written → {out_path}")

    # ── Summary ──
    print("\n── Summary ──")
    if "error" in result:
        print(f"  Error: {result['error']}")
    else:
        doc_type = result.get("document_type", "?")
        print(f"  Document Type: {doc_type}")

        if doc_type == "Unreadable":
            print(f"  ⚠  {result.get('message', 'Document is unreadable.')}")
            print("  → Please upload a higher-quality scan or photograph with better lighting and focus.")
        else:
            name = result.get("candidate_name", "")
            print(f"  Candidate: {name}")

            if doc_type == "Marksheet":
                univ = result.get("board_or_university_name", "")
                print(f"  Inst/BRD : {univ}")
                print(f"  Total    : {result.get('computed_total')} / {result.get('printed_grand_total')}")
                for s in result.get("subjects", []):
                    print(f"    {s.get('subject_name','?'):30s}  {s.get('marks_obtained','?')} / {s.get('max_marks','?')}")
            elif doc_type == "Degree Certificate":
                univ = result.get("university_name", "")
                print(f"  Inst/BRD : {univ}")
                print(f"  Degree   : {result.get('degree_name', '?')}")
                print(f"  Major    : {result.get('specialization_or_major', '?')}")
                print(f"  Class    : {result.get('division_or_class', '?')}")

            if "_validation_errors" in result:
                print(f"  Final Errors: {result['_validation_errors']}")
