"""
Marksheet extraction v2 — improved Gemma4 reasoning + image quality gate.

Changes vs v1:
  • Two-stage pipeline:
      Stage 1 — Image quality assessment: ask Gemma4 to rate readability.
                 If quality is too low, return a clear user-facing warning and stop.
      Stage 2 — Marksheet-specific extraction using a domain-aware prompt that:
                 - Knows Indian board marks are ALWAYS integers (never decimals)
                 - Re-reads each digit cell independently
                 - Cross-checks totals
  • Higher resolution sent to Gemma4 (max_dim=2048) to preserve digit detail.
  • Writes output to tests/output_marksheet_v2.json

Run from BaseTruth root:
    python tests/extract_marksheet_v2.py
"""
from __future__ import annotations

import base64
import io
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

TESTS_DIR = Path(__file__).resolve().parent
FILES = {
    "Psychology_Marksheet.jpg": TESTS_DIR / "Psychology_Marksheet.jpg",
    "SSC-Marksheet.pdf": TESTS_DIR / "SSC-Marksheet.pdf",
}

# ---------------------------------------------------------------------------
# Ollama helpers (inline, no dependency on basetruth.integrations)
# ---------------------------------------------------------------------------

OLLAMA_BASE = "http://localhost:11434"
OLLAMA_MODEL = "gemma4:latest"
CONNECT_TIMEOUT = 5
READ_TIMEOUT = 600


def _ollama_chat(messages: list, *, temperature: float = 0) -> str:
    """Send a chat request to Ollama and return the text content."""
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    resp = requests.post(
        f"{OLLAMA_BASE}/api/chat",
        json=payload,
        timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
    )  # nosemgrep: basetruth-ssrf
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


def _image_to_b64(path: Path, max_dim: int = 2048) -> Optional[str]:
    """Resize to max_dim and return base64-encoded JPEG string."""
    try:
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
    except Exception as exc:
        print(f"    ⚠  Could not load image {path.name}: {exc}")
        return None


def _pdf_first_page_b64(path: Path, max_dim: int = 2048) -> Optional[str]:
    """Rasterise first PDF page at high DPI and return base64 JPEG."""
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(str(path))
        page = doc.load_page(0)
        # 3× zoom → ~216 dpi — enough to read small print in mark tables
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
    except Exception as exc:
        print(f"    ⚠  PDF rasterisation failed for {path.name}: {exc}")
        return None


def get_image_b64(path: Path) -> Optional[str]:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}:
        return _image_to_b64(path)
    if suffix == ".pdf":
        return _pdf_first_page_b64(path)
    return None


# ---------------------------------------------------------------------------
# Stage 1 — Image quality assessment
# ---------------------------------------------------------------------------

IMAGE_QUALITY_PROMPT = """
Look at this document image carefully.

Assess its readability and return a JSON object:

{
  "quality_score": 0,
  "is_readable": true,
  "issues": [],
  "readable_regions": [],
  "unreadable_regions": [],
  "recommendation": ""
}

Rules:
- quality_score: integer 1 (completely unreadable) to 5 (crystal clear, every character legible).
- is_readable: true if you can reliably read AT LEAST 70% of the text content.
- issues: list of specific problems found (e.g. "blurry", "low resolution", "skewed",
  "shadow obscuring text", "glare", "torn/damaged", "overexposed", "dark background").
- readable_regions: list of regions where text IS legible (e.g. "header", "name field",
  "marks table top rows").
- unreadable_regions: list of regions where text CANNOT be reliably read.
- recommendation: if is_readable is false, write a clear user-facing instruction like
  "Please re-upload a higher resolution scan. The marks table is too blurry to read."
- Output strict JSON only.
""".strip()


def assess_image_quality(img_b64: str, filename: str) -> Dict[str, Any]:
    print(f"    Stage 1 — Quality assessment: {filename}")
    try:
        content = _ollama_chat(
            [
                {
                    "role": "user",
                    "content": IMAGE_QUALITY_PROMPT,
                    "images": [img_b64],
                }
            ]
        )
        json_text = _extract_json(content)
        if json_text:
            result = json.loads(json_text)
            result["_raw"] = content
            return result
        return {"quality_score": 0, "is_readable": False,
                "recommendation": "Quality assessment failed — Gemma4 returned no structured response.",
                "_raw": content}
    except Exception as exc:
        return {"quality_score": 0, "is_readable": False,
                "recommendation": f"Quality assessment failed: {exc}", "_raw": ""}


# ---------------------------------------------------------------------------
# Stage 2 — Marksheet-specific extraction with domain-aware reasoning
# ---------------------------------------------------------------------------

MARKSHEET_EXTRACTION_SYSTEM_PROMPT = (
    "You are an expert in reading Indian school and university mark sheets / transcripts. "
    "You have deep knowledge of Indian board examination formats (CBSE, Maharashtra State Board, "
    "ICSE, PSEB, RBSE, etc.). Return strict JSON only."
)

MARKSHEET_EXTRACTION_PROMPT = """
Extract ALL information from this mark sheet image.

CRITICAL RULES for reading marks (these MUST be followed):
1. Marks on Indian board mark sheets are ALWAYS whole integers — NEVER decimals.
   Examples of correct reading: 73, 103, 134, 85, 100.
   Examples of mistakes to AVOID: 13.4 ❌, 11.356 ❌, 1.34 ❌, 0.73 ❌.

2. When you see digits that look like they could be split across columns or lines,
   treat them as ONE integer. E.g., if you see "13" and "4" next to each other in
   a marks cell, the value is 134, not 13.4 or 13 and 4 separately.

3. Read each subject's marks cell INDEPENDENTLY. Zoom in mentally on each cell and
   count the digit characters. Typical mark ranges:
   - Out of 100: values 0–100
   - Out of 150: values 0–150
   - Out of 200: values 0–200
   Never report a mark like "11.35" — that is a misread of "113" or "135".

4. Verify totals: if you can compute the sum of all subject marks, check it against
   the grand total printed on the sheet. Report any mismatch.

5. Read the candidate's name EXACTLY as printed, character by character.

Return a JSON object with this structure:
{
  "document_type": "",
  "board_name": "",
  "examination_name": "",
  "examination_year": "",
  "candidate_name": "",
  "roll_number": "",
  "seat_number": "",
  "school_name": "",
  "subjects": [
    {
      "subject_name": "",
      "max_marks": null,
      "marks_obtained": null,
      "grade": "",
      "remarks": ""
    }
  ],
  "computed_total": null,
  "printed_grand_total": null,
  "total_max_marks": null,
  "percentage": "",
  "result": "",
  "division": "",
  "merit_distinction": "",
  "additional_fields": {},
  "data_quality_notes": [],
  "reasoning_steps": []
}

Rules:
- subjects: one entry per subject row found.
- marks_obtained: INTEGER (or null if unreadable). NEVER a decimal.
- max_marks: INTEGER per subject (or null if not printed).
- computed_total: your own sum of all marks_obtained (cross-check).
- data_quality_notes: list any cells where you were uncertain about
  the digit(s) and explain why, e.g.:
  "Subject MATHS: cell showed '13' in one column and '4' in the next —
   interpreted as 134 (out of 150)".
- reasoning_steps: walk through your logic for any ambiguous digit readings.
- Output strict JSON only — no markdown, no extra text.
""".strip()


def extract_marksheet(img_b64: str, filename: str) -> Dict[str, Any]:
    print(f"    Stage 2 — Marksheet extraction: {filename}")
    try:
        content = _ollama_chat(
            [
                {"role": "system", "content": MARKSHEET_EXTRACTION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": MARKSHEET_EXTRACTION_PROMPT,
                    "images": [img_b64],
                },
            ]
        )
        json_text = _extract_json(content)
        if json_text:
            result = json.loads(json_text)
            result["_raw_response"] = content
            return result
        return {"error": "Gemma4 returned no structured JSON", "_raw_response": content}
    except Exception as exc:
        return {"error": str(exc), "_raw_response": ""}


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def process_file(label: str, path: Path) -> Dict[str, Any]:
    print(f"\n  ── {label} ──")
    if not path.exists():
        return {"error": f"File not found: {path}"}

    img_b64 = get_image_b64(path)
    if not img_b64:
        return {"error": "Could not produce image for Gemma4 (PIL/PyMuPDF unavailable?)"}

    # Check Ollama is reachable
    try:
        requests.get(f"{OLLAMA_BASE}/api/tags", timeout=CONNECT_TIMEOUT)
    except Exception:
        return {"error": f"Ollama not reachable at {OLLAMA_BASE}. Start Ollama and retry."}

    # ── Stage 1: quality gate ────────────────────────────────────────────────
    quality = assess_image_quality(img_b64, label)
    score = quality.get("quality_score", 0)
    is_readable = quality.get("is_readable", False)

    print(f"    Quality score: {score}/5 | Readable: {is_readable}")
    if quality.get("issues"):
        print(f"    Issues: {', '.join(quality['issues'])}")

    if not is_readable or score < 3:
        recommendation = quality.get("recommendation") or (
            "Image quality is too poor to extract data reliably. "
            "Please re-upload a clearer, higher-resolution scan."
        )
        return {
            "quality_assessment": quality,
            "extraction": None,
            "user_message": f"⚠️  Cannot extract data — {recommendation}",
        }

    # ── Stage 2: marksheet extraction ───────────────────────────────────────
    extraction = extract_marksheet(img_b64, label)

    # Post-process: validate that marks_obtained are integers
    subjects = extraction.get("subjects") or []
    for subj in subjects:
        m = subj.get("marks_obtained")
        if m is not None:
            try:
                subj["marks_obtained"] = int(round(float(m)))
            except (TypeError, ValueError):
                pass

    # Cross-check computed vs printed total
    notes = extraction.setdefault("data_quality_notes", [])
    computed = extraction.get("computed_total")
    printed = extraction.get("printed_grand_total")
    if computed is not None and printed is not None:
        try:
            computed_int = int(round(float(computed)))
            printed_int = int(round(float(printed)))
            if computed_int != printed_int:
                notes.append(
                    f"Total mismatch: sum of extracted marks ({computed_int}) "
                    f"≠ printed grand total ({printed_int}). "
                    "One or more marks may have been misread."
                )
                extraction["total_verified"] = False
            else:
                extraction["total_verified"] = True
        except (TypeError, ValueError):
            pass

    return {
        "quality_assessment": quality,
        "extraction": extraction,
        "user_message": "✅ Extraction complete.",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results: Dict[str, Any] = {}

    for label, path in FILES.items():
        results[label] = process_file(label, path)

    out = TESTS_DIR / "output_marksheet_v2.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  ✓  Written → {out}")

    # Print a quick summary
    print("\n── Summary ──")
    for label, result in results.items():
        msg = result.get("user_message", "")
        extraction = result.get("extraction") or {}
        name = extraction.get("candidate_name", "")
        total = extraction.get("printed_grand_total", "")
        verified = extraction.get("total_verified")
        subjects = extraction.get("subjects") or []
        print(f"\n{label}:")
        print(f"  {msg}")
        if name:
            print(f"  Candidate : {name}")
        if total:
            print(f"  Grand total: {total}")
        if verified is not None:
            print(f"  Total verified: {verified}")
        for s in subjects:
            print(f"  {s.get('subject_name','?'):25s}  {s.get('marks_obtained','?')} / {s.get('max_marks','?')}")
