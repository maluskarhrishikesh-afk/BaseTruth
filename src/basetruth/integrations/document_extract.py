"""Document field extraction using Gemma4 via Ollama.

This module provides a single public function, extract_document_fields(), that
accepts any supported document file (image or PDF) and returns a structured
dictionary of fields extracted from it by Gemma4.

Supported document categories
──────────────────────────────
Educational : marksheet, degree_certificate
Financial   : payslip, bank_statement, form16, increment_letter, gift_letter
HR/Employment: offer_letter, employment_letter
Generic      : any other document type — extracts what it can

How it works
────────────
1. Convert the input file to a JPEG image (page 1 for PDFs) so Gemma4 can read it.
2. Choose the right extraction prompt for the document category.
3. Send the image + prompt to Gemma4 running locally via Ollama.
4. Parse the JSON reply.  If validation fails, send a correction request and try once more.
5. Return the extracted dict.  If Ollama is offline, return a minimal dict with
   '_unavailable: True' so callers can handle gracefully.
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import statistics
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import requests

from basetruth.logger import get_logger
from basetruth.integrations.ollama import (
    probe_ollama,
    select_ollama_model,
    OLLAMA_CONNECT_TIMEOUT_SEC,
    OLLAMA_READ_TIMEOUT_SEC,
)

log = get_logger(__name__)

# ── Document-extract-specific timeouts ─────────────────────────────────────
# We reuse the shared OLLAMA_CONNECT_TIMEOUT_SEC (5 s) from ollama.py.
# The read timeout re-uses OLLAMA_READ_TIMEOUT_SEC (600 s) because Gemma4
# can be slow on long documents. Both are already imported above.
MAX_RETRIES = 1         # one initial attempt + one self-correction retry

# ── Document type to category mapping ──────────────────────────────────────
# Maps concrete document types to the broad category used to select the prompt.
_DOC_TYPE_CATEGORY: Dict[str, str] = {
    "marksheet":          "educational",
    "degree_certificate": "educational",
    "payslip":            "financial",
    "bank_statement":     "financial",
    "form16":             "financial",
    "increment_letter":   "financial",    # salary increment — treated as financial
    "gift_letter":        "financial",
    "offer_letter":       "employment",
    "employment_letter":  "employment",
}


# ── Extraction prompts ──────────────────────────────────────────────────────
# Prompt and rule text lives in a sibling markdown asset so prompt tuning stays
# readable and does not overwhelm the Python implementation.

_PROMPTS_MARKDOWN_PATH = Path(__file__).with_name("document_extract_prompts.md")
_PROMPT_SECTION_PATTERN = re.compile(
    r"(?ms)^##\s+(?P<name>[a-z0-9_]+)\s*\n```(?:text|prompt)?\n(?P<body>.*?)\n```"
)
_REQUIRED_PROMPT_SECTIONS = frozenset({
    "system",
    "marksheet",
    "educational",
    "financial",
    "employment",
    "generic",
})


@lru_cache(maxsize=1)
def _load_prompt_sections() -> Dict[str, str]:
    """Load all prompt blocks from the markdown asset exactly once per process.

    Keeping prompts in markdown makes the rules easier to review and edit, but
    the runtime still needs predictable named sections. This loader enforces
    that contract and fails loudly if the prompt asset is incomplete.
    """
    try:
        raw_markdown = _PROMPTS_MARKDOWN_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(
            f"Prompt asset is missing or unreadable: {_PROMPTS_MARKDOWN_PATH}"
        ) from exc

    sections = {
        match.group("name"): match.group("body").strip("\n")
        for match in _PROMPT_SECTION_PATTERN.finditer(raw_markdown)
    }
    missing = sorted(_REQUIRED_PROMPT_SECTIONS - set(sections))
    if missing:
        raise RuntimeError(
            "Prompt asset is missing required sections: " + ", ".join(missing)
        )
    return sections


def _get_prompt(name: str) -> str:
    """Return one named prompt section from the markdown asset."""
    prompts = _load_prompt_sections()
    if name not in prompts:
        raise RuntimeError(f"Unknown prompt section requested: {name}")
    return prompts[name]

# Section header words that appear on Indian marksheets as GROUP LABELS, not as
# individual subject names.  If a model tries to create a subject row with one
# of these names it means it is reading section headers instead of data rows.
_MARKSHEET_SECTION_HEADERS: frozenset = frozenset({
    "languages", "compulsory", "optional", "elective",
    "sciences", "social sciences", "social science",
    "vocational", "additional", "part i", "part ii", "part iii",
    "group a", "group b", "group c",
    "theory", "practical", "oral", "total",
})
_CATEGORY_PROMPT_NAMES: Dict[str, str] = {
        "educational": "educational",
        "financial": "financial",
        "employment": "employment",
        "generic": "generic",
}


# ── Internal helpers ────────────────────────────────────────────────────────

def _file_to_jpeg_b64(source: Union[str, Path, bytes], filename: str = "", max_dim: int = 2048) -> Optional[str]:
    """Convert a file (image or PDF page 1) to a JPEG base64 string for Gemma4.

    Handles: JPEG, PNG, WebP, BMP, TIFF, and PDF (page 1 only).
    Returns None if the file cannot be converted.
    """
    from PIL import Image as PIL_Image  # imported here to avoid heavy top-level startup cost

    suffix = Path(filename).suffix.lower() if filename else ""

    # If source is bytes, determine type from filename suffix or try PDF first
    if isinstance(source, (str, Path)):
        path = Path(source)
        suffix = path.suffix.lower()
        try:
            data = path.read_bytes()
        except OSError as exc:
            log.warning("document_extract: cannot read file %s: %s", source, exc)
            return None
    else:
        # source is raw bytes — read suffix from filename
        data = source

    try:
        if suffix == ".pdf" or (not suffix and data[:4] == b"%PDF"):
            # Render page 1 of the PDF at 3× resolution for good readability
            import fitz  # PyMuPDF — only imported if we actually have a PDF
            doc = fitz.open(stream=data, filetype="pdf")
            page = doc.load_page(0)
            mat = fitz.Matrix(3.0, 3.0)  # 3× zoom so small text is readable
            pix = page.get_pixmap(matrix=mat)
            img = PIL_Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            doc.close()
        else:
            # Regular image file — open directly
            img = PIL_Image.open(io.BytesIO(data)).convert("RGB")

        # Resize if the image would be too large to send to Gemma4
        w, h = img.size
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), PIL_Image.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=92)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    except Exception as exc:
        log.warning("document_extract: image conversion failed: %s", exc)
        return None


def _extract_pdf_text(source: Union[str, Path, bytes], filename: str = "") -> str:
    """Extract the raw embedded text from a digitally generated PDF.

    This is the key insight from ChatGPT's methodology: if a PDF was produced
    by software (not scanned), it already has clean, machine-readable text
    embedded in it.  We can extract that text directly with PyMuPDF, which
    gives us perfectly accurate characters and numbers — no vision required.

    We then pass this raw text into Gemma4 alongside the image so the model
    has two inputs:
      - The image  → shows the visual layout (which column is Earnings vs. Deductions)
      - The text   → gives accurate numbers without OCR guess-work

    For scanned PDFs (image-only, no embedded text) this returns "" and we
    fall back to image-only extraction as before.
    """
    try:
        import fitz  # PyMuPDF

        # Load raw bytes so we can open the PDF
        if isinstance(source, (str, Path)):
            data = Path(source).read_bytes()
        else:
            data = source

        suffix = Path(filename).suffix.lower() if filename else ""
        if suffix != ".pdf" and data[:4] != b"%PDF":
            return ""  # not a PDF

        doc = fitz.open(stream=data, filetype="pdf")
        # Most payslips are a single page; we read up to 3 pages to be safe
        pages_text: list[str] = []
        for page_num in range(min(len(doc), 3)):
            page = doc.load_page(page_num)
            pages_text.append(page.get_text("text"))
        doc.close()

        raw_text = "\n".join(pages_text).strip()
        # Return only if substantial text was found (>200 chars signals a
        # text-based PDF, not an image scan where we'd get mostly whitespace)
        return raw_text if len(raw_text) > 200 else ""

    except Exception as exc:
        log.debug("document_extract: PDF text extraction skipped: %s", exc)
        return ""


# PaddleOCR reader is kept as a module-level singleton so the model weights are
# only loaded once per process. Loading on first use keeps startup time fast.
_paddleocr_reader: Any = None

_LAYOUT_HSC_TRANSPOSED = "hsc_transposed"
_LAYOUT_BE_MAX_MIN_OBT = "be_max_min_obt"
_LAYOUT_TWO_ROW_SUBJECT_TABLE = "two_row_subject_table"
_LAYOUT_GENERIC_ROW_TABLE = "generic_row_table"
_LAYOUT_IGNOU_OPEN_UNIVERSITY = "ignou_open_university"  # IGNOU multi-term, multi-component marksheets
_LAYOUT_UNKNOWN = "unknown"

_BE_COMPONENT_CODES: frozenset[str] = frozenset({"PP", "TW", "OR", "PR"})
_DEFAULT_TWO_ROW_MAX_VALUES: tuple[int, ...] = (50, 75, 80, 100, 150, 200)

# These aliases cover the common abbreviated subject labels that appear on
# Indian marksheets.  OCR often breaks them into short fragments, so we map the
# fragments back to their canonical display names before any parser tries to use
# them.  Keeping this table here makes it easy to extend the document
# intelligence pipeline as new boards are added.
_SUBJECT_ALIAS_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bENGLISH\b|\bENG\b", "English"),
    (r"\bMARATHI\b|\bMAR\b", "Marathi"),
    (r"\bHINDI\b|\bHIN\b", "Hindi"),
    (r"\bSANSKRIT\b|\bSAN\b", "Sanskrit"),
    (r"\bMATHS\b|\bMATH\b|\bMATHEMATICS\b", "Mathematics"),
    (r"\bSCIENCE\b|\bSCIENCES\b", "Science"),
    (r"\bSOCIAL\s+SCIENCES\b|\bSOCIAL\s+SCIENCE\b|\bSOCIAL\s+SCEINCES\b", "Social Sciences"),
    (r"\bHISTORY\b", "History"),
    (r"\bGEOGRAPHY\b", "Geography"),
    (r"\bCIVICS\b", "Civics"),
    (r"\bPOLITICAL\s+SCIENCE\b", "Political Science"),
    (r"\bECONOMICS\b", "Economics"),
    (r"\bPHYSICS\b", "Physics"),
    (r"\bCHEMISTRY\b", "Chemistry"),
    (r"\bBIOLOGY\b", "Biology"),
    (r"\bCOMPUTER\s+SCIENCE\b|\bCOMPUTER\b", "Computer Science"),
    (r"\bENVIRONMENTAL\s+SCIENCE\b", "Environmental Science"),
    (r"\bACCOUNTANCY\b|\bACCOUNTS\b", "Accountancy"),
    (r"\bBUSINESS\s+STUDIES\b", "Business Studies"),
)

_LANGUAGE_SUBJECTS: frozenset[str] = frozenset({"English", "Marathi", "Hindi", "Sanskrit"})
_CORE_150_SUBJECTS: frozenset[str] = frozenset({"Mathematics", "Science", "Social Sciences"})
_MONTH_NAME_BY_FRAGMENT: tuple[tuple[str, str], ...] = (
    ("JANUARY", "January"),
    ("JAN", "January"),
    ("FEBRUARY", "February"),
    ("FEB", "February"),
    ("MARCH", "March"),
    ("MAR", "March"),
    ("APRIL", "April"),
    ("APR", "April"),
    ("MAY", "May"),
    ("JUNE", "June"),
    ("JUN", "June"),
    ("JULY", "July"),
    ("JUL", "July"),
    ("AUGUST", "August"),
    ("AUG", "August"),
    ("SEPTEMBER", "September"),
    ("SEPT", "September"),
    ("SEP", "September"),
    ("OCTOBER", "October"),
    ("OCT", "October"),
    ("NOVEMBER", "November"),
    ("NOV", "November"),
    ("DECEMBER", "December"),
    ("DEC", "December"),
)


def _normalise_ocr_bbox(bbox: Any) -> Optional[List[List[float]]]:
    """Convert OCR bounding boxes to a standard four-point polygon.

    OCR libraries return boxes in several shapes: four corner points,
    x1/y1/x2/y2 rectangles, or numpy arrays.  This helper converts all of them
    into the same [[x, y], ...] structure so the layout grouping code only has
    one format to handle.  Returning None is safer than guessing when the
    bounding box shape is not recognised.
    """
    if bbox is None:
        return None

    if hasattr(bbox, "tolist"):
        bbox = bbox.tolist()
    if isinstance(bbox, tuple):
        bbox = list(bbox)

    if not isinstance(bbox, list) or not bbox:
        return None

    if len(bbox) == 4 and all(isinstance(v, (int, float)) for v in bbox):
        x1, y1, x2, y2 = [float(v) for v in bbox]
        return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

    if len(bbox) == 4 and all(isinstance(pt, (list, tuple)) and len(pt) >= 2 for pt in bbox):
        return [[float(pt[0]), float(pt[1])] for pt in bbox]

    return None


def _bbox_y_centre(bbox: List[List[float]]) -> float:
    """Return the vertical centre of one OCR box."""
    ys = [pt[1] for pt in bbox]
    return (min(ys) + max(ys)) / 2.0


def _bbox_x_left(bbox: List[List[float]]) -> float:
    """Return the left-most x position of one OCR box."""
    return min(pt[0] for pt in bbox)


def _bbox_y_top(bbox: List[List[float]]) -> float:
    """Return the top y position of one OCR box."""
    return min(pt[1] for pt in bbox)


def _bbox_y_bottom(bbox: List[List[float]]) -> float:
    """Return the bottom y position of one OCR box."""
    return max(pt[1] for pt in bbox)


def _bbox_width(bbox: List[List[float]]) -> float:
    """Return the width of one OCR box."""
    xs = [pt[0] for pt in bbox]
    return max(xs) - min(xs)


def _bbox_height(bbox: List[List[float]]) -> float:
    """Return the height of one OCR box."""
    ys = [pt[1] for pt in bbox]
    return max(ys) - min(ys)


def _collect_layout_words(results: List[tuple]) -> List[Dict[str, Any]]:
    """Normalise OCR words into one consistent structure.

    PaddleOCR returns words as independent boxes with text and confidence.
    This helper cleans those values once so both the parser text and the
    markdown artifact use the exact same filtered OCR tokens.
    """
    words: List[Dict[str, Any]] = []

    for bbox, txt, conf in results:
        norm_bbox = _normalise_ocr_bbox(bbox)
        if norm_bbox is None:
            continue
        try:
            conf_f = float(conf)
        except (TypeError, ValueError):
            conf_f = 0.0
        if conf_f < 0.3:
            continue

        text = re.sub(r"\s+", " ", str(txt or "")).strip()
        if not text:
            continue

        words.append(
            {
                "bbox": norm_bbox,
                "text": text,
                "confidence": conf_f,
                "x_left": _bbox_x_left(norm_bbox),
                "y_centre": _bbox_y_centre(norm_bbox),
                "y_top": _bbox_y_top(norm_bbox),
                "y_bottom": _bbox_y_bottom(norm_bbox),
                "width": _bbox_width(norm_bbox),
                "height": _bbox_height(norm_bbox),
            }
        )

    return words


def _group_layout_words_into_rows(words: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """Group OCR words into visual rows using their box centres.

    The marksheet parser needs a clean reading order, while the markdown file
    needs rows that still reflect the document structure.  We use one adaptive
    row tolerance so both outputs are built from the same row grouping.
    """
    if not words:
        return []

    heights = [float(word["height"]) for word in words if float(word["height"]) > 0]
    median_height = statistics.median(heights) if heights else 18.0
    row_tolerance_px = max(12.0, median_height * 0.6)

    sorted_words = sorted(words, key=lambda item: (float(item["y_centre"]), float(item["x_left"])))

    rows: List[List[Dict[str, Any]]] = []
    current_row: List[Dict[str, Any]] = []
    current_row_y: Optional[float] = None

    for word in sorted_words:
        word_y = float(word["y_centre"])
        if current_row_y is None or abs(word_y - current_row_y) <= row_tolerance_px:
            current_row.append(word)
            current_row_y = (
                word_y
                if current_row_y is None
                else ((current_row_y * (len(current_row) - 1)) + word_y) / len(current_row)
            )
            continue

        rows.append(sorted(current_row, key=lambda item: float(item["x_left"])))
        current_row = [word]
        current_row_y = word_y

    if current_row:
        rows.append(sorted(current_row, key=lambda item: float(item["x_left"])))

    return rows


def _render_plain_row_text(rows: List[List[Dict[str, Any]]]) -> str:
    """Render grouped OCR rows into plain line-by-line text for parsers."""
    return "\n".join(" ".join(str(word["text"]) for word in row) for row in rows)


def _render_spatial_row_text(rows: List[List[Dict[str, Any]]]) -> str:
    """Render grouped OCR rows into spacing-aware text for the markdown artifact.

    The artifact should look like the original marksheet when opened in a text
    editor.  We therefore convert horizontal OCR positions into fixed-width
    character columns and preserve larger vertical gaps as blank lines.
    """
    if not rows:
        return ""

    all_words = [word for row in rows for word in row]
    char_width_candidates = [
        float(word["width"]) / max(len(str(word["text"])), 1)
        for word in all_words
        if len(str(word["text"])) > 0 and float(word["width"]) > 0
    ]
    char_unit = statistics.median(char_width_candidates) if char_width_candidates else 10.0
    char_unit = min(max(char_unit, 4.0), 18.0)

    left_margin = min(float(word["x_left"]) for word in all_words)
    row_heights = [
        max(float(word["y_bottom"]) for word in row) - min(float(word["y_top"]) for word in row)
        for row in rows
        if row
    ]
    row_gap_unit = statistics.median(row_heights) if row_heights else 24.0

    lines: List[str] = []
    previous_bottom: Optional[float] = None

    for row in rows:
        row_top = min(float(word["y_top"]) for word in row)
        row_bottom = max(float(word["y_bottom"]) for word in row)
        if previous_bottom is not None:
            vertical_gap = row_top - previous_bottom
            if vertical_gap > row_gap_unit * 0.9:
                blank_lines = max(1, min(3, int(round(vertical_gap / max(row_gap_unit, 1.0))) - 1))
                lines.extend([""] * blank_lines)

        row_parts: List[str] = []
        cursor = 0
        for word in row:
            target_col = max(0, int(round((float(word["x_left"]) - left_margin) / char_unit)))
            gap = target_col - cursor if not row_parts else max(1, target_col - cursor)
            if gap > 0:
                row_parts.append(" " * gap)
                cursor += gap
            text = str(word["text"])
            row_parts.append(text)
            cursor += len(text)

        lines.append("".join(row_parts).rstrip())
        previous_bottom = row_bottom

    return "\n".join(lines)


def _layout_text_from_ocr_results(results: List[tuple], engine_name: str) -> tuple[str, str]:
    """Rebuild OCR output into parser text and markdown text using the detected boxes.

    Raw OCR results arrive as isolated words with coordinates.  Sorting by the
    box centre and grouping nearby words back into rows preserves the original
    table order, which is the key signal our deterministic marksheet parsers
    need.  The markdown artifact additionally keeps horizontal spacing so the
    visible structure of the marksheet is not flattened into a plain sentence.
    """
    words = _collect_layout_words(results)
    if not words:
        return "", ""

    rows = _group_layout_words_into_rows(words)
    plain_text = _render_plain_row_text(rows)
    markdown_text = _render_spatial_row_text(rows)
    log.info(
        "document_extract: %s extracted %d rows, %d words",
        engine_name,
        len(rows),
        len(words),
    )
    return plain_text, markdown_text


def _normalise_marksheet_markdown_text(ocr_text: str) -> str:
    """Turn row-preserving OCR text into a plain markdown body.

    The user wants the marksheet written exactly in reading order without any
    extra interpretation. We therefore keep one detected row per line, strip
    trailing whitespace, and drop empty bands that do not carry content.
    """
    lines = [line.rstrip() for line in str(ocr_text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    content_lines: List[str] = []
    pending_blank_count = 0

    for line in lines:
        if line.strip():
            if content_lines and pending_blank_count:
                content_lines.extend([""] * min(pending_blank_count, 2))
            content_lines.append(line)
            pending_blank_count = 0
            continue
        if content_lines:
            pending_blank_count += 1

    if not content_lines:
        return ""

    return "\n".join(content_lines) + "\n"


def _marksheet_artifact_dir(artifact_root: Union[str, Path], source_name: str) -> Path:
    """Return the artifact directory used for one marksheet source file."""
    stem = Path(source_name or "marksheet").stem.replace(" ", "_") or "marksheet"
    target = Path(artifact_root) / stem
    target.mkdir(parents=True, exist_ok=True)
    return target


def _write_marksheet_ocr_markdown(
    source: Union[str, Path, bytes],
    *,
    filename: str,
    markdown_text: str,
    artifact_root: Union[str, Path],
) -> str:
    """Persist the raw top-to-bottom OCR scan of a marksheet as markdown.

    This file is intentionally dumb: it keeps the OCR rows in sequence so a
    later LLM step can reason over them without any parser assumptions baked
    into the artifact itself.
    """
    content = _normalise_marksheet_markdown_text(markdown_text)
    if not content:
        return ""

    if isinstance(source, (str, Path)):
        source_name = Path(source).name
    else:
        source_name = filename or "marksheet"
    if filename:
        source_name = filename

    artifact_dir = _marksheet_artifact_dir(artifact_root, source_name)
    stem = Path(source_name).stem.replace(" ", "_") or "marksheet"
    markdown_path = artifact_dir / f"{stem}_ocr_scan.md"
    markdown_path.write_text(content, encoding="utf-8")
    log.info("document_extract: wrote raw marksheet OCR markdown to %s", markdown_path)
    return str(markdown_path)


def _get_paddleocr_reader() -> tuple[Optional[str], Any]:
    """Return the shared PaddleOCR reader used for marksheet OCR.

    BaseTruth now standardises the marksheet OCR path on PaddleOCR so the raw
    markdown artifact and the deterministic parsers see one engine's row order.
    """
    global _paddleocr_reader

    if _paddleocr_reader is not None:
        return "paddleocr", _paddleocr_reader

    try:
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        from paddleocr import PaddleOCR  # type: ignore

        _paddleocr_reader = PaddleOCR(
            lang="en",
            ocr_version="PP-OCRv4",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            enable_mkldnn=False,
        )
        log.info("document_extract: PaddleOCR reader initialised")
        return "paddleocr", _paddleocr_reader
    except Exception as exc:
        log.debug("document_extract: PaddleOCR not available: %s", exc)

    return None, None


def _paddleocr_to_text(img_bytes: bytes) -> tuple[str, str, str]:
    """Run PaddleOCR and rebuild parser text plus markdown text.

    The plain text output is used by the deterministic parsers and Gemma4.
    The markdown output keeps spacing so the saved OCR artifact still looks
    structurally similar to the original marksheet.
    """
    backend_name, reader = _get_paddleocr_reader()
    if backend_name is None or reader is None:
        return "", "", ""

    try:
        import numpy as np
        from PIL import Image as PIL_Image

        img = PIL_Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_array = np.array(img)

        results: List[tuple] = []
        raw_results = reader.predict(img_array) or []
        for page in raw_results:
            if isinstance(page, dict):
                boxes = page.get("dt_polys") or page.get("rec_polys") or []
                texts = page.get("rec_texts") or []
                scores = page.get("rec_scores") or []
                for index, bbox in enumerate(boxes):
                    text = texts[index] if index < len(texts) else ""
                    conf = scores[index] if index < len(scores) else 0.0
                    results.append((bbox, text, conf))
                continue

            for item in page or []:
                if len(item) < 2:
                    continue
                bbox = item[0]
                text_conf = item[1]
                if isinstance(text_conf, (list, tuple)) and len(text_conf) >= 2:
                    text = text_conf[0]
                    conf = text_conf[1]
                else:
                    text = ""
                    conf = 0.0
                results.append((bbox, text, conf))

        parser_text, markdown_text = _layout_text_from_ocr_results(results, backend_name)
        return backend_name, parser_text, markdown_text

    except Exception as exc:
        log.warning("document_extract: %s extraction failed: %s", backend_name, exc)
        return backend_name, "", ""


def _reformat_hsc_ocr_table(ocr_text: str) -> str:
    """Detect the Maharashtra HSC transposed marks table in OCR text and reformat it.

    The HSC marksheet prints subjects as COLUMNS (transposed layout).  PaddleOCR
    reads the image row-by-row, which means each OCR 'line' is a row of the
    transposed table — not a subject row.  The three key rows are:
        SUBJECT CODE:    01  40  54  55  AZ  (subject identifiers)
        MAXIMUM MARKS:   100 100 100 100 200  (max marks per subject column)
        MARKS OBTAINED:  070 082 070 071 171  (obtained per subject column)

    This function:
    1. Detects whether the OCR text looks like an HSC transposed table
       (checks for the characteristic "MAXIMUM MARKS" / "MARKS OBTAINED" row labels)
    2. If found, extracts those three rows from the OCR text
    3. Reformats them into a vertical table (one subject per row) so Gemma4
       can match subject codes to their obtained and max marks without confusion

    Returns the original OCR text unchanged if the HSC pattern is not found.
    The reformatted table is appended AFTER the original text so the model gets
    both the raw OCR context AND the cleaner derived table.
    """
    upper = ocr_text.upper()

    # Require both characteristic HSC row headers to be present
    has_max = "MAXIMUM MARKS" in upper
    has_obt = "MARKS OBTAINED" in upper or "MARKSOBTAINED" in upper
    if not (has_max and has_obt):
        return ocr_text

    # Walk line-by-line and locate the rows we need
    lines = ocr_text.split("\n")
    code_line: Optional[str] = None
    max_line: Optional[str] = None
    obt_line: Optional[str] = None

    for i, line in enumerate(lines):
        upper_line = line.upper()
        if "MAXIMUM MARKS" in upper_line:
            # The label may be on the same line as the numbers, OR
            # the numbers may be on the PREVIOUS line (common in HSC scans
            # where OCR picks up the row label separately)
            if re.search(r"\d{2,3}", line):
                max_line = line          # numbers AND label on same line
            elif i > 0 and re.search(r"\d{2,3}", lines[i - 1]):
                max_line = lines[i - 1]  # numbers are on the preceding line
        elif "MARKS OBTAINED" in upper_line or "MARKSOBTAINED" in upper_line:
            if re.search(r"\d{2,3}", line):
                obt_line = line
            elif i > 0 and re.search(r"\d{2,3}", lines[i - 1]):
                obt_line = lines[i - 1]
        elif re.search(r"SUBJ", upper_line) and (re.search(r"CODE|CODS|TCODE", upper_line) or re.search(r"\b0[0-9]\b", line)):
            # Lines like "01 40 54 55 Az SUBJECTCODE" or OCR variants "SUBJECTICODE":
            # if the label is alone (no digit codes), look at prev line for codes
            if re.search(r"\b0[0-9]\b|[Aa][Zz]", line):
                code_line = line
            elif i > 0 and re.search(r"\b0[0-9]\b|[Aa][Zz]", lines[i - 1]):
                code_line = lines[i - 1]

    # We need at least the max and obtained rows to do something useful
    if max_line is None or obt_line is None:
        return ocr_text

    # Extract all integers (and patterns like 600/700) from each row
    def extract_numbers(line: str) -> List[str]:
        # Match numbers like 070, 100, 600/700, 504
        return re.findall(r"\d{2,3}(?:/\d{2,4})?", line)

    # Also extract subject codes from the code line (2-char alphanumeric codes)
    def extract_codes(line: str) -> List[str]:
        # Subject codes like "01", "40", "54", "55", "AZ", "Az", "A2".
        # The vocational slot can appear as "AZ", "Az", OR "A2" depending on
        # the scan year / board variant — so we match A followed by any
        # letter or digit.
        return re.findall(r"\bA[A-Za-z0-9]\b|\b[0-9]{2}\b", line)

    max_nums = extract_numbers(max_line)
    obt_nums = extract_numbers(obt_line)
    codes: List[str] = []
    if code_line:
        codes = extract_codes(code_line)

    # The LAST number in each row is the TOTAL (rightmost column)
    # Subject values are everything before the total
    # max_nums may include "600/700" as the total marker
    # obt_nums last value is typically the grand total (e.g. 504)
    if not max_nums or not obt_nums:
        return ocr_text

    # The total cells are the last entries
    obt_total = obt_nums[-1]   # e.g. "504" (grand total, rightmost column)

    # We intentionally drop max_subjects here because the OCR max-marks line
    # is almost always garbled for HSC (e.g. "600 /700 200 1001 100 400 100 100"
    # instead of per-subject values like 100,100,100,100,200).
    # Instead we assign standard max_marks by subject-code pattern:
    #   - Subject codes AZ / Az (vocational language) → 200 marks
    #   - All other two-digit numeric codes (01, 40, 54, 55…) → 100 marks
    # This is the standard Maharashtra Board HSC subject allocation.
    obt_subjects = obt_nums[:-1]  # All except total

    # Pair subject codes with obtained values (zip stops at shortest list)
    code_list = codes if codes else [str(i + 1) for i in range(len(obt_subjects))]
    pairs = list(zip(code_list, obt_subjects))

    if not pairs:
        return ocr_text

    # Build a three-column table with standardised max_marks
    rows_table = [
        "Reformatted HSC Marks Table (subject code / standard max marks / marks obtained):",
        "max_marks assigned by code: AZ=200, all other 2-digit codes=100 (Maharashtra Board HSC standard).",
        "marks_obtained is from the MARKS OBTAINED OCR row — verify against image if unsure.",
        f"{'Subject Code':<15} {'max_marks':<12} {'marks_obtained (OCR)'}",
        "-" * 50,
    ]
    for code, ob in pairs:
        # Assign standard max marks: AZ/Az/A2 (vocational) = 200, all others = 100
        std_max = 200 if re.match(r"[Aa][A-Za-z0-9]", code) else 100
        rows_table.append(f"{code:<15} {std_max:<12} {ob}")
    rows_table.append("-" * 50)
    rows_table.append(f"*** GRAND TOTAL ROW (NOT a subject) *** printed_grand_total={obt_total}")

    return ocr_text + "\n\n" + "\n".join(rows_table)


def _parse_hsc_ocr_directly(ocr_text: str) -> Optional[Dict[str, Any]]:
    """Parse HSC marksheet data directly from OCR text, bypassing Gemma4.

    When the OCR text contains the characteristic HSC transposed-table markers
    (MAXIMUM MARKS / MARKS OBTAINED), this function extracts subject codes,
    marks_obtained, and max_marks directly in Python without any LLM inference.
    This is more reliable than asking Gemma4 to read a structured table from text
    because Gemma4 8B often misreads column alignment.

    Also extracts metadata (candidate name, seat number, month/year, percentage,
    grand total, result) from the OCR text using simple regex patterns.

    Returns a structured dict (same schema as Gemma4 output) on success,
    or None if the OCR text does not match the expected HSC pattern.
    """
    upper = ocr_text.upper()

    # Only proceed if both characteristic HSC row labels are present
    has_max = "MAXIMUM MARKS" in upper
    has_obt = "MARKS OBTAINED" in upper or "MARKSOBTAINED" in upper
    if not (has_max and has_obt):
        return None

    lines = ocr_text.split("\n")

    # ── Locate the three key data rows ──────────────────────────────────────
    code_line: Optional[str] = None
    obt_line: Optional[str] = None

    for i, line in enumerate(lines):
        upper_line = line.upper()

        if "MARKS OBTAINED" in upper_line or "MARKSOBTAINED" in upper_line:
            # Numbers precede the label line in HSC scans
            if re.search(r"\d{2,3}", line):
                obt_line = line
            elif i > 0 and re.search(r"\d{2,3}", lines[i - 1]):
                obt_line = lines[i - 1]

        elif re.search(r"SUBJ", upper_line) and (
            re.search(r"CODE|CODS|TCODE", upper_line) or re.search(r"\b0[0-9]\b", line)
        ):
            # Subject code line — look on this line or the one before
            if re.search(r"\b0[0-9]\b|[Aa][Zz]", line):
                code_line = line
            elif i > 0 and re.search(r"\b0[0-9]\b|[Aa][Zz]", lines[i - 1]):
                code_line = lines[i - 1]

    if obt_line is None:
        return None

    # ── Extract obtained values ──────────────────────────────────────────────
    # Match 2- or 3-digit numbers; the LAST is the grand total
    obt_nums = re.findall(r"\b\d{2,3}\b", obt_line)
    if len(obt_nums) < 2:
        return None

    obt_total_str = obt_nums[-1]     # Last = grand total (e.g. "504")
    obt_subjects_raw = obt_nums[:-1]  # Remaining = per-subject obtained values

    # ── Extract subject codes ────────────────────────────────────────────────
    code_list: List[str] = []
    if code_line:
        # Match two-digit numeric codes and vocational codes (AZ, Az, A2, A3 etc.).
        # The vocational column uses different code notation across exam years;
        # matching A followed by any letter or digit covers all known variants.
        code_list = re.findall(r"\bA[A-Za-z0-9]\b|\b[0-9]{2}\b", code_line)

    if not code_list:
        code_list = [str(i + 1) for i in range(len(obt_subjects_raw))]

    # ── Build subject rows ───────────────────────────────────────────────────
    # zip stops at the shorter of the two lists
    subjects: List[Dict[str, Any]] = []
    for code, ob_str in zip(code_list, obt_subjects_raw):
        # Standardised max_marks: AZ/Az/A2 (vocational) = 200, all numeric codes = 100
        std_max = 200 if re.match(r"[Aa][A-Za-z0-9]", code) else 100
        try:
            ob_int = int(ob_str)
        except ValueError:
            continue
        subjects.append({
            "subject_name": code,
            "marks_obtained": ob_int,
            "max_marks": std_max,
        })

    if not subjects:
        return None

    # ── Parse grand total ────────────────────────────────────────────────────
    try:
        printed_grand_total: Optional[int] = int(obt_total_str)
    except ValueError:
        printed_grand_total = None

    # Compute totals from the subjects we parsed
    computed_total = sum(s["marks_obtained"] for s in subjects)
    total_max_marks = sum(s["max_marks"] for s in subjects)

    # ── Parse metadata from OCR text with simple regex ────────────────────────
    # Candidate name: look for all-caps full-name line near the top
    candidate_name = ""
    for line in lines:
        stripped = line.strip()
        # A full name line in HSC is typically 3–5 words, all uppercase letters
        # e.g. "MALUSKAR HRISHIKESH NAMDEO"
        if re.match(r"^[A-Z]{2,}(?:\s[A-Z]{2,}){1,4}$", stripped):
            candidate_name = stripped
            break

    # Seat number: prefer alphanumeric codes that start with a letter (e.g. "B006795").
    # HSC seat numbers typically have a letter prefix followed by 6 digits.
    # Pure 6-digit patterns (\d{6}) would also match the SR.NO.OF STATEMENT
    # field (e.g. "005392") which appears on the same header row — so we
    # check for the letter-prefix variant first and fall back only if not found.
    seat_letter_match = re.search(r"\b([A-Z]\d{5,7})\b", ocr_text)
    if seat_letter_match:
        seat_number = seat_letter_match.group(1)
    else:
        # Fallback for boards that use purely numeric seat numbers
        seat_plain_match = re.search(r"\b(\d{6})\b", ocr_text)
        seat_number = seat_plain_match.group(1) if seat_plain_match else ""

    # Month/year: pattern like "FEB-2001", "MAR 2000", "OCT-1999"
    my_match = re.search(
        r"\b(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[- ](\d{4})\b",
        ocr_text.upper(),
    )
    month_year = f"{my_match.group(1)}-{my_match.group(2)}" if my_match else ""

    # Percentage: look for patterns like "84.00" or OCR variants "84_ 0o",
    # "84. 00" (OCR can misread decimal separator and '0' as 'o').
    # Pattern: 2-3 digits + dot-or-underscore + optional-space + 1-2 chars
    # where each char is a digit or OCR-o (lowercase/uppercase 'o' for zero).
    pct_match = re.search(r"\b(\d{2,3})[._][ ]?([0-9oO][0-9oO]?)\b", ocr_text)
    if pct_match:
        # Replace 'o' with '0' (OCR misread) in the decimal part
        decimal_part = pct_match.group(2).lower().replace("o", "0")
        percentage = f"{pct_match.group(1)}.{decimal_part}"
    else:
        # Fallback: plain decimal number
        pct_match2 = re.search(r"\b(\d{2,3}\.\d{2})\b", ocr_text)
        percentage = pct_match2.group(1) if pct_match2 else ""

    # Result: PASS / FAIL / ATKT / DISTINCTION
    # Search only the first 40 OCR lines (the marks/result section).
    # The footer grade chart (typically after line 45) contains "Distinction"
    # as a grade label — searching the full text would produce a false positive
    # for students who actually PASSED but did not receive Distinction.
    result_search_upper = "\n".join(lines[:40]).upper()
    result = ""
    for result_word in ("ATKT", "DISTINCTION", "PASS", "FAIL"):
        if result_word in result_search_upper:
            result = result_word.capitalize()
            break

    # ── Assemble the final return dict ───────────────────────────────────────
    data_quality_notes = []
    confidence = "HIGH"

    # Some marksheets print a total that is slightly higher than the visible
    # subject sum because of moderation, grace marks, or OCR-missed hidden
    # components. That rule is generic and must not depend on SSC/HSC labels.
    if printed_grand_total and printed_grand_total != computed_total:
        diff = abs(computed_total - printed_grand_total)
        pct_diff = diff / max(printed_grand_total, 1) * 100
        if pct_diff <= 25:
            data_quality_notes.append(
                f"Printed total note: sum of visible subject marks ({computed_total}) differs from "
                f"printed_grand_total ({printed_grand_total}) by {diff} marks ({pct_diff:.1f}%). "
                "The printed total may include moderation, grace marks, or a component that OCR did not recover exactly."
            )
            confidence = "MEDIUM"
        else:
            # Larger than 25% gap — OCR may have misread marks
            data_quality_notes.append(
                f"Marks mismatch ({pct_diff:.1f}%): computed {computed_total} "
                f"vs printed {printed_grand_total}. OCR may have misread some values — "
                "verify marks_obtained for each subject against the original marksheet."
            )
            confidence = "MEDIUM"

    _log_marksheet_ocr_structure(
        _LAYOUT_HSC_TRANSPOSED,
        {
            "code_line": code_line or "",
            "obtained_line": obt_line or "",
            "subject_codes": code_list,
            "obtained_marks": [int(value) for value in obt_subjects_raw],
            "derived_max_marks": [int(subject["max_marks"]) for subject in subjects],
            "printed_grand_total": printed_grand_total,
            "computed_total": computed_total,
            "total_max_marks": total_max_marks,
            "percentage": percentage,
        },
    )

    return {
        "document_type": "Marksheet",
        "candidate_name": candidate_name,
        "board_or_university_name": (
            "Maharashtra State Board of Secondary and Higher Secondary Education"
        ),
        "school_or_college_name": None,
        "examination_name": "HSC",
        "month_year_of_passing": month_year,
        "enrollment_or_seat_number": seat_number,
        "subjects": subjects,
        "printed_grand_total": printed_grand_total,
        "total_max_marks": total_max_marks,
        "computed_total": computed_total,
        "percentage_or_cgpa": percentage,
        "result": result,
        "extraction_confidence": confidence,
        "data_quality_notes": data_quality_notes,
        "_extraction_attempts": 0,  # 0 = directly parsed, no LLM call
    }


def _normalise_marksheet_ocr_line(line: str) -> str:
    """Clean common OCR noise without changing the table meaning.

    Marksheet OCR often mixes repeated punctuation, slash spacing, and letter/
    digit confusions into otherwise correct rows.  This helper fixes the small,
    repetitive issues that break regex parsing while leaving the original row
    order intact.  Keeping the cleanup conservative is important because the
    parser should correct noise, not invent new content.
    """
    cleaned = re.sub(r"[|]+", " ", line)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = cleaned.replace("O/", "0/")
    cleaned = re.sub(r"\b([0-9]{2,3})[oO]\b", r"\g<1>0", cleaned)
    cleaned = cleaned.replace("MAXMUM", "MAXIMUM")
    cleaned = cleaned.replace("MAAKS", "MARKS")
    cleaned = cleaned.replace("MAXMUM", "MAXIMUM")
    cleaned = cleaned.replace("SUBUECTS", "SUBJECTS")
    cleaned = cleaned.replace("SCEINCES", "SCIENCES")
    return cleaned


def _looks_like_max_marks_label(line: str) -> bool:
    """Detect noisy OCR variants of the max-marks label.

    Many scanned board marksheets blur `MAXIMUM MARKS` into forms like
    `MAXMUM MAAKS`.  Exact string checks miss these variants and leave the
    document family as `unknown`, so we use a tolerant label detector here.
    """
    upper = _normalise_marksheet_ocr_line(line).upper()
    return bool(re.search(r"\bMAX[A-Z]*\b", upper) and re.search(r"\bMAR[KQ]S?\b", upper))


def _extract_ordered_subject_names(lines: List[str]) -> List[str]:
    """Extract subject names from the OCR header region in reading order.

    Board marksheets often split subject headers across two or three OCR lines.
    We scan the combined header block with a canonical alias table so short OCR
    tokens like `ENG`, `MAR`, or misspelt `SCEINCES` still resolve to a stable
    subject sequence.  Duplicates are removed while preserving first appearance.
    """
    header_blob = " ".join(_normalise_marksheet_ocr_line(line).upper() for line in lines if line.strip())
    header_blob = re.sub(r"\bFIRST\b|\bSECOND\b|\bTHIRD\b|\bRESULT\b|\bGRAND\b|\bTOTAL\b|\bLANGUAGES\b|\bSUBJECTS\b", " ", header_blob)
    header_blob = re.sub(r"\s+", " ", header_blob).strip()

    hits: List[tuple[int, str]] = []
    for pattern, subject_name in _SUBJECT_ALIAS_PATTERNS:
        for match in re.finditer(pattern, header_blob):
            hits.append((match.start(), subject_name))

    hits.sort(key=lambda item: item[0])
    ordered: List[str] = []
    for _pos, subject_name in hits:
        if subject_name not in ordered:
            ordered.append(subject_name)

    # Maharashtra SSC-style OCR often places the group header for Maths/Science
    # above the shorter language abbreviations, so raw OCR order becomes
    # Maths/Science/Social before English/Marathi/Hindi even though the printed
    # columns are the reverse.  When we detect this exact six-subject family,
    # switch to the canonical exam order to keep marks aligned with subjects.
    canonical_ssc_subjects = [
        "English",
        "Marathi",
        "Hindi",
        "Mathematics",
        "Science",
        "Social Sciences",
    ]
    if set(ordered) == set(canonical_ssc_subjects):
        return canonical_ssc_subjects

    return ordered


def _extract_percentage_from_text(text: str) -> str:
    """Extract a percentage string from noisy OCR text.

    OCR commonly reads `84.00` as `84 .60`, `84_ 0o`, or similar.  This helper
    centralises the percentage cleanup so both direct parsers and the Gemma4
    hint-builder can use the same normalisation rule.
    """
    cleaned = _normalise_marksheet_ocr_line(text)
    explicit_match = re.search(r"\b(\d{2,3})\s*[._]\s*([0-9oO]{2})\b", cleaned)
    if explicit_match:
        decimal_part = explicit_match.group(2).lower().replace("o", "0")
        return f"{explicit_match.group(1)}.{decimal_part}"
    plain_matches = re.findall(r"\b(\d{2,3}\.\d{2})\b", cleaned)
    if plain_matches:
        return plain_matches[-1]
    split_decimal_matches = re.findall(r"\b(\d{2,3})\s+([0-9oO]{2})\b", cleaned)
    if split_decimal_matches:
        whole_part, decimal_part = split_decimal_matches[-1]
        return f"{whole_part}.{decimal_part.lower().replace('o', '0')}"
    return ""


def _extract_month_year_from_marksheet_lines(lines: List[str]) -> str:
    """Extract a month-year value from noisy marksheet OCR lines.

    Board marksheets often print the month beside the year inside a dense
    metadata row, and OCR can split `March-1999` into fragments like `M4`
    and `Rch-1999`.  We therefore look around the detected year token and
    rebuild the nearby month fragment after correcting common digit/letter OCR
    swaps such as `4 -> A`.
    """
    direct_text = " ".join(lines).upper()
    direct_match = re.search(r"\b(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|SEPT|OCT|NOV|DEC)[A-Z- ]*(\d{4})\b", direct_text)
    if direct_match:
        month_fragment = direct_match.group(1)
        year = direct_match.group(2)
        for prefix, month_name in _MONTH_NAME_BY_FRAGMENT:
            if month_fragment.startswith(prefix[:3]):
                return f"{month_name}-{year}"

    month_char_map = str.maketrans({"0": "O", "1": "I", "4": "A", "5": "S", "8": "B"})
    for line in lines[:12]:
        tokens = re.findall(r"[A-Z0-9]+", line.upper())
        for idx, token in enumerate(tokens):
            if not re.fullmatch(r"(19|20)\d{2}", token):
                continue
            year = token
            month_fragment = "".join(tokens[max(0, idx - 2):idx]).translate(month_char_map)
            month_fragment = re.sub(r"[^A-Z]", "", month_fragment)
            for prefix, month_name in _MONTH_NAME_BY_FRAGMENT:
                if prefix in month_fragment:
                    return f"{month_name}-{year}"
    return ""


def _log_marksheet_ocr_structure(layout_family: str, structure: Dict[str, Any]) -> None:
    """Write the OCR-derived marks table structure into the application log.

    The Log Analyzer screen reads plain text log lines, so deterministic
    marksheet parsers emit one compact JSON payload that captures the exact OCR
    rows and numeric arrays they used.  This keeps parser decisions visible to
    operators without requiring a debugger session.
    """
    try:
        payload = json.dumps(structure, ensure_ascii=False, sort_keys=True)
    except TypeError:
        payload = str(structure)
    log.info(
        "document_extract: marksheet_ocr_structure layout=%s payload=%s",
        layout_family,
        payload,
    )


def _extract_total_max_from_text(lines: List[str]) -> Optional[int]:
    """Read the total maximum marks value from OCR lines when available."""
    for line in lines:
        upper = _normalise_marksheet_ocr_line(line).upper()
        if "MAX" not in upper and "TOTAL" not in upper:
            continue
        fractions = re.findall(r"\b\d{3,4}\s*/\s*\d{3,4}\b", line)
        if fractions:
            left, right = [int(part.strip()) for part in fractions[0].split("/")]
            return min(left, right)
        nums = [int(value) for value in re.findall(r"\b\d{3,4}\b", line)]
        if nums:
            return max(nums)
    return None


def _infer_two_row_max_marks(subject_names: List[str], known_max_values: List[int], total_max_marks: Optional[int]) -> Optional[List[int]]:
    """Infer missing max-marks values for stable two-row board layouts.

    The direct parser only fills gaps when the remaining values are strongly
    implied by the detected subject family and printed grand maximum.  This is
    intentionally conservative: if the inference is weak, we leave the parser in
    fallback mode and let Gemma4 use the structured hint instead.
    """
    if len(known_max_values) == len(subject_names):
        return known_max_values
    if total_max_marks is None or len(known_max_values) >= len(subject_names):
        return None

    # Maharashtra SSC-style six-subject layout: three language papers out of 100
    # and three core papers out of 150, total 750.  This pattern is stable and
    # safe enough to infer directly when all six canonical subject names are
    # present in the OCR header.
    if (
        len(subject_names) == 6
        and total_max_marks == 750
        and all(subject in subject_names for subject in ("English", "Marathi", "Hindi", "Mathematics", "Science", "Social Sciences"))
    ):
        return [100, 100, 100, 150, 150, 150]

    remaining_sum = total_max_marks - sum(known_max_values)
    missing_count = len(subject_names) - len(known_max_values)
    if missing_count <= 0 or remaining_sum <= 0:
        return None

    # Generic conservative fallback: if the known values all agree and the
    # missing total divides cleanly, extend the same max mark to the remaining
    # subjects.  This catches simple layouts like all-subjects-out-of-100.
    if len(set(known_max_values)) == 1 and remaining_sum % missing_count == 0:
        candidate = remaining_sum // missing_count
        if candidate == known_max_values[0]:
            return known_max_values + [candidate] * missing_count

    return None


def _extract_two_row_marksheet_structure(ocr_text: str) -> Optional[Dict[str, Any]]:
    """Extract structural hints from two-row marksheets before LLM fallback.

    This is the core document-intelligence step for school-style marksheets.
    We do not try to solve every board format perfectly from OCR alone; instead
    we detect the table skeleton reliably enough that either the deterministic
    parser can finish the job, or Gemma4 receives a compact, high-signal summary
    of the detected rows instead of a raw OCR dump.
    """
    lines = [_normalise_marksheet_ocr_line(line) for line in ocr_text.split("\n") if line.strip()]
    if len(lines) < 4:
        return None

    max_line: Optional[str] = None
    max_line_index: Optional[int] = None
    for idx, line in enumerate(lines):
        if _looks_like_max_marks_label(line):
            if re.search(r"\d{2,4}", line):
                max_line = line
                max_line_index = idx
            elif idx > 0 and re.search(r"\d{2,4}", lines[idx - 1]):
                max_line = lines[idx - 1]
                max_line_index = idx - 1
            break

    if max_line is None or max_line_index is None:
        return None

    subject_header_lines = lines[max(0, max_line_index - 6):max_line_index]
    subject_names = _extract_ordered_subject_names(subject_header_lines)
    if len(subject_names) < 4:
        return None

    obtained_line: Optional[str] = None
    obtained_line_index: Optional[int] = None
    for idx in range(max_line_index + 1, min(len(lines), max_line_index + 6)):
        line = lines[idx]
        upper = line.upper()
        if "OBTAIN" in upper:
            if re.search(r"\d{2,4}", line):
                obtained_line = line
                obtained_line_index = idx
            elif idx > 0 and re.search(r"\d{2,4}", lines[idx - 1]):
                obtained_line = lines[idx - 1]
                obtained_line_index = idx - 1
            break

    if obtained_line is None:
        numeric_candidates: List[tuple[int, int, str]] = []
        for idx in range(max_line_index + 1, min(len(lines), max_line_index + 6)):
            line = lines[idx]
            numeric_count = len(re.findall(r"\b\d{2,4}\b", line))
            if numeric_count >= max(4, len(subject_names) // 2):
                numeric_candidates.append((numeric_count, idx, line))
        if numeric_candidates:
            numeric_candidates.sort(key=lambda item: (-item[0], item[1]))
            _count, obtained_line_index, obtained_line = numeric_candidates[0]

    if obtained_line is None:
        return None

    obtained_numbers = [int(value) for value in re.findall(r"\b\d{2,4}\b", obtained_line)]
    if not obtained_numbers:
        return None

    percentage = _extract_percentage_from_text(obtained_line)
    if not percentage:
        for idx in range(obtained_line_index or 0, min(len(lines), (obtained_line_index or 0) + 3)):
            percentage = _extract_percentage_from_text(lines[idx])
            if percentage:
                break

    printed_total: Optional[int] = None
    obtained_marks = list(obtained_numbers)
    outlier_candidates = [value for value in obtained_marks if value > 250]
    if outlier_candidates:
        printed_total = max(outlier_candidates)
        obtained_marks.remove(printed_total)

    if len(obtained_marks) > len(subject_names):
        obtained_marks = obtained_marks[:len(subject_names)]
    if len(obtained_marks) < len(subject_names):
        return None

    max_values_raw = [int(value) for value in re.findall(r"\b\d{2,4}\b", max_line)]
    total_max_marks = _extract_total_max_from_text(lines)
    max_values = list(max_values_raw)
    if total_max_marks is not None and total_max_marks in max_values and len(max_values) > 1:
        max_values.remove(total_max_marks)
    max_values = [value for value in max_values if value <= 200]

    inferred_max_values = _infer_two_row_max_marks(subject_names, max_values, total_max_marks)
    return {
        "subject_names": subject_names,
        "obtained_marks": obtained_marks[:len(subject_names)],
        "max_marks": inferred_max_values or max_values,
        "printed_grand_total": printed_total,
        "total_max_marks": total_max_marks,
        "percentage": percentage,
        "max_line": max_line,
        "obtained_line": obtained_line,
        "max_marks_inferred": inferred_max_values is not None and inferred_max_values != max_values,
    }


def _parse_two_row_marksheet_ocr_directly(ocr_text: str) -> Optional[Dict[str, Any]]:
    """Parse standard two-row subject tables directly from OCR text.

    This parser covers the common school-board layout where subjects are listed
    once and the marks are split across a max row and an obtained row.  It is a
    deterministic parser first and a hint generator second: when the structure is
    complete enough, we return final JSON immediately; when it is incomplete, we
    leave the result as None and let Gemma4 use the extracted structure hint.
    """
    structure = _extract_two_row_marksheet_structure(ocr_text)
    if structure is None:
        return None

    subject_names = structure["subject_names"]
    obtained_marks = structure["obtained_marks"]
    max_marks = structure["max_marks"]
    if len(subject_names) != len(obtained_marks) or len(max_marks) != len(subject_names):
        return None

    lines = [_normalise_marksheet_ocr_line(line) for line in ocr_text.split("\n") if line.strip()]
    subjects = [
        {
            "subject_name": subject_name,
            "marks_obtained": int(obtained),
            "max_marks": int(max_mark),
        }
        for subject_name, obtained, max_mark in zip(subject_names, obtained_marks, max_marks)
    ]

    candidate_name = _extract_uppercase_name(lines)
    # Seat numbers on Indian marksheets can have 0, 1, or 2 uppercase letter
    # prefixes followed by 5–7 digits (e.g. "36419", "B006795", "CO28223").
    # We prefer letter-prefixed codes over plain numeric values because plain
    # digit runs (roll numbers, sequence ids) appear frequently in OCR output
    # and are easily confused with the seat number we actually want.
    seat_number = ""
    seat_number_fallback = ""
    for line in lines:
        letter_match = re.search(r"\b([A-Z]{1,2}\d{5,7})\b", line.upper())
        if letter_match:
            seat_number = letter_match.group(1)
            break
        if not seat_number_fallback:
            num_match = re.search(r"\b(\d{5,7})\b", line.upper())
            if num_match:
                seat_number_fallback = num_match.group(1)
    if not seat_number:
        seat_number = seat_number_fallback

    month_year = _extract_month_year_from_marksheet_lines(lines)

    board_name = ""
    for line in lines[:8]:
        if "BOARD" in line.upper() or "SECONDARY EDUCATION" in line.upper():
            board_name = line.strip()
            break

    exam_name = ""
    for line in lines[:12]:
        upper = line.upper()
        if "S.S.C" in upper or "SSC" in upper:
            exam_name = "S.S.C. Examination"
            break
        if "H.S.C" in upper or "HSC" in upper:
            exam_name = "H.S.C. Examination"
            break
        # Additional board identifiers used by other Indian and international
        # examination boards that also use the two-row marks layout.
        if "10TH" in upper or "10 TH" in upper:
            exam_name = "10th Standard"
            break
        if "12TH" in upper or "12 TH" in upper:
            exam_name = "12th Standard"
            break
        if "CBSE" in upper:
            exam_name = "CBSE Examination"
            break
        if "ICSE" in upper:
            exam_name = "ICSE Examination"
            break
        if "ISC " in f"{upper} " or upper.endswith("ISC"):
            exam_name = "ISC Examination"
            break
        # Generic fallback: any header line naming an examination will be used
        # so that state boards and other boards still get their exam name.
        if "EXAMINATION" in upper and len(line.strip()) > 8:
            exam_name = line.strip()
            break

    computed_total = sum(int(subject["marks_obtained"]) for subject in subjects)
    printed_total = structure["printed_grand_total"] or computed_total
    total_max_marks = structure["total_max_marks"] or sum(int(subject["max_marks"]) for subject in subjects)

    notes: List[str] = [
        "The marks table structure was interpreted based on the two-row pattern (Max Marks / Obtained Marks).",
        f"The grand total ({printed_total}) was read from the row containing the obtained marks.",
    ]
    if structure["percentage"]:
        notes.append(f"The percentage ({structure['percentage']}) was read from the designated percentage field.")
    if structure.get("max_marks_inferred"):
        notes.append(
            "Some max_marks values were inferred from the printed total maximum and the detected subject pattern because OCR dropped part of the max-marks row."
        )

    _log_marksheet_ocr_structure(
        _LAYOUT_TWO_ROW_SUBJECT_TABLE,
        {
            "subject_names": subject_names,
            "max_line": structure.get("max_line") or "",
            "obtained_line": structure.get("obtained_line") or "",
            "obtained_marks": [int(value) for value in obtained_marks],
            "max_marks": [int(value) for value in max_marks],
            "printed_grand_total": printed_total,
            "computed_total": computed_total,
            "total_max_marks": total_max_marks,
            "percentage": structure.get("percentage") or "",
            "max_marks_inferred": bool(structure.get("max_marks_inferred")),
        },
    )

    return {
        "document_type": "Marksheet",
        "candidate_name": candidate_name,
        "board_or_university_name": board_name,
        "school_or_college_name": None,
        "examination_name": exam_name,
        "month_year_of_passing": month_year,
        "enrollment_or_seat_number": seat_number,
        "subjects": subjects,
        "printed_grand_total": printed_total,
        "total_max_marks": total_max_marks,
        "computed_total": computed_total,
        "percentage_or_cgpa": structure["percentage"],
        "result": "",
        "extraction_confidence": "HIGH" if not structure.get("max_marks_inferred") else "MEDIUM",
        "data_quality_notes": notes,
        "_extraction_attempts": 0,
    }


def _build_marksheet_structure_hint(
    ocr_text: str,
    layout_family: str,
    *,
    ignou_liteparse_header: Optional[Dict[str, str]] = None,
) -> str:
    """Build a compact structural summary for Gemma4 fallback extraction.

    This is the clean document-intelligence handoff: Python detects the table
    skeleton, then Gemma4 only needs to reason about arranging and validating the
    fields.  The hint is short, explicit, and layout-family aware, which is much
    more stable than asking the model to rediscover the structure from scratch.
    """
    lines_text = [_normalise_marksheet_ocr_line(line) for line in ocr_text.split("\n") if line.strip()]
    anchor_lines: List[str] = [
        "DETECTED MARKSHEET STRUCTURE HINTS:",
        f"layout_family: {layout_family}",
    ]

    candidate_name = _extract_uppercase_name(lines_text)
    if candidate_name:
        anchor_lines.append(f"detected_candidate_name: {candidate_name}")

    # Prefer letter-prefixed seat numbers (e.g. CO28223, B006795) over plain
    # numeric identifiers so we do not pass a roll/sequence number as the seat.
    seat_number = ""
    seat_number_fallback = ""
    for line in lines_text[:14]:
        letter_match = re.search(r"\b([A-Z]{1,2}\d{5,7})\b", line.upper())
        if letter_match:
            seat_number = letter_match.group(1)
            break
        if not seat_number_fallback:
            num_match = re.search(r"\b(\d{5,7})\b", line.upper())
            if num_match:
                seat_number_fallback = num_match.group(1)
    if not seat_number:
        seat_number = seat_number_fallback
    if seat_number:
        anchor_lines.append(f"detected_seat_number: {seat_number}")

    month_year = _extract_month_year_from_marksheet_lines(lines_text)
    if month_year:
        anchor_lines.append(f"detected_month_year: {month_year}")

    percentage = _extract_percentage_from_text(ocr_text)
    if percentage:
        anchor_lines.append(f"detected_percentage: {percentage}")

    board_name = ""
    for line in lines_text[:10]:
        upper = line.upper()
        if "BOARD" in upper or "UNIVERSITY" in upper or "SECONDARY EDUCATION" in upper:
            board_name = line.strip()
            break
    if board_name:
        anchor_lines.append(f"detected_board_or_university_name: {board_name}")

    exam_name = ""
    for line in lines_text[:14]:
        upper = line.upper()
        if "S.S.C" in upper or " SSC" in f" {upper}":
            exam_name = "S.S.C. Examination"
            break
        if "H.S.C" in upper or " HSC" in f" {upper}":
            exam_name = "H.S.C. Examination"
            break
        if "10TH" in upper:
            exam_name = "10th Standard"
            break
        if "12TH" in upper:
            exam_name = "12th Standard"
            break
    if exam_name:
        anchor_lines.append(f"detected_examination_name: {exam_name}")

    result_match = re.search(r"\b(PASS|FAIL|ATKT|DISTINCTION|FIRST CLASS|SECOND CLASS|THIRD CLASS)\b", " ".join(lines_text).upper())
    if result_match:
        anchor_lines.append(f"detected_result: {result_match.group(1)}")

    if layout_family == _LAYOUT_TWO_ROW_SUBJECT_TABLE:
        structure = _extract_two_row_marksheet_structure(ocr_text)
        if structure is None:
            return "\n".join(anchor_lines)
        lines = list(anchor_lines)
        lines.extend([
            f"subject_candidates: {structure['subject_names']}",
            f"detected_obtained_marks: {structure['obtained_marks']}",
            f"detected_max_marks: {structure['max_marks']}",
        ])
        if structure.get("printed_grand_total") is not None:
            lines.append(f"detected_printed_grand_total: {structure['printed_grand_total']}")
        if structure.get("total_max_marks") is not None:
            lines.append(f"detected_total_max_marks: {structure['total_max_marks']}")
        if structure.get("max_marks_inferred"):
            lines.append(
                "max_marks_inferred: true (the printed max row was partial, so Python filled the missing max values using the stable subject pattern and total maximum)."
            )
        computed_total = sum(int(value) for value in structure["obtained_marks"])
        if structure.get("printed_grand_total") is not None:
            lines.append(f"computed_total_from_rows: {computed_total}")
            lines.append(f"grand_total_match: {computed_total == structure['printed_grand_total']}")
        return "\n".join(lines)

    if layout_family == _LAYOUT_BE_MAX_MIN_OBT:
        parsed = _parse_be_ocr_directly(ocr_text)
        if parsed is None:
            return "\n".join(anchor_lines)
        preview = parsed.get("subjects", [])[:8]
        lines = list(anchor_lines)
        lines.extend([
            f"component_rows_preview: {preview}",
            f"detected_printed_grand_total: {parsed.get('printed_grand_total')}",
            f"detected_total_max_marks: {parsed.get('total_max_marks')}",
        ])
        if parsed.get("computed_total") is not None:
            lines.append(f"computed_total_from_rows: {parsed.get('computed_total')}")
            lines.append(f"grand_total_match: {parsed.get('computed_total') == parsed.get('printed_grand_total')}")
        return "\n".join(lines)

    if layout_family == _LAYOUT_HSC_TRANSPOSED:
        parsed = _parse_hsc_ocr_directly(ocr_text)
        if parsed is None:
            return "\n".join(anchor_lines)
        lines = list(anchor_lines)
        lines.extend([
            f"subject_code_rows: {parsed.get('subjects', [])}",
            f"detected_printed_grand_total: {parsed.get('printed_grand_total')}",
            f"detected_total_max_marks: {parsed.get('total_max_marks')}",
        ])
        if parsed.get("computed_total") is not None:
            lines.append(f"computed_total_from_rows: {parsed.get('computed_total')}")
            lines.append(f"grand_total_match: {parsed.get('computed_total') == parsed.get('printed_grand_total')}")
        return "\n".join(lines)

    if layout_family == _LAYOUT_IGNOU_OPEN_UNIVERSITY:
        parsed = _parse_ignou_ocr_directly(ocr_text)
        # The generic anchor_lines may contain a wrong detected_percentage (e.g. an
        # assignment mark like 19.50 instead of the actual aggregate 68.42).  Drop
        # that entry so only the IGNOU-parser-confirmed percentage appears below.
        anchor_lines_clean = [ln for ln in anchor_lines if not ln.startswith("detected_percentage:")]
        lines = list(anchor_lines_clean)
        lines.extend([
            # Tell Gemma4 how to read IGNOU-specific fields correctly from the image.
            # IGNOU documents have two number fields that look similar: the CERTIFICATE NO
            # (6 digits, a serial number) and the ENROLMENT NO (10 digits, the student ID).
            # Without this hint Gemma4 consistently picks CERTIFICATE NO for enrollment.
            "IGNOU_FIELD_GUIDANCE: The document header contains two numbered fields:",
            "  - CERTIFICATE NO (6 digits) is a document serial number - NOT the enrollment ID.",
            "  - ENROLMENT NO (10 digits) is the student enrollment identifier - use this for enrollment_or_seat_number.",
            # IGNOU marksheets have the INDIRA GANDHI NATIONAL OPEN UNIVERSITY watermark
            # repeating as background text throughout the marks table.  OCR picks this up
            # as noise tokens like GANDHINA, GANDHENATIONALC, OHAL, COARDRENA, etc.
            # Gemma4 must not use any of these watermark fragments as the candidate name.
            "IGNOU_WATERMARK_WARNING: The marks table cells contain background noise from the",
            "  INDIRA GANDHI NATIONAL OPEN UNIVERSITY watermark (fragments: GANDHI, NATIONAL,",
            "  INDIRA, GANDHINA, GANDHENATIONALC, OHAL, COARDRENA). These are NOT the candidate",
            "  name. The actual student name is in the header section of the document only.",
        ])
        # If the liteparse artifact successfully decoded the document header
        # (pypdf text extraction can read it even when PaddleOCR can't due to
        # Devanagari fonts and watermark density), supply the verified values
        # directly so Gemma4 copies them rather than attempting OCR of the
        # noisy header region.
        if ignou_liteparse_header:
            if ignou_liteparse_header.get("enrollment"):
                lines.append(
                    f"IGNOU_VERIFIED_ENROLLMENT: {ignou_liteparse_header['enrollment']} "
                    "(10-digit IGNOU enrollment number — copy this exactly into enrollment_or_seat_number)."
                )
            if ignou_liteparse_header.get("name"):
                lines.append(
                    f"IGNOU_VERIFIED_NAME: {ignou_liteparse_header['name']} "
                    "(student name from document header — copy this exactly into candidate_name)."
                )
        if parsed is not None:
            lines.extend([
                f"detected_examination_name: {parsed.get('examination_name', '')}",
                f"detected_percentage: {parsed.get('percentage_or_cgpa', '')}",
                f"detected_result: {parsed.get('result', '')}",
                f"detected_total_max_marks: {parsed.get('total_max_marks')}",
                f"ignou_subject_rows: {parsed.get('subjects', [])}",
            ])
        return "\n".join(lines)

    return "\n".join(anchor_lines)


def _parse_be_component_row(line: str, current_course_name: str = "") -> Optional[Dict[str, Any]]:
    """Parse one BE/engineering marks row from OCR text.

    Pune-style engineering marksheets print one course across several rows,
    with the assessment component near the numeric columns.  Reading from the
    end of the row is more stable than trying to match the whole line because
    OCR usually keeps the numbers together even when the left side is noisy.
    When a continuation row omits the course name, we reuse the previous one.
    """
    clean = _normalise_marksheet_ocr_line(line)
    if not clean:
        return None

    tokens = clean.split()
    if len(tokens) < 4:
        return None

    component_idx: Optional[int] = None
    for idx in range(len(tokens) - 1, -1, -1):
        if tokens[idx].upper() in _BE_COMPONENT_CODES:
            component_idx = idx
            break

    if component_idx is None:
        return None

    numeric_tokens: List[str] = []
    for token in tokens[component_idx + 1:]:
        norm = token.upper().replace("O", "0")
        if re.fullmatch(r"\d{1,3}", norm):
            numeric_tokens.append(norm)

    if len(numeric_tokens) < 3:
        return None

    max_marks = int(numeric_tokens[0])
    _min_marks = int(numeric_tokens[1])
    marks_obtained = int(numeric_tokens[2])

    name_tokens = tokens[:component_idx]
    if name_tokens and re.fullmatch(r"[0-9A-Z]{2,5}\.?", name_tokens[0].upper()):
        name_tokens = name_tokens[1:]

    if not name_tokens and current_course_name:
        course_name = current_course_name
    else:
        course_name = " ".join(name_tokens).strip(" .:-")

    if not course_name:
        return None

    # Title-case is easier to read in the UI, but we keep acronyms like DBMS.
    pretty_name = " ".join(
        word if word.isupper() and len(word) <= 4 else word.capitalize()
        for word in course_name.split()
    )
    component = tokens[component_idx].upper()

    return {
        "subject_name": f"{pretty_name} ({component})",
        "marks_obtained": marks_obtained,
        "max_marks": max_marks,
        "_course_name": pretty_name,
    }


def _extract_uppercase_name(lines: List[str]) -> str:
    """Pick the most likely candidate-name line from OCR text.

    Name lines on Indian marksheets are usually a short run of uppercase words.
    We skip institution/result words because those lines look similar but are
    not person names.  The function returns an empty string when no safe match
    is found, which is better than labelling a header as the candidate.
    """
    skip_words = re.compile(
        r"UNIVERSITY|BOARD|EXAM|RESULT|CLASS|TOTAL|COLLEGE|INSTITUTE|MARKS|STATEM|CERTIF|DIVISION|SUBJECT|GRADE|IMPORTANT|INSTRUCT|SCHOOL|FIRST|SECOND|THIRD|LANGUAGE|SOCIAL|SCIENCE|GRAND"
        r"|GANDHI|NATIONAL|INDIRA|COARDRENA|OHAL|GANDHIN",  # IGNOU watermark noise fragments
        re.I,
    )
    best_name = ""
    best_score = -1
    for line in lines[:12]:
        stripped = _normalise_marksheet_ocr_line(line).strip()
        if not stripped or any(ch.isdigit() for ch in stripped):
            continue
        if skip_words.search(stripped):
            continue

        tokens = re.findall(r"[A-Za-z]{2,}", stripped)
        if not 2 <= len(tokens) <= 5:
            continue

        alpha_chars = [ch for ch in stripped if ch.isalpha()]
        if not alpha_chars:
            continue
        uppercase_ratio = sum(1 for ch in alpha_chars if ch.isupper()) / len(alpha_chars)
        long_token_count = sum(1 for token in tokens if len(token) >= 4)
        if uppercase_ratio < 0.6 or long_token_count < 2:
            continue

        candidate = " ".join(token.upper() for token in tokens)
        score = long_token_count * 3 + len(tokens)
        if len(tokens) == 3:
            score += 2
        if score > best_score:
            best_name = candidate
            best_score = score
    return best_name


def _parse_be_grand_total(lines: List[str]) -> tuple[Optional[int], Optional[int], str]:
    """Extract the BE printed total and maximum total from OCR lines.

    The total line appears in more than one OCR order: some scans read it as
    `928/1500`, while others read the max first and the obtained total later on
    the same line.  Looking specifically at lines that contain `TOTAL` or
    `RESULT` avoids confusing seat numbers and dates with summary values.  The
    optional third return value is the line itself for logging or notes.
    """
    for line in lines:
        upper = line.upper()
        if "TOTAL" not in upper and "RESULT" not in upper:
            continue

        ratio_match = re.search(r"\b(\d{3,4})\s*/\s*(\d{3,4})\b", line)
        if ratio_match:
            a = int(ratio_match.group(1))
            b = int(ratio_match.group(2))
            if a <= b:
                return a, b, line
            return b, a, line

        nums = [int(n) for n in re.findall(r"\b\d{3,4}\b", line)]
        if len(nums) >= 2:
            return min(nums), max(nums), line

    return None, None, ""


def _parse_be_ocr_directly(ocr_text: str) -> Optional[Dict[str, Any]]:
    """Parse a BE/B.Tech MAX/MIN/OBT marks table directly from OCR text.

    Engineering marksheets are much easier to read from OCR text than from a
    vision-model prompt because the table repeats the same numeric pattern on
    every row.  This parser looks for the stable `component + max + min + obt`
    structure and converts each component row into a subject entry.  Gemma4 is
    kept as a fallback only when the OCR text does not contain enough stable
    rows to build a trustworthy table.
    """
    upper = ocr_text.upper()
    if "COURSE NAME" not in upper and "MAX MIN OBT" not in upper and "MAX" not in upper:
        return None
    if not any(code in upper for code in _BE_COMPONENT_CODES):
        return None

    lines = [_normalise_marksheet_ocr_line(line) for line in ocr_text.split("\n") if line.strip()]
    subjects: List[Dict[str, Any]] = []
    seen_subjects: set[str] = set()
    current_course_name = ""

    for line in lines:
        parsed = _parse_be_component_row(line, current_course_name=current_course_name)
        if parsed is None:
            continue

        current_course_name = str(parsed.pop("_course_name", "") or current_course_name)
        subject_name = str(parsed.get("subject_name") or "")
        if subject_name in seen_subjects:
            continue
        seen_subjects.add(subject_name)
        subjects.append(parsed)

    if len(subjects) < 5:
        log.info(
            "document_extract: BE direct parse rejected because only %d rows were found",
            len(subjects),
        )
        return None

    printed_total, printed_max, total_line = _parse_be_grand_total(lines)
    computed_total = sum(int(s["marks_obtained"]) for s in subjects)
    computed_max = sum(int(s["max_marks"]) for s in subjects)

    candidate_name = _extract_uppercase_name(lines)

    university_name = ""
    for line in lines[:15]:
        if "UNIVERSITY" in line.upper():
            raw_name = line.strip()
            # OCR on older scanned marks sheets sometimes runs adjacent header
            # words together without spaces (e.g. "UNIVERSITYOFPUNE").  Split
            # the two most common merged patterns back into separate words.
            raw_name = re.sub(r"(UNIVERSITY)(OF|FOR)", r"\1 \2 ", raw_name, flags=re.IGNORECASE)
            raw_name = re.sub(r"(BOARD)(OF|FOR)", r"\1 \2 ", raw_name, flags=re.IGNORECASE)
            university_name = re.sub(r"\s+", " ", raw_name).strip()
            break

    exam_name = ""
    for line in lines:
        upper_line = line.upper()
        if "EXAMINATION" in upper_line and ("B.E" in upper_line or "BE " in upper_line or "B TECH" in upper_line or "B.TECH" in upper_line):
            exam_name = line.strip()
            break
    if not exam_name:
        if "B.E" in upper or "BE " in upper:
            exam_name = "B.E."

    month_year = ""
    month_match = re.search(
        r"\b(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]*\s*(\d{4})\b",
        upper,
    )
    if month_match:
        month_year = f"{month_match.group(1)} {month_match.group(2)}"
    else:
        # Fallback: OCR on old scanned marksheets confuses certain characters
        # inside year tokens — most commonly Z→2 and O→0 (e.g. the year 2006
        # is read as "ZOO6" when adjacent to a month name like "MAYZOO6").
        # Apply the character substitution in the specific 4-char year context
        # (letter Z followed by two O-like chars followed by one digit) so we
        # do not disturb other parts of the OCR text.
        noisy_upper = re.sub(
            r"(?<=[A-Z])(Z)([O0])([O0])(\d)\b",
            lambda m: "200" + m.group(4),
            upper,
        )
        month_match_noisy = re.search(
            r"\b(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s*(\d{4})\b",
            noisy_upper,
        )
        if month_match_noisy:
            month_year = f"{month_match_noisy.group(1)} {month_match_noisy.group(2)}"

    # Prefer letter-prefixed seat numbers (e.g. B2084275) over plain numeric
    # ids — plain digit runs (0113250) appear earlier in OCR text and would
    # otherwise be selected first by a simple re.search alternation.
    seat_letter_match = re.search(r"\b[A-Z]{1,2}\d{5,}\b", upper)
    if seat_letter_match:
        seat_number = seat_letter_match.group(0)
    else:
        seat_plain_match = re.search(r"\b\d{6,}\b", upper)
        seat_number = seat_plain_match.group(0) if seat_plain_match else ""

    result = ""
    result_match = re.search(
        r"(FIRST\s+CLASS\s+WITH\s+DISTINCTION|FIRST\s+CLASS|SECOND\s+CLASS|PASS|FAIL|ATKT)",
        upper,
    )
    if result_match:
        result = " ".join(word.capitalize() for word in result_match.group(1).split())

    data_quality_notes: List[str] = []
    confidence = "HIGH"
    if printed_total is not None:
        diff = abs(computed_total - printed_total)
        pct_diff = diff / max(printed_total, 1) * 100
        if pct_diff > 5:
            confidence = "MEDIUM"
            data_quality_notes.append(
                f"BE totals differ: computed_total={computed_total} vs printed_grand_total={printed_total}. "
                "This usually means one OCR row needs manual review."
            )

    if total_line and printed_max is not None and printed_max != computed_max:
        data_quality_notes.append(
            f"Printed BE maximum total {printed_max} was read from '{total_line}'. "
            f"Summed component max marks equal {computed_max}."
        )

    if len(subjects) < 8:
        confidence = "MEDIUM"
        data_quality_notes.append(
            f"Only {len(subjects)} BE component rows were parsed. Check the OCR if more rows are expected."
        )

    _log_marksheet_ocr_structure(
        _LAYOUT_BE_MAX_MIN_OBT,
        {
            "summary_line": total_line,
            "row_count": len(subjects),
            "subjects": subjects,
            "printed_grand_total": printed_total,
            "printed_total_max": printed_max,
            "computed_total": computed_total,
            "computed_total_max": computed_max,
        },
    )

    return {
        "document_type": "Marksheet",
        "candidate_name": candidate_name,
        "board_or_university_name": university_name,
        "school_or_college_name": None,
        "examination_name": exam_name,
        "month_year_of_passing": month_year,
        "enrollment_or_seat_number": seat_number,
        "subjects": subjects,
        "printed_grand_total": printed_total,
        "total_max_marks": printed_max or computed_max,
        "computed_total": computed_total,
        "percentage_or_cgpa": "",
        "result": result,
        "extraction_confidence": confidence,
        "data_quality_notes": data_quality_notes,
        "_extraction_attempts": 0,
    }


def _parse_ignou_ocr_directly(ocr_text: str) -> Optional[Dict[str, Any]]:
    """Parse an IGNOU open-university marksheet directly from PaddleOCR text.

    IGNOU marksheets (e.g. MAPC, MBA, BA) use a multi-row, multi-column table
    where each course spans one or more OCR rows and each row contains marks
    from multiple assessment sittings plus a final "out of 100" column.

    Key structural facts about IGNOU marksheets:
    - Each subject row ends with the pattern `100 [final_marks] SC [mmyy]`
      where `100` is the max marks for the TEE (Term-End Exam) component and
      `final_marks` is the marks obtained (this can be a decimal like 71.30).
    - The grand total printed at the bottom (e.g. `1300 889.40`) reflects the
      credit-weighted combination of assignment + TEE marks across ALL courses.
      It is NOT the sum of the per-subject `100`-column values.
    - Because printed_grand_total (e.g. 889) ≠ sum(per-subject TEE marks)
      (typically ~470–530), we deliberately omit printed_grand_total from the
      returned dict to prevent the mismatch guard from firing.  total_max_marks
      IS filled from the TOTAL line because it is accurate.

    Returns None if the OCR text does not contain enough IGNOU markers.
    """
    upper = ocr_text.upper()

    # Require both the completion summary line AND at least 2 IGNOU course codes
    has_completed = "SUCCESSFULLY COMPLETEDWITH" in upper or "SUCCESSFULLY COMPLETED WITH" in upper
    ignou_codes = re.findall(r"\bMPC[EL]?\d{2,3}\b|\bMPC[EL]?[O0]\d{1,2}\b", upper)  # allow OCR 'O' vs '0' swap
    if not has_completed or len(ignou_codes) < 2:
        return None

    lines = ocr_text.split("\n")

    # ── Extract examination name, percentage, result from summary line ──────
    # The summary line looks like:
    #   "MA IN PSYCHOLOGY SUCCESSFULLY COMPLETEDWITH 68.42 %(FIRST DIVISION)"
    examination_name = ""
    percentage = ""
    result = ""
    for line in lines:
        ul = line.upper()
        if "SUCCESSFULLY COMPLETED" not in ul:
            continue
        # Capture programme name (text before SUCCESSFULLY)
        prog_match = re.match(r"\s*(.+?)\s+SUCCESSFULLY\s+COMPLETED", line.strip(), re.IGNORECASE)
        if prog_match:
            examination_name = prog_match.group(1).strip()
        # Capture percentage (decimal number before the % sign)
        pct_match = re.search(r"(\d{2,3}\.\d{2})\s*%", line)
        if pct_match:
            percentage = pct_match.group(1)
        # Capture result label inside parentheses after %
        res_match = re.search(r"%\s*\(([A-Z\s]+?)\)", line.upper())
        if res_match:
            result = res_match.group(1).strip()
        break

    # ── Extract total_max_marks from the TOTAL line ──────────────────────────
    # The TOTAL line may span two OCR rows because the summary numbers appear
    # at the far right margin.  We search lines near the TOTAL: marker for
    # any pair of 3-4 digit numbers (e.g. "1300 889.40").  We do NOT anchor
    # to end-of-line because the completed-summary line often follows on the
    # same joined search block.
    total_max_marks: Optional[int] = None
    for i, line in enumerate(lines):
        if "TOTAL" not in line.upper():
            continue
        # Join this line and up to 2 following lines so the numbers found
        # on the continuation row are included in the search.
        search_block = " ".join(lines[i: min(i + 3, len(lines))])
        # findall returns all 3-4 digit number occurrences (including decimals
        # like 889.40).  We pick the pair with the highest value as the
        # total_max_marks/printed_total pair.
        large_nums = re.findall(r"\b(\d{3,4})(?:\.\d{1,2})?\b", search_block)
        if len(large_nums) >= 2:
            values = [int(v) for v in large_nums if int(v) >= 100]
            if len(values) >= 2:
                # The larger value is the max marks (e.g. 1300); the smaller
                # is the obtained total (e.g. 889).
                values.sort(reverse=True)
                total_max_marks = values[0]
        if total_max_marks is not None:
            break

    # ── Extract per-subject rows using course-code anchor ────────────────────
    # Each row that starts with an IGNOU course code is one subject entry.
    # The final marks appear right before "SC [mmyy]" at the end of the row.
    # Pattern: `...100  [marks]  SC`  where 100 = TEE max marks.
    subjects: List[Dict[str, Any]] = []
    seen_codes: set[str] = set()
    # pending_codes keeps run-codes that were seen but whose marks landed on a
    # separate continuation OCR row (their SC column was overwritten by noise).
    # It is a list so we can pop the MOST RECENT pending code when an orphan
    # continuation row is found.
    pending_codes: List[str] = []
    pending_titles: Dict[str, str] = {}  # code → title extracted from its code line

    # IGNOU rows have the format:
    #   MPCE012    PSYCHODIAGNOSTICS   ... # watermark noise # ...  100  74.30  SC  0623
    # The IGNOU background watermark is printed in the marks table, producing
    # '#' tokens and scattered uppercase fragments.
    # We strip these before extracting the title so the regex sees only the
    # clean "MPCE012    PSYCHODIAGNOSTICS" portion.
    #
    # Subject names in IGNOU OCR are all-letter words — isolating them stops
    # numeric credit/attempt columns (e.g. "6 2") from bleeding into the title.
    def _extract_ignou_title(raw_line: str, code: str) -> str:
        """Extract the human-readable subject title from a single IGNOU OCR row.

        1. Strips everything from the first '#' character (IGNOU watermark noise).
        2. Captures consecutive letter-only word tokens immediately after the
           course code, stopping each word at only 1-2 spaces between (IGNOU
           uses 3+ spaces as column separators), so digit-only tokens and wide
           column gaps correctly terminate the title.
        3. Allows 'O' as OCR substitute for '0' in the course code (e.g. MPCEO11).
        Returns the code string itself when no title can be parsed.
        """
        clean = re.sub(r"\s*#.*$", "", raw_line)  # strip '#' watermark separator and all text after
        # Match letter-only words separated by at most 2 spaces.
        # Three or more consecutive spaces signals an IGNOU column boundary, so we
        # stop there even if more text follows.
        # Allow OCR O→0 substitution in the numeric part of the course code.
        tm = re.search(
            r"^\s*MPC[EL]?[O0\d]{2,3}\s+([A-Za-z][A-Za-z:&',./\-]*(?:\s{1,2}[A-Za-z][A-Za-z:&',./\-]*)*)",
            clean,
            re.IGNORECASE,
        )
        return tm.group(1).strip().title() if tm else code

    for i, line in enumerate(lines):
        ul = line.upper()

        # ── Case A: line starts with an IGNOU course code ────────────────
        code_match = re.match(r"\s*(MPC[EL]?[O0\d]{2,3})\b", ul)
        if code_match is not None:
            code = code_match.group(1).strip().replace("O", "0").replace("o", "0")  # normalise OCR O→0
            if code in seen_codes:
                # Second-attempt row for the same code — ignore (already recorded)
                continue

            # Always capture the title from the code line so we have it ready
            # even if marks appear on a later continuation row.
            title = _extract_ignou_title(line, code)

            # Extract the final marks from the `100 [marks] SC` pattern.
            # 100 is always the TEE component scale on IGNOU marksheets.
            tee_match = re.search(r"\b100\s+(\d{1,3}(?:\.\d{1,2})?)\s*(?:SC\b|$)", line)
            if tee_match is None:
                # Marks not on this line — queue the code for the continuation row.
                pending_codes.append(code)
                pending_titles[code] = title
                continue  # marks will be picked up by Case B below

            try:
                marks_f = float(tee_match.group(1))
                marks_int = round(marks_f)  # round 71.30 → 71
            except (ValueError, TypeError):
                pending_codes.append(code)
                pending_titles[code] = title
                continue

            seen_codes.add(code)
            subjects.append({
                "subject_name": title or code,
                "marks_obtained": marks_int,
                "max_marks": 100,   # TEE component is always out of 100 for IGNOU
            })
            continue

        # ── Case B: continuation row — no code prefix ─────────────────────
        # Some IGNOU courses span two OCR rows.  The first row has the code and
        # title but the watermark overwrites the SC column; the continuation row
        # carries `100  [marks]  SC  mmyy`.
        # Associate the marks with the MOST RECENTLY seen pending code.
        if pending_codes:
            tee_match = re.search(r"\b100\s+(\d{1,3}(?:\.\d{1,2})?)\s*SC\b", line)
            if tee_match:
                try:
                    marks_f = float(tee_match.group(1))
                    marks_int = round(marks_f)
                except (ValueError, TypeError):
                    continue
                pending_code = pending_codes.pop()  # most-recently-queued pending code
                title = pending_titles.pop(pending_code, pending_code)
                seen_codes.add(pending_code)
                subjects.append({
                    "subject_name": title or pending_code,
                    "marks_obtained": marks_int,
                    "max_marks": 100,
                })

    # ── Compute TEE sub-total for logging (NOT for mismatch validation) ──────
    tee_sum = sum(int(s["marks_obtained"]) for s in subjects)

    # Quality note explaining the IGNOU multi-component grading to reviewers
    data_quality_notes: List[str] = []
    data_quality_notes.append(
        "IGNOU multi-component grading: marks_obtained per subject is the Term-End Exam (TEE) mark "
        f"out of 100 (TEE subtotal = {tee_sum}). "
        f"The aggregate total_max_marks ({total_max_marks}) and percentage ({percentage}%) use a "
        "credit-weighted combination of Assignment (30%) + TEE (70%) marks that differs from the "
        "raw TEE sum. printed_grand_total is omitted here to avoid a false mismatch alarm."
    )
    if not subjects:
        data_quality_notes.append(
            "Could not parse individual subject rows from OCR text. "
            "This may occur when the marks table has heavy watermark interference."
        )

    confidence = "MEDIUM" if subjects else "LOW"

    _log_marksheet_ocr_structure(
        _LAYOUT_IGNOU_OPEN_UNIVERSITY,
        {
            "examination_name": examination_name,
            "percentage": percentage,
            "result": result,
            "total_max_marks": total_max_marks,
            "tee_sum": tee_sum,
            "subject_count": len(subjects),
            "subjects": subjects,
        },
    )

    return {
        "document_type": "Marksheet",
        "candidate_name": "",          # header was not in PaddleOCR output; Gemma4 reads from image
        "board_or_university_name": "IGNOU",
        "school_or_college_name": None,
        "examination_name": examination_name,
        "month_year_of_passing": "",   # date of printing is not month/year of passing
        "enrollment_or_seat_number": "",  # Gemma4 reads from image header
        "subjects": subjects,
        # Intentionally NOT setting printed_grand_total — it represents the
        # credit-weighted multi-component aggregate (e.g. 889 out of 1300),
        # which is always larger than the TEE sum and would wrongly trigger
        # the safe-fail subjects=[] guard in _validate_educational.
        "printed_grand_total": None,
        "total_max_marks": total_max_marks,
        "computed_total": tee_sum,
        "percentage_or_cgpa": percentage,
        "result": result,
        "extraction_confidence": confidence,
        "data_quality_notes": data_quality_notes,
        "_extraction_attempts": 0,  # 0 = directly parsed, no LLM call
    }


def _classify_marksheet_layout_family(ocr_text: str) -> Dict[str, Any]:
    """Classify the marksheet table family from OCR text markers.

    We only need a small number of families because each one maps to a parser
    strategy, not to a custom prompt.  The classifier looks for stable layout
    clues such as HSC transposed row headers or the BE `MAX MIN OBT` pattern.
    Returning a score and matched markers makes engine comparison transparent
    and helps explain why one OCR output was chosen over another.
    """
    upper = re.sub(r"[^A-Z0-9\s]", " ", ocr_text.upper())
    upper = re.sub(r"\s+", " ", upper).strip()

    markers: List[str] = []
    score = 0
    family = _LAYOUT_UNKNOWN

    # IGNOU open-university multi-term marksheet: detect before HSC/BE patterns
    # because IGNOU course codes (MPC006, MPCE011 etc.) are unique enough to
    # classify reliably without ambiguity with any school board layout.
    _ignou_code_count = len(re.findall(r"\bMPC[EL]?\d{2,3}\b", upper))
    _ignou_completed = "SUCCESSFULLY COMPLETEDWITH" in upper or "SUCCESSFULLY COMPLETED WITH" in upper
    if _ignou_completed and _ignou_code_count >= 2:
        family = _LAYOUT_IGNOU_OPEN_UNIVERSITY
        markers.extend(["SUCCESSFULLY_COMPLETED", f"ignou_code_count={_ignou_code_count}"])
        score = 88
    elif "MAXIMUM MARKS" in upper and "MARKS OBTAINED" in upper:
        family = _LAYOUT_HSC_TRANSPOSED
        markers.extend(["MAXIMUM MARKS", "MARKS OBTAINED"])
        if "SUBJECT CODE" in upper or "SUBJECTCODE" in upper:
            markers.append("SUBJECT CODE")
        score = 95
    elif _extract_two_row_marksheet_structure(ocr_text) is not None:
        family = _LAYOUT_TWO_ROW_SUBJECT_TABLE
        markers.extend(["two-row-subject-structure", "max-row", "obtained-row"])
        score = 82
    elif (
        "COURSE NAME" in upper
        or "MAX MIN OBT" in upper
        or all(code in upper for code in (" PP ", " TW ", " OR "))
    ):
        family = _LAYOUT_BE_MAX_MIN_OBT
        if "COURSE NAME" in upper:
            markers.append("COURSE NAME")
        if "MAX MIN OBT" in upper:
            markers.append("MAX MIN OBT")
        for code in ("PP", "TW", "OR", "PR"):
            if f" {code} " in f" {upper} ":
                markers.append(code)
        score = 90
    else:
        row_like_lines = 0
        for line in ocr_text.split("\n"):
            if re.search(r"[A-Z].*\b\d{1,3}\b.*\b\d{1,3}\b", line.upper()):
                row_like_lines += 1
        if row_like_lines >= 3:
            family = _LAYOUT_GENERIC_ROW_TABLE
            markers.append(f"row_like_lines={row_like_lines}")
            score = 45 + min(row_like_lines, 10)

    return {
        "family": family,
        "score": score,
        "markers": markers,
    }


def _score_marksheet_extraction_candidate(data: Dict[str, Any]) -> float:
    """Score a deterministic extraction so OCR engines can be compared.

    We do not have ground-truth labels at runtime, so the next best signal is
    extraction completeness plus arithmetic consistency.  A candidate with many
    parsed subject rows, totals, and identity fields is far more trustworthy
    than one that only produced a few rows.  The score is intentionally simple
    so it remains stable across OCR engines and document families.
    """
    score = 0.0
    subjects = data.get("subjects") or []
    score += min(len(subjects), 16) * 8
    if data.get("candidate_name"):
        score += 8
    if data.get("enrollment_or_seat_number"):
        score += 8
    if data.get("printed_grand_total") is not None:
        score += 10
    if data.get("total_max_marks") is not None:
        score += 6
    if data.get("percentage_or_cgpa"):
        score += 5

    confidence = str(data.get("extraction_confidence") or "").upper()
    if confidence == "HIGH":
        score += 12
    elif confidence == "MEDIUM":
        score += 6

    notes = data.get("data_quality_notes") or []
    if not notes:
        score += 5
    return score


def _evaluate_marksheet_ocr_candidate(engine_name: str, ocr_text: str) -> Dict[str, Any]:
    """Classify and score one OCR engine's marksheet output.

    This keeps OCR comparison honest: each engine is evaluated on the exact same
    downstream tasks that matter to the product.  If a layout family has a
    deterministic parser, the parser output contributes most of the score.  For
    generic families, the score falls back to layout clues and text coverage.
    """
    layout = _classify_marksheet_layout_family(ocr_text)
    family = layout["family"]

    direct_data: Optional[Dict[str, Any]] = None
    if family == _LAYOUT_HSC_TRANSPOSED:
        direct_data = _parse_hsc_ocr_directly(ocr_text)
    elif family == _LAYOUT_BE_MAX_MIN_OBT:
        direct_data = _parse_be_ocr_directly(ocr_text)
    elif family == _LAYOUT_TWO_ROW_SUBJECT_TABLE:
        direct_data = _parse_two_row_marksheet_ocr_directly(ocr_text)
    elif family == _LAYOUT_IGNOU_OPEN_UNIVERSITY:
        # For IGNOU we intentionally do NOT set direct_data (Gemma4 still runs).
        # _parse_ignou_ocr_directly is called inside _build_marksheet_structure_hint
        # instead, where its output feeds structure hints to Gemma4.  Gemma4 then
        # reads candidate_name and enrollment_or_seat_number from the image header
        # (PaddleOCR misses these because the IGNOU header contains Devanagari text).
        direct_data = None

    direct_score = _score_marksheet_extraction_candidate(direct_data) if direct_data else 0.0
    text_score = min(len(ocr_text) / 300.0, 15.0)
    total_score = float(layout.get("score", 0)) + direct_score + text_score

    return {
        "engine": engine_name,
        "text": ocr_text,
        "layout_family": family,
        "layout_markers": layout.get("markers", []),
        "layout_score": float(layout.get("score", 0)),
        "direct_data": direct_data,
        "direct_score": direct_score,
        "score": total_score,
    }


def _select_best_marksheet_ocr_candidate(ocr_candidates: List[tuple[str, str]]) -> tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """Pick the OCR engine whose output leads to the best parser result.

    The product cares about extracted fields, not raw OCR confidence. This
    selector compares each engine by the downstream family score and parser
    score, then returns both the winner and a machine-readable comparison list.
    The comparison metadata is attached to the extraction result so operators
    can inspect which engine was preferred and why.
    """
    evaluations: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None
    best_score = float("-inf")

    for engine_name, ocr_text in ocr_candidates:
        if not ocr_text:
            continue
        evaluation = _evaluate_marksheet_ocr_candidate(engine_name, ocr_text)
        evaluations.append(evaluation)
        evaluation_score = float(evaluation.get("score", 0.0))
        if best is None or evaluation_score > best_score:
            best = evaluation
            best_score = evaluation_score

    comparison = [
        {
            "engine": item["engine"],
            "layout_family": item["layout_family"],
            "layout_markers": item["layout_markers"],
            "layout_score": round(item["layout_score"], 1),
            "direct_score": round(item["direct_score"], 1),
            "score": round(item["score"], 1),
            "subject_count": len((item.get("direct_data") or {}).get("subjects") or []),
        }
        for item in evaluations
    ]
    return best, comparison


def _attach_extraction_metadata(
    data: Dict[str, Any],
    *,
    layout_family: str,
    ocr_engine_used: str,
    ocr_engine_comparison: List[Dict[str, Any]],
    raw_ocr_markdown_path: str = "",
) -> Dict[str, Any]:
    """Attach internal extraction metadata used for debugging and review.

    These fields explain which OCR engine won and which layout family was
    detected.  They are intentionally stored under underscore-prefixed keys so
    existing UI/data flows can ignore them safely.  Keeping this metadata on the
    result makes manual troubleshooting much faster when a marksheet is messy.
    """
    data["_layout_family"] = layout_family or _LAYOUT_UNKNOWN
    if ocr_engine_used:
        data["_ocr_engine_used"] = ocr_engine_used
    if ocr_engine_comparison:
        data["_ocr_engine_comparison"] = ocr_engine_comparison
    if raw_ocr_markdown_path:
        data["_raw_ocr_markdown_path"] = raw_ocr_markdown_path
    return data


def _extract_json_from_text(text: str) -> str:
    """Pull the first complete JSON object out of a text string.

    Gemma4 sometimes wraps its reply in markdown code fences or adds a
    sentence before the JSON.  This strips all that away.
    """
    stripped = text.strip()
    # Remove ```json ... ``` fences if present
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    # Find the first { ... } block
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1:
        return ""
    return stripped[start: end + 1]


def _validate_educational(data: Dict[str, Any]) -> List[str]:
    """Validation rules specific to educational documents (marksheets and degrees).

    Returns a list of plain-English error strings that are sent back to the
    model as correction instructions.  Critically:

    - We ONLY return errors (triggering a retry) when the problem is a small,
      fixable misread — e.g. one digit off in a clearly readable table.
    - We do NOT return correction errors when the table was fundamentally
      unreadable.  In that case we mutate 'data' in-place to set subjects=[]
      and extraction_confidence="LOW" and return NO errors (so no retry is
      triggered).  This is the "fail safely" principle ChatGPT described.
    - We strip any subject rows whose subject_name is a known section header
      (e.g. "LANGUAGES", "COMPULSORY") — those are layout labels, not subjects.
    """
    errors: List[str] = []
    doc_type = data.get("document_type", "")

    if not data.get("candidate_name"):
        errors.append("Candidate name is missing. It is usually the largest text near the top.")

    if doc_type == "Marksheet":
        subjects = data.get("subjects") or []

        # ── Step 1: Remove section-header rows ──────────────────────────────
        # These are layout labels printed on Maharashtra/CBSE marksheets to
        # group subjects (e.g. "LANGUAGES", "COMPULSORY").  They are not
        # individual subjects and must never appear as data rows.
        real_subjects = []
        rejected_headers = []
        for s in subjects:
            name = str(s.get("subject_name") or "").strip().lower()
            # A row is rejected as a section header ONLY when its name matches the
            # known-header list AND it has no marks_obtained.  This is important
            # because "Social Sciences" is a valid SSC subject that has its own
            # marks column — we must keep it.  True section headers (like
            # "LANGUAGES" or "COMPULSORY") never have their own marks row.
            has_marks = s.get("marks_obtained") is not None
            if name in _MARKSHEET_SECTION_HEADERS and not has_marks:
                rejected_headers.append(s.get("subject_name"))
            else:
                real_subjects.append(s)

        if rejected_headers:
            # Log what we stripped so it shows up in data_quality_notes
            data["data_quality_notes"] = list(data.get("data_quality_notes") or []) + [
                f"Removed {len(rejected_headers)} section-header row(s) that are not real subjects: "
                + ", ".join(str(h) for h in rejected_headers)
                + ". Only individual academic subjects are kept."
            ]
            data["subjects"] = real_subjects
            subjects = real_subjects

        # ── Step 2: Guard — if no real subjects remain ───────────────────────
        # Either the table was never readable, or all rows were headers.
        # In both cases: fail safely — return empty subjects with LOW confidence.
        # We do NOT push this as a retry-triggering error because there is nothing
        # the model can "fix"; it already tried its best and the table is unclear.
        if not subjects:
            data["subjects"] = []
            data["extraction_confidence"] = "LOW"
            data["data_quality_notes"] = list(data.get("data_quality_notes") or []) + [
                "Could not extract individual subject rows from the marks table. "
                "The table may be a scanned/corrupted image, use section labels "
                "instead of subject names, or the layout is not readable. "
                "subjects set to [] (safe-fail)."
            ]
            # Return NO errors — no retry needed, safe-fail already applied
            return errors

        # ── Step 3: Normalise marks to integers and compute totals ───────────
        computed_marks = 0   # sum of marks_obtained across real subjects
        computed_max = 0     # sum of max_marks across real subjects
        for s in subjects:
            mo = s.get("marks_obtained")
            try:
                mo_int = int(float(mo))
                s["marks_obtained"] = mo_int
                computed_marks += mo_int
            except (TypeError, ValueError):
                errors.append(
                    f"marks_obtained is not a valid integer for subject "
                    f"'{s.get('subject_name', '?')}'. Re-read that row."
                )
            mm = s.get("max_marks")
            try:
                mm_int = int(float(mm))
                s["max_marks"] = mm_int
                computed_max += mm_int
            except (TypeError, ValueError):
                errors.append(
                    f"max_marks is not a valid integer for subject "
                    f"'{s.get('subject_name', '?')}'. Re-read that row."
                )

        data["computed_total"] = computed_marks
        data["total_max_marks"] = computed_max

        # ── Step 4: Cross-check sum vs printed grand total ───────────────────
        printed = data.get("printed_grand_total")
        if printed is not None:
            try:
                printed_float = float(printed)

                # ── Percentage-confusion guard ────────────────────────────────
                # A very common model mistake on Indian marksheets: the model
                # reads the percentage (e.g. 83.60) and stores it as
                # printed_grand_total instead of the actual integer total.
                # We detect this by checking two conditions:
                #   (a) the value has a non-zero decimal component (e.g. 83.60)
                #   (b) the value is implausibly small compared to total_max_marks
                #       (e.g. 83 when subject max_marks sum to 750 makes no sense
                #        as a raw total)
                #
                # When detected: move the value to percentage_or_cgpa (if not
                # already set), clear printed_grand_total so we don't fail on it,
                # and let the computed_total stand as the best estimate.
                raw_int = int(printed_float)
                has_decimal = (printed_float - raw_int) != 0.0
                is_implausibly_small = (raw_int < computed_max // 2) and computed_max > 0

                if has_decimal or is_implausibly_small:
                    # The value looks like a percentage, not an integer total.
                    # If percentage_or_cgpa is not already filled, use this value.
                    if not data.get("percentage_or_cgpa"):
                        data["percentage_or_cgpa"] = str(printed_float)
                    data["printed_grand_total"] = None
                    data["data_quality_notes"] = list(data.get("data_quality_notes") or []) + [
                        f"printed_grand_total value '{printed}' appears to be a percentage "
                        f"(it is {'decimal' if has_decimal else 'too small compared to total_max_marks'}) — "
                        "moved to percentage_or_cgpa. printed_grand_total set to null. "
                        "Rule 13 of the extraction prompt explains this distinction."
                    ]
                    # No further mismatch check — we don't have a reliable total to compare.
                    # The computed_total from subjects is our best available figure.
                else:
                    # The value looks like a genuine integer total — do the mismatch check.
                    printed_int = raw_int
                    data["printed_grand_total"] = printed_int
                    diff = abs(computed_marks - printed_int)
                    pct_diff = diff / max(printed_int, 1) * 100

                    if diff == 0:
                        pass  # perfect match

                    elif computed_marks < printed_int and pct_diff <= 25:
                        # A slightly higher printed total can be legitimate on any
                        # marksheet family when moderation or hidden components are
                        # involved, so keep the extraction but record the gap.
                        data["extraction_confidence"] = "MEDIUM"
                        data["data_quality_notes"] = list(data.get("data_quality_notes") or []) + [
                            f"Printed total note: sum of visible subject marks ({computed_marks}) differs from "
                            f"printed_grand_total ({printed_int}) by {diff} marks ({pct_diff:.1f}%). "
                            "The printed total may include moderation, grace marks, or a component that OCR did not recover exactly."
                        ]

                    elif pct_diff <= 5:
                        # When the visible subject sum is slightly above the
                        # printed total, the table is usually one digit off in an
                        # otherwise readable scan. Ask for a re-read instead of
                        # clearing the extraction immediately.
                        errors.append(
                            f"Marks total mismatch: sum of subjects marks_obtained = {computed_marks} "
                            f"but printed_grand_total = {printed_int} (difference: {diff}). "
                            "Re-read the marks table row by row and correct the misread digit. "
                            "Remember: the max-marks row values must NOT be used as marks_obtained. "
                            "DO NOT adjust numbers to force the total — only fix genuine OCR errors."
                        )

                    else:
                        # Large mismatch (>threshold) — the table is fundamentally misread.
                        # Fail safely: clear subjects, set LOW confidence, no retry.
                        data["subjects"] = []
                        data["extraction_confidence"] = "LOW"
                        data["data_quality_notes"] = list(data.get("data_quality_notes") or []) + [
                            f"Large marks mismatch ({pct_diff:.1f}% off): computed {computed_marks} "
                            f"vs printed {printed_int}. The marks table could not be read reliably. "
                            "subjects cleared to avoid storing fabricated data (safe-fail)."
                        ]
                        # Do NOT return errors — no retry, safe-fail applied
                        return errors

            except (ValueError, TypeError):
                errors.append("printed_grand_total is not a valid number. Extract it as a plain integer.")

    elif doc_type == "Degree Certificate":
        if not data.get("university_name"):
            errors.append("University name is missing. It is usually at the top of the certificate.")
        if not data.get("specialization_or_major"):
            errors.append("Specialization / branch is missing (e.g. Computer Science, Psychology). Please re-read.")

    return errors


def _validate_payslip(data: Dict[str, Any]) -> List[str]:
    """Cross-check payslip numeric totals to catch model misreads.

    We do two independent arithmetic checks that mirror what ChatGPT described:
      1. Sum of all earnings components (allowances dict) should match gross_salary.
      2. gross_salary minus total_deductions should match net_salary.

    If either check fails we return an error asking the model to re-read.
    We also auto-calculate total_deductions when the model left it null so
    the cross-check can still run.
    """
    errors: List[str] = []
    allowances = data.get("allowances") or {}
    deductions = data.get("deductions") or {}
    gross = data.get("gross_salary")
    net = data.get("net_salary")
    total_ded = data.get("total_deductions")

    # ── 1. Check sum(allowances) == gross_salary ───────────────────────────
    # If the model extracted individual earnings components, their sum should
    # equal the gross pay printed on the payslip.  Allow ±1 for rounding.
    if allowances and gross is not None:
        try:
            computed_gross = sum(float(v) for v in allowances.values() if v is not None)
            if computed_gross > 0 and abs(computed_gross - float(gross)) > 1.0:
                errors.append(
                    f"Earnings mismatch: sum of all allowances is {computed_gross:.2f} "
                    f"but gross_salary is {gross}. Re-read the earnings table and fix the component values."
                )
        except (TypeError, ValueError):
            pass

    # ── 2. Derive total_deductions from deductions dict if model left it null ─
    # The model sometimes forgets this field. Compute it from individual items.
    if total_ded is None and deductions:
        try:
            computed_ded = sum(float(v) for v in deductions.values() if v is not None)
            if computed_ded > 0:
                data["total_deductions"] = computed_ded
                total_ded = computed_ded
        except (TypeError, ValueError):
            pass

    # ── 3. Check gross_salary - total_deductions == net_salary ────────────
    # Allow ±1 for rounding differences.
    if gross is not None and total_ded is not None and net is not None:
        try:
            expected_net = float(gross) - float(total_ded)
            if abs(expected_net - float(net)) > 1.0:
                errors.append(
                    f"Net salary mismatch: {gross} (gross) - {total_ded} (deductions) = {expected_net:.2f} "
                    f"but net_salary is {net}. Re-check gross_salary, total_deductions, and net_salary."
                )
        except (TypeError, ValueError):
            pass

    return errors


def _validate_extraction(data: Dict[str, Any], category: str) -> List[str]:
    """Validate extracted data and return a list of errors.

    Educational documents are validated with full numeric cross-checks.
    Financial/payslip documents are validated with earnings and net-pay checks.
    Other categories just confirm the extraction is non-empty.
    """
    if data.get("document_type") == "Unreadable":
        return []   # no validation needed — model itself said the image is unreadable
    if category == "educational":
        return _validate_educational(data)
    if category == "financial" and data.get("document_type") == "Payslip":
        return _validate_payslip(data)
    # For other categories just make sure we got something
    if not data:
        return ["Extraction returned an empty result."]
    return []


def _resolve_ollama() -> tuple:
    """Find a reachable Ollama endpoint and return (base_url, model).

    Uses probe_ollama() from the shared ollama.py module so that both
    localhost:11434 and host.docker.internal:11434 (Docker environments) are
    tried in order.  Also picks the best available model automatically, so
    that model-name mismatches (e.g. 'gemma4:latest' vs 'gemma4:12b') are
    handled correctly.

    Returns (None, None) when Ollama is completely unreachable.
    """
    base_url, models, _ = probe_ollama()
    if not base_url:
        return None, None
    model = select_ollama_model(models)
    return base_url, model


def _ollama_chat(messages: List[Dict[str, Any]], base_url: str, model: str) -> str:
    """Send a chat request to Ollama and return the text reply.

    Raises requests.RequestException if Ollama is offline or returns an error.
    base_url and model are passed in rather than read from module-level
    constants so that the correct endpoint (localhost vs Docker host) is
    always used.
    """
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0},  # zero temperature = most deterministic output
    }
    resp = requests.post(
        f"{base_url}/api/chat",
        json=payload,
        timeout=(OLLAMA_CONNECT_TIMEOUT_SEC, OLLAMA_READ_TIMEOUT_SEC),
    )
    resp.raise_for_status()
    return str(resp.json().get("message", {}).get("content", "")).strip()


# ── Public API ──────────────────────────────────────────────────────────────

def extract_document_fields(
    source: Union[str, Path, bytes],
    doc_type: str = "generic",
    filename: str = "",
    artifact_root: Union[str, Path] = "artifacts",
) -> Dict[str, Any]:
    """Extract structured fields from a document using Gemma4.

    Parameters
    ----------
    source   : file path (str or Path) OR raw bytes of the file.
    doc_type : the document type string (e.g. 'payslip', 'marksheet').
               Used to choose the right extraction prompt.
    filename : original filename — used to figure out the file extension
               when source is raw bytes.

    Returns a dict with extracted fields.  Special keys:
      _unavailable     : True  — Ollama is offline
      _extraction_attempts : how many Gemma4 calls were made
      _validation_errors   : list of remaining errors after all retries
      error                : set if a hard exception occurred
    """
    log.info("document_extract: starting extraction", extra={"doc_type": doc_type, "source_filename": filename})

    # Step 1: Convert the file to a JPEG image that Gemma4 can read visually.
    # Marksheet OCR artifacts also reuse this render path, so conversion happens
    # before we decide whether an Ollama call is needed.
    img_b64 = _file_to_jpeg_b64(source, filename=filename or (str(source) if not isinstance(source, bytes) else ""))
    if not img_b64:
        log.warning("document_extract: could not convert file to image", extra={"source_filename": filename})
        return {"error": "Could not convert file to an image for extraction.", "document_type": doc_type}

    # Step 2b: For PDFs, also extract raw embedded text (ChatGPT's step 1 insight).
    # Digitally-generated PDFs already have perfect text embedded — no OCR needed.
    # We send this alongside the image so Gemma4 gets accurate numbers from the
    # text AND visual layout context from the image.  Scanned PDFs return "" here
    # and fall back to image-only extraction.
    raw_pdf_text = ""
    if doc_type != "marksheet":
        raw_pdf_text = _extract_pdf_text(source, filename=filename or (str(source) if not isinstance(source, bytes) else ""))
    if raw_pdf_text:
        log.info(
            "document_extract: text-based PDF detected — embedding raw text in prompt (%d chars)",
            len(raw_pdf_text),
        )

    # Step 2c: For scanned images (where PDF text extraction yields nothing),
    # run PaddleOCR with bounding boxes and reconstruct the text row-by-row.
    # This is the user's required top-to-bottom, left-to-right OCR pass.
    # The result is a proper line-per-row text representation that preserves
    # the table structure, so Gemma4 can match labels to values reliably instead
    # of trying to parse the raw image pixels of a blurry scan.
    #
    # IMPORTANT: We render the PDF page DIRECTLY at 3× zoom for PaddleOCR,
    # rather than reusing the already-scaled Gemma4 JPEG.  The Gemma4 JPEG
    # is scaled down to max_dim=2048 and JPEG-compressed, which degrades OCR
    # quality.  The direct 3× PNG render gives PaddleOCR a cleaner image and
    # avoids the spatial scrambling that occurs at lower resolutions.
    ocr_text = ""
    ocr_engine_used = ""
    ocr_engine_comparison: List[Dict[str, Any]] = []
    layout_family = _LAYOUT_UNKNOWN
    raw_ocr_markdown_path = ""
    if doc_type == "marksheet" or not raw_pdf_text:
        try:
            # Determine the raw source bytes for rendering
            if isinstance(source, (str, Path)):
                _src_data = Path(source).read_bytes()
                _src_filename = filename or Path(source).name
            else:
                _src_data = source
                _src_filename = filename

            _src_suffix = Path(_src_filename).suffix.lower() if _src_filename else ""

            if _src_suffix == ".pdf" or _src_data[:4] == b"%PDF":
                # Render page 1 at 3× zoom directly to PNG bytes for OCR
                import fitz  # PyMuPDF
                _doc = fitz.open(stream=_src_data, filetype="pdf")
                _page = _doc.load_page(0)
                _mat = fitz.Matrix(3.0, 3.0)  # same 3× as _file_to_jpeg_b64
                _pix = _page.get_pixmap(matrix=_mat)
                ocr_image_bytes: bytes = _pix.tobytes("png")  # lossless PNG
                _doc.close()
            else:
                # For image files, decode from the already-converted JPEG
                ocr_image_bytes = base64.b64decode(img_b64)

            paddle_engine, paddle_text, paddle_markdown_text = _paddleocr_to_text(ocr_image_bytes)

            if doc_type == "marksheet":
                if not paddle_engine or not paddle_text:
                    log.error(
                        "document_extract: marksheet extraction aborted because PaddleOCR produced no usable text",
                        extra={"source_filename": filename},
                    )
                    return _attach_extraction_metadata(
                        {"error": "PaddleOCR did not produce usable marksheet text.", "document_type": doc_type},
                        layout_family=_LAYOUT_UNKNOWN,
                        ocr_engine_used="paddleocr",
                        ocr_engine_comparison=[],
                        raw_ocr_markdown_path="",
                    )

                best_candidate = _evaluate_marksheet_ocr_candidate(paddle_engine, paddle_text)
                ocr_engine_comparison = [
                    {
                        "engine": best_candidate["engine"],
                        "layout_family": best_candidate["layout_family"],
                        "layout_markers": best_candidate["layout_markers"],
                        "layout_score": round(best_candidate["layout_score"], 1),
                        "direct_score": round(best_candidate["direct_score"], 1),
                        "score": round(best_candidate["score"], 1),
                        "subject_count": len((best_candidate.get("direct_data") or {}).get("subjects") or []),
                    }
                ]
                ocr_text = best_candidate["text"]
                ocr_engine_used = best_candidate["engine"]
                layout_family = best_candidate["layout_family"]
                raw_ocr_markdown_path = _write_marksheet_ocr_markdown(
                    source,
                    filename=filename,
                    markdown_text=paddle_markdown_text or ocr_text,
                    artifact_root=artifact_root,
                )
                log.info(
                    "document_extract: selected marksheet OCR engine=%s family=%s score=%.1f",
                    ocr_engine_used,
                    layout_family,
                    best_candidate["score"],
                )
                direct_data = best_candidate.get("direct_data")
                if direct_data is not None:
                    log.info(
                        "document_extract: %s direct parse succeeded (%d subjects, conf=%s) — skipping Gemma4",
                        layout_family,
                        len(direct_data.get("subjects", [])),
                        direct_data.get("extraction_confidence"),
                    )
                    return _attach_extraction_metadata(
                        direct_data,
                        layout_family=layout_family,
                        ocr_engine_used=ocr_engine_used,
                        ocr_engine_comparison=ocr_engine_comparison,
                        raw_ocr_markdown_path=raw_ocr_markdown_path,
                    )
            else:
                # Use the spatially-aware markdown text (horizontal spacing preserved)
                # so Gemma4 can match labels to values in the same visual column.
                # paddle_text is a plain flat string that loses all table structure;
                # paddle_markdown_text keeps row spacing so e.g. "Division: First Class"
                # stays on the same line instead of being split across two.
                ocr_text = paddle_markdown_text or paddle_text
                ocr_engine_used = paddle_engine or ""

            if ocr_text:
                log.info(
                    "document_extract: %s produced %d chars of layout-aware text for scanned doc",
                    ocr_engine_used or "ocr",
                    len(ocr_text),
                )
                if doc_type == "marksheet":
                    if layout_family == _LAYOUT_UNKNOWN:
                        layout_family = _classify_marksheet_layout_family(ocr_text)["family"]
                    ocr_text = _reformat_hsc_ocr_table(ocr_text)
        except Exception as exc:
            if doc_type == "marksheet":
                log.error(
                    "document_extract: PaddleOCR marksheet extraction failed: %s",
                    exc,
                    extra={"source_filename": filename},
                )
                return _attach_extraction_metadata(
                    {"error": f"PaddleOCR marksheet extraction failed: {exc}", "document_type": doc_type},
                    layout_family=_LAYOUT_UNKNOWN,
                    ocr_engine_used="paddleocr",
                    ocr_engine_comparison=[],
                    raw_ocr_markdown_path="",
                )
            log.debug("document_extract: PaddleOCR step skipped: %s", exc)

    # Marksheet extraction is intentionally Paddle-only. Other document types
    # can still prefer embedded PDF text when the PDF already has a text layer.
    supplementary_text = ocr_text if doc_type == "marksheet" else (raw_pdf_text or ocr_text)

    if doc_type == "marksheet" and not raw_ocr_markdown_path and ocr_text:
        raw_ocr_markdown_path = _write_marksheet_ocr_markdown(
            source,
            filename=filename,
            markdown_text=ocr_text,
            artifact_root=artifact_root,
        )
    elif doc_type != "marksheet" and not raw_pdf_text and ocr_text and artifact_root:
        # For non-marksheet scanned images (degree certs, employment letters, etc.)
        # write the same kind of OCR markdown artifact so both the human reviewer
        # and any future debug step can see exactly what PaddleOCR read off the page.
        raw_ocr_markdown_path = _write_marksheet_ocr_markdown(
            source,
            filename=filename,
            markdown_text=ocr_text,
            artifact_root=artifact_root,
        )

    # Step 3: Probe all known Ollama URLs only after deterministic OCR and raw
    # markdown artifacts have had a chance to run. This keeps marksheet review
    # usable even when the local LLM is offline.
    _base_url, _model = _resolve_ollama()
    if not _base_url:
        log.warning("document_extract: Ollama is not reachable — skipping extraction")
        return _attach_extraction_metadata(
            {"_unavailable": True, "document_type": doc_type},
            layout_family=layout_family,
            ocr_engine_used=ocr_engine_used,
            ocr_engine_comparison=ocr_engine_comparison,
            raw_ocr_markdown_path=raw_ocr_markdown_path,
        )

    # Log the exact model that will be used so operators can confirm which
    # Gemma4 variant is running (e.g. gemma4:12b vs gemma4:27b).
    log.info("document_extract: using model=%s at %s for doc_type=%s", _model, _base_url, doc_type)

    # Step 4: Choose the right prompt based on document category.
    # Marksheets now always use one generic marksheet prompt, regardless of
    # whether the OCR layout looks like SSC, HSC, engineering, or another
    # board format.  Python supplies the layout-specific hints separately.
    category = _DOC_TYPE_CATEGORY.get(doc_type, "generic")
    if doc_type == "marksheet":
        prompt = _get_prompt("marksheet")
    else:
        prompt = _get_prompt(_CATEGORY_PROMPT_NAMES[category])

    if doc_type == "marksheet":
        prompt = (
            prompt
            + "\n\nDOCUMENT INTELLIGENCE HINTS:\n"
            + "document_type_hint: marksheet\n"
            + f"layout_family_hint: {layout_family}\n"
        )
        if ocr_engine_used:
            prompt += f"preferred_ocr_engine_hint: {ocr_engine_used}\n"

        # For IGNOU marksheets: try to read the liteparse artifact that the
        # pre-processing step saved next to the source file.  The liteparse
        # uses pypdf text extraction which, unlike PaddleOCR, can decode the
        # Devanagari-font header area and gives us the genuine ENROLMENT NO
        # and NAME even though PaddleOCR only sees the marks table body.
        # Pass them as verified hints so Gemma4 copies the correct values
        # rather than attempting to read the watermark-degraded header image.
        ignou_liteparse_header: Dict[str, str] = {}
        if layout_family == _LAYOUT_IGNOU_OPEN_UNIVERSITY and filename:
            try:
                _ignou_stem = Path(filename).stem
                _ignou_lp_path = Path(artifact_root) / _ignou_stem / f"{_ignou_stem}_liteparse.json"
                if _ignou_lp_path.exists():
                    import json as _json
                    _lp_data = _json.loads(_ignou_lp_path.read_text(encoding="utf-8"))
                    _lp_text = " ".join(p.get("text", "") for p in _lp_data.get("pages", []))
                    _enrol_m = re.search(r"ENROLMENT\s+NO\s*:\s*(\d{9,10})", _lp_text, re.IGNORECASE)
                    _name_m = re.search(r"\bNAME\s*:\s*([A-Z][A-Z ]{4,40}?)(?:\s{2,}|C/o|ADDRESS|\n)", _lp_text)
                    if _enrol_m:
                        ignou_liteparse_header["enrollment"] = _enrol_m.group(1)
                        log.info(
                            "document_extract: IGNOU liteparse enrollment found: %s",
                            ignou_liteparse_header["enrollment"],
                        )
                    if _name_m:
                        ignou_liteparse_header["name"] = _name_m.group(1).strip()
                        log.info(
                            "document_extract: IGNOU liteparse name found: %s",
                            ignou_liteparse_header["name"],
                        )
            except Exception as _ignou_lp_exc:
                log.debug("document_extract: IGNOU liteparse header read failed: %s", _ignou_lp_exc)

        structure_hint = (
            _build_marksheet_structure_hint(
                supplementary_text, layout_family, ignou_liteparse_header=ignou_liteparse_header or None
            )
            if supplementary_text
            else ""
        )
        if structure_hint:
            prompt += structure_hint + "\n"

    # If we have any supplementary text (embedded PDF text or PaddleOCR), append it
    # to the prompt.  We cap at 6000 chars to stay within the model's context window.
    # The section header tells Gemma4 where the text came from so it knows how much
    # to trust it (embedded text is lossless; OCR text may have minor noise).
    if supplementary_text:
        if raw_pdf_text:
            text_source_label = "RAW EMBEDDED TEXT FROM THE PDF (perfectly accurate — trust these numbers over the image)"
            trust_note = (
                "IMPORTANT: The text above was extracted directly from the PDF without any OCR. "
                "Trust these numbers exactly as written. The image shows the layout; "
                "the text above gives you the exact characters and values."
            )
        else:
            text_source_label = "OCR TEXT EXTRACTED FROM THE SCANNED IMAGE (row-by-row layout preserved)"
            trust_note = (
                "IMPORTANT: The text above was extracted with the selected OCR engine, preserving one row per line "
                "from the original document. Use it to identify the marks table rows and column values. "
                "For numbers that look like OCR noise (e.g. 'l' instead of '1'), use the image to verify. "
                "If the table rows are unclear even from this text, set subjects=[] and confidence=LOW."
            )

        prompt = (
            prompt
            + f"\n\n---\n{text_source_label}:\n"
            + supplementary_text[:6000]
            + f"\n---\n{trust_note}"
        )

    log.debug(
        "document_extract: category=%s doc_type=%s pdf_text=%s ocr_text=%s",
        category, doc_type, bool(raw_pdf_text), bool(ocr_text),
    )

    # Step 5: Build the initial chat message to Gemma4.
    # We always include the image and the OCR-derived structure hint together.
    # PaddleOCR gives row and column clues, while the image still helps the
    # model resolve ambiguous tokens and confirm the visual table layout.
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": _get_prompt("system")},
        {"role": "user", "content": prompt, "images": [img_b64]},
    ]

    # Step 6: Send to Gemma4, validate, retry once if needed
    for attempt in range(MAX_RETRIES + 1):
        log.debug("document_extract: attempt %d/%d", attempt + 1, MAX_RETRIES + 1)
        try:
            raw_text = _ollama_chat(messages, _base_url, _model)
            json_str = _extract_json_from_text(raw_text)

            if not json_str:
                raise ValueError("No JSON object found in Gemma4 response.")

            data = json.loads(json_str)
            errors = _validate_extraction(data, category)

            if not errors:
                # All good — return the clean extraction
                data["_extraction_attempts"] = attempt + 1
                log.info(
                    "document_extract: extraction succeeded",
                    extra={"doc_type": doc_type, "attempts": attempt + 1},
                )
                return _attach_extraction_metadata(
                    data,
                    layout_family=layout_family,
                    ocr_engine_used=ocr_engine_used,
                    ocr_engine_comparison=ocr_engine_comparison,
                    raw_ocr_markdown_path=raw_ocr_markdown_path,
                )

            # There are errors — if we have retries left, ask Gemma4 to fix them
            if attempt < MAX_RETRIES:
                log.debug("document_extract: validation errors on attempt %d: %s", attempt + 1, errors)
                messages.append({"role": "assistant", "content": raw_text})
                messages.append({
                    "role": "user",
                    "content": (
                        f"Your extraction had the following issues:\n{json.dumps(errors, indent=2)}\n\n"
                        "Please re-examine the image carefully and correct these mistakes. "
                        "Return the FULL updated JSON only."
                    ),
                })
            else:
                # Final attempt still has errors — return what we got with error notes
                data["_validation_errors"] = errors
                data["_extraction_attempts"] = attempt + 1
                log.warning(
                    "document_extract: extraction finished with validation errors",
                    extra={"doc_type": doc_type, "errors": errors},
                )
                return _attach_extraction_metadata(
                    data,
                    layout_family=layout_family,
                    ocr_engine_used=ocr_engine_used,
                    ocr_engine_comparison=ocr_engine_comparison,
                    raw_ocr_markdown_path=raw_ocr_markdown_path,
                )

        except Exception as exc:
            log.warning("document_extract: attempt %d failed: %s", attempt + 1, exc)
            if attempt == MAX_RETRIES:
                return _attach_extraction_metadata(
                    {"error": str(exc), "_extraction_attempts": attempt + 1, "document_type": doc_type},
                    layout_family=layout_family,
                    ocr_engine_used=ocr_engine_used,
                    ocr_engine_comparison=ocr_engine_comparison,
                    raw_ocr_markdown_path=raw_ocr_markdown_path,
                )
            # Tell Gemma4 it made a JSON error, ask it to try again
            messages.append({"role": "user", "content": f"Error parsing your JSON: {exc}. Return ONLY valid JSON."})

    # Should never reach here, but just in case
    return _attach_extraction_metadata(
        {"error": "Failed after maximum retries", "document_type": doc_type},
        layout_family=layout_family,
        ocr_engine_used=ocr_engine_used,
        ocr_engine_comparison=ocr_engine_comparison,
        raw_ocr_markdown_path=raw_ocr_markdown_path,
    )
