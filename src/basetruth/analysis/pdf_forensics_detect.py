"""
pdf_forensics_detect.py  —  Universal PDF tampering detector.

Designed to complement forensics_detect.py (which handles images).
No LLM dependency.  Applies 11 forensic layers that are meaningful for PDF
documents — offer letters, payslips, certificates, bank statements, etc.
Output JSON is structured to be passed directly to an LLM for narrative analysis.

Usage:
    python pdf_forensics_detect.py <pdf_path> [<reference_pdf_path>]

Single-file mode:  absolute signal scoring.
Two-file mode:     peer-relative comparison (higher score = more suspicious).

Techniques applied to every PDF; gracefully skipped where not applicable:
  1  Incremental Update Detection  %%EOF count → incremental saves = tampering
  2  Metadata Analysis             Creator/Producer/dates, editing-tool fingerprinting
  3  Font Consistency              Embedded vs. non-embedded, unique families per page
  4  Invisible / Hidden Text       White / zero-size text that hides content
  5  Suspicious Object Detection   JavaScript, embedded files, OpenAction, XFA forms
  6  Content Consistency           Page count, sizes, blank pages, text-density per page
  7  Digital Signature Check       Presence / absence of digital signature
  8  Page Render ELA               ELA on rasterized page-1 to catch pixel-level tampering
  9  Embedded Image Analysis       Noise residual on images extracted from PDF streams
  10 File Entropy                  Shannon entropy of raw PDF bytes
  11 Object / XRef Integrity       pikepdf object count, ObjStm, orphaned xrefs

Outputs (written alongside the first input PDF):
    pdf_forensics_report.json
    ela_page1_<label>.png         — ELA heatmap of rasterized page 1
    noise_page1_<label>.png       — Noise heatmap of rasterized page 1
    embedded_imgs_<label>.png     — Noise heatmap of largest embedded image (when present)
"""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import sys
import tempfile
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from basetruth.logger import get_logger

log = get_logger(__name__)

import cv2
import fitz  # PyMuPDF
import numpy as np
import pikepdf
from PIL import Image, ImageChops

warnings.filterwarnings("ignore")


# ── configuration constants ────────────────────────────────────────────────────

# JPEG quality used as ELA re-compression baseline (same as forensics_detect.py)
ELA_QUALITY  = 75
ELA_AMPLIFY  = 10

# DPI used when rasterizing a PDF page to run image-level checks on it
PAGE_RENDER_DPI = 150

# Tools known to be PDF editors / online converters — presence flags tampering risk
_PDF_EDIT_TOOLS = [
    "ilovepdf", "smallpdf", "sejda", "pdf24", "pdfzorro", "adobe acrobat",
    "pdfescape", "foxit phantom", "nitro", "pdfelement",
    "pdf editor", "edit pdf", "modify pdf", "split pdf", "merge pdf",
    "compress pdf", "rotate pdf", "pdf candy", "pdfsam",
]

# Tools that are legitimate document-creation / publishing tools (not edit-flags)
_PDF_CREATE_TOOLS = [
    "microsoft word", "google docs", "libreoffice", "openoffice",
    "reportlab", "itext", "fpdf", "fpdf2", "wkhtmltopdf",
    "pdfkit", "weasyprint", "quartz", "ghostscript",
    "adobe indesign", "adobe illustrator", "illustrator", "indesign",
    "texlive", "miktex", "latex", "xelatex", "pdflatex",
    "crystal reports", "greytip", "kofax",
    "pandoc", "aspose",
]


# ── helpers ────────────────────────────────────────────────────────────────────

def _label(path: str) -> str:
    """Return a filesystem-safe label (filename without extension)."""
    return Path(path).stem


def _out_dir(path: str) -> str:
    """Return the directory of the given path."""
    return str(Path(path).parent.resolve())


def _parse_pdf_date(date_str: str) -> datetime | None:
    """Parse a PDF date string (D:YYYYMMDDHHmmss[±HH'mm']) into a datetime.

    PDF dates look like: D:20231015120000+05'30' or D:20231015120000Z
    Returns None when the string is absent or malformed.
    """
    if not date_str:
        return None
    # Strip the D: prefix that fitz sometimes leaves in
    clean = re.sub(r"^D:", "", date_str.strip())
    # Keep only the 14 numeric characters (YYYYMMDDHHmmss)
    digits = re.match(r"(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})", clean)
    if not digits:
        # Try 8-digit date only (YYYYMMDD)
        d8 = re.match(r"(\d{4})(\d{2})(\d{2})", clean)
        if not d8:
            return None
        try:
            return datetime(int(d8.group(1)), int(d8.group(2)), int(d8.group(3)),
                            tzinfo=timezone.utc)
        except ValueError:
            return None
    try:
        return datetime(
            int(digits.group(1)), int(digits.group(2)), int(digits.group(3)),
            int(digits.group(4)), int(digits.group(5)), int(digits.group(6)),
            tzinfo=timezone.utc,
        )
    except ValueError:
        return None


# ── Layer 1: Incremental Update Detection ────────────────────────────────────

def incremental_update_analysis(pdf_path: str) -> dict:
    """Detect incremental saves — a leading indicator of post-creation tampering.

    SIMPLE EXPLANATION:
    The PDF format normally has a single 'end-of-file' marker (%%EOF) at the bottom.
    When someone opens a PDF, types a new number (say, altering a salary figure),
    and saves the file, the editing tool APPENDS the changed data before a second
    %%EOF marker instead of rewriting the whole file.  This is called an 'incremental
    update'.  Counting %%EOF markers tells us how many times the file has been
    saved after it was originally created.  A legitimate payslip or offer letter
    generated by payroll/HR software should have exactly ONE %%EOF.  Two or more
    means something changed after creation.
    """
    raw = Path(pdf_path).read_bytes()

    # Count %%EOF markers (each incremental save adds one)
    eof_count = raw.count(b"%%EOF")

    # startxref keyword appears once per creation + once per incremental update
    startxref_count = len(re.findall(rb"startxref", raw))

    # PDF version string — e.g. %PDF-1.7
    ver_match = re.search(rb"%PDF-(\d+\.\d+)", raw[:64])
    pdf_version = ver_match.group(1).decode() if ver_match else "unknown"

    # Number of distinct xref tables or xref streams
    xref_kw_count = len(re.findall(rb"\bxref\b", raw))

    # Incremental updates = extra EOF markers beyond the required one
    incremental_updates = max(0, eof_count - 1)

    suspicious_flags = []
    if incremental_updates > 0:
        suspicious_flags.append(
            f"PDF was saved {incremental_updates} time(s) after initial creation "
            f"(incremental updates detected — common when editing salary/date fields)"
        )
    if xref_kw_count > 1 and incremental_updates == 0:
        # Multiple xref tables but only one EOF: could be xref streams (PDF 1.5+)
        # This is normal in newer PDFs, but worth recording.
        suspicious_flags.append(
            f"Multiple xref tables ({xref_kw_count}) — may indicate cross-reference stream format "
            f"(normal in PDF ≥ 1.5, but confirm alongside other signals)"
        )

    return {
        "pdf_version":         pdf_version,
        "file_size_bytes":     len(raw),
        "eof_marker_count":    eof_count,
        "startxref_count":     startxref_count,
        "xref_table_count":    xref_kw_count,
        "incremental_updates": incremental_updates,
        "suspicious_flags":    suspicious_flags,
        "interpretation": (
            f"CRITICAL — {incremental_updates} incremental update(s) detected "
            f"(file modified after creation; common tampering vector)"
            if incremental_updates > 0 else
            "CLEAN — single creation event, no incremental modifications detected"
        ),
    }


# ── Layer 2: Metadata Analysis ────────────────────────────────────────────────

def metadata_analysis(pdf_path: str) -> dict:
    """Analyse PDF Information Dictionary to fingerprint creation tool and detect edits.

    SIMPLE EXPLANATION:
    Every PDF carries a hidden information card (like a label on the back of a photo)
    listing who created it, which software generated it, and when it was created and
    last modified.  We check three things:
    (a) Was an online PDF editor or a known manipulation tool used at any point?
    (b) Is the modification date later than the creation date — and by how many days?
    (c) Was the metadata completely stripped (which is suspicious, as a pay-slipped
        by payroll software always keeps its creation details)?
    """
    doc = fitz.open(pdf_path)
    meta = doc.metadata or {}
    doc.close()

    creation_date = meta.get("creationDate") or ""
    mod_date      = meta.get("modDate")      or ""
    author        = meta.get("author")       or ""
    title         = meta.get("title")        or ""
    subject       = meta.get("subject")      or ""
    keywords      = meta.get("keywords")     or ""
    creator       = meta.get("creator")      or ""   # application that made the source doc
    producer      = meta.get("producer")     or ""   # PDF library that produced the file

    suspicious_flags = []

    # ── Check for known PDF-editing tools in Creator or Producer ──────────────
    # We check both fields because some editors replace Producer, others modify Creator.
    for field_name, field_val in [("Creator", creator), ("Producer", producer)]:
        lower_val = field_val.lower()
        for tool in _PDF_EDIT_TOOLS:
            if tool in lower_val:
                suspicious_flags.append(
                    f"PDF editing/manipulation tool found in {field_name}: '{field_val}'"
                )
                break

    # ── Date gap check ─────────────────────────────────────────────────────────
    creation_dt = _parse_pdf_date(creation_date)
    mod_dt      = _parse_pdf_date(mod_date)
    gap_days: int | None = None

    if creation_dt and mod_dt:
        gap = (mod_dt - creation_dt).total_seconds()
        if gap > 60:  # more than 60 seconds — genuine edit
            gap_days = int(gap // 86400)
            suspicious_flags.append(
                f"ModDate is {gap_days} day(s) after CreationDate — "
                f"document was modified after it was originally generated"
            )

    # ── Absent metadata ────────────────────────────────────────────────────────
    has_any_meta = any([creation_date, mod_date, author, title, creator, producer])
    if not has_any_meta:
        suspicious_flags.append(
            "PDF metadata is completely absent — likely stripped by an online PDF tool "
            "or generated by a script that omits document information"
        )
    if has_any_meta and not creation_date:
        suspicious_flags.append(
            "No CreationDate — metadata was partially stripped after creation"
        )

    return {
        "title":          title,
        "author":         author,
        "subject":        subject,
        "keywords":       keywords,
        "creator":        creator,
        "producer":       producer,
        "creation_date":  creation_date,
        "mod_date":       mod_date,
        "date_gap_days":  gap_days,
        "suspicious_flags": suspicious_flags,
        "tamper_risk": (
            "HIGH"   if len(suspicious_flags) >= 2 else
            "MEDIUM" if suspicious_flags else
            "LOW"
        ),
        "interpretation": (
            f"HIGH RISK — {'; '.join(suspicious_flags)}"
            if len(suspicious_flags) >= 2 else
            f"MEDIUM RISK — {suspicious_flags[0]}"
            if suspicious_flags else
            f"CLEAN — Creator: '{creator}', Producer: '{producer}', "
            f"no editing tools or date anomalies detected"
        ),
    }


# ── Layer 3: Font Consistency Analysis ───────────────────────────────────────

def font_analysis(pdf_path: str) -> dict:
    """Check font embedding consistency across all pages.

    SIMPLE EXPLANATION:
    A genuine payslip or offer letter is generated by one piece of software in
    one pass: every character uses the same font family and all fonts are either
    fully embedded in the PDF or all referenced externally.  If someone opened
    the PDF in a viewer, typed new text (e.g. changing a salary figure), and
    re-saved it, the newly typed text will typically use a *different* font or
    embedding style than the original.  We catalogue every font used across every
    page and flag mismatches in embedding status or unexpected font family changes.
    """
    doc = fitz.open(pdf_path)

    # Collect font info per page.
    # fitz.page.get_fonts() → list of (xref, ext, type, basefont, name, encoding, referencer)
    all_fonts: dict[str, dict] = {}   # basefont → {type, embedded, pages}
    non_embedded: list[str] = []
    page_font_sets: list[set[str]] = []

    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        fonts = page.get_fonts(full=True)
        page_set: set[str] = set()

        for _xref, ext, font_type, basefont, name, encoding, _ref in fonts:
            # The font key is basefont if available, otherwise name.
            key = (basefont or name or "unknown").strip("/")
            is_embedded = bool(ext)  # ext is non-empty when the font data is embedded

            if key not in all_fonts:
                all_fonts[key] = {
                    "type":      font_type,
                    "encoding":  encoding,
                    "embedded":  is_embedded,
                    "pages":     [],
                }
            all_fonts[key]["pages"].append(page_num + 1)

            if not is_embedded and key not in non_embedded:
                non_embedded.append(key)

            page_set.add(key)

        page_font_sets.append(page_set)

    doc.close()

    unique_families = list(all_fonts.keys())
    n_non_embedded  = len(non_embedded)
    n_total         = len(unique_families)

    # Check if fonts used on page 1 differ from later pages — a signal that
    # additions were made in a different tool than the original.
    page1_only_fonts: list[str] = []
    other_only_fonts: list[str] = []
    if len(page_font_sets) > 1:
        p1 = page_font_sets[0]
        rest = set().union(*page_font_sets[1:])
        page1_only_fonts = sorted(p1 - rest)
        other_only_fonts = sorted(rest - p1)

    suspicious_flags: list[str] = []
    if n_non_embedded > 0:
        pct = round(n_non_embedded / max(n_total, 1) * 100, 1)
        suspicious_flags.append(
            f"{n_non_embedded}/{n_total} font(s) not embedded ({pct}%) — "
            f"text may render differently on other machines and could indicate re-saving"
        )
    if other_only_fonts and len(other_only_fonts) <= 4:
        suspicious_flags.append(
            f"Font(s) appear only on later pages, not on page 1: "
            f"{', '.join(other_only_fonts[:4])} — possible post-creation insertion"
        )
    if n_total > 10:
        suspicious_flags.append(
            f"Unusually high number of distinct font families ({n_total}) — "
            f"typical in merge/paste operations"
        )

    # Summarise font list (cap at 15 for readability)
    font_summary = [
        {
            "name":     k,
            "type":     v["type"],
            "embedded": v["embedded"],
            "encoding": v["encoding"],
            "pages":    v["pages"][:5],
        }
        for k, v in list(all_fonts.items())[:15]
    ]

    return {
        "total_unique_fonts":    n_total,
        "non_embedded_count":    n_non_embedded,
        "non_embedded_names":    non_embedded[:10],
        "page1_exclusive_fonts": page1_only_fonts[:5],
        "later_pages_only_fonts": other_only_fonts[:5],
        "fonts":                 font_summary,
        "suspicious_flags":      suspicious_flags,
        "interpretation": (
            f"SUSPICIOUS — {'; '.join(suspicious_flags)}"
            if suspicious_flags else
            f"CONSISTENT — {n_total} font(s), all properly embedded or uniformly referenced"
        ),
    }


# ── Layer 4: Invisible / Hidden Text Detection ────────────────────────────────

def invisible_text_analysis(pdf_path: str) -> dict:
    """Detect text that exists in the PDF but is invisible to the reader.

    SIMPLE EXPLANATION:
    One clever way to hide tampered data inside a PDF is to place 'invisible' text
    behind the visible content.  This can be done by setting text colour to white
    (matching the page background), making the font size near-zero (0.001 pt), or
    using PDF rendering mode 3 which explicitly tells the viewer to skip drawing.
    We scan every text span on every page for these conditions.  In a legitimate
    payslip or offer letter there should be zero hidden text.
    """
    doc = fitz.open(pdf_path)

    total_spans      = 0
    white_spans      = 0
    tiny_size_spans  = 0
    shadow_spans     = 0
    examples: list[dict] = []

    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        # "dict" format gives per-span colour and size information
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE).get("blocks", [])
        page_spans_list = []

        for block in blocks:
            if block.get("type") != 0:  # type 0 = text block
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    total_spans += 1
                    text  = span.get("text", "").strip()
                    color = span.get("color", 0)   # 24-bit RGB packed int
                    size  = span.get("size", 1.0)
                    bbox  = span.get("bbox")

                    if text and bbox:
                        page_spans_list.append({"text": text, "bbox": bbox, "color": color, "size": size})

                    # White text on white background — colour == 0xFFFFFF (16777215)
                    # Near-white: R,G,B all > 240 → a packed int > 0xF0F0F0 (15790320)
                    r = (color >> 16) & 0xFF
                    g = (color >> 8)  & 0xFF
                    b = color & 0xFF
                    is_white = (r > 240 and g > 240 and b > 240)

                    if is_white and text:
                        white_spans += 1
                        if len(examples) < 5:
                            examples.append({
                                "page":   page_num + 1,
                                "text":   text[:80],
                                "reason": "white-coloured text (invisible on white background)",
                                "color":  f"#{color:06X}",
                                "size":   round(size, 2),
                            })

                    # Zero / near-zero font size (text with size < 0.5 pt is invisible)
                    if size < 0.5 and text:
                        tiny_size_spans += 1
                        if len(examples) < 5:
                            examples.append({
                                "page":   page_num + 1,
                                "text":   text[:80],
                                "reason": f"near-zero font size ({size:.3f} pt)",
                                "color":  f"#{color:06X}",
                                "size":   round(size, 3),
                            })

        # Check for shadow attacks: overlapping text with different content
        for i in range(len(page_spans_list)):
            for j in range(i + 1, len(page_spans_list)):
                s1, s2 = page_spans_list[i], page_spans_list[j]
                t1, t2 = s1["text"], s2["text"]
                
                if t1 == t2 or t1 in t2 or t2 in t1:
                    continue
                    
                b1, b2 = s1["bbox"], s2["bbox"]
                
                x_left = max(b1[0], b2[0])
                y_top = max(b1[1], b2[1])
                x_right = min(b1[2], b2[2])
                y_bottom = min(b1[3], b2[3])
                
                if x_right > x_left and y_bottom > y_top:
                    inter_area = (x_right - x_left) * (y_bottom - y_top)
                    area1 = max(0, b1[2] - b1[0]) * max(0, b1[3] - b1[1])
                    area2 = max(0, b2[2] - b2[0]) * max(0, b2[3] - b2[1])
                    min_area = min(area1, area2)
                    
                    if min_area > 0 and (inter_area / min_area) > 0.8:
                        shadow_spans += 1
                        if len(examples) < 10:
                            examples.append({
                                "page": page_num + 1,
                                "text": f"'{t1[:20]}' overlapping with '{t2[:20]}'",
                                "reason": "overlapping text bounds (shadow attack padding/edit)",
                                "color": f"#{max(s1['color'], s2['color']):06X}",
                                "size": round((s1['size'] + s2['size']) / 2, 2)
                            })

    doc.close()

    total_hidden = white_spans + tiny_size_spans + shadow_spans
    suspicious_flags: list[str] = []
    if white_spans > 0:
        suspicious_flags.append(
            f"{white_spans} span(s) of white text found — hidden content on page background"
        )
    if tiny_size_spans > 0:
        suspicious_flags.append(
            f"{tiny_size_spans} span(s) with near-zero font size — invisible text fragments"
        )
    if shadow_spans > 0:
        suspicious_flags.append(
            f"{shadow_spans} span(s) overlapping with different text — potential shadow edit mask"
        )

    return {
        "total_text_spans":   total_spans,
        "white_text_spans":   white_spans,
        "tiny_size_spans":    tiny_size_spans,
        "total_hidden_spans": total_hidden,
        "examples":           examples,
        "suspicious_flags":   suspicious_flags,
        "interpretation": (
            f"SUSPICIOUS — {'; '.join(suspicious_flags)}"
            if suspicious_flags else
            f"CLEAN — {total_spans} visible text spans, no hidden / invisible text found"
        ),
    }


# ── Layer 5: Suspicious Object Detection ─────────────────────────────────────

def suspicious_objects_analysis(pdf_path: str) -> dict:
    """Detect JavaScript, auto-actions, embedded files, and encryption.

    SIMPLE EXPLANATION:
    A genuine salary slip or offer letter is a dead-simple PDF: text, a logo image,
    and maybe a table.  It should NOT contain JavaScript (mini programs that can
    auto-fill fields), embedded executables, or form Actions that fire automatically
    when you open the file.  We scan the raw bytes of the PDF for these patterns
    because they are sometimes used to dynamically generate or alter displayed values
    — for example, a script that shows a different salary figure than what is stored
    in the PDF source text.
    """
    raw = Path(pdf_path).read_bytes()
    findings: list[str] = []

    # ── JavaScript ─────────────────────────────────────────────────────────────
    js_hits = len(re.findall(rb"/JS[\s\n\r(/<]", raw))
    jscript_hits = len(re.findall(rb"/JavaScript[\s\n\r(/<]", raw))
    total_js = js_hits + jscript_hits
    if total_js > 0:
        findings.append(
            f"JavaScript embedded in PDF ({total_js} occurrence(s)) — "
            f"legitimate HR documents should contain zero JavaScript"
        )

    # ── Embedded files ─────────────────────────────────────────────────────────
    ef_hits = len(re.findall(rb"/EmbeddedFile", raw))
    if ef_hits > 0:
        findings.append(
            f"{ef_hits} embedded file(s) found — suspicious if this is a plain payslip or letter"
        )

    # ── OpenAction (auto-runs on open) ──────────────────────────────────────────
    open_action = bool(re.search(rb"/OpenAction[\s\n\r/<]", raw))
    if open_action:
        findings.append(
            "OpenAction found — the PDF performs an action automatically when opened "
            "(not expected in plain HR documents)"
        )

    # ── Additional Actions (form triggers) ────────────────────────────────────
    aa_hits = len(re.findall(rb"/AA[\s\n\r/<]", raw))
    if aa_hits > 0:
        findings.append(
            f"/AA (additional actions / form triggers — {aa_hits} occurrence(s)) — "
            f"used in interactive forms; unexpected in payslips or letters"
        )

    # ── XFA dynamic form ─────────────────────────────────────────────────────
    xfa_present = b"/XFA" in raw
    if xfa_present:
        findings.append(
            "XFA form structure detected — a dynamic XML-based form layer that can display "
            "different values from what is stored in the static text layer"
        )

    # ── Encryption ────────────────────────────────────────────────────────────
    is_encrypted = bool(re.search(rb"/Encrypt[\s\n\r/<]", raw))
    if is_encrypted:
        findings.append("Encryption dictionary present — may restrict forensic analysis")

    # ── Launch / URI actions (can redirect to malicious pages) ────────────────
    launch_hits = len(re.findall(rb"/Launch[\s\n\r/<]", raw))
    uri_hits    = len(re.findall(rb"/URI[\s\n\r/<]", raw))
    if launch_hits > 0:
        findings.append(f"/Launch action found ({launch_hits}) — triggers external execution on click")

    return {
        "javascript_count":       total_js,
        "embedded_files_count":   ef_hits,
        "has_open_action":        open_action,
        "has_xfa_form":           xfa_present,
        "is_encrypted":           is_encrypted,
        "additional_actions":     aa_hits,
        "launch_actions":         launch_hits,
        "uri_actions":            uri_hits,
        "findings":               findings,
        "interpretation": (
            "HIGH RISK — " + "; ".join(findings)
            if len(findings) >= 2 else
            "MEDIUM RISK — " + findings[0]
            if findings else
            "CLEAN — no JavaScript, embedded files, auto-actions, or dynamic forms detected"
        ),
    }


# ── Layer 6: Content Consistency ──────────────────────────────────────────────

def content_consistency_analysis(pdf_path: str) -> dict:
    """Check page-level structural consistency — sizes, blank pages, text density.

    SIMPLE EXPLANATION:
    A payslip generated in one pass by a payroll system will have:
    — a consistent A4 / letter page size throughout
    — no completely blank pages (which might hide reference or original content)
    — a relatively uniform text density per page
    If the page sizes change mid-document, or one page suddenly has much more text than
    all others, those are signs the file may have been assembled from different sources.
    """
    doc = fitz.open(pdf_path)
    page_count  = doc.page_count

    page_info: list[dict] = []
    page_sizes: list[tuple] = []

    for page_num in range(page_count):
        page = doc.load_page(page_num)
        rect = page.rect
        w = round(rect.width,  1)
        h = round(rect.height, 1)
        text = page.get_text()
        char_count = len(text)
        image_list = page.get_images()
        page_sizes.append((w, h))
        page_info.append({
            "page":         page_num + 1,
            "width_pt":     w,
            "height_pt":    h,
            "char_count":   char_count,
            "image_count":  len(image_list),
            "is_blank":     char_count < 5,
        })

    doc.close()

    unique_sizes = list(set(page_sizes))
    blank_pages  = [p["page"] for p in page_info if p["is_blank"]]

    # Detect pages with anomalous text density (±3 standard deviations from mean)
    char_counts = [p["char_count"] for p in page_info]
    density_anomalies: list[int] = []
    if len(char_counts) > 2:
        mean_c = float(np.mean(char_counts))
        std_c  = float(np.std(char_counts))
        density_anomalies = [
            page_info[i]["page"]
            for i, c in enumerate(char_counts)
            if std_c > 0 and abs(c - mean_c) > 3 * std_c
        ]

    suspicious_flags: list[str] = []
    if len(unique_sizes) > 2:
        suspicious_flags.append(
            f"Inconsistent page sizes: {len(unique_sizes)} different dimensions found "
            f"— suggests the document was assembled from multiple sources"
        )
    if blank_pages:
        suspicious_flags.append(
            f"Blank page(s) detected at position(s): {blank_pages} "
            f"— may conceal original content or act as separator pages"
        )
    if density_anomalies:
        suspicious_flags.append(
            f"Text-density outlier page(s): {density_anomalies} "
            f"— unusual amounts of text vs rest of document"
        )

    return {
        "page_count":          page_count,
        "unique_page_sizes":   [list(s) for s in unique_sizes],
        "blank_pages":         blank_pages,
        "text_density_anomalies": density_anomalies,
        "total_chars_in_doc":  sum(char_counts),
        "pages":               page_info,
        "suspicious_flags":    suspicious_flags,
        "interpretation": (
            f"SUSPICIOUS — {'; '.join(suspicious_flags)}"
            if suspicious_flags else
            f"CONSISTENT — {page_count} page(s), uniform page size, "
            f"no blank pages, text density normal"
        ),
    }


# ── Layer 7: Digital Signature Check ─────────────────────────────────────────

def digital_signature_analysis(pdf_path: str) -> dict:
    """Detect the presence or absence of a digital signature.

    SIMPLE EXPLANATION:
    A digital signature in a PDF works like a tamper-evident wax seal: it
    covers (via a 'ByteRange') the exact bytes of the document at signing time.
    Any subsequent edit (adding a page, changing a number) will invalidate the
    seal.  Absence of a signature on a document that claims to be official is
    worth noting.  Presence of a signature that covers only part of the file
    (ByteRange gap) is highly suspicious — it means the document was modified
    AFTER it was signed.
    """
    raw = Path(pdf_path).read_bytes()

    # Look for signature field type
    has_sig_field = bool(re.search(rb"/Sig[\s\n\r(<]", raw))

    # ByteRange appears in signature dictionaries and specifies which bytes are covered
    byte_range_matches = re.findall(
        rb"/ByteRange\s*\[\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*\]", raw
    )

    signatures: list[dict] = []
    coverage_gaps: list[dict] = []
    file_size = len(raw)

    for match in byte_range_matches:
        b0, l0, b1, l1 = (int(x) for x in match)
        covered = l0 + l1           # bytes covered by this signature
        total   = b1 + l1           # how far the coverage extends
        gap     = file_size - total # bytes after the last covered region
        signatures.append({
            "start1": b0, "len1": l0,
            "start2": b1, "len2": l1,
            "covered_bytes": covered,
        })
        if gap > 64:  # more than 64 bytes after the signed region = modification after signing
            coverage_gaps.append({
                "uncovered_bytes_after_signature": gap,
                "file_size": file_size,
                "signed_region_end": total,
            })

    # Check for certificate data
    has_cert = b"/Cert" in raw

    suspicious_flags: list[str] = []
    if coverage_gaps:
        suspicious_flags.append(
            f"Signature ByteRange does not cover the full file "
            f"({coverage_gaps[0]['uncovered_bytes_after_signature']:,} bytes uncovered) — "
            f"document was likely modified AFTER signing"
        )

    note = ""
    if not has_sig_field and not signatures:
        note = (
            "No digital signature found — document integrity cannot be cryptographically verified. "
            "This is normal for most payslips and offer letters but note the absence."
        )

    return {
        "has_signature_field": has_sig_field,
        "signature_count":     len(signatures),
        "has_certificate":     has_cert,
        "signatures":          signatures,
        "coverage_gaps":       coverage_gaps,
        "suspicious_flags":    suspicious_flags,
        "note":                note,
        "interpretation": (
            f"SUSPICIOUS — {'; '.join(suspicious_flags)}"
            if suspicious_flags else
            (
                f"SIGNED — {len(signatures)} digital signature(s) with full ByteRange coverage"
                if signatures else
                "UNSIGNED — no digital signature (normal for standard HR documents)"
            )
        ),
    }


# ── Layer 8: Page Render ELA ──────────────────────────────────────────────────

def page_render_ela_analysis(pdf_path: str, out_dir: str, label: str) -> dict:
    """Rasterize page 1 and run ELA on the resulting image.

    SIMPLE EXPLANATION:
    We convert page 1 of the PDF into a high-quality JPEG photograph, then run
    'Error Level Analysis' (ELA) on it.  ELA works by re-compressing the photo at
    a lower quality and measuring the pixel-level differences.  If someone typed
    a new number into the PDF (e.g. changing 50,000 to 80,000 in a salary field)
    and that number was then rasterized into the image, it will typically show a
    distinct brightness spike in the ELA heatmap because its JPEG 'aging' pattern
    is different from the surrounding text which was generated at the same time.
    This catches tampering even when the document looks visually identical to the
    original.
    """
    doc = fitz.open(pdf_path)
    if doc.page_count == 0:
        doc.close()
        return {"skipped": True, "reason": "PDF has no pages"}

    page    = doc.load_page(0)
    mat     = fitz.Matrix(PAGE_RENDER_DPI / 72.0, PAGE_RENDER_DPI / 72.0)
    pix     = page.get_pixmap(matrix=mat, alpha=False)
    doc.close()

    # Save the rasterized page as a high-quality JPEG
    page_img_path = os.path.join(out_dir, f"page1_{label}.jpg")
    pix.save(page_img_path, jpg_quality=95)

    # ── Run ELA (same algorithm as forensics_detect.py) ───────────────────────
    ela_save_path = os.path.join(out_dir, f"ela_page1_{label}.png")
    orig = Image.open(page_img_path).convert("RGB")
    buf  = io.BytesIO()
    orig.save(buf, format="JPEG", quality=ELA_QUALITY)
    buf.seek(0)
    recompressed = Image.open(buf).convert("RGB")

    ela_arr     = np.array(ImageChops.difference(orig, recompressed), dtype=np.float32)
    ela_vis     = np.clip(ela_arr * ELA_AMPLIFY, 0, 255).astype(np.uint8)
    Image.fromarray(ela_vis).save(ela_save_path)

    mean_ela = float(np.mean(ela_arr))
    h_img, w_img = ela_arr.shape[:2]
    block = 32
    high = total = 0
    for y in range(0, h_img - block, block):
        for x in range(0, w_img - block, block):
            if np.mean(ela_arr[y:y+block, x:x+block]) > mean_ela * 2.5:
                high += 1
            total += 1
    suspicious_ratio = round(high / total, 4) if total else 0.0

    # ── Also run noise residual on the same rendered page ────────────────────
    noise_save_path = os.path.join(out_dir, f"noise_page1_{label}.png")
    img_cv  = cv2.imread(page_img_path)
    gray    = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY).astype(np.float32)
    residual = np.abs(gray - cv2.GaussianBlur(gray, (5, 5), 0))
    mean_noise = float(np.mean(residual))

    tile_sz = 64
    cv_vals = []
    for y in range(0, gray.shape[0] - tile_sz, tile_sz):
        for x in range(0, gray.shape[1] - tile_sz, tile_sz):
            patch = residual[y:y+tile_sz, x:x+tile_sz]
            cv_vals.append(np.std(patch) / (np.mean(patch) + 1e-6))
    cv_arr  = np.array(cv_vals) if cv_vals else np.array([0.0])
    hotspot = round(float(np.sum(cv_arr > np.mean(cv_arr) * 2.0)) / len(cv_arr), 4)

    vis = np.clip(residual * 4, 0, 255).astype(np.uint8)
    cv2.imwrite(noise_save_path, cv2.applyColorMap(vis, cv2.COLORMAP_JET))

    return {
        "page_rasterized_path":  page_img_path,
        "ela_heatmap_path":      ela_save_path,
        "noise_heatmap_path":    noise_save_path,
        "ela": {
            "mean_ela":               round(mean_ela, 3),
            "max_ela":                round(float(np.max(ela_arr)), 3),
            "std_ela":                round(float(np.std(ela_arr)), 3),
            "suspicious_block_ratio": suspicious_ratio,
            "interpretation": (
                "HIGH — many blocks have anomalous re-compression levels (pixel-level tampering likely)"
                if suspicious_ratio > 0.05 else
                "LOW — uniform ELA across page (consistent with original)"
            ),
        },
        "noise": {
            "mean_noise":         round(mean_noise, 4),
            "hotspot_tile_ratio": hotspot,
            "interpretation": (
                "ANOMALOUS — localised noise spikes (possible content insertion boundary)"
                if hotspot > 0.10 else
                "UNIFORM — noise residual consistent across the page"
            ),
        },
    }


# ── Layer 9: Embedded Image Analysis ─────────────────────────────────────────

def embedded_image_analysis(pdf_path: str, out_dir: str, label: str) -> dict:
    """Extract embedded images from the PDF and run noise analysis on the largest one.

    SIMPLE EXPLANATION:
    Many HR documents (especially scanned documents, or those with a company logo)
    contain embedded images inside the PDF.  A genuine logo embedded at creation
    time will have a consistent noise pattern throughout.  If someone pasted a new
    logo (or signature, or stamp) on top of the original and then saved the PDF,
    the newly inserted image will have a different noise fingerprint than the rest
    of the document's image content.
    """
    doc = fitz.open(pdf_path)

    # Collect unique images across all pages (by xref to avoid duplicates)
    seen_xrefs: set[int] = set()
    image_records: list[dict] = []

    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        for img_tuple in page.get_images(full=True):
            xref = img_tuple[0]
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)
            try:
                img_info = doc.extract_image(xref)
                image_records.append({
                    "xref":       xref,
                    "page":       page_num + 1,
                    "width":      img_info.get("width",  0),
                    "height":     img_info.get("height", 0),
                    "ext":        img_info.get("ext",    "unknown"),
                    "size_bytes": len(img_info.get("image", b"")),
                    "_bytes":     img_info.get("image", b""),
                })
            except Exception:  # noqa: BLE001
                pass

    doc.close()

    if not image_records:
        return {
            "total_embedded_images": 0,
            "analysis":              None,
            "skipped":               True,
            "reason":                "No images embedded in this PDF",
        }

    # Sort by size — analyse the largest (most likely to contain forged content)
    image_records.sort(key=lambda r: r["size_bytes"], reverse=True)
    top_img = image_records[0]
    img_bytes = top_img.pop("_bytes")  # remove raw bytes from report
    for rec in image_records[1:]:
        rec.pop("_bytes", None)

    # Write the largest image to disk and run noise analysis
    noise_result: dict | None = None
    noise_save_path = os.path.join(out_dir, f"embedded_imgs_{label}.png")
    try:
        img_pil   = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        tmp_path  = os.path.join(out_dir, f"_tmp_embed_{label}.png")
        img_pil.save(tmp_path)

        img_cv    = cv2.imread(tmp_path)
        gray      = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY).astype(np.float32)
        residual  = np.abs(gray - cv2.GaussianBlur(gray, (5, 5), 0))
        mean_n    = float(np.mean(residual))

        tile_sz  = 32
        cv_vals  = []
        for y in range(0, gray.shape[0] - tile_sz, tile_sz):
            for x in range(0, gray.shape[1] - tile_sz, tile_sz):
                patch = residual[y:y+tile_sz, x:x+tile_sz]
                cv_vals.append(np.std(patch) / (np.mean(patch) + 1e-6))
        cv_arr   = np.array(cv_vals) if cv_vals else np.array([0.0])
        hotspot  = round(float(np.sum(cv_arr > np.mean(cv_arr) * 2.0)) / len(cv_arr), 4)

        vis = np.clip(residual * 4, 0, 255).astype(np.uint8)
        cv2.imwrite(noise_save_path, cv2.applyColorMap(vis, cv2.COLORMAP_JET))

        # Clean up temp file
        try:
            os.remove(tmp_path)
        except OSError:
            pass

        noise_result = {
            "mean_noise":         round(mean_n, 4),
            "hotspot_tile_ratio": hotspot,
            "noise_heatmap_path": noise_save_path,
            "interpretation": (
                "ANOMALOUS — noise hotspots suggest composite or pasted regions in this image"
                if hotspot > 0.12 else
                "UNIFORM — noise residual of embedded image is consistent"
            ),
        }
    except Exception as exc:  # noqa: BLE001
        noise_result = {"skipped": True, "reason": str(exc)}

    # Summary list for the report (no raw bytes, capped at 10 records)
    img_summary = [
        {k: v for k, v in rec.items() if k != "_bytes"}
        for rec in image_records[:10]
    ]

    return {
        "total_embedded_images": len(image_records),
        "analysed_image":        {k: v for k, v in top_img.items() if k != "_bytes"},
        "noise_analysis":        noise_result,
        "all_images":            img_summary,
    }


# ── Layer 10: File Entropy ────────────────────────────────────────────────────

def file_entropy_analysis(pdf_path: str) -> dict:
    """Compute Shannon entropy of raw PDF bytes.

    SIMPLE EXPLANATION:
    Entropy measures how 'random' a file's data is.  A normal PDF full of compressed
    streams and fonts will have high entropy (near 8.0 bits per byte).  PDFs that
    have been repeatedly converted, re-saved, or had large uncompressed sections
    inserted sometimes show slightly lower entropy.  On its own this is a weak
    signal, but it becomes significant when combined with other indicators.
    """
    raw = Path(pdf_path).read_bytes()
    data   = np.frombuffer(raw, dtype=np.uint8)
    counts = np.bincount(data, minlength=256).astype(np.float64)
    counts = counts[counts > 0]
    p      = counts / counts.sum()
    entropy = round(float(-np.sum(p * np.log2(p))), 4)

    return {
        "file_entropy_bits":  entropy,
        "max_possible":       8.0,
        "file_size_bytes":    len(raw),
        "interpretation": (
            "LOW — entropy below 7.0 bits/byte; unusually low for a normal PDF"
            if entropy < 7.0 else
            "MODERATE — entropy 7.0–7.5; may indicate partially uncompressed content"
            if entropy < 7.5 else
            "HEALTHY — entropy ≥ 7.5 bits/byte, consistent with normally compressed PDF"
        ),
    }


# ── Layer 11: Object / Cross-Reference Integrity ──────────────────────────────

def object_integrity_analysis(pdf_path: str) -> dict:
    """Use pikepdf to check object count, types, and cross-reference consistency.

    SIMPLE EXPLANATION:
    A PDF is made up of numbered objects (fonts, images, page content streams, etc.)
    All of these are listed in a 'cross-reference table' (xref) at the end of the file.
    When someone inserts a new object (like a replacement salary figure rendered as a
    hidden layer), the xref table should be updated.  If the declared object count in
    the PDF header does not match the objects actually present in the file, or if
    compressed 'ObjStm' object streams are found (which can hide objects from simple
    scanners), those are signals worth noting.
    """
    try:
        with pikepdf.open(pdf_path, suppress_warnings=True) as pdf:
            # Total objects in the PDF (includes indirect and stream objects)
            all_objs = list(pdf.objects)
            total_objects = len(all_objs)

            type_counts: dict[str, int] = {}
            stream_count  = 0
            obj_stm_count = 0  # compressed object streams — can hide object data

            for obj in all_objs:
                try:
                    if isinstance(obj, pikepdf.Dictionary):
                        obj_type = str(obj.get("/Type", "")).strip("/")
                        if obj_type:
                            type_counts[obj_type] = type_counts.get(obj_type, 0) + 1
                        if obj.is_stream:
                            stream_count += 1
                            # ObjStm — a stream that contains other objects compressed inside
                            if obj_type == "ObjStm":
                                obj_stm_count += 1
                except Exception:  # noqa: BLE001
                    pass

            # Trailer /Size declares the expected object count
            declared_size: int | None = None
            try:
                declared_size = int(pdf.trailer.get("/Size", 0)) or None
            except Exception:  # noqa: BLE001
                pass

        suspicious_flags: list[str] = []
        if declared_size is not None and abs(total_objects - declared_size) > 5:
            suspicious_flags.append(
                f"Object count mismatch: trailer declares {declared_size} objects, "
                f"but {total_objects} objects found in file "
                f"— may indicate xref table was not properly updated after modification"
            )
        if obj_stm_count > 0:
            # ObjStm is valid in PDF ≥ 1.5 but worth flagging as it can obscure analysis
            suspicious_flags.append(
                f"{obj_stm_count} compressed object stream(s) (ObjStm) found — "
                f"objects are packed inside streams and cannot be read by basic text scanners"
            )

        # Compute xref_mismatch_score as a 0–100 continuous score for the ML model.
        # A 10% object-count difference yields ≈10 points; a 100%+ mismatch saturates at 100.
        # This is more informative than a binary suspicious flag for the ML classifier.
        xref_mismatch_score = 0.0
        if declared_size is not None and declared_size > 0:
            mismatch_abs = abs(total_objects - declared_size)
            xref_mismatch_score = min(100.0, (mismatch_abs / declared_size) * 100.0)

        return {
            "total_objects":      total_objects,
            "declared_size":      declared_size,
            "stream_count":       stream_count,
            "obj_stm_count":      obj_stm_count,
            "type_counts":        type_counts,
            "xref_mismatch_score": round(xref_mismatch_score, 2),  # new — used by ML model
            "suspicious_flags":   suspicious_flags,
            "interpretation": (
                f"SUSPICIOUS — {'; '.join(suspicious_flags)}"
                if suspicious_flags else
                f"CONSISTENT — {total_objects} objects, xref integrity normal"
            ),
        }

    except pikepdf.PasswordError:
        return {"skipped": True, "reason": "PDF is password-protected — cannot inspect objects"}
    except Exception as exc:  # noqa: BLE001
        return {"skipped": True, "reason": f"pikepdf error: {exc}"}


# ── Scoring ────────────────────────────────────────────────────────────────────

def compute_score(result: dict, peer_result: dict | None = None) -> tuple[float, list[str]]:
    """Apply a weighted penalty model across all 11 layers and return a 0–100 score.

    SIMPLE EXPLANATION:
    Each layer above flags suspicious findings.  This function is the judge: it adds
    up 'guilt points' from all the flags.  Heavier penalties go to stronger signals
    (incremental updates, JavaScript, post-signing modifications) and lighter ones
    to ambiguous signals (low entropy, absent metadata).  A final score ≥ 55 means
    TAMPERED, 30–54 means LIKELY TAMPERED, 15–29 means UNCERTAIN, and < 15 is ORIGINAL.
    """
    score: float        = 0.0
    evidence: list[str] = []

    # ── Layer 1: Incremental updates ──────────────────────────────────────────
    inc = result.get("incremental_updates", {})
    n_updates = inc.get("incremental_updates", 0)
    if n_updates > 0:
        pts = min(n_updates * 30, 45)
        score += pts
        evidence.append(
            f"Incremental updates: {n_updates} post-creation save(s) detected "
            f"(%%EOF count = {inc.get('eof_marker_count', '?')}) — "
            f"strongest PDF tampering signal ({pts:.0f} pts)"
        )

    # ── Layer 2: Metadata ──────────────────────────────────────────────────────
    meta = result.get("metadata", {})
    for flag in meta.get("suspicious_flags", []):
        pts = 20 if "editing" in flag.lower() or "manipulation" in flag.lower() else 10
        score += pts
        evidence.append(f"Metadata: {flag} ({pts:.0f} pts)")

    # Peer comparison: more metadata was stripped than reference → edited
    if peer_result:
        own_meta_fields  = sum(1 for f in ["creator", "producer", "author", "title",
                                            "creation_date"] if meta.get(f))
        peer_meta_fields = sum(1 for f in ["creator", "producer", "author", "title",
                                            "creation_date"]
                               if peer_result.get("metadata", {}).get(f))
        if peer_meta_fields > 0 and own_meta_fields < peer_meta_fields * 0.5:
            score += 12
            evidence.append(
                f"Metadata: only {own_meta_fields} field(s) vs peer's {peer_meta_fields} "
                f"— metadata stripped after editing (12 pts)"
            )

    # ── Layer 3: Font consistency ──────────────────────────────────────────────
    font = result.get("fonts", {})
    for flag in font.get("suspicious_flags", []):
        pts = 15 if "post-creation" in flag else 10
        score += pts
        evidence.append(f"Font: {flag} ({pts:.0f} pts)")

    # ── Layer 4: Invisible text ────────────────────────────────────────────────
    inv = result.get("invisible_text", {})
    hidden = inv.get("total_hidden_spans", 0)
    if hidden > 0:
        score += 25
        evidence.append(
            f"Invisible text: {hidden} hidden span(s) found "
            f"(white, zero-size, or shadow text — 25 pts)"
        )

    # ── Layer 5: Suspicious objects ───────────────────────────────────────────
    objs = result.get("suspicious_objects", {})
    if objs.get("javascript_count", 0) > 0:
        score += 40
        evidence.append(
            f"JavaScript embedded ({objs['javascript_count']} occurrence(s)) — "
            f"never expected in payslips or offer letters (40 pts)"
        )
    if objs.get("embedded_files_count", 0) > 0:
        score += 20
        evidence.append(f"Embedded file(s) found (20 pts)")
    if objs.get("has_open_action"):
        score += 15
        evidence.append("OpenAction present — PDF performs an action on open (15 pts)")
    if objs.get("has_xfa_form"):
        score += 12
        evidence.append("XFA form detected — dynamic display layer (can show forged values) (12 pts)")

    # ── Layer 6: Content consistency ──────────────────────────────────────────
    content = result.get("content", {})
    for flag in content.get("suspicious_flags", []):
        score += 12
        evidence.append(f"Content: {flag} (12 pts)")

    # ── Layer 7: Digital signature ────────────────────────────────────────────
    sig = result.get("signature", {})
    for flag in sig.get("suspicious_flags", []):
        score += 35
        evidence.append(
            f"Signature: {flag} — document tampered after signing (35 pts)"
        )

    # ── Layer 8: Page render ELA ───────────────────────────────────────────────
    render = result.get("page_render_ela", {})
    if not render.get("skipped"):
        ela_ratio = render.get("ela", {}).get("suspicious_block_ratio", 0.0)
        if ela_ratio > 0.05:
            score += 20
            evidence.append(
                f"Page ELA: {ela_ratio*100:.1f}% of blocks have anomalous re-compression "
                f"— pixel-level tampering likely (20 pts)"
            )
        elif ela_ratio > 0.02:
            score += 10
            evidence.append(
                f"Page ELA: {ela_ratio*100:.1f}% of blocks slightly elevated (10 pts)"
            )
        noise_hotspot = render.get("noise", {}).get("hotspot_tile_ratio", 0.0)
        if noise_hotspot > 0.10:
            score += 15
            evidence.append(
                f"Page noise: {noise_hotspot*100:.1f}% hotspot tiles "
                f"— possible insertion boundary (15 pts)"
            )

    # ── Layer 9: Embedded image noise ─────────────────────────────────────────
    emb = result.get("embedded_images", {})
    if not emb.get("skipped") and emb.get("noise_analysis"):
        img_hotspot = emb["noise_analysis"].get("hotspot_tile_ratio", 0.0)
        if img_hotspot > 0.12:
            score += 10
            evidence.append(
                f"Embedded image noise hotspot: {img_hotspot*100:.1f}% (10 pts)"
            )

    # ── Layer 10: File entropy ─────────────────────────────────────────────────
    ent = result.get("file_entropy", {})
    if ent.get("file_entropy_bits", 8.0) < 7.0:
        score += 8
        evidence.append(
            f"File entropy low ({ent['file_entropy_bits']:.4f} bits/byte) — "
            f"unusual for a normal PDF (8 pts)"
        )

    # Peer entropy comparison
    if peer_result:
        own_ent  = ent.get("file_entropy_bits", 0.0)
        peer_ent = peer_result.get("file_entropy", {}).get("file_entropy_bits", 0.0)
        if peer_ent > 0 and own_ent < peer_ent * 0.97:
            score += 7
            evidence.append(
                f"Entropy: {own_ent:.4f} bits/byte vs peer {peer_ent:.4f} — "
                f"lower entropy suggests re-compression loss (7 pts)"
            )

    # ── Layer 11: Object integrity ─────────────────────────────────────────────
    obj_int = result.get("object_integrity", {})
    for flag in obj_int.get("suspicious_flags", []):
        pts = 20 if "mismatch" in flag else 8
        score += pts
        evidence.append(f"Object integrity: {flag} ({pts:.0f} pts)")

    # ── Peer file size ─────────────────────────────────────────────────────────
    if peer_result:
        own_sz  = result.get("file_size_bytes", 0)
        peer_sz = peer_result.get("file_size_bytes", 0)
        if peer_sz > 0 and own_sz < peer_sz * 0.80:
            score += 8
            evidence.append(
                f"File size: {own_sz:,} B vs peer {peer_sz:,} B — "
                f"{100*(1-own_sz/peer_sz):.0f}% smaller (re-compression, 8 pts)"
            )

    return min(score, 100.0), evidence


# ── Per-PDF orchestrator ───────────────────────────────────────────────────────

def analyse_pdf(path: str, out_dir: str) -> dict:
    """Run all 11 forensic layers on a single PDF and return the full result dict.

    SIMPLE EXPLANATION:
    This is the central 'coordinator' function.  It calls each of the 11 checks in
    sequence, prints progress to the terminal, and assembles everything into one big
    result dictionary that will later be scored and written to JSON.
    """
    label = _label(path)
    fsize = os.path.getsize(path)

    print(f"\n  [1/11] Incremental update detection …")
    layer1 = incremental_update_analysis(path)
    print(f"         eof_markers={layer1['eof_marker_count']}  "
          f"incremental_updates={layer1['incremental_updates']}  "
          f"→ {layer1['interpretation'][:55]}")

    print(f"  [2/11] Metadata analysis …")
    layer2 = metadata_analysis(path)
    print(f"         creator='{layer2['creator'][:40]}'  "
          f"flags={len(layer2['suspicious_flags'])}")
    for flag in layer2["suspicious_flags"]:
        print(f"         ⚑ {flag[:80]}")

    print(f"  [3/11] Font consistency …")
    layer3 = font_analysis(path)
    print(f"         unique_fonts={layer3['total_unique_fonts']}  "
          f"non_embedded={layer3['non_embedded_count']}  "
          f"flags={len(layer3['suspicious_flags'])}")

    print(f"  [4/11] Invisible / hidden text …")
    layer4 = invisible_text_analysis(path)
    print(f"         total_spans={layer4['total_text_spans']}  "
          f"white={layer4['white_text_spans']}  tiny={layer4['tiny_size_spans']}")

    print(f"  [5/11] Suspicious objects (JS / embeds / actions) …")
    layer5 = suspicious_objects_analysis(path)
    print(f"         js={layer5['javascript_count']}  "
          f"embedded_files={layer5['embedded_files_count']}  "
          f"open_action={layer5['has_open_action']}  "
          f"xfa={layer5['has_xfa_form']}")

    print(f"  [6/11] Content consistency …")
    layer6 = content_consistency_analysis(path)
    print(f"         pages={layer6['page_count']}  "
          f"blank={layer6['blank_pages']}  "
          f"sizes={len(layer6['unique_page_sizes'])}  "
          f"flags={len(layer6['suspicious_flags'])}")

    print(f"  [7/11] Digital signature check …")
    layer7 = digital_signature_analysis(path)
    print(f"         has_sig={layer7['has_signature_field']}  "
          f"sig_count={layer7['signature_count']}  "
          f"coverage_gaps={len(layer7['coverage_gaps'])}")

    print(f"  [8/11] Page render ELA + noise …")
    layer8 = page_render_ela_analysis(path, out_dir, label)
    if layer8.get("skipped"):
        print(f"         skipped — {layer8.get('reason')}")
    else:
        ela_r = layer8.get("ela", {})
        noi_r = layer8.get("noise", {})
        print(f"         ela_mean={ela_r.get('mean_ela', '?')}  "
              f"ela_suspicious={ela_r.get('suspicious_block_ratio', 0)*100:.1f}%  "
              f"noise_hotspot={noi_r.get('hotspot_tile_ratio', 0)*100:.1f}%")

    print(f"  [9/11] Embedded image analysis …")
    layer9 = embedded_image_analysis(path, out_dir, label)
    if layer9.get("skipped"):
        print(f"         skipped — {layer9.get('reason')}")
    else:
        print(f"         total_images={layer9['total_embedded_images']}  "
              f"analysing largest …")

    print(f"  [10/11] File entropy …")
    layer10 = file_entropy_analysis(path)
    print(f"          {layer10['file_entropy_bits']} bits/byte")

    print(f"  [11/11] Object / xref integrity …")
    layer11 = object_integrity_analysis(path)
    if layer11.get("skipped"):
        print(f"          skipped — {layer11.get('reason')}")
    else:
        print(f"          total_objects={layer11['total_objects']}  "
              f"obj_stm={layer11['obj_stm_count']}  "
              f"flags={len(layer11['suspicious_flags'])}")

    return {
        "path":             path,
        "file_size_bytes":  fsize,
        "incremental_updates": layer1,
        "metadata":            layer2,
        "fonts":               layer3,
        "invisible_text":      layer4,
        "suspicious_objects":  layer5,
        "content":             layer6,
        "signature":           layer7,
        "page_render_ela":     layer8,
        "embedded_images":     layer9,
        "file_entropy":        layer10,
        "object_integrity":    layer11,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python pdf_forensics_detect.py <pdf_file> [<reference_pdf_file>]")
        print()
        print("Examples:")
        print("  Single file:  python pdf_forensics_detect.py payslip.pdf")
        print("  Two-file comparison:  python pdf_forensics_detect.py suspect.pdf original.pdf")
        sys.exit(1)

    paths  = sys.argv[1:]
    labels = [_label(p) for p in paths]
    out_dir = _out_dir(paths[0])

    for path in paths:
        if not os.path.isfile(path):
            print(f"ERROR: file not found — {path}")
            sys.exit(1)
        if not path.lower().endswith(".pdf"):
            print(f"WARNING: {path} does not appear to be a PDF")

    mode   = "pair" if len(paths) == 2 else "single"
    report: dict[str, Any] = {"mode": mode, "pdfs": {}}

    for path, label in zip(paths, labels):
        print(f"\n{'='*65}")
        print(f"  Analysing: {label}")
        print(f"  File:      {path}  ({os.path.getsize(path):,} bytes)")
        print(f"{'='*65}")
        report["pdfs"][label] = analyse_pdf(path, out_dir)

    # ── Scoring ────────────────────────────────────────────────────────────────
    pdf_labels = list(report["pdfs"].keys())
    for i, label in enumerate(pdf_labels):
        result = report["pdfs"][label]
        peer   = report["pdfs"][pdf_labels[1 - i]] if mode == "pair" else None
        score, evidence = compute_score(result, peer)

        verdict = (
            "TAMPERED"        if score >= 55 else
            "LIKELY TAMPERED" if score >= 30 else
            "UNCERTAIN"       if score >= 15 else
            "ORIGINAL"
        )

        result["forgery_score_0_100"] = round(score, 1)
        result["verdict"]             = verdict
        result["evidence"]            = evidence

        print(f"\n  ► {label}  score={score:.0f}/100  verdict={verdict}")
        for ev in evidence:
            print(f"    • {ev}")

    # ── Pair verdict ──────────────────────────────────────────────────────────
    if mode == "pair":
        s0 = report["pdfs"][pdf_labels[0]]["forgery_score_0_100"]
        s1 = report["pdfs"][pdf_labels[1]]["forgery_score_0_100"]

        if abs(s0 - s1) < 10:
            original = tampered = "UNCERTAIN"
            conclusion = "Scores too close to determine with confidence — manual review recommended."
        elif s0 > s1:
            original, tampered = pdf_labels[1], pdf_labels[0]
            conclusion = (
                f"{pdf_labels[0]} scores {s0:.0f}/100 (tampered). "
                f"{pdf_labels[1]} is likely the original ({s1:.0f}/100)."
            )
        else:
            original, tampered = pdf_labels[0], pdf_labels[1]
            conclusion = (
                f"{pdf_labels[1]} scores {s1:.0f}/100 (tampered). "
                f"{pdf_labels[0]} is likely the original ({s0:.0f}/100)."
            )

        report["pair_verdict"] = {
            "original":   original,
            "tampered":   tampered,
            "scores":     {pdf_labels[0]: s0, pdf_labels[1]: s1},
            "conclusion": conclusion,
        }
        print(f"\n{'='*65}")
        print(f"  PAIR VERDICT")
        print(f"{'='*65}")
        print(f"  {conclusion}")

    # ── Write report ──────────────────────────────────────────────────────────
    out_path = os.path.join(out_dir, "pdf_forensics_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n  Report   → {out_path}")
    print(f"  Heatmaps → {out_dir}/")
    for label in pdf_labels:
        print(f"    ela_page1_{label}.png  "
              f"noise_page1_{label}.png  "
              f"embedded_imgs_{label}.png")


if __name__ == "__main__":
    main()


# ── Public library entry point (called from Bulk Scan page) ───────────────────

def run_pdf_forensics(pdf_path: str) -> Dict[str, Any]:
    """Run the full 11-layer PDF forensic pipeline on a structured PDF.

    This is the library entry point called by the Bulk Scan page when Gemma4
    classifies an uploaded PDF as a digitally-created structured document
    (payslip, offer letter, bank statement, form16, etc.).

    Unlike the CLI-oriented ``analyse_pdf()`` function, this function:
    - Does not print any progress to stdout (suitable for server-side use).
    - Writes ELA/noise heatmap images to a temp directory that is cleaned up
      automatically once the analysis completes.
    - Returns a result dict shaped identically to ``run_forensics()`` in
      ``image_forensics_detect.py`` so the Bulk Scan UI can render both image
      and PDF forensic results with the same ``_render_forensics_detail()``
      function.

    Returns
    -------
    dict with keys:
        scan_summary — source_file, format, file_size_bytes, forensic_verdict,
                       forgery_score_0_100, overall_explanation, evidence (list)
        layers       — dict of 11 layers, each with: name, status
                       (CLEAN / SUSPICIOUS / N/A), plain_english, metrics
    """
    path = str(pdf_path)
    fsize = os.path.getsize(path) if os.path.exists(path) else 0
    label = _label(path)

    log.info("run_pdf_forensics: starting", extra={"path": path, "size_bytes": fsize})

    # Use a temporary directory for the ELA/noise heatmap output images.
    # These are written by page_render_ela_analysis() and embedded_image_analysis().
    # We remove the directory unconditionally in the finally block.
    tmp_dir = tempfile.mkdtemp(prefix="bt_pdf_forensics_")
    try:
        # ── Run all 11 forensic layers independently ─────────────────────────
        # Each layer is wrapped so a failure in one never stops the others.
        def _safe(fn, *args):  # noqa: ANN202
            """Call fn(*args), returning a skip dict on any exception."""
            try:
                return fn(*args)
            except Exception as exc:  # noqa: BLE001
                log.warning("PDF forensics layer failed: %s — %s", fn.__name__, exc)
                return {"skipped": True, "reason": str(exc)}

        r_inc  = _safe(incremental_update_analysis, path)
        r_meta = _safe(metadata_analysis, path)
        r_font = _safe(font_analysis, path)
        r_inv  = _safe(invisible_text_analysis, path)
        r_objs = _safe(suspicious_objects_analysis, path)
        r_cont = _safe(content_consistency_analysis, path)
        r_sig  = _safe(digital_signature_analysis, path)
        # ELA and embedded-image layers need an output directory for heatmap PNGs.
        r_ela  = _safe(page_render_ela_analysis, path, tmp_dir, label)
        r_emb  = _safe(embedded_image_analysis, path, tmp_dir, label)
        r_ent  = _safe(file_entropy_analysis, path)
        r_obj  = _safe(object_integrity_analysis, path)

        # ── Detect whether this is a scanned (image-only) PDF ─────────────────
        # A scanned PDF has no machine-readable embedded text layer — it is just
        # a sequence of images.  We check by extracting text from up to 3 pages:
        # if fewer than 200 characters are found, the PDF is treated as scanned.
        # This flag is passed to the ML model as the `is_scanned_pdf` feature.
        _is_scanned_pdf = False
        try:
            _scan_doc = fitz.open(path)
            _scan_text = ""
            for _pn in range(min(len(_scan_doc), 3)):
                _scan_text += _scan_doc.load_page(_pn).get_text("text")
            _scan_doc.close()
            _is_scanned_pdf = len(_scan_text.strip()) < 200
            log.debug(
                "run_pdf_forensics: is_scanned_pdf=%s (embedded_text_chars=%d)",
                _is_scanned_pdf, len(_scan_text.strip()),
            )
        except Exception as _scan_exc:
            log.debug("run_pdf_forensics: is_scanned_pdf detection skipped — %s", _scan_exc)

        # Assemble the raw dict that compute_score() expects (same keys as
        # the dict returned by analyse_pdf()).
        raw_result: Dict[str, Any] = {
            "file_size_bytes":     fsize,
            "incremental_updates": r_inc,
            "metadata":            r_meta,
            "fonts":               r_font,
            "invisible_text":      r_inv,
            "suspicious_objects":  r_objs,
            "content":             r_cont,
            "signature":           r_sig,
            "page_render_ela":     r_ela,
            "embedded_images":     r_emb,
            "file_entropy":        r_ent,
            "object_integrity":    r_obj,
        }

        score, evidence = compute_score(raw_result, None)

        # Classify the overall verdict using the same thresholds as main().
        verdict = (
            "TAMPERED"        if score >= 55 else
            "LIKELY TAMPERED" if score >= 30 else
            "UNCERTAIN"       if score >= 15 else
            "ORIGINAL"
        )

        # Count how many layers raised at least one suspicion flag.
        _all_layers = [r_inc, r_meta, r_font, r_inv, r_objs, r_cont,
                       r_sig, r_ela, r_emb, r_ent, r_obj]
        suspicious_count = sum(
            1 for r in _all_layers
            if not r.get("skipped") and (
                r.get("suspicious_flags") or
                r.get("findings") or
                r.get("incremental_updates", 0) > 0 or
                r.get("total_hidden_spans", 0) > 0
            )
        )
        clean_count = 11 - suspicious_count

        # Build a single-sentence summary for the UI banner.
        if verdict == "ORIGINAL":
            explanation = (
                f"All {clean_count} applicable PDF forensic layers came back clean. "
                "No significant tampering signals detected in this digitally-created PDF."
            )
        elif verdict in ("LIKELY TAMPERED", "TAMPERED"):
            explanation = (
                f"{suspicious_count} of 11 PDF forensic layers flagged suspicious signals "
                f"(score {score:.0f}/100). Key evidence: {'; '.join(evidence[:3])}."
            )
        else:
            explanation = (
                f"Mixed results — {suspicious_count} suspicious, {clean_count} clean layers "
                f"(score {score:.0f}/100). Manual review recommended."
            )

        # ── Helper: derive CLEAN/SUSPICIOUS/N/A status from a flags list ─────
        def _status_from_flags(flags) -> str:
            """Return SUSPICIOUS when at least one flag is present, else CLEAN."""
            return "SUSPICIOUS" if flags else "CLEAN"

        # ── Map raw layer dicts to the standard UI layer shape ───────────────
        # Each entry must have: name, status, plain_english, metrics.
        # The 'plain_english' field is written in simple everyday language so
        # even a non-technical reviewer can understand what each check found.
        layers: Dict[str, Any] = {
            "layer_1_incremental_updates": {
                "name": "Incremental Update Detection",
                "status": (
                    "N/A" if r_inc.get("skipped") else
                    "SUSPICIOUS" if r_inc.get("incremental_updates", 0) > 0 else
                    "CLEAN"
                ),
                # Plain-English explanation: tell the reviewer exactly what was found
                # and why it matters, in plain words.
                "plain_english": (
                    f"⚠️ This file was saved {r_inc['incremental_updates']} additional time(s) after it was first created. "
                    f"Every time a PDF is opened and re-saved after creation, a new section is appended to the file. "
                    f"Genuine payslips and offer letters from payroll software are created once and never re-saved. "
                    f"Number of end-of-file markers found: {r_inc.get('eof_marker_count', '?')} (expected 1)."
                    if not r_inc.get("skipped") and r_inc.get("incremental_updates", 0) > 0
                    else "✅ The file was created in one go and never re-saved after creation. "
                         "This is normal for genuine payslips and official documents."
                    if not r_inc.get("skipped")
                    else "⏭️ Could not inspect end-of-file markers — check skipped."
                ),
                "metrics": {
                    "eof_marker_count":    r_inc.get("eof_marker_count"),
                    "incremental_updates": r_inc.get("incremental_updates"),
                    "pdf_version":         r_inc.get("pdf_version"),
                },
            },
            "layer_2_metadata": {
                "name": "Metadata Analysis",
                "status": "N/A" if r_meta.get("skipped") else _status_from_flags(r_meta.get("suspicious_flags", [])),
                "plain_english": (
                    # Build a simple sentence based on what was actually found
                    ("⚠️ " + " | ".join(r_meta["suspicious_flags"]))
                    if not r_meta.get("skipped") and r_meta.get("suspicious_flags")
                    else (
                        f"✅ Clean metadata. Created by: '{r_meta.get('creator', '—')}'. "
                        f"No online PDF editors or editing tools were detected. "
                        f"The creation date looks normal."
                        if not r_meta.get("skipped")
                        else "⏭️ Metadata check skipped."
                    )
                ),
                "metrics": {
                    "creator":       r_meta.get("creator"),
                    "producer":      r_meta.get("producer"),
                    "date_gap_days": r_meta.get("date_gap_days"),
                    "tamper_risk":   r_meta.get("tamper_risk"),
                },
            },
            "layer_3_font_consistency": {
                "name": "Font Consistency",
                "status": "N/A" if r_font.get("skipped") else _status_from_flags(r_font.get("suspicious_flags", [])),
                "plain_english": (
                    ("⚠️ Font inconsistency found — " + " | ".join(r_font["suspicious_flags"]) + ". "
                     "When a field (e.g. salary amount) is changed in a PDF editor, the new text often uses "
                     "a different font than the rest of the document. Mixed fonts are a common sign of editing.")
                    if not r_font.get("skipped") and r_font.get("suspicious_flags")
                    else (
                        f"✅ All {r_font.get('total_unique_fonts', 0)} font(s) appear consistent. "
                        "No suspicious font mixing that would indicate text was replaced."
                        if not r_font.get("skipped")
                        else "⏭️ Font check skipped."
                    )
                ),
                "metrics": {
                    "total_unique_fonts": r_font.get("total_unique_fonts"),
                    "non_embedded_count": r_font.get("non_embedded_count"),
                },
            },
            "layer_4_invisible_text": {
                "name": "Invisible / Hidden Text",
                "status": (
                    "N/A" if r_inv.get("skipped") else
                    "SUSPICIOUS" if r_inv.get("total_hidden_spans", 0) > 0 else
                    "CLEAN"
                ),
                "plain_english": (
                    f"⚠️ {r_inv['total_hidden_spans']} hidden text span(s) found "
                    f"({r_inv.get('white_text_spans', 0)} in white ink, {r_inv.get('tiny_size_spans', 0)} in zero-size text). "
                    "Hidden text is a common trick to overlay a different number or name on top of the original "
                    "while keeping the original text invisible in the background."
                    if not r_inv.get("skipped") and r_inv.get("total_hidden_spans", 0) > 0
                    else (
                        "✅ No hidden or invisible text found. All text in the document is visible."
                        if not r_inv.get("skipped")
                        else "⏭️ Hidden text check skipped."
                    )
                ),
                "metrics": {
                    "total_hidden_spans": r_inv.get("total_hidden_spans"),
                    "white_text_spans":   r_inv.get("white_text_spans"),
                    "tiny_size_spans":    r_inv.get("tiny_size_spans"),
                },
            },
            "layer_5_suspicious_objects": {
                "name": "Suspicious Object Detection",
                "status": "N/A" if r_objs.get("skipped") else _status_from_flags(r_objs.get("findings", [])),
                "plain_english": (
                    ("⚠️ Unexpected elements found — " + ", ".join(r_objs["findings"]) + ". "
                     "These features (JavaScript, dynamic forms, auto-run actions, embedded files) "
                     "are never present in genuine payslips, offer letters, or bank statements. "
                     "Their presence suggests the file may have been constructed or manipulated using a PDF tool.")
                    if not r_objs.get("skipped") and r_objs.get("findings")
                    else (
                        "✅ No unexpected elements found. No JavaScript, embedded files, "
                        "dynamic forms, or auto-run actions detected — normal for genuine HR documents."
                        if not r_objs.get("skipped")
                        else "⏭️ Object scan skipped."
                    )
                ),
                "metrics": {
                    "javascript_count":     r_objs.get("javascript_count"),
                    "embedded_files_count": r_objs.get("embedded_files_count"),
                    "has_open_action":      r_objs.get("has_open_action"),
                    "has_xfa_form":         r_objs.get("has_xfa_form"),
                },
            },
            "layer_6_content_consistency": {
                "name": "Content Consistency",
                "status": "N/A" if r_cont.get("skipped") else _status_from_flags(r_cont.get("suspicious_flags", [])),
                "plain_english": (
                    ("⚠️ Content structure issues — " + " | ".join(r_cont["suspicious_flags"]) + ". "
                     "Genuine documents from one system have consistent page sizes throughout and no blank pages. "
                     "Mixed sizes or blank pages can indicate the file was assembled from different sources.")
                    if not r_cont.get("skipped") and r_cont.get("suspicious_flags")
                    else (
                        f"✅ Content looks consistent across all {r_cont.get('page_count', '?')} page(s). "
                        "Page sizes are uniform and there are no blank or suspicious padding pages."
                        if not r_cont.get("skipped")
                        else "⏭️ Content consistency check skipped."
                    )
                ),
                "metrics": {
                    "page_count":        r_cont.get("page_count"),
                    "blank_pages":       r_cont.get("blank_pages"),
                    "unique_page_sizes": len(r_cont.get("unique_page_sizes", [])),
                },
            },
            "layer_7_digital_signature": {
                "name": "Digital Signature Check",
                "status": "N/A" if r_sig.get("skipped") else _status_from_flags(r_sig.get("suspicious_flags", [])),
                "plain_english": (
                    ("⚠️ Digital signature problem — " + " | ".join(r_sig["suspicious_flags"]) + ". "
                     "If a document was signed digitally and then someone changed even one character, "
                     "the signature becomes invalid. This is the most reliable tampering proof for signed PDFs.")
                    if not r_sig.get("skipped") and r_sig.get("suspicious_flags")
                    else (
                        "✅ No digital signature problems found. "
                        + ("The document has no digital signature — this is normal for most payslips and letters."
                           if not r_sig.get("has_signature_field")
                           else f"Digital signature is present and covers the full document ({r_sig.get('signature_count', 0)} signature(s)).")
                        if not r_sig.get("skipped")
                        else "⏭️ Digital signature check skipped."
                    )
                ),
                "metrics": {
                    "has_signature_field": r_sig.get("has_signature_field"),
                    "signature_count":     r_sig.get("signature_count"),
                    "coverage_gaps":       len(r_sig.get("coverage_gaps", [])),
                },
            },
            "layer_8_page_render_ela": {
                "name": "Page Render ELA (Error Level Analysis)",
                "status": (
                    "N/A" if r_ela.get("skipped") else
                    "SUSPICIOUS" if r_ela.get("ela", {}).get("suspicious_block_ratio", 0) > 0.02 else
                    "CLEAN"
                ),
                "plain_english": (
                    # ELA works by re-compressing the page image and looking for blocks that
                    # compress differently — pasted/edited regions stand out clearly.
                    f"⚠️ Error-Level Analysis found {r_ela.get('ela', {}).get('suspicious_block_ratio', 0)*100:.1f}% "
                    "of the page has unusual compression patterns. "
                    "This happens when a number or name is replaced in a PDF viewer — "
                    "the pasted block compresses differently from the rest of the page, leaving a visible signal."
                    if not r_ela.get("skipped") and r_ela.get("ela", {}).get("suspicious_block_ratio", 0) > 0.02
                    else (
                        "✅ Page rendering looks consistent. No areas with unusual compression patterns "
                        "that would indicate copy-paste or text replacement."
                        if not r_ela.get("skipped")
                        else f"⏭️ Page render ELA skipped — {r_ela.get('reason', 'unknown reason')}."
                    )
                ),
                "metrics": {} if r_ela.get("skipped") else {
                    "ela_mean":               r_ela.get("ela", {}).get("mean_ela"),
                    "suspicious_block_ratio": r_ela.get("ela", {}).get("suspicious_block_ratio"),
                    "noise_hotspot_ratio":    r_ela.get("noise", {}).get("hotspot_tile_ratio"),
                },
            },
            "layer_9_embedded_image_analysis": {
                "name": "Embedded Image Analysis",
                "status": (
                    "N/A" if r_emb.get("skipped") else
                    "SUSPICIOUS" if r_emb.get("noise_analysis", {}).get("hotspot_tile_ratio", 0) > 0.12 else
                    "CLEAN"
                ),
                "plain_english": (
                    f"⚠️ Embedded image noise hotspot ratio is {r_emb.get('noise_analysis', {}).get('hotspot_tile_ratio', 0)*100:.1f}% "
                    "which is high. When an image (e.g. a stamp or signature) is copied from elsewhere and pasted into a PDF, "
                    "the noise pattern of that image is different from the surrounding page — this check detects that difference."
                    if not r_emb.get("skipped") and r_emb.get("noise_analysis", {}).get("hotspot_tile_ratio", 0) > 0.12
                    else (
                        f"✅ {r_emb.get('total_embedded_images', 0)} embedded image(s) analysed — noise looks consistent."
                        if not r_emb.get("skipped") and not r_emb.get("skipped")
                        else "⏭️ No embedded images found in this PDF, or check was skipped."
                    )
                ),
                "metrics": {} if r_emb.get("skipped") else {
                    "total_embedded_images": r_emb.get("total_embedded_images"),
                    "noise_hotspot_ratio":   r_emb.get("noise_analysis", {}).get("hotspot_tile_ratio"),
                },
            },
            "layer_10_file_entropy": {
                "name": "File Entropy",
                "status": (
                    "N/A" if r_ent.get("skipped") else
                    "SUSPICIOUS" if r_ent.get("file_entropy_bits", 8.0) < 7.0 else
                    "CLEAN"
                ),
                "plain_english": (
                    f"⚠️ File entropy is low ({r_ent.get('file_entropy_bits', 0):.3f} bits/byte, expected ≥ 7.0). "
                    "Entropy measures randomness. A real PDF file has lots of compressed data that looks very random. "
                    "Low entropy can mean the file is simple/generated text without real complexity, "
                    "or that it was processed by a compression or stripping tool."
                    if not r_ent.get("skipped") and r_ent.get("file_entropy_bits", 8.0) < 7.0
                    else (
                        f"✅ File entropy is normal ({r_ent.get('file_entropy_bits', 0):.3f} bits/byte). "
                        "The file has the expected level of data complexity."
                        if not r_ent.get("skipped")
                        else "⏭️ File entropy check skipped."
                    )
                ),
                "metrics": {"file_entropy_bits": r_ent.get("file_entropy_bits")},
            },
            "layer_11_object_xref_integrity": {
                "name": "Object / XRef Integrity",
                "status": "N/A" if r_obj.get("skipped") else _status_from_flags(r_obj.get("suspicious_flags", [])),
                "plain_english": (
                    ("⚠️ Internal PDF structure issues — " + " | ".join(r_obj["suspicious_flags"]) + ". "
                     "A PDF is made of numbered objects (images, fonts, page content, etc.) with an index (xref) at the end. "
                     "If the index count does not match the actual objects in the file, "
                     "it usually means objects were added or removed without properly updating the index — "
                     "a common side effect of editing tools.")
                    if not r_obj.get("skipped") and r_obj.get("suspicious_flags")
                    else (
                        f"✅ Internal PDF structure is intact. "
                        f"{r_obj.get('total_objects', '?')} objects found with a consistent cross-reference index."
                        if not r_obj.get("skipped")
                        else f"⏭️ Object integrity check skipped — {r_obj.get('reason', 'unknown reason')}."
                    )
                ),
                "metrics": {} if r_obj.get("skipped") else {
                    "total_objects":      r_obj.get("total_objects"),
                    "obj_stm_count":      r_obj.get("obj_stm_count"),
                    # xref_mismatch_score: 0–100 continuous score for the ML model.
                    # Derived from declared_size vs actual object count mismatch percentage.
                    "xref_mismatch_score": r_obj.get("xref_mismatch_score", 0.0),
                },
            },
        }

        # ── Attach scanned-PDF flag as metadata on the layers dict ───────────
        # The ML model needs is_scanned_pdf as a feature.  Rather than adding a
        # 12th "layer", we expose it under a private _meta key that extract_feature_vector_pdf
        # reads.  It does not appear in the UI layer list.
        layers["_meta"] = {
            "is_scanned_pdf": float(_is_scanned_pdf),
        }

        # ── Attempt ML scoring; fall back to heuristic if model not available ─
        # This mirrors the same pattern used in image_forensics_detect._compute_score().
        scoring_method = "heuristic"
        feature_contributions: Optional[Dict[str, float]] = None  # type: ignore[type-arg]
        try:
            from basetruth.analysis.ml_scorer_pdf import (  # noqa: PLC0415
                extract_feature_vector_pdf,
                predict_pdf,
                explain_pdf,
            )
            feature_vec = extract_feature_vector_pdf(layers)
            ml_result   = predict_pdf(feature_vec)
            if ml_result:
                score  = ml_result["score"]
                scoring_method = "ML"   # same value emitted by image_forensics_detect
                # Re-derive the verdict from the ML score using the same thresholds.
                verdict = (
                    "TAMPERED"        if score >= 55 else
                    "LIKELY TAMPERED" if score >= 30 else
                    "UNCERTAIN"       if score >= 15 else
                    "ORIGINAL"
                )
                # Rebuild the explanation to reflect the ML-driven verdict.
                if verdict == "ORIGINAL":
                    explanation = (
                        f"All {clean_count} applicable PDF forensic layers came back clean. "
                        "No significant tampering signals detected in this digitally-created PDF."
                    )
                elif verdict in ("LIKELY TAMPERED", "TAMPERED"):
                    explanation = (
                        f"{suspicious_count} of 11 PDF forensic layers flagged suspicious signals "
                        f"(ML score {score:.0f}/100). Key evidence: {'; '.join(evidence[:3])}."
                    )
                else:
                    explanation = (
                        f"Mixed results — {suspicious_count} suspicious, {clean_count} clean layers "
                        f"(ML score {score:.0f}/100). Manual review recommended."
                    )
                log.info(
                    "run_pdf_forensics: scoring_method=ML (XGBoost)",
                    extra={"ml_score": score, "verdict": verdict},
                )
                # Compute per-feature SHAP contributions so the UI can show which
                # signals drove the score.  This is a non-blocking call — failures
                # are silently ignored and contributions stays None.
                feature_contributions = explain_pdf(feature_vec)
        except Exception as _ml_exc:
            log.warning(
                "run_pdf_forensics: ML scoring failed — falling back to heuristic: %s", _ml_exc,
            )

        log.info(
            "run_pdf_forensics: complete",
            extra={"path": path, "verdict": verdict, "score": score, "evidence_count": len(evidence)},
        )

        return {
            "scan_summary": {
                "source_file":         os.path.basename(path),
                "format":              "PDF (structured / digital)",
                "file_size_bytes":     fsize,
                "forensic_verdict":    verdict,
                "forgery_score_0_100": round(score, 1),
                "overall_explanation": explanation,
                "evidence":            evidence,
                # scoring_method is read by forensics_utils.py to show the ML/heuristic badge
                "scoring_method":      scoring_method,
                # feature_contributions is None when heuristic was used or SHAP failed
                "feature_contributions": feature_contributions,
            },
            "layers": layers,
        }

    except Exception as exc:  # noqa: BLE001
        log.error("run_pdf_forensics: unexpected failure — %s", exc, exc_info=True)
        return {
            "scan_summary": {
                "source_file":         os.path.basename(path),
                "forensic_verdict":    "ERROR",
                "forgery_score_0_100": 0.0,
                "overall_explanation": f"PDF forensic analysis failed: {exc}",
                "evidence":            [],
            },
            "layers": {},
        }
    finally:
        # Always remove the temp directory where ELA/noise heatmaps were written.
        shutil.rmtree(tmp_dir, ignore_errors=True)
