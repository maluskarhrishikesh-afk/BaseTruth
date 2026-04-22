"""collect_training_samples_pdf.py — Build a PDF training CSV from a folder of PDF documents.

Usage
-----
    # Label a folder of ORIGINAL (clean) PDFs  →  label=0
    python scripts/collect_training_samples_pdf.py \
        --folder tests/sample/original_pdfs \
        --label 0 \
        --output data/training_data_pdf.csv

    # Label a folder of TAMPERED PDFs  →  label=1  (append to same CSV)
    python scripts/collect_training_samples_pdf.py \
        --folder tests/sample/tampered_pdfs \
        --label 1 \
        --output data/training_data_pdf.csv \
        --append

How it works
------------
Each PDF is passed through the structured-PDF forensic engine (run_pdf_forensics
from pdf_forensics_detect.py).  This engine has 11 PDF-specific layers that are
completely different from the image forensic layers:

  Layer 1  — Incremental Update Detection (how many times was the file re-saved?)
  Layer 2  — Metadata Analysis (software, date gaps, suspicious creator strings)
  Layer 3  — Font Consistency (mixed fonts often means text was replaced)
  Layer 4  — Invisible / Hidden Text (white or zero-size text overlaid on content)
  Layer 5  — Suspicious Objects (JavaScript, embedded files, XFA forms)
  Layer 6  — Content Consistency (mixed page sizes, blank pages)
  Layer 7  — Digital Signature Integrity (signature coverage gaps)
  Layer 8  — Page Render ELA (re-compress the rendered page, look for paste artefacts)
  Layer 9  — Embedded Image Analysis (noise in pasted stamps / signatures)
  Layer 10 — File Entropy (low entropy = simple / stripped / generated file)
  Layer 11 — Object / XRef Integrity (internal PDF structure consistency)

The raw numeric metrics from each layer are written as one row per PDF.
The 'label' column records whether the document is original (0) or tampered (1).
This CSV is the PDF equivalent of training_data_image.csv.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

# ── Put the src/ folder on sys.path so basetruth imports work
# when this script is run from the repo root without installing the package.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

# ── Column definitions — each raw metric from the 11 PDF forensic layers ──────
_COLUMNS = [
    # --- identity / provenance ---
    "filename",
    "file_size_bytes",
    "heuristic_score",        # score produced by the PDF heuristic engine
    "heuristic_verdict",      # ORIGINAL / UNCERTAIN / LIKELY TAMPERED / TAMPERED

    # --- Layer 1: Incremental Update Detection ---
    # How many times was the PDF re-opened and re-saved after initial creation?
    # Genuine payroll-generated documents are created once and never re-saved.
    "incremental_updates",    # count of extra save cycles (0 = clean)
    "eof_marker_count",       # number of %%EOF markers (should be 1 for clean files)
    "pdf_version",            # PDF version string (e.g. "1.4")

    # --- Layer 2: Metadata ---
    "metadata_flag_count",    # number of suspicious metadata flags found
    "metadata_creator",       # software that created the PDF (free-text, may be empty)
    "metadata_date_gap_days", # gap in days between creation and modification dates

    # --- Layer 3: Font Consistency ---
    # Mixed fonts are the single most reliable indicator that text was replaced.
    "total_unique_fonts",     # total distinct fonts across all pages
    "non_embedded_fonts",     # number of fonts not embedded (harder to render consistently)

    # --- Layer 4: Invisible / Hidden Text ---
    # White-on-white text or zero-size text is used to overlay a fake value.
    "total_hidden_spans",     # total hidden text spans (white ink or 0-pt size)
    "white_text_spans",       # subset: spans rendered in white colour
    "tiny_size_spans",        # subset: spans with font size 0 or near-zero

    # --- Layer 5: Suspicious Objects ---
    # JavaScript and embedded files are never present in genuine HR documents.
    "javascript_count",       # number of JavaScript actions in the PDF
    "embedded_files_count",   # number of attached/embedded files
    "has_open_action",        # 1 if the PDF auto-runs something on open
    "has_xfa_form",           # 1 if the file uses XFA dynamic forms

    # --- Layer 6: Content Consistency ---
    "page_count",             # total number of pages
    "blank_pages",            # number of completely blank pages
    "unique_page_sizes",      # number of distinct page dimensions (>1 is suspicious)

    # --- Layer 7: Digital Signature ---
    "has_signature_field",    # 1 if any digital signature field exists in the PDF
    "signature_count",        # number of digital signature fields
    "signature_coverage_gaps",# number of unsigned byte ranges after the last signature

    # --- Layer 8: Page Render ELA ---
    # ELA on the rendered page image detects pasted/replaced regions.
    "render_ela_mean",        # mean ELA pixel difference after re-compression
    "render_ela_suspicious_block_ratio",  # fraction of 32×32 blocks above 2.5× mean
    "render_noise_hotspot_ratio",         # fraction of tiles with anomalous noise

    # --- Layer 9: Embedded Image Analysis ---
    "embedded_image_count",         # number of embedded images in the PDF
    "embedded_noise_hotspot_ratio", # noise hotspot ratio of the largest embedded image

    # --- Layer 10: File Entropy ---
    "file_entropy_bits",      # Shannon entropy of raw bytes (0–8, real PDFs ≥ 7.0)

    # --- Layer 11: Object / XRef Integrity ---
    "total_pdf_objects",      # total count of PDF objects in the cross-reference table
    "obj_stm_count",          # number of compressed object streams (can hide edits)

    # --- Ground truth label ---
    "label",                  # 0 = original/clean,  1 = tampered/forged
]


def _safe_int(value, default: int = 0) -> int:
    """Convert *value* to int safely.

    Some PDF analysis functions return a list (e.g. a list of blank-page
    indices, or a list of font names) where you might expect a plain count.
    When that happens we use the length of the list as the numeric proxy,
    which is the most meaningful integer to put in the training CSV.
    """
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        # e.g. blank_pages=[2, 5]  →  len=2 (number of blank pages)
        return len(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _extract_features(layers: dict, score: float, verdict: str, filename: str, file_size: int) -> dict:
    """Pull the raw numeric metrics out of a run_pdf_forensics() result dict.

    Each metric has a safe default (0 / 0.0 / empty string) so the CSV row
    always has the same shape even when a layer was skipped or failed.
    """
    # ── Layer 1: Incremental Updates ─────────────────────────────────────────
    inc = layers.get("layer_1_incremental_updates", {}).get("metrics", {})
    incremental_updates = _safe_int(inc.get("incremental_updates"))
    eof_marker_count    = _safe_int(inc.get("eof_marker_count"), default=1)
    pdf_version         = str(inc.get("pdf_version") or "")

    # ── Layer 2: Metadata ─────────────────────────────────────────────────────
    meta = layers.get("layer_2_metadata", {})
    meta_metrics = meta.get("metrics", {})
    # The PDF engine stores suspicious_flags internally but only exposes them
    # via the status and plain_english fields in the layer dict.
    # Derive a numeric flag count from the plain_english text: each flag is
    # separated by " | " so count occurrences + 1 when the layer is SUSPICIOUS.
    meta_status = meta.get("status", "N/A")
    if meta_status == "SUSPICIOUS":
        pe = meta.get("plain_english", "")
        # plain_english starts with "⚠️ flag1 | flag2 | flag3"
        # Strip the leading warning emoji then count pipe-separated chunks.
        pe_stripped = pe.replace("⚠️ ", "").split(" | ")
        metadata_flag_count = len(pe_stripped)
    else:
        # CLEAN or N/A — no suspicious metadata flags.
        metadata_flag_count = 0
    metadata_creator = str(meta_metrics.get("creator") or "")
    date_gap_days    = float(meta_metrics.get("date_gap_days") or 0.0)

    # ── Layer 3: Font Consistency ─────────────────────────────────────────────
    font = layers.get("layer_3_font_consistency", {}).get("metrics", {})
    total_unique_fonts = _safe_int(font.get("total_unique_fonts"))
    non_embedded_fonts = _safe_int(font.get("non_embedded_count"))

    # ── Layer 4: Invisible Text ───────────────────────────────────────────────
    inv = layers.get("layer_4_invisible_text", {}).get("metrics", {})
    total_hidden_spans = _safe_int(inv.get("total_hidden_spans"))
    white_text_spans   = _safe_int(inv.get("white_text_spans"))
    tiny_size_spans    = _safe_int(inv.get("tiny_size_spans"))

    # ── Layer 5: Suspicious Objects ───────────────────────────────────────────
    objs = layers.get("layer_5_suspicious_objects", {}).get("metrics", {})
    javascript_count    = _safe_int(objs.get("javascript_count"))
    embedded_files      = _safe_int(objs.get("embedded_files_count"))
    has_open_action     = int(bool(objs.get("has_open_action")))
    has_xfa_form        = int(bool(objs.get("has_xfa_form")))

    # ── Layer 6: Content Consistency ──────────────────────────────────────────
    cont = layers.get("layer_6_content_consistency", {}).get("metrics", {})
    page_count        = _safe_int(cont.get("page_count"))
    blank_pages       = _safe_int(cont.get("blank_pages"))
    unique_page_sizes = _safe_int(cont.get("unique_page_sizes"), default=1)

    # ── Layer 7: Digital Signature ────────────────────────────────────────────
    sig = layers.get("layer_7_digital_signature", {}).get("metrics", {})
    has_sig_field      = int(bool(sig.get("has_signature_field")))
    signature_count    = _safe_int(sig.get("signature_count"))
    coverage_gaps      = _safe_int(sig.get("coverage_gaps"))

    # ── Layer 8: Page Render ELA ──────────────────────────────────────────────
    ela = layers.get("layer_8_page_render_ela", {}).get("metrics", {})
    render_ela_mean    = round(float(ela.get("ela_mean") or 0.0), 4)
    render_ela_sbr     = round(float(ela.get("suspicious_block_ratio") or 0.0), 6)
    render_noise       = round(float(ela.get("noise_hotspot_ratio") or 0.0), 6)

    # ── Layer 9: Embedded Image Analysis ──────────────────────────────────────
    emb = layers.get("layer_9_embedded_image_analysis", {}).get("metrics", {})
    embedded_img_count  = _safe_int(emb.get("total_embedded_images"))
    embedded_noise_hr   = round(float(emb.get("noise_hotspot_ratio") or 0.0), 6)

    # ── Layer 10: File Entropy ────────────────────────────────────────────────
    ent = layers.get("layer_10_file_entropy", {}).get("metrics", {})
    file_entropy_bits = round(float(ent.get("file_entropy_bits") or 0.0), 6)

    # ── Layer 11: Object / XRef Integrity ─────────────────────────────────────
    xref = layers.get("layer_11_object_xref_integrity", {}).get("metrics", {})
    total_pdf_objects = _safe_int(xref.get("total_objects"))
    obj_stm_count     = _safe_int(xref.get("obj_stm_count"))

    return {
        "filename":            filename,
        "file_size_bytes":     file_size,
        "heuristic_score":     round(score, 2),
        "heuristic_verdict":   verdict,

        # Layer 1
        "incremental_updates": incremental_updates,
        "eof_marker_count":    eof_marker_count,
        "pdf_version":         pdf_version,

        # Layer 2
        "metadata_flag_count":    metadata_flag_count,
        "metadata_creator":       metadata_creator,
        "metadata_date_gap_days": date_gap_days,

        # Layer 3
        "total_unique_fonts": total_unique_fonts,
        "non_embedded_fonts": non_embedded_fonts,

        # Layer 4
        "total_hidden_spans": total_hidden_spans,
        "white_text_spans":   white_text_spans,
        "tiny_size_spans":    tiny_size_spans,

        # Layer 5
        "javascript_count":    javascript_count,
        "embedded_files_count":embedded_files,
        "has_open_action":     has_open_action,
        "has_xfa_form":        has_xfa_form,

        # Layer 6
        "page_count":        page_count,
        "blank_pages":       blank_pages,
        "unique_page_sizes": unique_page_sizes,

        # Layer 7
        "has_signature_field":      has_sig_field,
        "signature_count":          signature_count,
        "signature_coverage_gaps":  coverage_gaps,

        # Layer 8
        "render_ela_mean":                    render_ela_mean,
        "render_ela_suspicious_block_ratio":  render_ela_sbr,
        "render_noise_hotspot_ratio":         render_noise,

        # Layer 9
        "embedded_image_count":        embedded_img_count,
        "embedded_noise_hotspot_ratio": embedded_noise_hr,

        # Layer 10
        "file_entropy_bits": file_entropy_bits,

        # Layer 11
        "total_pdf_objects": total_pdf_objects,
        "obj_stm_count":     obj_stm_count,

        # Ground truth (filled by the caller)
        "label": None,
    }


def collect_samples(folder: str, label: int, output_csv: str, append: bool) -> None:
    """Run PDF forensics on every PDF in *folder*, then write/append rows to *output_csv*."""
    from basetruth.analysis.pdf_forensics_detect import run_pdf_forensics  # noqa: PLC0415

    folder_path = Path(folder).resolve()
    if not folder_path.is_dir():
        print(f"ERROR: folder not found — {folder_path}")
        sys.exit(1)

    # Collect all PDF files in the folder
    pdfs = sorted(
        p for p in folder_path.iterdir()
        if p.suffix.lower() == ".pdf"
    )
    if not pdfs:
        print(f"No PDF files found in {folder_path}")
        sys.exit(1)

    print(f"\n{'─'*60}")
    print(f"  Folder  : {folder_path}")
    print(f"  PDFs    : {len(pdfs)} found")
    print(f"  Label   : {label}  ({'original/clean' if label == 0 else 'tampered/forged'})")
    print(f"  Output  : {output_csv}  ({'append' if append else 'create new'})")
    print(f"{'─'*60}\n")

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    write_header = not append or not output_path.exists()

    rows_written = 0
    rows_failed  = 0

    with open(output_path, "a" if append else "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_COLUMNS)
        if write_header:
            writer.writeheader()

        for i, pdf_path in enumerate(pdfs, 1):
            print(f"  [{i:>3}/{len(pdfs)}] {pdf_path.name} …", end=" ", flush=True)
            try:
                result  = run_pdf_forensics(str(pdf_path))
                summary = result.get("scan_summary", {})
                score   = float(summary.get("forgery_score_0_100", 0.0))
                verdict = str(summary.get("forensic_verdict", "UNAVAILABLE"))
                fsize   = int(summary.get("file_size_bytes", pdf_path.stat().st_size))
                layers  = result.get("layers", {})

                row = _extract_features(
                    layers=layers,
                    score=score,
                    verdict=verdict,
                    filename=pdf_path.name,
                    file_size=fsize,
                )
                row["label"] = label

                writer.writerow(row)
                f.flush()  # Flush after every row so partial results survive a crash

                verdict_icon = "🟢" if verdict == "ORIGINAL" else "🟡" if verdict == "UNCERTAIN" else "🔴"
                print(f"{verdict_icon} {verdict}  (score={score:.1f})")
                rows_written += 1

            except Exception as exc:
                print(f"❌ FAILED — {exc}")
                rows_failed += 1

    print(f"\n{'─'*60}")
    print(f"  Done. {rows_written} rows written, {rows_failed} failed.")
    print(f"  CSV saved to: {output_path.resolve()}")
    print(f"{'─'*60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run PDF forensic analysis on a folder of PDFs and save raw metrics to CSV."
    )
    parser.add_argument(
        "--folder",
        required=True,
        help="Path to the folder containing PDFs (relative to repo root or absolute).",
    )
    parser.add_argument(
        "--label",
        required=True,
        type=int,
        choices=[0, 1],
        help="Ground truth label: 0 = original/clean, 1 = tampered/forged.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the output CSV file (relative to repo root or absolute).",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        default=False,
        help="Append to an existing CSV instead of overwriting it. "
             "Use this when labelling a second folder (e.g. tampered) after the first.",
    )
    args = parser.parse_args()
    collect_samples(
        folder=args.folder,
        label=args.label,
        output_csv=args.output,
        append=args.append,
    )


if __name__ == "__main__":
    main()
