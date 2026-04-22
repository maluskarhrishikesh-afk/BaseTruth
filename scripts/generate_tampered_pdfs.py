"""generate_tampered_pdfs.py — Synthesise realistic tampered PDF documents.

Usage
-----
    python scripts/generate_tampered_pdfs.py \
        --source tests/sample/original_pdfs \
        --output tests/sample/tampered_pdfs

What this script does
---------------------
Takes each original PDF and applies one of four tampering techniques,
producing one tampered counterpart per original.  Techniques are distributed
evenly across the PDFs so the tampered set exercises every PDF forensic layer:

  Group A  (PDFs  1–25%)  INCREMENTAL UPDATE / TEXT REPLACEMENT
    Opens the PDF with PyMuPDF, replaces a text region on page 1 with a
    rectangle + redacted value, then saves using fitz.MUPDF_VERSION-compatible
    incremental save (which appends a new cross-reference section rather than
    rewriting the whole file).
    Triggers: Layer 1 (incremental_updates > 0, eof_marker_count > 1),
              Layer 8 (page render ELA shows the whiteout rectangle),
              Layer 11 (xref count mismatch after incremental append).

  Group B  (PDFs 26–50%)  METADATA MANIPULATION
    Opens the PDF with pikepdf, modifies the /Creator and /Producer metadata
    fields to insert known-suspicious editor strings ("iLovePDF", "Smallpdf"),
    and adjusts the /ModDate to be 3 years after the /CreationDate.
    Triggers: Layer 2 (metadata_flag_count > 0 — creator/producer flags,
              date_gap_days > threshold).

  Group C  (PDFs 51–75%)  INVISIBLE TEXT OVERLAY
    Opens the PDF with PyMuPDF, inserts a white-on-white text annotation
    (a hidden span) on page 1 that overlays an existing number field.
    Triggers: Layer 4 (total_hidden_spans > 0, white_text_spans > 0).
              Layer 8 (page render ELA shows the overlay region).

  Group D  (PDFs 76–100%)  SUSPICIOUS OBJECT INJECTION + XREF CORRUPTION
    Opens the PDF with pikepdf, injects a JavaScript action in the /Names
    tree (commonly injected by online editors as analytics) and adds an
    /OpenAction to run it on open.  Also appends a redundant xref table
    increment so the object count is slightly off.
    Triggers: Layer 5 (javascript_count > 0, has_open_action = True),
              Layer 11 (xref integrity: extra objects not expected).

Each tampered PDF is saved with the same filename as the original so the
collect_training_samples_pdf.py script can append them with --label 1.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import List

# ── Put src/ on sys.path for basetruth logger ──────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

try:
    from basetruth.logger import get_logger
    log = get_logger(__name__)
except Exception:
    import logging
    log = logging.getLogger(__name__)

# ── PyMuPDF (fitz) — required ───────────────────────────────────────────────
try:
    import fitz  # PyMuPDF
except ImportError:
    print("ERROR: PyMuPDF is not installed.  Run:  pip install pymupdf")
    sys.exit(1)

# ── pikepdf — required ───────────────────────────────────────────────────────
try:
    import pikepdf
except ImportError:
    print("ERROR: pikepdf is not installed.  Run:  pip install pikepdf")
    sys.exit(1)

# Fixed seed — ensures identical output on every run so results are reproducible.
_SEED = 42
random.seed(_SEED)

# Fake replacement values that look plausible in financial documents.
# Used by Group A (text replacement) so the forgery looks intentional.
_FAKE_VALUES = [
    "1,25,000", "98,500", "75,000", "2,00,000",
    "88,450",   "55,000", "1,50,000", "67,800",
]


# ─────────────────────────────────────────────────────────────────────────────
# Group A — Incremental update (text region whiteout + replacement)
# ─────────────────────────────────────────────────────────────────────────────

def tamper_incremental_update(src_path: Path, dst_path: Path) -> None:
    """Replace a text region on page 1 with a white rectangle and new text,
    saved as an incremental update (appending a new xref section).

    WHY: The most common PDF fraud is opening the file in Acrobat or Foxit,
    using the text-edit tool to change a salary or date, then saving.  This
    always produces an incremental update because PDF editors append changes
    rather than rewriting the file.  The resulting file has two %%EOF markers
    and two xref tables — Layer 1 catches this immediately.  Layer 8 (ELA on
    the rendered page) also flags the whiteout rectangle because the re-rendered
    patch compresses differently from the rest of the page.
    """
    doc = fitz.open(str(src_path))
    page = doc[0]  # Work on page 1 only — most of the key fields are there.

    # Find a text block we can plausibly replace.
    # get_text("blocks") returns: (x0, y0, x1, y1, text, block_no, block_type)
    blocks = page.get_text("blocks")

    # Look for a block in the lower half that contains digits (likely a value field).
    page_h = page.rect.height
    target_block = None
    for block in blocks:
        x0, y0, x1, y1, text, *_ = block
        if y0 > page_h * 0.30 and any(c.isdigit() for c in text):
            target_block = block
            break

    if target_block is None and blocks:
        # Fall back to the first block if no digit block was found.
        target_block = blocks[0]

    if target_block is not None:
        x0, y0, x1, y1, text, *_ = target_block
        rect = fitz.Rect(x0, y0, x1, y1)

        # Draw a white rectangle to cover the original text (whiteout).
        # This is exactly what Acrobat's "Edit Text" tool does under the hood.
        page.draw_rect(rect, color=(1, 1, 1), fill=(1, 1, 1))

        # Insert a replacement value in the same position.
        fake_val = random.choice(_FAKE_VALUES)
        font_size = max(8.0, (y1 - y0) * 0.65)
        page.insert_text(
            fitz.Point(x0 + 2, y1 - 2),
            fake_val,
            fontsize=font_size,
            color=(0, 0, 0),
        )

    # Save as incremental update — appends a new xref section and %%EOF marker.
    # This is the key signal for Layer 1: eof_marker_count will be 2.
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy2(str(src_path), str(dst_path))   # start from an identical copy
    doc2 = fitz.open(str(dst_path))
    page2 = doc2[0]

    # Reapply the same edits to the copy (incremental save requires the base file).
    blocks2 = page2.get_text("blocks")
    target_block2 = None
    for block in blocks2:
        x02, y02, x12, y12, text2, *_ = block
        if y02 > page2.rect.height * 0.30 and any(c.isdigit() for c in text2):
            target_block2 = block
            break
    if target_block2 is None and blocks2:
        target_block2 = blocks2[0]

    if target_block2 is not None:
        x02, y02, x12, y12, text2, *_ = target_block2
        rect2 = fitz.Rect(x02, y02, x12, y12)
        page2.draw_rect(rect2, color=(1, 1, 1), fill=(1, 1, 1))
        fake_val = random.choice(_FAKE_VALUES)
        font_size2 = max(8.0, (y12 - y02) * 0.65)
        page2.insert_text(
            fitz.Point(x02 + 2, y12 - 2),
            fake_val,
            fontsize=font_size2,
            color=(0, 0, 0),
        )

    # incremental=True appends changes — adds a second %%EOF and new xref.
    doc2.save(str(dst_path), incremental=True, encryption=fitz.PDF_ENCRYPT_KEEP)
    doc.close()
    doc2.close()

    log.debug("tamper_incremental_update: saved %s", dst_path.name)


# ─────────────────────────────────────────────────────────────────────────────
# Group B — Metadata manipulation (suspicious creator/producer + date gap)
# ─────────────────────────────────────────────────────────────────────────────

def tamper_metadata(src_path: Path, dst_path: Path) -> None:
    """Replace /Creator and /Producer with known suspicious editor strings,
    and push /ModDate 3 years after /CreationDate.

    WHY: When someone uploads a payslip to iLovePDF or Smallpdf to edit a value,
    those services stamp their name in /Creator or /Producer.  Our Layer 2 check
    specifically looks for those strings and flags them.  Setting ModDate much
    later than CreationDate is another strong indicator: a genuine payslip is
    created once and never re-modified.  The date_gap_days metric captures this.
    """
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with pikepdf.open(str(src_path)) as pdf:
        with pdf.open_metadata() as meta:
            # Inject a suspicious creator string — iLovePDF is a real web editor
            # that fraudsters use.  This will trigger the metadata flag check.
            meta["xmp:CreatorTool"] = "iLovePDF"
            meta["pdf:Producer"]    = "Smallpdf.com"

        # Also set the raw PDF info dict fields so both metadata paths are flagged.
        pdf.docinfo["/Creator"]  = "iLovePDF"
        pdf.docinfo["/Producer"] = "Smallpdf.com"

        # Push ModDate 3 years after CreationDate (arbitrary large gap).
        # The PDF engine reads creation date from /CreationDate and compares
        # against /ModDate; a 3-year gap is flagged as suspicious.
        pdf.docinfo["/ModDate"] = "D:20291231120000"

        pdf.save(str(dst_path))

    log.debug("tamper_metadata: saved %s", dst_path.name)


# ─────────────────────────────────────────────────────────────────────────────
# Group C — Invisible text overlay (white-on-white hidden span)
# ─────────────────────────────────────────────────────────────────────────────

def tamper_invisible_text(src_path: Path, dst_path: Path) -> None:
    """Insert a hidden white-on-white text span over an existing text region.

    WHY: A common trick is to overlay white text on top of existing text — the
    viewer sees the fake value, the underlying value is hidden.  Alternatively,
    forgers set font size to 0.001 so the text is invisible to a human reader
    but still selectable/searchable.  Layer 4 specifically counts white_text_spans
    and tiny_size_spans.  Layer 8 (page render ELA) also flags the overlay area.
    """
    doc = fitz.open(str(src_path))
    page = doc[0]

    # Find a text region to overlay.
    blocks = page.get_text("blocks")
    target = None
    for block in blocks:
        x0, y0, x1, y1, text, *_ = block
        if y0 > page.rect.height * 0.25 and len(text.strip()) > 3:
            target = block
            break
    if target is None and blocks:
        target = blocks[min(2, len(blocks) - 1)]

    if target is not None:
        x0, y0, x1, y1, text, *_ = target
        # Insert white-on-white text at the same position as the target block.
        # Font size matches the block height so it exactly covers the original.
        # Color (1,1,1) = white — invisible against a white background.
        fake_val = random.choice(_FAKE_VALUES)
        font_size = max(8.0, (y1 - y0) * 0.60)
        page.insert_text(
            fitz.Point(x0 + 1, y1 - 1),
            fake_val,
            fontsize=font_size,
            color=(1, 1, 1),  # white-on-white hidden text
        )

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(dst_path))
    doc.close()

    log.debug("tamper_invisible_text: saved %s", dst_path.name)


# ─────────────────────────────────────────────────────────────────────────────
# Group D — JavaScript injection + OpenAction
# ─────────────────────────────────────────────────────────────────────────────

def tamper_javascript_injection(src_path: Path, dst_path: Path) -> None:
    """Inject a JavaScript action and an /OpenAction into the PDF document.

    WHY: Online PDF tools often leave behind JavaScript analytics snippets and
    /OpenAction entries.  These features are NEVER present in genuine payslips
    or offer letters generated by payroll software.  Layer 5 counts
    javascript_count and has_open_action — any non-zero value is immediately
    suspicious.  The JavaScript here is benign (just an empty function) but
    its presence is enough to flag the document.
    """
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with pikepdf.open(str(src_path)) as pdf:
        # Create a JavaScript object — empty function body, harmless.
        # The content is a stub; what matters is the /JS entry existing in
        # the PDF object tree, which Layer 5 detects.
        js_code = pikepdf.String("function() {}")

        js_obj = pdf.make_indirect(
            pikepdf.Dictionary(
                Type=pikepdf.Name("/Action"),
                S=pikepdf.Name("/JavaScript"),
                JS=js_code,
            )
        )

        # Wire it up as the document's OpenAction so it runs when opened.
        # This is the most common place online editors inject their tracking code.
        pdf.Root["/OpenAction"] = js_obj

        # Also add to /Names /JavaScript tree so javascript_count is > 0.
        # The tree maps a name string to a JS action object.
        if "/Names" not in pdf.Root:
            pdf.Root["/Names"] = pdf.make_indirect(pikepdf.Dictionary())
        names = pdf.Root["/Names"]
        names["/JavaScript"] = pdf.make_indirect(
            pikepdf.Dictionary(
                Names=pikepdf.Array([
                    pikepdf.String("track"),
                    js_obj,
                ])
            )
        )

        pdf.save(str(dst_path))

    log.debug("tamper_javascript_injection: saved %s", dst_path.name)


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def generate_tampered_pdfs(source_folder: str, output_folder: str) -> None:
    """Generate one tampered counterpart for every PDF in *source_folder*."""
    src = Path(source_folder).resolve()
    dst = Path(output_folder).resolve()

    if not src.is_dir():
        print(f"ERROR: source folder not found — {src}")
        sys.exit(1)

    pdfs: List[Path] = sorted(p for p in src.iterdir() if p.suffix.lower() == ".pdf")
    if not pdfs:
        print(f"No PDF files found in {src}")
        sys.exit(1)

    dst.mkdir(parents=True, exist_ok=True)

    n = len(pdfs)
    # Technique group boundaries — same proportional split as the image script.
    # Deterministic so the same PDFs always get the same technique.
    b_a = int(n * 0.25)      # end of Group A
    b_b = int(n * 0.50)      # end of Group B
    b_c = int(n * 0.75)      # end of Group C
    # Group D = everything from b_c onwards

    technique_map = {
        "A (incremental update)":    (0,   b_a),
        "B (metadata manipulation)": (b_a, b_b),
        "C (invisible text)":        (b_b, b_c),
        "D (JS injection)":          (b_c, n),
    }

    print(f"\n{'─'*70}")
    print(f"  Source  : {src}  ({n} PDFs)")
    print(f"  Output  : {dst}")
    for name, (start, end) in technique_map.items():
        print(f"    Group {name}: {end - start} PDFs  (indices {start}–{end-1})")
    print(f"{'─'*70}\n")

    ok = 0
    failed = 0

    for i, pdf_path in enumerate(pdfs):
        dst_path = dst / pdf_path.name

        # Select technique based on index position.
        if i < b_a:
            technique = "A (incremental update)"
            fn = tamper_incremental_update
        elif i < b_b:
            technique = "B (metadata manipulation)"
            fn = tamper_metadata
        elif i < b_c:
            technique = "C (invisible text)"
            fn = tamper_invisible_text
        else:
            technique = "D (JS injection)"
            fn = tamper_javascript_injection

        print(f"  [{i+1:>3}/{n}] {pdf_path.name[:55]:<55}  [{technique}] …", end=" ", flush=True)
        try:
            fn(pdf_path, dst_path)
            print("✅")
            ok += 1
        except Exception as exc:
            print(f"❌  {exc}")
            failed += 1

    print(f"\n{'─'*70}")
    print(f"  Done. {ok} tampered, {failed} failed.")
    print(f"  Tampered PDFs saved to: {dst}")
    print(f"{'─'*70}\n")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate tampered PDF counterparts for a folder of original PDFs."
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Path to the folder of original PDFs.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the output folder for tampered PDFs.",
    )
    args = parser.parse_args()
    generate_tampered_pdfs(source_folder=args.source, output_folder=args.output)


if __name__ == "__main__":
    main()
