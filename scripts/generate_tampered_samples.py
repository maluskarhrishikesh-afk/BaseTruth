"""generate_tampered_samples.py — Synthesise realistic tampered document images.

Usage
-----
    python scripts/generate_tampered_samples.py \
        --source tests/sample/original \
        --output tests/sample/tampered

What this script does
---------------------
Takes each original image and applies one of four tampering techniques,
producing one tampered counterpart per original.  Techniques are distributed
evenly across the 31 images so the tampered set exercises every forensic layer:

  Group A  (images  1–8 )  TEXT OVERLAY
    Paint a filled rectangle over a text region, then write new fake text.
    Simulates salary/date value substitution.
    Triggers: ELA (the painted patch compresses differently), edge density,
              font stroke-width inconsistency.

  Group B  (images  9–15)  WITHIN-IMAGE COPY-MOVE
    Copy a rectangular tile from one corner of the same image, paste it at a
    different position.  Simulates cloning a stamp or signature to a new spot.
    Triggers: clone detection (SIFT matched keypoints), ELA at paste boundary,
              noise inconsistency.

  Group C  (images 16–22)  CROSS-IMAGE REGION PASTE
    Slice a region from the NEXT image in the sorted list and paste it onto
    the current image.  Simulates a photo/signature inserted from a different
    source document.
    Triggers: ELA strongly (two different JPEG compression histories merge),
              colour anomaly (different lighting/sensor), noise inconsistency.

  Group D  (images 23–31)  MULTI-RECOMPRESSION + LOCALISED COLOUR SHIFT
    Re-save the image through 4 JPEG quality cycles (95→80→65→50), then
    brighten/tint a rectangular sub-region.  Simulates editing via an online
    PDF-to-image converter followed by a value highlight.
    Triggers: DCT double-compression comb, ELA at the tinted patch,
              colour anomaly in the altered region.

Each tampered file is saved as a JPEG and given the same filename as the
original so the collect_training_samples.py script can append them easily
with --label 1.
"""
from __future__ import annotations

import io
import os
import random
import sys
from pathlib import Path
from typing import List

# ── put src/ on sys.path for basetruth imports (logger only) ─────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

try:
    from basetruth.logger import get_logger
    log = get_logger(__name__)
except Exception:
    import logging
    log = logging.getLogger(__name__)

# ── PIL is required ─────────────────────────────────────────────────────────
try:
    from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
except ImportError:
    print("ERROR: Pillow is not installed.  Run:  pip install Pillow")
    sys.exit(1)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# Fixed random seed — ensures identical output on every run so results are
# reproducible and the training CSV stays consistent.
_SEED = 42
random.seed(_SEED)


# ─────────────────────────────────────────────────────────────────────────────
# Technique A — Text overlay
# ─────────────────────────────────────────────────────────────────────────────

# Fake values that might plausibly appear in financial documents.  These are
# used to overwrite a random text region so the forgery looks intentional.
_FAKE_NUMBERS = [
    "₹ 1,25,000", "₹ 98,500", "2026-03-31", "04/2026",
    "TX9982341", "EMP00421", "₹ 75,000", "NET 88,450",
]


def tamper_text_overlay(img: Image.Image) -> Image.Image:
    """Paint a filled rectangle over a text area and write fake text on it.

    WHY: The most common forgery is replacing a number — salary, date, PAN.
    The forger selects a text region, covers it with the background colour,
    and types the new value.  ELA always catches this: the painted rectangle
    and the new text have a different compression history from the rest of the
    image, so they 'glow' in the ELA heatmap.
    """
    out = img.copy().convert("RGB")
    draw = ImageDraw.Draw(out)

    w, h = out.size
    # Sample the background colour from a pixel near the top-left of the image
    # (likely the document paper colour for a photo taken on a desk).
    bg_x = max(0, min(w - 1, int(w * 0.05)))
    bg_y = max(0, min(h - 1, int(h * 0.05)))
    bg_colour = out.getpixel((bg_x, bg_y))
    # For RGB tuples, keep as-is; PIL getpixel on JPEG may return (r,g,b)
    if isinstance(bg_colour, int):
        bg_colour = (bg_colour, bg_colour, bg_colour)

    # Choose a random position in the lower-middle of the image — where
    # financial fields (salary, date) typically appear on a payslip/statement.
    x0 = random.randint(int(w * 0.15), int(w * 0.55))
    y0 = random.randint(int(h * 0.40), int(h * 0.75))
    rect_w = random.randint(int(w * 0.20), int(w * 0.35))
    rect_h = random.randint(int(h * 0.025), int(h * 0.05))
    x1, y1 = x0 + rect_w, y0 + rect_h

    # Fill the rectangle with the background colour to erase existing text.
    draw.rectangle([x0, y0, x1, y1], fill=bg_colour)

    # Write fake text inside the erased region.
    fake_value = random.choice(_FAKE_NUMBERS)
    # Font size scaled to ~3% of image height so it looks proportional.
    font_size = max(20, int(h * 0.025))
    try:
        # Try to load a sans-serif system font; fall back to PIL default.
        font = ImageFont.truetype("arial.ttf", font_size)
    except OSError:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        except OSError:
            font = ImageFont.load_default()

    # Text colour: slightly darker than background to look like ink.
    text_colour = (30, 30, 30)
    draw.text((x0 + 4, y0 + 2), fake_value, fill=text_colour, font=font)

    log.debug(
        "tamper_text_overlay: applied",
        extra={"rect": (x0, y0, x1, y1), "text": fake_value},
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Technique B — Within-image copy-move
# ─────────────────────────────────────────────────────────────────────────────

def tamper_copy_move(img: Image.Image) -> Image.Image:
    """Copy a rectangular tile from one part of the image and paste it elsewhere.

    WHY: Stamps, logos, and signatures are commonly cloned — the forger copies
    an official stamp from a genuine document and pastes it onto a fake one.
    SIFT-based clone detection catches this because the keypoints inside the
    pasted region match keypoints in the source region of the same image.
    ELA also flags the paste boundary because the pasted tile and the
    destination have mismatched compression histories.
    """
    out = img.copy().convert("RGB")
    w, h = out.size

    # Source tile: from the upper-right quadrant (where letterheads/logos live).
    src_x = random.randint(int(w * 0.55), int(w * 0.70))
    src_y = random.randint(int(h * 0.05), int(h * 0.20))
    tile_w = random.randint(int(w * 0.12), int(w * 0.22))
    tile_h = random.randint(int(h * 0.06), int(h * 0.12))

    tile = out.crop((src_x, src_y, src_x + tile_w, src_y + tile_h))

    # Destination: lower-left quadrant (body of the document).
    dst_x = random.randint(int(w * 0.05), int(w * 0.25))
    dst_y = random.randint(int(h * 0.60), int(h * 0.80))
    # Clamp destination so the tile stays inside the image bounds.
    dst_x = min(dst_x, w - tile_w - 1)
    dst_y = min(dst_y, h - tile_h - 1)

    out.paste(tile, (dst_x, dst_y))

    log.debug(
        "tamper_copy_move: applied",
        extra={"src": (src_x, src_y), "dst": (dst_x, dst_y), "tile": (tile_w, tile_h)},
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Technique C — Cross-image region paste
# ─────────────────────────────────────────────────────────────────────────────

def tamper_cross_paste(img: Image.Image, donor: Image.Image) -> Image.Image:
    """Paste a region from a different source image onto this image.

    WHY: Identity fraud often involves inserting a face/photo from one document
    into another, or pasting a company header from a genuine letter onto a
    fabricated one.  This creates the strongest ELA signature because the two
    images have completely different JPEG compression histories — the pasted
    region 'glows' intensely at the boundary in the ELA heatmap.  Colour
    anomaly also fires because lighting and camera sensor differ between images.
    """
    out = img.copy().convert("RGB")
    w, h = out.size

    # Resize donor to the same size so crop coordinates are comparable.
    donor_resized = donor.convert("RGB").resize((w, h), Image.LANCZOS)

    # Slice a large-ish central strip from the donor — simulates a header or
    # photo section being transplanted.
    strip_y0 = random.randint(int(h * 0.10), int(h * 0.25))
    strip_y1 = strip_y0 + random.randint(int(h * 0.15), int(h * 0.25))
    strip_x0 = random.randint(int(w * 0.10), int(w * 0.30))
    strip_x1 = strip_x0 + random.randint(int(w * 0.30), int(w * 0.50))
    strip = donor_resized.crop((strip_x0, strip_y0, strip_x1, strip_y1))

    # Place the transplanted strip at a different vertical position on the target.
    dst_y = random.randint(int(h * 0.50), int(h * 0.70))
    dst_x = random.randint(int(w * 0.05), int(w * 0.20))
    dst_x = min(dst_x, w - (strip_x1 - strip_x0) - 1)
    dst_y = min(dst_y, h - (strip_y1 - strip_y0) - 1)

    out.paste(strip, (dst_x, dst_y))

    log.debug(
        "tamper_cross_paste: applied",
        extra={"src_strip": (strip_x0, strip_y0, strip_x1, strip_y1), "dst": (dst_x, dst_y)},
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Technique D — Multi-recompression + localised colour shift
# ─────────────────────────────────────────────────────────────────────────────

def tamper_recompress_and_tint(img: Image.Image) -> Image.Image:
    """Re-save through 4 JPEG quality cycles, then tint a sub-region.

    WHY: Online document converters (iLovePDF, SmallPDF, etc.) always re-save
    the JPEG multiple times.  Each round-trip through different quality settings
    leaves a 'comb' pattern in the DCT histogram — the double-compression
    artefact.  Adding a tinted rectangle on top simulates highlighting a value
    or using the 'whiteout' tool in an online editor.  The tinted patch has a
    different compression history from the rest and triggers ELA + colour anomaly.
    """
    # ── Step 1: multiple JPEG re-save cycles ──────────────────────────────────
    # Simulates the image being processed by 4 different tools, each re-saving
    # at a different quality setting.  This is the primary DCT comb trigger.
    current = img.copy().convert("RGB")
    for quality in [92, 78, 63, 50]:
        buf = io.BytesIO()
        current.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        current = Image.open(buf).copy()  # .copy() detaches from the BytesIO buffer

    out = current.convert("RGB")
    w, h = out.size

    # ── Step 2: brighten and tint a sub-region ────────────────────────────────
    # Simulate a yellow-green highlight applied to a specific field value.
    x0 = random.randint(int(w * 0.20), int(w * 0.50))
    y0 = random.randint(int(h * 0.30), int(h * 0.65))
    region_w = random.randint(int(w * 0.18), int(w * 0.32))
    region_h = random.randint(int(h * 0.03), int(h * 0.06))
    x1, y1 = min(x0 + region_w, w - 1), min(y0 + region_h, h - 1)

    region = out.crop((x0, y0, x1, y1))

    # Brighten the cropped region by 1.6×, then add a warm tint overlay.
    region = ImageEnhance.Brightness(region).enhance(1.6)
    # Blend a semi-transparent yellow tint: multiply each pixel towards (255,255,150)
    tinted = Image.new("RGB", region.size, (255, 255, 120))
    region = Image.blend(region, tinted, alpha=0.30)

    out.paste(region, (x0, y0))

    log.debug(
        "tamper_recompress_and_tint: applied",
        extra={"region": (x0, y0, x1, y1)},
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# JPEG save helper — always saves as JPEG so the output matches the input format
# ─────────────────────────────────────────────────────────────────────────────

def _save_as_jpeg(img: Image.Image, path: Path, quality: int = 88) -> None:
    """Save the PIL image as JPEG, stripping EXIF to avoid metadata leaking
    the tampering technique back to the feature extractor."""
    # Convert to RGB first (in case it's RGBA or palette mode from PNG operations).
    rgb = img.convert("RGB")
    # Save without copying EXIF — a deliberate choice: a real forger would not
    # preserve the original camera EXIF when re-saving a tampered image.
    rgb.save(str(path), format="JPEG", quality=quality, optimize=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def generate_tampered(source_folder: str, output_folder: str) -> None:
    """Generate one tampered counterpart for every image in *source_folder*."""
    src = Path(source_folder).resolve()
    dst = Path(output_folder).resolve()

    if not src.is_dir():
        print(f"ERROR: source folder not found — {src}")
        sys.exit(1)

    images: List[Path] = sorted(
        p for p in src.iterdir() if p.suffix.lower() in _IMAGE_EXTS
    )
    if not images:
        print(f"No images found in {src}")
        sys.exit(1)

    dst.mkdir(parents=True, exist_ok=True)

    n = len(images)
    # Assign technique groups proportionally.
    # Groups are deterministic so the same 31 images always get the same technique.
    boundaries = [
        int(n * 0.26),   # end of group A (text overlay)
        int(n * 0.48),   # end of group B (copy-move)
        int(n * 0.71),   # end of group C (cross-paste)
        n,               # end of group D (recompress + tint)
    ]
    technique_names = [
        "text_overlay",
        "copy_move",
        "cross_paste",
        "recompress_tint",
    ]

    print(f"\n{'─'*65}")
    print(f"  Source  : {src}  ({n} images)")
    print(f"  Output  : {dst}")
    print(f"  Groups  : A=text_overlay({boundaries[0]})"
          f"  B=copy_move({boundaries[1]-boundaries[0]})"
          f"  C=cross_paste({boundaries[2]-boundaries[1]})"
          f"  D=recompress_tint({boundaries[3]-boundaries[2]})")
    print(f"{'─'*65}\n")

    ok = 0
    failed = 0

    for i, img_path in enumerate(images):
        # Determine which technique group this image falls into.
        if i < boundaries[0]:
            group = 0
        elif i < boundaries[1]:
            group = 1
        elif i < boundaries[2]:
            group = 2
        else:
            group = 3

        technique = technique_names[group]
        out_path = dst / img_path.name

        print(f"  [{i+1:>3}/{n}] {img_path.name:<42} technique={technique} …", end=" ", flush=True)

        try:
            img = Image.open(img_path)

            if group == 0:
                # Group A: text overlay
                tampered = tamper_text_overlay(img)

            elif group == 1:
                # Group B: within-image copy-move
                tampered = tamper_copy_move(img)

            elif group == 2:
                # Group C: cross-image paste — use the NEXT image as donor.
                # Wraps around at the end so every image has a donor.
                donor_path = images[(i + 1) % n]
                donor = Image.open(donor_path)
                tampered = tamper_cross_paste(img, donor)

            else:
                # Group D: multi-recompression + colour tint
                tampered = tamper_recompress_and_tint(img)

            _save_as_jpeg(tampered, out_path)
            print(f"✅  saved → {out_path.name}")
            ok += 1

        except Exception as exc:
            print(f"❌  FAILED — {exc}")
            log.error(
                "generate_tampered: image failed",
                extra={"image": img_path.name, "technique": technique, "error": str(exc)},
                exc_info=True,
            )
            failed += 1

    print(f"\n{'─'*65}")
    print(f"  Done. {ok} tampered images saved, {failed} failed.")
    print(f"  Output folder: {dst}")
    print(f"{'─'*65}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate synthetic tampered document images for ML training."
    )
    parser.add_argument(
        "--source",
        default="tests/sample/original",
        help="Folder containing original images (default: tests/sample/original).",
    )
    parser.add_argument(
        "--output",
        default="tests/sample/tampered",
        help="Folder to write tampered images into (default: tests/sample/tampered).",
    )
    args = parser.parse_args()
    generate_tampered(args.source, args.output)


if __name__ == "__main__":
    main()
