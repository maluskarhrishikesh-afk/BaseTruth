"""collect_training_samples.py — Build a training CSV from a folder of document images.

Usage
-----
    # Label a folder of ORIGINAL (clean) documents  →  label=0
    python scripts/collect_training_samples.py \
        --folder tests/sample/original \
        --label 0 \
        --output data/training_data_image.csv

    # Label a folder of TAMPERED documents  →  label=1  (append to same CSV)
    python scripts/collect_training_samples.py \
        --folder tests/sample/tampered \
        --label 1 \
        --output data/training_data_image.csv \
        --append

How it works
------------
Each image is passed through the same 11-layer forensic engine used by the
Forensic Scan screen (run_forensics from image_forensics_detect.py).
The raw numeric metrics from each layer are extracted and written as one row
in the CSV.  The 'label' column records whether the image is original (0) or
tampered (1) — this becomes the target variable for the ML classifier later.

The CSV is intentionally raw and human-readable so you can open it in Excel,
inspect individual rows, and correct any wrong labels before training.
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

# ── Supported image extensions ─────────────────────────────────────────────────
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# ── Column names written to the CSV ───────────────────────────────────────────
# Columns are grouped by forensic layer so the CSV stays readable.
# Each value is the raw numeric metric from that layer — not a derived score.
_COLUMNS = [
    # --- identity / provenance ---
    "filename",
    "file_size_bytes",
    "heuristic_score",        # score produced by the existing fixed-weight heuristic
    "heuristic_verdict",      # ORIGINAL / UNCERTAIN / LIKELY TAMPERED / TAMPERED

    # --- Layer 1: ELA ---
    "ela_mean",               # mean pixel difference after re-compression at 75%
    "ela_max",                # peak pixel difference
    "ela_std",                # standard deviation across all pixels
    "ela_suspicious_block_ratio",  # fraction of 32×32 blocks above 2.5× mean ELA

    # --- Layer 2: Metadata ---
    "metadata_flag_count",    # number of suspicious metadata flags found

    # --- Layer 3: File Entropy ---
    "file_entropy_bits",      # Shannon entropy of raw bytes (bits per byte, 0–8)

    # --- Layer 4: Noise ---
    "noise_hotspot_ratio",    # fraction of tiles with above-average noise level

    # --- Layer 5: DCT (JPEG only) ---
    "dct_comb_ratio",         # double-compression comb ratio (>1.3 is suspicious)
    "dct_skipped",            # 1 if DCT was skipped (non-JPEG), 0 if it ran

    # --- Layer 6: Clone detection ---
    "clone_ratio",            # fraction of keypoints flagged as cloned regions

    # --- Layer 7: Color anomaly ---
    "color_anomaly_ratio",    # fraction of pixels outside the document colour palette
    "color_largest_blob_px",  # area in pixels of the largest anomalous colour blob

    # --- Layer 8: Edge density ---
    "edge_high_density_ratio",# fraction of tiles with unnaturally high edge density

    # --- Layer 9: Saturation ---
    "saturation_ratio",       # fraction of tiles that are over-saturated

    # --- Layer 10: Font consistency ---
    "font_stroke_cv",         # coefficient of variation of stroke widths (text uniformity)
    "font_suspicious_regions",# count of spatially-coherent font anomaly clusters
    "font_sharpness_outlier_ratio",  # fraction of character regions with outlier sharpness
    "font_skipped",           # 1 if font analysis was skipped, 0 if it ran

    # --- Layer 11: AI / FFT ---
    "ai_spike_ratio",         # GAN/diffusion upsampling grid artefact ratio in FFT

    # --- Ground truth label ---
    "label",                  # 0 = original/clean,  1 = tampered/forged
]


def _extract_features(layers: dict, score: float, verdict: str, filename: str, file_size: int) -> dict:
    """Pull the raw numeric metrics out of a run_forensics() result dict.

    Every value defaults to 0.0 / 0 when the layer failed or was skipped,
    so the CSV always has the same shape regardless of the file format.
    """
    ela = layers.get("layer_1_ela", {}).get("metrics", {})
    meta = layers.get("layer_2_metadata", {})
    entropy = layers.get("layer_3_entropy", {}).get("metrics", {})
    noise = layers.get("layer_4_noise", {}).get("metrics", {})
    dct = layers.get("layer_5_dct", {}).get("metrics", {})
    clone = layers.get("layer_6_clone", {}).get("metrics", {})
    color = layers.get("layer_7_color", {}).get("metrics", {})
    blobs = color.get("anomaly_blobs", [])
    edge = layers.get("layer_8_edge", {}).get("metrics", {})
    sat = layers.get("layer_9_saturation", {}).get("metrics", {})
    font = layers.get("layer_10_font", {}).get("metrics", {})
    ai = layers.get("layer_11_ai", {}).get("metrics", {})

    return {
        "filename": filename,
        "file_size_bytes": file_size,
        "heuristic_score": round(score, 2),
        "heuristic_verdict": verdict,

        # Layer 1 — ELA
        "ela_mean": round(float(ela.get("mean_ela", 0.0)), 4),
        "ela_max": round(float(ela.get("max_ela", 0.0)), 4),
        "ela_std": round(float(ela.get("std_ela", 0.0)), 4),
        "ela_suspicious_block_ratio": round(float(ela.get("suspicious_block_ratio", 0.0)), 6),

        # Layer 2 — Metadata
        "metadata_flag_count": int(len(meta.get("suspicious_flags", []))),

        # Layer 3 — Entropy
        "file_entropy_bits": round(float(entropy.get("entropy_bits", 0.0)), 6),

        # Layer 4 — Noise
        "noise_hotspot_ratio": round(float(noise.get("hotspot_tile_ratio", 0.0)), 6),

        # Layer 5 — DCT
        "dct_comb_ratio": round(float(dct.get("comb_ratio", 0.0)), 6),
        "dct_skipped": int(bool(dct.get("skipped", False))),

        # Layer 6 — Clone
        "clone_ratio": round(min(1.0, float(clone.get("clone_ratio", 0.0))), 6),

        # Layer 7 — Color
        "color_anomaly_ratio": round(float(color.get("anomaly_ratio", 0.0)), 6),
        "color_largest_blob_px": int(blobs[0]["area_px"] if blobs else 0),

        # Layer 8 — Edge
        "edge_high_density_ratio": round(float(edge.get("high_density_tile_ratio", 0.0)), 6),

        # Layer 9 — Saturation
        "saturation_ratio": round(float(sat.get("high_saturation_tile_ratio", 0.0)), 6),

        # Layer 10 — Font
        "font_stroke_cv": round(float(font.get("stroke_cv", 0.0)), 6),
        "font_suspicious_regions": int(font.get("n_suspicious_regions", 0)),
        "font_sharpness_outlier_ratio": round(float(font.get("sharpness_outlier_ratio", 0.0)), 6),
        "font_skipped": int(bool(font.get("skipped", False))),

        # Layer 11 — AI
        "ai_spike_ratio": round(float(ai.get("spike_ratio", 0.0)), 6),

        # Ground truth  (filled in by the caller — not extracted from forensics)
        "label": None,
    }


def collect_samples(folder: str, label: int, output_csv: str, append: bool) -> None:
    """Run forensics on every image in *folder*, then write/append rows to *output_csv*."""
    from basetruth.analysis.image_forensics_detect import run_forensics  # noqa: PLC0415

    folder_path = Path(folder).resolve()
    if not folder_path.is_dir():
        print(f"ERROR: folder not found — {folder_path}")
        sys.exit(1)

    # Collect all image files in the folder
    images = sorted(
        p for p in folder_path.iterdir()
        if p.suffix.lower() in _IMAGE_EXTS
    )
    if not images:
        print(f"No image files found in {folder_path}")
        sys.exit(1)

    print(f"\n{'─'*60}")
    print(f"  Folder  : {folder_path}")
    print(f"  Images  : {len(images)} found")
    print(f"  Label   : {label}  ({'original/clean' if label == 0 else 'tampered/forged'})")
    print(f"  Output  : {output_csv}  ({'append' if append else 'create new'})")
    print(f"{'─'*60}\n")

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Decide whether to write the header row
    write_header = not append or not output_path.exists()

    rows_written = 0
    rows_failed = 0

    with open(output_path, "a" if append else "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_COLUMNS)
        if write_header:
            writer.writeheader()

        for i, img_path in enumerate(images, 1):
            print(f"  [{i:>3}/{len(images)}] {img_path.name} …", end=" ", flush=True)
            try:
                result = run_forensics(str(img_path))

                # run_forensics() returns a dict with scan_summary + layers
                summary = result.get("scan_summary", {})
                score = float(summary.get("forgery_score_0_100", 0.0))
                verdict = str(summary.get("forensic_verdict", "UNAVAILABLE"))
                file_size = int(summary.get("file_size_bytes", img_path.stat().st_size))
                layers = result.get("layers", {})

                row = _extract_features(
                    layers=layers,
                    score=score,
                    verdict=verdict,
                    filename=img_path.name,
                    file_size=file_size,
                )
                # Attach the human-provided ground truth label
                row["label"] = label

                writer.writerow(row)
                f.flush()  # Write immediately so partial results survive crashes

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
        description="Run forensic analysis on a folder of images and save raw metrics to CSV."
    )
    parser.add_argument(
        "--folder",
        required=True,
        help="Path to the folder containing images (relative to repo root or absolute).",
    )
    parser.add_argument(
        "--label",
        type=int,
        choices=[0, 1],
        required=True,
        help="Ground truth label:  0 = original/clean,  1 = tampered/forged.",
    )
    parser.add_argument(
        "--output",
        default="data/training_data_image.csv",
        help="Path to the output CSV file (default: data/training_data_image.csv).",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to an existing CSV instead of overwriting it.",
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
