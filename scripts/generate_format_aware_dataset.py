"""generate_format_aware_dataset.py — Create a 50,000-row format-aware synthetic
image fraud dataset that is directly compatible with ml_scorer.py.

Usage:
    python scripts/generate_format_aware_dataset.py

Output:
    data/format_aware_image_fraud_50000.csv

After generating, retrain the model:
    python scripts/train_ml_scorer.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import random

# ── Seed for reproducibility ──────────────────────────────────────────────────
np.random.seed(33)
random.seed(33)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_OUT_PATH  = _REPO_ROOT / "data" / "format_aware_image_fraud_50000.csv"

# ── Config ────────────────────────────────────────────────────────────────────
TOTAL_ROWS = 50_000
HALF       = TOTAL_ROWS // 2

# Real-world document format distribution (typical HR/banking scan mix).
# JPEG dominates; HEIC growing from mobile devices; scan = physical paper.
formats = ["jpg",  "png",  "tiff", "webp", "heic", "bmp",  "scan"]
weights = [0.34,   0.18,   0.10,   0.10,   0.12,   0.06,   0.10 ]


# ── Helpers ───────────────────────────────────────────────────────────────────
def clamp(x: float, lo: float, hi: float) -> float:
    """Clamp x to [lo, hi] — prevents features from escaping valid range."""
    return max(lo, min(hi, x))


def rnd(v: float) -> float:
    """Round to 2 decimal places for readable CSV output."""
    return round(float(v), 2)


# ── Main generation loop ──────────────────────────────────────────────────────
rows = []

for i in range(TOTAL_ROWS):

    # Balanced labels — first HALF genuine, second HALF tampered.
    # Training shuffles, so sequential ordering causes no problems.
    label = 0 if i < HALF else 1

    # Pick a random file format according to real-world frequency weights.
    fmt = random.choices(formats, weights=weights)[0]

    # Latent fraud intensity — one hidden variable drives ALL features for this
    # row so that tampered documents have coherently elevated signals across
    # multiple layers, not random independent spikes.
    latent = np.random.normal(0, 1)
    if label == 1:
        latent += 1.0  # tampered docs are systematically more suspicious

    # Hard case: rows where genuine/tampered overlap (low |latent|).
    # Used for hard-case evaluation during training — matches the expert CSV.
    hard_case = 1 if abs(latent) < 0.4 else 0

    # ── Format-specific signal baselines ─────────────────────────────────────
    # Each format has different forensic fingerprints that the model must learn:
    #   JPG  — lossy JPEG compression → high DCT and ELA sensitivity
    #   PNG  — lossless → near-zero DCT; no compression mismatch signal
    #   TIFF — lossless scan archive format → very low DCT
    #   WebP — modern lossy → moderate DCT
    #   HEIC — Apple mobile format → compression artefacts differ from JPEG
    #   BMP  — raw uncompressed → no DCT, very low ELA baseline
    #   scan — physical paper scan → high inherent noise, near-zero DCT
    if fmt == "jpg":
        ela              = 35 + 14 * latent + np.random.normal(0, 10)
        dct              = 42 + 12 * latent + np.random.normal(0, 8)
        compression_base = 30 + 16 * latent + np.random.normal(0, 12)

    elif fmt == "png":
        ela              = 18 + 10 * latent + np.random.normal(0, 8)
        dct              = 4  + np.random.normal(0, 2)   # lossless → almost no DCT
        compression_base = 8  + np.random.normal(0, 4)

    elif fmt == "tiff":
        ela              = 22 + 10 * latent + np.random.normal(0, 8)
        dct              = 2  + np.random.normal(0, 2)   # near-zero for lossless
        compression_base = 8  + np.random.normal(0, 4)

    elif fmt == "webp":
        ela              = 28 + 12 * latent + np.random.normal(0, 10)
        dct              = 20 + 8  * latent + np.random.normal(0, 6)
        compression_base = 25 + 12 * latent + np.random.normal(0, 10)

    elif fmt == "heic":
        ela              = 24 + 10 * latent + np.random.normal(0, 8)
        dct              = 10 + 6  * latent + np.random.normal(0, 5)
        compression_base = 12 + 8  * latent + np.random.normal(0, 6)

    elif fmt == "bmp":
        ela              = 12 + 8 * latent + np.random.normal(0, 6)
        dct              = np.random.normal(0, 1)         # essentially zero
        compression_base = 4  + np.random.normal(0, 2)

    else:  # scan
        ela              = 45 + 16 * latent + np.random.normal(0, 12)
        dct              = 8  + np.random.normal(0, 3)   # scanned image, near-zero
        compression_base = 6  + np.random.normal(0, 3)

    # ── Shared forensic features (same formula for every format) ─────────────

    # metadata_flag_count: integer 0–5 suspicious EXIF / metadata flags.
    metadata_flag_count = int(clamp(round(np.random.normal(1 + 0.5 * label, 1)), 0, 5))

    # clone_ratio: fraction of image that matches another region (copy-move).
    # Beta distribution keeps this naturally bounded in [0, 1] with low baseline.
    clone_ratio = clamp(np.random.beta(1 + label, 18) + 0.02 * max(latent, 0), 0, 1)

    # text_alignment_score: NOT YET IMPLEMENTED in the engine — always 0.
    # Setting it to 0 here matches exactly what the engine produces at inference
    # so the model never sees variance in this feature during training either.
    # Training on variance it cannot extract at runtime is a training/inference mismatch.
    text_alignment_score = 0.0

    # font_inconsistency: mixed font strokes / weights suggesting copy-paste edits.
    font_inconsistency = 15 + 16 * latent + np.random.normal(0, 13)

    # signature_mismatch: -1 = "document has no signature" (N/A sentinel).
    # 55% of documents have no signature — keeps this realistic.
    if np.random.rand() < 0.55:
        signature_mismatch = -1.0
    else:
        signature_mismatch = 15 + 24 * label + 10 * latent + np.random.normal(0, 16)
        signature_mismatch = clamp(signature_mismatch, 0, 100)

    # noise_hotspots: physical scans have a higher inherent noise baseline.
    noise_hotspots = (
        18
        + np.random.normal(0, 10)
        + (20 if fmt == "scan" else 0)  # scans start with elevated noise floor
        + 5 * latent
    )

    # color_patch_score: localised colour anomalies suggesting image splicing.
    color_patch_score = 14 + 14 * latent + np.random.normal(0, 12)

    # ai_artifact_score: 10% of tampered docs show strong AI-generation artefacts.
    ai_artifact_score = (
        8
        + (30 if (label == 1 and np.random.rand() < 0.10) else 0)
        + 8 * latent
        + np.random.normal(0, 10)
    )

    # compression_mismatch: uses the format-specific baseline computed above.
    compression_mismatch = compression_base

    # ── Clamp all values to valid ranges ─────────────────────────────────────
    ela                  = clamp(ela, 0, 100)
    dct                  = clamp(dct, 0, 100)
    font_inconsistency   = clamp(font_inconsistency, 0, 100)
    noise_hotspots       = clamp(noise_hotspots, 0, 100)
    color_patch_score    = clamp(color_patch_score, 0, 100)
    ai_artifact_score    = clamp(ai_artifact_score, 0, 100)
    compression_mismatch = clamp(compression_mismatch, 0, 100)

    # ── Assemble row ──────────────────────────────────────────────────────────
    # NOTE: Only the 11 FEATURE_NAMES columns plus metadata (image_id,
    # file_format) and labels (label, hard_case) are written.
    # The three format-specific extra signals (alpha_anomaly_score,
    # scan_texture_score, sensor_noise_score) are intentionally EXCLUDED because
    # the engine cannot extract them at inference time yet.  Including phantom
    # features in training creates a mismatch that hurts real-world accuracy.
    # file_format is metadata only — train() selects by FEATURE_NAMES so it is
    # automatically ignored during training.
    rows.append({
        "image_id":             f"IMG_{i+1:06d}",
        "file_format":          fmt,            # metadata — NOT a model feature
        # ── 11 model features — exactly FEATURE_NAMES in ml_scorer.py ────────
        "ela_score":            rnd(ela),
        "dct_score":            rnd(dct),
        "metadata_flag_count":  metadata_flag_count,
        "clone_ratio":          round(clone_ratio, 4),
        "text_alignment_score": 0.0,
        "font_inconsistency":   rnd(font_inconsistency),
        "signature_mismatch":   rnd(signature_mismatch),
        "noise_hotspots":       rnd(noise_hotspots),
        "color_patch_score":    rnd(color_patch_score),
        "ai_artifact_score":    rnd(ai_artifact_score),
        "compression_mismatch": rnd(compression_mismatch),
        # ── Labels ───────────────────────────────────────────────────────────
        "label":                label,
        "hard_case":            hard_case,
    })

# ── Build DataFrame and save ──────────────────────────────────────────────────
df = pd.DataFrame(rows)
_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(_OUT_PATH, index=False)

print(f"\nSaved: {_OUT_PATH}")
print(f"Rows:  {len(df):,}")
print(f"\nLabel balance:")
print(df["label"].value_counts())
print(f"\nHard cases: {df['hard_case'].sum():,} ({df['hard_case'].mean()*100:.1f}%)")
print(f"\nFormat distribution:")
print(df["file_format"].value_counts())
print(f"\nFeature stats:")
from basetruth.analysis.ml_scorer import FEATURE_NAMES  # noqa: E402
print(df[FEATURE_NAMES].describe().round(2))
