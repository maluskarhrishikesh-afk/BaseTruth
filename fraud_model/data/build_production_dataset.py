"""
build_production_dataset.py
───────────────────────────────────────────────────────────────────────────────
Builds a production-grade forensic document fraud training dataset by:

  1. Using training_data_image.csv (220 real rows) as the ground-truth
     distribution for every feature.
  2. Recalibrating forensic_training_10000_rows.csv and
     format_aware_image_fraud_50000.csv (both synthetically generated) to match
     real-world feature ranges using per-class quantile mapping.
  3. Sampling real label-conditional distributions for columns that only exist
     in the real dataset (ela_max, file_size_bytes, etc.).
  4. Combining all three sources with sample weights:
       real = 5.0 × | syn1 = 1.0 × | syn2 = 0.5 ×
  5. Running a Kolmogorov-Smirnov drift report before and after calibration so
     you can confirm the recalibration worked.
  6. Saving the final merged dataset to data/production_dataset.csv.

Usage:
    python fraud_model/data/build_production_dataset.py
"""

import numpy as np
import pandas as pd
from scipy import stats
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Paths ─────────────────────────────────────────────────────────────────────────────
# This file lives in fraud_model/data/.  Real data is alongside it.
# Synthetic CSVs stay in the repo-level data/ folder.
_HERE = Path(__file__).resolve().parent          # fraud_model/data/
_REPO = _HERE.parent.parent                      # repo root
REAL_PATH = str(_HERE / "training_data_image.csv")
SYN1_PATH = str(_REPO / "data" / "forensic_training_10000_rows.csv")
SYN2_PATH = str(_REPO / "data" / "format_aware_image_fraud_50000.csv")
OUT_PATH  = str(_HERE / "production_dataset.csv")

# ── Heuristic score → verdict thresholds (derived from real data analysis) ───
# Real data breakdown:
#   ORIGINAL       : score 4.7  – 14.2   (mean ~10.5)
#   UNCERTAIN      : score 15.3 – 29.9   (mean ~22.1)
#   LIKELY TAMPERED: score 30.3 – 54.7   (mean ~41.7)
#   TAMPERED       : score 55.2 – 99.5   (mean ~78.8)
def _score_to_verdict(score: float) -> str:
    """Map a numeric heuristic score to a text verdict using real data thresholds."""
    if score < 15.0:
        return "ORIGINAL"
    if score < 30.0:
        return "UNCERTAIN"
    if score < 55.0:
        return "LIKELY TAMPERED"
    return "TAMPERED"


# ── Column mapping: synthetic column → real equivalent ───────────────────────
# Each entry: (synthetic_col_name, real_col_name)
# The quantile recalibration maps synthetic percentile ranks to real-world values,
# so the relative ordering is preserved while the value range matches reality.
COLUMN_MAP = [
    # Synthetic scale 0-100 → real natural measurement scale
    ("ela_score",           "ela_mean"),           # 0-100 → 0-2.6
    ("dct_score",           "dct_comb_ratio"),      # 0-100 → 0-1.886
    ("noise_hotspots",      "noise_hotspot_ratio"), # 0-78  → 0-0.073
    ("font_inconsistency",  "font_stroke_cv"),      # 0-100 → 0.257-0.568
    ("ai_artifact_score",   "ai_spike_ratio"),      # 0-87  → 1.43-1.85
    ("color_patch_score",   "color_anomaly_ratio"), # 0-100 → 0-0.276
    # Same column name but very different range (real mean=0.47, syn mean=0.09)
    ("clone_ratio",         "clone_ratio"),
    # Same name, close but slightly off (real mean=1.0, syn mean=1.86)
    ("metadata_flag_count", "metadata_flag_count"),
]

# Columns that exist in real data but have no direct synthetic equivalent.
# We fill them by bootstrap-sampling from the real label-conditional distribution.
REAL_ONLY_COLS = [
    "file_size_bytes",
    "heuristic_score",
    "ela_max",
    "ela_std",
    "ela_suspicious_block_ratio",
    "edge_high_density_ratio",
    "saturation_ratio",
    "font_suspicious_regions",
    "font_sharpness_outlier_ratio",
    "color_largest_blob_px",
    "dct_skipped",
    "font_skipped",
    "file_entropy_bits",  # always 0 in real data
]

# Final ordered schema for the output CSV
FINAL_COLS = [
    "label",
    "source",
    "sample_weight",
    # ELA features
    "ela_mean",
    "ela_max",
    "ela_std",
    "ela_suspicious_block_ratio",
    # Core forensic signals
    "noise_hotspot_ratio",
    "dct_comb_ratio",
    "dct_skipped",
    "clone_ratio",
    "metadata_flag_count",
    "file_entropy_bits",
    # Color / edge
    "color_anomaly_ratio",
    "color_largest_blob_px",
    "edge_high_density_ratio",
    "saturation_ratio",
    # Font
    "font_stroke_cv",
    "font_suspicious_regions",
    "font_sharpness_outlier_ratio",
    "font_skipped",
    # AI detection
    "ai_spike_ratio",
    # File metadata
    "file_size_bytes",
    "heuristic_score",
    "heuristic_verdict",
    # Extra useful synthetic-only columns (added value beyond real data)
    "file_format",
    "hard_case",
    "signature_mismatch_score",   # recalibrated from synthetic signature_mismatch
    "text_alignment_score",       # normalised from synthetic (syn1 only; 0 in syn2)
    "compression_mismatch_score", # recalibrated from synthetic compression_mismatch
]


# ── Core recalibration helpers ────────────────────────────────────────────────

def _quantile_map(syn_vals: np.ndarray, real_vals: np.ndarray) -> np.ndarray:
    """
    Quantile (percentile-to-percentile) mapping of synthetic values onto the
    real data distribution.

    Algorithm:
      1. Rank each synthetic value within the synthetic array to get its
         percentile position (0 → lowest, 1 → highest).
      2. Sort the real values to create a reference lookup table.
      3. Interpolate: for each synthetic percentile, find the corresponding
         real value at that position.

    This preserves relative ordering within the synthetic data while replacing
    the value range with the real-world range.  It is parameter-free and works
    for any distribution shape.
    """
    # Rank → percentile.  'average' gives tied values the same rank.
    pct = stats.rankdata(syn_vals, method="average") / len(syn_vals)
    # Clip slightly inward to avoid extrapolation artifacts at the extremes.
    pct = np.clip(pct, 0.001, 0.999)

    # Reference: sorted real values at evenly-spaced percentile positions
    real_sorted = np.sort(real_vals)
    real_pct    = np.linspace(0.0, 1.0, len(real_sorted))

    # Linear interpolation: synthetic percentile → real value
    return np.interp(pct, real_pct, real_sorted)


def _quantile_map_by_label(
    syn_vals:   np.ndarray,
    syn_labels: np.ndarray,
    real_vals:  np.ndarray,
    real_labels: np.ndarray,
) -> np.ndarray:
    """
    Per-class quantile mapping.

    Fraud documents are recalibrated against the real fraud distribution and
    genuine documents against the real genuine distribution.  This is more
    accurate than overall recalibration because it preserves the class-
    conditional signal that the model needs to learn from.

    If a class has no real reference rows the entire-population fallback is used.
    """
    result = np.empty(len(syn_vals), dtype=float)
    for lbl in [0, 1]:
        syn_mask  = syn_labels  == lbl
        real_mask = real_labels == lbl
        if syn_mask.sum() == 0:
            continue
        # Fall back to global real distribution if the class has too few real rows
        ref_vals = real_vals[real_mask] if real_mask.sum() >= 5 else real_vals
        result[syn_mask] = _quantile_map(syn_vals[syn_mask], ref_vals)
    return result


def _bootstrap_sample(
    real: pd.DataFrame,
    col: str,
    target_labels: np.ndarray,
    real_labels:   np.ndarray,
) -> np.ndarray:
    """
    Fill a column that only exists in real data by resampling (bootstrap) from
    the real label-conditional distribution.

    For each target row we pick a random real row with the same class label so
    the sampled values retain realistic fraud vs genuine differences.
    """
    result = np.empty(len(target_labels))
    for lbl in [0, 1]:
        t_mask = target_labels == lbl
        r_mask = real_labels   == lbl
        if t_mask.sum() == 0:
            continue
        # If the class has no real rows at all, use the full real population
        pool = real.loc[r_mask, col].values if r_mask.sum() >= 1 else real[col].values
        result[t_mask] = np.random.choice(pool.astype(float), size=t_mask.sum(), replace=True)
    return result


# ── KS drift report helper ───────────────────────────────────────────────────

def _ks_report(tag: str, real_df: pd.DataFrame, cal_df: pd.DataFrame) -> None:
    """
    Print Kolmogorov-Smirnov statistics for key columns to measure how closely
    the calibrated synthetic distribution matches the real distribution.
    KS statistic ranges 0 (identical) → 1 (completely different).
    """
    cols = [
        "ela_mean", "clone_ratio", "noise_hotspot_ratio",
        "dct_comb_ratio", "metadata_flag_count",
        "font_stroke_cv", "ai_spike_ratio",
    ]
    print(f"\n  {'Column':<35} KS stat ({tag})")
    print(f"  {'-'*55}")
    for c in cols:
        if c not in cal_df.columns or c not in real_df.columns:
            continue
        ks, _ = stats.ks_2samp(real_df[c].values, cal_df[c].dropna().values)
        bar = "█" * int(ks * 30)
        print(f"  {c:<35} {ks:.4f}  {bar}")


# ── Main recalibration pipeline ───────────────────────────────────────────────

def recalibrate_synthetic(
    syn_df:      pd.DataFrame,
    real_df:     pd.DataFrame,
    source_name: str,
) -> pd.DataFrame:
    """
    Recalibrate a synthetic dataset so its feature ranges match the real data.

    Steps:
      1. Map each synthetic column to its real equivalent using per-class
         quantile matching.
      2. Bootstrap-sample columns that exist only in the real dataset.
      3. Carry over extra synthetic-only columns after normalisation.
      4. Derive heuristic_verdict from the sampled heuristic_score.

    Returns a DataFrame aligned to FINAL_COLS (missing cols → NaN).
    """
    real_labels = real_df["label"].values
    syn_labels  = syn_df["label"].values
    out = pd.DataFrame({"label": syn_labels})

    # ── Step 1: recalibrate shared columns ────────────────────────────────────
    for syn_col, real_col in COLUMN_MAP:
        if syn_col in syn_df.columns:
            out[real_col] = _quantile_map_by_label(
                syn_df[syn_col].values.astype(float),
                syn_labels,
                real_df[real_col].values.astype(float),
                real_labels,
            )
        else:
            # Column missing in this synthetic set → bootstrap from real
            out[real_col] = _bootstrap_sample(real_df, real_col, syn_labels, real_labels)

    # ── Step 2: fill real-only columns via bootstrap sampling ─────────────────
    for col in REAL_ONLY_COLS:
        if col not in out.columns:
            out[col] = _bootstrap_sample(real_df, col, syn_labels, real_labels)

    # ── Step 3: carry over extra synthetic-only columns (normalised) ──────────

    # signature_mismatch: values in [-1, 100] in synthetic; -1 means no sig present.
    # We clip negatives to 0 (no mismatch), then recalibrate 0-100 onto the
    # real metadata_flag_count range as the closest available proxy.
    if "signature_mismatch" in syn_df.columns:
        sig_clipped = np.clip(syn_df["signature_mismatch"].values.astype(float), 0.0, None)
        out["signature_mismatch_score"] = _quantile_map_by_label(
            sig_clipped, syn_labels,
            real_df["metadata_flag_count"].values.astype(float), real_labels,
        )
    else:
        out["signature_mismatch_score"] = 0.0

    # text_alignment_score: only present in syn1; already 0-100 scale; normalise to 0-1.
    # syn2 has it as a constant 0.0 column.
    if "text_alignment_score" in syn_df.columns:
        out["text_alignment_score"] = (
            syn_df["text_alignment_score"].values.astype(float) / 100.0
        ).clip(0.0, 1.0)
    else:
        out["text_alignment_score"] = 0.0

    # compression_mismatch: recalibrate onto noise_hotspot_ratio range (closest proxy).
    if "compression_mismatch" in syn_df.columns:
        out["compression_mismatch_score"] = _quantile_map_by_label(
            syn_df["compression_mismatch"].values.astype(float), syn_labels,
            real_df["noise_hotspot_ratio"].values.astype(float), real_labels,
        )
    else:
        out["compression_mismatch_score"] = 0.0

    # hard_case flag: keep as-is (0/1 binary flag indicating borderline cases)
    out["hard_case"] = syn_df["hard_case"].values if "hard_case" in syn_df.columns else 0

    # file_format: keep as-is (tiff/jpg/webp/png); default to jpg if absent
    out["file_format"] = (
        syn_df["file_format"].values if "file_format" in syn_df.columns else "jpg"
    )

    # ── Step 4: derive heuristic_verdict from sampled heuristic_score ─────────
    out["heuristic_verdict"] = out["heuristic_score"].apply(_score_to_verdict)

    # ── Step 5: tag source and weight ─────────────────────────────────────────
    out["source"] = source_name

    return out


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 65)
    print("BaseTruth Production Dataset Builder")
    print("=" * 65)

    # ── Load ─────────────────────────────────────────────────────────────────
    print("\n[1/5] Loading raw datasets...")
    real = pd.read_csv(REAL_PATH)
    syn1 = pd.read_csv(SYN1_PATH)
    syn2 = pd.read_csv(SYN2_PATH)
    print(f"      Real  : {real.shape[0]:>6,} rows × {real.shape[1]} cols")
    print(f"      Syn1  : {syn1.shape[0]:>6,} rows × {syn1.shape[1]} cols")
    print(f"      Syn2  : {syn2.shape[0]:>6,} rows × {syn2.shape[1]} cols")

    # ── Pre-calibration KS drift ─────────────────────────────────────────────
    print("\n[2/5] Pre-calibration KS drift (lower = closer to real):")
    # Build a quick aligned view of syn1 just for the drift report
    _pre_syn1 = pd.DataFrame({
        "ela_mean":           syn1["ela_score"],
        "clone_ratio":        syn1["clone_ratio"],
        "noise_hotspot_ratio":syn1["noise_hotspots"],
        "dct_comb_ratio":     syn1["dct_score"],
        "metadata_flag_count":syn1["metadata_flag_count"],
        "font_stroke_cv":     syn1["font_inconsistency"],
        "ai_spike_ratio":     syn1["ai_artifact_score"],
    })
    _ks_report("syn1 raw", real, _pre_syn1)

    # ── Recalibrate ──────────────────────────────────────────────────────────
    print("\n[3/5] Recalibrating synthetic datasets...")
    syn1_cal = recalibrate_synthetic(syn1, real, "synthetic_forensic")
    print(f"      Syn1 calibrated: {syn1_cal.shape}")
    syn2_cal = recalibrate_synthetic(syn2, real, "synthetic_format_aware")
    print(f"      Syn2 calibrated: {syn2_cal.shape}")

    # ── Post-calibration KS drift ────────────────────────────────────────────
    print("\n      Post-calibration KS drift:")
    _ks_report("syn1 cal", real, syn1_cal)

    # ── Prepare real data for merge ──────────────────────────────────────────
    real_out = real.copy()
    real_out["source"]                   = "real"
    real_out["sample_weight"]            = 5.0
    real_out["hard_case"]                = 0
    real_out["file_format"]              = "jpg"
    real_out["signature_mismatch_score"] = 0.0
    real_out["text_alignment_score"]     = 0.0
    real_out["compression_mismatch_score"] = 0.0
    # Drop columns not in FINAL_COLS (filename, heuristic_verdict already present)
    real_out = real_out.drop(columns=["filename"], errors="ignore")

    # Assign sample weights to calibrated synthetic sets
    syn1_cal["sample_weight"] = 1.0
    syn2_cal["sample_weight"] = 0.5

    # ── Align all three to FINAL_COLS schema ─────────────────────────────────
    def _align(df: pd.DataFrame) -> pd.DataFrame:
        """Add missing FINAL_COLS as NaN; drop columns outside the schema."""
        for c in FINAL_COLS:
            if c not in df.columns:
                df[c] = np.nan
        return df[FINAL_COLS]

    real_aligned = _align(real_out)
    syn1_aligned = _align(syn1_cal)
    syn2_aligned = _align(syn2_cal)

    # ── Merge and shuffle ────────────────────────────────────────────────────
    print("\n[4/5] Merging and shuffling...")
    production = pd.concat([real_aligned, syn1_aligned, syn2_aligned], ignore_index=True)
    production = production.sample(frac=1, random_state=42).reset_index(drop=True)

    fraud_n   = (production["label"] == 1).sum()
    genuine_n = (production["label"] == 0).sum()
    print(f"      Total rows : {len(production):>7,}")
    print(f"      Fraud (1)  : {fraud_n:>7,}  ({fraud_n/len(production)*100:.1f}%)")
    print(f"      Genuine (0): {genuine_n:>7,}  ({genuine_n/len(production)*100:.1f}%)")
    print(f"      Sources    : {production['source'].value_counts().to_dict()}")

    # ── Per-source value-range sanity check ──────────────────────────────────
    print("\n      Recalibrated column ranges (real vs calibrated synthetic):")
    check_cols = ["ela_mean", "clone_ratio", "noise_hotspot_ratio", "font_stroke_cv", "ai_spike_ratio"]
    for col in check_cols:
        r_min, r_max = real[col].min(), real[col].max()
        s1 = production.loc[production["source"] == "synthetic_forensic", col]
        s2 = production.loc[production["source"] == "synthetic_format_aware", col]
        print(
            f"      {col:<35} real=[{r_min:.3f},{r_max:.3f}]"
            f"  syn1=[{s1.min():.3f},{s1.max():.3f}]"
            f"  syn2=[{s2.min():.3f},{s2.max():.3f}]"
        )

    # ── Save ─────────────────────────────────────────────────────────────────
    print(f"\n[5/5] Saving → {OUT_PATH}")
    production.to_csv(OUT_PATH, index=False)
    print(f"      Saved {len(production):,} rows × {len(production.columns)} columns.")

    print("\n" + "=" * 65)
    print("Production dataset ready.")
    print("  Use sample_weight column when fitting XGBoost.")
    print("  Validate ONLY on rows where source == 'real'.")
    print("=" * 65)


if __name__ == "__main__":
    main()
