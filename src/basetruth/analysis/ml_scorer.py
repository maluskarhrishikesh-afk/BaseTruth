"""ml_scorer.py — ML-based fraud scoring for the image forensic engine.

This module is an optional drop-in replacement for the fixed-weight heuristic
in image_forensics_detect._compute_score().  It works as follows:

  1. extract_feature_vector(layers)
       Converts raw engine layer output into 11 normalised (0–100) floats that
       match the schema of the expert training dataset
       (data/forensic_training_10000_rows.csv).

  2. predict(feature_vector)
       Loads data/ml_scorer_image.pkl and calls predict_proba() to get P(tampered).
       Returns None if the model file does not exist yet — the caller falls back
       to the heuristic automatically.

  3. train(csv_paths, output_pkl)
       Trains an XGBoost classifier on one or more CSV files, runs 5-fold CV,
       and saves the model only if ROC AUC >= 0.80.

Cold-start guarantee: if ml_scorer_image.pkl is absent, every call to predict()
returns None immediately and the existing heuristic runs unchanged.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from basetruth.logger import get_logger

log = get_logger(__name__)

# ── Path to the saved model ────────────────────────────────────────────────────
# Resolved relative to the repo root so the path works whether the code is run
# from the project root, src/, or as a package inside a virtualenv.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent  # src/basetruth/analysis → repo
_MODEL_PATH = _REPO_ROOT / "data" / "ml_scorer_image.pkl"

# Cache the loaded model in module-level state so we only deserialise it once
# per process, not once per document scan.
_MODEL_CACHE: Any = None
_MODEL_LOAD_ATTEMPTED: bool = False


# ── Feature names — must match the expert training CSV column order exactly ───
FEATURE_NAMES: List[str] = [
    "ela_score",
    "dct_score",
    "metadata_flag_count",
    "clone_ratio",
    "text_alignment_score",
    "font_inconsistency",
    "signature_mismatch",
    "noise_hotspots",
    "color_patch_score",
    "ai_artifact_score",
    "compression_mismatch",
]


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_feature_vector(layers: Dict[str, Any]) -> "np.ndarray":  # type: ignore[name-defined]
    """Convert raw engine layer dict into 11 normalised floats (0–100 each).

    The normalisation formulas map each engine metric into the same 0–100 scale
    used by the expert training dataset so the model's learned thresholds apply
    directly.  All values are clamped to [0, 100] (or [0, 1] for clone_ratio).

    Returns a numpy array of shape (11,) with dtype float32.
    Missing or skipped layers produce 0.0 — not NaN — so the vector is always
    complete and the model can still make a prediction.
    """
    import numpy as np  # noqa: PLC0415

    ela = layers.get("layer_1_ela", {}).get("metrics", {})
    meta = layers.get("layer_2_metadata", {})
    noise = layers.get("layer_4_noise", {}).get("metrics", {})
    dct = layers.get("layer_5_dct", {}).get("metrics", {})
    clone = layers.get("layer_6_clone", {}).get("metrics", {})
    color = layers.get("layer_7_color", {}).get("metrics", {})
    blobs = color.get("anomaly_blobs", [])
    font = layers.get("layer_10_font", {}).get("metrics", {})
    ai = layers.get("layer_11_ai", {}).get("metrics", {})

    # ── ela_score (0–100) ─────────────────────────────────────────────────────
    # Combine suspicious block ratio (primary signal) and mean ELA (secondary).
    # block_ratio > 0.05 is the existing threshold for high suspicion.
    # Multiplying by 800 puts a ratio of 0.125 at score=100, matching the
    # expert dataset's average tampered ela_score of ~56.
    ela_block = float(ela.get("suspicious_block_ratio", 0.0))
    ela_mean = float(ela.get("mean_ela", 0.0))
    ela_score = min(100.0, ela_block * 800 + ela_mean * 2)

    # ── dct_score (0–100) ─────────────────────────────────────────────────────
    # comb_ratio of 1.0 = no double-compression.  Ratios above 1.3 indicate a
    # double-compression comb.  Mapping: (ratio - 1.0) * 60 so ratio=1.5 → 30,
    # ratio=2.67 → 100.  Skipped (non-JPEG) → 0.
    raw_comb = float(dct.get("comb_ratio", 0.0)) if not dct.get("skipped") else 0.0
    dct_score = min(100.0, max(0.0, (raw_comb - 1.0) * 60.0))

    # ── metadata_flag_count (0–5) ─────────────────────────────────────────────
    # Integer count of suspicious metadata flags — direct pass-through.
    metadata_flag_count = min(5, int(len(meta.get("suspicious_flags", []))))

    # ── clone_ratio (0–1) ─────────────────────────────────────────────────────
    # Clamp to [0, 1] — values above 1.0 are a normalisation error in SIFT
    # keypoint counting.
    clone_ratio = min(1.0, max(0.0, float(clone.get("clone_ratio", 0.0))))

    # ── text_alignment_score (0–100) ──────────────────────────────────────────
    # Not yet extracted by any engine layer.  Set to 0 so the model uses the
    # other 10 features.  Phase 2 will add a real alignment detector.
    text_alignment_score = 0.0

    # ── font_inconsistency (0–100) ────────────────────────────────────────────
    # Combines stroke coefficient-of-variation and anomaly cluster count.
    # stroke_cv > 0.4 is the existing high-suspicion threshold.
    # Multiplying by 120 puts cv=0.83 at 100; adding 5 per anomaly region.
    font_stroke_cv = float(font.get("stroke_cv", 0.0)) if not font.get("skipped") else 0.0
    font_regions = int(font.get("n_suspicious_regions", 0))
    font_inconsistency = min(100.0, font_stroke_cv * 120 + font_regions * 5)

    # ── signature_mismatch (–1 = not applicable) ─────────────────────────────
    # Image forensics does not yet have a signature comparison layer.
    # Use -1 as the "not applicable" sentinel; the model handles this via
    # SimpleImputer(strategy="median") fitted during training.
    signature_mismatch = -1.0

    # ── noise_hotspots (0–100) ────────────────────────────────────────────────
    # hotspot_tile_ratio is a fraction (0–1).  Multiply by 1000 so a ratio of
    # 0.10 (existing threshold for suspicious) maps to score=100.
    noise_hotspot_ratio = float(noise.get("hotspot_tile_ratio", 0.0))
    noise_hotspots = min(100.0, noise_hotspot_ratio * 1000)

    # ── color_patch_score (0–100) ─────────────────────────────────────────────
    # anomaly_ratio is a fraction (0–1).  Multiplying by 2000 puts 0.05 at 100.
    # Adding a log-scaled blob size boosts large anomalous regions.
    color_ratio = float(color.get("anomaly_ratio", 0.0))
    largest_blob = int(blobs[0]["area_px"]) if blobs else 0
    color_patch_score = min(100.0, color_ratio * 2000 + math.log1p(largest_blob) * 3)

    # ── ai_artifact_score (0–100) ─────────────────────────────────────────────
    # spike_ratio from FFT analysis.  Values below 1.0 are baseline noise;
    # values above 3.5 indicate strong AI/GAN upsampling artefacts.
    # Mapping: (spike_ratio - 1.0) * 20 so ratio=6.0 → 100.
    spike_ratio = float(ai.get("spike_ratio", 0.0))
    ai_artifact_score = min(100.0, max(0.0, (spike_ratio - 1.0) * 20.0))

    # ── compression_mismatch (0–100) ──────────────────────────────────────────
    # Derived from ELA standard deviation (high std = uneven compression history)
    # and the dct_score computed above (double-compression component).
    ela_std = float(ela.get("std_ela", 0.0))
    compression_mismatch = min(100.0, ela_std * 6 + dct_score * 0.4)

    values = [
        ela_score,
        dct_score,
        float(metadata_flag_count),
        clone_ratio,
        text_alignment_score,
        font_inconsistency,
        signature_mismatch,
        noise_hotspots,
        color_patch_score,
        ai_artifact_score,
        compression_mismatch,
    ]

    log.debug(
        "ml_scorer: feature_vector extracted",
        extra=dict(zip(FEATURE_NAMES, [round(v, 3) for v in values])),
    )
    return np.array(values, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Predict
# ─────────────────────────────────────────────────────────────────────────────

def _load_model() -> Any:
    """Load (and cache) the XGBoost model from disk.

    Returns None when the model file does not exist yet — this is the normal
    cold-start state before train_ml_scorer.py has been run.
    """
    global _MODEL_CACHE, _MODEL_LOAD_ATTEMPTED
    if _MODEL_LOAD_ATTEMPTED:
        return _MODEL_CACHE

    _MODEL_LOAD_ATTEMPTED = True
    if not _MODEL_PATH.exists():
        # Warn at INFO so this is always visible in the default log level.
        # If the UI shows '📐 Heuristic' instead of '🤖 ML', this message
        # tells you exactly which path was checked and found missing.
        log.warning(
            "ml_scorer: model file not found — falling back to heuristic for all scans",
            extra={"checked_path": str(_MODEL_PATH)},
        )
        return None

    try:
        import joblib  # noqa: PLC0415
        _MODEL_CACHE = joblib.load(_MODEL_PATH)
        log.info(
            "ml_scorer: XGBoost model loaded successfully",
            extra={"path": str(_MODEL_PATH)},
        )
        return _MODEL_CACHE
    except Exception as exc:
        log.warning(
            "ml_scorer: model load failed — falling back to heuristic for all scans",
            extra={"path": str(_MODEL_PATH), "error": str(exc)},
        )
        return None


def predict(feature_vector: "np.ndarray") -> Optional[Dict[str, Any]]:  # type: ignore[name-defined]
    """Score a feature vector using the trained XGBoost model.

    Returns a dict with keys:
        score          float (0–100)  — forgery probability mapped to 0–100
        confidence     float (0–1)    — raw predict_proba output
        scoring_method str            — always "ML"

    Returns None when the model is not loaded, so the caller can fall back to
    the heuristic without any special handling.
    """
    model = _load_model()
    if model is None:
        return None

    try:
        import numpy as np  # noqa: PLC0415

        # Replace the signature_mismatch sentinel (-1) with NaN so the
        # pipeline imputer handles it correctly.
        vec = feature_vector.copy().reshape(1, -1)
        vec[vec == -1.0] = float("nan")

        # predict_proba returns [[P(genuine), P(tampered)]]
        proba = model.predict_proba(vec)[0]
        p_tampered = float(proba[1])
        score = round(p_tampered * 100, 1)

        log.info(
            "ml_scorer: XGBoost prediction complete",
            extra={"p_tampered": round(p_tampered, 4), "ml_score": score},
        )
        return {
            "score": score,
            "confidence": round(p_tampered, 4),
            "scoring_method": "ML",
        }
    except Exception as exc:
        log.warning(
            "ml_scorer: prediction failed — falling back to heuristic",
            extra={"error": str(exc)},
        )
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Train  (called by scripts/train_ml_scorer.py)
# ─────────────────────────────────────────────────────────────────────────────

# Column names to drop before training — identity/leakage columns that must
# never be fed to the model.
_DROP_COLS = {"doc_id", "filename", "hard_case", "heuristic_score", "heuristic_verdict",
              "file_entropy_bits", "dct_skipped", "font_skipped",
              "file_size_bytes", "ela_mean", "ela_max", "ela_std",
              "ela_suspicious_block_ratio", "noise_hotspot_ratio", "dct_comb_ratio",
              "color_anomaly_ratio", "color_largest_blob_px", "edge_high_density_ratio",
              "saturation_ratio", "font_stroke_cv", "font_suspicious_regions",
              "font_sharpness_outlier_ratio", "ai_spike_ratio", "metadata_flag_count_raw"}


def train(csv_paths: List[str], output_pkl: str) -> Dict[str, Any]:
    """Train an XGBoost classifier and save to *output_pkl*.

    Accepts a list of CSV paths so both the expert 10k dataset and our own
    real-image CSV can be combined.  Each CSV must have a 'label' column.

    CSVs that use our raw engine column schema are automatically remapped to
    the 11 normalised features via _remap_raw_csv().

    Returns a metrics dict with accuracy, f1, roc_auc.
    Raises ValueError if ROC AUC < 0.80 (prevents saving a bad model).
    """
    import numpy as np  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415
    from sklearn.model_selection import StratifiedKFold, cross_validate  # noqa: PLC0415
    from sklearn.metrics import roc_auc_score, f1_score, accuracy_score  # noqa: PLC0415
    from sklearn.pipeline import Pipeline  # noqa: PLC0415
    from sklearn.impute import SimpleImputer  # noqa: PLC0415
    import joblib  # noqa: PLC0415

    try:
        from xgboost import XGBClassifier  # noqa: PLC0415
    except ImportError:
        # Fall back to Random Forest if XGBoost not installed
        from sklearn.ensemble import RandomForestClassifier as XGBClassifier  # noqa: PLC0415
        log.warning("ml_scorer: xgboost not found, using RandomForestClassifier")

    frames = []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        # If the CSV uses our raw engine schema, remap it to the expert schema.
        if "ela_suspicious_block_ratio" in df.columns:
            df = _remap_raw_csv(df)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    log.info("ml_scorer: training on combined dataset", extra={"rows": len(combined), "sources": len(frames)})

    # Drop identity / leakage columns that are present in some CSVs
    drop_present = [c for c in _DROP_COLS if c in combined.columns]
    combined.drop(columns=drop_present, inplace=True)

    # Separate features from label
    X = combined[FEATURE_NAMES].copy().astype(float)
    # Replace the signature_mismatch sentinel (-1) with NaN for imputation
    X.replace(-1.0, float("nan"), inplace=True)
    y = combined["label"].astype(int)

    # Build a pipeline: median imputation (handles -1/NaN) → XGBoost
    try:
        from xgboost import XGBClassifier as _XGB  # noqa: PLC0415
        model_step = _XGB(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            scale_pos_weight=1,
            eval_metric="logloss",
            random_state=42,
        )
    except ImportError:
        from sklearn.ensemble import RandomForestClassifier as _RF  # noqa: PLC0415
        model_step = _RF(n_estimators=300, class_weight="balanced", random_state=42)

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", model_step),
    ])

    # 5-fold stratified cross-validation for honest evaluation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(
        pipe, X, y, cv=cv,
        scoring=["accuracy", "f1", "roc_auc"],
        return_train_score=False,
    )

    mean_accuracy = float(np.mean(cv_results["test_accuracy"]))
    mean_f1 = float(np.mean(cv_results["test_f1"]))
    mean_roc_auc = float(np.mean(cv_results["test_roc_auc"]))

    log.info(
        "ml_scorer: cross-validation results",
        extra={"accuracy": round(mean_accuracy, 4), "f1": round(mean_f1, 4), "roc_auc": round(mean_roc_auc, 4)},
    )

    # Hard-case evaluation (rows where hard_case=1 in original expert CSV)
    # These are the expert-level forgeries that are hardest to detect.
    hard_case_metrics = {}
    for csv_path in csv_paths:
        raw = pd.read_csv(csv_path)
        if "hard_case" in raw.columns:
            hard = raw[raw["hard_case"] == 1].copy()
            if len(hard) > 0:
                if "ela_suspicious_block_ratio" in hard.columns:
                    hard = _remap_raw_csv(hard)
                hard_X = hard[FEATURE_NAMES].copy().astype(float).replace(-1.0, float("nan"))
                hard_y = hard["label"].astype(int)
                # Fit on full data, evaluate on hard cases
                pipe.fit(X, y)
                hard_pred = pipe.predict(hard_X)
                hard_case_metrics = {
                    "n_hard": len(hard),
                    "accuracy": round(float(accuracy_score(hard_y, hard_pred)), 4),
                    "f1": round(float(f1_score(hard_y, hard_pred, zero_division=0)), 4),
                }
                log.info("ml_scorer: hard_case evaluation", extra=hard_case_metrics)
            break

    # Minimum quality gate — refuse to save a model that is clearly underfitting
    if mean_roc_auc < 0.80:
        raise ValueError(
            f"Training aborted: ROC AUC {mean_roc_auc:.4f} < 0.80 minimum target. "
            "Improve the dataset before saving."
        )

    # Fit on the full combined dataset before saving
    pipe.fit(X, y)

    # Check for dominant single feature (>50% importance = likely leakage)
    try:
        importances = pipe.named_steps["model"].feature_importances_
        max_importance = float(max(importances))
        dominant_feature = FEATURE_NAMES[list(importances).index(max(importances))]
        if max_importance > 0.50:
            log.warning(
                "ml_scorer: dominant feature detected — possible leakage",
                extra={"feature": dominant_feature, "importance": round(max_importance, 4)},
            )
        feature_importance_dict = dict(zip(FEATURE_NAMES, [round(float(v), 4) for v in importances]))
    except AttributeError:
        feature_importance_dict = {}

    # Save the trained pipeline
    output_path = Path(output_pkl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, output_path)
    log.info("ml_scorer: model saved", extra={"path": str(output_path)})

    # Invalidate the module-level cache so the new model is loaded next time
    global _MODEL_CACHE, _MODEL_LOAD_ATTEMPTED
    _MODEL_CACHE = None
    _MODEL_LOAD_ATTEMPTED = False

    return {
        "rows_trained": len(combined),
        "accuracy": round(mean_accuracy, 4),
        "f1": round(mean_f1, 4),
        "roc_auc": round(mean_roc_auc, 4),
        "hard_case": hard_case_metrics,
        "feature_importances": feature_importance_dict,
        "model_path": str(output_path),
    }


def _remap_raw_csv(df: "pd.DataFrame") -> "pd.DataFrame":  # type: ignore[name-defined]
    """Remap our raw engine CSV schema to the 11 normalised expert feature schema.

    Our collect_training_samples.py writes raw engine values (e.g. ela_suspicious
    _block_ratio as a raw fraction).  The expert dataset uses 0–100 scores.
    This function applies the same normalisation formulas as extract_feature_vector()
    so both datasets end up in the same feature space.
    """
    import numpy as np  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415

    out = pd.DataFrame()

    ela_block = df.get("ela_suspicious_block_ratio", pd.Series(0.0, index=df.index)).astype(float)
    ela_mean = df.get("ela_mean", pd.Series(0.0, index=df.index)).astype(float)
    ela_std = df.get("ela_std", pd.Series(0.0, index=df.index)).astype(float)
    dct_comb = df.get("dct_comb_ratio", pd.Series(1.0, index=df.index)).astype(float)
    dct_skip = df.get("dct_skipped", pd.Series(0, index=df.index)).astype(int)
    meta_flags = df.get("metadata_flag_count", pd.Series(0, index=df.index)).astype(float)
    clone_raw = df.get("clone_ratio", pd.Series(0.0, index=df.index)).astype(float)
    noise_h = df.get("noise_hotspot_ratio", pd.Series(0.0, index=df.index)).astype(float)
    color_r = df.get("color_anomaly_ratio", pd.Series(0.0, index=df.index)).astype(float)
    blob_px = df.get("color_largest_blob_px", pd.Series(0, index=df.index)).astype(float)
    font_cv = df.get("font_stroke_cv", pd.Series(0.0, index=df.index)).astype(float)
    font_reg = df.get("font_suspicious_regions", pd.Series(0, index=df.index)).astype(float)
    font_skip = df.get("font_skipped", pd.Series(0, index=df.index)).astype(int)
    ai_spike = df.get("ai_spike_ratio", pd.Series(0.0, index=df.index)).astype(float)

    out["ela_score"] = (ela_block * 800 + ela_mean * 2).clip(0, 100)
    raw_dct = dct_comb.where(dct_skip == 0, other=1.0)
    out["dct_score"] = ((raw_dct - 1.0) * 60).clip(0, 100)
    out["metadata_flag_count"] = meta_flags.clip(0, 5)
    out["clone_ratio"] = clone_raw.clip(0.0, 1.0)
    out["text_alignment_score"] = 0.0
    font_cv_clean = font_cv.where(font_skip == 0, other=0.0)
    out["font_inconsistency"] = (font_cv_clean * 120 + font_reg * 5).clip(0, 100)
    out["signature_mismatch"] = -1.0  # not available in our dataset
    out["noise_hotspots"] = (noise_h * 1000).clip(0, 100)
    out["color_patch_score"] = (color_r * 2000 + np.log1p(blob_px) * 3).clip(0, 100)
    out["ai_artifact_score"] = ((ai_spike - 1.0) * 20).clip(0, 100)
    out["compression_mismatch"] = (ela_std * 6 + out["dct_score"] * 0.4).clip(0, 100)
    out["label"] = df["label"].astype(int)

    return out
