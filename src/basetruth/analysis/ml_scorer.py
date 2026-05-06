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

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from basetruth.logger import get_logger

log = get_logger(__name__)

# ── Path to the saved model ────────────────────────────────────────────────────
# Resolved relative to the repo root so the path works whether the code is run
# from the project root, src/, or as a package inside a virtualenv.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent  # src/basetruth/analysis → repo
_MODEL_PATH = _REPO_ROOT / "fraud_model" / "models" / "ml_scorer_image.pkl"

# Cache the loaded model in module-level state so we only deserialise it once
# per process, not once per document scan.
_MODEL_CACHE: Any = None
_MODEL_LOAD_ATTEMPTED: bool = False


# ── Binary verdict taxonomy ───────────────────────────────────────────────────
# The ML model is a binary XGBoost classifier.
#
#   0  ORIGINAL    — genuine document with no manipulation of any kind
#   1  FAKE/EDITED — covers re-saved originals (ORIGINAL-DERIVED), directly
#                    tampered documents (TAMPERED), and re-saved tampered
#                    documents (TAMPERED-DERIVED).
#
# Re-saves of genuine documents are treated as suspicious in a KYC fraud context
# because legitimate users do not re-encode their bank statements or ID scans.
#
# The "fraud score" (0-100) is p[FAKE/EDITED] × 100.
ML_VERDICT_LABELS: Dict[int, str] = {
    0: "ORIGINAL",
    1: "FAKE/EDITED",
}

# ── Feature names — 18 signals used for training ─────────────────────────────
# file_entropy_bits (Layer 3) was removed — all values are 0.0 in the dataset.
# XGBoost is scale-invariant so raw fractions and raw pixel counts work fine
# without manual normalisation.
FEATURE_NAMES: List[str] = [
    # Layer 1: ELA — all 4 sub-signals kept separately so the model can learn
    # their individual contribution rather than a hard-coded weighted sum.
    "ela_mean",
    "ela_max",
    "ela_std",
    "ela_suspicious_block_ratio",
    # Layer 2: Metadata
    "metadata_flag_count",
    # Layer 4: Noise  (Layer 3 file_entropy_bits dropped — all values 0.0 in dataset)
    "noise_hotspot_ratio",
    # Layer 5: DCT double-compression
    "dct_comb_ratio",
    "dct_skipped",       # binary: 1 = non-JPEG, 0 = JPEG processed
    # Layer 6: Clone detection
    "clone_ratio",
    # Layer 7: Colour anomaly
    "color_anomaly_ratio",
    "color_largest_blob_px",
    # Layer 8: Edge density — unnaturally sharp tile boundaries
    "edge_high_density_ratio",
    # Layer 9: Saturation
    "saturation_ratio",
    # Layer 10: Font consistency
    "font_stroke_cv",
    "font_suspicious_regions",
    "font_sharpness_outlier_ratio",
    "font_skipped",      # binary: 1 = no text found / skipped
    # Layer 11: AI/FFT artefacts
    "ai_spike_ratio",
]


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_feature_vector(layers: Dict[str, Any]) -> "np.ndarray":  # type: ignore[name-defined]
    """Return all 19 raw engine metrics as a float32 array (one value per FEATURE_NAMES entry).

    Each value is taken directly from the forensic layer result — no manual
    aggregation is applied.  XGBoost is scale-invariant so raw fractions (0–1)
    and raw pixel counts work as well as normalised 0–100 scores.

    Layers 3 (entropy), 8 (edge) and 9 (saturation) were previously ignored;
    they are now included so all 19 raw signals reach the model.

    Missing or skipped layers produce 0.0 so the vector is always complete.
    """
    import numpy as np  # noqa: PLC0415

    # Pull the metrics sub-dict from each forensic layer result.
    ela   = layers.get("layer_1_ela", {}).get("metrics", {})
    meta  = layers.get("layer_2_metadata", {})
    # layer_3_entropy (file_entropy_bits) dropped — all values are 0.0 in real scans
    noise = layers.get("layer_4_noise", {}).get("metrics", {})
    dct   = layers.get("layer_5_dct", {}).get("metrics", {})
    clone = layers.get("layer_6_clone", {}).get("metrics", {})
    color = layers.get("layer_7_color", {}).get("metrics", {})
    blobs = color.get("anomaly_blobs", [])
    edge  = layers.get("layer_8_edge", {}).get("metrics", {})      # was never read before
    sat   = layers.get("layer_9_saturation", {}).get("metrics", {}) # was never read before
    font  = layers.get("layer_10_font", {}).get("metrics", {})
    ai    = layers.get("layer_11_ai", {}).get("metrics", {})

    # Boolean skip guards — keep as 0.0/1.0 float so the model can learn that
    # "dct_skipped=1" means the file is a PNG (not a tampered JPEG), and
    # "font_skipped=1" means no text regions were found.
    dct_skipped  = 1.0 if dct.get("skipped") else 0.0
    font_skipped = 1.0 if font.get("skipped") else 0.0

    values = [
        # Layer 1 — ELA (4 raw sub-signals; model learns their optimal weights)
        float(ela.get("mean_ela", 0.0)),
        float(ela.get("max_ela", 0.0)),
        float(ela.get("std_ela", 0.0)),
        float(ela.get("suspicious_block_ratio", 0.0)),

        # Layer 2 — Metadata flag count
        float(len(meta.get("suspicious_flags", []))),

        # Layer 4 — Noise hotspot fraction  (Layer 3 entropy removed — always 0.0)
        float(noise.get("hotspot_tile_ratio", 0.0)),

        # Layer 5 — DCT double-compression ratio and skip flag
        float(dct.get("comb_ratio", 0.0)) if not dct_skipped else 0.0,
        dct_skipped,

        # Layer 6 — Clone detection ratio
        min(1.0, float(clone.get("clone_ratio", 0.0))),

        # Layer 7 — Colour anomaly
        float(color.get("anomaly_ratio", 0.0)),
        float(blobs[0]["area_px"]) if blobs else 0.0,

        # Layer 8 — Edge density (previously ignored entirely)
        float(edge.get("high_density_tile_ratio", 0.0)),

        # Layer 9 — Saturation (previously ignored entirely)
        float(sat.get("high_saturation_tile_ratio", 0.0)),

        # Layer 10 — Font consistency (3 sub-signals + skip flag)
        float(font.get("stroke_cv", 0.0)) if not font_skipped else 0.0,
        float(font.get("n_suspicious_regions", 0)) if not font_skipped else 0.0,
        float(font.get("sharpness_outlier_ratio", 0.0)) if not font_skipped else 0.0,
        font_skipped,

        # Layer 11 — AI/FFT artefact spike ratio
        float(ai.get("spike_ratio", 0.0)),
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
    """Score a feature vector using the trained binary XGBoost model.

    Returns a dict with keys:
        score          float (0–100)  — fraud probability: P(FAKE/EDITED) × 100
        confidence     float (0–1)    — same value in 0–1 range
        scoring_method str            — always "ML"
        ml_verdict     str            — predicted class name from ML_VERDICT_LABELS
                                        (ORIGINAL | FAKE/EDITED)

    Returns None when the model is not loaded, so the caller can fall back to
    the heuristic without any special handling.

    The score (0–100) is the probability of the FAKE/EDITED class.
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

        # predict_proba returns shape (1, n_classes).
        # For binary: [[P(ORIGINAL), P(FAKE/EDITED)]]
        proba = model.predict_proba(vec)[0]
        n_classes = len(proba)

        if n_classes == 2:
            # Binary model — class 1 = FAKE/EDITED
            p_fraud = float(proba[1])
            ml_class = int(np.argmax(proba))
            ml_verdict = ML_VERDICT_LABELS.get(ml_class, "FAKE/EDITED")
        elif n_classes == 4:
            # Legacy 4-class mode support
            p_fraud = float(proba[2]) + float(proba[3])
            ml_class = int(np.argmax(proba))
            # Fallback to names if labels dict was updated
            _old_labels = {0: "ORIGINAL", 1: "ORIGINAL-DERIVED", 2: "TAMPERED", 3: "TAMPERED-DERIVED"}
            ml_verdict = _old_labels.get(ml_class, "TAMPERED")
        else:
            p_fraud = float(proba[-1])
            ml_verdict = "FAKE/EDITED" if np.argmax(proba) > 0 else "ORIGINAL"

        score = round(p_fraud * 100, 1)

        log.info(
            "ml_scorer: XGBoost prediction complete",
            extra={"p_fraud": round(p_fraud, 4), "ml_score": score, "ml_verdict": ml_verdict},
        )
        return {
            "score": score,
            "confidence": round(p_fraud, 4),
            "scoring_method": "ML",
            "ml_verdict": ml_verdict,
        }
    except Exception as exc:
        log.warning(
            "ml_scorer: prediction failed — falling back to heuristic",
            extra={"error": str(exc)},
        )
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Step 2b — Per-sample SHAP contributions (tree SHAP, built into XGBoost)
# ─────────────────────────────────────────────────────────────────────────────

def explain(feature_vector: "np.ndarray") -> Optional[Dict[str, float]]:  # type: ignore[name-defined]
    """Compute per-feature SHAP contribution values for a single image scan.

    Uses XGBoost's built-in tree SHAP — no separate 'shap' package required.
    Each value shows how much a feature pushed the prediction toward FAKE/EDITED
    (positive) or ORIGINAL (negative) in log-odds units.

    signature_mismatch sentinels (-1) are replaced with NaN before imputation
    so the pipeline's SimpleImputer fills them with the training-set median,
    exactly as predict() does.

    Returns a dict {feature_name: shap_value} with 18 entries, or None if
    the model is unavailable or the booster raises any error.
    """
    model = _load_model()
    if model is None:
        return None

    try:
        import numpy as np  # noqa: PLC0415
        import xgboost as xgb  # noqa: PLC0415

        # Replace the signature_mismatch sentinel (-1) with NaN so the imputer
        # treats it as missing, consistent with how predict() handles it.
        vec = feature_vector.copy().reshape(1, -1)
        vec[vec == -1.0] = float("nan")

        # Apply the sklearn imputer step to fill any NaN/missing values.
        imputer = model.named_steps["imputer"]
        imputed = imputer.transform(vec)

        booster = model.named_steps["model"].get_booster()

        # Guard: the saved booster may have been trained with fewer features than
        # the current FEATURE_NAMES list (e.g. a new feature was added to the code
        # after the last model training run).  Trimming the imputed array to the
        # booster's actual feature count prevents the XGBoost SHAP call from
        # raising "expected N, got M" — features beyond the booster's training
        # horizon simply produce no contribution until the model is retrained.
        n_booster = booster.num_features()
        imputed_trimmed = imputed[:, :n_booster]
        active_names = FEATURE_NAMES[:n_booster]

        # Wrap in DMatrix with named features for a consistent column order.
        dmat = xgb.DMatrix(imputed_trimmed, feature_names=active_names)

        # pred_contribs=True triggers XGBoost tree SHAP.
        # For binary models it returns a 2-D array:
        #   shape (n_samples, n_booster + 1)
        # For legacy 4-class models it returns a 3-D array:
        #   shape (n_samples, n_classes, n_booster + 1)
        shap_matrix = booster.predict(dmat, pred_contribs=True)

        if shap_matrix.ndim == 2:
            # Binary model — last column is bias, drop it.
            shap_values = shap_matrix[0, :-1]
        elif shap_matrix.ndim == 3:
            # Legacy 4-class multiclass support
            tampered_shap = shap_matrix[0, 2, :-1] + shap_matrix[0, 3, :-1]
            genuine_shap  = shap_matrix[0, 0, :-1] + shap_matrix[0, 1, :-1]
            shap_values = tampered_shap - genuine_shap
        else:
            shap_values = shap_matrix[0, ..., :-1].flatten()

        contributions = {
            name: round(float(v), 4)
            for name, v in zip(active_names, shap_values)
        }
        log.debug(
            "ml_scorer: SHAP contributions computed",
            extra={"n_features": len(contributions)},
        )
        return contributions
    except Exception as exc:
        log.warning(
            "ml_scorer: explain failed — contributions unavailable",
            extra={"error": str(exc)},
        )
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Train  (called by scripts/train_ml_scorer.py)
# ─────────────────────────────────────────────────────────────────────────────

# Columns to drop before training — identity / leakage columns only.
# The raw forensic signal columns are now ALL used as features (FEATURE_NAMES)
# so they must NOT be listed here.
_DROP_COLS = {
    "doc_id", "filename",
    "hard_case",               # data-collection flag, not a forensic signal
    "heuristic_score",         # derived from the same signals — would be target leakage
    "heuristic_verdict",       # same reason
    "file_size_bytes",         # file size alone is not a reliable fraud signal
    "file_entropy_bits",       # all values are 0.0 — entropy layer not populating correctly
    "metadata_flag_count_raw", # raw pre-dedup version; metadata_flag_count is used instead
    "source",                  # present in the production dataset CSV; not a feature
    "sample_weight",           # sampling weight; not a forensic signal
    "file_format",             # categorical string; not currently encoded
    "signature_mismatch_score",   # not implemented; always -1
    "text_alignment_score",       # not implemented; always 0
    "compression_mismatch_score", # derived aggregate; raw sub-signals already present
}


def train(csv_paths: List[str], output_pkl: str, progress_cb: Optional[Any] = None) -> Dict[str, Any]:
    """Train an XGBoost classifier and save to *output_pkl*.

    Accepts a list of CSV paths so both the expert 10k dataset and our own
    real-image CSV can be combined.  Each CSV must have a 'label' column.

    CSVs that use our raw engine column schema are automatically remapped to
    the 11 normalised features via _remap_raw_csv().

    progress_cb is an optional callable(step: str, pct: int) that receives
    human-readable step descriptions and a 0-100 completion percentage.  It is
    called from this thread so it must be non-blocking.

    Returns a metrics dict with accuracy, f1, roc_auc.
    Raises ValueError if ROC AUC < 0.80 (prevents saving a bad model).
    """
    def _emit(step: str, pct: int) -> None:
        """Fire the optional progress callback — swallow exceptions so training never fails because of it."""
        if progress_cb:
            try:
                progress_cb(step, pct)
            except Exception:  # noqa: BLE001
                pass

    import numpy as np  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415
    from sklearn.model_selection import StratifiedKFold  # noqa: PLC0415
    from sklearn.metrics import f1_score, accuracy_score, roc_auc_score  # noqa: PLC0415
    from sklearn.pipeline import Pipeline  # noqa: PLC0415
    from sklearn.impute import SimpleImputer  # noqa: PLC0415
    from sklearn.base import clone as _clone_pipe  # noqa: PLC0415
    import joblib  # noqa: PLC0415

    try:
        from xgboost import XGBClassifier  # noqa: PLC0415
    except ImportError:
        # Fall back to Random Forest if XGBoost not installed
        from sklearn.ensemble import RandomForestClassifier as XGBClassifier  # noqa: PLC0415
        log.warning("ml_scorer: xgboost not found, using RandomForestClassifier")

    _emit("Loading training data files...", 3)
    frames = []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        # If the CSV uses our raw engine schema, remap it to the expert schema.
        if "ela_suspicious_block_ratio" in df.columns:
            df = _remap_raw_csv(df)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    # Show original raw-label distribution before binarising
    _raw_label_names = {0: "ORIGINAL", 1: "ORIGINAL-DERIVED", 2: "TAMPERED", 3: "TAMPERED-DERIVED"}
    raw_labels = sorted(combined["label"].astype(int).unique().tolist())
    _emit(
        f"Loaded {len(combined)} samples  —  raw classes: "
        + ", ".join(_raw_label_names.get(l, str(l)) for l in raw_labels)
        + "  →  binarising to ORIGINAL / FAKE/EDITED",
        8,
    )
    log.info(
        "ml_scorer: training on combined dataset",
        extra={"rows": len(combined), "sources": len(frames), "raw_labels": raw_labels},
    )

    # Drop identity / leakage columns that are present in some CSVs
    drop_present = [c for c in _DROP_COLS if c in combined.columns]
    combined.drop(columns=drop_present, inplace=True)

    # Separate features from label.
    # For columns not present in a CSV (e.g. an older CSV missing edge/saturation),
    # fill with 0.0 so both old and new datasets can be combined safely.
    for col in FEATURE_NAMES:
        if col not in combined.columns:
            combined[col] = 0.0
    X = combined[FEATURE_NAMES].copy().astype(float)

    # ── Binary label conversion ──────────────────────────────────────────────────
    # 0 = ORIGINAL    (phone-fresh genuine document)
    # 1 = FAKE/EDITED (ORIGINAL-DERIVED re-saves + TAMPERED + TAMPERED-DERIVED)
    # Re-saved genuine documents are suspicious in a KYC context and the dataset
    # is too small (~470 rows) to reliably learn four separate classes.
    y = (combined["label"].astype(int) > 0).astype(int)
    unique_labels = [0, 1]
    is_multiclass = False

    # Build a pipeline: median imputation (handles -1/NaN) → XGBoost (binary)
    try:
        from xgboost import XGBClassifier as _XGB  # noqa: PLC0415
        model_step = _XGB(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
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

    # 5-fold stratified cross-validation — manual loop so we can emit per-fold
    # progress messages to the UI in real time.
    _emit("Starting 5-fold cross-validation...", 18)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_accuracies: List[float] = []
    fold_f1s:        List[float] = []
    fold_aucs:       List[float] = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X.values, y.values)):
        _emit(f"Cross-validation  —  fold {fold_idx + 1} / 5  (training...)", 20 + fold_idx * 10)

        # Clone the pipeline so each fold gets a fresh, unfitted copy.
        pipe_fold = _clone_pipe(pipe)
        pipe_fold.fit(X.iloc[train_idx], y.iloc[train_idx])

        fold_pred  = pipe_fold.predict(X.iloc[val_idx])
        fold_proba = pipe_fold.predict_proba(X.iloc[val_idx])
        fold_y     = y.iloc[val_idx]

        fold_acc = float(accuracy_score(fold_y, fold_pred))
        fold_f1  = float(f1_score(fold_y, fold_pred, average="binary", zero_division=0))
        fold_auc = float(roc_auc_score(fold_y, fold_proba[:, 1]))

        fold_accuracies.append(fold_acc)
        fold_f1s.append(fold_f1)
        fold_aucs.append(fold_auc)
        _emit(
            f"Fold {fold_idx + 1} / 5 done  —  accuracy {fold_acc:.1%}  |  F1 {fold_f1:.1%}  |  AUC {fold_auc:.3f}",
            22 + (fold_idx + 1) * 10,
        )

    mean_accuracy = float(np.mean(fold_accuracies))
    mean_f1       = float(np.mean(fold_f1s))
    mean_roc_auc  = float(np.mean(fold_aucs))

    log.info(
        "ml_scorer: cross-validation results",
        extra={
            "accuracy": round(mean_accuracy, 4),
            "f1": round(mean_f1, 4),
            "roc_auc": round(mean_roc_auc, 4),
            "mode": "binary (ORIGINAL vs FAKE/EDITED)",
        },
    )
    _emit(
        f"Cross-validation complete  —  avg accuracy {mean_accuracy:.1%}  |  F1 {mean_f1:.1%}  |  AUC {mean_roc_auc:.3f}",
        73,
    )

    # Hard-case evaluation (rows where hard_case=1 in original expert CSV)
    _emit("Checking hard-case performance...", 76)
    hard_case_metrics = {}
    for csv_path in csv_paths:
        raw = pd.read_csv(csv_path)
        if "hard_case" in raw.columns:
            hard = raw[raw["hard_case"] == 1].copy()
            if len(hard) > 0:
                if "ela_suspicious_block_ratio" in hard.columns:
                    hard = _remap_raw_csv(hard)
                hard_X = hard[FEATURE_NAMES].copy().astype(float).replace(-1.0, float("nan"))
                # Binarize hard-case labels to match the training scheme
                hard_y = (hard["label"].astype(int) > 0).astype(int)
                # Fit on full data, evaluate on hard cases
                pipe.fit(X, y)
                hard_pred = pipe.predict(hard_X)
                # Use weighted averaging so hard-case F1 works in both binary and multiclass
                avg = "binary"
                hard_case_metrics = {
                    "n_hard": len(hard),
                    "accuracy": round(float(accuracy_score(hard_y, hard_pred)), 4),
                    "f1": round(float(f1_score(hard_y, hard_pred, average=avg, zero_division=0)), 4),
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
    _emit("Fitting final model on all training data...", 84)
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
    _emit("Saving model to disk...", 93)
    output_path = Path(output_pkl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, output_path)
    log.info("ml_scorer: model saved", extra={"path": str(output_path)})
    _emit(f"Model saved  —  {str(output_path)}", 97)

    # Invalidate the module-level cache so the new model is loaded next time
    global _MODEL_CACHE, _MODEL_LOAD_ATTEMPTED
    _MODEL_CACHE = None
    _MODEL_LOAD_ATTEMPTED = False

    _emit(f"Training complete!  Accuracy {mean_accuracy:.1%}  |  F1 {mean_f1:.1%}  |  AUC {mean_roc_auc:.3f}", 100)
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
    """Pass the 19 raw engine columns through unchanged so they reach the model directly.

    Previously this function aggregated the raw columns into 11 normalised scores,
    discarding information in the process.  Now that FEATURE_NAMES lists the 19 raw
    column names, this function simply selects those columns and fills in 0.0 for
    any that are absent (e.g. old CSVs collected before edge/saturation layers were added).
    XGBoost learns the optimal combination of each sub-signal from data rather than
    relying on our hard-coded aggregation weights.
    """
    import pandas as pd  # noqa: PLC0415

    out = pd.DataFrame(index=df.index)
    for col in FEATURE_NAMES:
        if col in df.columns:
            out[col] = df[col].astype(float)
        else:
            # Column absent in this CSV (older dataset) — fill with 0.0 so the
            # SimpleImputer can still handle it cleanly during training.
            out[col] = 0.0
    out["label"] = df["label"].astype(int)
    return out
