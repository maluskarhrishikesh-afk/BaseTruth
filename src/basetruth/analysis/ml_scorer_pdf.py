"""ml_scorer_pdf.py — ML-based fraud scoring for the PDF forensic engine.

This module is the PDF counterpart of ml_scorer.py.  It works the same way:

  1. extract_feature_vector_pdf(layers)
       Converts raw PDF engine layer output into 18 features that match the
       schema of the expert training dataset
       (data/pdf_ultimate_dataset_10000_rows.csv).

  2. predict_pdf(feature_vector)
       Loads data/ml_scorer_pdf.pkl and calls predict_proba() to get
       P(tampered).  Returns None if the model file does not exist — the
       caller falls back to the heuristic automatically.

  3. train_pdf(csv_paths, output_pkl)
       Trains an XGBoost classifier, runs 5-fold stratified CV, and saves the
       model only if ROC AUC >= 0.80.

The 18 feature names exactly match the expert PDF dataset columns (minus
the non-feature columns pdf_id and label).  Our own 66-row CSV is remapped
to this schema by _remap_raw_pdf_csv() before training.

Cold-start guarantee: if ml_scorer_pdf.pkl is absent, every call to
predict_pdf() returns None and the existing heuristic runs unchanged.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from basetruth.logger import get_logger

log = get_logger(__name__)

# ── Model file path ────────────────────────────────────────────────────────────
# Resolved from here: src/basetruth/analysis/ → repo root is 4 levels up.
_REPO_ROOT  = Path(__file__).resolve().parent.parent.parent.parent
_MODEL_PATH = _REPO_ROOT / "data" / "ml_scorer_pdf.pkl"

# Module-level model cache — load once per process.
_MODEL_CACHE:          Any  = None
_MODEL_LOAD_ATTEMPTED: bool = False


# ── Feature names — must match the expert PDF training CSV column order ────────
# These 17 features are used verbatim by the trained XGBoost model.
# hard_subtle_case was removed from training because it is a dataset meta-flag
# (1 = difficult case), not a real forensic signal.  It is always 0 in live
# prediction, which means including it during training causes the model to learn
# a feature that never varies at runtime — reducing real-world discriminability.
# The model now trains on 17 real forensic features only.
PDF_FEATURE_NAMES: List[str] = [
    "incremental_updates",              # times the file was re-saved after creation
    "eof_marker_count",                 # number of %%EOF markers (clean=1)
    "metadata_anomaly_score",           # 0–100 score from metadata flags + date gap
    "hidden_text_spans",                # total spans with white/invisible text
    "white_text_spans",                 # subset of hidden spans in white colour
    "javascript_count",                 # number of JavaScript actions in PDF
    "embedded_files_count",             # number of attached/embedded files
    "signature_gap_score",              # unsigned byte-range gap after signatures
    "render_ela_suspicious_block_ratio",# ELA ratio on page-render (0–1)
    "render_noise_hotspot_ratio",       # noise hotspot ratio on page-render (0–1)
    "object_count",                     # total PDF objects in cross-reference
    "stream_entropy",                   # Shannon entropy of raw bytes (0–8)
    "xref_mismatch_score",              # 0–100 score for xref integrity issues (now LIVE)
    "font_switch_score",                # 0–100 proxy for font switching / mixing
    "ocr_text_layer_gap",               # gap between OCR and text layer (0–100)
    "is_scanned_pdf",                   # 1 if document appears to be scanned (now LIVE)
    "has_signature",                    # 1 if any digital signature field exists
]

# Columns to drop before training — identity, leakage, and constant fields.
# Per expert feedback: filename/heuristic fields cause leakage; several of our
# own columns (pdf_version, metadata_creator, non_embedded_fonts, etc.) are
# either non-numeric or constant across our 66-row supplement and add no signal.
_PDF_DROP_COLS = {
    "pdf_id",                       # identity column in expert CSV
    "filename",                     # identity column in our CSV
    "heuristic_score",              # our rule-engine output — leakage
    "heuristic_verdict",            # our rule-engine output — leakage
    "pdf_version",                  # version string, not a fraud signal
    "metadata_creator",             # free text, incompatible with XGBoost directly
    "metadata_date_gap_days",       # partially captured by metadata_anomaly_score
    "non_embedded_fonts",           # constant in our supplement data
    "tiny_size_spans",              # constant in our supplement data
    "has_open_action",              # not in expert schema; constant for most docs
    "has_xfa_form",                 # constant in our supplement data
    "page_count",                   # not in expert schema
    "blank_pages",                  # not in expert schema
    "unique_page_sizes",            # not in expert schema
    "signature_count",              # not in expert schema (gap_score captures coverage)
    "render_ela_mean",              # expert uses block_ratio, not raw mean
    "embedded_image_count",         # not in expert schema
    "embedded_noise_hotspot_ratio", # not in expert schema
    "obj_stm_count",                # constant in our supplement data
    "file_size_bytes",              # not a forensic signal
    "hard_subtle_case",             # dataset meta-flag — always 0 in live prediction;
                                    # removing it forces the model to learn from real
                                    # forensic signals only, improving live accuracy.
}


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Feature extraction (live path)
# ─────────────────────────────────────────────────────────────────────────────

def extract_feature_vector_pdf(layers: Dict[str, Any]) -> "np.ndarray":  # type: ignore[name-defined]
    """Convert raw PDF engine layer dict into 18 floats for the ML model.

    Each value is mapped to the same scale used by the expert training CSV so
    the model's learned thresholds apply directly.  All values default to 0.0
    when a layer was skipped or failed — the vector is always complete.

    Returns a numpy array of shape (18,) with dtype float32.
    """
    import numpy as np  # noqa: PLC0415

    # ── Layer 1: Incremental Updates ─────────────────────────────────────────
    inc = layers.get("layer_1_incremental_updates", {}).get("metrics", {})
    incremental_updates = float(inc.get("incremental_updates") or 0)
    eof_marker_count    = float(inc.get("eof_marker_count") or 1)

    # ── Layer 2: Metadata ─────────────────────────────────────────────────────
    # Derive a 0–100 metadata anomaly score from the layer status.
    # SUSPICIOUS status → look at how many pipe-separated flags appear in the
    # plain_english text.  Scale: each flag contributes ~20 points (max 5=100).
    meta = layers.get("layer_2_metadata", {})
    if meta.get("status") == "SUSPICIOUS":
        pe = meta.get("plain_english", "")
        n_flags = len(pe.replace("⚠️ ", "").split(" | "))
        metadata_anomaly_score = min(100.0, float(n_flags) * 20.0)
    else:
        metadata_anomaly_score = 0.0

    # ── Layer 4: Invisible Text ───────────────────────────────────────────────
    inv = layers.get("layer_4_invisible_text", {}).get("metrics", {})
    hidden_text_spans = float(inv.get("total_hidden_spans") or 0)
    white_text_spans  = float(inv.get("white_text_spans") or 0)

    # ── Layer 5: Suspicious Objects ───────────────────────────────────────────
    objs = layers.get("layer_5_suspicious_objects", {}).get("metrics", {})
    javascript_count    = float(objs.get("javascript_count") or 0)
    embedded_files_count= float(objs.get("embedded_files_count") or 0)

    # ── Layer 7: Digital Signature ────────────────────────────────────────────
    # The expert's signature_gap_score ranges 0–98 in the dataset.
    # Our engine returns an integer coverage_gaps count.  Scale by 10 to
    # produce a score in a comparable range; cap at 100.
    sig = layers.get("layer_7_digital_signature", {}).get("metrics", {})
    raw_gaps         = float(sig.get("coverage_gaps") or 0)
    signature_gap_score = min(100.0, raw_gaps * 10.0)
    has_signature    = float(bool(sig.get("has_signature_field")))

    # ── Layer 8: Page Render ELA ──────────────────────────────────────────────
    ela = layers.get("layer_8_page_render_ela", {}).get("metrics", {})
    render_ela_sbr   = float(ela.get("suspicious_block_ratio") or 0.0)
    render_noise_hr  = float(ela.get("noise_hotspot_ratio") or 0.0)

    # ── Layer 10: File Entropy ────────────────────────────────────────────────
    ent = layers.get("layer_10_file_entropy", {}).get("metrics", {})
    stream_entropy = float(ent.get("file_entropy_bits") or 0.0)

    # ── Layer 3: Font Consistency ─────────────────────────────────────────────
    # Expert's font_switch_score ranges 0–100 (mean ≈ 21).  Our engine gives
    # an integer total_unique_fonts which is on a similar natural scale.
    font = layers.get("layer_3_font_consistency", {}).get("metrics", {})
    font_switch_score = float(font.get("total_unique_fonts") or 0)

    # ── Layer 11: Object / XRef Integrity ─────────────────────────────────────
    xref = layers.get("layer_11_object_xref_integrity", {}).get("metrics", {})
    object_count = float(xref.get("total_objects") or 0)
    # xref_mismatch_score is now computed by object_integrity_analysis() and stored in
    # layer_11 metrics.  If the layer was skipped (older engine), default to 0.
    xref_mismatch_score = float(xref.get("xref_mismatch_score") or 0.0)

    # ── Scanned-PDF flag (from _meta key set by run_pdf_forensics) ────────────
    # is_scanned_pdf is 1 when the PDF has no machine-readable text layer (image-only).
    # run_pdf_forensics detects this by checking embedded text length with fitz and
    # writes the result to layers["_meta"]["is_scanned_pdf"].
    meta = layers.get("_meta", {})
    is_scanned_pdf = float(meta.get("is_scanned_pdf") or 0.0)

    # ── Features not yet extractable by the current engine ───────────────────
    # ocr_text_layer_gap: requires running OCR and diffing against the embedded text
    # layer — a future enhancement.  Set to 0.0 so the vector is always complete.
    ocr_text_layer_gap = 0.0

    values = [
        incremental_updates,
        eof_marker_count,
        metadata_anomaly_score,
        hidden_text_spans,
        white_text_spans,
        javascript_count,
        embedded_files_count,
        signature_gap_score,
        render_ela_sbr,
        render_noise_hr,
        object_count,
        stream_entropy,
        xref_mismatch_score,
        font_switch_score,
        ocr_text_layer_gap,
        is_scanned_pdf,
        has_signature,
    ]

    log.debug(
        "ml_scorer_pdf: feature_vector extracted",
        extra=dict(zip(PDF_FEATURE_NAMES, [round(v, 3) for v in values])),
    )
    return np.array(values, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Predict
# ─────────────────────────────────────────────────────────────────────────────

def _load_model() -> Any:
    """Load (and cache) the PDF XGBoost model from disk.

    Returns None when the model file does not exist — normal cold-start state
    before train_ml_scorer_pdf.py has been run.
    """
    global _MODEL_CACHE, _MODEL_LOAD_ATTEMPTED
    if _MODEL_LOAD_ATTEMPTED:
        return _MODEL_CACHE

    _MODEL_LOAD_ATTEMPTED = True
    if not _MODEL_PATH.exists():
        log.warning(
            "ml_scorer_pdf: model file not found — falling back to PDF heuristic for all scans",
            extra={"checked_path": str(_MODEL_PATH)},
        )
        return None

    try:
        import joblib  # noqa: PLC0415
        _MODEL_CACHE = joblib.load(_MODEL_PATH)
        log.info(
            "ml_scorer_pdf: XGBoost PDF model loaded successfully",
            extra={"path": str(_MODEL_PATH)},
        )
        return _MODEL_CACHE
    except Exception as exc:
        log.warning(
            "ml_scorer_pdf: model load failed — falling back to PDF heuristic",
            extra={"path": str(_MODEL_PATH), "error": str(exc)},
        )
        return None


def predict_pdf(feature_vector: "np.ndarray") -> Optional[Dict[str, Any]]:  # type: ignore[name-defined]
    """Score a PDF feature vector using the trained XGBoost model.

    Returns a dict with keys:
        score          float (0–100)  — forgery probability mapped to 0–100
        confidence     float (0–1)    — raw predict_proba output
        scoring_method str            — always "ML"

    Returns None when the model is not loaded so the caller falls back to the
    PDF heuristic without any special handling.
    """
    model = _load_model()
    if model is None:
        return None

    try:
        import numpy as np  # noqa: PLC0415

        proba = model.predict_proba(feature_vector.reshape(1, -1))[0]
        p_tampered = float(proba[1])
        score = round(p_tampered * 100, 1)

        log.info(
            "ml_scorer_pdf: XGBoost PDF prediction complete",
            extra={"p_tampered": round(p_tampered, 4), "ml_score": score},
        )
        return {
            "score": score,
            "confidence": round(p_tampered, 4),
            "scoring_method": "ML",
        }
    except Exception as exc:
        log.warning(
            "ml_scorer_pdf: prediction failed — falling back to PDF heuristic",
            extra={"error": str(exc)},
        )
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Train  (called by scripts/train_ml_scorer_pdf.py)
# ─────────────────────────────────────────────────────────────────────────────

def _remap_raw_pdf_csv(df: "pd.DataFrame") -> "pd.DataFrame":  # type: ignore[name-defined]
    """Remap our own 66-row PDF CSV columns to the expert 18-feature schema.

    Our CSV (training_data_pdf.csv) was produced by collect_training_samples_pdf.py
    and has different column names and different scales for several metrics.
    This function bridges the two schemas so both CSVs can be concatenated.

    WHY each mapping exists:
      total_hidden_spans    → hidden_text_spans    (rename only)
      metadata_flag_count   → metadata_anomaly_score  (scale 0-5 → 0-100: *20)
      signature_coverage_gaps → signature_gap_score   (scale int count → 0-100: *10)
      total_pdf_objects     → object_count             (rename only)
      file_entropy_bits     → stream_entropy            (rename only)
      total_unique_fonts    → font_switch_score         (rename only; ranges are similar)
      has_signature_field   → has_signature             (rename only)
      xref_mismatch_score, ocr_text_layer_gap, is_scanned_pdf, hard_subtle_case
                            → added as zeros (not available in our engine output)
    """
    import pandas as pd  # noqa: PLC0415

    df = df.copy()

    # Direct renames — feature meaning is equivalent, only the column name differs.
    rename_map = {
        "total_hidden_spans":     "hidden_text_spans",
        "total_pdf_objects":      "object_count",
        "file_entropy_bits":      "stream_entropy",
        "total_unique_fonts":     "font_switch_score",
        "has_signature_field":    "has_signature",
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    # Scale metadata_flag_count (0–5 integer) to 0–100 anomaly score.
    # Each flag contributes 20 points — consistent with the expert dataset
    # where metadata_anomaly_score mean is ~22 (≈ 1 flag per genuine doc).
    if "metadata_flag_count" in df.columns:
        df["metadata_anomaly_score"] = (df["metadata_flag_count"] * 20.0).clip(0, 100)

    # Scale signature_coverage_gaps (integer count) to 0–100.
    # Expert dataset: signature_gap_score ranges 0–98, mean 16.5.
    # Our count * 10 puts 1 gap at 10, 9 gaps at 90 — reasonable calibration.
    if "signature_coverage_gaps" in df.columns:
        df["signature_gap_score"] = (df["signature_coverage_gaps"] * 10.0).clip(0, 100)

    # Add columns required by the expert schema that we cannot produce.
    # Set to 0 so they do not introduce leakage but also do not mislead the model.
    for col in ("xref_mismatch_score", "ocr_text_layer_gap", "is_scanned_pdf", "hard_subtle_case"):
        if col not in df.columns:
            df[col] = 0.0

    return df


def train_pdf(csv_paths: List[str], output_pkl: str) -> Dict[str, Any]:
    """Train a PDF XGBoost classifier and save to *output_pkl*.

    Accepts a list of CSV paths so the expert 10k dataset and our own 66-row
    supplement can be combined.  Each CSV must have a 'label' column.

    CSVs that use our raw engine column schema are detected by the presence of
    'total_hidden_spans' and are remapped automatically before concatenation.

    Returns a metrics dict.  Raises ValueError if ROC AUC < 0.80.
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
        from sklearn.ensemble import RandomForestClassifier as XGBClassifier  # noqa: PLC0415
        log.warning("ml_scorer_pdf: xgboost not found, using RandomForestClassifier")

    frames = []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        # Detect our own CSV schema by presence of 'total_hidden_spans'
        if "total_hidden_spans" in df.columns:
            df = _remap_raw_pdf_csv(df)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    log.info(
        "ml_scorer_pdf: training on combined dataset",
        extra={"rows": len(combined), "sources": len(frames)},
    )

    # Drop identity / leakage / constant columns.
    drop_present = [c for c in _PDF_DROP_COLS if c in combined.columns]
    combined.drop(columns=drop_present, inplace=True)

    # Select only the 18 model features — ignore any remaining extra columns.
    missing = [f for f in PDF_FEATURE_NAMES if f not in combined.columns]
    if missing:
        log.warning("ml_scorer_pdf: missing feature columns, filling with 0", extra={"missing": missing})
        for col in missing:
            combined[col] = 0.0

    X = combined[PDF_FEATURE_NAMES].copy().astype(float)
    y = combined["label"].astype(int)

    # Pipeline: median imputation (handles any NaN) → XGBoost.
    # Hyperparameters match the expert notebook exactly:
    # n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9.
    try:
        from xgboost import XGBClassifier as _XGB  # noqa: PLC0415
        model_step = _XGB(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42,
        )
    except ImportError:
        from sklearn.ensemble import RandomForestClassifier as _RF  # noqa: PLC0415
        model_step = _RF(n_estimators=400, class_weight="balanced", random_state=42)

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model",   model_step),
    ])

    # 5-fold stratified cross-validation — honest out-of-fold evaluation.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(
        pipe, X, y, cv=cv,
        scoring=["accuracy", "f1", "roc_auc"],
        return_train_score=False,
    )

    mean_accuracy = float(np.mean(cv_results["test_accuracy"]))
    mean_f1       = float(np.mean(cv_results["test_f1"]))
    mean_roc_auc  = float(np.mean(cv_results["test_roc_auc"]))

    log.info(
        "ml_scorer_pdf: cross-validation results",
        extra={
            "accuracy": round(mean_accuracy, 4),
            "f1": round(mean_f1, 4),
            "roc_auc": round(mean_roc_auc, 4),
        },
    )

    # Hard-case evaluation using hard_subtle_case column in the expert CSV.
    hard_case_metrics: Dict[str, Any] = {}
    for csv_path in csv_paths:
        raw = pd.read_csv(csv_path)
        if "hard_subtle_case" in raw.columns:
            hard = raw[raw["hard_subtle_case"] == 1].copy()
            if len(hard) > 0:
                if "total_hidden_spans" in hard.columns:
                    hard = _remap_raw_pdf_csv(hard)
                # Fill any still-missing expert columns with 0
                for col in PDF_FEATURE_NAMES:
                    if col not in hard.columns:
                        hard[col] = 0.0
                hard_X = hard[PDF_FEATURE_NAMES].copy().astype(float)
                hard_y = hard["label"].astype(int)
                pipe.fit(X, y)   # fit on full data, test on hard cases
                hard_pred = pipe.predict(hard_X)
                hard_case_metrics = {
                    "n_hard":   len(hard),
                    "accuracy": round(float(accuracy_score(hard_y, hard_pred)), 4),
                    "f1":       round(float(f1_score(hard_y, hard_pred, zero_division=0)), 4),
                }
                log.info("ml_scorer_pdf: hard_case evaluation", extra=hard_case_metrics)
            break

    # Quality gate — refuse to save a model that cannot outperform a coin-flip.
    if mean_roc_auc < 0.80:
        raise ValueError(
            f"Training aborted: ROC AUC {mean_roc_auc:.4f} < 0.80 minimum target. "
            "Improve the dataset before saving."
        )

    # Final fit on the entire combined dataset before serialising.
    pipe.fit(X, y)

    # Feature importance check — warn if one feature dominates (possible leakage).
    try:
        importances = pipe.named_steps["model"].feature_importances_
        max_imp = float(max(importances))
        dominant = PDF_FEATURE_NAMES[list(importances).index(max(importances))]
        if max_imp > 0.50:
            log.warning(
                "ml_scorer_pdf: dominant feature — possible leakage",
                extra={"feature": dominant, "importance": round(max_imp, 4)},
            )
        feature_importance_dict = dict(
            zip(PDF_FEATURE_NAMES, [round(float(v), 4) for v in importances])
        )
    except AttributeError:
        feature_importance_dict = {}

    # Save the trained pipeline.
    output_path = Path(output_pkl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, output_path)
    log.info("ml_scorer_pdf: model saved", extra={"path": str(output_path)})

    # Invalidate module cache so the new model is picked up immediately.
    global _MODEL_CACHE, _MODEL_LOAD_ATTEMPTED
    _MODEL_CACHE          = pipe
    _MODEL_LOAD_ATTEMPTED = True

    return {
        "rows_trained":       len(combined),
        "accuracy":           round(mean_accuracy, 4),
        "f1":                 round(mean_f1, 4),
        "roc_auc":            round(mean_roc_auc, 4),
        "hard_case":          hard_case_metrics,
        "feature_importances":feature_importance_dict,
        "model_path":         str(output_path),
    }
