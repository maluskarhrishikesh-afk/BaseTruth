"""ml_scorer_live.py — XGBoost-based fraud scoring for the Face Scan Live pipeline.

This module is an optional drop-in replacement for the fixed-weight heuristic
in build_live_face_scan_result().  It works as follows:

  1. build_feature_vector(checks_dict, session)
       Converts the completed result checks dict into a 24-feature float array
       that matches the schema of fraud_model/data/training_data_face_scan_live.csv.

  2. predict(feature_vector)
       Loads fraud_model/models/ml_scorer_face_scan_live.pkl and returns the spoof
       probability as a 0-100 risk score.
       Returns None if the model file does not exist — the caller falls back to the
       heuristic automatically.

  3. train(csv_path, output_pkl, progress_cb)
       Trains an XGBoost binary classifier (genuine=0, spoof=1) on the labeled
       CSV, runs 5-fold stratified CV, and saves the model only if ROC AUC >= 0.75.
       The lower bar (vs. 0.80 for image scoring) reflects that live-session data
       is inherently noisier than static forensic signals.

  4. explain(feature_vector)
       Returns per-feature SHAP contributions using XGBoost's built-in tree SHAP —
       no external shap package required.

Cold-start guarantee: if ml_scorer_face_scan_live.pkl is absent, every call to
predict() returns None and the existing heuristic formula runs unchanged.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from basetruth.logger import get_logger

log = get_logger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
# Resolved from this file: src/basetruth/face_scan/ml_scorer_live.py → repo root
_REPO_ROOT  = Path(__file__).resolve().parent.parent.parent.parent
_MODEL_PATH = _REPO_ROOT / "fraud_model" / "models" / "ml_scorer_face_scan_live.pkl"
_CSV_PATH   = _REPO_ROOT / "fraud_model" / "data"   / "training_data_face_scan_live.csv"

# Module-level model cache — loaded once per process, not once per session.
_MODEL_CACHE: Any = None
_MODEL_LOAD_ATTEMPTED: bool = False

# ── Binary verdict labels ────────────────────────────────────────────────────
# 0 = GENUINE   — real live person, challenges completed legitimately
# 1 = SPOOF     — replay attack, screen recording, virtual camera injection,
#                 or other non-live source
ML_VERDICT_LABELS: Dict[int, str] = {
    0: "GENUINE",
    1: "SPOOF",
}

# ── Feature schema — 24 tabular signals ─────────────────────────────────────
# All are already computed by build_live_face_scan_result() today, so feature
# engineering is free — we just need to package them in order.
FEATURE_NAMES: List[str] = [
    # Temporal consistency (how jerky/smooth head movement was)
    "yaw_jerk",
    "pitch_jerk",
    "nose_jitter",
    "temporal_consistency_score",
    # Replay heuristics (repeated-frame hash and brightness flicker)
    "repeat_frame_score",
    "flicker_score",
    "brightness_instability",
    # Eye micro-jitter (involuntary saccades — frozen eyes = suspicious)
    "mean_eye_jitter",
    # 3D depth (IOD vs yaw correlation — flat photo/screen = near 0)
    "iod_yaw_correlation",
    # Screen frequency (FFT moiré grid — filmed screen = elevated)
    "mean_fft_grid_peak",
    # Frame delivery timing (metronomic = replay tool)
    "interval_cv",
    # Session metadata
    "observed_fps",
    "frame_drop_rate",
    # Face quality
    "mean_face_area_ratio",
    "blur_risk_0_100",
    "brightness_risk_0_100",
    # Active liveness signals
    "wrong_action_count",
    "challenge_count",
    # Face tracking reliability
    "frames_without_face",
    # Device flag
    "virtual_camera_suspected",
    # ── Tier 1 ML signals (added with 24-feature schema) ─────────────────────
    # Variance of frame-to-frame yaw velocity (high = natural human; low = replay)
    "head_velocity_variance",
    # Mean blink duration from EAR during blink challenge (NaN if not applicable)
    "blink_duration_ms",
    # Mean time from challenge prompt to completion (NaN until ≥2 challenges timed)
    "challenge_reaction_latency_ms",
    # Mean face landmark confidence proxy across all history frames
    "mean_landmark_confidence",
]


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_vector(checks: Dict[str, Any], environment: Dict[str, Any]) -> "np.ndarray":  # type: ignore[name-defined]
    """Convert the result checks dict and environment into a float32 feature array.

    The checks dict is the 'checks' key from build_live_face_scan_result().
    The environment dict is session.environment.
    Both are already present in the final result payload, so no re-computation
    is needed — we are just extracting and ordering values.

    Missing keys fall back to safe neutral defaults so the imputer handles them.
    NaN is used for truly absent optional signals (e.g. virtual_camera_suspected
    when the browser did not send device metadata) so the sklearn SimpleImputer
    fills them with the training-set median rather than a hard-coded zero.
    Produces a 24-element float array matching FEATURE_NAMES.
    """
    import numpy as np  # noqa: PLC0415

    temporal = checks.get("temporal_consistency", {})
    replay   = checks.get("replay_heuristics", {})
    saccade  = checks.get("saccade_analysis", {})
    depth    = checks.get("depth_consistency", {})
    screen   = checks.get("screen_frequency", {})
    timing   = checks.get("frame_timing", {})
    quality  = checks.get("quality_assessment", {})
    liveness = checks.get("active_liveness", {})
    face_det = checks.get("face_detection", {})

    # virtual_camera_suspected: 1 if flagged, 0 if clean, NaN if unknown.
    # We use NaN rather than 0 so the imputer treats an absent metadata send
    # differently from a confirmed non-virtual camera — the imputer will fill
    # NaN with the training median rather than assuming clean.
    vc_raw = environment.get("virtual_camera_suspected")
    virtual_camera: float = float(vc_raw) if vc_raw is not None else float("nan")

    _nan = float("nan")

    # Helper to safely convert a value that may be None to float.
    # We cannot use float(None) — it raises TypeError — so we fall back to NaN
    # (which the SimpleImputer will fill with the training median).
    def _f(v: Any, default: float = _nan) -> float:
        return float(v) if v is not None else default

    values: List[float] = [
        _f(temporal.get("yaw_jerk"),              0.0),
        _f(temporal.get("pitch_jerk"),            0.0),
        _f(temporal.get("nose_jitter"),           0.0),
        _f(temporal.get("score_0_100"),           0.0),
        _f(replay.get("repeat_frame_score"),      0.0),
        _f(replay.get("flicker_score"),           0.0),
        _f(replay.get("brightness_instability"),  0.0),
        _f(saccade.get("mean_eye_jitter")),
        _f(depth.get("iod_yaw_correlation")),
        _f(screen.get("mean_fft_grid_peak")),
        _f(timing.get("interval_cv")),
        _f(environment.get("observed_fps")),
        _f(environment.get("frame_drop_rate")),
        _f(quality.get("mean_face_area_ratio")),
        _f(quality.get("blur_risk_0_100"),        0.0),
        _f(quality.get("brightness_risk_0_100"),  0.0),
        _f(liveness.get("wrong_action_count"),    0.0),
        _f(liveness.get("challenge_count"),       0.0),
        _f(face_det.get("frames_without_face"),   0.0),
        virtual_camera,
        # Tier 1 signals — NaN when absent so SimpleImputer uses training median
        _f(temporal.get("head_velocity_variance")),
        _f(liveness.get("blink_duration_ms")),
        _f(liveness.get("challenge_reaction_latency_ms")),
        _f(face_det.get("mean_landmark_confidence")),
    ]

    log.debug(
        "ml_scorer_live: feature vector built",
        extra=dict(zip(FEATURE_NAMES, [round(v, 4) if v == v else None for v in values])),
    )
    return __import__("numpy").array(values, dtype=float)  # type: ignore[return-value]


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Load model
# ─────────────────────────────────────────────────────────────────────────────

def _load_model() -> Any:
    """Load (and cache) the XGBoost model from disk.

    Returns None when the model file does not exist — this is the expected
    cold-start state before any training data has been collected and the model
    has been trained for the first time.  The caller falls back to heuristics.
    """
    global _MODEL_CACHE, _MODEL_LOAD_ATTEMPTED
    if _MODEL_LOAD_ATTEMPTED:
        return _MODEL_CACHE

    _MODEL_LOAD_ATTEMPTED = True
    if not _MODEL_PATH.exists():
        log.info(
            "ml_scorer_live: model not found — heuristic fallback active",
            extra={"checked_path": str(_MODEL_PATH)},
        )
        return None

    try:
        import joblib  # noqa: PLC0415
        _MODEL_CACHE = joblib.load(_MODEL_PATH)
        log.info(
            "ml_scorer_live: XGBoost model loaded",
            extra={"path": str(_MODEL_PATH)},
        )
        return _MODEL_CACHE
    except Exception as exc:
        log.warning(
            "ml_scorer_live: model load failed — heuristic fallback active",
            extra={"path": str(_MODEL_PATH), "error": str(exc)},
        )
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Predict
# ─────────────────────────────────────────────────────────────────────────────

def predict(feature_vector: "np.ndarray") -> Optional[Dict[str, Any]]:  # type: ignore[name-defined]
    """Score a feature vector using the trained binary XGBoost model.

    Returns a dict:
        score          float (0–100) — P(SPOOF) × 100
        scoring_method str           — "ML"
        ml_verdict     str           — "GENUINE" or "SPOOF"

    Returns None if the model is unavailable — caller continues with heuristics.
    """
    model = _load_model()
    if model is None:
        return None

    try:
        import numpy as np  # noqa: PLC0415

        # Reshape for sklearn pipeline and replace sentinels with NaN.
        vec = feature_vector.copy().astype(float).reshape(1, -1)
        vec[vec == -1.0] = float("nan")

        # predict_proba shape: (1, 2) — [[P(genuine), P(spoof)]]
        proba     = model.predict_proba(vec)[0]
        p_spoof   = float(proba[1])
        ml_class  = int(np.argmax(proba))
        ml_verdict = ML_VERDICT_LABELS.get(ml_class, "SPOOF")

        score = round(p_spoof * 100, 1)
        log.info(
            "ml_scorer_live: XGBoost prediction complete",
            extra={"p_spoof": round(p_spoof, 4), "ml_score": score, "ml_verdict": ml_verdict},
        )
        return {
            "score": score,
            "scoring_method": "ML",
            "ml_verdict": ml_verdict,
        }
    except Exception as exc:
        log.warning(
            "ml_scorer_live: prediction failed — heuristic fallback",
            extra={"error": str(exc)},
        )
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — SHAP explanations (tree SHAP built into XGBoost)
# ─────────────────────────────────────────────────────────────────────────────

def explain(feature_vector: "np.ndarray") -> Optional[Dict[str, float]]:  # type: ignore[name-defined]
    """Compute per-feature SHAP contribution values for a single live session.

    Uses XGBoost's built-in tree SHAP — no external shap package needed.
    Positive values push the prediction toward SPOOF; negative toward GENUINE.

    Returns {feature_name: shap_value} with 20 entries, or None if the model
    is unavailable or the computation raises any error.
    """
    model = _load_model()
    if model is None:
        return None

    try:
        import numpy as np  # noqa: PLC0415
        import xgboost as xgb  # noqa: PLC0415

        vec = feature_vector.copy().astype(float).reshape(1, -1)
        vec[vec == -1.0] = float("nan")

        # Apply the imputer step so NaN values are filled with training medians.
        imputer = model.named_steps["imputer"]
        imputed = imputer.transform(vec)

        booster = model.named_steps["model"].get_booster()

        # Guard against feature count mismatch if new features were added after
        # the last training run — trim to booster's actual feature count.
        n_booster = booster.num_features()
        imputed_trimmed = imputed[:, :n_booster]
        active_names = FEATURE_NAMES[:n_booster]

        dmat = xgb.DMatrix(imputed_trimmed, feature_names=active_names)

        # pred_contribs=True returns shape (n_samples, n_features + 1).
        # Last column is the bias term — drop it.
        shap_matrix = booster.predict(dmat, pred_contribs=True)
        shap_values = shap_matrix[0, :-1]

        contributions = {
            name: round(float(v), 4)
            for name, v in zip(active_names, shap_values)
        }
        log.debug("ml_scorer_live: SHAP computed", extra={"n_features": len(contributions)})
        return contributions

    except Exception as exc:
        log.warning("ml_scorer_live: explain failed", extra={"error": str(exc)})
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Auto-collect training data
# ─────────────────────────────────────────────────────────────────────────────

# All columns written to the CSV — mirrors collect_live_scan_samples.py.
_CSV_COLUMNS: List[str] = ["session_id", "timestamp_utc", "verdict"] + FEATURE_NAMES + ["label"]

# Number of metadata columns that precede the features in _CSV_COLUMNS.
# (session_id, timestamp_utc, verdict)
_CSV_META_COLS = 3


def migrate_csv_if_stale(csv_path: Optional[Path] = None) -> bool:
    """Repair a training CSV whose header no longer matches _CSV_COLUMNS.

    When new features are added to FEATURE_NAMES the header written at file-
    creation time becomes stale. New rows carry more columns than the header
    declares, which mis-aligns the 'label' column and breaks pandas.read_csv.

    How the migration works:
    - Reads every row with csv.reader (raw, position-based — not DictReader,
      which silently drops values that overflow the header).
    - Old rows (len == old column count): inserts empty strings for each new
      feature column right before the final 'label' value.
    - New rows (len == current column count): kept as-is.
    - Any other length: kept as-is with a warning (could be a hand-edited row).
    - Rewrites the file atomically using a temp file then rename.

    Returns True if the file was migrated, False if it was already up-to-date
    or did not exist yet.
    """
    import csv as _csv  # noqa: PLC0415
    import tempfile  # noqa: PLC0415
    import os  # noqa: PLC0415

    target = csv_path or _CSV_PATH
    if not target.exists() or target.stat().st_size == 0:
        return False  # nothing to migrate

    expected_header = _CSV_COLUMNS  # what the header should be
    expected_len    = len(expected_header)  # target row length (includes header)

    with target.open("r", newline="", encoding="utf-8") as fh:
        reader = _csv.reader(fh)
        rows   = list(reader)

    if not rows:
        return False

    actual_header = rows[0]
    if actual_header == expected_header:
        # Header already matches — nothing to do.
        return False

    old_len = len(actual_header)
    # Number of new feature columns that must be back-filled into old rows.
    # They sit between the last known feature and the label column.
    new_feature_count = expected_len - old_len  # e.g. 28 - 24 = 4

    if new_feature_count <= 0:
        # Header has MORE columns than expected — schema regressed, do not touch.
        log.warning(
            "ml_scorer_live: CSV header wider than _CSV_COLUMNS — manual review needed",
            extra={"csv_path": str(target), "header_len": old_len, "expected_len": expected_len},
        )
        return False

    log.info(
        "ml_scorer_live: migrating stale CSV header",
        extra={
            "csv_path": str(target),
            "old_col_count": old_len,
            "new_col_count": expected_len,
            "new_feature_count": new_feature_count,
        },
    )

    migrated_rows: List[List[str]] = [expected_header]  # write the correct header first

    for raw_row in rows[1:]:  # skip the old header
        if len(raw_row) == old_len:
            # Old row: splice in empty strings before the final 'label' value.
            # Position of label in old row = old_len - 1
            label_value = raw_row[-1]
            features    = raw_row[:-1]
            new_row     = features + [""] * new_feature_count + [label_value]
        elif len(raw_row) == expected_len:
            # Already the new length — left by a new session after the feature
            # was added but before the header was fixed.
            new_row = raw_row
        else:
            # Unexpected length — preserve as-is and log so we can investigate.
            log.warning(
                "ml_scorer_live: row with unexpected column count kept unchanged",
                extra={"row_len": len(raw_row), "expected": expected_len, "old": old_len},
            )
            new_row = raw_row
        migrated_rows.append(new_row)

    # Write to a temp file in the same directory then atomically rename.
    # This ensures the CSV is never left in a half-written state if the process
    # is interrupted mid-write.
    # shutil.move is used instead of os.replace because on Windows, os.replace
    # can raise PermissionError when the file is in a directory that has been
    # recently scanned by antivirus or the OS file cache hasn't released the
    # destination handle yet.
    import shutil  # noqa: PLC0415
    tmp_fd, tmp_path = tempfile.mkstemp(dir=target.parent, suffix=".csv.tmp")
    try:
        with os.fdopen(tmp_fd, "w", newline="", encoding="utf-8") as tmp_fh:
            writer = _csv.writer(tmp_fh)
            writer.writerows(migrated_rows)
        # shutil.move handles Windows file-locking edge cases better than os.replace
        shutil.move(tmp_path, str(target))
    except Exception:  # noqa: BLE001
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    log.info(
        "ml_scorer_live: CSV migration complete",
        extra={"csv_path": str(target), "rows_migrated": len(migrated_rows) - 1},
    )
    return True


def append_training_sample(result: Dict[str, Any], label: int = -1) -> None:
    """Append one row to the training CSV from a completed live-scan result dict.

    Called automatically by the API WebSocket handler immediately after
    build_live_face_scan_result() returns.  Every completed scan is recorded
    with label=-1 (unlabeled) unless the caller passes a known label.  Users
    then open the CSV in a spreadsheet, fill in the label column (0=GENUINE,
    1=SPOOF), and train the model from the ML Training Pipeline screen.

    The function is idempotent per session_id: if the CSV already contains a
    row with the same decision_trace_id the write is skipped so re-connections
    or replayed WebSocket results don't create duplicate rows.

    Errors are caught and logged rather than raised so a disk/permissions issue
    never interrupts the scan result being delivered to the user.
    """
    import csv  # noqa: PLC0415

    checks      = result.get("checks", {})
    environment = result.get("environment", {})
    trace       = result.get("trace", {})

    # Build the feature vector for the current schema (24 features).
    vec = build_feature_vector(checks, environment)

    session_id    = trace.get("decision_trace_id", result.get("filename", "unknown"))
    timestamp_utc = trace.get("timestamp_utc", "")
    verdict       = result.get("verdict", "")

    # Repair the CSV header if it was created under an older FEATURE_NAMES schema.
    # This runs on every call but is a no-op when the header already matches.
    try:
        migrate_csv_if_stale()
    except Exception as exc:  # noqa: BLE001
        log.warning("ml_scorer_live: CSV migration failed — proceeding anyway", extra={"error": str(exc)})

    # Check for existing session_id to stay idempotent without loading pandas.
    try:
        if _CSV_PATH.exists():
            with _CSV_PATH.open("r", newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for existing_row in reader:
                    if existing_row.get("session_id") == session_id:
                        log.debug(
                            "ml_scorer_live: sample already in CSV, skipping",
                            extra={"session_id": session_id},
                        )
                        return
    except Exception as exc:  # noqa: BLE001
        log.warning("ml_scorer_live: could not check CSV for duplicates", extra={"error": str(exc)})

    # Build the row dict — NaN stored as empty string so spreadsheets and
    # pandas both interpret it as a missing value on reload.
    row: Dict[str, Any] = {
        "session_id":    session_id,
        "timestamp_utc": timestamp_utc,
        "verdict":       verdict,
    }
    for name, value in zip(FEATURE_NAMES, vec):
        row[name] = "" if (value != value) else round(float(value), 6)  # NaN check: NaN != NaN
    row["label"] = label

    try:
        # Create parent directories if they don't exist yet (first-run safety).
        _CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

        # Append mode: write the header only when the file is new or empty.
        write_header = not _CSV_PATH.exists() or _CSV_PATH.stat().st_size == 0
        with _CSV_PATH.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=_CSV_COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        log.info(
            "ml_scorer_live: training sample appended",
            extra={"session_id": session_id, "verdict": verdict, "label": label},
        )
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "ml_scorer_live: failed to write training sample — scan result unaffected",
            extra={"csv_path": str(_CSV_PATH), "error": str(exc)},
        )


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — Train  (called by the ML Training Pipeline page via API WebSocket)
# ─────────────────────────────────────────────────────────────────────────────

# Columns to drop before training — not forensic signals.
_DROP_COLS = {
    "session_id",
    "filename",
    "timestamp_utc",
    "verdict",          # derived from risk score — would be target leakage
    "risk_score_0_100", # same reason
    "confidence_0_100", # derived — would be target leakage
    "scoring_method",   # metadata
    "narrative",        # text — not a numeric signal
}


def train(
    csv_path: str,
    output_pkl: str,
    progress_cb: Any = None,
) -> Dict[str, Any]:
    """Train an XGBoost binary classifier and save to *output_pkl*.

    Expects a CSV at csv_path with at least:
      - one column per FEATURE_NAMES entry (numeric)
      - a 'label' column  (0 = genuine, 1 = spoof)

    progress_cb is an optional callable(step: str, pct: int) that receives
    human-readable step descriptions and a 0–100 completion percentage.

    Returns a metrics dict.  Raises ValueError if ROC AUC < 0.75 (prevents
    saving a model that is worse than random chance on sparse early data).
    """

    def _emit(step: str, pct: int) -> None:
        """Fire the optional progress callback — swallow exceptions."""
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
        from sklearn.ensemble import RandomForestClassifier as XGBClassifier  # noqa: PLC0415
        log.warning("ml_scorer_live: xgboost not found, using RandomForestClassifier")

    _emit("Loading training data...", 3)

    df = pd.read_csv(csv_path)
    log.info("ml_scorer_live: loaded CSV", extra={"rows": len(df), "columns": list(df.columns)})

    if "label" not in df.columns:
        raise ValueError(f"Training CSV must have a 'label' column. Found: {list(df.columns)}")

    # Drop non-feature columns and keep only known feature columns that exist.
    drop = _DROP_COLS & set(df.columns)
    df = df.drop(columns=list(drop))

    # Select only the feature columns we know about — ignore any extra columns
    # added by the collection script that are not yet in FEATURE_NAMES.
    available_features = [f for f in FEATURE_NAMES if f in df.columns]
    missing_features   = [f for f in FEATURE_NAMES if f not in df.columns]
    if missing_features:
        log.warning(
            "ml_scorer_live: some features absent from CSV — will be imputed",
            extra={"missing": missing_features},
        )

    X = df[available_features].values.astype(float)
    y = df["label"].values.astype(int)

    _emit(f"Loaded {len(df)} rows, {len(available_features)} features.", 8)

    n_genuine = int((y == 0).sum())
    n_spoof   = int((y == 1).sum())
    log.info("ml_scorer_live: class distribution", extra={"genuine": n_genuine, "spoof": n_spoof})

    if n_genuine < 5 or n_spoof < 5:
        raise ValueError(
            f"Need at least 5 samples per class. Got genuine={n_genuine}, spoof={n_spoof}. "
            "Collect more labeled sessions before training."
        )

    _emit("Building XGBoost pipeline...", 12)

    # scale_pos_weight balances the class imbalance that is common during early
    # data collection (more genuine sessions than spoof sessions).
    scale_pos = n_genuine / max(n_spoof, 1)

    pipe = Pipeline([
        # SimpleImputer fills NaN values (absent optional signals) with the training
        # median so the model never receives NaN at prediction time.
        ("imputer", SimpleImputer(strategy="median")),
        ("model",   XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )),
    ])

    _emit("Running 5-fold cross-validation...", 18)

    # 5-fold stratified CV gives honest estimates even with small datasets.
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_auc: List[float] = []
    fold_acc: List[float] = []
    fold_f1:  List[float] = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        fold_pipe = _clone_pipe(pipe)
        fold_pipe.fit(X_tr, y_tr)

        y_prob = fold_pipe.predict_proba(X_val)[:, 1]
        y_pred = fold_pipe.predict(X_val)

        fold_auc.append(float(roc_auc_score(y_val, y_prob)))
        fold_acc.append(float(accuracy_score(y_val, y_pred)))
        fold_f1.append(float(f1_score(y_val, y_pred, zero_division=0)))

        pct = 18 + int(fold_idx / 5 * 50)
        _emit(f"Fold {fold_idx}/5 — AUC {fold_auc[-1]:.3f}", pct)

    mean_auc = float(np.mean(fold_auc))
    mean_acc = float(np.mean(fold_acc))
    mean_f1  = float(np.mean(fold_f1))

    log.info(
        "ml_scorer_live: CV complete",
        extra={"roc_auc": round(mean_auc, 4), "accuracy": round(mean_acc, 4), "f1": round(mean_f1, 4)},
    )

    _emit(f"CV AUC {mean_auc:.3f} | Acc {mean_acc:.3f} | F1 {mean_f1:.3f}", 72)

    # Minimum ROC AUC guard — prevents saving a model with no discriminative power.
    # 0.75 is lower than the image scorer (0.80) because live-session data is noisier
    # and datasets are small in the early collection phase.
    if mean_auc < 0.75:
        raise ValueError(
            f"ROC AUC {mean_auc:.3f} is below the minimum threshold of 0.75. "
            "Collect more labeled samples or review feature quality before retraining."
        )

    _emit("Training final model on full dataset...", 78)

    # Train the final model on the full dataset (not just one fold).
    pipe.fit(X, y)

    _emit("Saving model...", 92)

    out_path = Path(output_pkl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out_path)

    # Reset the module-level cache so the freshly trained model is loaded
    # on the next prediction call without restarting the process.
    global _MODEL_CACHE, _MODEL_LOAD_ATTEMPTED
    _MODEL_CACHE = pipe
    _MODEL_LOAD_ATTEMPTED = True

    log.info(
        "ml_scorer_live: model saved",
        extra={"path": str(out_path), "roc_auc": round(mean_auc, 4)},
    )

    metrics = {
        "roc_auc":  round(mean_auc, 4),
        "accuracy": round(mean_acc, 4),
        "f1":       round(mean_f1, 4),
        "n_samples":        len(df),
        "n_genuine":        n_genuine,
        "n_spoof":          n_spoof,
        "n_features_used":  len(available_features),
        "features_used":    available_features,
    }
    _emit(f"Done — ROC AUC {mean_auc:.3f}", 100)
    return metrics
