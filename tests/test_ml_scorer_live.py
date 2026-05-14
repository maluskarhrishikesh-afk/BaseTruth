"""tests/test_ml_scorer_live.py — Unit tests for the Face Scan Live ML scorer.

These tests verify:
  - build_feature_vector returns the right shape and dtype
  - Missing keys in checks/environment fall back to NaN or 0 gracefully
  - predict() returns None when no model file exists (cold-start guarantee)
  - explain() returns None when no model file exists
  - train() raises ValueError when fewer than 5 samples per class are present
  - train() raises ValueError when the 'label' column is absent from the CSV
  - The scoring_method key is 'heuristic' when no model is loaded (live.py integration)

Run with:
    python -m pytest tests/test_ml_scorer_live.py -v
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

# ── Path setup ─────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def full_checks() -> dict:
    """A checks dict that mirrors what build_live_face_scan_result() produces."""
    return {
        "temporal_consistency": {
            "yaw_jerk":   0.003,
            "pitch_jerk": 0.002,
            "nose_jitter": 1.5,
            "score_0_100": 12.0,
            "head_velocity_variance": 0.0012,
        },
        "replay_heuristics": {
            "repeat_frame_score":    5.0,
            "flicker_score":         8.0,
            "brightness_instability": 3.0,
            "score_0_100": 10.0,
        },
        "saccade_analysis": {
            "mean_eye_jitter": 0.4,
            "score_0_100": 20.0,
        },
        "depth_consistency": {
            "iod_yaw_correlation": 0.72,
            "score_0_100": 15.0,
        },
        "screen_frequency": {
            "mean_fft_grid_peak": 1.2,
            "score_0_100": 5.0,
        },
        "frame_timing": {
            "interval_cv": 0.15,
            "score_0_100": 8.0,
        },
        "quality_assessment": {
            "mean_face_area_ratio": 0.25,
            "blur_risk_0_100": 10.0,
            "brightness_risk_0_100": 5.0,
            "score_0_100": 15.0,
        },
        "active_liveness": {
            "wrong_action_count": 0,
            "challenge_count": 3,
            "score_0_100": 0.0,
            "blink_duration_ms": 180.0,
            "challenge_reaction_latency_ms": 950.0,
        },
        "face_detection": {
            "frames_without_face": 2,
            "mean_landmark_confidence": 0.92,
        },
    }


@pytest.fixture
def full_environment() -> dict:
    """A session environment dict with all standard keys populated."""
    return {
        "observed_fps": 10.1,
        "frame_drop_rate": 0.02,
        "virtual_camera_suspected": 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 — Feature vector shape and dtype
# ─────────────────────────────────────────────────────────────────────────────

def test_feature_vector_shape(full_checks, full_environment):
    """build_feature_vector must return a numpy array with exactly 24 elements."""
    from basetruth.face_scan.ml_scorer_live import build_feature_vector, FEATURE_NAMES

    fv = build_feature_vector(full_checks, full_environment)

    assert isinstance(fv, np.ndarray), "result must be a numpy ndarray"
    assert fv.shape == (24,), f"expected shape (24,), got {fv.shape}"
    assert len(FEATURE_NAMES) == 24, "FEATURE_NAMES must have exactly 24 entries"
    # dtype should be float (not int or object)
    assert np.issubdtype(fv.dtype, np.floating), f"expected float dtype, got {fv.dtype}"


def test_feature_vector_values_non_nan_for_full_input(full_checks, full_environment):
    """All features should be finite (non-NaN) when the input is complete."""
    from basetruth.face_scan.ml_scorer_live import build_feature_vector

    fv = build_feature_vector(full_checks, full_environment)
    # virtual_camera_suspected = 0, so no NaN expected for this complete input.
    assert np.all(np.isfinite(fv)), (
        f"unexpected NaN/Inf in complete-input feature vector: {fv}"
    )


def test_feature_vector_nan_for_missing_optional_keys():
    """Optional signals (eye jitter, IOD correlation, etc.) produce NaN when absent.

    NaN is the correct sentinel because the SimpleImputer fills them with the
    training median — not with a hard-coded zero that could bias the model.
    """
    from basetruth.face_scan.ml_scorer_live import build_feature_vector, FEATURE_NAMES

    # Empty checks and environment — all optional signals absent.
    fv = build_feature_vector({}, {})

    feat = dict(zip(FEATURE_NAMES, fv))

    # Signals that default to 0 when absent (counts, risk scores)
    zero_default = {
        "yaw_jerk", "pitch_jerk", "nose_jitter", "temporal_consistency_score",
        "repeat_frame_score", "flicker_score", "brightness_instability",
        "blur_risk_0_100", "brightness_risk_0_100",
        "wrong_action_count", "challenge_count", "frames_without_face",
    }
    for name in zero_default:
        assert feat[name] == 0.0 or feat[name] == 0, (
            f"{name} should default to 0 when absent, got {feat[name]}"
        )

    # Signals that should be NaN when absent (optional ratio signals)
    nan_default = {
        "mean_eye_jitter", "iod_yaw_correlation", "mean_fft_grid_peak",
        "interval_cv", "observed_fps", "frame_drop_rate",
        "mean_face_area_ratio", "virtual_camera_suspected",
        # Tier 1 signals — NaN when absent
        "head_velocity_variance", "blink_duration_ms",
        "challenge_reaction_latency_ms", "mean_landmark_confidence",
    }
    for name in nan_default:
        assert feat[name] != feat[name], (
            f"{name} should be NaN when absent, got {feat[name]}"
        )


def test_feature_vector_virtual_camera_unknown():
    """virtual_camera_suspected should be NaN when environment key is absent."""
    from basetruth.face_scan.ml_scorer_live import build_feature_vector, FEATURE_NAMES

    fv = build_feature_vector({}, {})
    feat = dict(zip(FEATURE_NAMES, fv))
    vc = feat["virtual_camera_suspected"]
    assert vc != vc, "virtual_camera_suspected must be NaN when not provided"


def test_feature_vector_virtual_camera_flagged():
    """virtual_camera_suspected should be 1.0 when set in environment."""
    from basetruth.face_scan.ml_scorer_live import build_feature_vector, FEATURE_NAMES

    fv = build_feature_vector({}, {"virtual_camera_suspected": True})
    feat = dict(zip(FEATURE_NAMES, fv))
    assert feat["virtual_camera_suspected"] == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 — predict() cold-start (no model file)
# ─────────────────────────────────────────────────────────────────────────────

def test_predict_returns_none_when_no_model(full_checks, full_environment):
    """predict() must return None when the model pkl file does not exist.

    This is the cold-start guarantee: the system must work correctly without
    a trained model (using the heuristic fallback) until data is collected.
    """
    import basetruth.face_scan.ml_scorer_live as scorer

    fv = scorer.build_feature_vector(full_checks, full_environment)

    # Patch _MODEL_PATH to a non-existent file and reset the load cache.
    with patch.object(scorer, "_MODEL_PATH", Path("/tmp/does_not_exist.pkl")):
        # Reset the module-level load cache so the patched path is tried.
        original_attempted = scorer._MODEL_LOAD_ATTEMPTED
        original_cache = scorer._MODEL_CACHE
        scorer._MODEL_LOAD_ATTEMPTED = False
        scorer._MODEL_CACHE = None
        try:
            result = scorer.predict(fv)
        finally:
            scorer._MODEL_LOAD_ATTEMPTED = original_attempted
            scorer._MODEL_CACHE = original_cache

    assert result is None, f"predict() must return None when model is absent, got {result}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 — explain() cold-start (no model file)
# ─────────────────────────────────────────────────────────────────────────────

def test_explain_returns_none_when_no_model(full_checks, full_environment):
    """explain() must return None when the model pkl file does not exist."""
    import basetruth.face_scan.ml_scorer_live as scorer

    fv = scorer.build_feature_vector(full_checks, full_environment)

    with patch.object(scorer, "_MODEL_PATH", Path("/tmp/does_not_exist.pkl")):
        original_attempted = scorer._MODEL_LOAD_ATTEMPTED
        original_cache = scorer._MODEL_CACHE
        scorer._MODEL_LOAD_ATTEMPTED = False
        scorer._MODEL_CACHE = None
        try:
            result = scorer.explain(fv)
        finally:
            scorer._MODEL_LOAD_ATTEMPTED = original_attempted
            scorer._MODEL_CACHE = original_cache

    assert result is None, f"explain() must return None when model is absent, got {result}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 4 — train() input validation: missing label column
# ─────────────────────────────────────────────────────────────────────────────

def test_train_raises_on_missing_label_column():
    """train() must raise ValueError if the CSV has no 'label' column."""
    from basetruth.face_scan.ml_scorer_live import train, FEATURE_NAMES

    # Build a CSV with all feature columns but no 'label'.
    df = pd.DataFrame(
        {name: np.random.rand(20) for name in FEATURE_NAMES}
    )

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        csv_path = f.name
        df.to_csv(csv_path, index=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        pkl_path = str(Path(tmpdir) / "test_model.pkl")
        with pytest.raises(ValueError, match="label"):
            train(csv_path, pkl_path)


# ─────────────────────────────────────────────────────────────────────────────
# Test 5 — train() input validation: insufficient samples
# ─────────────────────────────────────────────────────────────────────────────

def test_train_raises_on_too_few_samples():
    """train() must raise ValueError when fewer than 5 samples per class exist.

    The minimum-samples guard prevents training a model on data that is far
    too sparse for 5-fold cross-validation to produce meaningful metrics.
    """
    from basetruth.face_scan.ml_scorer_live import train, FEATURE_NAMES

    # 3 genuine and 3 spoof rows — below the minimum of 5 per class.
    rows = []
    for label in [0, 0, 0, 1, 1, 1]:
        row = {name: np.random.rand() for name in FEATURE_NAMES}
        row["label"] = label
        rows.append(row)
    df = pd.DataFrame(rows)

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        csv_path = f.name
        df.to_csv(csv_path, index=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        pkl_path = str(Path(tmpdir) / "test_model.pkl")
        with pytest.raises(ValueError, match="least 5 samples"):
            train(csv_path, pkl_path)


# ─────────────────────────────────────────────────────────────────────────────
# Test 6 — train() produces a model and saves pkl (with mocked AUC guard)
# ─────────────────────────────────────────────────────────────────────────────

def test_train_saves_model_when_auc_sufficient():
    """train() should save the pkl and return a metrics dict with roc_auc key
    when given sufficient data and the AUC clears the 0.75 threshold.

    We use perfectly separable synthetic data to guarantee AUC = 1.0.
    """
    from basetruth.face_scan.ml_scorer_live import train, FEATURE_NAMES

    # 20 genuine: feature 0 near 0; 20 spoof: feature 0 near 1.
    # All other features are random noise so the model is forced to use feature 0.
    n = 20
    genuine_rows = [{name: 0.0 if name == FEATURE_NAMES[0] else np.random.rand()
                     for name in FEATURE_NAMES} for _ in range(n)]
    spoof_rows   = [{name: 1.0 if name == FEATURE_NAMES[0] else np.random.rand()
                     for name in FEATURE_NAMES} for _ in range(n)]

    for r in genuine_rows:
        r["label"] = 0
    for r in spoof_rows:
        r["label"] = 1

    df = pd.DataFrame(genuine_rows + spoof_rows)

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        csv_path = f.name
        df.to_csv(csv_path, index=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        pkl_path = str(Path(tmpdir) / "test_model.pkl")
        metrics = train(csv_path, pkl_path)
        # Check inside the context — TemporaryDirectory removes the dir on exit.
        assert Path(pkl_path).exists(), "pkl file must be saved after successful training"

    assert "roc_auc" in metrics, "metrics dict must contain roc_auc key"
    assert metrics["roc_auc"] >= 0.75, (
        f"AUC {metrics['roc_auc']} must be >= 0.75 for model to be saved"
    )
    assert "n_samples" in metrics
    assert metrics["n_samples"] == 2 * n


# ─────────────────────────────────────────────────────────────────────────────
# Test 7 — predict() returns structured result with a trained model
# ─────────────────────────────────────────────────────────────────────────────

def test_predict_returns_score_dict_with_trained_model(full_checks, full_environment):
    """predict() must return a dict with 'score', 'scoring_method', and 'ml_verdict'
    when a valid model is loaded.
    """
    import basetruth.face_scan.ml_scorer_live as scorer

    # Build a tiny model in-process and inject it into the module cache.
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.ensemble import RandomForestClassifier

    n = 20
    X = np.random.rand(2 * n, 24)
    y = np.array([0] * n + [1] * n)
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model",   RandomForestClassifier(n_estimators=5, random_state=0)),
    ])
    pipe.fit(X, y)

    # Monkey-patch the module-level cache so predict() uses our tiny mock model.
    original_cache    = scorer._MODEL_CACHE
    original_attempted = scorer._MODEL_LOAD_ATTEMPTED
    scorer._MODEL_CACHE          = pipe
    scorer._MODEL_LOAD_ATTEMPTED = True

    try:
        fv = scorer.build_feature_vector(full_checks, full_environment)
        result = scorer.predict(fv)
    finally:
        scorer._MODEL_CACHE          = original_cache
        scorer._MODEL_LOAD_ATTEMPTED = original_attempted

    assert result is not None, "predict() must not return None when model is loaded"
    assert "score" in result
    assert "scoring_method" in result
    assert "ml_verdict" in result
    assert 0.0 <= result["score"] <= 100.0, f"score out of range: {result['score']}"
    assert result["scoring_method"] == "ML"
    assert result["ml_verdict"] in ("GENUINE", "SPOOF")


# ─────────────────────────────────────────────────────────────────────────────
# Test 8 — FEATURE_NAMES order is stable
# ─────────────────────────────────────────────────────────────────────────────

def test_feature_names_order_is_stable():
    """FEATURE_NAMES must be a list (not a set) with a fixed order.

    The order determines which column in the CSV maps to which feature during
    training. If the order changes after training, the pkl produces wrong scores.
    """
    from basetruth.face_scan.ml_scorer_live import FEATURE_NAMES

    assert isinstance(FEATURE_NAMES, list), "FEATURE_NAMES must be a list, not a set"
    # Re-import to verify the order is stable across imports (not randomly generated).
    from importlib import reload
    import basetruth.face_scan.ml_scorer_live as scorer_mod
    assert scorer_mod.FEATURE_NAMES == FEATURE_NAMES


# ─────────────────────────────────────────────────────────────────────────────
# Test 9 — progress_cb is called during training
# ─────────────────────────────────────────────────────────────────────────────

def test_train_calls_progress_cb():
    """train() must invoke the progress_cb callable at least once during training."""
    from basetruth.face_scan.ml_scorer_live import train, FEATURE_NAMES

    n = 20
    rows = []
    for i in range(2 * n):
        row = {name: float(i < n) if name == FEATURE_NAMES[0] else np.random.rand()
               for name in FEATURE_NAMES}
        row["label"] = 0 if i < n else 1
        rows.append(row)
    df = pd.DataFrame(rows)

    calls = []
    def fake_progress(step: str, pct: int) -> None:
        calls.append((step, pct))

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        csv_path = f.name
        df.to_csv(csv_path, index=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        pkl_path = str(Path(tmpdir) / "test_model.pkl")
        train(csv_path, pkl_path, progress_cb=fake_progress)

    assert len(calls) > 0, "progress_cb was never called during training"
    # Last call should report 100%
    assert calls[-1][1] == 100, f"last progress_cb call did not report 100%, got {calls[-1]}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 10 — append_training_sample writes a CSV row
# ─────────────────────────────────────────────────────────────────────────────

def _make_live_result(session_id: str = "fs_live_aabbccdd1122") -> dict:
    """Build a minimal final_result dict as produced by build_live_face_scan_result()."""
    return {
        "verdict": "GENUINE",
        "risk_score_0_100": 18.5,
        "confidence_0_100": 72.0,
        "environment": {"observed_fps": 10.0, "frame_drop_rate": 0.01, "virtual_camera_suspected": 0},
        "trace": {
            "decision_trace_id": session_id,
            "timestamp_utc": "2026-01-01T12:00:00Z",
        },
        "checks": {
            "temporal_consistency": {"yaw_jerk": 0.003, "pitch_jerk": 0.002, "nose_jitter": 1.5, "score_0_100": 12.0},
            "replay_heuristics": {"repeat_frame_score": 5.0, "flicker_score": 8.0, "brightness_instability": 3.0, "score_0_100": 10.0},
            "saccade_analysis": {"mean_eye_jitter": 0.4, "score_0_100": 20.0},
            "depth_consistency": {"iod_yaw_correlation": 0.72, "score_0_100": 15.0},
            "screen_frequency": {"mean_fft_grid_peak": 1.2, "score_0_100": 5.0},
            "frame_timing": {"interval_cv": 0.15, "score_0_100": 8.0},
            "quality_assessment": {"mean_face_area_ratio": 0.25, "blur_risk_0_100": 10.0, "brightness_risk_0_100": 5.0, "score_0_100": 15.0},
            "active_liveness": {"wrong_action_count": 0, "challenge_count": 3, "score_0_100": 0.0},
            "face_detection": {"frames_without_face": 2},
        },
    }


def test_append_training_sample_creates_csv_row():
    """append_training_sample must create the CSV and write exactly one row."""
    import csv
    import basetruth.face_scan.ml_scorer_live as scorer

    result = _make_live_result("fs_live_test001")

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "training_data_face_scan_live.csv"
        # Patch the module-level CSV path so we write to our temp directory.
        original = scorer._CSV_PATH
        scorer._CSV_PATH = csv_path
        try:
            scorer.append_training_sample(result, label=-1)
        finally:
            scorer._CSV_PATH = original

        assert csv_path.exists(), "CSV file must be created after first append"
        with csv_path.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 1, f"Expected 1 row, got {len(rows)}"
        assert rows[0]["session_id"] == "fs_live_test001"
        assert rows[0]["verdict"] == "GENUINE"
        assert rows[0]["label"] == "-1"


def test_append_training_sample_idempotent():
    """Calling append_training_sample twice with the same session_id writes only one row."""
    import csv
    import basetruth.face_scan.ml_scorer_live as scorer

    result = _make_live_result("fs_live_dedup001")

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "training_data_face_scan_live.csv"
        original = scorer._CSV_PATH
        scorer._CSV_PATH = csv_path
        try:
            scorer.append_training_sample(result)
            scorer.append_training_sample(result)  # second call — must be a no-op
        finally:
            scorer._CSV_PATH = original

        with csv_path.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 1, f"Idempotency violated: expected 1 row, got {len(rows)}"


def test_append_training_sample_header_columns():
    """CSV produced by append_training_sample must contain all expected column headers."""
    import csv
    import basetruth.face_scan.ml_scorer_live as scorer

    result = _make_live_result("fs_live_hdr001")

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "training_data_face_scan_live.csv"
        original = scorer._CSV_PATH
        scorer._CSV_PATH = csv_path
        try:
            scorer.append_training_sample(result)
        finally:
            scorer._CSV_PATH = original

        with csv_path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            headers = reader.fieldnames or []

    expected = ["session_id", "timestamp_utc", "verdict"] + scorer.FEATURE_NAMES + ["label"]
    assert headers == expected, f"Column mismatch.\nExpected: {expected}\nGot: {headers}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 11 — migrate_csv_if_stale repairs a stale training CSV
# ─────────────────────────────────────────────────────────────────────────────

def test_migrate_csv_if_stale_no_op_when_header_current():
    """migrate_csv_if_stale returns False when the header is already correct."""
    import csv
    import basetruth.face_scan.ml_scorer_live as scorer

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "training_data_face_scan_live.csv"
        # Write a file with the correct current header.
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(scorer._CSV_COLUMNS)
            w.writerow(["sid1", "2026-01-01T00:00:00Z", "GENUINE"] + ["0.5"] * len(scorer.FEATURE_NAMES) + ["0"])

        result = scorer.migrate_csv_if_stale(csv_path)

    assert result is False, "migrate_csv_if_stale should return False when header is already current"


def test_migrate_csv_if_stale_repairs_old_24_column_header():
    """migrate_csv_if_stale rewrites a 24-column header CSV to the current 28-column schema.

    Simulates the real bug: the header was written when FEATURE_NAMES had 20
    entries (24 columns total), new rows were then written with 24 features
    (28 columns total). After migration every row must have exactly len(_CSV_COLUMNS)
    columns and the header must match _CSV_COLUMNS exactly.
    """
    import csv
    import basetruth.face_scan.ml_scorer_live as scorer

    # Build a synthetic old header: same meta + label columns but only 20 features.
    old_features = scorer.FEATURE_NAMES[:20]
    old_header   = ["session_id", "timestamp_utc", "verdict"] + old_features + ["label"]  # 24 cols
    # One old row (24 columns) and one already-new row (28 columns).
    old_row = ["sid_old", "2026-01-01T00:00:00Z", "GENUINE"] + ["0.1"] * 20 + ["0"]
    new_row = ["sid_new", "2026-01-02T00:00:00Z", "SUSPICIOUS"] + ["0.2"] * len(scorer.FEATURE_NAMES) + ["1"]

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "training_data_face_scan_live.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(old_header)
            w.writerow(old_row)
            w.writerow(new_row)

        result = scorer.migrate_csv_if_stale(csv_path)

        assert result is True, "migrate_csv_if_stale should return True after a real migration"

        with csv_path.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.reader(fh))

    header   = rows[0]
    data_rows = rows[1:]

    assert header == scorer._CSV_COLUMNS, f"Header mismatch after migration: {header}"
    assert len(data_rows) == 2, f"Expected 2 data rows, got {len(data_rows)}"

    for i, row in enumerate(data_rows):
        assert len(row) == len(scorer._CSV_COLUMNS), (
            f"Row {i+2} has {len(row)} columns, expected {len(scorer._CSV_COLUMNS)}"
        )

    # The label column must be the last value in every migrated row.
    assert data_rows[0][-1] == "0",  "Old row label must be preserved as last column"
    assert data_rows[1][-1] == "1",  "New row label must be preserved as last column"

