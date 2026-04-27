"""tests/test_ml_scorer.py — Unit tests for the ML forensic scorer module.

These tests verify:
  - Feature extraction produces the right shape and types
  - Empty input doesn't crash (all zeros, graceful handling)
  - predict() returns None when no model file exists
  - predict() returns a valid score dict when a real model is loaded
  - The heuristic fallback path in _compute_score() is unaffected by ML

Run with:
    python -m pytest tests/test_ml_scorer.py -v
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ── Path setup ─────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from basetruth.analysis.ml_scorer import (
    FEATURE_NAMES,
    extract_feature_vector,
    predict,
    _remap_raw_csv,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures — reusable fake layer dicts
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def minimal_layers() -> dict:
    """A minimal but valid layers dict with all 11 layers present."""
    return {
        "layer_1_ela": {"metrics": {"suspicious_block_ratio": 0.10, "mean_ela": 12.0, "std_ela": 5.0, "max_ela": 30.0}},
        "layer_2_metadata": {"suspicious_flags": ["Software mismatch", "GPS stripped"]},
        "layer_3_entropy": {"metrics": {"entropy_bits": 7.5}},
        "layer_4_noise": {"metrics": {"hotspot_tile_ratio": 0.05}},
        "layer_5_dct": {"metrics": {"comb_ratio": 1.5, "skipped": False}},
        "layer_6_clone": {"metrics": {"clone_ratio": 0.30}},
        "layer_7_color": {"metrics": {"anomaly_ratio": 0.005, "anomaly_blobs": [{"area_px": 1500}]}},
        "layer_8_edge": {"metrics": {"high_density_tile_ratio": 0.08}},
        "layer_9_saturation": {"metrics": {"high_saturation_tile_ratio": 0.03}},
        "layer_10_font": {"metrics": {"stroke_cv": 0.50, "n_suspicious_regions": 2,
                                      "sharpness_outlier_ratio": 0.10, "skipped": False}},
        "layer_11_ai": {"metrics": {"spike_ratio": 2.0}},
    }


@pytest.fixture
def empty_layers() -> dict:
    """An entirely empty layers dict — every metric will be missing."""
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 — Feature vector shape and type
# ─────────────────────────────────────────────────────────────────────────────

def test_feature_vector_shape(minimal_layers):
    """extract_feature_vector must return a numpy array of exactly 19 floats."""
    fv = extract_feature_vector(minimal_layers)
    assert isinstance(fv, np.ndarray), "result should be a numpy ndarray"
    assert fv.shape == (19,), f"expected shape (19,), got {fv.shape}"
    assert fv.dtype == np.float32, f"expected float32 dtype, got {fv.dtype}"
    assert len(FEATURE_NAMES) == 19, "FEATURE_NAMES must have exactly 19 entries"


def test_feature_vector_all_finite(minimal_layers):
    """No NaN or Inf values should appear in the feature vector for valid input."""
    fv = extract_feature_vector(minimal_layers)
    assert np.all(np.isfinite(fv)), "unexpected NaN or Inf in feature vector"


def test_feature_vector_values_in_range(minimal_layers):
    """Raw feature values must meet their natural domain constraints.

    Ratio-type features (fractions, CVs, ratios) must be >= 0.
    Boolean skip flags must be 0 or 1.
    Pixel-count features may be any non-negative number.
    clone_ratio must be in [0, 1].
    """
    fv = extract_feature_vector(minimal_layers)
    feat_dict = dict(zip(FEATURE_NAMES, fv))

    # Ratios and fractions must be non-negative
    ratio_features = {
        "ela_mean", "ela_max", "ela_std", "ela_suspicious_block_ratio",
        "metadata_flag_count", "file_entropy_bits", "noise_hotspot_ratio",
        "dct_comb_ratio", "color_anomaly_ratio",
        "edge_high_density_ratio", "saturation_ratio",
        "font_stroke_cv", "font_suspicious_regions", "font_sharpness_outlier_ratio",
        "ai_spike_ratio",
    }
    for name in ratio_features:
        assert feat_dict[name] >= 0.0, f"{name} must be >= 0, got {feat_dict[name]}"

    # clone_ratio must be clamped to [0, 1]
    assert 0.0 <= feat_dict["clone_ratio"] <= 1.0, (
        f"clone_ratio out of [0,1]: {feat_dict['clone_ratio']}"
    )

    # Skip flags must be 0.0 or 1.0
    for flag in ("dct_skipped", "font_skipped"):
        assert feat_dict[flag] in (0.0, 1.0), (
            f"{flag} must be 0.0 or 1.0, got {feat_dict[flag]}"
        )

    # color_largest_blob_px is a raw pixel count — just check non-negative
    assert feat_dict["color_largest_blob_px"] >= 0.0, "blob pixel count must be >= 0"


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 — Empty layers dict does not crash
# ─────────────────────────────────────────────────────────────────────────────

def test_feature_vector_empty_layers(empty_layers):
    """Empty dict must produce all zeros (no sentinels any more)."""
    fv = extract_feature_vector(empty_layers)
    assert fv.shape == (19,), "shape must still be (19,) for empty input"
    assert np.all(fv == 0.0), (
        f"expected all zeros for empty layers, non-zero positions: "
        f"{[(FEATURE_NAMES[i], float(v)) for i, v in enumerate(fv) if v != 0.0]}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 — predict() returns None when model file is absent
# ─────────────────────────────────────────────────────────────────────────────

def test_predict_returns_none_when_no_model():
    """When the pkl file does not exist, predict() must return None, not raise."""
    import basetruth.analysis.ml_scorer as ms

    # Reset the module-level cache to force a fresh load attempt
    original_cache = ms._MODEL_CACHE
    original_attempted = ms._MODEL_LOAD_ATTEMPTED
    ms._MODEL_CACHE = None
    ms._MODEL_LOAD_ATTEMPTED = False

    try:
        # Patch the model path to a non-existent file
        with patch.object(ms, "_MODEL_PATH", Path("/nonexistent/path/ml_scorer_image.pkl")):
            fv = np.zeros(19, dtype=np.float32)
            result = predict(fv)
        assert result is None, "predict() must return None when model file is absent"
    finally:
        ms._MODEL_CACHE = original_cache
        ms._MODEL_LOAD_ATTEMPTED = original_attempted


# ─────────────────────────────────────────────────────────────────────────────
# Test 4 — predict() returns valid dict when a mock model is available
# ─────────────────────────────────────────────────────────────────────────────

def test_predict_returns_valid_dict_with_mock_model():
    """When a model is loaded, predict() must return score (0–100) and confidence (0–1)."""
    import basetruth.analysis.ml_scorer as ms

    # Build a minimal mock pipeline that returns realistic probabilities
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.25, 0.75]])

    original_cache = ms._MODEL_CACHE
    original_attempted = ms._MODEL_LOAD_ATTEMPTED
    ms._MODEL_CACHE = mock_model
    ms._MODEL_LOAD_ATTEMPTED = True

    try:
        fv = np.zeros(19, dtype=np.float32)
        result = predict(fv)
        assert result is not None, "predict() must return a dict when model is loaded"
        assert "score" in result
        assert "confidence" in result
        assert "scoring_method" in result
        assert 0.0 <= result["score"] <= 100.0, f"score out of range: {result['score']}"
        assert 0.0 <= result["confidence"] <= 1.0, f"confidence out of range: {result['confidence']}"
        assert result["scoring_method"] == "ML"
        assert result["score"] == pytest.approx(75.0, abs=0.1)
    finally:
        ms._MODEL_CACHE = original_cache
        ms._MODEL_LOAD_ATTEMPTED = original_attempted


def test_predict_score_range_boundary():
    """predict() must clamp score to [0.0, 100.0] regardless of model output."""
    import basetruth.analysis.ml_scorer as ms

    # Model outputs extreme probabilities close to 0 and 1
    for proba_tampered in [0.0, 0.5, 1.0]:
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[1 - proba_tampered, proba_tampered]])

        original_cache = ms._MODEL_CACHE
        original_attempted = ms._MODEL_LOAD_ATTEMPTED
        ms._MODEL_CACHE = mock_model
        ms._MODEL_LOAD_ATTEMPTED = True
        try:
            fv = np.zeros(19, dtype=np.float32)
            result = predict(fv)
            assert result is not None
            assert 0.0 <= result["score"] <= 100.0
        finally:
            ms._MODEL_CACHE = original_cache
            ms._MODEL_LOAD_ATTEMPTED = original_attempted


# ─────────────────────────────────────────────────────────────────────────────
# Test 5 — Heuristic fallback in _compute_score()
# ─────────────────────────────────────────────────────────────────────────────

def test_heuristic_fallback_when_ml_returns_none(minimal_layers):
    """When ML returns None, _compute_score() must use the heuristic score unchanged
    and return scoring_method='heuristic'.
    """
    from basetruth.analysis.image_forensics_detect import _compute_score, _heuristic_score
    import basetruth.analysis.ml_scorer as ms

    # Force ML to be unavailable
    original_cache = ms._MODEL_CACHE
    original_attempted = ms._MODEL_LOAD_ATTEMPTED
    ms._MODEL_CACHE = None
    ms._MODEL_LOAD_ATTEMPTED = False

    try:
        with patch.object(ms, "_MODEL_PATH", Path("/nonexistent/path/model.pkl")):
            h_score, h_verdict, h_evidence = _heuristic_score(minimal_layers)
            c_score, c_verdict, c_evidence, c_method, c_contribs = _compute_score(minimal_layers, 100_000)

        assert c_method == "heuristic", f"expected 'heuristic', got '{c_method}'"
        assert c_score == h_score, f"heuristic fallback score mismatch: {c_score} != {h_score}"
        assert c_verdict == h_verdict
        assert c_evidence == h_evidence
        assert c_contribs is None, "feature_contributions must be None when heuristic ran"
    finally:
        ms._MODEL_CACHE = original_cache
        ms._MODEL_LOAD_ATTEMPTED = original_attempted


def test_ml_score_used_when_model_available(minimal_layers):
    """When ML model is loaded, _compute_score() must use ML score and return
    scoring_method='ML'.
    """
    from basetruth.analysis.image_forensics_detect import _compute_score
    import basetruth.analysis.ml_scorer as ms

    mock_model = MagicMock()
    # Mock returns P(tampered)=0.90 → score=90.0
    mock_model.predict_proba.return_value = np.array([[0.10, 0.90]])

    original_cache = ms._MODEL_CACHE
    original_attempted = ms._MODEL_LOAD_ATTEMPTED
    ms._MODEL_CACHE = mock_model
    ms._MODEL_LOAD_ATTEMPTED = True

    try:
        c_score, c_verdict, c_evidence, c_method, c_contribs = _compute_score(minimal_layers, 100_000)
        assert c_method == "ML", f"expected 'ML', got '{c_method}'"
        assert c_score == pytest.approx(90.0, abs=0.1)
        assert c_verdict == "TAMPERED"
        # Evidence strings must still come from the heuristic regardless of scorer
        assert isinstance(c_evidence, list)
    finally:
        ms._MODEL_CACHE = original_cache
        ms._MODEL_LOAD_ATTEMPTED = original_attempted


# ─────────────────────────────────────────────────────────────────────────────
# Test 6 — _remap_raw_csv produces correct output shape
# ─────────────────────────────────────────────────────────────────────────────

def test_remap_raw_csv_columns():
    """_remap_raw_csv must output exactly FEATURE_NAMES + 'label' columns.

    All output values must be non-negative.  clone_ratio must be in [0, 1].
    color_largest_blob_px is a raw pixel count so it can exceed 100.
    """
    import pandas as pd

    # Minimal one-row raw CSV as would be produced by collect_training_samples.py
    raw = pd.DataFrame([{
        "filename": "test.jpg",
        "ela_mean": 12.0,
        "ela_max": 30.0,
        "ela_std": 5.0,
        "ela_suspicious_block_ratio": 0.10,
        "metadata_flag_count": 2,
        "file_entropy_bits": 7.5,
        "noise_hotspot_ratio": 0.05,
        "dct_comb_ratio": 1.5,
        "dct_skipped": 0,
        "clone_ratio": 0.30,
        "color_anomaly_ratio": 0.005,
        "color_largest_blob_px": 1500,
        "edge_high_density_ratio": 0.08,
        "saturation_ratio": 0.03,
        "font_stroke_cv": 0.50,
        "font_suspicious_regions": 2,
        "font_sharpness_outlier_ratio": 0.10,
        "font_skipped": 0,
        "ai_spike_ratio": 2.0,
        "label": 1,
    }])

    remapped = _remap_raw_csv(raw)
    expected_cols = set(FEATURE_NAMES) | {"label"}
    assert set(remapped.columns) == expected_cols, (
        f"unexpected columns: {set(remapped.columns) ^ expected_cols}"
    )
    assert len(remapped) == 1

    # clone_ratio clamped to [0, 1]; all others non-negative
    for col in FEATURE_NAMES:
        val = float(remapped[col].iloc[0])
        assert val >= 0.0, f"{col} must be non-negative, got {val}"
    assert float(remapped["clone_ratio"].iloc[0]) <= 1.0, "clone_ratio must be <= 1.0"
