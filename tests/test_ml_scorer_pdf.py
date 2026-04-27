"""tests/test_ml_scorer_pdf.py — Unit tests for the PDF ML forensic scorer.

These tests verify:
  - Feature extraction produces the right shape and types
  - Empty input doesn't crash (all zeros)
  - predict_pdf() returns None when no model file exists
  - predict_pdf() returns a valid score dict when a mock model is loaded
  - Score is clamped to [0.0, 100.0] regardless of model output
  - explain_pdf() returns None when the model is absent
  - explain_pdf() returns a valid contributions dict with a mock model
  - _remap_raw_pdf_csv() remaps our raw engine columns to the expert schema

Run with:
    python -m pytest tests/test_ml_scorer_pdf.py -v
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ── Path setup ────────────────────────────────────────────────────────────────
import sys
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from basetruth.analysis.ml_scorer_pdf import (
    PDF_FEATURE_NAMES,
    extract_feature_vector_pdf,
    predict_pdf,
    explain_pdf,
    _remap_raw_pdf_csv,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures — reusable fake PDF layer dicts
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def minimal_pdf_layers() -> dict:
    """A minimal but valid PDF layers dict with all supported layers present."""
    return {
        "layer_1_incremental_updates": {
            "metrics": {"incremental_updates": 2, "eof_marker_count": 3}
        },
        "layer_2_metadata": {
            "status": "SUSPICIOUS",
            "plain_english": "⚠️ Creator mismatch | ⚠️ Modified after creation",
        },
        "layer_3_font_consistency": {
            "metrics": {"total_unique_fonts": 5}
        },
        "layer_4_invisible_text": {
            "metrics": {"total_hidden_spans": 3, "white_text_spans": 2}
        },
        "layer_5_suspicious_objects": {
            "metrics": {"javascript_count": 1, "embedded_files_count": 0}
        },
        "layer_7_digital_signature": {
            "metrics": {"coverage_gaps": 2, "has_signature_field": True}
        },
        "layer_8_page_render_ela": {
            "metrics": {"suspicious_block_ratio": 0.08, "noise_hotspot_ratio": 0.04}
        },
        "layer_10_file_entropy": {
            "metrics": {"file_entropy_bits": 7.2}
        },
        "layer_11_object_xref_integrity": {
            "metrics": {"total_objects": 250, "xref_mismatch_score": 15.0}
        },
        "_meta": {"is_scanned_pdf": 0.0},
    }


@pytest.fixture
def empty_pdf_layers() -> dict:
    """An entirely empty layers dict — every metric will be missing."""
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 — Feature vector shape and type
# ─────────────────────────────────────────────────────────────────────────────

def test_pdf_feature_vector_shape(minimal_pdf_layers):
    """extract_feature_vector_pdf must return a numpy array of exactly 17 floats."""
    fv = extract_feature_vector_pdf(minimal_pdf_layers)
    assert isinstance(fv, np.ndarray), "result should be a numpy ndarray"
    assert fv.shape == (17,), f"expected shape (17,), got {fv.shape}"
    assert fv.dtype == np.float32, f"expected float32 dtype, got {fv.dtype}"
    assert len(PDF_FEATURE_NAMES) == 17, "PDF_FEATURE_NAMES must have exactly 17 entries"


def test_pdf_feature_vector_all_finite(minimal_pdf_layers):
    """No NaN or Inf values should appear in the feature vector for valid input."""
    fv = extract_feature_vector_pdf(minimal_pdf_layers)
    assert np.all(np.isfinite(fv)), "unexpected NaN or Inf in PDF feature vector"


def test_pdf_feature_vector_non_negative(minimal_pdf_layers):
    """All feature values must be >= 0 (no sentinel values for PDF features)."""
    fv = extract_feature_vector_pdf(minimal_pdf_layers)
    for name, val in zip(PDF_FEATURE_NAMES, fv):
        assert val >= 0.0, f"{name} is negative: {val}"


def test_pdf_feature_vector_capped(minimal_pdf_layers):
    """No feature should exceed its natural maximum (100 for score features)."""
    fv = extract_feature_vector_pdf(minimal_pdf_layers)
    score_features = {
        "metadata_anomaly_score", "signature_gap_score", "font_switch_score",
        "xref_mismatch_score", "ocr_text_layer_gap",
    }
    ratio_features = {
        "render_ela_suspicious_block_ratio", "render_noise_hotspot_ratio",
        "is_scanned_pdf", "has_signature",
    }
    for name, val in zip(PDF_FEATURE_NAMES, fv):
        if name in score_features:
            assert val <= 100.0, f"{name} exceeds 100: {val}"
        elif name in ratio_features:
            assert val <= 1.0, f"{name} exceeds 1.0: {val}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 — Empty layers dict does not crash
# ─────────────────────────────────────────────────────────────────────────────

def test_pdf_feature_vector_empty_layers(empty_pdf_layers):
    """Empty dict must produce all zeros (no crash, correct shape)."""
    fv = extract_feature_vector_pdf(empty_pdf_layers)
    assert fv.shape == (17,), "shape must still be (17,) for empty input"
    # eof_marker_count defaults to 1 (clean doc assumption) — check others are 0
    feat_dict = dict(zip(PDF_FEATURE_NAMES, fv))
    for name, val in feat_dict.items():
        if name == "eof_marker_count":
            # Default for a clean document is 1 (one %%EOF marker expected)
            assert val == 1.0, f"eof_marker_count default should be 1.0, got {val}"
        else:
            assert val == 0.0, f"expected 0.0 for {name} with empty layers, got {val}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 — predict_pdf() returns None when model file is absent
# ─────────────────────────────────────────────────────────────────────────────

def test_predict_pdf_returns_none_when_no_model():
    """When the pkl file does not exist, predict_pdf() must return None, not raise."""
    import basetruth.analysis.ml_scorer_pdf as ms

    original_cache = ms._MODEL_CACHE
    original_attempted = ms._MODEL_LOAD_ATTEMPTED
    ms._MODEL_CACHE = None
    ms._MODEL_LOAD_ATTEMPTED = False

    try:
        with patch.object(ms, "_MODEL_PATH", Path("/nonexistent/path/ml_scorer_pdf.pkl")):
            fv = np.zeros(17, dtype=np.float32)
            result = predict_pdf(fv)
        assert result is None, "predict_pdf() must return None when model file is absent"
    finally:
        ms._MODEL_CACHE = original_cache
        ms._MODEL_LOAD_ATTEMPTED = original_attempted


# ─────────────────────────────────────────────────────────────────────────────
# Test 4 — predict_pdf() returns valid dict when a mock model is available
# ─────────────────────────────────────────────────────────────────────────────

def test_predict_pdf_returns_valid_dict_with_mock_model():
    """When a model is loaded, predict_pdf() must return score (0–100), confidence (0–1),
    and scoring_method='ML'."""
    import basetruth.analysis.ml_scorer_pdf as ms

    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.30, 0.70]])

    original_cache = ms._MODEL_CACHE
    original_attempted = ms._MODEL_LOAD_ATTEMPTED
    ms._MODEL_CACHE = mock_model
    ms._MODEL_LOAD_ATTEMPTED = True

    try:
        fv = np.zeros(17, dtype=np.float32)
        result = predict_pdf(fv)
        assert result is not None, "predict_pdf() must return a dict when model is loaded"
        assert "score" in result
        assert "confidence" in result
        assert "scoring_method" in result
        assert 0.0 <= result["score"] <= 100.0, f"score out of range: {result['score']}"
        assert 0.0 <= result["confidence"] <= 1.0, f"confidence out of range: {result['confidence']}"
        assert result["scoring_method"] == "ML"
        assert result["score"] == pytest.approx(70.0, abs=0.1)
    finally:
        ms._MODEL_CACHE = original_cache
        ms._MODEL_LOAD_ATTEMPTED = original_attempted


# ─────────────────────────────────────────────────────────────────────────────
# Test 5 — Score boundary clamping
# ─────────────────────────────────────────────────────────────────────────────

def test_predict_pdf_score_range_boundary():
    """predict_pdf() must produce scores in [0.0, 100.0] for extreme probabilities."""
    import basetruth.analysis.ml_scorer_pdf as ms

    for p_tampered in [0.0, 0.5, 1.0]:
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[1 - p_tampered, p_tampered]])

        original_cache = ms._MODEL_CACHE
        original_attempted = ms._MODEL_LOAD_ATTEMPTED
        ms._MODEL_CACHE = mock_model
        ms._MODEL_LOAD_ATTEMPTED = True
        try:
            fv = np.zeros(17, dtype=np.float32)
            result = predict_pdf(fv)
            assert result is not None
            assert 0.0 <= result["score"] <= 100.0, (
                f"score {result['score']} out of [0,100] for p_tampered={p_tampered}"
            )
        finally:
            ms._MODEL_CACHE = original_cache
            ms._MODEL_LOAD_ATTEMPTED = original_attempted


# ─────────────────────────────────────────────────────────────────────────────
# Test 6 — explain_pdf() returns None when model file is absent
# ─────────────────────────────────────────────────────────────────────────────

def test_explain_pdf_returns_none_when_no_model():
    """explain_pdf() must return None (not raise) when no model is loaded."""
    import basetruth.analysis.ml_scorer_pdf as ms

    original_cache = ms._MODEL_CACHE
    original_attempted = ms._MODEL_LOAD_ATTEMPTED
    ms._MODEL_CACHE = None
    ms._MODEL_LOAD_ATTEMPTED = False

    try:
        with patch.object(ms, "_MODEL_PATH", Path("/nonexistent/path/ml_scorer_pdf.pkl")):
            fv = np.zeros(17, dtype=np.float32)
            result = explain_pdf(fv)
        assert result is None, "explain_pdf() must return None when model is absent"
    finally:
        ms._MODEL_CACHE = original_cache
        ms._MODEL_LOAD_ATTEMPTED = original_attempted


# ─────────────────────────────────────────────────────────────────────────────
# Test 7 — explain_pdf() returns contributions dict with mock XGBoost model
# ─────────────────────────────────────────────────────────────────────────────

def test_explain_pdf_returns_contributions_with_mock_model():
    """explain_pdf() must return a {feature_name: float} dict with 17 entries."""
    import basetruth.analysis.ml_scorer_pdf as ms
    import xgboost as xgb

    # Build a mock pipeline with imputer + booster.
    mock_imputer = MagicMock()
    mock_imputer.transform.return_value = np.zeros((1, 17), dtype=np.float32)

    # Booster.predict(dmat, pred_contribs=True) returns shape (1, n_features+1).
    # We have 17 features so the output is (1, 18) — last col is bias.
    fake_shap = np.random.uniform(-0.5, 0.5, (1, 18)).astype(np.float32)
    mock_booster = MagicMock()
    mock_booster.predict.return_value = fake_shap

    mock_model_step = MagicMock()
    mock_model_step.get_booster.return_value = mock_booster

    mock_pipeline = MagicMock()
    mock_pipeline.named_steps = {"imputer": mock_imputer, "model": mock_model_step}

    original_cache = ms._MODEL_CACHE
    original_attempted = ms._MODEL_LOAD_ATTEMPTED
    ms._MODEL_CACHE = mock_pipeline
    ms._MODEL_LOAD_ATTEMPTED = True

    try:
        fv = np.zeros(17, dtype=np.float32)
        # Patch DMatrix so the test runs even when XGBoost C libs behave differently
        with patch.object(xgb, "DMatrix", return_value=MagicMock()):
            contributions = explain_pdf(fv)

        assert contributions is not None, "explain_pdf() must return a dict with a mock model"
        assert isinstance(contributions, dict)
        assert len(contributions) == 17, f"expected 17 features, got {len(contributions)}"
        for name in PDF_FEATURE_NAMES:
            assert name in contributions, f"missing feature: {name}"
            assert isinstance(contributions[name], float), (
                f"{name} contribution should be float, got {type(contributions[name])}"
            )
    finally:
        ms._MODEL_CACHE = original_cache
        ms._MODEL_LOAD_ATTEMPTED = original_attempted


# ─────────────────────────────────────────────────────────────────────────────
# Test 8 — _remap_raw_pdf_csv() produces expected schema
# ─────────────────────────────────────────────────────────────────────────────

def test_remap_raw_pdf_csv_columns():
    """_remap_raw_pdf_csv must output a DataFrame with PDF_FEATURE_NAMES columns."""
    import pandas as pd

    # One-row raw CSV as produced by collect_training_samples_pdf.py
    raw = pd.DataFrame([{
        "filename": "payslip.pdf",
        "incremental_updates": 1,
        "eof_marker_count": 2,
        "metadata_flag_count": 3,          # → metadata_anomaly_score = 60.0
        "total_hidden_spans": 5,           # → hidden_text_spans
        "white_text_spans": 2,
        "javascript_count": 0,
        "embedded_files_count": 0,
        "signature_coverage_gaps": 1,      # → signature_gap_score = 10.0
        "render_ela_suspicious_block_ratio": 0.05,
        "render_noise_hotspot_ratio": 0.02,
        "total_pdf_objects": 180,          # → object_count
        "file_entropy_bits": 6.8,          # → stream_entropy
        "total_unique_fonts": 3,           # → font_switch_score
        "ocr_text_layer_gap": 0.0,
        "is_scanned_pdf": 0,
        "has_signature_field": 0,          # → has_signature
        "label": 0,
    }])

    remapped = _remap_raw_pdf_csv(raw)

    # All 17 model features must be present
    for col in PDF_FEATURE_NAMES:
        assert col in remapped.columns, f"missing column after remap: {col}"

    # Verify scale transformations
    assert float(remapped["metadata_anomaly_score"].iloc[0]) == pytest.approx(60.0)
    assert float(remapped["signature_gap_score"].iloc[0]) == pytest.approx(10.0)
    assert float(remapped["object_count"].iloc[0]) == pytest.approx(180.0)
    assert float(remapped["stream_entropy"].iloc[0]) == pytest.approx(6.8)
    assert float(remapped["font_switch_score"].iloc[0]) == pytest.approx(3.0)
    assert float(remapped["has_signature"].iloc[0]) == pytest.approx(0.0)


def test_remap_raw_pdf_csv_metadata_anomaly_capped():
    """metadata_anomaly_score must never exceed 100 regardless of flag count."""
    import pandas as pd

    raw = pd.DataFrame([{
        "metadata_flag_count": 10,  # 10 * 20 = 200 → should be capped at 100
        "label": 1,
    }])
    remapped = _remap_raw_pdf_csv(raw)
    assert float(remapped["metadata_anomaly_score"].iloc[0]) == pytest.approx(100.0)


# ─────────────────────────────────────────────────────────────────────────────
# Test 9 — metadata layer status path in feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def test_pdf_metadata_anomaly_clean():
    """A CLEAN metadata layer must produce metadata_anomaly_score = 0."""
    layers = {
        "layer_2_metadata": {"status": "CLEAN", "plain_english": "No issues"},
        "_meta": {"is_scanned_pdf": 0.0},
    }
    fv = extract_feature_vector_pdf(layers)
    feat_dict = dict(zip(PDF_FEATURE_NAMES, fv))
    assert feat_dict["metadata_anomaly_score"] == 0.0


def test_pdf_metadata_anomaly_suspicious():
    """A SUSPICIOUS metadata layer must produce a positive metadata_anomaly_score."""
    layers = {
        "layer_2_metadata": {
            "status": "SUSPICIOUS",
            "plain_english": "⚠️ Flag one | ⚠️ Flag two | ⚠️ Flag three",
        },
        "_meta": {"is_scanned_pdf": 0.0},
    }
    fv = extract_feature_vector_pdf(layers)
    feat_dict = dict(zip(PDF_FEATURE_NAMES, fv))
    # 3 flags × 20 = 60.0
    assert feat_dict["metadata_anomaly_score"] == pytest.approx(60.0, abs=1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Test 10 — is_scanned_pdf flag propagates correctly
# ─────────────────────────────────────────────────────────────────────────────

def test_pdf_is_scanned_flag():
    """is_scanned_pdf = 1 in _meta must appear as 1.0 in the feature vector."""
    layers = {"_meta": {"is_scanned_pdf": 1.0}}
    fv = extract_feature_vector_pdf(layers)
    feat_dict = dict(zip(PDF_FEATURE_NAMES, fv))
    assert feat_dict["is_scanned_pdf"] == pytest.approx(1.0)
