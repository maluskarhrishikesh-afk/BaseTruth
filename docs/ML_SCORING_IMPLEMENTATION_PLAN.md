# ML Scoring — Implementation Plan

## Background

BaseTruth currently uses a fixed point-weight heuristic in two files:
- `src/basetruth/analysis/image_forensics_detect.py` → `_compute_score()`
- `src/basetruth/analysis/pdf_forensics_detect.py` → `compute_score()`

The expert review of our first training dataset (62 rows from `data/training_data_image.csv`) and the provision of a 10,000-row synthetic reference dataset (`data/forensic_training_10000_rows.csv`) plus an XGBoost + SHAP notebook (`data/forensic_xgboost_shap_notebook.ipynb`) have validated the approach and identified exactly what needs to be fixed.

---

## What the Expert Analysis Found

### Issues in our first 62-row CSV

| # | Issue | Impact | Fix |
|---|-------|--------|-----|
| 1 | Same filename appears twice (label=0 and label=1) | Model memorises filenames → fake accuracy | Remove `filename` from training features |
| 2 | `metadata_flag_count` perfectly predicts label (correlation=1.0) | Label leakage → useless in production | Add natural variance to synthetic data |
| 3 | `file_entropy_bits` = 0 for all rows | Zero information content | Drop column from training |
| 4 | `dct_skipped` = 0 always | Zero information content | Drop column from training |
| 5 | `font_skipped` = 0 always | Zero information content | Drop column from training |
| 6 | `clone_ratio` goes above 1.0 | Invalid metric range | Fix normalisation formula |
| 7 | Only 62 rows | Far too small for XGBoost generalisation | Use the expert 10,000-row dataset as core |

### Good signals confirmed by expert
- ELA, DCT, Clone detection, Color anomaly, Font inconsistencies, Noise hotspots, AI spike ratio — all valid and worth keeping.

---

## The Expert-Provided Assets

Three files were provided in `data/`:

| File | Rows | Columns | Purpose |
|------|------|---------|---------|
| `forensic_training_1000_rows.csv` | 1,000 | 12 features + label | Quick baseline training set |
| `forensic_training_10000_rows.csv` | 10,000 | 12 features + label + hard_case | Primary training dataset |
| `forensic_xgboost_shap_notebook.ipynb` | — | 4 cells | Reference training + SHAP analysis |

**Expert 10k dataset characteristics (validated):**
- Perfectly balanced: 5,000 genuine (label=0) / 5,000 tampered (label=1)
- 1,054 hard-case rows (label=1 only) — expert-level forgeries
- Realistic feature overlap: `dct_score`, `noise_hotspots`, `compression_mismatch` overlap between classes (no leakage)
- Features use normalised 0–100 scale (not raw engine values)

**Column mapping — expert CSV → our engine:**

| Expert column | Our engine column | Notes |
|---|---|---|
| `ela_score` | `ela_suspicious_block_ratio` × 100 + `ela_mean` composite | Needs normalisation to 0–100 |
| `dct_score` | `dct_comb_ratio` mapped to 0–100 | comb_ratio 0–3+ maps to 0–100 |
| `metadata_flag_count` | `metadata_flag_count` | Direct match |
| `clone_ratio` | `clone_ratio` clamped to 0–1 | Fix > 1.0 values |
| `text_alignment_score` | Not yet extracted — **new layer** | Needs implementation |
| `font_inconsistency` | `font_stroke_cv` + `font_suspicious_regions` composite | Normalise to 0–100 |
| `signature_mismatch` | Not yet extracted for images — **new layer** | Optional / -1 if absent |
| `noise_hotspots` | `noise_hotspot_ratio` × 100 | Already available |
| `color_patch_score` | `color_anomaly_ratio` × 100 + `color_largest_blob_px` composite | Normalise to 0–100 |
| `ai_artifact_score` | `ai_spike_ratio` mapped to 0–100 | spike_ratio 0–6+ maps to 0–100 |
| `compression_mismatch` | Not yet computed for images — derived from ELA std + DCT | Needs computation |

---

## Architecture: Where the ML Scorer Plugs In

```
image_forensics_detect.py::run_forensics(path)
    │
    ├── runs layers 1–11 (unchanged)
    │
    ▼
_compute_score(layers)           ← TODAY: fixed weights
    │
    ├── extract_image_feature_vector(layers)   [new]
    │        │
    │        ▼  11 normalised floats (0–100 each)
    │
    ├── ml_scorer.predict(feature_vector)      [new]
    │        │
    │        ├── model loaded? ──→ ML score (0–100) + confidence
    │        │
    │        └── no model ────→ None (fall back to heuristic)
    │
    ▼
Final: score, verdict, evidence, scoring_method ("ML" or "heuristic")
```

---

## Files to Create or Modify

| # | Action | File | What changes |
|---|--------|------|--------------|
| 1 | **Create** | `src/basetruth/analysis/ml_scorer.py` | Feature extractor (engine→0–100 normalised values) + sklearn Random Forest/XGBoost wrapper + `predict()` + `train()` |
| 2 | **Create** | `scripts/train_ml_scorer.py` | CLI: load expert 10k CSV + our 62-row CSV → merge → train → 5-fold CV → save `data/ml_scorer_image.pkl` |
| 3 | **Modify** | `src/basetruth/analysis/image_forensics_detect.py` | In `_compute_score()`: after computing `layers`, call `ml_scorer.predict()` — use ML score if available, fall back to heuristic |
| 4 | **Modify** | `src/basetruth/ui/pages/forensic_scan.py` | Add "Scoring method" badge: `🤖 ML Model` or `📐 Heuristic` in the classification banner |
| 5 | **Modify** | `src/basetruth/ui/pages/forensics_utils.py` | Pass `scoring_method` through the payload so the UI badge can display it |
| 6 | **Modify** | `data/training_data_image.csv` | Rebuild with corrected columns (drop leaky/constant cols, fix clone_ratio > 1.0) |
| 7 | **Modify** | `scripts/collect_training_samples.py` | Fix: clamp `clone_ratio` to [0, 1]; add normalised composite columns matching expert schema |

PDF forensics ML scoring is explicitly **out of scope for this phase** — the expert dataset covers image documents only. The PDF heuristic remains unchanged.

---

## Detailed Step-by-Step Plan

### Step 1 — `src/basetruth/analysis/ml_scorer.py`

This module has three responsibilities:

**1a. `extract_feature_vector(layers: dict) → np.ndarray`**

Converts raw engine output into the 11 normalised (0–100) features matching the expert training schema:

```python
ela_score             = min(100, ela_suspicious_block_ratio * 800 + ela_mean * 2)
dct_score             = min(100, max(0, (dct_comb_ratio - 1.0) * 60))
metadata_flag_count   = metadata_flag_count  # already integer 0–5
clone_ratio           = min(1.0, clone_ratio)  # clamp > 1.0
text_alignment_score  = 0.0  # not yet implemented — placeholder
font_inconsistency    = min(100, font_stroke_cv * 120 + font_suspicious_regions * 5)
signature_mismatch    = -1.0  # not applicable for image scans — handle as missing
noise_hotspots        = min(100, noise_hotspot_ratio * 1000)
color_patch_score     = min(100, color_anomaly_ratio * 2000 + log(1 + color_largest_blob_px) * 3)
ai_artifact_score     = min(100, max(0, (ai_spike_ratio - 1.0) * 20))
compression_mismatch  = min(100, ela_std * 6 + dct_score * 0.4)
```

**1b. `predict(feature_vector: np.ndarray) → dict | None`**

- Load `data/ml_scorer_image.pkl` (joblib)
- If file missing → return `None` (caller uses heuristic)
- Call `model.predict_proba(feature_vector.reshape(1, -1))[0][1]` → P(tampered)
- Map P(tampered) → score 0–100
- Return `{"score": float, "confidence": float, "scoring_method": "ML"}`

**1c. `train(csv_path: str, output_pkl: str) → dict`**

- Load CSV, drop `doc_id`, `hard_case` columns
- Replace `signature_mismatch = -1` with `NaN`, impute with median
- Train `XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, scale_pos_weight=1)`
- 5-fold stratified CV — print accuracy, precision, recall, ROC AUC per fold
- Evaluate on hard_case subset separately
- Save model with joblib

---

### Step 2 — `scripts/train_ml_scorer.py`

CLI script to train and evaluate the model:

```bash
python scripts/train_ml_scorer.py
```

What it does:
1. Loads `data/forensic_training_10000_rows.csv` (expert 10k — primary training base)
2. Loads `data/training_data_image.csv` (our 62 real-image rows) and **remaps columns** to the expert schema using the normalisation formulas above
3. Concatenates both, giving ~10,062 rows
4. Trains XGBClassifier with 5-fold stratified CV
5. Prints evaluation table:
   - Overall accuracy, F1, ROC AUC
   - Hard-case accuracy (from `hard_case=1` subset)
   - Feature importances (SHAP-style ranked)
6. Saves `data/ml_scorer_image.pkl`
7. Prints clear PASS/FAIL: refuses to save if ROC AUC < 0.80

---

### Step 3 — Modify `_compute_score()` in `image_forensics_detect.py`

Current flow:
```python
def _compute_score(layers, file_size_bytes):
    score = 0.0
    # ... 11 blocks of "score += X if signal > threshold" ...
    return score, verdict, evidence
```

New flow (backwards-compatible, zero regression):
```python
def _compute_score(layers, file_size_bytes):
    # Always compute heuristic score + evidence (for human-readable explanation)
    heuristic_score, heuristic_verdict, evidence = _heuristic_score(layers)

    # Attempt ML scoring (returns None if no model loaded)
    ml_result = _try_ml_score(layers)
    if ml_result is not None:
        final_score = ml_result["score"]
        scoring_method = "ML"
    else:
        final_score = heuristic_score
        scoring_method = "heuristic"

    verdict = (
        "TAMPERED"        if final_score >= 55 else
        "LIKELY TAMPERED" if final_score >= 30 else
        "UNCERTAIN"       if final_score >= 15 else
        "ORIGINAL"
    )
    return final_score, verdict, evidence, scoring_method
```

The `evidence` list is always generated from the heuristic logic — it provides the plain-English explanations shown to the reviewer regardless of which scorer produced the numeric score.

---

### Step 4 — UI badge in `forensic_scan.py`

In the classification banner, add a third badge line:

```
🤖 ML Model (XGBoost)      ← when ml_scorer.pkl exists and scored this document
📐 Heuristic (rule-based)  ← when model not loaded
```

This is a one-line addition to the `st.markdown()` call for the banner — no new components needed.

---

### Step 5 — Fix `training_data_image.csv`

The current 62-row CSV has leaky/invalid columns. The fix happens inside `collect_training_samples.py`:

**Columns to drop from training (keep in CSV but never feed to model):**
- `filename` — memorisation risk
- `file_entropy_bits` — always 0 (bug in entropy layer)
- `dct_skipped` — always 0
- `font_skipped` — always 0
- `heuristic_score` — target leakage (derived from same signals)
- `heuristic_verdict` — target leakage

**Fix `clone_ratio`:**  
SIFT keypoint matching returns a count, not a fraction — values > 1.0 are a normalisation error. Fix: `clone_ratio = matched_keypoints / max(total_keypoints, 1)`, clamped to [0, 1].

---

## Training Data Strategy

| Source | Rows | Quality | Role |
|--------|------|---------|------|
| `forensic_training_10000_rows.csv` | 10,000 | Synthetic, expert-crafted, realistic distributions, hard cases | Primary training backbone |
| `training_data_image.csv` (ours, remapped) | 62 | Real photos, label=0 anchors real baseline | Grounding supplement |
| Future: more real tampered scans | TBD | Real-world ground truth | Progressive improvement |

The synthetic 10k dataset is used **as-is** because the expert confirmed it has:
- Realistic overlapping distributions (no leakage)
- Hard fraud cases (`hard_case=1`)
- `noise_hotspots`, `dct_score`, `compression_mismatch` correctly overlapping between classes

---

## Cold Start & Fallback Behaviour

| Scenario | Behaviour |
|---|---|
| `ml_scorer_image.pkl` not present | `ml_scorer.predict()` returns `None` → full heuristic path used — zero regression |
| Model loaded, prediction fails | Exception caught → fallback to heuristic, log warning |
| Score from ML differs significantly from heuristic | Both scores included in JSON output for audit trail |
| First run (day 1) | Heuristic only; run `train_ml_scorer.py` once to activate ML |

---

## Evaluation Targets (must pass before saving model)

| Metric | Minimum target | Target |
|---|---|---|
| ROC AUC (all rows) | 0.80 | > 0.90 |
| F1 Score (tampered class) | 0.75 | > 0.85 |
| Accuracy on hard_case=1 | 0.55 | > 0.65 |
| No single feature importance > 0.50 | Required | — |

The last rule guards against a new leakage sneaking through — if one feature dominates at >50% importance, it's a leakage signal and training is aborted.

---

## SHAP Integration (Phase 2 — after model confirmed good)

The expert notebook (`forensic_xgboost_shap_notebook.ipynb`) contains the SHAP summary plot and waterfall explanation code. In Phase 2 this can be wired into the Forensic Scan UI to show:
- **Which features contributed most** to the score for this specific document
- **Why** it was flagged (e.g. "ELA contributed +18 points, DCT contributed +12 points")

This is explicitly **Phase 2** — not part of the current implementation scope.

---

## Testing

After implementation, add to `tests/`:

```
tests/test_ml_scorer.py
  - test_feature_vector_shape: 11 floats returned for a valid layers dict
  - test_feature_vector_all_zeros: empty layers → all zeros, no crash
  - test_predict_returns_none_when_no_model: delete pkl → returns None
  - test_predict_range: score always 0–100
  - test_heuristic_fallback: when ml returns None, final score == heuristic score
```

---

## Implementation Order

1. Fix `collect_training_samples.py` (clamp clone_ratio, drop leaky columns from feature set)
2. Rebuild `training_data_image.csv` with fixed values
3. Create `ml_scorer.py` (feature extractor + predict + train)
4. Create `train_ml_scorer.py` script — run it, confirm ROC AUC target met
5. Modify `image_forensics_detect.py` (`_compute_score` integrates ML)
6. Modify `forensic_scan.py` (add scoring method badge)
7. Modify `forensics_utils.py` (pass scoring_method through payload)
8. Write `tests/test_ml_scorer.py`
9. Run full test suite: `python -m pytest tests/ -q --tb=short`

---

## Files NOT to change

- `pdf_forensics_detect.py` — PDF heuristic unchanged (no PDF training data)
- `tamper.py` — structural/content signals unchanged
- All UI pages except `forensic_scan.py` and `forensics_utils.py`
- `api.py` — the forensic scan API endpoint output shape is unchanged (scoring_method is additive)
