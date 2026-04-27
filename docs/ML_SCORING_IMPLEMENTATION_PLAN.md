# ML Scoring — Current Status

This file gives a simple status view of the ML scoring system.

---

## Short Summary

BaseTruth now has live ML scoring for both:

- image and scanned-document forensics
- software-generated PDF forensics

If a model file is present, BaseTruth uses it.
If a model file is missing or loading fails, BaseTruth falls back to the rule-based score.

That keeps the app safe on a fresh machine, during local development, and during partial deployments.

---

## Current Status

| Area | Status | Simple note |
|---|---|---|
| Image ML scoring | Done | Image forensics can use the trained XGBoost model |
| PDF ML scoring | Done | PDF forensics can use the trained XGBoost model |
| Heuristic fallback | Done | The app still works when model files are missing |
| Forensic Scan badge | Done | The UI clearly shows ML vs heuristic |
| PDF leakage fix | Done | Training-only helper fields were removed from live features |
| Live PDF feature extraction | Done | Key PDF features now come from real document analysis |
| PDF tests | Done | Dedicated tests exist for the PDF scorer |
| PDF explanations in UI | Done | PDF scans now show feature-contribution bars when ML is used |
| More real training data | Future work | More labelled documents will keep improving the models |

---

## What Is Live Today

### 1. Image ML scoring

- The image forensic engine still runs its normal layers.
- Then it builds a feature vector from those layers.
- If the image model exists, the final score comes from ML.
- If not, the system falls back to the old heuristic score.

Main files:

- `src/basetruth/analysis/ml_scorer.py`
- `src/basetruth/analysis/image_forensics_detect.py`
- `fraud_model/scripts/train_ml_scorer.py`
- `tests/test_ml_scorer.py`

### 2. PDF ML scoring

- The PDF forensic engine runs its normal PDF checks.
- Then it builds a PDF feature vector.
- Then it calls `predict_pdf()`.
- If the PDF model exists, the final score comes from ML.
- If not, the system falls back automatically.

Main files:

- `src/basetruth/analysis/ml_scorer_pdf.py`
- `src/basetruth/analysis/pdf_forensics_detect.py`
- `fraud_model/scripts/train_ml_scorer_pdf.py`
- `tests/test_ml_scorer_pdf.py`

### 3. UI support

- The Forensic Scan screen shows whether the score came from ML or heuristic.
- When ML runs, the UI can also show feature-contribution explanations.
- The ML Training Pipeline page can build training CSVs, train the models, and explain the signals in plain language.

Main files:

- `src/basetruth/ui/pages/forensic_scan.py`
- `src/basetruth/ui/pages/forensics_utils.py`
- `src/basetruth/ui/pages/ml_training.py`

---

## Safe Fallback Rule

This rule must stay true:

- If an ML model file is missing, BaseTruth must not fail.
- It must fall back to heuristic scoring and continue normally.

---

## What Still Needs Work

These are useful improvements, but they are not blockers:

1. Add more real image and PDF training data.
2. Keep refining weaker PDF features such as `ocr_text_layer_gap`.
3. Keep improving the training dataset mix so the models see more real edge cases.

---

## Bottom Line

The ML scoring system is not just a plan anymore.

- Image ML scoring is live.
- PDF ML scoring is live.
- The UI shows the correct scoring source.
- Tests exist for both scorers.
- The fallback path is still safe.
