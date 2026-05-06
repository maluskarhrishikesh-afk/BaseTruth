# BaseTruth — ML Fraud Detection Pipeline

> Plain-English guide to the current ML training pipeline.

---

## The Big Picture

BaseTruth has two scoring paths:

1. **Heuristic scoring** — fixed rules from the forensic engines
2. **ML scoring** — trained binary XGBoost models for image/scanned documents and PDFs (0: ORIGINAL, 1: FAKE/EDITED)

If a trained model exists, BaseTruth uses ML.
If a model is missing or fails to load, BaseTruth falls back to the heuristic path.

---

## Where Training Happens Now

You can train from two places:

- the **ML Training Pipeline** page in the UI
- the command-line scripts in `fraud_model/scripts/`

The UI is the easiest way because it shows progress, charts, and signal guides.

---

## The Current Training Flow

### Step 1 — Put labelled files into the sample folders

BaseTruth reads from these folders:

| Data type | Folder | Meaning |
|---|---|---|
| Images | `fraud_model/sample/original_images/` | Genuine image documents |
| Images | `fraud_model/sample/original_derived_images/` | Save-as copies of genuine images |
| Images | `fraud_model/sample/tampered_images/` | Clearly tampered image documents |
| Images | `fraud_model/sample/tampered_derived_images/` | Save-as copies of tampered images |
| PDFs | `fraud_model/sample/original_pdfs/` | Genuine PDFs |
| PDFs | `fraud_model/sample/tampered_pdfs/` | Tampered PDFs |

### Step 2 — Build the training CSVs

The extraction step runs the forensic engine on every sample file and writes rows to:

- `fraud_model/data/training_data_image.csv`
- `fraud_model/data/training_data_pdf.csv`

This is what the **Data Extraction** tab does in the UI.

### Step 3 — Train the models

The training step reads those CSVs and writes model files to:

- `fraud_model/models/ml_scorer_image.pkl`
- `fraud_model/models/ml_scorer_pdf.pkl`

This is what the **Model Training** tab does in the UI.

### Step 4 — Use the trained models at runtime

The forensic scan flow loads those model files automatically when they exist.

---

## Current Feature Sets

The two models do not use the same features.

### Image model

The image model currently uses **18 signals** such as:

- ELA values
- metadata flags
- file entropy
- noise hotspots
- DCT double-compression indicators
- clone ratio
- colour anomalies
- edge density
- saturation anomalies
- font consistency signals
- AI artefact signals

### PDF model

The PDF model currently uses **17 signals** such as:

- incremental update count
- EOF marker count
- metadata anomaly score
- hidden text
- white text
- JavaScript count
- embedded files count
- signature gap score
- rendered-page ELA and noise signals
- object count
- stream entropy
- xref mismatch score
- font switch score
- OCR text-layer gap
- scanned-PDF flag
- signature flag

---

## What the UI Shows

The **ML Training Pipeline** page has three tabs:

### 1. Data Extraction

- browse the sample folders
- start extraction
- stop extraction early if needed
- review extraction charts

### 2. Model Training

- choose image model, PDF model, or both
- watch live training progress
- review metrics and charts after training

### 3. Signal Reference

- read the image signals in simple language
- read the PDF signals in simple language
- see which signals are in the currently saved model

---

## Key Files

| File | Role |
|---|---|
| `src/basetruth/ui/pages/ml_training.py` | UI page for extraction, training, and signal reference |
| `fraud_model/scripts/collect_training_samples.py` | Builds the image training CSV |
| `fraud_model/scripts/collect_training_samples_pdf.py` | Builds the PDF training CSV |
| `fraud_model/scripts/train_ml_scorer.py` | Trains the image model |
| `fraud_model/scripts/train_ml_scorer_pdf.py` | Trains the PDF model |
| `fraud_model/scripts/run_ml_pipeline.py` | End-to-end launcher for the pipeline |
| `src/basetruth/analysis/ml_scorer.py` | Runtime image ML scorer |
| `src/basetruth/analysis/ml_scorer_pdf.py` | Runtime PDF ML scorer |
| `src/basetruth/analysis/image_forensics_detect.py` | Image forensic engine + ML fallback |
| `src/basetruth/analysis/pdf_forensics_detect.py` | PDF forensic engine + ML fallback |

---

## How to Add More Training Data

1. Put the files into the correct sample folder.
2. Run **Data Extraction** from the ML Training page, or run the collect scripts directly.
3. Run **Model Training** from the ML Training page, or run the train scripts directly.

You do not need to delete old CSV rows unless you intentionally want a fresh dataset.

---

## Why There Are Two Models

Image documents and PDFs behave differently.

- Scanned images care about things like clone detection, ELA, and texture/noise mismatch.
- PDFs care about things like hidden text, incremental updates, JavaScript, and xref problems.

Keeping separate models makes the predictions more accurate and easier to explain.

---

## Safe Fallback Rule

This must stay true:

- missing model file → no crash
- model load failure → no crash
- prediction failure → heuristic fallback

---

## Bottom Line

The current ML pipeline is built around the UI page and the `fraud_model/scripts/` tools.

It supports:

- labelled sample-folder extraction
- separate image and PDF models
- live progress in the UI
- charts and plain-language signal guides
- safe fallback when models are missing
