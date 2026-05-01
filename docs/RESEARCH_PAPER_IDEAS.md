# Research Paper Ideas — BaseTruth

> Status: Draft brainstorm — revisit before submission.
> Last updated: April 2026.

---

## Overview

Three publishable contributions have been identified from the BaseTruth codebase. They are listed in order of recommended priority.

---

## Paper 1 (Recommended First) — 4-Class Forensic Taxonomy for Laundering-Resistant Document Fraud Detection

### Proposed Title

**"TAMPERED-DERIVED: A 4-Class Forensic Taxonomy and XGBoost Classifier for Laundering-Resistant Financial Document Fraud Detection"**

### Core Insight — The Laundering Attack

When a fraudster forges a document and then re-saves it as JPEG, the re-compression cycle **washes out Error Level Analysis (ELA) and copy-clone detection signals**, making the forgery appear clean to a binary genuine/forged classifier. This is an **active evasion technique** — essentially a laundering step that costs the attacker nothing.

Prior work in document forensics uses binary (genuine/forged) or at most 3-class taxonomies. Neither catches this.

### The 4-Class Taxonomy

| Class | Label | Meaning |
|---|---|---|
| 0 | ORIGINAL | Phone-fresh genuine document — no re-save, strongest authentic signals |
| 1 | ORIGINAL-DERIVED | Save-as copy of a genuine document — still authentic, but carries an extra JPEG re-compression cycle |
| 2 | TAMPERED | Directly manipulated/forged document — strong ELA and clone signals |
| 3 | TAMPERED-DERIVED | Save-as copy of a tampered document — forensic "laundering"; ELA/clone signals are softer because re-compression partially masks the edit |

The "derived" split on the genuine side serves a second purpose: it prevents false positives. Users legitimately re-save genuine documents (scan-to-email, WhatsApp share, etc.). A binary model trained only on phone-fresh originals would incorrectly flag these as suspicious due to elevated `dct_comb_ratio`.

### Model Architecture

- XGBoost 4-class multiclass (`objective='multi:softprob', num_class=4`)
- 19-feature vector extracted from 11 forensic engine layers (ELA × 4 sub-signals, DCT, clone, metadata, entropy, noise hotspots, font, AI-artifact, saturation, edge)
- Fraud score: `(p[TAMPERED] + p[TAMPERED-DERIVED]) × 100` — backward-compatible with the existing 0-100 score bar
- Pipeline: `SimpleImputer → XGBClassifier`; cold-start safe (falls back to heuristic if model file absent)

### Training Data Philosophy

Trained **exclusively on real documents** — no synthetic data. Rationale documented in `fraud_model/scripts/run_ml_pipeline.py`:

> "In fraud detection, models trained on fake data learn fake patterns — leading to false confidence on real documents. Every real document you scan and label is worth far more than thousands of generated rows."

- Training set: 220+ real-world financial documents (payslips, bank statements, identity documents)
- 8-step pipeline: collect features per class (labels 0–3) → merge CSVs → train image model → train PDF model

### Measured Performance

| Metric | Value |
|---|---|
| ROC AUC | 0.9849 |
| Overall Accuracy | 93.82% |
| Overall F1 | 93.81% |
| Hard-case Accuracy | 94.69% |
| Hard-case F1 | 97.27% |

Dominant feature: `clone_ratio` (66.9% importance) — not data leakage; it is an expert-designed signal that directly measures pixel-level copy-paste repetition.

### Explainability — SHAP Feature Contributions

At inference time, XGBoost tree SHAP (`predict(dmat, pred_contribs=True)`) is used to compute per-feature contributions without an external `shap` package. The UI renders a horizontal bar chart (red = toward TAMPERED, green = toward GENUINE) for every scan. This means each verdict is explainable to a non-technical reviewer in plain English.

### Heuristic Fallback

When the trained model file is absent the engine falls back to a fixed-weight heuristic:

| Threshold | Verdict |
|---|---|
| ≥ 55 | TAMPERED |
| 30–54 | LIKELY TAMPERED |
| 15–29 | UNCERTAIN |
| < 15 | ORIGINAL |

Heuristic verdicts (`UNCERTAIN`, `LIKELY TAMPERED`) do not appear in ML mode — the model always picks one of the 4 classes.

### Suggested Sections

1. Introduction — document fraud in financial due diligence; binary classifiers as insufficient
2. Background — ELA, DCT, clone detection; JPEG re-compression as a laundering vector
3. Threat Model — attacker cost vs. detection evasion
4. 4-Class Taxonomy — design rationale; why derived classes are necessary on both sides
5. Feature Engineering — 19 signals from 11 forensic layers
6. Experimental Setup — dataset, training protocol, 5-fold CV
7. Results — per-class precision/recall/F1; confusion matrix; hard-case evaluation
8. Explainability — SHAP analysis; dominant features; failure cases
9. Ablation — binary baseline vs. 4-class; with/without derived classes
10. Conclusion

### Target Venues

- IEEE WIFS (Workshop on Information Forensics and Security) — primary
- ICDAR (International Conference on Document Analysis and Recognition)
- Pattern Recognition Letters

---

## Paper 2 — Hybrid 11-Signal Forensic Engine for Financial Document Verification

### Proposed Title

**"A Hybrid 11-Signal Forensic Engine with Dual ML/Heuristic Scoring for Financial Document Integrity Verification"**

### Core Contribution

An ensemble of 11 forensic signal layers — spanning pixel-level, structural, metadata, and AI-artifact domains — applied specifically to **financial documents** (payslips, bank statements, Form 16, identity cards), with a switchable ML/heuristic scoring layer and per-inference SHAP explainability.

Most prior document forensics papers evaluate one or two signals in isolation. This work demonstrates that combining 11 complementary signals, each targeting a different forgery vector, substantially outperforms any single-signal baseline.

### The 11 Forensic Layers

| # | Layer | What it detects | Tool |
|---|---|---|---|
| 1 | ELA (4 sub-signals) | Copy-paste and region edits from JPEG re-compression artefacts | Pillow + NumPy |
| 2 | Metadata | EXIF Software/Make/Model tags; known PDF editing tool fingerprints | Pillow + exifread |
| 3 | Entropy | Shannon entropy — uniformly low or high entropy flags generated/synthetic documents | NumPy |
| 4 | Noise consistency | Local editing leaves mismatched noise patterns at block boundaries | OpenCV + NumPy |
| 5 | DCT coefficients | Double-compression artefacts in JPEG DCT histogram | OpenCV |
| 6 | Copy-clone detection | Repeated blocks with pixel-shift matching (spliced content) | OpenCV + NumPy |
| 7 | Color correlation | Unnatural channel correlation and histogram shapes — flags AI-generated images | NumPy |
| 8 | Edge continuity | Cut-paste edges show unnatural density discontinuities | OpenCV |
| 9 | Saturation | Oversaturation patterns characteristic of AI-generated imagery | OpenCV |
| 10 | Font uniformity | Baseline alignment jitter — flags cut-and-paste text replacement | Pillow |
| 11 | AI-artifact detection | Distinctive blob and colour-blob patterns left by AI image generators | OpenCV |

A **parallel 11-layer PDF-native engine** (`pdf_forensics_detect.py`) targets digitally-created structured PDFs specifically:

| # | PDF Layer | Signal |
|---|---|---|
| 1 | Incremental updates | `%%EOF` marker count — each extra EOF = file saved after creation |
| 2 | Metadata fingerprinting | Creator/Producer field identifies editing tools; date-gap analysis |
| 3 | Font consistency | Fonts appearing only on later pages = post-creation insertion |
| 4 | Invisible text | White, zero-size, or Shadow Attack overlapping bounding boxes |
| 5 | Suspicious objects | JavaScript, embedded files, OpenAction, XFA, Launch actions |
| 6 | Content consistency | Page count, blank pages, text-density variance |
| 7 | Digital signature coverage | Signature field presence; coverage gaps indicate post-sign modification |
| 8 | Page render ELA | ELA on rasterised page 1 — detects pixel-level text replacement |
| 9 | Embedded image noise | Noise residual on images extracted from PDF streams |
| 10 | File entropy | Shannon entropy of raw PDF bytes |
| 11 | Object/XRef integrity | pikepdf object count vs. declared trailer size; ObjStm streams |

### Key Design Decisions Worth Noting

- **Dual routing**: Gemma4 classifies each uploaded document as structured PDF or scanned image, and the routing decision determines which forensic engine runs (PDF-native vs. image-based), avoiding category mismatch false signals.
- **Graceful degradation**: `_FORENSICS_AVAILABLE` guard means the engine returns safely when numpy/cv2/Pillow are absent — no crashes in minimal environments.
- **Identical output shape**: Both engines return `{scan_summary, layers}` with the same field schema, so the UI renders both without modification.

### Target Venues

- ICDAR (International Conference on Document Analysis and Recognition)
- ACM DocEng (Document Engineering)
- Forensic Science International: Digital Investigation

---

## Paper 3 — System Paper: End-to-End Financial Document Intelligence Platform

### Proposed Title

**"BaseTruth: An End-to-End Document Integrity and Identity Verification Platform for Financial Due Diligence"**

### Core Contribution

A **system paper** describing the full pipeline from document ingestion to case verdict, with emphasis on:

1. **Dual-path multimodal extraction** — structured PDFs use embedded text + page image sent together to Gemma4 (text gives exact values; image gives column layout context); scanned images use PaddleOCR spatial bounding box coordinates + image sent to Gemma4 (coordinates resolve label-to-value pairing in dense tables)

2. **Industry-specific arithmetic validation packs** — seven domain packs (payroll, banking, insurance, healthcare, compliance, invoice, mortgage) each validate arithmetic consistency, required field presence, domain-specific format rules (IFSC, UAN, GSTIN, PAN), and amount/date plausibility; designed Open/Closed: adding a new industry requires only a new module, no changes to existing files

3. **Real-time Video KYC** — WebSocket stream; InsightFace RetinaFace + ArcFace for face match; MediaPipe Eye Aspect Ratio (EAR) for blink liveness; graceful fallback to MediaPipe-only on Python 3.13+; shareable session URL allows remote KYC without an app install

4. **Human-in-the-loop two-level approval workflow** — scans are invisible to downstream screens until explicitly approved; case triage with priority, assignee, and labels; cross-document anomaly detection in bulk batches

5. **Explainable verdicts** — per-scan SHAP bar chart, plain-English layer summaries, traffic-light PDF audit reports; every verdict traceable to specific signal evidence

### Architecture Summary

```
Input document
     │
     ├─ Gemma4 lightweight classifier (document type routing)
     │
     ├─ Forensic engine (image 11-layer or PDF 11-layer)
     │
     ├─ OCR / text extraction (PaddleOCR or PyMuPDF embedded text)
     │
     ├─ LLM field extraction (Gemma4 via Ollama; cloud provider fallback)
     │
     ├─ Validation pack (domain-specific arithmetic + format checks)
     │
     ├─ ML fraud scorer (XGBoost 4-class; heuristic fallback)
     │
     └─ Persistence + case management (PostgreSQL; MinIO; REST API)
```

### Target Venues

- ACM DocEng demo/system track
- IEEE IST (International Conference on Imaging Systems and Techniques)
- UIST demo track

---

## Shared Experiment Ideas (applicable across all three papers)

- **Ablation study**: binary vs. 3-class vs. 4-class taxonomy — measure false negative rate on TAMPERED-DERIVED documents specifically
- **Laundering experiment**: take a confirmed tampered document, run it through 1, 2, 3 re-save cycles; measure how the fraud score degrades with each save for both heuristic and ML models
- **Cross-document type generalisation**: train on payslips, test on bank statements — does the feature vector generalise?
- **Feature importance stability**: retrain 10× with bootstrap samples; measure SHAP value variance per feature
- **Threshold sensitivity**: ROC curve per class; optimal operating point for financial due diligence use case (asymmetric cost: false negative = missed fraud >> false positive = extra review)

---

## Next Steps

- [ ] Choose Paper 1 as the first submission target
- [ ] Identify a concrete dataset split to publish alongside (anonymised, redacted real documents or a curated synthetic version for reproducibility)
- [ ] Write an ablation comparing binary baseline vs. 4-class on TAMPERED-DERIVED documents specifically
- [ ] Confirm venue deadlines (IEEE WIFS typically submits May–June)
- [ ] Draft abstract and introduction
