# Research Paper Ideas — BaseTruth

> Status: Active — four identified contributions, Paper 1 in progress.
> Last updated: May 2026.

---

## Overview

Four publishable contributions have been identified from the BaseTruth codebase. They are listed in order of recommended priority. Papers 1–3 cover document forensics and system architecture. Paper 4 (added May 2026) covers the Face Scan Live anti-spoofing pipeline, which is a distinct and novel contribution independent of the document fraud work.

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
Input document (static)               Live face via browser WebSocket
        │                                           │
        ├─ Gemma4 type router              ├─ Active liveness FSM
        │                                  │    (5 hold-and-return challenges)
        ├─ Forensic engine                 │
        │   (image 11-layer /              ├─ Passive anti-spoofing signals
        │    PDF 11-layer)                 │    (24-feature vector: IOD-yaw depth,
        │                                  │    replay hash, screen FFT, eye jitter,
        ├─ OCR / text extraction           │    temporal consistency, frame timing)
        │  (PaddleOCR / PyMuPDF)           │
        │                                  ├─ XGBoost binary scorer (GENUINE/SPOOF)
        ├─ LLM field extraction            │    + heuristic fallback (cold-start safe)
        │  (Gemma4 via Ollama)             │
        │                                  ├─ GDPR-compliant video recording
        ├─ Validation pack                 │    (SUSPICIOUS/DEEPFAKE only; H.264 MP4;
        │  (7 domain packs)                │    configurable retention TTL)
        │                                  │
        ├─ ML fraud scorer                 └─ face_scan_live_results table
        │  (XGBoost 4-class)
        │
        └─ Persistence + case management (PostgreSQL; MinIO; REST API)
```

### Target Venues

- ACM DocEng demo/system track
- IEEE IST (International Conference on Imaging Systems and Techniques)
- UIST demo track

---

## Paper 4 — Active + Passive Hybrid Liveness Detection for Financial KYC

### Proposed Title

**"Hold-and-Return: A 24-Signal Passive Anti-Spoofing Framework Fused with Active Liveness Challenges for Remote Financial KYC"**

### Core Insight

Existing liveness papers focus on either:

- **Active challenges** (instruction-following gestures) — gameable with prepared replay tooling once the attacker knows the challenge sequence, and inconclusive when the user hesitates or stumbles.
- **Passive signals** (texture maps, rPPG, depth sensors) — most assume depth cameras or controlled lighting unavailable on commodity smartphone/laptop webcams.

This work shows that a set of **passive signals derivable entirely from a standard webcam JPEG stream** — when correctly fused — achieves reliable liveness detection without any additional hardware, while the active challenge component provides an independent evidence channel that passive signals cannot simultaneously fake.

### Novel Contributions

#### 1. IOD-Yaw Geometric Depth Cue

The inter-ocular distance (IOD) measured from 5-point face landmarks correlates with head yaw in a real 3D face but is independent of yaw in a flat representation (photo print, screen display, mask). The signal is computed as the Pearson correlation between per-frame IOD and yaw across the session:

```
iod_yaw_correlation = pearsonr(iod_sequence, yaw_sequence)
```

Real humans: correlation typically −0.40 to −0.99 (IOD shrinks as face turns away from camera).
Flat photo / printed mask: correlation near 0 (IOD stays constant regardless of paper tilt).
Plastic doll measured at −0.12: scores ~77/100 risk — correctly above the SUSPICIOUS threshold.

This requires no depth sensor, no infrared, no second camera. It is derived from landmark geometry already available whenever a face detector runs.

#### 2. Hold-and-Return Challenge Protocol

Prior work uses either a single head-turn gesture or a binary present/absent detection. The hold-and-return protocol introduced here has three stages:

1. **Threshold crossing**: yaw must exceed `|0.16|` (normalised by IOD) — filters out micro-movements that a rigid mask can produce without the head actually turning.
2. **Sustained hold** (10 consecutive frames ≈ 1 s at 10 FPS): the user must *maintain* the turned position, not just briefly cross it. Replay tools injecting single frames cannot satisfy this.
3. **Stability gate reset**: after hold completes, the system resets to a fresh stability gate before accepting the next challenge — making it impossible to exploit a drifting replay that happens to cross the threshold twice.

Wrong-action self-correction (user reverses and retries) is explicitly allowed and tracked as a feature (`wrong_action_count`) so that hesitation by a genuine user does not inflate the risk score.

#### 3. Screen Replay Detection via FFT Moiré Grid

When a face video is filmed from a screen (monitor, smartphone, TV), the screen pixel grid introduces a regular moiré pattern in the captured frames. The signal:

```
mean_fft_grid_peak = mean amplitude of dominant non-DC frequency bins
                     in the 2-D FFT of each frame's luminance channel
```

This fires on filmed screens regardless of the video content and is complementary to repeat-frame hashing — which detects software replay tools but misses filmed-screen attacks where each frame is genuinely different.

#### 4. Self-Labeling Training Pipeline

Most liveness papers train on dedicated spoofing databases (e.g., CelebA-Spoof, LCC FASD). This work presents an alternative for production deployments where labelled data does not exist at launch:

1. Cold start: heuristic formula runs; `scoring_method = "heuristic"` in every result JSON.
2. Every completed session appends a row to `training_data_face_scan_live.csv` with `label=-1` (unconfirmed).
3. Operators label confirmed genuine and spoof sessions → rows updated to `label=0` or `label=1`.
4. When enough labelled rows accumulate, `train()` runs 5-fold stratified CV and saves the model only if ROC AUC ≥ 0.75.
5. On next server restart the model activates automatically; `scoring_method = "ML"` from that point.

This is a genuine deployment contribution: liveness systems that require a pre-labelled dataset before going live are impractical for small financial institutions building their first KYC pipeline.

#### 5. GDPR-Compliant Forensic Video Recording

Completed SUSPICIOUS, DEEPFAKE, or INCONCLUSIVE sessions are assembled from the per-frame JPEG buffer into an H.264 MP4 and uploaded to object storage. GENUINE sessions are **not** recorded. Rationale: biometric video is Article 9 GDPR special-category data — recording it without investigative necessity violates the data minimisation principle. The recording TTL is operator-configurable (`FACE_SCAN_RECORD_VIDEO_RETENTION_DAYS`). This design decision and its legal basis are worth a short discussion section in the paper.

### The 24-Feature Vector

| Group | Features | Anti-spoofing target |
|---|---|---|
| Temporal consistency | yaw_jerk, pitch_jerk, nose_jitter, temporal_consistency_score | Rigid objects / smoothed replay |
| Replay detection | repeat_frame_score, flicker_score, brightness_instability | Software replay tools |
| Eye micro-jitter | mean_eye_jitter | Static images / masks (eyes never move) |
| 3D depth geometry | iod_yaw_correlation | Flat photos, printed masks, screens |
| Screen moiré | mean_fft_grid_peak | Filmed-screen attacks |
| Frame timing | interval_cv | Metronomic replay tools |
| Session metadata | observed_fps, frame_drop_rate | Frame injection / rate manipulation |
| Face quality | mean_face_area_ratio, blur_risk, brightness_risk | Occlusion, darkness, extreme distance |
| Active liveness | wrong_action_count, challenge_count | Challenge bypass attempts |
| Face tracking | frames_without_face | Occlusion attacks |
| Device flag | virtual_camera_suspected | OBS / virtual camera injection |
| Tier-1 ML signals | head_velocity_variance, blink_duration_ms, challenge_reaction_latency_ms, mean_landmark_confidence | Human micro-behaviour baseline |

### Model Architecture

- XGBoost binary classifier (`objective='binary:logistic'`)
- Pipeline: `SimpleImputer(strategy='median') → XGBClassifier`
- Training gate: saves only if 5-fold stratified CV ROC AUC ≥ 0.75
- Inference: risk_score = spoof probability × 100
- Explainability: XGBoost tree SHAP (no external `shap` package)
- Cold-start: `predict()` returns `None` → heuristic runs unchanged; `scoring_method` field distinguishes the two in the result JSON

### Suggested Sections

1. Introduction — webcam-only liveness; gap between passive-only and active-only approaches
2. Background — existing active challenge protocols; passive signal survey; smartphone webcam constraints
3. Threat Model — replay attack taxonomy: software replay, filmed-screen, printed photo, 3D mask, virtual camera injection
4. Hold-and-Return Challenge Protocol — FSM design; threshold vs. hold vs. reset stages; wrong-action self-correction
5. 24-Signal Passive Feature Set — per-signal design rationale; which attacks each targets
6. IOD-Yaw Geometric Depth Cue — derivation; empirical results (real face vs. photo vs. doll vs. mask)
7. XGBoost Fusion Model — training pipeline; self-labeling bootstrapping; cold-start guarantee
8. GDPR-Compliant Video Recording — design rationale; data minimisation; forensic value
9. Experimental Evaluation — per-signal ablation; binary ML vs. heuristic baseline; held-out attack types
10. Conclusion

### Target Venues

- IEEE BTAS (Biometrics: Theory, Applications and Systems) — primary
- IJCB (International Joint Conference on Biometrics)
- IEEE Access (open access; system/application paper track)
- IET Biometrics

---

## Shared Experiment Ideas (applicable across all three papers)

- **Ablation study**: binary vs. 3-class vs. 4-class taxonomy — measure false negative rate on TAMPERED-DERIVED documents specifically
- **Laundering experiment**: take a confirmed tampered document, run it through 1, 2, 3 re-save cycles; measure how the fraud score degrades with each save for both heuristic and ML models
- **Cross-document type generalisation**: train on payslips, test on bank statements — does the feature vector generalise?
- **Feature importance stability**: retrain 10× with bootstrap samples; measure SHAP value variance per feature
- **Threshold sensitivity**: ROC curve per class; optimal operating point for financial due diligence use case (asymmetric cost: false negative = missed fraud >> false positive = extra review)

---

## Next Steps

- [x] Choose Paper 1 as the first submission target (decided)
- [ ] Identify a concrete dataset split to publish alongside Paper 1 (anonymised, redacted real documents or a curated synthetic version for reproducibility)
- [ ] Write an ablation comparing binary baseline vs. 4-class on TAMPERED-DERIVED documents specifically
- [ ] ⚠️ **IEEE WIFS deadline is May–June 2026 — submit Paper 1 abstract now**
- [ ] Draft Paper 1 abstract and introduction
- [ ] Collect labelled genuine/spoof Face Scan Live sessions to train the Paper 4 XGBoost model
- [ ] Run IOD-yaw correlation experiment on a held-out set of printed photos and filmed-screen attacks to generate Table 1 for Paper 4
- [ ] Decide whether Paper 4 targets IEEE BTAS (November 2026 deadline) or IJCB
- [ ] Update Paper 3 system description to include the Face Scan Live pipeline once video recording is in production
