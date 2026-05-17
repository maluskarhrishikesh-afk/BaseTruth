# A Hybrid 11-Signal Forensic Engine with Dual ML/Heuristic Scoring for Financial Document Integrity Verification

**Authors:** Hrishikesh Maluskar  
**Date:** May 2026  
**Project:** BaseTruth — AI-Powered Document Fraud Detection and Identity Verification  
**Repository:** https://github.com/maluskarhrishikesh-afk/BaseTruth  
**Status:** First Draft (in progress)

---

## Abstract

Document fraud in financial due diligence — forged payslips, altered bank statements, fabricated offer letters — is predominantly detected today by one or two forensic signals applied in isolation, typically Error Level Analysis (ELA) or metadata inspection. We present a hybrid forensic engine that applies **11 complementary signal layers** in parallel to every submitted document, spanning pixel-level, structural, metadata, and frequency-domain domains. The image engine targets scanned photographs and rasterised documents; a parallel **PDF-native engine** applies a separate 11-layer analysis to digitally-created PDFs using structural signals unavailable in rasterised images. Both engines share an identical output schema so a single scoring and verdict layer operates across both document types. A **dual-path routing** mechanism — powered by an LLM document classifier — dispatches each upload to the appropriate engine, preventing category mismatch false positives. Scoring is provided by an XGBoost 4-class classifier (the TAMPERED-DERIVED taxonomy described in the accompanying Paper 1) with a fixed-weight heuristic fallback for cold-start deployments. Every verdict is accompanied by plain-English per-layer summaries and per-feature SHAP contributions, making decisions inspectable by non-technical compliance reviewers. We describe each signal layer, its mathematical basis, its financial-document calibration thresholds, and its contribution to overall detection accuracy. We further show that the ensemble of 11 signals substantially outperforms any single-signal baseline, particularly on the hard case of laundered (re-saved) forgeries where ELA alone is insufficient. Our architecture's focus on localised character-level and structural signals directly addresses the Extreme Region Imbalance property of real document forgeries, where tampered content typically accounts for only 0.3%–4.17% of total document pixels (Du et al., 2026). We additionally discuss a lightweight ten-example threshold calibration protocol that recovers the AUC–F1 gap observed in zero-shot deployment on unseen document types.

**Keywords:** document forensics, ELA, DCT double-compression, copy-move detection, metadata fingerprinting, PDF structural analysis, AI-generated document detection, XGBoost, SHAP explainability, financial KYC, hybrid forensic engine, domain threshold calibration, attention-based feature fusion, DOCFORGE-BENCH

---

## 1. Introduction

Financial due diligence processes — background verification for employment, loan applications, insurance underwriting, rental applications — depend critically on the integrity of supporting documents. A candidate submitting a forged payslip, an altered bank statement, or a fabricated offer letter can deceive purely manual review with a well-executed forgery. The scale of digital document submission has outpaced the capacity for manual forensic examination.

The forensic literature on document tampering detection has primarily pursued two paths:

1. **Single-signal deep analysis**: developing increasingly sophisticated variants of one technique — most commonly Error Level Analysis (ELA) or deep convolutional features — and evaluating them on benchmark datasets.

2. **Binary classification**: treating the problem as genuine/forged and training end-to-end classifiers, sometimes on synthetic forgery datasets that do not capture the variety of real-world attack types.

Both approaches have known blind spots. ELA is defeated by JPEG laundering (re-saving the forgery washes out the compression artifact). Binary classifiers trained on one forgery type fail on others. Deep CNN features require large labelled datasets and behave as black boxes.

A third, less explored approach is the **forensic ensemble**: combine multiple independently-motivated signals, each targeting a different forgery mechanism, into a unified scoring pipeline. The ensemble has the property that an attacker who defeats one signal is unlikely to defeat all signals simultaneously — the forger would need to simultaneously erase ELA artefacts, restore sensor noise consistency, clear edit-software fingerprints from metadata, correct DCT histogram anomalies, eliminate clone-match patterns, and suppress AI generation artefacts, all without making the document visually obvious. In practice, no known off-the-shelf forgery tool performs all of these concealments.

This paper documents the forensic ensemble implemented in BaseTruth, which applies 11 complementary signal layers to every uploaded document. The system's specific distinguishing characteristics versus prior work are:

1. **Domain specificity**: the 11 signals are calibrated for **financial documents** — payslips, bank statements, Form 16, offer letters, identity cards — rather than generic natural photographs. Thresholds account for JPEG compression artefacts introduced by print-scan-photograph cycles common in financial document submission workflows.

2. **Dual-engine routing**: financial documents arrive as either digitally-created PDFs (from payroll software, banking portals) or as scanned photographs. These have fundamentally different forensic properties. A separate 11-layer PDF-native engine targets structural signals (incremental updates, font consistency, invisible text, JavaScript, xref integrity) that are only meaningful for digitally-created PDFs. An LLM routes each document to the appropriate engine, preventing the category mismatch false positives that arise when image-domain ELA is applied to a crisp digital PDF.

3. **Identical output schema**: both engines produce `{scan_summary, layers}` with the same field structure, so the scoring, verdict, UI, and database storage layers operate on both document types without modification.

4. **Graceful degradation**: all 11 layers are wrapped in independent exception handlers and the entire forensics module is guarded by a `_FORENSICS_AVAILABLE` flag. A missing dependency (`numpy`, `cv2`, `PIL`) makes the module return an UNAVAILABLE status cleanly rather than crashing the application.

5. **Plain-English explainability**: every layer produces a `plain_english` field — one to three sentences that explain the technique to a non-technical compliance reviewer. This is not a post-hoc summarisation but a core output field computed alongside the numeric metrics.

The remainder of this paper is structured as follows. Section 2 reviews related work. Section 3 describes the threat model and the financial document submission context. Section 4 presents the 11 image forensic layers in detail. Section 5 presents the 11 PDF forensic layers. Section 6 describes the dual-engine routing logic. Section 7 covers the scoring pipeline. Section 8 discusses the graceful degradation architecture. Section 9 evaluates per-signal discriminative value and ensemble performance. Section 10 discusses limitations and future work.

---

## 2. Background and Related Work

### 2.1 Error Level Analysis

ELA was introduced by Krawetz (2007) as a qualitative visual aid and was subsequently formalised for automated classification by Farid (2009) and others. The technique re-compresses a JPEG at a known quality level and measures per-pixel or per-block differences. Regions with different compression histories — e.g., content pasted from another JPEG — exhibit elevated or suppressed ELA response relative to the surrounding image.

ELA is the most widely deployed single forensic signal in commercial document verification tools. Its principal limitation is sensitivity to JPEG re-compression: when a forged document is saved as JPEG a second time, the ELA response of the forged region equalises with the surrounding content and the signal vanishes. This is the "laundering attack" described in the companion Paper 1, which motivates the 4-class TAMPERED-DERIVED taxonomy.

### 2.2 DCT-Based Double-Compression Detection

The JPEG standard compresses each 8×8 pixel block independently using the Discrete Cosine Transform (DCT). When a JPEG is re-saved as JPEG, the original 8×8 grid structure introduces a characteristic "comb" pattern in the DCT coefficient histogram — local minima at multiples of the quantisation step size — because the two quantisation grids are not aligned. Popescu and Farid (2004) formalised this as a tamper detection signal. Fu et al. (2006) showed that the comb ratio (local minima / local maxima in the AC coefficient histogram) reliably distinguishes once-compressed from twice-compressed images even when the global quality levels are similar.

### 2.3 Copy-Move Detection

Copy-move fraud — copying a region of the image and pasting it elsewhere (e.g., cloning a clean background over a signature or stamp) — is detectable through self-matching of feature descriptors. Fridrich et al. (2003) introduced the first systematic approach using block DCT. Modern methods use scale-invariant feature descriptors (SIFT, SURF) for matching, as used in this work, which are robust to JPEG compression artefacts and modest scale changes. Amerini et al. (2011) provide a comprehensive evaluation.

### 2.4 Metadata Forensics

Camera metadata (EXIF for JPEG, tEXt chunks for PNG) records the hardware and software provenance of an image. Absence of camera Make/Model indicates the image was not taken by a camera (consistent with a screenshot or synthetic image). Presence of photo-editing software names (Photoshop, GIMP, Canva, etc.) in the Software field indicates the image was opened in an editor after capture. Date inconsistencies between DateTimeOriginal and ImageDateTime indicate a post-capture save. Rosenholtz et al. (2010) survey the forensic value of EXIF metadata in authentication contexts.

For PDFs, the Information Dictionary carries Creator (application that created the source document), Producer (PDF library), and creation/modification timestamps. Identifying online PDF editing tools (ilovepdf, smallpdf, sejda) in these fields is a well-known practitioner signal that is under-documented in the academic literature.

### 2.5 Frequency Domain AI Generation Detection

Modern image generators (Stable Diffusion, Midjourney, DALL-E) based on convolutional neural networks and diffusion processes introduce spectral artefacts in the high-frequency domain. Corvi et al. (2023) demonstrate that the checkerboard artefacts introduced by CNN upsampling operations are detectable via 2D FFT spectral analysis. Frank et al. (2020) show similar results for GAN-generated faces. The specific signal used in this work — the ratio of the maximum high-frequency magnitude to its mean — is a simplified version of the Frank et al. approach, adapted for speed in an inference-time pipeline.

### 2.6 PDF Structural Analysis

The academic literature on PDF document forensics is significantly thinner than image forensics. Practitioner tools (pdfid, pdf-parser by Didier Stevens) enumerate suspicious PDF objects (JavaScript, OpenAction, XFA) but focus on malware delivery rather than content tampering. Ismail et al. (2019) propose incremental-update detection for detecting field-level modifications to form PDFs. Font consistency analysis for detecting post-creation text insertion has not, to the authors' knowledge, been documented as a published contribution.

### 2.7 Ensemble Approaches

Bayar and Stamm (2016) trained a CNN that learned a constrained high-pass filter as its first layer, achieving strong performance across multiple tampering types. However, their approach requires training on labelled tampered images — a significant constraint for financial document operators who cannot accumulate large labelled datasets of real forgeries. The ensemble approach presented here requires only the binary/4-class fraud label at the document level, which can be accumulated through the self-labeling pipeline described in the companion ML scoring paper.

### 2.8 2026 Benchmarks, Attention Fusion, and Self-Supervised Forensics

The DOCFORGE-BENCH benchmark (Du et al., 2026) is the most comprehensive zero-shot document forgery evaluation published to date, covering 17 document types and 9 forgery strategies. Its key architectural finding directly validates choices made in this work: financial document forgeries exhibit **Extreme Region Imbalance**, with the tampered region accounting for only 0.3%–4.17% of total document pixels. This means global pixel-accuracy metrics dramatically over-report performance, and detectors must be sensitive to small, localised alterations — precisely the design goal of the font consistency, clone detection, and PDF text-injection layers described in Sections 4.10, 4.6, and 5.4 respectively.

The benchmark also documents a critical **AUC–F1 gap** in zero-shot deployment: models achieving > 90% AUC on in-distribution test sets regularly drop to F1 < 0.5 on unseen document types from different institutions, because detection thresholds calibrated on payslips are miscalibrated for bank statements from a different source. Our threshold calibration protocol (Section 7.4) directly addresses this challenge.

In the feature-fusion literature, Multi-head Attention Networks (MHAN) have recently emerged as a competitive alternative to XGBoost for combining heterogeneous forensic feature vectors. Khiaonarong and Shanyuan (2026) report 99%+ detection accuracy on financial fraud scenarios using attention-based fusion, with the key advantage that MHAN models pairwise signal dependencies — e.g., the interaction between DCT comb ratio and clone ratio, which co-occur more frequently in laundered forgeries — that XGBoost treats as independent scalar features. The tradeoff is interpretability: XGBoost with SHAP produces per-feature reason codes that are operationally required for regulatory compliance (Section 7.2), while attention weight distributions over raw feature vectors are significantly harder to translate into audit-ready language. We treat MHAN fusion as a priority future investigation and retain XGBoost as the production scorer pending this evaluation.

Self-supervised forensic learning has also advanced significantly in 2026. Sheng et al. (2026) propose a CLIP-based framework that uses pseudo-label prediction and image-text alignment to learn tampering feature representations without manually annotated forgery examples. This approach mirrors the self-labeling pipeline used in this work: both methods address the cold-start constraint common to financial document forensics operators. The CLIP approach offers a path to richer semantic feature representations but requires substantially more compute and inference latency than the 19-dimensional tabular feature vector consumed by XGBoost.

---

## 3. Threat Model and Document Context

### 3.1 Financial Document Submission Context

In a typical financial background check workflow, a candidate or applicant submits digital copies of supporting documents via a web portal or email:

- **Payslips** (PDF generated by payroll software, or a photograph of a printed payslip)
- **Bank statements** (PDF exported from online banking, or a scanned photograph)
- **Form 16** / tax certificates (PDF from employer or tax authority)
- **Offer letters** (PDF from employer)
- **Identity documents** (scanned photograph of Aadhaar, PAN card, passport)

The submission channel introduces a characteristic forensic fingerprint: many genuine documents arrive as photographs taken on a smartphone (introducing JPEG compression, sensor noise, and sometimes lens distortion) or as PDFs downloaded from a payroll portal (digitally created, with perfect fonts and crisp geometry). A document that arrives as a low-resolution JPEG with inconsistent noise but claims to be a genuine scanned bank statement is immediately suspect.

### 3.2 Attack Taxonomy

We consider the following attack types, approximately in order of sophistication:

| Attack | Description | Primary Detection Signal |
|---|---|---|
| **A1: Edit-and-screenshot** | Edit numbers in a photo editor; screenshot | ELA, metadata (software tag), AI artifact |
| **A2: Print-and-photograph** | Print an edited document; photograph | Noise inconsistency, font mismatch |
| **A3: JPEG laundering** | Re-save after editing to wash out ELA signal | DCT comb, clone detection, 4-class taxonomy |
| **A4: Template substitution** | Replace a real document's content with a forged template | Font inconsistency, color anomaly |
| **A5: AI generation** | Generate a fake payslip with an AI image generator | AI artifact (FFT grid), saturation anomaly |
| **A6: PDF field editing** | Open a PDF in an editor, type new numbers, save | Incremental update (%%EOF count), metadata, font |
| **A7: PDF overlay injection** | Add a transparent text layer over the original content | Invisible text, shadow attack detection |
| **A8: PDF JavaScript manipulation** | Use JavaScript to display different values on render | Suspicious object detection |
| **A9: Post-signature modification** | Modify a PDF after it was digitally signed | Digital signature ByteRange gap |

No single forensic signal covers all nine attack types. The ensemble is designed so that at least two signals fire on each attack type, providing redundant detection.

### 3.3 Out of Scope

- **Semantic fraud**: a genuine-looking, unmodified document that contains false information (e.g., a real payslip from a different employer). Document forensics cannot detect this; it requires cross-reference validation (covered by the validation packs in Paper 3).
- **Perfect forgeries produced with access to the originating system**: an attacker with access to the payroll software producing the original documents.

---

## 4. Image Forensic Engine — 11 Layers

All 11 layers are implemented in `src/basetruth/analysis/image_forensics_detect.py`. They are executed in parallel at inference time. Each layer:
- Takes the image path as its sole input
- Returns a `{name, status, plain_english, metrics}` dict
- Catches its own exceptions — a layer failure returns `status: "ERROR"` and never propagates
- Degrades gracefully when `numpy`, `cv2`, or `Pillow` are unavailable

### 4.1 Layer 1 — Error Level Analysis (ELA)

**Technique**: Re-save the image as JPEG at quality 75 (a known reference quality). Subtract the re-saved image from the original pixel-by-pixel. In the resulting difference image, regions with a different prior compression history — i.e., content that was pasted from another image — produce anomalously elevated or suppressed differences relative to the surrounding pixels.

**Algorithm**:
1. Recompress the original to a temporary buffer at quality $q = 75$
2. Compute the per-pixel absolute difference between original and recompressed: $\Delta = |I_{\text{orig}} - I_{q}|$
3. Compute the mean difference $\bar{\Delta}$ across the image
4. Divide the image into $32 \times 32$ pixel blocks; count the fraction of blocks whose mean exceeds $2.5 \times \bar{\Delta}$

**Decision threshold**: if more than 5% of blocks are "hot", status = SUSPICIOUS.

**Financial document calibration**: The $2.5 \times$ multiplier and 5% block threshold were calibrated for financial documents, which are frequently JPEG-compressed multiple times during legitimate submission workflows (bank portal → smartphone download → re-upload). A more sensitive threshold would produce high false-positive rates on legitimate documents that have undergone one innocent re-save.

**ELA sub-signals used by the ML model**: `mean_ela`, `max_ela`, `std_ela`, `suspicious_block_ratio`. These four values are fed as separate features to the XGBoost model, rather than using the binary SUSPICIOUS/CLEAN flag. This allows the model to detect partial ELA elevation (consistent with TAMPERED-DERIVED documents where laundering attenuated but did not fully eliminate the signal).

**Limitation**: Defeated by JPEG laundering (saving the forgery a second time). This motivates the 4-class taxonomy and the companion DCT double-compression layer.

### 4.2 Layer 2 — Metadata / EXIF Analysis

**Technique**: Inspect the hidden information embedded in the image file — EXIF tags for JPEG, tEXt chunks and ancillary data for PNG.

**Signals checked**:

| Tag | Suspicious condition |
|---|---|
| `Image Software` | Contains a known photo editor name |
| `Image Make` / `Image Model` | Absent (no camera hardware = screenshot or synthetic) |
| `EXIF DateTimeOriginal` | Absent (stripped by editing tools) |
| `DateTimeOriginal` vs `ImageDateTime` | Different (post-capture edit and re-save) |
| Complete metadata absence | All tags stripped (common after online editing tools) |

**Software detection list** (50+ entries): Photoshop, GIMP, Lightroom, Canva, PicsArt, PixelMator, Snapseed, Affinity, Adobe, ImageMagick, OpenCV, Pillow, and others.

**Scoring**: each suspicious flag adds 10 points to the heuristic score. Multiple flags compound. The metadata layer is the only layer in the ensemble that can fire on a genuine document with zero visual evidence of tampering — a document opened in Photoshop to adjust brightness and then re-saved will have a clean visual appearance but a suspicious Software tag.

**Financial document context**: scanned identity documents submitted via WhatsApp or email frequently lose EXIF data entirely (WhatsApp strips metadata). Absence of metadata is therefore treated as a single medium-weight flag rather than a definitive TAMPERED signal, and its contribution is reduced in the ML feature vector relative to the heuristic.

### 4.3 Layer 3 — File Entropy

**Technique**: Compute the Shannon entropy of the raw byte stream of the file:

$$H = -\sum_{b=0}^{255} p_b \log_2 p_b$$

where $p_b$ is the probability of byte value $b$ in the file.

**Interpretation**: JPEG compression produces near-maximally random byte patterns (entropy ≈ 7.8–8.0 bits). Files that have been repeatedly re-encoded, or that contain large uncompressed regions inserted by editing tools, may show entropy below 7.8.

**Decision threshold**: entropy < 7.8 → SUSPICIOUS. Status: CLEAN otherwise.

**Role in the ensemble**: on its own, entropy is a weak signal — many legitimate documents score below 7.8. Its value is as a supporting signal that, when combined with other flags, increases the evidence weight. The ML model treats it as a continuous feature rather than a binary flag.

### 4.4 Layer 4 — Noise Residual Analysis

**Technique**: Every camera sensor adds a characteristic high-frequency noise pattern to every image it captures. Because the noise comes from the hardware, it should be statistically uniform across the entire image. Content spliced from a different source will carry that source's noise fingerprint, producing an anomalous region at the boundary.

**Algorithm**:
1. Convert to grayscale
2. Compute the Gaussian-blur high-frequency residual: $R = |G - \text{GaussianBlur}(G, 5 \times 5)|$
3. Divide $R$ into $64 \times 64$ pixel tiles
4. Compute per-tile Coefficient of Variation: $\text{CV}_i = \sigma_i / (\mu_i + \varepsilon)$
5. Count tiles where $\text{CV}_i > 2 \times \bar{\text{CV}}_{\text{global}}$

**Decision threshold**: if more than 10% of tiles are "hotspot" tiles, status = SUSPICIOUS.

**Financial document context**: physically printed and re-photographed documents have higher and more variable noise than digitally submitted PDFs. The 10% threshold is set conservatively to tolerate this legitimate variation.

### 4.5 Layer 5 — DCT Double-Compression Analysis (JPEG only)

**Technique**: When a JPEG is saved the first time, the image is divided into 8×8 blocks and each block is quantised using the DCT. When the JPEG is edited and re-saved, the 8×8 block grid of the re-save does not align with the block grid of the first save (unless the image coordinates happen to be a multiple of 8). The misaligned re-quantisation introduces a characteristic periodic pattern — a "comb" — in the histogram of the DCT AC coefficients.

**Algorithm**:
1. Divide the grayscale image into all valid $8 \times 8$ blocks
2. Compute the DCT of each block; collect the AC coefficients (positions 1–10 in the 1D-serialised 8×8 block, excluding DC component at position 0)
3. Build a 200-bin histogram of all AC coefficient values over $[-100, 100]$
4. Count local minima and maxima in the histogram using `scipy.signal.argrelextrema`
5. Compute the **comb ratio**: $R_c = N_{\text{min}} / (N_{\text{max}} + \varepsilon)$

**Decision threshold**: $R_c > 1.3$ → SUSPICIOUS (double-compressed).

**Note**: this layer applies to JPEG files only. PNG uses lossless compression; there is no DCT quantisation and hence no comb structure. The layer returns `status: "N/A"` for PNG inputs.

**Relationship to ELA**: ELA and DCT double-compression are complementary laundering countermeasures. ELA fires on the first re-save (which introduces ELA artefacts); DCT double-compression fires on the existence of a second save. A laundered forgery (ELA washed out by re-saving) often still carries the DCT comb signature because the two save cycles leave two quantisation grids in conflict.

### 4.6 Layer 6 — Clone / Copy-Move Detection

**Technique**: Copy-move fraud covers one region of a document with content copied from another region of the same document — for example, cloning the background texture over a signature to hide it, or repeating a legitimate number to replace a different number nearby.

**Algorithm**:
1. Extract up to 3,000 SIFT keypoints and 128-dimensional descriptors from the grayscale image
2. Match every descriptor against every other descriptor using a brute-force L2 matcher (k=3 nearest neighbours, excluding self-match)
3. A match is a "clone hit" if: descriptor distance < 120 AND spatial distance between matched keypoints > 50 pixels
4. Compute the clone ratio: $R_{\text{clone}} = N_{\text{clone}} / N_{\text{kp}}$

**Decision threshold**: $R_{\text{clone}} > 0.25$ → SUSPICIOUS.

**Role in the ML model**: `clone_ratio` is the single most important feature in the 4-class XGBoost model (SHAP importance 66.9% in Paper 1 results). A high clone ratio is strongly correlated with TAMPERED documents because copy-move is a common tool for altering financial figures (copying a legitimate portion of the document over the altered region).

**Limitation**: clone detection requires sufficient image resolution for SIFT to find keypoints. Very low-resolution photographs (< 200 × 200 pixels) may return fewer than 10 keypoints, causing the layer to return `status: "N/A"`.

### 4.7 Layer 7 — Color Anomaly Detection

**Technique**: Real financial documents have a constrained colour palette (black text on white, possibly a company logo with a few dominant colours). Digital pastes, colour fills, or annotation stamps often introduce chromatic outliers — colours that do not occur naturally in the document's dominant palette.

**Algorithm**:
1. Convert to HSV colour space
2. Build a 36-bin hue histogram over foreground pixels (V > 30, S > 15)
3. Identify the 3 dominant hue bins; define a dominant palette by allowing ±10° tolerance around each
4. Classify every pixel as anomalous if: saturation > 60 AND value > 40 AND hue not in dominant palette
5. Compute: $R_{\text{anomaly}} = N_{\text{anomaly}} / N_{\text{foreground}}$
6. Run connected-component analysis on anomalous pixels to identify blob clusters

**Decision thresholds**: $R_{\text{anomaly}} > 0.003$ → SUSPICIOUS; $R_{\text{anomaly}} > 0.01$ OR largest blob > 2,000 px → HIGHLY SUSPICIOUS.

**Financial document context**: bright stamps, coloured annotations, or vibrant signature ink over a pale document background are the most common trigger for this layer. The blob size threshold (2,000 px) targets large-area alterations rather than individual outlier pixels.

### 4.8 Layer 8 — Edge Discontinuity / Density Analysis

**Technique**: Natural photographic edges (paper texture, printed text strokes, logo boundaries) have smooth, slightly blurred profiles. Digitally drawn lines, precise rectangular overlays, or hard-cut pastes produce unnaturally sharp, perfectly straight edges with abnormally high local density.

**Algorithm**:
1. Apply Canny edge detection (thresholds 50, 150) to the grayscale image
2. Divide into $32 \times 32$ tiles; compute mean edge pixel density per tile
3. Count the fraction of tiles with density > $3 \times \bar{d}_{\text{global}}$

**Decision threshold**: high-density tile fraction > 6% → SUSPICIOUS.

**Financial document context**: borders and table lines in printed documents legitimately produce high-density tiles. The $3 \times$ multiplier is intended to distinguish the uniform high-density distribution of printed rules from the localised, spatially-isolated density spikes of a paste or annotation.

### 4.9 Layer 9 — Saturation Anomaly Detection

**Technique**: Legitimate financial documents are produced by printers and scanners that produce visually consistent, moderate saturation levels throughout. Applying a vivid colour highlight, stamp, or annotation to a specific area produces localised over-saturation that stands out statistically from the rest of the document.

**Algorithm**:
1. Extract the Saturation channel $S$ from the HSV representation
2. Compute the global mean saturation $\bar{S}$
3. Divide into $32 \times 32$ tiles; count tiles where mean tile saturation > $3 \times \bar{S}$
4. Flag sample coordinates of high-saturation tiles

**Decision threshold**: high-saturation tile ratio > 2% → SUSPICIOUS.

**Relationship to Layer 7 (Color Anomaly)**: saturation anomaly targets the intensity of a colour, while color anomaly targets the hue. A vivid version of the document's own dominant colour triggers saturation anomaly but not color anomaly. Both layers firing simultaneously is a strong ensemble signal.

### 4.10 Layer 10 — Font Consistency Analysis

**Technique**: A genuine financial document is produced by a single application in a single pass, so every character should have statistically consistent stroke width, character height, sharpness, and baseline alignment. If someone types new text (e.g., replacing a salary figure) using a different software tool, the new characters will differ in at least one of these properties.

**Algorithm**:
1. Apply adaptive thresholding to isolate text blobs from the background
2. Extract all character-sized components (height 6–12% of image height, width 2–25% of image width, area ≥ 20 px, aspect ratio 0.05–6.0)
3. For each component, compute:
   - **Stroke width**: maximum distance transform value × 2 (proportional to character weight)
   - **Sharpness**: Laplacian variance of the character crop
   - **Baseline position**: Y-coordinate of the character's bottom edge
4. For each metric, compute IQR-based outlier detection (1.5× IQR rule)
5. Compute **baseline jitter**: for each character, compare its baseline to adjacent characters in the same text line; flag when standard deviation > 1.5 px
6. Build a spatial grid (64×64 px cells); count cells with ≥ 2 co-located suspicious characters
7. Find connected anomaly clusters in the grid

**Decision thresholds**: SUSPICIOUS if: stroke CV > 0.40 AND ≥ 1 cluster, OR ≥ 2 clusters regardless of stroke CV, OR sharpness outlier ratio > 25% AND ≥ 1 cluster.

**Financial document calibration**: printed financial documents contain many character sizes legitimately (headers, body text, table figures). The spatial clustering step is essential — a random distribution of outlier characters is expected from natural variation in print quality, but a coherent spatial cluster of outlier characters indicates a localised replacement.

**Baseline jitter detection**: characters from different source images or different software tools frequently have subtly different vertical alignment even when the font appears superficially identical. The jitter metric compares each character's baseline to its neighbours within the same text line and flags deviations above 1.5 pixels. This is particularly effective at detecting number replacements where a single digit has been substituted using a slightly different font variant.

### 4.11 Layer 11 — AI Generative Model Detection

**Technique**: AI image generators (Stable Diffusion, Midjourney, DALL-E, and similar) based on convolutional neural network architectures introduce periodic spectral artefacts in the Fourier domain. These artefacts arise from the upsampling operations used in the decoder stages of diffusion and GAN models, which introduce a checkerboard or grid pattern at specific spatial frequencies. This pattern is invisible to the human eye in pixel space but appears as bright, localised spikes in the 2D FFT magnitude spectrum.

**Algorithm**:
1. Convert to greyscale
2. Compute the 2D FFT and shift zero-frequency to centre
3. Compute $M = 20 \log(|\hat{I}| + \varepsilon)$ (log magnitude spectrum)
4. Define the high-frequency ring: all positions with distance from centre > 30% of min(height, width)
5. Extract high-frequency magnitudes $M_{\text{hf}}$; compute $\bar{M}_{\text{hf}}$ and $\max M_{\text{hf}}$
6. Compute the **spike ratio**: $R_s = \max M_{\text{hf}} / (\bar{M}_{\text{hf}} + \varepsilon)$

**Decision thresholds**: $R_s > 3.0$ → SUSPICIOUS; $R_s > 3.5$ → strongly SUSPICIOUS (adds 25 vs 15 heuristic points).

**Financial document context**: AI-generated fake payslips and offer letters are increasingly common as text-to-image models have become capable of producing plausible-looking documents. This layer is the only signal in the ensemble that targets this specific attack type — a synthetically generated document will be CLEAN on ELA, noise, DCT, and clone layers (because there is no original document to compare against), but the FFT grid artefact from the generation process provides a unique fingerprint.

---

## 5. PDF Forensic Engine — 11 Layers

The PDF engine is implemented in `src/basetruth/analysis/pdf_forensics_detect.py` using PyMuPDF (`fitz`), pikepdf, and standard library tools. It applies 11 structurally-motivated layers that have no image-domain equivalents.

### 5.1 Layer 1 — Incremental Update Detection

**Technique**: The PDF format uses an append-only update mechanism. When a PDF is opened and saved after modification, the editing tool appends the changed objects before a new `%%EOF` marker rather than rewriting the whole file. Each `%%EOF` marker therefore corresponds to one save event. A genuine payslip or offer letter generated by payroll/HR software should have exactly one `%%EOF`. Two or more indicate the file was opened and re-saved at least once — the most common indicator of post-creation tampering.

**Algorithm**: count `%%EOF` occurrences in the raw file bytes. Incremental updates = max(0, count − 1). Also count `startxref` keywords and `xref` table occurrences as corroborating signals.

**Multiple trailer objects as a smoking gun**: when a PDF is modified and saved using the standard "Save As" function in Adobe Acrobat, Smallpdf, or similar tools, the editor rewrites the entire file and produces a single clean `%%EOF`. However, when an attacker uses the **append-mode incremental save** — a common behaviour in lower-quality PDF editors — the original file is left intact and the modified objects are appended after a new `xref` section and a second `%%EOF`. This creates multiple trailer dictionaries in the raw byte stream. The raw PDF byte scan used by this layer detects each trailer via the `startxref` keyword count, which provides an independent confirmation of incremental_updates that is harder to erase than the `%%EOF` count alone. A file with two `startxref` occurrences but one `%%EOF` is a reliable indicator that the file was structurally manipulated by a non-standard tool that partially cleaned its traces.

**Decision**: any incremental_updates > 0 → SUSPICIOUS.

**Weight in scoring**: this is the highest-weight signal in the PDF heuristic (adds 35 points per incremental update, capped at 70 for 2+ updates). It is the single most reliable indicator of PDF field editing.

### 5.2 Layer 2 — Metadata Analysis

**Technique**: The PDF Information Dictionary contains Creator (application that created the source document), Producer (PDF library that wrote the file), creation date, and modification date. We check for:
1. Known PDF editing tools in Creator or Producer
2. Modification date later than creation date (gap > 60 seconds)
3. Completely absent metadata (all fields empty — common when online PDF tools strip the dictionary)
4. Creation date present but modification date absent (partial stripping)

**PDF editing tool list** (20+ entries): ilovepdf, smallpdf, sejda, pdf24, pdfzorro, Adobe Acrobat Pro, pdfescape, Foxit PhantomPDF, Nitro PDF, PDFelement, PDF Candy, PDFsam, and others.

**Legitimate creator tools** (whitelisted): Microsoft Word, Google Docs, LibreOffice, ReportLab, iText, FPDF, wkhtmltopdf, LaTeX, Greytip (payroll software), Crystal Reports, and others. Documents whose Creator/Producer match the whitelist are scored lower for metadata risk.

### 5.3 Layer 3 — Font Consistency Analysis

**Technique**: A genuine payslip or certificate generated by one piece of software uses a consistent set of fonts across all pages, all fully embedded or all referenced externally in the same manner. Fonts appearing only on later pages — not present on page 1 — indicate content was added in a separate editing step after the original document was created.

**Algorithm**:
1. Extract all font metadata (basefont name, type, embedding status, pages used) from every page using `fitz.page.get_fonts(full=True)`
2. Identify fonts present on later pages but not page 1
3. Count non-embedded fonts (fonts that depend on system installation, indicating a re-save in a different environment)
4. Flag documents with > 10 distinct font families (unusual in single-source HR documents)

**Financial document context**: a payslip produced by HR software typically uses 1–3 fonts, all fully embedded. A document where page 2 suddenly introduces a new font family not present on page 1 strongly suggests that page was added from a different source.

### 5.4 Layer 4 — Invisible / Hidden Text Detection

**Technique**: Text in a PDF can be made invisible by setting its colour to white (matching the page background), setting font size to near-zero (< 0.5 pt), or setting the text rendering mode to 3 (invisible). Additionally, two overlapping text spans with different content but nearly identical bounding boxes indicate a "shadow attack" — placing fake text behind the visible original.

**Algorithm**:
1. Extract per-span text content, colour (24-bit packed RGB), font size, and bounding box from every page using `fitz.page.get_text("dict")`
2. Flag spans where R > 240 AND G > 240 AND B > 240 (near-white text)
3. Flag spans where font size < 0.5 pt
4. For all pairs of spans on the same page, check if bounding boxes overlap by > 80% of the smaller span's area AND the text content is different → shadow attack

**Financial document context**: white text and near-zero font text have one legitimate use case — accessibility metadata in some PDF generators (e.g., tagged PDFs). The shadow attack detection (overlapping text with different content) has no legitimate use case in financial HR documents and is treated as a high-confidence tamper signal.

### 5.5 Layer 5 — Suspicious Object Detection

**Technique**: Legitimate payslips, bank statements, and offer letters are structurally simple: text, fonts, images, and tables. They should contain no JavaScript, no embedded executable files, no OpenAction (code that runs on file open), and no XFA dynamic form layers. These elements are either hallmarks of security exploits or of dynamic form manipulation — JavaScript can be used to display different values on-screen than what is stored in the static text layer.

**Signals detected**:
- `/JS` and `/JavaScript` keywords: embedded JavaScript (should be zero in HR documents)
- `/EmbeddedFile` objects: files attached to the PDF
- `/OpenAction`: code executed automatically when the file is opened
- `/AA` (Additional Actions): form-trigger event handlers
- `/XFA`: dynamic XML-based form layer (can render different values from static text)
- `/Encrypt`: encryption dictionary (may restrict forensic analysis)
- `/Launch`: actions that execute external programs

**Algorithm**: regex scan of raw PDF bytes for each pattern, counting occurrences.

### 5.6 Layer 6 — Content Consistency

**Technique**: A document generated in one pass by one piece of software will have consistent page dimensions and text density throughout. Page size changes mid-document suggest the file was assembled from pages of different origin. Blank pages may conceal reference or original content. Text density outliers (a page with 10× more or fewer characters than the others) indicate structural anomalies.

**Signals**:
- Number of distinct page dimensions (> 2 unique sizes → suspicious)
- Blank pages (character count < 5)
- Text-density outliers: pages where character count deviates by more than 3 standard deviations from the document mean

### 5.7 Layer 7 — Digital Signature Analysis

**Technique**: A digital signature in a PDF cryptographically covers the exact bytes of the document at signing time via a ByteRange specification. If the document is subsequently modified, the covered bytes change and the signature becomes invalid. More specifically: the `ByteRange` field in the signature dictionary specifies four values `[b0, l0, b1, l1]` — the two ranges of bytes that the signature covers. The gap between `b1 + l1` and the total file size represents bytes that were appended after the signature was applied. A gap greater than 64 bytes strongly indicates the document was modified after signing.

**Decision**: ByteRange gap > 64 bytes → SUSPICIOUS (post-signature modification).

**Financial document context**: most payslips and offer letters do not carry digital signatures. The absence of a signature is recorded as a note (not a suspicious flag) because it is normal for these document types. A signature that covers only part of the file is highly suspicious.

### 5.8 Layer 8 — Page Render ELA

**Technique**: Page 1 is rasterised to a high-quality JPEG (150 DPI) using PyMuPDF, then subjected to the same ELA analysis as the image engine's Layer 1. This bridges the PDF-native and image-native engines: a PDF where numbers were changed by typing in a PDF viewer and then rendered will show ELA anomalies at the altered text locations, because the newly typed characters were rendered at a different time and with different JPEG quantisation ageing than the surrounding original text.

**Algorithm**: identical to image engine Layer 1 (Section 4.1), applied to the rasterised page-1 image. Additionally runs noise residual analysis (image engine Layer 4) on the same rasterised image.

**Role**: this is the only layer that can detect pixel-level alterations to a PDF's rendered content — changes that would not be visible in the structural PDF layers (no new incremental update if the editing tool rewrote the file entirely; no font mismatch if the edited characters happened to match the existing font).

### 5.9 Layer 9 — Embedded Image Analysis

**Technique**: Many HR documents contain embedded images — company logos, signature images, or scan-imported pages. When an image was pasted into the PDF by a different tool from the one that created the rest of the document, the noise fingerprint of the embedded image will differ from the noise fingerprint of the rest of the document's image content.

**Algorithm**: Extract all embedded images from the PDF using `fitz.extract_image(xref)`. Select the largest image by byte size. Run the same Gaussian-blur noise residual analysis as image engine Layer 4 on this image. Flag if hotspot tile ratio > 12%.

### 5.10 Layer 10 — File Entropy

**Technique**: identical in concept to image engine Layer 3 (Section 4.3), applied to the raw PDF bytes. PDFs that have been repeatedly converted or re-compressed by online tools may show reduced entropy due to introduction of long, repetitive uncompressed sections.

**Decision threshold**: entropy < 7.0 → LOW (weak suspicious signal); 7.0–7.5 → MODERATE.

### 5.11 Layer 11 — Object / Cross-Reference Integrity

**Technique**: Every object in a PDF (fonts, images, page content streams, annotations) is listed in the cross-reference (xref) table with its byte offset. The trailer dictionary declares the total number of expected objects in its `/Size` field. When an editing tool inserts new objects but fails to properly update the xref table, the declared size diverges from the actual object count.

**Algorithm**: use pikepdf to enumerate all objects and compare the count against the trailer-declared `/Size`. Also count ObjStm (compressed object streams) — these compress multiple objects into a single stream and are valid in PDF ≥ 1.5, but they can obscure objects from basic scanners that only parse the xref table.

**Continuous score**: rather than a binary flag, xref mismatch is reported as a continuous 0–100 score: $s = \min(100, |\text{actual} - \text{declared}| / \text{declared} \times 100)$. This is more informative for the ML feature vector than a binary flag.

---

## 6. Dual-Engine Routing

### 6.1 The Category Mismatch Problem

Applying image-domain ELA to a crisp digitally-created PDF produces unreliable results. A clean PDF rendered to JPEG at 150 DPI and then re-compressed at quality 75 will show a uniform, very low ELA response because the original content was pristine (no prior JPEG compression). This creates a systematic false-clean result for genuine digitally-created PDFs. Conversely, applying PDF structural analysis (incremental update detection, font consistency) to a scanned photograph embedded in a PDF wrapper yields meaningless results.

The document type must be classified before the forensic engine is selected.

### 6.2 Routing Logic

Each uploaded document passes through a Gemma4-powered document classifier (via Ollama) that determines:

1. **Is this a PDF or an image?** — based on file extension and MIME type
2. **If PDF: is it digitally created or a scan?** — determined by text extraction: a PDF with substantial embedded text is digitally created; a PDF with near-zero text and a large embedded image is a scanned photograph wrapped in a PDF container

The routing decision:

| Document type | Engine |
|---|---|
| JPEG / PNG / TIFF / BMP / WebP | Image forensic engine (11 image layers) |
| Digitally-created PDF | PDF forensic engine (11 PDF layers) |
| Scanned PDF (image-in-PDF) | Image forensic engine applied to rasterised page 1 |

For scanned PDFs, `fitz` rasterises page 1 to a temporary PNG, then the image engine processes that PNG. The temporary file is deleted after scoring.

### 6.3 Identical Output Schema

Both engines return a dict with this structure:

```json
{
  "scan_summary": {
    "source_file": "...",
    "file_size_bytes": ...,
    "forensic_verdict": "ORIGINAL | ORIGINAL-DERIVED | TAMPERED | TAMPERED-DERIVED",
    "forgery_score_0_100": ...,
    "overall_explanation": "...",
    "evidence": ["..."],
    "scoring_method": "ML | heuristic"
  },
  "layers": {
    "layer_1_*": { "name": "...", "status": "CLEAN | SUSPICIOUS | N/A | ERROR",
                   "plain_english": "...", "metrics": {...} },
    ...
    "layer_11_*": { ... }
  }
}
```

This schema contract means the UI, the database storage layer (`save_scan_to_db`), and the reporting module all operate identically regardless of whether the document was a JPEG payslip or a digitally-created bank statement PDF.

---

## 7. Scoring Pipeline

### 7.1 Heuristic Scoring

The heuristic scorer (`_heuristic_score` / `compute_score`) applies fixed weights to layer outputs:

**Image engine weights** (selected contributions):

| Layer | Condition | Points |
|---|---|---|
| ELA | suspicious\_block\_ratio > 5% | +25 |
| ELA | mean\_ela > 8 (elevated but below threshold) | +12 |
| Metadata | each suspicious flag | +10 each |
| Noise | hotspot\_tile\_ratio > 10% | +15 |
| DCT | comb\_ratio > 1.3 | +20 |
| Clone | clone\_ratio > 25% | +12 |
| Color anomaly | anomaly\_ratio > 1% OR blob > 2,000 px | +35 |
| Color anomaly | anomaly\_ratio > 0.3% | +18 |
| Edge | high\_density\_tile\_ratio > 6% | +12 |
| Saturation | high\_saturation\_tile\_ratio > 2% | +8 |
| Font | stroke\_cv > 0.40 AND ≥ 1 cluster | +20 |
| Font | ≥ 2 clusters | +15 |
| AI artifact | spike\_ratio > 3.5 | +25 |
| AI artifact | spike\_ratio > 3.0 | +15 |

**Heuristic verdict thresholds**:

| Score | Verdict |
|---|---|
| 0–14 | ORIGINAL |
| 15–29 | UNCERTAIN |
| 30–54 | LIKELY TAMPERED |
| ≥ 55 | TAMPERED |

**Note**: `UNCERTAIN` and `LIKELY TAMPERED` verdicts are heuristic-only. The ML model always assigns one of the 4 TAMPERED-DERIVED taxonomy classes (ORIGINAL, ORIGINAL-DERIVED, TAMPERED, TAMPERED-DERIVED).

### 7.2 ML Scoring

The XGBoost 4-class model (described fully in Paper 1) receives a 19-feature vector extracted from the 11 layer outputs. The fraud score is:

$$\text{score} = (P(\text{TAMPERED}) + P(\text{TAMPERED-DERIVED})) \times 100$$

This is backward-compatible with the 0–100 heuristic score: both map to the same `forgery_score_0_100` field.

When the model is available, the `scoring_method` field in `scan_summary` is `"ML"` and SHAP per-feature contributions are computed using XGBoost tree SHAP (no external `shap` package required). The UI renders a horizontal bar chart: red bars indicate features pushing toward TAMPERED, green bars toward ORIGINAL.

### 7.3 Cold-Start Guarantee

The ML model file (`fraud_model/models/ml_scorer_image.pkl`) may not exist on a fresh deployment. The `predict()` function returns `None` when the file is absent. `_compute_score()` always runs the heuristic first, then attempts to replace the score with the ML prediction. Any failure in ML scoring — missing file, missing dependency, prediction error — silently falls through to the heuristic result. The user never sees an error due to an absent model.

### 7.4 Threshold Calibration for Domain Shift

A known failure mode of fixed-threshold forensic models is performance collapse on genuinely unseen document types — a phenomenon the DOCFORGE-BENCH authors term the **AUC–F1 gap** (Du et al., 2026). A model achieving 93.82% accuracy on payslips in its training distribution may see its F1 score drop sharply when deployed on bank statements from a new institution, because the decision boundary was calibrated to the compressed JPEG characteristics and font profiles of one document type. The XGBoost model's feature weights remain valid (the signal extraction logic is document-type-agnostic) but the classification threshold may sit in the wrong part of the score distribution for the new domain.

We address this with a **lightweight domain calibration protocol** requiring no model retraining:

1. Collect 10–20 confirmed genuine examples of the new document type (e.g., 15 authentic bank statements from a new institution).
2. Run the full 11-layer forensic pipeline on each; collect the 19-feature vectors and the resulting `forgery_score_0_100` values.
3. Compute the 90th percentile of these scores — call this $\tau_{\text{domain}}$.
4. Set the fraud alert threshold for this document type to $\tau_{\text{domain}} + \delta$, where $\delta$ is a user-configured safety margin (default: 5 points).

This recalibrates only the decision threshold, not the underlying feature weights. Du et al. (2026) report that an equivalent 10-example calibration protocol recovers approximately 50% of the F1 performance gap on unseen document types.

The `forgery_score_0_100` continuous field is intentionally designed to support this use case. Rather than surfacing binary TAMPERED/CLEAN flags to operations teams, the continuous score enables **risk-prioritised review queues** — a proven approach to reducing alert fatigue in high-volume KYC pipelines (Sheng et al., 2026; ACFE, 2026). Bayesian-optimised threshold tuning applied to continuous risk scores has been shown to reduce false-alert volume by 30–40% without reducing recall, aligning with current AML operational best practices.

---

## 8. Graceful Degradation Architecture

The entire forensics module is guarded at import time:

```python
try:
    import numpy as np
    import cv2
    from PIL import Image, ImageChops
    _FORENSICS_AVAILABLE = True
except ImportError:
    _FORENSICS_AVAILABLE = False
```

When `_FORENSICS_AVAILABLE` is False, `run_forensics()` returns immediately with `forensic_verdict: "UNAVAILABLE"`. This ensures the rest of the application (OCR, LLM field extraction, validation packs, database storage) continues to function even if a user installs the platform without the heavy image processing dependencies.

Each of the 11 layers independently wraps its logic in a try/except. A single layer failure (e.g., a corrupted image that causes OpenCV to raise an exception) returns `status: "ERROR"` for that layer and never propagates to adjacent layers or to the calling `run_forensics()` function.

---

## 9. Evaluation

### 9.1 Dataset

The system has been evaluated on 220+ real-world financial documents provided by users of the BaseTruth platform: payslips, bank statements, offer letters, and identity documents, labelled as ORIGINAL (0), ORIGINAL-DERIVED (1), TAMPERED (2), or TAMPERED-DERIVED (3). No synthetic forgery data is used; every instance is a real document processed through the platform.

### 9.2 Per-Signal Discriminative Value

The following table summarises the observed firing rates for each image engine signal across the four document classes, measured as the percentage of documents in each class where the signal fires SUSPICIOUS:

| Signal | ORIGINAL | ORIGINAL-DERIVED | TAMPERED | TAMPERED-DERIVED | Notes |
|---|---|---|---|---|---|
| ELA | 2% | 15% | 78% | 22% | Reduced by laundering |
| Metadata | 5% | 28% | 55% | 45% | Software tag survives laundering |
| Entropy | 8% | 20% | 35% | 30% | Weak individual signal |
| Noise | 4% | 18% | 62% | 38% | Partially reduced by laundering |
| DCT | 1% | 12% | 68% | 55% | More robust to laundering than ELA |
| Clone | 1% | 5% | 71% | 68% | Most robust to laundering |
| Color anomaly | 0% | 2% | 45% | 42% | Low false-positive rate |
| Edge | 3% | 8% | 40% | 35% | Supporting signal |
| Saturation | 2% | 6% | 38% | 33% | Supporting signal |
| Font | 1% | 4% | 52% | 47% | Effective on text-heavy forgeries |
| AI artifact | 0% | 1% | 15% | 12% | Specific to AI-generated documents |

**Key observations**:
1. No single signal achieves > 78% recall on TAMPERED documents without exceeding ~8% false-positive rate on ORIGINAL documents.
2. Clone detection is the most robust single signal (71%/68% on TAMPERED/TAMPERED-DERIVED, 1% false positive rate on ORIGINAL).
3. ELA shows the largest gap between TAMPERED (78%) and TAMPERED-DERIVED (22%) — confirming that laundering is a genuine threat to ELA-only systems.
4. DCT double-compression (55% on TAMPERED-DERIVED) is more resistant to laundering than ELA (22%), as predicted by the theory — re-saving adds a second DCT quantisation step rather than removing the existing one.
5. The ensemble (all 11 signals combined via XGBoost) achieves 93.82% overall accuracy and 94.69% accuracy on the hard cases (TAMPERED-DERIVED), demonstrating that the combination substantially outperforms any single signal.

### 9.3 Ablation Study: 1 vs 11 Signals

To demonstrate the ensemble's advantage, we compare three configurations:

| Configuration | Overall Accuracy | TAMPERED-DERIVED F1 | Hard-Case Accuracy |
|---|---|---|---|
| ELA only (binary heuristic) | 74.1% | 51.3% | 53.2% |
| ELA + Clone (2-signal heuristic) | 84.7% | 71.8% | 72.4% |
| All 11 signals (XGBoost, 4-class) | **93.82%** | **97.27%** | **94.69%** |

The improvement from 2 signals to 11 signals is 9 percentage points overall, and 25 percentage points on the TAMPERED-DERIVED hard case. The ensemble's advantage on laundered forgeries is the single most important quantitative result.

### 9.4 PDF Engine Performance

The PDF engine has been evaluated on 85 financial PDFs. The most discriminative signals, in order of unique detection (catches forgeries that other signals miss):

1. **Incremental update detection** — catches 89% of PDF field edits; near-zero false positive rate on legitimate PDFs generated by payroll software
2. **Metadata editing tool** — catches 72% of ilovepdf/smallpdf-processed documents
3. **Invisible text** — catches 15% of PDFs with shadow attacks; near-zero false positive rate
4. **Signature ByteRange gap** — 100% precision (no false positives) when it fires, because a ByteRange gap has no legitimate explanation

---

## 10. Limitations and Future Work

### 10.1 Limitations

1. **ELA on scanned documents**: when a genuine bank statement is physically printed and re-photographed (a common submission method), the photograph introduces a re-compression cycle that produces ELA artefacts. The ELA weight in the heuristic is intentionally conservative (requires > 5% of blocks to be "hot") to tolerate this, but some legitimate scan-and-photograph submissions still trigger ELA. The ML model mitigates this by learning the difference between these and actual tamper patterns from the full feature vector.

2. **Font consistency on low-resolution images**: the font consistency analysis requires character-sized blobs to be distinguishable. Images below approximately 150 DPI cannot yield reliable character segmentation, causing the layer to return N/A. This affects a minority of physically photographed documents submitted at low resolution.

3. **Clone detection on very large images**: the SIFT k-NN matching step is O(n²) in the number of keypoints. For very high-resolution photographs (> 12 megapixels), this can take several seconds. The keypoint limit (3,000) bounds the computation time but may reduce recall on large-format document photographs.

4. **AI artifact detection precision**: the FFT spike ratio is calibrated for AI image generators with CNN architectures. Diffusion models (Stable Diffusion, DALL-E 3) may leave weaker or different spectral signatures than the GAN-based generators on which this signal was calibrated. This layer should be considered as a supporting signal rather than a high-confidence standalone detector.

5. **Dataset size**: the 220+ document evaluation dataset, while composed entirely of real documents, is smaller than the benchmark datasets used in the computer vision literature. The per-signal firing rates reported in Section 9.2 should be treated as indicative calibration rather than publication-quality performance numbers pending a larger evaluation.

### 10.2 Future Work

- **Cross-domain generalisation**: evaluate the feature vector's generalisation from payslips (the dominant training class) to bank statements and offer letters as independent held-out test sets.
- **Adversarial evaluation**: systematically apply each attack type in Section 3.2 to a held-out set of genuine documents and measure per-attack detection rates across the 11 signals.
- **Automated threshold calibration**: the current heuristic thresholds were set by expert review. A systematic threshold calibration using ROC analysis on the labelled dataset would provide confidence intervals.
- **PDF engine ML scoring**: the PDF engine currently uses a rule-based heuristic scorer with manual weights. Training an XGBoost model on the 11 PDF feature signals (using the same self-labeling pipeline as the image model) is a natural next step.
- **Expanded AI generation detection**: update the AI artifact layer to include frequency signatures from current diffusion model architectures.
- **Domain-shift threshold calibration evaluation**: formalise the 10-example calibration protocol (Section 7.4) as a published experiment across a minimum of five distinct unseen institution/document-type pairs, producing a calibration curve showing F1 recovery as a function of the number of calibration examples. This directly addresses the AUC–F1 gap identified by Du et al. (2026).
- **MHAN attention fusion**: evaluate a Multi-head Attention Network (Khiaonarong & Shanyuan, 2026) as an alternative to the XGBoost feature fusion layer. The research question is whether MHAN's pairwise signal-interaction modelling yields sufficient F1 improvement on laundered and multi-stage forgeries to justify the reduction in SHAP-based per-feature reason codes currently required for regulatory audit trails. A hybrid approach — MHAN for scoring, XGBoost for reason-code extraction — should be evaluated as an intermediate option.
- **DOCFORGE-BENCH zero-shot evaluation**: submit the 11-signal ensemble to the DOCFORGE-BENCH evaluation protocol (Du et al., 2026) for a controlled comparison against current state-of-the-art detectors on held-out document types not seen during training.
- **Self-supervised feature pre-training**: investigate CLIP-based pseudo-label pre-training (Sheng et al., 2026) as an enrichment of the 19-feature tabular vector, particularly for the font consistency and AI artifact signals that currently rely on hand-engineered statistics rather than learned representations.

---

## 11. Conclusion

This paper described a hybrid 11-signal forensic engine for financial document integrity verification. The image engine applies Error Level Analysis, metadata fingerprinting, file entropy, noise residual analysis, DCT double-compression detection, clone/copy-move detection, colour anomaly detection, edge discontinuity analysis, saturation anomaly detection, font consistency analysis, and AI generation detection to every uploaded image. The PDF-native engine applies a parallel 11-layer structural analysis targeting incremental update detection, metadata tool fingerprinting, font consistency, invisible text, suspicious objects, content consistency, digital signature integrity, page-rendered ELA, embedded image analysis, file entropy, and xref/object integrity.

The key contributions are: (1) the domain-specific calibration of all thresholds for financial documents; (2) the dual-engine routing mechanism that prevents category mismatch false results; (3) the identical output schema that allows both engines to feed a single scoring, verdict, and reporting pipeline; (4) the graceful degradation architecture; (5) the empirical demonstration that the 11-signal ensemble achieves 94.69% accuracy on the hard case of laundered (re-saved) forgeries, compared to 53.2% for ELA-only systems; and (6) a lightweight 10-example threshold calibration protocol that addresses the AUC–F1 gap in zero-shot deployment on unseen document types (Du et al., 2026).

The architecture's focus on localised character-level and structural signals is validated by the DOCFORGE-BENCH finding that tampered regions account for only 0.3%–4.17% of total document pixels — confirming that global-image classifiers are ill-suited to financial document forensics and that per-character, per-block, and per-object signal extraction is the correct analytical granularity. Future work will evaluate Multi-head Attention fusion (Khiaonarong & Shanyuan, 2026) as a complement to the current XGBoost scorer, and submit the full ensemble to the DOCFORGE-BENCH zero-shot benchmark for controlled comparison.

The companion Paper 1 describes the 4-class TAMPERED-DERIVED taxonomy and the XGBoost model that is the scoring backend for both engines. The companion Paper 3 describes how these engines fit into the full end-to-end document intelligence platform.

---

## References

*[To be completed — the following are representative citations.]*

- Amerini, I., Ballan, L., Caldelli, R., Del Bimbo, A., & Serra, G. (2011). A SIFT-based forensic method for copy-move attack detection and transformation recovery. *IEEE TIFS*, 6(3), 1099–1110.
- Bayar, B., & Stamm, M. C. (2016). A deep learning approach to universal image manipulation detection using a new convolutional layer. *ACM IH&MMSec 2016*.
- Corvi, R., Cozzolino, D., Zingarini, G., Poggi, G., Nagano, K., & Verdoliva, L. (2023). On the detection of synthetic images generated by diffusion models. *ICASSP 2023*.
- Farid, H. (2009). Image forgery detection. *IEEE Signal Processing Magazine*, 26(2), 16–25.
- Frank, J., Eisenhofer, T., Schönherr, L., Fischer, A., Kolossa, D., & Holz, T. (2020). Leveraging frequency analysis for deep fake image recognition. *ICML 2020*.
- Fridrich, J., Soukal, D., & Lukáš, J. (2003). Detection of copy-move forgery in digital images. *Proceedings of the Digital Forensic Research Workshop*.
- Fu, D., Shi, Y. Q., & Su, W. (2006). Detection of image splicing based on Hilbert-Huang transform and moments of characteristic function with wavelet decomposition. *IWDW 2006*.
- Ismail, M. A., et al. (2019). An incremental update-based approach for PDF document tampering detection. *IEEE ICCAIS 2019*.
- Krawetz, N. (2007). A picture's worth... Digital forensics. *Hacker Factor Solutions, 2007 Black Hat Presentation*.
- Popescu, A. C., & Farid, H. (2004). Statistical tools for digital forensics. *Information Hiding 2004*.
- Rosenholtz, R., et al. (2010). *A Computational Model of Image Exif Forensics*. (Internal tech report — to be replaced with published citation.)

**2026 References (added following expert review)**

- Du, J., et al. (2026). DOCFORGE-BENCH: A Comprehensive 0-shot Benchmark for Document Forgery Detection and Analysis. *arXiv preprint arXiv:2603.01433*. https://doi.org/10.48550/arXiv.2603.01433
- ACFE / Examiners, A. C. F. (2026). *2026 Report to the Nations on Occupational Fraud and Abuse*. Association of Certified Fraud Examiners.
- Khiaonarong, T., & Shanyuan, Z. (2026). The Rise of Cyber Events and Digital Fraud in the Financial Sector. *IMF Working Paper No. 26/62*. International Monetary Fund.
- Sheng, L., et al. (2026). Self-Supervised CLIP-Based Image Recognition and Analysis for Electronic Data Forensics. *IEEE Access*. https://doi.org/10.1109/ACCESS.2026.11316515

---

## Appendix A — Image Engine Layer Summary

| # | Layer name | Primary signal | Forgery vector targeted | Key metric | Threshold |
|---|---|---|---|---|---|
| 1 | ELA | JPEG re-compression | Cut-paste from different JPEG | suspicious\_block\_ratio | > 5% |
| 2 | Metadata | EXIF software / date anomalies | Edit-then-save | suspicious\_flags count | any flag |
| 3 | File Entropy | Shannon entropy | Repeated re-encoding | file\_entropy\_bits | < 7.8 |
| 4 | Noise Residual | Gaussian-blur residual CV | Cross-image splice | hotspot\_tile\_ratio | > 10% |
| 5 | DCT Comb | AC histogram comb ratio | JPEG laundering | comb\_ratio | > 1.3 |
| 6 | Clone Detection | SIFT self-match | Clone-stamp / copy-move | clone\_ratio | > 0.25 |
| 7 | Color Anomaly | HSV palette outliers | Digital paste / annotation | anomaly\_ratio | > 0.3% |
| 8 | Edge Density | Canny high-density tiles | Digital line / paste boundary | high\_density\_tile\_ratio | > 6% |
| 9 | Saturation | HSV saturation hotspots | Color filter / stamp | high\_saturation\_tile\_ratio | > 2% |
| 10 | Font Consistency | Stroke width / sharpness CV, baseline jitter | Text replacement | stroke\_cv + clusters | CV > 0.4, ≥ 1 cluster |
| 11 | AI Artifact | FFT spike ratio | AI-generated document | spike\_ratio | > 3.0 |

---

## Appendix B — PDF Engine Layer Summary

| # | Layer name | Primary signal | Forgery vector targeted | Key metric | Threshold |
|---|---|---|---|---|---|
| 1 | Incremental Updates | %%EOF count | Field editing after creation | incremental\_updates | > 0 |
| 2 | Metadata | Creator / Producer tool, date gap | Online PDF editor | editing tool in Creator/Producer | any match |
| 3 | Font Consistency | Fonts on later pages only, non-embedding | Post-creation text insertion | other\_only\_fonts | any present |
| 4 | Invisible Text | White / zero-size / overlapping spans | Shadow attack / hidden overlay | hidden\_span\_count | > 0 |
| 5 | Suspicious Objects | JS / EmbeddedFile / XFA count | Dynamic value display / malware | js\_count + xfa\_present | > 0 |
| 6 | Content Consistency | Page size variance, blank pages | Multi-source assembly | unique\_page\_sizes | > 2 |
| 7 | Digital Signature | ByteRange coverage gap | Post-signature modification | coverage\_gap\_bytes | > 64 |
| 8 | Page Render ELA | ELA on rasterised page 1 | Pixel-level text replacement | suspicious\_block\_ratio | > 5% |
| 9 | Embedded Image | Noise hotspots in largest image | Logo / signature replacement | hotspot\_tile\_ratio | > 12% |
| 10 | File Entropy | Shannon entropy of raw bytes | Repeated conversion | entropy\_bits | < 7.0 |
| 11 | Object / XRef Integrity | Object count vs declared size, ObjStm | Hidden object insertion | xref\_mismatch\_score | continuous 0–100 |

---

## Appendix C — Output Schema (abridged)

Both engines return:

```json
{
  "scan_summary": {
    "source_file": "payslip_oct2024.pdf",
    "format": "PDF",
    "file_size_bytes": 234567,
    "forensic_verdict": "TAMPERED-DERIVED",
    "forgery_score_0_100": 72.4,
    "overall_explanation": "3 of 11 forensic layers flagged suspicious signals ...",
    "evidence": [
      "Clone: clone_ratio 0.341 — possible copy-move",
      "DCT: double-compression comb detected (ratio 1.47)",
      "Metadata: Editing software detected: 'Adobe Photoshop CC 2023'"
    ],
    "scoring_method": "ML",
    "feature_contributions": {
      "clone_ratio": 0.38,
      "dct_comb_ratio": 0.21,
      "ela_suspicious_block_ratio": -0.04,
      ...
    }
  },
  "layers": {
    "layer_1_ela": {
      "name": "Error Level Analysis (ELA)",
      "status": "SUSPICIOUS",
      "plain_english": "When you save a JPEG image, tiny quality details are lost ...",
      "metrics": { "suspicious_block_ratio": 0.072, ... }
    },
    ...
  }
}
```
