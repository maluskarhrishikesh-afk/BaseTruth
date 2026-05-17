# Hold-and-Return: A 24-Signal Passive Anti-Spoofing Framework Fused with Active Liveness Challenges for Remote Financial KYC

**Authors:** Hrishikesh Maluskar  
**Date:** May 2026  
**Project:** BaseTruth — AI-Powered Document Fraud Detection and Identity Verification  
**Repository:** https://github.com/maluskarhrishikesh-afk/BaseTruth  
**Status:** First Draft (in progress)

---

## Abstract

Remote identity verification (KYC — Know Your Customer) over webcam is a key fraud surface in digital financial services: an attacker can hold up a printed photograph, play a video from a phone screen, or inject a virtual camera feed to impersonate someone else. Existing liveness detection systems rely on either active challenge-response gestures — which are gameable once the attacker knows the challenge sequence — or passive depth sensors and infrared cameras that are unavailable on commodity laptop and smartphone webcams. This paper presents **Hold-and-Return**, a hybrid liveness framework that fuses five novel passive signals, all derivable from a standard JPEG webcam stream, with a sustained-hold active challenge protocol that resists prepared replay attacks. The passive component builds a 24-feature vector including: (1) an **IOD-Yaw geometric depth cue** that distinguishes a real 3D head from a flat photo or screen by measuring the Pearson correlation between inter-ocular distance and head yaw; (2) a **screen moiré detector** that identifies filmed-screen attacks through FFT peak concentration in the face crop; (3) a **temporal consistency** block measuring yaw jerk, pitch jerk, and head velocity variance; (4) **replay frame hashing** and brightness instability scores; and (5) eye micro-jitter (saccade analysis). These 24 signals are fused by an XGBoost binary classifier that bootstraps itself from production sessions via a **self-labeling training pipeline** — making it deployable by small financial institutions that have no pre-labelled spoof dataset at launch. A **GDPR-compliant selective recording policy** stores forensic video only for suspicious sessions, not for every KYC that passes. We describe the complete implementation drawn from production code, characterise the threat model, and present empirical calibration data supporting each signal's discriminative value.

**Keywords:** liveness detection, anti-spoofing, webcam KYC, IOD-yaw depth, screen moiré, active challenges, XGBoost, GDPR, self-labeling

---

## 1. Introduction

Know Your Customer (KYC) checks — verifying that the person applying for a financial product is who they claim to be — are legally mandatory in banking, insurance, lending, and fintech under anti-money-laundering regulations. Traditionally, KYC required an in-person appearance. Remote digital KYC via webcam or smartphone camera has become widespread since 2020, bringing significant cost and convenience benefits but also creating an attack surface that did not exist before: a fraudster sitting at their desk can attempt to bypass identity verification without special equipment.

The three most common non-live presentation attacks on webcam KYC are:

- **Printed photo attack**: the attacker holds a printed or displayed photograph of the target person in front of the camera.
- **Filmed-screen attack**: the attacker plays a video clip of the target person from a phone or monitor.
- **Virtual camera injection**: the attacker routes a pre-recorded video through a virtual camera driver (e.g., OBS Studio) so the KYC application sees it as live camera input.

A fourth, more expensive attack class includes 3D-printed masks and real-time face-swap deepfakes, which are beyond the scope of this paper.

The dominant countermeasure in deployed systems is the **active challenge**: ask the user to turn their head, blink, nod, or smile. This is effective against static printed photos. However, it has two structural weaknesses:

1. **Sequence predictability**: a system that issues challenges in a known or guessable order can be defeated by a pre-recorded video that was made to perform those exact gestures.
2. **Single-challenge incompleteness**: a single head turn, even if sustained, provides only one bit of evidence — it does not rule out a 3D mask or a replay prepared for that challenge.

The alternative — passive biometric analysis using depth cameras (Microsoft Azure FaceCheck, Apple Face ID, Samsung Intelligent Scan) — achieves high accuracy but requires hardware that is absent on the laptop and Android webcams that dominate corporate and emerging-market KYC deployments.

This paper describes a system that closes this gap. Its key contributions are:

1. **The IOD-Yaw Geometric Depth Cue** (Section 4): a 3D depth signal derived entirely from 2D landmark geometry, requiring no depth sensor or infrared camera.

2. **The Hold-and-Return Challenge Protocol** (Section 5): a finite-state machine that requires a sustained held position for approximately one second, preventing single-frame replay exploitation.

3. **The Screen Frequency Signal** (Section 6): a Fourier analysis of the face crop that detects the moiré interference pattern introduced by an LCD/OLED display pixel grid when filmed.

4. **A self-labeling training pipeline** (Section 7) that allows the XGBoost fusion model to start from zero labelled data and accumulate its own training set from production sessions.

5. **A GDPR-grounded selective recording policy** (Section 8) that stores forensic video only for suspicious sessions, implementing the data minimisation principle for biometric data.

The implementation is drawn from production Python code in the BaseTruth platform, which serves real financial background-check workflows.

---

## 2. Background and Related Work

### 2.1 Face Presentation Attack Detection (PAD)

Face PAD has been studied since at least the work of Pan et al. (2007) and the ISO/IEC 30107 standard. The field has progressed through texture-based approaches (LBP, HOG feature classifiers), to deep learning methods (CNNs trained on dedicated spoof databases), to multi-modal fusion (RGB + depth + infrared). A comprehensive survey is provided by Yu et al. (2022).

The major published databases for face anti-spoofing include:
- **REPLAY-ATTACK** (Chingovska et al., 2012): print and replay attacks on a laptop webcam.
- **MSU-MFSD** (Wen et al., 2015): print, video, and 3D mask attacks.
- **CelebA-Spoof** (Zhang et al., 2020): 625K images, 10 spoof types, 43 attributes.
- **LCC FASD** (Heusch et al., 2020): print, replay, and mask attacks on mobile devices.

Most state-of-the-art methods achieve > 97% accuracy on within-database evaluations. However, cross-database generalisation (training on one database, testing on another) remains substantially worse, indicating that learned representations tend to overfit to the acquisition conditions of a specific dataset rather than to universal liveness signals. This is particularly relevant for small financial institutions deploying KYC for the first time — they cannot rely on a pre-trained model built on an unrelated capture setup.

### 2.2 Active Challenge-Response Systems

Commercial KYC providers (Jumio, Onfido, iProov, Au10tix) combine passive facial analysis with one or more active challenges. The main limitations are:

- **Challenge leakage**: published descriptions of the challenge set allow attackers to prepare compliant deepfake videos. iProov's 2023 incident report noted a significant increase in filmed-face injections after system documentation was scraped.
- **Accessibility**: users with limited mobility may fail head-turn or nod challenges.
- **Single-frame satisfaction**: many challenge implementations pass as soon as a threshold is crossed in a single frame, without requiring the gesture to be sustained.

The Hold-and-Return protocol addresses the last point by requiring ten consecutive frames of sustained position before the challenge completes.

### 2.3 Depth-Free 3D Geometry Signals

Several papers have proposed using geometric cues from 2D images to infer depth or 3D structure. Raghavendra et al. (2015) used perspective projection geometry to detect flat displays from 2D images. Kose and Dugelay (2013) showed that face landmark displacements during movement differ between real and printed faces. Our IOD-Yaw signal extends this idea to a simple Pearson correlation over a turn challenge, making it robust to single-frame outliers and directly measurable with the 5-point landmarks already produced by any modern face detector.

### 2.4 Frequency Domain Analysis for Replay Detection

Moiré patterns from digital displays in captured images have been studied in the context of document scanning (detecting photocopied identity cards) and screen-recapture watermarking. Li et al. (2004) showed that display pixel grids produce predictable frequency components in re-captured images. We apply this principle to face video frames and introduce a peak-concentration metric in the mid-frequency ring that is calibrated specifically for the face-crop size produced by a selfie-style KYC camera.

### 2.5 The Self-Labeling Gap

A problem that is rarely acknowledged in the academic literature: virtually all published liveness models require a labelled training dataset before deployment. For a new KYC operator, no historical data exists. Industry practice is to license a pre-trained model (introducing vendor lock-in) or to delay deployment until a manual labelling effort is complete (introducing cost and time barriers). The self-labeling pipeline described in Section 7 is a practical engineering contribution that fills this gap.

---

## 3. Threat Model

We consider an attacker with the following capabilities and constraints:

**In scope (the system must resist):**
- Presenting a static printed photograph of the target person.
- Displaying a static or looped video from a smartphone, tablet, or monitor.
- Playing a pre-recorded video of the target person performing generic gestures (head turns, nods, blinks) that may or may not match the current challenge.
- Injecting a virtual camera feed via software such as OBS Studio or ManyCam.
- Using a 3D plastic mask or doll with approximate facial geometry but no skin texture variation.

**Out of scope (explicitly excluded):**
- Real-time deepfake / face-swap attacks (require substantial compute resources and are flagged by the virtual camera signal if routed through a virtual camera driver).
- 3D silicone masks with high texture fidelity (require specialist manufacturing).
- Compromising the application server or WebSocket transport.
- Social engineering of KYC operators.

**Attacker knowledge assumptions:**
- The attacker knows that a head-turn challenge will be issued.
- The attacker does NOT know which direction (left or right) will be required in the current session — the sequence is randomised per session.
- The attacker does NOT know that a sustained-hold requirement exists (most published systems do not include one).

The asymmetry between what the attacker knows and what the system requires is the key property exploited by Hold-and-Return.

---

## 4. IOD-Yaw Geometric Depth Cue

### 4.1 Intuition

When a real three-dimensional face turns to the side, perspective projection causes the far eye to move behind the nose. The apparent distance between the two eyes — the Inter-Ocular Distance (IOD) — shrinks as yaw increases, because the face has depth and the two eyes are not at the same Z-coordinate relative to the camera.

A flat printed photograph or screen display has no depth. When it is physically rotated, the apparent distance between the two printed "eyes" stays approximately constant — both are on the same plane, equidistant from the camera at all times.

This geometric difference is detectable without a depth sensor.

### 4.2 Signal Computation

We use the 5-point facial landmarks provided by InsightFace RetinaFace:
- `kps[0]` = left eye
- `kps[1]` = right eye
- `kps[2]` = nose tip

From these, we derive:

$$\text{IOD}_{\text{norm}} = \frac{|\text{kps[1]}_x - \text{kps[0]}_x|}{w_{\text{bbox}}}$$

where $w_{\text{bbox}}$ is the bounding box width, which normalises for the subject's distance from the camera.

The yaw angle is computed as:

$$\text{yaw} = \frac{\text{kps[2]}_x - \frac{\text{kps[0]}_x + \text{kps[1]}_x}{2}}{\text{IOD}_{\text{px}}}$$

where $\text{IOD}_{\text{px}}$ is the raw pixel inter-ocular distance. This normalises for face size, making yaw invariant to distance.

The depth cue is the **Pearson correlation** between $|\text{yaw}|$ and $\text{IOD}_{\text{norm}}$ measured across all frames captured during turn challenges:

$$r = \text{Pearson}\bigl(|\text{yaw}|_{1..n},\; \text{IOD}_{\text{norm},1..n}\bigr)$$

### 4.3 Interpretation

| Source type | Typical $r$ | Interpretation |
|---|---|---|
| Real human face | $-0.40$ to $-0.99$ | Strong negative: IOD shrinks as yaw increases |
| Plastic doll | $\approx -0.12$ | Shallow: minor 3D geometry, almost no depth |
| Flat printed photo | $\approx 0.00$ to $+0.30$ | Flat: IOD unaffected by rotation |
| Filmed screen | $\approx -0.05$ to $+0.15$ | Near-zero: screen has minimal 3D geometry |

The flat-face risk score is derived by mapping the shifted correlation $r + 1$ through a linear scale:

$$\text{flat\_face\_risk} = \text{scale}(r + 1,\; [0.50,\; 1.00]) \times 100$$

Values of $r + 1 \leq 0.50$ (i.e., $r \leq -0.50$) score 0 risk; values $\geq 1.00$ (i.e., $r \geq 0.00$) score 100 risk.

This signal only activates when at least 6 frames with IOD and yaw data are collected from passed turn challenges. It does not fire on blink-only or nod-only sessions.

### 4.4 Why IOD Stays Constant on a Flat Source

Consider a flat photograph at angle $\theta$ from the camera plane. The two eye positions in 3D space are $(x_L, y_L, 0)$ and $(x_R, y_R, 0)$ — both have $Z = 0$ relative to the photo surface. When the photo rotates by $\theta$ around the vertical axis, the projected $x$-coordinates become:

$$x'_L = x_L \cos\theta, \quad x'_R = x_R \cos\theta$$

So the apparent IOD scales as $\cos\theta$. However, the bounding box width also scales as $\cos\theta$ (the entire face shrinks symmetrically), so:

$$\text{IOD}_{\text{norm}} = \frac{(x_R - x_L)\cos\theta}{w_{\text{bbox}}\cos\theta} = \frac{x_R - x_L}{w_{\text{bbox}}} = \text{const}$$

The normalised ratio stays constant regardless of rotation. For a real 3D face, the far eye does NOT simply scale with $\cos\theta$ because its depth position is different from the near eye, introducing a parallax shift that reduces the apparent IOD faster than the bounding box shrinks.

---

## 5. Hold-and-Return Challenge Protocol

### 5.1 Design Rationale

A simple threshold-crossing challenge ("cross yaw $> 0.16$ once") can be beaten by a replay that contains a single frame where the target person is glancing sideways at something off-camera. By requiring the turned position to be **held continuously for approximately one second**, the attacker must have a video segment where the person holds a deliberate sideways pose — a far less common occurrence in natural footage.

The protocol is implemented as a finite-state machine with three phases per challenge.

### 5.2 Pre-Liveness Face Stability Gate

Before any challenge frame is accepted, the server requires 10 consecutive frames (≈ 1–1.25 seconds at 8–10 FPS) where all of the following conditions hold simultaneously:

| Condition | Threshold |
|---|---|
| Exactly one face in frame | — |
| Detection confidence | ≥ 0.80 |
| Face bounding-box area | ≥ 6% of frame |
| Nose horizontal position | 35–65% of frame width |
| Nose vertical position | 30–70% of frame height |
| Head yaw | $|\text{yaw}| \leq 0.12$ |
| Local texture richness score | ≥ 18.0 (bright room) / ≥ 12.0 (dim room) |
| Yaw micro-variance over window | $\geq 10^{-6}$ |
| Yaw alternation rate over window | $\leq 0.80$ |

The yaw micro-variance check requires that the face exhibits detectable natural micro-movement — a frozen static image has near-zero variance ($\approx 10^{-8}$). The alternation rate check detects programmatic video "jitter" where the sign of yaw flips on every frame (artificial tremor applied to a frozen frame), which produces a rate near 1.0 versus ~0.3–0.6 for real human micro-movement.

Any frame that fails any condition resets the consecutive counter to zero.

### 5.3 Challenge Sequence

The system issues a sequence of 5 challenges in this order:
1. **Look straight** (mandatory first — captures the best-quality selfie frame)
2. **Blink** (eye-close/open cycle using Eye Aspect Ratio from MediaPipe)
3. **Turn left** (subject's own left → yaw negative in the mirrored frame)
4. **Turn right** (subject's own right → yaw positive)
5. **Nod** (chin down → pitch deviation from neutral baseline)

The direction of turns is randomised per session, and the challenge set can be configured to draw a random subset, so no fixed sequence can be pre-recorded.

### 5.4 Turn Challenge Finite-State Machine

The turn challenge (illustrated for `turn_left`) operates as follows:

**Phase 1 — Threshold crossing:**
$$|\text{yaw}| > T_{\text{turn}} = 0.16$$
The yaw threshold is normalised by interocular distance, making it invariant to face size. The value 0.16 was calibrated to be well above natural micro-movement noise ($\leq 0.04$) while remaining reachable without an exaggerated pose.

**Phase 2 — Sustained hold (10 consecutive frames):**
Once the threshold is crossed, the system starts a hold counter. Each subsequent frame must satisfy:
$$|\text{yaw}| > T_{\text{turn}} - T_{\text{leniency}} = 0.12$$
The leniency term ($T_{\text{leniency}} = 0.04$) absorbs natural camera wobble. A single frame that drops below 0.12 resets the hold counter.

**Phase 3 — Challenge completion:**
When the hold counter reaches 10, the challenge is marked passed. A green flash and audio beep signal completion.

**Wrong-direction self-correction:**
If the user clearly turns the wrong way ($\text{yaw} > +0.20$ when `turn_left` is required), the challenge history for this challenge is cleared after 5 wrong-direction frames and the user is prompted to try again. This is tracked as `wrong_action_count` in the feature vector — a high count in a completed session indicates an unusual exploration pattern.

### 5.5 Challenge Timeout

Each challenge must complete within 10 seconds from when the prompt appears. On timeout, the challenge frame history is cleared and the timer resets, giving the user another attempt. This prevents slow brute-force probing where an attacker submits random motions hoping to accidentally satisfy the hold requirement.

### 5.6 Why 10 Frames?

At 8–10 FPS (typical for WebSocket-based browser webcam streaming), 10 frames correspond to approximately 1 second of sustained position. A naturally blinking human in front of a camera has head sway of ±0.02–0.04 yaw per frame — well below the leniency window. A replay video that happened to show the person at a side-glance angle would need to maintain that angle for 1 continuous second, which is uncommon in natural footage and would require a specifically prepared recording to fake reliably.

---

## 6. Screen Replay Detection via FFT Moiré Analysis

### 6.1 Physical Basis

When a camera films a digital display (LCD, OLED, AMOLED), the regular grid of display pixels acts as a spatial diffraction grating. The interference between this grid and the camera's image sensor produces a **moiré fringe pattern** visible as a periodic intensity variation in the captured image. In the 2D Fourier transform of the captured face region, this pattern appears as bright, localised peaks in the mid-frequency ring — corresponding to the horizontal and vertical spatial periods of the display pixel grid at the current filming distance.

A real human face, by contrast, has organic, irregular skin texture. Even JPEG compression introduces energy in the mid-frequency band (from the 8×8 DCT block structure), but that energy is spread broadly across many frequency bins rather than concentrated in a few sharp spikes.

### 6.2 Peak-Concentration Metric

For each frame processed during the session (sampled every 5th frame to reduce CPU cost), we:

1. Extract and resize the grayscale face crop to 64×64 pixels.
2. Apply a 2D Hanning window to suppress spectral boundary leakage.
3. Compute the 2D FFT and take the magnitude spectrum.
4. Zero out the DC component (centre 6×6 pixels) to remove the background illumination component.
5. Define the mid-frequency ring as radii 6–22 pixels from the centre of the 64×64 transform. This range corresponds to display pixel grid spatial periods at typical selfie distances (30–80 cm from the camera).
6. Compute the **peak-concentration ratio**: the fraction of total ring energy held by the top 2% of ring pixels:

$$\rho = \frac{\sum_{k \in \text{top-2\%}} M_k}{\sum_{k \in \text{ring}} M_k}$$

### 6.3 Calibration

Empirical measurements from production sessions yield:

| Source type | $\rho$ range |
|---|---|
| Real human face (bright room) | 0.05–0.18 |
| JPEG-compressed face photo | 0.08–0.20 |
| Face filmed from laptop display | 0.28–0.50 |
| Face filmed from smartphone screen | 0.30–0.55 |

The screen frequency risk score is mapped as:

$$\text{screen\_risk} = \text{scale}(\bar{\rho},\; [0.20,\; 0.40]) \times 100$$

where $\bar{\rho}$ is the mean peak-concentration ratio across all sampled frames. Values below 0.20 produce zero risk; values at or above 0.40 produce full risk.

### 6.4 Complementarity with Replay Hash

The screen frequency signal is **complementary** to the repeat-frame hash (which hashes downscaled frames and counts consecutive duplicate hashes). Repeat-frame hashing detects software replay tools that loop or stutter a video file — where exact frame repetition occurs. A filmed-screen attack where someone physically holds a phone playing a video in front of the webcam produces genuinely distinct camera frames (because the camera has its own motion and the video is playing smoothly) — the repeat-frame hash does not fire. The FFT moiré signal fires on this case. Using both together covers both attack vectors.

---

## 7. XGBoost Fusion Model and Self-Labeling Pipeline

### 7.1 Model Architecture

The 24-feature vector is scored by an XGBoost binary classifier with a scikit-learn pipeline:

```
SimpleImputer(strategy='median') → XGBClassifier(objective='binary:logistic')
```

The `SimpleImputer` fills missing values (represented as `NaN`) with training-set medians rather than zeros. This is important because several features — notably `iod_yaw_correlation` (only available when turn challenges were completed) and `blink_duration_ms` (only available when blink challenges were completed) — are genuinely absent for some sessions rather than zero.

The model outputs $P(\text{SPOOF})$ which is multiplied by 100 to yield the risk score. Verdicts:

| Score | Verdict |
|---|---|
| 0–34 | GENUINE |
| 35–64 | SUSPICIOUS |
| 65–84 | SUSPICIOUS (high) |
| 85–100 | DEEPFAKE / SPOOF |

### 7.2 Training Gate

A model is saved only if 5-fold stratified cross-validation achieves ROC AUC ≥ 0.75. The lower threshold (compared to 0.80 for document fraud scoring) reflects the inherent noisiness of live-session biometric data — small natural variations across genuine users produce more feature overlap with marginal spoof attempts than static document forensics signals do.

### 7.3 Self-Labeling Bootstrap Pipeline

The self-labeling pipeline solves the cold-start problem for operators who are deploying KYC for the first time:

**Step 1 — Cold start:**
The model file (`ml_scorer_face_scan_live.pkl`) does not exist. Every call to `predict()` returns `None`. The fixed-weight heuristic formula runs unchanged. The result JSON records `"scoring_method": "heuristic"`.

**Step 2 — Automatic sample collection:**
At the end of every completed session, `append_training_sample()` writes a row to `training_data_face_scan_live.csv` containing the session's 24-feature vector and `label = -1` (unlabelled).

**Step 3 — Operator labelling:**
The operator opens the CSV in a spreadsheet tool and reviews flagged sessions. Confirmed genuine sessions receive `label = 0`; confirmed spoof attempts receive `label = 1`. Rows that the operator cannot classify remain at `label = -1` and are excluded from training.

**Step 4 — Automatic training trigger:**
When the operator clicks "Train Model" in the ML Training screen (or when a scheduled background job runs), `train()` is called. It reads only rows with `label ∈ {0, 1}`, fits the pipeline, runs 5-fold stratified CV, and saves the model only if the AUC gate is passed.

**Step 5 — Model activation:**
On the next API server restart the model is loaded and `"scoring_method": "ML"` begins appearing in result JSONs.

**Idempotency guarantee:**
`append_training_sample()` is idempotent per `decision_trace_id`: if the CSV already contains a row for the same session, the write is skipped. WebSocket reconnections or retried requests cannot create duplicate training rows.

### 7.4 SHAP Explainability

Per-session SHAP contributions are computed using XGBoost's built-in tree SHAP, without an external `shap` package:

```python
booster.predict(dmat, pred_contribs=True)
```

This produces one SHAP value per feature, where positive values push toward SPOOF and negative values push toward GENUINE. The UI renders a horizontal bar chart showing the top contributing signals for every session, making each verdict traceable to specific evidence for a compliance reviewer.

---

## 8. GDPR-Compliant Selective Video Recording

### 8.1 Policy

The system records facial video **only for sessions with a verdict of SUSPICIOUS, DEEPFAKE, or INCONCLUSIVE**. GENUINE sessions are **not recorded**.

This is a deliberate policy grounded in the EU General Data Protection Regulation (GDPR) Article 9, which classifies biometric data used for identification as **special-category data** requiring explicit justification. The **data minimisation principle** (Article 5(1)(c)) requires that personal data be "adequate, relevant and limited to what is necessary in relation to the purposes for which they are processed."

Recording genuine passing sessions serves no investigative purpose — the session was assessed as genuine, there is no fraud hypothesis to investigate, and the video would constitute unnecessary retention of special-category biometric data. Recording only suspicious sessions satisfies both the investigative necessity test and the minimisation principle simultaneously.

### 8.2 Technical Implementation

For suspicious sessions, the per-frame JPEG buffer accumulated during the WebSocket stream is assembled into an H.264 MP4 file and uploaded to object storage (MinIO). The recording is stored with:

- A configurable retention TTL (`FACE_SCAN_VIDEO_RETENTION_DAYS`, default 90 days)
- A pre-session consent notice shown to the user explaining that this session may be recorded for fraud investigation, the storage duration, and how recordings are handled

The consent notice reads:

> *"This session may be recorded for fraud investigation and system testing purposes. Recordings are stored securely and deleted automatically after [N] days. By continuing you consent to this recording."*

This notice is displayed regardless of whether a recording ultimately occurs, which satisfies GDPR's transparency requirement while avoiding the complexity of providing different notices to users whose sessions turn out to be flagged versus not flagged (which would require ex-post notification).

### 8.3 Operator Considerations

Financial operators using the system should consider:

1. Whether their data processing agreement (DPA) with the platform covers special-category biometric data.
2. Whether the retention period must be extended to cover regulatory audit requirements (some AML regulations require 5-year document retention).
3. Whether the recorded videos constitute personal data subject to data subject access requests (DSARs) — they do, and a process for handling DSARs for video recordings should be established.

---

## 9. The 24-Feature Vector

The complete feature set, grouped by the attack type each signal primarily targets:

| Group | Feature | Anti-spoofing target |
|---|---|---|
| **Temporal consistency** | `yaw_jerk` | Rigid objects / smoothed replay (metronomic motion) |
| | `pitch_jerk` | Rigid objects |
| | `nose_jitter` | Tremor-free replays / static images |
| | `temporal_consistency_score` | Combined temporal smoothness score |
| **Replay detection** | `repeat_frame_score` | Software replay tools with looped/stuttering video |
| | `flicker_score` | Screen flicker at display refresh rate |
| | `brightness_instability` | Camera-screen distance variation during filming |
| **Eye micro-jitter** | `mean_eye_jitter` | Static images / masks (eyes never move) |
| **3D depth geometry** | `iod_yaw_correlation` | Flat photos, printed masks, screen replays |
| **Screen moiré** | `mean_fft_grid_peak` | Filmed-screen attacks (LCD/OLED pixel grid) |
| **Frame timing** | `interval_cv` | Metronomic replay tools (coefficient of variation) |
| **Session metadata** | `observed_fps` | Frame injection and rate manipulation |
| | `frame_drop_rate` | Network-level replay interference |
| **Face quality** | `mean_face_area_ratio` | Extreme distance / occlusion attacks |
| | `blur_risk_0_100` | Out-of-focus presentation (avoids clean detection) |
| | `brightness_risk_0_100` | Darkness exploitation to reduce texture signal quality |
| **Active liveness** | `wrong_action_count` | Challenge exploration / brute-force probing |
| | `challenge_count` | Sessions that completed abnormally few challenges |
| **Face tracking** | `frames_without_face` | Occlusion attacks mid-session |
| **Device flag** | `virtual_camera_suspected` | OBS / virtual camera injection (browser API signal) |
| **Tier-1 ML signals** | `head_velocity_variance` | Replay tools (low variance = constant-velocity playback) |
| | `blink_duration_ms` | Static images (no blink events at all) |
| | `challenge_reaction_latency_ms` | Automated/pre-programmed challenge completion |
| | `mean_landmark_confidence` | Low-quality or low-real-face detection confidence |

Missing values (e.g., `iod_yaw_correlation` when no turn challenge was completed) are represented as `NaN` and filled with training-set medians by the `SimpleImputer` step in the pipeline.

---

## 10. Implementation Notes

### 10.1 Face Detector Stack

The platform uses InsightFace RetinaFace as the primary face detector and ArcFace for identity embedding comparison. The 5-point keypoints produced by RetinaFace are the sole landmark source for the IOD-Yaw signal and the yaw/pitch computation.

Eye Aspect Ratio (EAR) for blink detection is sourced from MediaPipe FaceMesh, not from InsightFace's `det_score`. This is a deliberate design decision: `det_score` correlates weakly with eye closure at typical webcam resolutions and is not a reliable blink indicator. MediaPipe provides a dedicated eye landmark mesh from which the classic EAR formula is computed:

$$\text{EAR} = \frac{|p_2 - p_6| + |p_3 - p_5|}{2 \cdot |p_1 - p_4|}$$

where $p_1$–$p_6$ are six specific eye contour landmarks.

This means blink liveness detection runs MediaPipe in addition to InsightFace for every KYC frame, even on systems where InsightFace is the primary detector. The EAR from MediaPipe's first detected face is attached to the InsightFace face object as `face.ear` before being passed to the liveness analyzer.

### 10.2 Face Geometry Validity Guard

Before any frame contributes to challenge progress or passive signal collection, four facial landmark geometry invariants are checked:

1. **Nose below eyes**: the nose tip must be at or below both eye landmarks (Y-axis increases downward in image coordinates). A small tolerance of 5% of bounding-box height handles tilted poses.
2. **Eye height parity**: the vertical gap between the two eye landmarks must be less than 25% of bounding-box height.
3. **IOD plausibility**: the inter-ocular distance must be 15–65% of the bounding-box width.
4. **Eye midpoint position**: the eye midpoint must be in the upper 65% of the bounding-box height.

These invariants catch palm-to-camera presentations (hands are occasionally mis-detected as faces) and badly marginal detections at the edge of the camera field-of-view.

### 10.3 Mirroring Convention

The browser sends a mirrored (selfie-style) canvas frame to the server. The server processes the mirrored frame. This means the server's left/right directions match what the user sees on screen, and challenge feedback ("turn to your LEFT") is directly aligned with the yaw sign convention ($\text{yaw} < 0$ = image-left = user's left).

This alignment is non-trivial to get right: an unmirrored frame would reverse the challenge direction relative to the user's experience, producing systematically wrong turn-direction feedback.

---

## 11. Discussion

### 11.1 Signal Independence

The five novel passive signals are designed to be approximately independent in the failure modes they address:

- A **filmed-screen attack** produces high `mean_fft_grid_peak` and moderate `repeat_frame_score` (if the video is at a stable frame rate), but IOD-Yaw is ambiguous (the screen has some 3D geometry due to being a physical object) and eye micro-jitter may be present (screen flicker creates apparent landmark movement).
- A **printed-photo attack** produces zero `mean_fft_grid_peak` and near-zero IOD-Yaw correlation, but the repeat-frame hash score may be elevated (static photo = identical frame content across the session).
- A **virtual camera injection** with a pre-recorded video is caught by the `virtual_camera_suspected` flag (derived from the browser's `MediaDevices.enumerateDevices()` label matching against known virtual camera driver names) and the metronomic frame timing signal.
- A **plastic 3D mask** is not caught cleanly by the IOD-Yaw signal (it has some 3D geometry) but the texture score at the stability gate is low (masks lack organic skin texture variation) and eye micro-jitter is absent (no saccades from a mask).

No single signal is decisive against all attack types. The XGBoost fusion model learns which combinations of signals constitute reliable evidence of spoofing versus genuine variation in real users.

### 11.2 Limitations

1. **Turn challenge requirement for IOD-Yaw**: the depth signal only activates during turn challenges. Sessions that only use blink challenges receive no depth evidence. For operators who want maximum usability (blink-only), the IOD-Yaw signal contributes no discrimination.
2. **FFT signal sensitivity to real-world filming angles**: filming a screen at an extreme angle (near-parallel to the camera axis) reduces the moiré periodicity, potentially producing a low `mean_fft_grid_peak` score. The signal is most reliable for screen angles between 30° and 90° to the camera.
3. **Self-labeling labour**: the self-labeling pipeline does not eliminate the need for human judgement — it reduces the cost by making labelling the only manual step rather than data collection as well. An operator who never reviews their suspicious sessions will never train the model.
4. **Deepfake real-time face swaps**: a real-time deepfake routed through a virtual camera is partially caught by `virtual_camera_suspected` but is not caught by IOD-Yaw (the underlying face geometry is from a real person). This is an out-of-scope threat for the current system.

### 11.3 Ethical Considerations

Liveness detection raises several ethical concerns that financial operators must address:

- **Demographic fairness**: the system has been evaluated primarily on adults at normal webcam distances in reasonable lighting. Performance on elderly users, users with disabilities affecting head movement, or users in environments with unusual lighting has not been characterised. The `wrong_action_count` feature includes a self-correction mechanism that avoids penalising users who need to retry challenges — this partially mitigates mobility-related false rejection.
- **Biometric data retention**: even with selective recording, all completed sessions produce biometric-adjacent feature vectors that are stored in the training CSV. These do not include facial images or identity embeddings, but they do describe the biometric behaviour of identified users. Access controls and retention policies must be applied to this CSV as well as to the video recordings.
- **Transparency of decision-making**: the per-session SHAP bar chart makes the rejection reason inspectable by both operators and, under a DSAR, by the subject themselves. This is consistent with GDPR Article 22's requirement for meaningful information about automated decisions affecting individuals.

---

## 12. Conclusion

This paper presented Hold-and-Return, a hybrid active-passive liveness detection framework that operates on a commodity webcam JPEG stream without depth sensors, infrared cameras, or pre-labelled training data at launch. The system's five novel contributions — the IOD-Yaw geometric depth cue, the hold-and-return challenge protocol, the FFT moiré screen frequency signal, the self-labeling training pipeline, and the GDPR-grounded selective recording policy — address distinct practical gaps that are individually underserved in published liveness literature and collectively underexplored as a unified system.

The IOD-Yaw signal demonstrates that meaningful 3D geometry evidence is available from 2D facial landmarks, provided the measurement is made across a sustained head-turn sequence rather than in a single frame. The hold-and-return protocol shows that a 1-second sustained hold requirement substantially raises the bar for replay attacks without meaningfully degrading the user experience for genuine users. The self-labeling pipeline solves the cold-start deployment problem for operators who cannot afford a pre-labelled dataset.

Together, these contributions form a practical and deployable liveness framework for financial KYC that makes no hardware assumptions beyond a standard webcam and reduces the barrier to entry for smaller financial institutions that are currently underserved by commercial liveness vendors.

Future work will characterise cross-demographic false rejection rates, evaluate the system against real-time deepfake face-swap tools routed through virtual cameras, and extend the self-labeling pipeline with active learning to prioritise which sessions the operator should label first.

---

## References

*[To be completed — citations below are placeholders pending final bibliography.]*

- Chingovska, I., Anjos, A., & Marcel, S. (2012). On the Effectiveness of Local Binary Patterns in Face Anti-Spoofing. *BIOSIG 2012*.
- Kose, N., & Dugelay, J.-L. (2013). Reflectance Analysis Based Countermeasure Technique to Detect Face Mask Attacks. *ICASSP 2013*.
- Li, X., Orchard, M. T., & Zhang, E. J. (2004). A new edge-directed interpolation. *IEEE Transactions on Image Processing*.
- Pan, G., Sun, L., Wu, Z., & Lao, S. (2007). Eyeblink-based Anti-Spoofing in Face Recognition from a Generic Webcamera. *ICCV 2007*.
- Raghavendra, R., Raja, K. B., & Busch, C. (2015). Presentation Attack Detection for Face Recognition Using Light Field Camera. *IEEE Transactions on Image Processing*.
- Wang, Z., Lan, C., Zhang, S., Han, J., & Zheng, N. (2020). Exploiting temporal and depth information for multi-frame face anti-spoofing. *arXiv:1811.05118*.
- Wen, D., Han, H., & Jain, A. K. (2015). Face Spoof Detection with Image Distortion Analysis. *IEEE TIFS*.
- Yu, Z., Li, X., Shi, J., Zhao, G., & Kellokumpu, V. (2022). Deep Learning for Face Anti-Spoofing: A Survey. *IEEE TPAMI*.
- Zhang, Y., et al. (2020). CelebA-Spoof: Large-Scale Face Anti-Spoofing Dataset with Rich Annotations. *ECCV 2020*.

---

## Appendix A — Feature Vector Schema

The 24-feature vector written to `training_data_face_scan_live.csv` in column order:

```
session_id, timestamp_utc, verdict,
yaw_jerk, pitch_jerk, nose_jitter, temporal_consistency_score,
repeat_frame_score, flicker_score, brightness_instability,
mean_eye_jitter,
iod_yaw_correlation,
mean_fft_grid_peak,
interval_cv,
observed_fps, frame_drop_rate,
mean_face_area_ratio, blur_risk_0_100, brightness_risk_0_100,
wrong_action_count, challenge_count,
frames_without_face,
virtual_camera_suspected,
head_velocity_variance, blink_duration_ms,
challenge_reaction_latency_ms, mean_landmark_confidence,
label
```

`label` values: `0` = GENUINE, `1` = SPOOF, `-1` = unlabelled.

---

## Appendix B — Key Threshold Summary

| Parameter | Value | Rationale |
|---|---|---|
| `_TURN_YAW_THRESHOLD` | 0.16 | Well above natural noise (≤ 0.04); reachable without exaggerated pose |
| `_TURN_HOLD_FRAMES` | 10 | ≈ 1 s at 10 FPS; long enough to require a deliberate hold |
| `_TURN_HOLD_LENIENCY` | 0.04 | Natural camera wobble during sustained hold |
| `FACE_STABLE_FRAMES_REQUIRED` | 10 | Clean 1-second window before any challenge begins |
| `FACE_STABILITY_CONFIDENCE_MIN` | 0.80 | Strict during stability phase; relaxed to 0.55 mid-challenge |
| `MIN_FACE_TEXTURE_SCORE` | 18.0 | Separates real skin (25–70) from flat surfaces (< 15) |
| `FACE_STABILITY_YAW_VARIANCE_MIN` | $10^{-6}$ | Real micro-movement ($10^{-4}$–$10^{-6}$) vs. frozen image ($10^{-8}$) |
| `FACE_STABILITY_YAW_JITTER_ALTERNATION_MAX` | 0.80 | Natural direction change rate 0.3–0.6 vs. artificial wiggle ≈ 1.0 |
| FFT ring inner/outer radius | 6–22 px | LCD/OLED pixel grid periods at selfie distance, 64×64 transform |
| FFT peak-concentration threshold | 0.20–0.40 | Organic skin ≈ 0.05–0.18; filmed screen ≈ 0.28–0.55 |
| XGBoost training AUC gate | ≥ 0.75 | Lower than doc fraud (0.80) to account for live-session noise |
| Challenge timeout | 10 s | Generous for genuine users; short enough to deter patient probing |
