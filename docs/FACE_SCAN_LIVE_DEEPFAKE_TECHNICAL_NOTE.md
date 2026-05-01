# Face Scan Live Deepfake Detection Technical Note

## Short Answer

Yes, we can detect some classes of deepfakes and spoofing attempts during liveness challenges.

But the correct technical position is:

- liveness challenges can materially improve live deepfake detection,
- they are very effective against replay attacks, screen attacks, printed-photo attacks, and weak face-swap pipelines,
- they are not a guarantee against all high-end real-time generative deepfakes.

So the product should present this as **live face authenticity and spoof-risk detection**, not as an absolute promise that every deepfake will be caught.

## Why Liveness Helps Deepfake Detection

Static-image detection only sees one frame. A liveness session gives us a time series.

That unlocks signals that are much harder for a spoof or live deepfake system to fake consistently:

- blink timing and eyelid dynamics,
- head-pose continuity during challenge-response motion,
- mouth and eye landmark consistency over time,
- optical-flow smoothness across adjacent frames,
- remote photoplethysmography (rPPG) pulse consistency,
- screen replay artefacts such as moire, glare, and refresh-banding,
- latency between challenge prompt and human response,
- face-boundary blending instability during motion.

This is why live challenge mode should be a core part of the Face Scan roadmap.

## Threat Models To Cover

Face Scan should explicitly target these threat classes:

### 1. Printed photo attack

Examples:

- printed passport photo held to camera,
- phone showing a still selfie,
- tablet displaying a cropped headshot.

Typical indicators:

- flat surface geometry,
- specular glare patterns,
- no true blink dynamics,
- weak 3D head motion,
- paper or screen edges in the frame,
- moire or pixel-grid artefacts.

### 2. Replay attack

Examples:

- pre-recorded video replayed on another phone,
- laptop/webcam feed pointed at a screen.

Typical indicators:

- challenge timing mismatch,
- repeated compression artefacts,
- refresh-line banding,
- double camera noise chain,
- screen glare and moire,
- inaccurate challenge-response alignment.

### 3. Live face swap / live deepfake

Examples:

- webcam feed altered in real time,
- avatar or face-reenactment software,
- GPU-based live face replacement in conferencing software.

Typical indicators:

- unstable face boundary during head turns,
- warped mouth/teeth/eye regions,
- inconsistent skin texture across time,
- landmark jitter during expressions,
- temporal lag between head motion and rendered face motion,
- weak physiological cues such as rPPG inconsistency.

## Recommended Detection Strategy

Use a layered approach. Do not depend on one model or one heuristic.

Also add a separate uncertainty channel. The system should never output only a spoof-risk score. It should always output:

- `live_authenticity_risk_score_0_100`
- `confidence_0_100`
- `confidence_reason`

This is essential because weak lighting, motion blur, low FPS, or unstable tracking can make a low-risk score unreliable.

### Layer 1 - Existing active liveness challenges

Use the current challenge engine as the base gate:

- `look_straight`
- `blink`
- `nod`
- `turn_left`
- `turn_right`

Current reusable code:

- `src/basetruth/kyc/liveness.py`
- `src/basetruth/api.py` WebSocket frame flow
- `src/basetruth/vision/face.py` detector and landmarks

This layer answers:

- did the subject respond correctly,
- was the motion human-like enough,
- did the session produce a reliable frontal frame.

This alone is not enough for deepfake detection, but it is the correct first layer.

### Layer 2 - Temporal landmark consistency

Measure how facial landmarks evolve across time.

Recommended features:

- landmark velocity and acceleration smoothness,
- yaw and pitch continuity,
- eyelid opening/closing curve shape,
- mouth-corner and nose-tip motion coherence,
- landmark jitter score per region: eyes, nose, mouth, jaw.

Why it works:

- live swaps and reenactment systems often keep average pose correct but introduce frame-to-frame instability in local regions.

Recommended libraries:

- `MediaPipe FaceLandmarker` for dense landmarks,
- `NumPy` for time-series aggregation,
- `SciPy` for smoothing, derivatives, and signal statistics.

### Layer 3 - Optical-flow and face-boundary analysis

Measure whether the facial region moves like a real connected surface.

Recommended features:

- dense optical flow inside the face mask,
- motion coherence between central face and boundary pixels,
- residual motion around jawline, hairline, cheeks, and forehead,
- boundary halo score during head turns,
- face-mask versus background motion discrepancy.

Why it works:

- composited or generated faces often align the centre of the face better than the contour edges.

Recommended libraries:

- `OpenCV` Farneback optical flow or Lucas-Kanade tracking,
- `NumPy` for motion statistics,
- optional `scikit-image` for edge and mask morphology helpers.

### Layer 4 - Screen replay / presentation attack detection

Detect whether the camera is pointed at another display.

Recommended features:

- moire pattern score,
- horizontal or vertical refresh-band frequency peaks,
- rolling-shutter banding,
- repeated glare patches,
- pixel-grid / subpixel pattern suspicion,
- histogram spikes caused by backlit displays.

Why it works:

- replay attacks often pass face detection and even some simple challenge checks, but they leave display-specific artefacts.

Recommended libraries:

- `OpenCV`,
- `NumPy` FFT,
- optional `PyWavelets` or `SciPy FFT` for frequency-domain analysis.

### Layer 4A - Device and environment signals

Collect and fuse the environment metadata that is available from the browser and API session:

- browser family and version,
- OS family,
- platform type (`web`, later `android` or `ios` if native clients appear),
- reported camera resolution,
- observed FPS,
- frame drop rate,
- repeated-frame suspicion,
- virtual-camera suspicion.

Important note:

- browser-side virtual-camera detection is imperfect,
- so this must be treated as a risk signal, not an absolute block by itself.

Why it matters:

- many live deepfake attacks run through OBS, virtual webcams, meeting software filters, or emulator-like environments,
- ignoring environment signals leaves a major production blind spot.

### Layer 5 - rPPG physiological signal check

Estimate a weak pulse signal from facial color changes across a short temporal window.

Recommended features:

- pulse presence confidence,
- dominant frequency in plausible heart-rate range,
- temporal consistency of pulse estimate across forehead and cheek regions,
- pulse-to-motion robustness score.

Why it works:

- a live human face often contains a measurable remote photoplethysmography signal,
- screens, prints, and many generated streams do not preserve it reliably.

Important caveat:

- this signal is fragile under poor lighting, compression, low FPS, and strong motion,
- it should be a supporting signal, never a sole rejection rule.

Recommended libraries:

- `NumPy`,
- `SciPy signal`,
- optional later use of `pyVHR` concepts or an internal lightweight implementation instead of adding a heavy dependency immediately.

### Layer 6 - Optional learned anti-spoof classifier

After the deterministic stack is stable, add a model-based classifier as an extra signal, not as the only decision-maker.

Recommended options:

- ONNX anti-spoof model run locally with `onnxruntime`,
- a lightweight PyTorch anti-spoof model converted to ONNX for deployment,
- challenge-frame classifier over the best frontal frame plus a few temporal crops.

Recommended deployment rule:

- keep deterministic heuristics as the always-available baseline,
- treat the learned model as an additive confidence signal.

## Recommended Libraries

### Keep from current stack

- `OpenCV` for image processing, masks, optical flow, FFT pre-processing, blur, edge, glare, and replay heuristics.
- `MediaPipe FaceLandmarker` for dense landmarks, blink blendshapes, and reliable cross-platform live geometry.
- `InsightFace` for face detection and embeddings where available.
- `ONNX Runtime` for any future anti-spoof inference model.
- `NumPy` for feature extraction, arrays, temporal aggregation, and frequency-domain work.

### Add carefully

- `SciPy` for filtering, signal statistics, derivatives, peak detection, FFT helpers, and rPPG support.
- optional `scikit-image` for texture, morphology, and mask refinement.
- optional `PyWavelets` only if wavelet-based replay analysis proves useful in experiments.

### Do not add in v1 unless a real need appears

- heavyweight research-only deepfake packages with unclear licensing,
- cloud-only face APIs,
- model stacks that cannot run fully offline in the deployment environments already supported by BaseTruth.

## Recommended Algorithm Pipeline

The live Face Scan pipeline should look like this:

```text
Browser camera frames
  -> decode + resize
  -> face detection + landmarks
  -> challenge engine state update
  -> temporal feature buffers
  -> environment/device feature update
  -> replay / spoof heuristics
  -> rPPG supporting signal
  -> optional learned anti-spoof model
  -> score fusion
  -> confidence estimation
  -> verdict builder
  -> final Face Scan payload
```

## Proposed Feature Buffers

Maintain a rolling window of recent frames, for example 2 to 5 seconds depending on FPS.

Per-frame values to store:

- timestamp,
- face box,
- face area ratio,
- yaw, pitch, and optional roll,
- EAR / blink-related metrics,
- detector confidence,
- selected landmark coordinates,
- optical-flow summary values,
- face-boundary residual score,
- replay-frequency features,
- brightness and blur measures,
- small region color traces for rPPG,
- frame hash or fingerprint for repeated-frame detection.

Why this matters:

- the final decision should come from a temporal aggregate, not a single frame.

## Score Fusion Recommendation

Do not fuse everything into one opaque score without traceability.

Use sub-scores first:

- `challenge_response_score_0_100`
- `presentation_attack_score_0_100`
- `temporal_consistency_score_0_100`
- `physiology_score_0_100`
- `synthetic_artifact_score_0_100`
- `model_spoof_score_0_100` when optional model is enabled
- `environment_risk_score_0_100`

Then compute:

- `live_authenticity_risk_score_0_100`

Suggested initial fusion template from the expert review:

```text
final_score =
  0.25 * liveness +
  0.20 * temporal +
  0.20 * replay +
  0.15 * optical_flow +
  0.10 * quality +
  0.10 * rPPG
```

This is a starting point only. It must be calibrated later with real evaluation data.

Confidence should be computed separately, not derived directly from the same weights. It should be penalized by:

- low light,
- low FPS,
- heavy motion blur,
- intermittent landmark loss,
- very short sessions,
- inconsistent detector confidence.

Recommended verdict mapping:

- `0-24` -> `GENUINE`
- `25-49` -> `SUSPICIOUS`
- `50-74` -> `LIKELY SPOOFED`
- `75-100` -> `DEEPFAKE_OR_PRESENTATION_ATTACK`

The exact thresholds should be calibrated later using real evaluation data.

Suggested hard-rule ordering from the expert review:

- if `liveness_failed` -> `LIVENESS_FAILED`
- else if `replay_score > 80` -> `DEEPFAKE`
- else if temporal instability is high -> `SUSPICIOUS`
- else -> `GENUINE`

These rules are pre-calibration placeholders and should not be treated as final production thresholds.

## Operational Requirements For Production Use

Before a public Face Scan API is exposed, the live pipeline also needs:

- rate limiting per IP and per API credential,
- session TTL and expiry,
- max attempts per session,
- duplicate-frame / replay-loop protection,
- authenticated API access,
- schema versioning in every response,
- trace metadata: `decision_trace_id`, `rules_version`, `model_version`, `timestamp_utc`, `processing_time_ms`.

Recommended baseline performance targets:

- process about 10 FPS rather than every camera frame,
- use a 3-second sliding analysis window,
- keep max session duration around 20 seconds,
- target under 200 ms processing per analysed frame on the supported CPU path.

These are part of making the system safe, operable, and auditable, not optional extras.

## What We Should Say Publicly Versus Internally

### Public or operator-facing wording

Use wording like:

- `live authenticity check`,
- `presentation attack detection`,
- `deepfake and spoof-risk analysis`,
- `liveness and live face integrity signals`.

Avoid overclaiming with wording like:

- `guaranteed deepfake detection`,
- `100% AI face detection`,
- `proof that the face is real`.

### Internal or expert-facing wording

We can say:

- the system performs multi-signal live anti-spoof and deepfake-risk analysis,
- it is strongest against presentation attacks and weak to medium live face swaps,
- it is a probabilistic fraud detector, not a mathematical proof system.

## Recommended V1 Implementation Order

### Phase 1 - Deterministic live heuristics only

Implement first:

- current liveness challenge engine reuse,
- temporal landmark consistency score,
- optical-flow boundary score,
- replay-screen heuristics,
- final fused live-authenticity risk score.

Why first:

- fully offline,
- explainable,
- testable,
- no dataset dependency to start building the product surface.

### Phase 2 - rPPG as a supporting signal

Add:

- ROI color-trace extraction from forehead and cheeks,
- band-pass filtering,
- pulse plausibility confidence.

Keep it soft-weighted in the score fusion.

### Phase 3 - Optional learned anti-spoof model

Add after we have validation data:

- ONNX local anti-spoof classifier,
- calibration step,
- shadow-mode evaluation before making it decision-bearing.

## Evaluation Plan Before Final Product Decision

Before taking a final architecture decision, compare the approach against real examples in these buckets:

- genuine webcam sessions,
- low-light genuine sessions,
- glasses / beard / partial occlusion sessions,
- printed photo attacks,
- phone-screen replay attacks,
- tablet / laptop replay attacks,
- low-end live face swap tools,
- higher-quality live face reenactment tools,
- weak network / compressed sessions.

Also evaluate:

- virtual-camera inputs,
- duplicate-frame replay loops,
- emulator-like browser sessions,
- rapid repeated API probing from the same client.

Metrics to compare:

- false reject rate on genuine users,
- attack detection rate by threat type,
- challenge completion time,
- performance on CPU-only deployment,
- explainability quality of the final evidence output.

## Recommended Final Position For Expert Review

If you discuss this with other experts, the position I would recommend reviewing is:

- **Yes**, BaseTruth should use liveness sessions to detect deepfake and spoof-risk signals.
- **No**, we should not rely on liveness challenges alone as a binary deepfake oracle.
- **Best design:** layered live anti-spoof detection built on top of challenge-response, temporal consistency, replay heuristics, rPPG support, and optional ONNX anti-spoof models.

That is the most defensible technical path for an offline-first product like BaseTruth.