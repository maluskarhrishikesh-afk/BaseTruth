# Face Scan Live — ML Evolution Plan

**Created:** 2026-05-07  
**Updated:** 2026-05-08  
**Status:** Phase 1 IMPLEMENTED — XGBoost scorer, CSV data collection, and heuristic calibration fixes are all live  
**Trigger:** First successful live scan verdict (GENUINE, risk 29.15, all 4 challenges passed)  
**Expert review:** See session notes for full JSON report

---

## Baseline Assessment (Original System, 2026-05-07)

The first real live scan confirmed the heuristics-only pipeline works end-to-end:

| What worked | Result |
|---|---|
| Active liveness (4 challenges) | ✅ All passed: look_straight, nod, turn_left, turn_right |
| Wrong-action self-correction | ✅ User corrected turn_right×2 — system allowed it |
| Depth consistency | ✅ Strong — `iod_yaw_correlation = -0.9398` |
| Screen frequency | ✅ Pass |
| Frame timing | ✅ Pass |
| Eye micro-jitter | ✅ Pass |

| Known weak spots | Value | Why it matters |
|---|---|---|
| Temporal consistency | 37.51 / 100 | Jerky motion — caused by low FPS |
| Repeat frame score | 56.95 / 100 | Webcam driver buffering at 10 FPS creates near-duplicate frames |
| Processing time | 223 766 ms (~3.7 min) | Far above the 3–10 s production target |
| Observed FPS | 10.0 | Low — root cause of temporal and replay signals |

The system correctly trusted challenge-response over the weaker passive signals. Final verdict was GENUINE despite two `"review"` signals — the right decision for a real user.

---

## What Has Been Implemented (as of 2026-05-08)

### Heuristic Fixes and Calibration

#### Default challenge list
`DEFAULT_FACE_SCAN_CHALLENGES` changed from `["blink", "turn_left", "nod"]` to `["blink", "nod", "turn_left", "turn_right"]`.
The API always prepends `look_straight` automatically, making the full default sequence:
**look_straight → blink → nod → turn_left → turn_right**.
`turn_right` was previously missing entirely from the default set.

#### Relative-delta turn thresholds tightened
`_TURN_RELATIVE_YAW_DELTA` raised from 0.09 → **0.12**  
`_TURN_RELATIVE_NOSE_SHIFT` raised from 0.10 → **0.12**  
After the stability gate passes, the user's resting baseline may be slightly off-centre (e.g. yaw = -0.04). Drifting back to neutral from that baseline was accumulating a delta of ~0.09 and triggering a false turn pass. The tighter 0.12 threshold requires a deliberate bilateral head movement.

#### Stability gate reset only on turn challenges
Previously the stability gate reset (`face_stable_frames = 0`) after every challenge completion. Now it resets **only after completing a turn challenge** (`turn_left` or `turn_right`). Non-turn challenges (`look_straight`, `blink`, `nod`) leave the gate state intact, eliminating the 4–5 retry cycle that users experienced.

#### SUSPICIOUS verdict dual-guard
The verdict path `max(replay, temporal) >= 50 → SUSPICIOUS` now requires **both** the sub-score being elevated **and** `risk_score >= 35.0`. A single sub-score spike from a low-light dark room (hash collisions from near-identical frames) can no longer override a healthy overall risk score and produce a false SUSPICIOUS for a genuine user.

#### Depth consistency recalibrated for doll/mask detection
The depth scoring formula changed from `_scale_risk(corr + 1.0, 0.70, 1.20)` to `_scale_risk(corr + 1.0, 0.50, 1.00)`.
Calibrated against real session data:
- Real human: `iod_yaw_correlation` typically −0.40 to −0.99 → risk 0
- Plastic doll (measured at −0.12): now scores ~77 → exceeds the 65-point SUSPICIOUS threshold ✓  
- Flat photo/mask (≈0.0 to +0.30): scores 100 → SUSPICIOUS ✓  

Previous thresholds (0.70, 1.20) left the doll at ~37, below the 65-point gate.

A standalone depth gate was also added: `depth_score >= 65 → SUSPICIOUS` fires independently of the dual-guard sub-score path. This catches flat objects (masks, printed photos, palm attacks) that pass all challenges but fail the IOD geometry test.

#### Face size risk recalibrated for live-scan oval
`_scale_risk(mean_face_area, 0.06, 0.18, inverse=True)` changed to `_scale_risk(mean_face_area, 0.05, 0.10, inverse=True)`.
Real live sessions with the face in the browser oval produce `mean_face_area` of 0.07–0.12. The old (0.06, 0.18) scale scored these at 28–83% risk, triggering the "face too small" narrative message even when the face was correctly positioned. The new (0.05, 0.10) scale treats 10%+ area as zero risk — the "face too small" message now only fires for genuinely small faces.

#### Texture threshold for dark frames lowered
`LOW_BRIGHTNESS_TEXTURE_SCORE` lowered from 14.0 → **12.0**. Prevents the stability gate from falsely rejecting genuine users in moderately dim rooms.

#### Nod range threshold lowered
`_NOD_RANGE_THRESHOLD` lowered from 0.12 → **0.10**. Accepts slightly smaller nods, improving usability for users with limited neck mobility.

---

### Phase 1 — XGBoost Fusion Scorer (IMPLEMENTED)

`src/basetruth/face_scan/ml_scorer_live.py` is live.

| Component | Status |
|---|---|
| `build_feature_vector(checks, environment)` | ✅ Extracts 20 float features from the result dict |
| `predict(feature_vector)` | ✅ Loads XGBoost model, returns risk_score + scoring_method; returns None on cold-start |
| `explain(feature_vector)` | ✅ Returns per-feature SHAP contributions (XGBoost tree SHAP, no external package) |
| `append_training_sample(result, label)` | ✅ Appends one CSV row per completed session to `fraud_model/data/training_data_face_scan_live.csv` |
| `train(csv_path, output_pkl)` | ✅ SimpleImputer → XGBClassifier pipeline; 5-fold CV; saves only if AUC ≥ 0.75 |
| Heuristic fallback | ✅ When model file absent, `predict()` returns None and existing heuristic runs unchanged |
| `scoring_method` field | ✅ `"ML"` or `"heuristic"` in every result JSON |
| Unit tests | ✅ `tests/test_ml_scorer_live.py` — 16 tests, all passing |

**Current state:** Cold-start (no model file yet). The heuristic formula runs. Every session appends a training row to the CSV with `label=-1` (unconfirmed). Once operators label confirmed genuine and spoof sessions and run `train()`, the model activates automatically on next server restart.

**CSV schema (23 columns):**  
`session_id, timestamp_utc, verdict, yaw_jerk, pitch_jerk, nose_jitter, temporal_consistency_score, repeat_frame_score, flicker_score, brightness_instability, mean_eye_jitter, iod_yaw_correlation, mean_fft_grid_peak, interval_cv, observed_fps, frame_drop_rate, mean_face_area_ratio, blur_risk_0_100, brightness_risk_0_100, wrong_action_count, challenge_count, frames_without_face, virtual_camera_suspected, label`

---

---

## What Should Not Change

These components work correctly and should remain rule/state-machine based:

- **Active challenge FSM** — challenge orchestration, wrong-action handling, timeouts
- **Hard-reject heuristics** — no face, multiple faces, virtual camera flag, FPS too low, challenge timeout, impossible motion
- **Narrative engine** — Gemma4 LLM explanation layer
- **SHAP explainability** — already planned for Phase 1 XGBoost (mirrors `ml_scorer.py` pattern)

---

## Architecture Target

```
Current:
  final_risk_score = fixed_weights(replay, temporal, quality, eye_jitter, screen, timing, depth)

Phase 1:
  final_risk_score = XGBoost(tabular_features)    ← replaces manual formula
  fallback         = current heuristics            ← when .pkl model is absent

Phase 2:
  final_risk_score = XGBoost(tabular + replay_cnn_prob + passive_liveness_score)

Phase 3 (high-risk escalation only):
  if 30 < final_risk_score < 70:
      escalate → temporal_transformer(face_sequence)
      final_risk_score = blend(xgb_score, transformer_score)
```

---

## Phase 1 — XGBoost Fusion Scorer

**Goal:** Replace the fixed-weight `risk_score` formula with a trained XGBoost model that learns optimal signal weights from labeled session data.

**Status: IMPLEMENTED** — see “What Has Been Implemented” section above for full details. The architecture below describes the design as originally planned; it has been built as specified.

### Input Feature Vector (all already computed today)

| Feature | Source function |
|---|---|
| `yaw_jerk` | `_compute_temporal_consistency()` |
| `pitch_jerk` | `_compute_temporal_consistency()` |
| `nose_jitter` | `_compute_temporal_consistency()` |
| `temporal_consistency_score` | `_compute_temporal_consistency()` |
| `repeat_frame_score` | `_compute_replay_heuristics()` |
| `flicker_score` | `_compute_replay_heuristics()` |
| `brightness_instability` | `_compute_replay_heuristics()` |
| `mean_eye_jitter` | `_compute_saccade_analysis()` |
| `iod_yaw_correlation` | `_compute_depth_consistency()` |
| `mean_fft_grid_peak` | `_compute_screen_frequency()` |
| `interval_cv` | `_compute_frame_timing()` |
| `observed_fps` | session metadata |
| `frame_drop_rate` | session metadata |
| `mean_face_area_ratio` | `_compute_quality_assessment()` |
| `blur_risk_0_100` | `_compute_quality_assessment()` |
| `brightness_risk_0_100` | `_compute_quality_assessment()` |
| `wrong_action_count` | active liveness result |
| `challenge_count` | active liveness result |
| `frames_without_face` | face detection check |
| `virtual_camera_suspected` | virtual camera flag (0/1) |

**Output:** `spoof_probability (0–1)` → `risk_score_0_100 = spoof_probability × 100`

### New Files

| File | Purpose |
|---|---|
| `src/basetruth/face_scan/ml_scorer_live.py` | Feature vector builder + `predict()` + `train()` + `explain()` |
| `data/face_scan_live_training.csv` | Labeled session data (label: 0=genuine, 1=spoof) |
| `data/face_scan_live_scorer.pkl` | Trained XGBoost pipeline (SimpleImputer → XGBClassifier) |
| `scripts/collect_live_scan_samples.py` | Export sessions from DB/JSON to labeled CSV rows |
| `tests/test_ml_scorer_live.py` | Unit tests (mirror pattern from `tests/test_ml_scorer.py`) |

### Architecture Pattern

Mirror `src/basetruth/analysis/ml_scorer.py` exactly:
- `SimpleImputer(strategy="median") → XGBClassifier` pipeline
- `predict(feature_dict)` returns `(score_0_100, verdict, scoring_method)`
- `explain(feature_dict)` returns SHAP contributions dict (no external `shap` package — use XGBoost tree SHAP via `predict_contribs`)
- Heuristic fallback when `.pkl` is absent
- `scoring_method` = `"ML"` or `"heuristic"` flows into the result JSON

### Integration Point

`src/basetruth/face_scan/live.py` → `_compute_final_result()`:  
Replace the manual weighted-sum with `ml_scorer_live.predict(feature_vector)` when model file exists.

### Training Data Strategy

| Split | Source |
|---|---|
| Genuine | Real sessions from real users (export from `your_data/` or DB) |
| Spoof | Synthetic — reuse genuine session JSON, zero out variance fields, inflate `repeat_frame_score`, set `virtual_camera_suspected=True`, add screen-replay field patterns |

- Minimum viable: ~50 rows  
- Production target: 500+  
- Label column: `label` (0 = genuine, 1 = spoof)

---

## Phase 1b — FPS Improvement (do first, unblocks everything)

**Goal:** Raise `observed_fps` from 10 → 20+. Almost everything else improves automatically.

**Impact cascade:**
- `repeat_frame_score` drops (fewer duplicate frames from buffering)
- `temporal_consistency` improves (smoother motion curves)
- `processing_time_ms` falls (fewer stacked frames to process at session end)

**Root cause candidates to investigate:**

1. **Browser canvas capture interval** — currently `setInterval` in the live HTML JS; lower from 100 ms → 50 ms
2. **API WebSocket frame queue** — skip frames if queue depth > 1 to prevent backlog  
3. **Server-side profiling** — InsightFace init on first frame is ~3 s; subsequent frames should be < 50 ms

**Files to change:** `src/basetruth/face_scan/live.py` (HTML JS `setInterval`), possibly `_process_kyc_frame` in `api.py`

---

## Phase 2 — Replay CNN

**Goal:** Replace hash-based `repeat_frame_score` with a lightweight CNN trained on real screen-replay texture artifacts.

**Why heuristics are not enough:** Average-hash repeat detection fires on genuine low-FPS webcam buffering (as seen in the first scan: score=56.95 despite GENUINE verdict). A CNN trained on actual screen photos learns moiré patterns, refresh-line artifacts, and pixel-grid signatures — signals that the hash check cannot distinguish from natural low-FPS jitter.

**Approach:**

| Parameter | Value |
|---|---|
| Model | MobileNetV3-Small or EfficientNet-Lite0 (< 3 MB) |
| Input | 96×96 face crop |
| Output | `screen_replay_probability (0–1)` |
| Inference format | ONNX (consistent with InsightFace deployment) |
| Training data | Genuine face crops vs. crops photographed from phone/laptop screens |

**New files:**
- `src/basetruth/face_scan/replay_cnn.py`
- `data/replay_cnn.onnx`

**Integration point:** `_compute_replay_heuristics()` in `live.py` — run CNN per-frame, average probability, replace or supplement `repeat_frame_score`.

---

## Phase 2b — Passive Liveness Model

**Goal:** Add a per-frame liveness score that does not require active challenges.

**Approach:** XGBoost on per-frame features:
- texture score (already computed for stability gate)
- EAR stability
- IOD variance
- brightness gradient
- FFT peak

Aggregated across all frames → `passive_liveness_score`.  
Used as an additional input feature for the Phase 1 fusion scorer.

---

## Phase 3 — Temporal Transformer (high-risk escalation only)

**Goal:** Catch sophisticated deepfakes and high-quality replay attacks that pass all earlier layers.

**Trigger condition:** Only when Phase 1 XGBoost score falls in the 30–70 "review" band. Clear genuine (< 30) or clear spoof (> 70) sessions are not escalated.

**Approach:**

| Parameter | Value |
|---|---|
| Model | Small VideoMAE or TimeSformer fine-tuned on face sequences |
| Input | 16-frame face crop sequence at 112×112 |
| Output | `deepfake_probability (0–1)` |
| Inference format | ONNX |

**New files:**
- `src/basetruth/face_scan/temporal_model.py`
- `data/temporal_liveness.onnx`

---

## Implementation Order

| Priority | Phase | Effort | Status |
|---|---|---|---|
| 1 | Phase 1b — FPS fix | 1 day | ⏳ Pending |
| 2 | Phase 1 — XGBoost fusion scorer | 2–3 days | ✅ IMPLEMENTED (cold-start; needs labeled data) |
| 3 | Heuristic calibration fixes (default challenges, relative-delta thresholds, face size, depth, verdict guard) | Completed | ✅ IMPLEMENTED |
| 4 | Phase 2 — Replay CNN | ~1 week | ⏳ Pending (needs labeled training data) |
| 5 | Phase 2b — Passive liveness | 1–2 days | ⏳ Pending (after replay CNN) |
| 6 | Phase 3 — Temporal transformer | Several weeks | ⏳ Pending (needs 500+ real sessions) |

---

## What This Enables (Remaining Coverage Gaps)

| Attack type | Current coverage | Covered after |
|---|---|---|
| Webcam buffering false positives | ⚠️ Partially mitigated (dual-guard verdict fix) | Phase 1b (FPS) + Phase 2 (CNN) fully resolve |
| Plastic doll / close-fitting 3D mask | ✅ Covered by recalibrated depth consistency (65-point gate) | — |
| Flat printed photo | ✅ Covered (depth 100, + static-scan halo/compression checks) | — |
| Smooth deepfake video | ⚠️ Partial (motion + depth checks) | Phase 3 (transformer) |
| Mask attack (opaque 3D) | ❌ Not covered | Requires IR sensor or dedicated anti-spoof model |
| GAN live stream injection | ❌ Not covered | Phase 3 (transformer) |

---

## Related Files

| File | Role |
|---|---|
| `src/basetruth/face_scan/live.py` | Integration point for all ML scorers |
| `src/basetruth/analysis/ml_scorer.py` | Pattern to mirror for `ml_scorer_live.py` |
| `src/basetruth/kyc/liveness.py` | Challenge constants and stability gate |
| `docs/FACE_SCAN_WORKING.md` | Full behaviour specification |
| `docs/ML_PIPELINE.md` | Document/image ML pipeline reference |
| `docs/ML_SCORING_IMPLEMENTATION_PLAN.md` | Image ML scorer plan (already implemented) |
