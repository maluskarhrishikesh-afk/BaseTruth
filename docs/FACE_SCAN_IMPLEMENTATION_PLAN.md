# Face Scan Implementation Plan

## Goal

Turn the new Face Scan screen into a production-ready verification surface that can:

1. analyse a static face image for fake-photo and deepfake signals,
2. run active liveness challenges from the browser,
3. return one clean final result payload in the same spirit as Scan Document,
4. expose the same capability through a stable public API later without rebuilding the logic a second time.

The key design rule is:

- build Face Scan as its own bounded workflow with a single canonical result schema,
- reuse existing liveness and face-detection primitives where they are already correct,
- do not couple the feature to Video KYC storage fields or UI assumptions.

## Current State

Current implementation status in the codebase:

- `src/basetruth/face_scan/service.py` now owns the static Face Scan orchestration.
- `src/basetruth/face_scan/live.py` now owns dedicated live Face Scan session state, result building, live-page HTML, replay heuristics, temporal consistency checks, and frame-quality scoring.
- `src/basetruth/api.py` now exposes the dedicated live session contract:
  - `POST /api/v1/face-scan/sessions`
  - `GET /api/v1/face-scan/sessions/{session_id}`
  - `WS /api/v1/face-scan/ws/{session_id}`
  - `GET /face-scan/live/{session_id}`
- `src/basetruth/ui/pages/face_scan.py` now uses the Face Scan-specific live routes instead of piggybacking on Video KYC JSON.
- Static and live mode now both return the canonical Face Scan payload with verdict, risk score, confidence, evidence, `trace`, `environment`, `checks`, and `artifacts`.
- The live page now requests a higher-quality browser stream, captures at roughly 8 FPS, retries dropped WebSocket connections, and keeps interrupted sessions resumable until expiry.
- Shared liveness detection in `src/basetruth/kyc/liveness.py` has already been hardened for short realistic blinks and baseline-relative left/right turns.
- Focused test coverage exists in:
  - `tests/test_face_scan_service.py`
  - `tests/test_face_scan_api.py`
  - `tests/test_face_scan_live.py`
  - `tests/test_kyc_liveness.py`

This means Face Scan is no longer a stub. The plan below should now track calibration, operator clarity, and higher-bar anti-spoofing improvements rather than the original scaffolding work.

## Current Gaps Still Open

The scratch note in `docs/new_features.txt` is useful, but it is not a reliable plan document because it mixes one live result, repeated JSON, product questions, and outside suggestions. The real tracked follow-ups from that note are:

- operator-facing explanation still needs to show the captured live signals more clearly, not just the top-level verdict text,
- replay heuristics still need calibration because `repeat_frame_score` can look suspicious even when the challenge flow completed cleanly,
- the live quality model still penalises small faces heavily (`face_size_risk_0_100` can hit 100 easily on otherwise valid sessions),
- challenge-level telemetry is still missing, which makes real-world tuning slower than it should be,
- advanced anti-spoof signals suggested by outside reviewers are still roadmap items, not implemented v1 checks.

## Tracked Next Work

These are the concrete items that should stay on the active plan now:

1. Improve operator clarity in the live result.
  - Return the important raw live metrics in a more readable UI section.
  - Make it easy to see why a session is `GENUINE`, `SUSPICIOUS`, or `INCONCLUSIVE` without reading raw JSON only.

2. Calibrate replay heuristics with real sessions.
  - Review `repeat_frame_score`, `flicker_score`, and brightness-instability thresholds.
  - Separate genuine low-motion sessions from actual replay loops more cleanly.

3. Add challenge-level telemetry for tuning.
  - Capture measured yaw, pitch, EAR, nose shift, reconnect count, and detector dropouts per challenge.
  - Keep this in logs or diagnostics so live failures can be debugged from data rather than guesswork.

4. Review face-size and quality thresholds.
  - Revisit the current quality weighting so valid sessions with a slightly small face do not lose too much confidence.
  - Keep the guidance aligned with the browser capture changes already shipped.

5. Prioritise next-wave anti-spoofing signals for future phases.
  - Active illumination / chromatic reflection.
  - rPPG pulse estimation.
  - 3D depth consistency from challenge motion.
  - Device / transport fingerprinting.
  - FFT / moire screen-replay checks.
  - Eye saccades and other involuntary responses.

## Best Technical Direction

The best way to implement this is to separate Face Scan into three layers:

### 1. Face Scan domain layer

Create a dedicated orchestration layer that owns the Face Scan result contract and decision rules.

Recommended module shape:

- `src/basetruth/face_scan/models.py`
- `src/basetruth/face_scan/service.py`
- `src/basetruth/face_scan/static_analysis.py`
- `src/basetruth/face_scan/live_session.py`
- `src/basetruth/face_scan/result_builder.py`

Why this is the right shape:

- UI and API can both call the same service.
- response formatting stays in one place,
- detector logic stays testable,
- later API exposure becomes a thin wrapper instead of a rewrite.

### 2. Reuse existing primitives, not existing product flows

Reuse these existing primitives:

- `src/basetruth/vision/face.py` for face detection and embeddings,
- `src/basetruth/kyc/liveness.py` for challenge logic,
- the existing camera/WebSocket challenge transport patterns from `src/basetruth/api.py`,
- the existing Scan Document result styling helpers from `src/basetruth/ui/pages/scan.py`.

Do not reuse Video KYC as the product boundary.

Reason:

- Video KYC also carries Aadhaar, PAN, address proof, scheduling, GPS, and entity-linking concerns.
- Face Scan should stay focused on face integrity and liveness only.
- If Face Scan keeps piggybacking on Video KYC session semantics, the later public API will inherit unrelated baggage.

### 3. One canonical Face Scan result contract

Face Scan should always produce one final payload shaped like Scan Document: a summary verdict, a score, plain-English review text, evidence bullets, and detailed raw signals.

Recommended response shape:

```json
{
  "filename": "selfie.jpg",
  "scan_type": "face_scan",
  "mode": "static|live|hybrid",
  "schema_version": "1.0.0",
  "verdict": "GENUINE|SUSPICIOUS|DEEPFAKE|LIVENESS_FAILED|INCONCLUSIVE",
  "risk_score_0_100": 18.4,
  "confidence_0_100": 82.1,
  "confidence_reason": "Good lighting and stable face tracking produced reliable live signals.",
  "overall_explanation": "Technical summary of the strongest signals.",
  "honest_review": "Plain-English explanation for the operator.",
  "evidence": [
    "One clear frontal face detected.",
    "No strong replay or screen-glare artefacts found.",
    "Blink and head-turn challenges passed."
  ],
  "trace": {
    "decision_trace_id": "fs_20260501_abc123",
    "timestamp_utc": "2026-05-01T14:22:17Z",
    "processing_time_ms": 1640,
    "rules_version": "face-scan-rules-1.0.0",
    "model_version": "heuristics-only"
  },
  "environment": {
    "platform": "web",
    "browser": "Chrome 136",
    "os": "Windows 11",
    "camera_resolution": [1280, 720],
    "observed_fps": 14.7,
    "virtual_camera_suspected": false
  },
  "checks": {
    "face_detection": {
      "face_count": 1,
      "primary_face_box": [90, 44, 280, 250],
      "confidence": 0.98
    },
    "photo_authenticity": {
      "status": "pass",
      "score_0_100": 12.0,
      "signals": {
        "screen_replay": 8.0,
        "print_attack": 14.0,
        "compression_anomaly": 10.0,
        "edge_halo": 6.0
      }
    },
    "deepfake_signals": {
      "status": "pass",
      "score_0_100": 16.0,
      "signals": {
        "landmark_stability": 0.92,
        "skin_texture_consistency": 0.88,
        "warp_artifacts": 0.07,
        "blending_artifacts": 0.05
      }
    },
    "active_liveness": {
      "status": "pass",
      "passed": true,
      "completed_challenges": ["look_straight", "blink", "nod"],
      "challenge_count": 3,
      "best_frame_available": true
    }
  },
  "artifacts": {
    "best_frame_available": true,
    "challenge_snapshots_available": true
  }
}
```

This gives the UI and the future API the same shape from day one.

For production use, the added confidence, trace, and environment fields are required so that operators and API clients can distinguish between a low-risk high-confidence result and a low-risk low-confidence result. These fields also provide the audit trail needed for disputes, reviews, and later compliance requirements.

## Product Recommendation

Face Scan should support three operator modes.

### Mode A - Static Photo Scan

Input:

- one uploaded selfie or ID-photo-style face image.

Checks:

- face detection,
- fake-photo / replay-photo heuristics,
- deepfake-like artifact heuristics,
- image quality checks,
- optional multi-face rejection.

Output:

- immediate Face Scan result payload,
- no live challenge section,
- verdict can still be `SUSPICIOUS`, `DEEPFAKE`, or `INCONCLUSIVE`.

### Mode B - Live Liveness Challenge

Input:

- camera stream only.

Checks:

- look straight,
- blink,
- nod,
- turn left,
- turn right,
- optional future replay-screen heuristics from the live frames.

Output:

- live-only Face Scan result payload,
- no static deepfake claim if no uploaded photo was analysed,
- verdict should reflect challenge result clearly.

### Mode C - Hybrid Face Scan

Input:

- uploaded face image plus live challenge.

Checks:

- all static checks,
- all selected live challenges,
- optional future reference comparison if product later needs it.

Output:

- one final payload that merges static and live results,
- this should become the default long-term product mode.

## Detector Strategy

Do not claim true deepfake detection unless the signal is backed by deterministic heuristics or a real model.

For a deeper discussion focused specifically on live-camera deepfake detection from liveness sessions, see:

- `docs/FACE_SCAN_LIVE_DEEPFAKE_TECHNICAL_NOTE.md`

Recommended v1 strategy:

### Static image checks

Build a deterministic anti-spoof / authenticity layer first.

Candidate signals:

- face-count validation,
- minimum face size and frontal-quality checks,
- blur and motion blur,
- moire / screen-recapture patterns,
- print-photo cues such as flat highlights and paper edges,
- JPEG double-compression and halo edges near the face region,
- landmark symmetry anomalies,
- local texture inconsistency inside the facial mask,
- eye-region / mouth-region blending artefacts,
- background-face boundary artefacts.

This should be framed as a fraud-risk score, not as a magical absolute deepfake detector.

### Live checks

Reuse the liveness challenge engine already in `src/basetruth/kyc/liveness.py`, but expose it through a Face Scan-specific session layer.

Needed outputs:

- current challenge,
- completed challenges,
- pass/fail,
- best live frame,
- one retained frame per completed challenge,
- challenge-level reasoning for the final response.

### Verdict rules

Recommended v1 verdict mapping:

- `GENUINE` when static risk is low and live challenges pass,
- `SUSPICIOUS` when some risk signals are elevated but not decisive,
- `DEEPFAKE` only when strong synthetic/blending/replay signals are above threshold,
- `LIVENESS_FAILED` when the live challenge fails,
- `INCONCLUSIVE` when image quality or detector confidence is too low.

## API Design Recommendation

Do not keep the future API as a single overloaded upload endpoint only.

Use two API surfaces:

### 1. Immediate static scan

- `POST /api/v1/face-scan`

Purpose:

- upload one image,
- run static checks,
- return the canonical Face Scan result payload.

### 2. Live session flow

- `POST /api/v1/face-scan/sessions`
- `GET /api/v1/face-scan/sessions/{session_id}`
- `WS /api/v1/face-scan/ws/{session_id}`
- optional `GET /api/v1/face-scan/sessions/{session_id}/best-frame`

Purpose:

- create a live-only or hybrid session,
- drive challenges over WebSocket,
- return the same canonical Face Scan result payload when done.

This keeps the external API simple:

- one-shot static image clients call one endpoint,
- live clients use a session lifecycle,
- both still receive the same final result shape.

## UI Recommendation

Rework the Face Scan page into three sections, not two disconnected prototypes.

### Section 1 - Scan Setup

Controls:

- mode selector: `Static Photo`, `Live Challenge`, `Hybrid`,
- optional face-image uploader,
- challenge multiselect,
- optional strictness selector,
- `Run Face Scan` or `Start Live Session` action.

### Section 2 - Live Challenge Panel

Show only when live mode is active.

Display:

- open-challenge link or embedded camera surface,
- session status,
- progress bar,
- current instruction,
- retained best frame once available.

### Section 3 - Final Result Panel

This should visually mirror the Scan Document result experience.

Display:

- result banner,
- verdict badge,
- risk score,
- operator-facing review card,
- evidence bullet list,
- syntax-highlighted JSON for the detailed payload,
- download JSON button.

The page should stop showing raw KYC session JSON directly.

## Recommended Implementation Phases

Status key:

- `Done` = shipped in the current codebase.
- `Next` = should be tackled in the near term.
- `Later` = real roadmap work, but not needed to keep the current Face Scan contract working.

### Phase 1 - Freeze the Face Scan contract (`Done`)

Decide and document:

- canonical response schema,
- verdict set,
- score semantics,
- supported modes,
- which checks are available in static vs live mode.

Deliverables:

- `docs/FUNCTIONALITY.md` Face Scan section,
- `docs/ARCHITECTURE.md` Face Scan layer entry,
- `docs/IDENTITY_VERIFICATION.md` mention of shared liveness primitives if needed,
- API schema update in `src/basetruth/api.py`.

### Phase 2 - Extract a dedicated Face Scan service (`Done`)

Create a service entry point like:

- `run_face_scan_static(...)`
- `create_face_scan_session(...)`
- `get_face_scan_session_status(...)`
- `build_face_scan_result(...)`

Goal:

- UI and API stop calling ad-hoc helpers directly.

### Phase 3 - Replace the fake static scorer (`Done`)

Remove the random score from `vision.face.analyze_face()`.

Replace it with deterministic face-authenticity heuristics and a structured score breakdown.

Important rule:

- if a real deepfake model is not present, the payload must say it is heuristic-based,
- never pretend a simulated score is a real model verdict.

### Phase 3A - Add uncertainty and environment signals (`Done`)

Before Face Scan is treated as production-grade, the payload must also capture:

- `confidence_0_100`,
- `confidence_reason`,
- browser / OS / platform metadata,
- observed FPS and camera resolution,
- virtual-camera suspicion,
- detector drop-rate and frame-quality summaries.

Why this matters:

- poor lighting and low FPS can invalidate otherwise good heuristics,
- virtual cameras and emulators are a major attack path for public APIs,
- production systems need confidence and environment context, not only a raw risk score.

### Phase 4 - Separate Face Scan live sessions from Video KYC sessions (`Done`)

Extract shared liveness/session logic where useful, but introduce Face Scan-specific session orchestration.

Reason:

- Face Scan should not require customer name, entity ref, address fields, or reference-doc semantics.

Expected result:

- Face Scan sessions become lightweight,
- future public API is cleaner,
- current Video KYC behavior stays isolated.

### Phase 5 - Build the unified result builder (`Done`)

Create one result builder that converts static signals and live-session outcomes into the canonical payload.

It should generate:

- verdict,
- risk score,
- confidence score,
- overall explanation,
- honest review,
- evidence list,
- checks dictionary,
- artifact availability metadata,
- audit trace fields,
- environment metadata.

This is the most important phase because it makes Face Scan feel like Scan Document instead of a debug tool.

### Phase 5A - Add operational safeguards (`In Progress`)

Already implemented:

- session expiry,
- canonical response schema versioning,
- duplicate-frame and replay-loop heuristics,
- resumable live sessions after disconnects,
- browser reconnect handling.

Still to add before public API exposure:

- per-IP and per-API-key rate limiting,
- max attempts per session,
- request authentication,
- stronger diagnostics for challenge-level failures.

Recommended baseline targets:

- browser capture target: 8-12 FPS on the supported path,
- sliding analysis window: 3 seconds,
- max live session TTL: 20 minutes,
- target per-frame processing latency: under 200 ms on the supported CPU path.

### Phase 6 - Rework the Face Scan UI page (`In Progress`)

Update `src/basetruth/ui/pages/face_scan.py` so it:

- uses the new Face Scan service,
- supports the live session lifecycle cleanly,
- reuses the Scan Document presentation pattern,
- never shows raw internal session payloads as the only operator-facing explanation,
- explains captured live parameters more clearly than the current raw JSON-only detail view.

### Phase 7 - Rework the API endpoints (`Done for current contract`)

Update `src/basetruth/api.py` so it:

- exposes the canonical static response,
- exposes Face Scan session lifecycle endpoints,
- documents both in Swagger with explicit Pydantic models,
- includes versioned contracts and authentication requirements.

### Phase 7A - Calibrate thresholds before release (`Next`)

The score ranges in this plan are design placeholders, not final thresholds.

Before release, calibrate using evaluation data across genuine sessions, replay attacks, printed-photo attacks, live face-swap sessions, and compressed / low-FPS sessions.

Expected outputs:

- operating thresholds,
- false accept / false reject tradeoffs,
- confidence-score tuning,
- revised score-fusion weights if needed.

### Phase 8 - Add reports/persistence only if product really needs them (`Later`)

Recommendation:

- keep v1 Face Scan stateless like Scan Document,
- return downloadable JSON only,
- add DB persistence later only if operators explicitly need historical storage.

If persistence is later required, use a new `face_scans` table rather than overloading `video_kyc_checks`.

## Files Likely To Change During Implementation

- `src/basetruth/ui/pages/face_scan.py`
- `src/basetruth/api.py`
- `src/basetruth/vision/face.py`
- `src/basetruth/kyc/liveness.py` only if shared challenge outputs need richer metadata
- new package under `src/basetruth/face_scan/`
- `docs/FUNCTIONALITY.md`
- `docs/ARCHITECTURE.md`
- `docs/IDENTITY_VERIFICATION.md`
- `docs/TESTING.md` only if Face Scan adds a new testing rule or command example

## Testing Plan For The Later Implementation

Minimum expected test coverage:

### Unit tests

- static authenticity score builder,
- verdict mapping,
- confidence-score and confidence-reason mapping,
- evidence generation,
- face-count and no-face failure handling,
- session result builder,
- live challenge aggregation,
- API response model serialization,
- environment metadata normalization,
- duplicate-frame / replay-guard helper logic.

### Service tests

- static endpoint returns canonical payload,
- Face Scan session lifecycle returns final result payload on completion,
- hybrid flow merges static and live results correctly,
- low-quality image yields `INCONCLUSIVE` cleanly,
- low-signal sessions yield low confidence without overclaiming certainty,
- rate limiting and session-expiry behavior fail safely.

### Regression tests

- no random scoring remains,
- UI helper renders final result without raw debug JSON assumptions,
- Video KYC behavior is unchanged by Face Scan extraction.

Run order when implementation starts:

1. narrow Face Scan unit tests,
2. narrow API/session tests,
3. full `python -m pytest tests/ -q --tb=short`.

## Main Risks

- keeping Face Scan tied to Video KYC and inheriting unrelated complexity,
- exposing a public API before the result contract is stable,
- shipping a fake deepfake score that looks authoritative but is actually random,
- mixing UI formatting logic with detector logic,
- adding persistence too early and locking in the wrong schema,
- using the word `deepfake` too confidently when the system is really returning heuristic fraud signals,
- returning risk scores without confidence context,
- ignoring device-level attack vectors like virtual cameras and replay loops,
- releasing thresholds that were never calibrated on real attack data.

## Practical Recommendation

Implement Face Scan as a stateless Scan Document-style product first, with optional live session support.

That means:

- one canonical Face Scan payload,
- dedicated Face Scan service/orchestrator,
- shared detector primitives underneath,
- static mode and live mode both feeding the same result builder,
- API and UI both consuming that same contract.

This is the cleanest path technically and the safest path for the later external API.