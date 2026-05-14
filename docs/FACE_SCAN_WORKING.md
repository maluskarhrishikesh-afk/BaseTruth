# Face Scan — How It Works

This document explains exactly how the Face Scan screen works, step by step, in plain language.
No jargon. Every number and decision rule is explained in everyday terms.

---

## What Is Face Scan?

Face Scan is a screen in BaseTruth that checks whether a face is real.
It has two ways of checking: **uploading a photo** or **doing a live camera challenge**.

Both methods give you a final verdict at the end:

- **GENUINE** — the face looks real and the check passed.
- **SUSPICIOUS** — something looked a bit off; a human should review it.
- **DEEPFAKE** — strong signs of a fake, replayed, or screen-recorded face.
- **INCONCLUSIVE** — the image or video quality was too poor to decide.
- **LIVENESS_FAILED** — the person did not finish the required live challenges.

The result also includes:
- a **risk score** from 0 to 100 (0 = completely safe, 100 = definitely fake),
- a **confidence score** from 0 to 100 (how sure the system is of its verdict),
- an **honest review** sentence telling the operator what to do,
- a list of **evidence bullets** explaining why the verdict was reached.

---

## Tab 1: Static Photo Scan

This tab lets you upload a single photo and check if the face in it looks genuine.

### Step 1 — Upload the photo

The operator uploads a photo from their computer.
The system reads the file bytes and converts them into a usable image.
If the file cannot be decoded (e.g. corrupt file), it immediately returns an error.

### Step 2 — Find the face

The system tries to find a face using **InsightFace** (our primary face detector).
If InsightFace is not available, it falls back to **MediaPipe** (a lighter detector from Google).

- If no face is found, the check stops immediately with the message "No face found in the image."
- If multiple faces are found, the largest face is used for analysis, but the result is flagged as less reliable.

### Step 3 — Crop and measure the face

Once the face is found, the system cuts out just the face region from the photo.
It then measures five things about that crop:

**a) Sharpness (blur check)**
It runs a "Laplacian" filter on the face, which highlights edges.
A real, sharp photo has a lot of clearly defined edges. A blurry photo has fewer.
The system computes how much variation the edges have:
- High variation (above 140) = sharp image — no blur penalty.
- Low variation (below 25) = blurry image — high blur risk.

> **Threshold note:** These values (140 and 25, as well as the brightness, face size, and extreme-quality thresholds further below) are named constants in `service.py`. They are calibrated for typical webcam and phone-camera images at standard resolutions. Deployments with unusually low-resolution cameras, strong artificial lighting, or unusual demographics may need to recalibrate them — ideally by measuring the distribution of these metrics on a representative sample of images from that environment.

**b) Brightness (lighting check)**
It measures the average pixel brightness of the face (0 = pitch black, 255 = pure white).
It calculates how far the brightness is from a comfortable middle value of 128.
- Difference less than 18 — balanced lighting — no penalty.
- Difference greater than 90 — too dark or overexposed — high brightness risk.

**c) Face size (proportion check)**
It measures what percentage of the total image area the face occupies.
- Face covers more than 18% of the photo — good size — no penalty.
- Face covers less than 6% — the face is too small — high size risk.

**d) Edge halo (screen or print check)**
This checks whether the visual edges in the photo are strangely concentrated around the border of the face rather than spread evenly across it.
When someone photographs a printed photo or a screen showing a photo, the edges of the face region look unnaturally sharp at the boundary but flat in the middle.
The system measures:
- Edge density at the outer border of the face crop (10% margin around the edges).
- Edge density at the centre of the face crop.
- If the border is much denser than the centre, it adds a spoof risk signal.

**e) Compression leftovers (double-compression check)**
This checks whether the photo has been saved as a JPEG twice.
When someone takes a screenshot of a face from another JPEG (like photographing a screen), the image goes through JPEG compression twice — once when the original was saved, and once when the screenshot was taken.
The system re-encodes the face crop at high JPEG quality and compares it to the original.
If the difference is large, it means the face crop already had embedded JPEG artefacts baked in — a sign of a re-photograph or screen capture.

**f) Landmark asymmetry (face geometry check)**
The face detector provides five keypoints: left eye, right eye, nose, left mouth corner, right mouth corner.
The system measures whether the eyes are at different heights and whether the mouth corners are uneven.
Large asymmetry for a single still portrait suggests the geometry has been altered or is unstable.

### Step 4 — Calculate the risk score

The system combines the edge halo score and compression score into a presentation risk (how likely is this a photo of a photo or screen):

    Presentation risk = 60% x edge halo + 40% x compression score

The landmark asymmetry score becomes the synthetic risk (how likely is this a manipulated or generated image).

The final risk score depends on whether one face or multiple faces were found:

**Single-face path (normalised):**

    Risk score = 70% x presentation risk + 30% x synthetic risk

When only one face is found, multi-face risk is always 0.
Using a fixed 35% weight for a signal that is always 0 would permanently cap the maximum possible risk at 65, making the DEEPFAKE verdict (threshold 75) mathematically unreachable.
Normalising to the two active signals restores the full 0–100 scale.

**Multi-face path:**

    Risk score = 45% x presentation risk + 20% x synthetic risk + 35% x multi-face risk

Multi-face risk jumps to 85 if more than one face was found.
The risk score is always clamped between 0 and 100.

### Step 5 — Calculate the confidence score

The confidence score tells you how much to trust the verdict.
Low confidence does not mean the face is fake — it just means the photo quality was not good enough, or the sub-signals were too inconsistent, for a strong conclusion.

Confidence is built from two components:

**Quality component (70% weight)**

How well-suited the image is for reliable analysis:

    Quality = 100 - (35% × blur risk) - (30% × brightness risk) - (20% × face size risk) - (15% × detector penalty)

The detector penalty adds a small deduction if the face detector was not fully certain it found a real face.

**Signal agreement component (30% weight)**

How consistently the independent fraud sub-signals agree with each other.
If presentation risk and synthetic risk both point strongly in the same direction (both high, or both low), the system has a clear, coherent picture and confidence is higher.
If one signal indicates fraud while the other shows none, the evidence conflicts and confidence is reduced.

    Agreement = 100 - 2 × (standard deviation of sub-scores)

A standard deviation of 0 (all signals identical) → Agreement = 100.
A standard deviation of 50 (e.g. one signal at 0, another at 100) → Agreement = 0.

**Combined:**

    Confidence = (70% × quality component) + (30% × signal agreement)

Confidence is capped at 99 and floored at 5.

Two additional hard caps apply:
- If more than one face was found — confidence is capped at 55.
- If the image is extremely blurry (Laplacian below 12) or the face is tiny (less than 3% of the image) — confidence is capped at 25.

### Step 6 — Decide the verdict

| Condition | Verdict |
|---|---|
| Confidence below 35 | INCONCLUSIVE |
| Risk score 75 or above | DEEPFAKE |
| Risk score 35 or above, or more than one face | SUSPICIOUS |
| Everything else | GENUINE |

---

## Tab 2: Live Camera Challenge

This tab opens a live camera session in the browser.
The person is asked to perform a short set of movements in front of their camera.
The system watches the camera feed in real time and checks whether the behaviour is consistent with a real live human.

### How the session starts

1. The operator goes to the Face Scan screen and picks the challenges they want: Blink, Nod, Turn Left, Turn Right, or Look Straight. The default set is **blink, nod, turn left, and turn right**. The API always prepends `look_straight` automatically, so the full default sequence is: **look_straight → blink → nod → turn_left → turn_right**.
2. The operator clicks "Generate Live Challenge Link". The system creates a new live session with a unique session ID and stores it in memory.
3. The system generates a link like: http://127.0.0.1:8000/face-scan/live/{session_id}
4. That link opens a dedicated browser page served directly by the backend.

### What happens in the browser (the live page)

The person clicks "Start Live Face Scan" on the browser page.

The browser asks for camera permission and starts the front-facing camera at up to 1280x720 resolution.

A WebSocket connection opens between the browser and the backend.
A WebSocket is a continuous two-way connection — unlike a regular web request, it stays open the whole time, and both sides can send messages at any moment.

The browser then starts capturing and sending a frame (a still image from the camera) every 125 milliseconds — roughly 8 frames per second.

Each frame is:
- resized to 640 pixels wide,
- horizontally mirrored (so the person sees their own reflection as expected, and the server uses the same left-right orientation as the person),
- encoded as a JPEG at 82% quality to balance file size and image detail,
- sent to the server as a base64 string (text encoding of the image bytes),
- tagged with a high-precision timestamp from the browser.

Before sending frames, the browser also sends one metadata message containing:
- the camera width and height,
- the camera device name (e.g. "FaceTime Camera" or "OBS Virtual Camera"),
- the browser's user agent string,
- the operating system platform.

If the WebSocket connection drops (e.g. the network glitches), the browser automatically tries to reconnect up to 8 times, with a 1.2-second pause between attempts.
Once reconnected, the session continues from where it left off.
Sessions stay valid for up to 20 minutes.

### What the backend does with each frame

When a frame arrives at the server:

1. The server records the exact time it was received using a monotonic clock (a clock that never jumps or drifts). This timestamp is used later for timing analysis.
2. The server decodes the JPEG back into an image.
3. The server runs the face detector on the image.
   - For all challenges except blink: only InsightFace runs (faster).
   - For the blink challenge only: InsightFace runs first, and then MediaPipe also runs to get the Eye Aspect Ratio. Running both detectors is only done for blink because it is the only challenge that needs the eye opening measurement.
4. If a face is found, the server extracts pose measurements from the five face keypoints (left eye, right eye, nose, left mouth, right mouth):
   - Yaw — how far the nose has moved left or right relative to the midpoint between the eyes, divided by the distance between the eyes. This measures head turning.
   - Pitch — how far the nose is below or above the midpoint between the eyes, divided by the distance between the eyes. This measures nodding.
   - Nose horizontal position — where the nose sits between the left and right edges of the face bounding box (0 = far left, 1 = far right). Used for the look-straight check.
   - Interocular distance ratio — the pixel distance between the two eyes, divided by the face width. Used for the 3D depth check.
   - Eye landmark positions — where the left and right eyes sit within the face bounding box. Used for the eye micro-jitter check.
   - Detection confidence — how sure the face detector is that it found a real face.
   - Frame hash — a tiny 8x8 fingerprint of the face crop, used for the replay check.
   - Blur score — how sharp the frame is.
   - Brightness — how bright the face crop is.
5. Every 5th frame, the server also runs a screen-frequency check on the face crop (see below). Only every 5th frame to save processing time.
6. All measurements are saved into the session's frame history.

The server then runs the challenge detector on the frame history for the current challenge.
If the challenge passes, it moves to the next challenge.

The server sends a status message back to the browser with:
- which challenge is currently active,
- how many challenges are done,
- feedback text to show the person (e.g. "Keep turning — turn further to YOUR left...").

When all challenges pass, the server runs the full analysis and sends back a result message with the final verdict, risk score, confidence, and evidence bullets.

---

## Face Stability Gate (Pre-Challenge)

Before any challenge frame is accepted or counted, the server requires the face to be **stable for approximately 1 second**.
This is called the Face Stable gate and it follows the expert-recommended pipeline:

    Face Detection → Face Validation → Face Tracking (stability) → Liveness Challenge

### Why it is needed

Without a stability requirement, an attacker could flash a known face image past the camera for one or two frames, then perform the challenge with a different face or object.
The gate ensures the person is genuinely present, well-positioned, and clearly visible **before** any challenge credits start accumulating.

### How it works

The server maintains a per-session counter called `face_stable_frames`.
This counter only increments when a frame passes **all five** of these checks simultaneously:

1. **Exactly one face** is detected in the frame — no more, no less.
2. **Detection confidence ≥ 0.80** — the face must be clearly recognisable. This threshold is deliberately stricter than the 0.55 used during live challenges (where turns, blinks, and nods temporarily reduce confidence). The pre-challenge phase allows the server to demand high certainty while the person is stationary.
3. **Face area ≥ 6% of frame** — the stability gate is stricter than the 5% live-challenge threshold. At 6%, the bounding box is at least ~125×125 px at 640×480, giving the landmark model reliable results before challenges start.
4. **Nose within the horizontal centre band** (35%–65% of frame width) — the person's face must be roughly centred horizontally.
5. **Nose within the vertical centre band** (30%–70% of frame height) — prevents partial faces, tilted cameras, and bad landmark geometry from a very low or high camera position.
6. **Head near-frontal** — |yaw| ≤ 0.12 (roughly ±7°). The user must face roughly straight at the camera before challenges begin. Pitch is intentionally not checked here — the 2D pitch value (nose-y minus eye-midpoint-y, normalised by interocular distance) is an anatomical constant (~0.4–0.7 for any real forward-facing face) and cannot distinguish head tilt from a face simply being close to the camera. The vertical centering check (condition 5) already guards against extreme camera angles.
7. **Texture richness (adaptive)** — the face crop must have sufficient local texture variation (measured as mean per-patch standard deviation across a 6×6 grid). The threshold adapts to ambient brightness: in a dark frame (brightness mean < 80) the threshold relaxes to 14.0 to avoid false rejects; in normal light it is 18.0. A flat screen, printed photo, or uniform background scores < 15; a real face at selfie distance typically scores 25–70.
8. **Geometry invariants pass** — the face must look like a real human face (eyes above nose, mouth below eyes, proportions within normal ranges).

Additionally, once the window of qualifying frames is complete, the server checks two anti-spoofing signals:

1. **Yaw micro-movement variance** — A real person always has small involuntary oscillations (breathing, muscle tremor). With our 5-point InsightFace landmark yaw formula, real users at a laptop webcam produce variance in the range 4×10⁻⁶ to 1×10⁻⁴. A static screen or printed photo with a frozen frame has near-zero floating-point variance (≈ 10⁻⁸ to 10⁻⁷). If the variance is below 1×10⁻⁶, the window is rejected and the counter resets with the message: "No natural movement detected — ensure you are using a live camera."

2. **Yaw jitter naturalness** — A real person's micro-movement direction changes occasionally but not on every frame. An attacker who shakes a screen to pass the variance check produces a high-frequency alternating signal: yaw direction reverses on almost every frame. If more than 80% of consecutive diff-pairs alternate sign, the motion is flagged as unnatural. The window resets with the message: "Unnatural movement detected — ensure you are using a live camera, not a video." This is controlled by `FACE_STABILITY_YAW_JITTER_ALTERNATION_MAX = 0.80`.

If any condition fails, the counter resets to zero. The counter must reach **10 consecutive qualifying frames** (approximately 1 second at 10 FPS) before challenge history starts accumulating.

While the counter is below 10, the server responds with:
- `face_stable: false`
- `face_stable_progress` (current count)
- `face_stable_required` (10)
- A human-readable feedback message explaining what the person needs to fix (e.g. "Move closer to the camera — your face is too small in the oval.")

Once the counter reaches 10 and the yaw variance check passes, `face_stable: true` is set on all subsequent status messages, and the normal challenge-accumulation logic begins.

### Stability Gate Reset Between Challenges

When one challenge completes and the session moves to the next, the stability gate is **only reset for turn challenges** (`turn_left` and `turn_right`). Non-turn challenges (`look_straight`, `blink`, `nod`) leave the gate state intact.

Why: turn challenges require the person to move significantly away from the frontal position, so after a turn challenge the face needs to return to stable frontal position before the next challenge begins. For still challenges (look straight, blink, nod), the person typically remains near-frontal throughout, so resetting the gate would waste time and require the user to pass 10 more stability frames unnecessarily.

### Constants (in `src/basetruth/kyc/liveness.py`)

| Constant | Value | Meaning |
|---|---|---|
| `FACE_STABLE_FRAMES_REQUIRED` | 10 | Consecutive qualifying frames before challenges start |
| `FACE_STABILITY_CONFIDENCE_MIN` | 0.80 | Minimum detection confidence for the stability gate |
| `FACE_STABILITY_AREA_MIN` | 0.06 | Minimum face area (6%) for the stability gate |
| `FACE_STABILITY_X_MIN` / `X_MAX` | 0.35 / 0.65 | Horizontal centering band |
| `FACE_STABILITY_Y_MIN` / `Y_MAX` | 0.30 / 0.70 | Vertical centering band |
| `FACE_STABILITY_YAW_MAX` | 0.12 | Maximum absolute yaw before challenges (~±7°); pitch is intentionally not checked (see note above) |
| `FACE_STABILITY_YAW_VARIANCE_MIN` | 1×10⁻⁶ | Minimum yaw variance over stability window (micro-movement check); calibrated for 5-point landmark formula |
| `FACE_STABILITY_YAW_JITTER_ALTERNATION_MAX` | 0.80 | Maximum fraction of alternating yaw diffs before jitter is flagged as artificial |
| `MIN_FACE_TEXTURE_SCORE` | 18.0 | Minimum local texture richness score in normal light |
| `LOW_BRIGHTNESS_THRESHOLD` | 80 | Brightness mean below which the relaxed texture threshold applies |
| `LOW_BRIGHTNESS_TEXTURE_SCORE` | 14.0 | Relaxed texture threshold in dark/low-light frames |
| `CHALLENGE_TIMEOUT_SECONDS` | 10.0 | Seconds before a challenge that has not been completed resets |
| `MIN_FACE_DETECTION_CONFIDENCE` | 0.55 | Minimum confidence during active challenges |
| `MIN_FACE_AREA_RATIO` | 0.05 | Minimum face area (5%) at all times during active challenges |
| `_NOD_DOWN_DELTA` | 0.08 | Minimum pitch deviation from baseline to count as a nod (gentle chin-dip) |
| `_NOD_HOLD_FRAMES` | 6 | Consecutive frames of sustained nod position required (~0.6 s at 10 FPS) |
| `_NOD_BASELINE_FRAMES` | 3 | Frames at challenge start used to compute the neutral-pitch baseline |
| `_STRAIGHT_STABLE_FRAMES` | 10 | Consecutive centred + frontal frames required for look_straight (~1 s at 10 FPS) |
| `_STRAIGHT_X_MIN` / `X_MAX` | 0.40 / 0.60 | Horizontal centering band for look_straight |
| `_STRAIGHT_YAW_MAX` | 0.12 | Maximum absolute yaw accepted during look_straight |

---

## How Each Challenge Works

All measurements are done on the mirrored frame (the same orientation the person sees in their selfie preview).
So "the person turns to their own left" means the nose moves toward the left side of the image.

### Wrong-Action Handling

Each directional challenge (turn left, turn right, nod) has a built-in wrong-direction guard.
If the person moves in the exact wrong direction before completing the challenge, the server detects the `wrong_motion` flag from `analyze_challenge`.
The server then:
1. Appends the event to `challenge_wrong_actions` (an audit trail on the session object).
2. Resets the frame history for the current challenge to the last 2 frames only — this keeps the baseline near-neutral and avoids a junk accumulation of wrong-direction frames.
3. Returns a sticky feedback message (e.g. "Move the opposite direction") so the browser displays it persistently until the next frame.

The audit trail (`challenge_wrong_actions`) is not shown to the user but is available in session logs.
A high wrong-action count can itself be a soft signal of confusion or spoofing.

### Challenge Timeout

Once the stability gate passes and the first challenge begins, every challenge has a hard timeout of **10 seconds** (`CHALLENGE_TIMEOUT_SECONDS = 10.0`).
The timer is tracked in `session.challenge_started_at` (Unix monotonic timestamp).
On every incoming frame during challenge processing, the server checks `time.monotonic() - session.challenge_started_at`.
If the elapsed time exceeds 10 seconds:
- The frame history for the current challenge is reset to an empty list.
- The timer restarts.
- The user receives the feedback: "Time's up — please perform the challenge again."

The timer also resets whenever a challenge passes (so each challenge gets a fresh 10-second window).

### Early Blink Signal

During the stability window (while `face_stable_frames` is accumulating), the server watches the eye aspect ratio (EAR) on every frame.
If any frame has `ear < 0.20` (a blink or partial close), `session.blink_observed_in_stability` is set to `True`.
This is a **soft signal only** — it does not gate any decision.
Its purpose is to confirm that the person is alive and interacting before the formal Blink challenge starts.
It is logged and available for future automated risk scoring.

### Look Straight

This challenge asks the person to face the camera directly and hold still.
It is optional but useful — when it passes, the server saves the current frame as the best face capture for the session.

How it detects this:
It looks at the nose's horizontal position within the face bounding box across the last **10 consecutive frames** (`_STRAIGHT_STABLE_FRAMES = 10`).
All 10 frames must satisfy three conditions simultaneously:

1. **Horizontal centering** — nose stays between 40% and 60% of the face width (roughly centred).
2. **Low yaw** — nose is no more than 0.12 away from the eye midpoint (direct measure of head rotation; catches a face that is centred yet still rotated sideways).
3. **Detection confidence** ≥ 0.65 — rejects blurry or marginal detections so the saved selfie is always a clean shot.

10 frames at 10 FPS takes about 1 second of steady hold.

If the nose is too far left, the feedback says "Move slightly to YOUR right to centre."
If the nose is too far right, the feedback says "Move slightly to YOUR left to centre."
If the yaw is too large (head is rotated), the feedback says "Look directly into the camera lens…"

### Blink

This challenge asks the person to close both eyes fully and then open them again.

How it detects this:
The server uses Eye Aspect Ratio (EAR) from MediaPipe.
EAR is a number between roughly 0.05 (eyes fully closed) and 0.35 (eyes fully open).
It is calculated from how tall the eye opening is relative to the width of the eye.
Open eye = tall opening = high EAR. Closed eye = narrow opening = low EAR.

The server looks for this sequence in the history:
1. Eyes were open earlier (EAR above 0.20) — the baseline.
2. Eyes closed at some point (EAR dropped below 0.20) — the dip.
3. Eyes are open again now (EAR recovered above 0.18) — the reopen.

If MediaPipe EAR is not available, the system falls back to watching InsightFace's face-detection confidence score.
A blink causes a small but measurable dip in that score because the closing eyes reduce facial feature visibility.

The minimum frame count for the blink check is 3 frames (about 300 ms at 10 FPS).
This is enough to capture the baseline, dip, and reopen.

### Nod

This challenge asks the person to gently tilt their chin down toward their chest.
No return-to-neutral is required — once the hold is met the challenge passes immediately.

How it detects this (hold-based approach):
1. **Baseline** — the server records the neutral pitch from the first 3 challenge frames (the user was just looking straight after the stability gate).
2. **Threshold** — pitch must deviate from that baseline by at least **0.08** (`_NOD_DOWN_DELTA = 0.08`). This is a gentle chin-dip; the user does not need to bend far down.
3. **Hold** — the server finds the longest contiguous run of frames where the deviation is ≥ 0.08. That run must reach **6 consecutive frames** (`_NOD_HOLD_FRAMES = 6`) — approximately 0.6 seconds at 10 FPS.
4. **Direction-agnostic** — both positive and negative pitch deviations are accepted, because different camera heights produce different pitch directions for the same "chin down" motion.

Once the 6-frame hold quota is met, the challenge passes and a green flash signals completion.

Wrong-motion guard: if the person shakes their head side-to-side (yaw range much larger than pitch range, yaw range > 0.10), the server detects it as a wrong motion and tells the user to tilt their chin DOWN, not sideways.

Feedback progression:
- Before any deviation: **"Gently tilt your chin down…"**
- Once deviation starts building: **"Keep looking down… (N more frames)"**
- Hold quota met: **"✅ Nod completed!"

### Turn Left

This challenge asks the person to slowly turn their head to their own left.

How it detects this:
It watches the yaw measurement.
When you turn left, the nose moves toward the left side of the image (away from the eye midpoint), making yaw negative.

The check passes if the most negative yaw value seen in the last 20 frames is at or below -0.16
(meaning the nose has swung at least 16% of the eye-to-eye distance to the left).

There is also a fallback: if the person started centred and the nose shifted left by at least **12%** of the face width AND the yaw changed by at least **0.12** from the starting position, that also counts as a turn.
This fallback helps people who cannot do a large turn due to physical constraints.

The 0.12/0.12 thresholds were tightened from 0.09/0.10 specifically to prevent false passes. After the stability gate passes, the user's face may be resting at a very slightly off-centre baseline. Drifting back to neutral from that starting pose could accumulate a small delta, which at the older looser thresholds would incorrectly register as a successful turn. The tighter thresholds require a clear deliberate turn, not just a return-to-centre drift.

### Turn Right

Exactly the same logic as Turn Left, but in the opposite direction.
Yaw must reach at least +0.16 (nose swung to the right).
The relative-delta fallback also requires at least 0.12 yaw delta and 0.12 nose shift (same tightened thresholds as turn left).

---

## The Seven Safety Checks (Live Mode)

After all the challenges are done, the server runs seven background checks across all the collected frames to look for signs of fraud.
These checks run on the full frame history — everything captured from when the session started until it ended.

### Check 1 — Replay Detection (weight: 50%)

This is the most important check. It looks for repeated frames — a sign that someone is playing a pre-recorded video instead of using a live camera.

How it works:
Every frame has a fingerprint called an "average hash" — an 8x8 grid of 64 bits (ones and zeros) representing the rough brightness pattern of the face crop.
For each frame, the system finds another frame approximately 300 milliseconds later (using the server-recorded timestamps).
It compares the two fingerprints by counting how many of the 64 bits are different (the Hamming distance).

A real live person is always slightly moving — breathing, micro-expressions, tiny head movements, detector noise.
Over 300 milliseconds, at least 2 to 4 bits will change even if they are sitting perfectly still.
A looped video or replayed recording sends identical frames over and over.
Two frames 300 ms apart from a replay loop will have a Hamming distance of 0 (or at most 1).

The system counts what percentage of all frame pairs have a Hamming distance of 1 or less.
That percentage becomes the repeat frame score (0–100).

A secondary check also looks at brightness flicker — how much the average face brightness jumps between consecutive frames.
Natural lighting has small stable fluctuations. A screen being filmed flickers with the screen's refresh rate, causing larger and more irregular brightness changes.

    Final replay score = 70% x repeat frame score + 30% x flicker score

### Check 2 — Movement Consistency (weight: 7%)

This checks whether the head movements during the challenges look smooth and natural, or jerky and unnatural.

How it works:
It looks at the yaw (left-right) and pitch (up-down) values across each challenge frame group.
For each group, it calculates the "jerk" — the change-in-change of the movement.
If your speed changes smoothly, that is natural human motion. If your speed changes abruptly in every frame, something is wrong.

Formally: jerk = average of |change in (change in yaw)| across consecutive frames.

Important: this check only uses frames from still challenges (Look Straight and Blink).
During turn and nod challenges, the person is intentionally accelerating and decelerating, which naturally produces high jerk even for a real person.
Using those frames for this check would give false alarms.
Restricting to still challenges means we are only measuring jitter in frames where a real face should be almost completely motionless.

This check now detects **two distinct attack patterns**, not just one:

**High-jitter risk (replay loops):** A looped or injected video stream sometimes causes erratic, unnaturally high jerk because the face tracking jumps between positions at fixed intervals. This is captured by the existing formula `score = jerk × 240 + ...`.

**Stillness risk (static photographs):** A live person at rest always has micro-tremors from breathing and natural head sway. These produce a combined (yaw + pitch + nose) jitter of at least ~0.025. A static photograph held in front of the camera produces only face-detector rounding noise, which lands below 0.020 combined. If the combined jitter across the still-challenge group falls below a **liveness floor of 0.025**, a proportional stillness-risk penalty is added — up to 55 points at absolute zero motion. This means a static photo used for Look Straight will raise the temporal risk score substantially, pushing the overall risk score above the SUSPICIOUS threshold when the replay check is also elevated.

In the real-world hybrid Attack 1 (static photo for Look Straight, `combined_jitter ≈ 0.014`): `stillness_risk ≈ 24 points`. This raises the temporal score from ~4 to ~28.

In the harder **hybrid Attack 2** (static photo for Look Straight + many wrong-motion attempts diluting overall replay), `combined_jitter ≈ 0.024` is just barely below the liveness floor, so the stillness penalty only adds ~3 points. The temporal weight is 7% (reduced from 22%), so this check alone cannot push the score over SUSPICIOUS. The **Still-Challenge Replay check (Check 2b)** covers this gap instead.

    Temporal score = replay_jitter_risk + stillness_risk
    where:
        replay_jitter_risk = (yaw_jerk × 240) + (pitch_jerk × 240) + (nose_jitter × 320)
        stillness_risk     = max(0, liveness_floor − combined_jitter) / liveness_floor × 55
        liveness_floor     = 0.025 (minimum expected combined jitter for a live person at rest)

### Check 2b — Still-Challenge Replay (weight: 15%)

This is the primary defence against the hybrid attack where a static photo is held for the Look Straight challenge while the real person performs the motion challenges.

**Why a dedicated check is needed:**  
The overall Replay Detection check (Check 1) operates on *all session frames*. When 9 wrong-motion attempts are made before passing a turn challenge, the 30+ genuine motion frames produce varied hashes that pull the session-wide `repeat_frame_score` below 50, preventing the Check 1 SUSPICIOUS gate from firing — even though the 15 Look Straight photo frames are near-identical.

**Per-challenge scoring (the key architectural change):**  
Every completed challenge is now scored independently in its own frame window. For still challenges (Look Straight and Blink), both temporal jitter and replay heuristics are computed on only that challenge's frames. For motion challenges (Nod, Turn Left, Turn Right), only replay is computed (temporal jitter during intentional motion is naturally high for any genuine person).

Scoring each challenge independently means:
- A clean blink window cannot dilute the score of a suspicious look_straight window, or vice versa.
- A looped Turn Left window is detected even if Look Straight was genuine.
- The session result always reflects the *worst* single challenge — the most suspicious evidence window.

The session-level `temporal` and `still_replay` scores used in the risk formula are taken from the worst (highest-risk) single still-challenge, not a pooled average.

**Per-challenge SUSPICIOUS gate (asymmetric thresholds):**  
Because still and motion challenges have fundamentally different natural replay baselines, the gate uses different thresholds for each type:

- **Still challenges** (`look_straight`, `blink`): `combined_risk ≥ 50` → SUSPICIOUS.
  A breathing, blinking person sitting in front of the camera always has enough micro-movement (head sway, expression changes, detector noise) to keep per-still-challenge replay well below 50%. Any score above this floor is therefore very unlikely to be innocent.
  Combined formula: `40% × temporal_score + 60% × replay_score`

- **Motion challenges** (`nod`, `turn_left`, `turn_right`): `combined_risk ≥ 70` → SUSPICIOUS.
  During a nod or turn the subject must hold the confirmed position for the camera to register it. That hold phase produces several consecutive near-identical frames, naturally pushing `repeat_frame_score` into the 50–65 range even for a completely genuine person. Setting the threshold at 70% eliminates these false positives while still catching a looped or replayed motion challenge video, which produces 90–100% near-identical pairs.
  Combined formula: `100% × replay_score` (temporal is excluded — see verdict table notes above)

**Hard gate (independent of overall risk score):**  
If `repeat_frame_score ≥ 70%` *and* at least 8 frames were captured in the worst still-challenge window, the verdict is automatically SUSPICIOUS regardless of the overall risk score. A live person breathing during look_straight never produces 70%+ near-identical frame pairs; only a static photo source does. The frame-count guard (≥ 8) prevents accidental hash collisions in very short windows from triggering this gate.

**Evidence — named challenge:**  
When any challenge fires the suspicious gate, the evidence bullets name the exact challenge and explain why: e.g. *"Challenge 'Look Straight' was flagged suspicious (score=82/100): 95% of frame pairs were near-identical (expected < 30% for a live face) — static photo suspected."*

    Per-challenge combined_risk:
        still challenge = 40% × temporal_score + 60% × replay_score
        motion challenge = 100% × replay_score
    Hard gate: worst_still_challenge.repeat_frame_score ≥ 70 AND frames_used ≥ 8 → SUSPICIOUS

### Check 3 — Image Quality (weight: 10%)

This checks whether the live frames were sharp enough and well-lit enough to trust the other checks.

How it works:
It averages the blur score, brightness balance, and face size ratio across all frames.
- Blur risk: same as the static check — low edge variance = blurry = high blur risk.
- Brightness risk: same as static — too dark or too bright = high risk.
- Face size risk (live-calibrated): the scale is calibrated for the live session oval, not for photos. Real sessions where the person fills the browser oval have a face area ratio of roughly 7–12% of the frame. The scoring scale is (0.05, 0.10): face area at or below 5% scores risk 100, face area at or above 10% scores risk 0. This is intentionally tighter than the static photo scale because the oval guides the user to a predictable framing — if the face is genuinely small during a live session, it means the person is too far away or not positioned in the oval.

    Quality risk = 50% x blur risk + 30% x brightness risk + 20% x face size risk

The narrative reports "face too small in frame" when face size risk reaches 50 or above (face area below ~7.5%). At normal webcam distance with the face filling the oval, the face area ratio is above 10%, which scores 0 risk and suppresses this warning entirely.

This does not affect the verdict directly; it mainly reduces the confidence score.

### Check 4 — Eye Micro-Jitter (weight: 5%)

This checks whether the eyes made tiny involuntary movements — something real eyes always do.

How it works:
Human eyes never stay perfectly still. Even when staring at a fixed point, they make tiny rapid movements roughly every 100–200 milliseconds. These are completely involuntary.
A printed photo, a static image on a screen, or a looped video has perfectly frozen eye positions frame after frame.

The system collects the left-eye and right-eye x and y positions (as a fraction of the face width and height) across at least 6 frames.
It then removes the slow drift caused by head movement, by subtracting the best-fit straight-line trend from the position series.
What remains is the micro-jitter — the tiny residual fluctuation after the main movement has been removed.

It calculates the standard deviation of that residual for each of the four coordinates and averages them.

- Standard deviation above 0.003 to 0.004 — normal live eye movement — low risk.
- Standard deviation below 0.0005 — eyes are suspiciously frozen — high stillness risk.

### Check 5 — Screen Frequency Pattern (weight: 4%)

This checks whether the face was being filmed from a phone or computer screen rather than in real life.

How it works:
When a camera films a digital screen (phone, laptop, TV), the LCD or OLED pixel grid creates a subtle pattern of repeating bright lines.
This pattern shows up when you analyse the image mathematically for repeating spatial patterns.

Every 5 frames, the system takes the face crop and runs this frequency analysis.
It looks at a specific range of medium-sized repeating patterns that match the pixel pitch of typical screens at selfie distances.
The key measurement is: what fraction of the energy in that pattern range is concentrated in just the top 2% of hotspot locations?

A filmed screen has 2–4 very bright, localised spots in this range (from the pixel grid), so the concentration is high (typically 0.35–0.70).
A real face has organic irregular texture, so the energy is spread across many locations — low concentration (typically 0.05–0.18).
JPEG compression artefacts also add energy to this range, but it is spread broadly, not concentrated.

The system averages the concentration ratio across all measured frames.
- Mean ratio below 0.20 — looks like a real face — no additional risk.
- Mean ratio above 0.40 — looks like a filmed screen — high screen risk.

### Check 6 — Frame Delivery Timing (weight: 1%)

This checks whether frames arrived at suspiciously uniform intervals — a sign that a replay tool is sending them rather than a real browser.

How it works:
The server records the exact time it receives each frame using a server-side clock that the client cannot manipulate.
It calculates the time gaps between consecutive frames in milliseconds.
It then computes the Coefficient of Variation (CV) — the standard deviation of those gaps divided by the average gap.
CV measures how irregular the timing is. High CV = irregular = natural. Low CV = perfectly regular = suspicious.

A real browser: the browser event loop, OS scheduling, encoding time, and network buffering all add unpredictable, variable delays. Real sessions have a CV of 0.15–0.50.
A replay injection tool: it sends frames at a perfectly fixed interval, like a metronome. CV below 0.05.

- CV below 0.03 — metronomic timing — very suspicious — high timing risk.
- CV above 0.15 — organic irregular timing — consistent with a real browser — low risk.

Gaps larger than 2 seconds are excluded (those are reconnection pauses, not the normal delivery rhythm).

### Check 7 — 3D Depth Consistency (weight: 8%)

This checks whether the face has real 3D depth by watching how the eye separation changes during head turns.

How it works:
When a real 3D head turns to one side, the far eye gradually moves behind the nose (perspective effect).
The visible distance between the two eyes gets smaller as the head turns further.

A flat 2D object — a printed photo, a mask, or a screen — has no depth.
If you physically rotate a flat object sideways, both "eyes" are still on the same flat surface, so their apparent separation stays constant no matter how much it is tilted.

The system looks at all the frames captured during turn challenges.
For each frame, it records how much the head has turned (yaw) and the eye separation ratio (distance between eyes divided by face width).
It then computes the Pearson correlation between these two series.

- Strong negative correlation (below -0.50): as the head turned more, eye separation shrank — real 3D face — low risk.
- Correlation between -0.50 and 0.00: shallow or absent depth — borderline source (e.g. a close-fitting 3D-printed mask or a doll) — moderate to high risk.
- Correlation near zero or positive: eye separation did not decrease even when the head turned — flat source — high risk.
- If the eye separation never changes at all across all turn frames — that is extremely suspicious — risk score of 80 immediately.

The risk formula is: `flat_face_risk = scale(correlation + 1.0, 0.50, 1.00)`. This was calibrated against real session data:
- Real human faces: `iod_yaw_correlation` typically −0.40 to −0.99 → risk 0.
- Plastic doll (shallow 3D, measured at −0.12): risk ≈ 77 → SUSPICIOUS.
- Flat photo or mask (correlation ≈ 0 to +0.30): risk 100 → SUSPICIOUS.

The previous (0.70, 1.20) calibration was too lenient — a plastic doll scored only ~37, staying below the 65-point SUSPICIOUS threshold.

This check is skipped entirely if no turn challenges were included in the session.

### Virtual Camera Check

Before any frames are processed, when the browser sends its metadata message, the server reads the video track device label — the name the operating system gives to the camera.

A real camera has a name like "FaceTime HD Camera", "Logitech C920", or "USB Camera".
Virtual camera software (tools that can feed pre-recorded video into the camera input) has a recognisable name — like "OBS Virtual Camera", "Snap Camera", "ManyCam", "DroidCam", or similar.

The server checks the device label against a list of 15 known virtual camera tool name fragments:
obs, virtual, manycam, snap camera, droidcam, epoccam, ivcam, xsplit, mmhmm, iriun, camo, e2esim, ndi virtual input, wirecast, logitech capture.

If any of those words appear in the device name (case-insensitive), the session is flagged with virtual_camera_suspected: true.
This flag does not change the risk score directly, but it appears in the evidence and is visible to the operator.

---

## How the Final Live Risk Score Is Calculated

Once all challenges are done and all seven checks have run, the heuristic risk score is:

    Risk score = (50% × replay) + (7% × temporal) + (15% × still_replay) + (10% × quality) + (5% × eye_jitter) + (4% × screen_fft) + (1% × timing) + (8% × depth)
    Weights: 0.50 + 0.07 + 0.15 + 0.10 + 0.05 + 0.04 + 0.01 + 0.08 = 1.00

All scores are 0–100. The total is clamped between 0 and 100.

| Signal | Weight | Why this weight |
|---|---|---|
| Replay (session-wide) | 50% | Repeated-frame injection is the most common and most directly measurable attack vector. Session-wide scope catches looped video that spans the whole session. |
| Still-Challenge Replay | 15% | Isolates replay specifically within Look Straight and Blink windows. Prevents dilution by genuine motion frames in hybrid attacks (e.g. photo for look_straight + live person for turns). |
| Temporal consistency | 7% | Reduced from 22% because `still_replay` now covers the overlap. Temporal still contributes the stillness-risk penalty for near-zero jitter (static photo signal). |
| Image quality | 10% | Poor quality (blur, lighting) can mask the other checks; factoring it in reduces false confidence. |
| Eye micro-jitter | 5% | Sessions without the correct challenge produce 0 on this check — kept small so short sessions are not penalised. |
| Screen frequency | 4% | Reliable when a screen is filmed; absent for non-screen attacks. Small weight for consistency. |
| Frame delivery timing | 1% | Very small because many legitimate setups (screen recorders, remote desktop) naturally produce metronomic timing even when the user is genuine. |
| 3D depth | 8% | Strong signal against flat-face attacks; skipped (0 risk) for sessions with no turn challenges. |

### Phase 1 — XGBoost ML Scorer (active, cold-start mode)

A machine-learning risk scorer (`src/basetruth/face_scan/ml_scorer_live.py`) has been implemented alongside the heuristic formula.
It uses the same 20 signals as inputs and trains an XGBoost binary classifier (genuine=0, spoof=1) to learn optimal weights from labeled session data.

How it works:
- When the model file `fraud_model/models/ml_scorer_face_scan_live.pkl` exists, the ML model replaces the fixed-weight heuristic formula and returns a spoof probability (0–1) converted to a 0–100 risk score.
- When the model file does not exist (cold-start, before enough training data has been collected), the heuristic formula above runs unchanged. The `scoring_method` field in the result JSON shows `"ML"` or `"heuristic"` so operators can see which mode is active.
- Every completed session — regardless of verdict — is appended as a row to `fraud_model/data/training_data_face_scan_live.csv`. The CSV stores all 20 feature values plus the system verdict and a `label` column (−1 = unconfirmed, 0 = confirmed genuine, 1 = confirmed spoof).
- Once enough verified examples are collected (~50 minimum, 500+ for production), the model is trained by running: `src/basetruth/face_scan/ml_scorer_live.py train`. A 5-fold stratified cross-validation is run and the model is saved only if ROC AUC ≥ 0.75.
- SHAP feature contributions are supported via `explain(feature_vector)` — the same pattern as the image document ML scorer.

The 20 features the ML scorer uses:

| Feature | Source check |
|---|---|
| `yaw_jerk` | Temporal consistency |
| `pitch_jerk` | Temporal consistency |
| `nose_jitter` | Temporal consistency |
| `temporal_consistency_score` | Temporal consistency |
| `repeat_frame_score` | Replay heuristics |
| `flicker_score` | Replay heuristics |
| `brightness_instability` | Replay heuristics |
| `mean_eye_jitter` | Eye micro-jitter |
| `iod_yaw_correlation` | 3D depth consistency |
| `mean_fft_grid_peak` | Screen frequency |
| `interval_cv` | Frame delivery timing |
| `observed_fps` | Session environment |
| `frame_drop_rate` | Session environment |
| `mean_face_area_ratio` | Quality assessment |
| `blur_risk_0_100` | Quality assessment |
| `brightness_risk_0_100` | Quality assessment |
| `wrong_action_count` | Active liveness |
| `challenge_count` | Active liveness |
| `frames_without_face` | Face detection |
| `virtual_camera_suspected` | Virtual camera flag |

### Confidence Score (Live)

    Confidence = 100 - (35% x blur risk) - (25% x brightness risk) - (20% x frame count penalty) - (20% x face tracking dropout)

Frame count penalty: if fewer than 30 frames were captured, confidence drops.
Face tracking dropout: if the face was not found in more than 25% of frames, confidence drops.
Confidence is clamped between 5 and 99.

### Verdict Thresholds (Live)

Conditions are evaluated **in this exact order**. The first match wins.

| Step | Condition | Verdict | Notes |
|---|---|---|---|
| 1 | Not all challenges completed | LIVENESS_FAILED | Session abandoned or timed out |
| 2 | `confidence < 35` | INCONCLUSIVE | Image or tracking quality too poor to decide |
| 3 | `replay_score > 80` | DEEPFAKE | Extremely high session-wide repeated frames — near-certainty of looped source |
| 4 | **Per-challenge gate** — any still challenge (`look_straight`, `blink`) has `combined_risk ≥ 50`, **or** any motion challenge (`nod`, `turn_left`, `turn_right`) has `combined_risk ≥ 70` | SUSPICIOUS | Still threshold is 50: even moderate repeat frames in a still window are suspicious (a breathing person never holds steady enough to hit 50%). Motion threshold is raised to 70 because the hold phase of a genuine nod or turn naturally produces 50–65% repeat frames — requiring 70% eliminates these false positives while still catching looped motion video. |
| 5 | `max(replay_score, temporal_score, still_replay_score) ≥ 50` **AND** `risk_score ≥ 35` | SUSPICIOUS | Belt-and-braces: any one strongly elevated sub-score, corroborated by a 35+ overall risk. The `still_replay_score` is included so a photo used for `look_straight` (which drives still_replay very high) is caught here even if overall session replay is diluted by genuine motion frames. The `risk_score ≥ 35` guard prevents isolated spikes (e.g. dark-room hash collisions) from triggering SUSPICIOUS in an otherwise clean session. |
| 6 | **Hard still-replay gate** — `still_replay.repeat_frame_score ≥ 70` **AND** `still_frames_used ≥ 8` | SUSPICIOUS | No risk_score guard needed: 70%+ of 300 ms-apart pairs near-identical within the worst still-challenge window is virtually impossible for a live person. The frame-count condition (≥ 8) prevents spurious collisions in very short windows from triggering this gate. |
| 7 | `depth_score ≥ 65` | SUSPICIOUS | Flat-face gate: turn challenges completed but 3D IOD change is near zero — consistent with a physical photo or flat object being moved in front of the camera. |
| 8 | None of the above | GENUINE | — |

**Per-challenge combined_risk formula (used in step 4):**

    Still challenge  (look_straight, blink):  combined_risk = 40% × temporal_score + 60% × replay_score
    Motion challenge (nod, turn_left, turn_right): combined_risk = 100% × replay_score

Temporal score is excluded from motion challenges because intentional acceleration and deceleration during a turn or nod naturally produces high jitter even for a genuinely live person — using temporal there would cause constant false alarms.

**Worked examples:**

| Scenario | Key scores | Gate that fires | Verdict |
|---|---|---|---|
| Live person, all clear | replay=8, still_replay=12, temporal=5, risk=9 | None | GENUINE |
| Genuine nod, hold phase (57% repeat) | nod combined_risk=57 (< 70 motion gate), risk=27 | None | GENUINE |
| Photo for look_straight, live for rest | look_straight combined_risk=64 (≥ 50 still gate) | Step 4 | SUSPICIOUS |
| Photo for look_straight (few frames), live motion fills session | still_replay=85, risk=38 | Step 5 (still_replay ≥ 50 AND risk ≥ 35) | SUSPICIOUS |
| Full replay loop, all challenges looped | replay=91 | Step 3 | DEEPFAKE |
| Flat photo tilted for turns, zero IOD change | depth=78 | Step 7 | SUSPICIOUS |

---

## Known Attack Types and Coverage

This table documents which spoofing attack categories the current system covers, partially covers, or does not yet address. It is intended to set honest expectations with operators and stakeholders.

| Attack type | Description | Coverage |
|---|---|---|
| Photo of a photo (print attack) | Attacker holds a printed photo of the target in front of the camera | ✅ Covered — edge halo and compression residual checks detect the re-photograph signature |
| Screen replay | Pre-recorded video played from a phone or laptop | ✅ Covered — repeated-frame hashing, screen-frequency pattern, and frame-timing checks |
| Deepfake video | AI-generated or face-swapped video of the target | ⚠️ Partial — movement consistency, eye micro-jitter, and 3D depth checks catch many deepfakes; high-quality deepfakes with natural motion may pass |
| Mask attack | Physical 3D mask of the target worn by an attacker | ❌ Not covered — requires infrared depth sensors or texture-based anti-spoofing models not yet integrated |
| 3D avatar / rendered face | Photorealistic CGI avatar presented live | ❌ Not covered — screen-frequency check may catch rendered output displayed on a screen, but direct GPU injection bypasses it |
| GAN-generated live stream | Fully synthetic face stream injected directly into the video pipeline | ❌ Not covered — requires dedicated GAN-detection models; virtual camera flag provides a partial signal |

---

## What the Result Looks Like

Both the static and live modes return the same shaped result:

    verdict: GENUINE
    risk_score_0_100: 19.3
    confidence_0_100: 94.8
    honest_review: "The live session looks genuine based on the current challenge-response and replay checks."
    evidence:
      - "Completed live challenges: blink, turn_left, nod."
      - "No strong repeated-frame or replay-screen pattern was found in the live capture."
      - "Head and eye motion stayed reasonably consistent across the live challenge sequence."

The result also contains a checks section with the raw numbers from every individual check,
a trace section with a unique ID, timestamp, and which version of the rules was used,
and an environment section recording the camera resolution, observed FPS, browser, operating system, and whether a virtual camera was suspected.

---

## Evaluation Metrics

The system does not yet have a labeled ground-truth dataset for formal evaluation. Operators deploying BaseTruth in production are strongly encouraged to collect and label verified examples (known-genuine and known-spoofed) so the following metrics can be measured:

| Metric | What it measures | Target |
|---|---|---|
| **FAR** (False Acceptance Rate) | How often a spoof image is accepted as genuine (risk < 35) | As low as possible |
| **FRR** (False Rejection Rate) | How often a genuine face is rejected as suspicious or deepfake | Keep below acceptable UX threshold |
| **ROC AUC** | Overall discrimination ability across all thresholds | Closer to 1.0 = better separation |
| **INCONCLUSIVE rate** | How often the system cannot decide due to quality | Should be low for controlled capture environments |

The current verdict thresholds (risk ≥ 75 → DEEPFAKE, risk ≥ 35 → SUSPICIOUS) were chosen to be conservative: it is safer to flag a suspicious case for review than to silently accept a spoof. This biases the system toward lower FAR at the cost of a higher FRR. Once a labeled dataset is available, these thresholds can be tuned on a FAR/FRR operating-point curve to match the specific risk tolerance of the deployment.

---

## Scoring Philosophy

The static and live modes use independent analysis pipelines because the available signals are entirely different — a still photo provides no motion data and a live session provides no compression artefacts. Both pipelines output a risk score on the same shared 0–100 scale with identical verdict thresholds (GENUINE, SUSPICIOUS, DEEPFAKE, INCONCLUSIVE), so operators can compare results across both modes without needing to understand the internal differences.

---

## Files That Implement This

| What | Where |
|---|---|
| Static photo check logic | src/basetruth/face_scan/service.py |
| Live session orchestration, all 7 checks, browser page HTML | src/basetruth/face_scan/live.py |
| Challenge pass/fail logic (blink, nod, turns, look straight, stability gate) | src/basetruth/kyc/liveness.py |
| Face Scan live session state (face_stable_frames counter, challenge history) | src/basetruth/face_scan/live.py |
| XGBoost ML scorer: feature extraction, predict(), train(), explain(), CSV append | src/basetruth/face_scan/ml_scorer_live.py |
| Plain-English narrative generation (Gemma4 LLM + rule-based fallback) | src/basetruth/face_scan/narrative.py |
| Face detection (InsightFace + MediaPipe fallback) | src/basetruth/vision/face.py |
| API routes (session create, WebSocket, result fetch) | src/basetruth/api.py |
| Streamlit UI page (two tabs, challenge picker, result display) | src/basetruth/ui/pages/face_scan.py |
| Unit tests for live checks and ML scorer | tests/test_face_scan_live.py |
| Unit tests for challenge detection and stability gate | tests/test_kyc_liveness.py |
| Unit tests for static scan | tests/test_face_scan_service.py |
| Unit tests for API routes | tests/test_face_scan_api.py |
| Unit tests for ML scorer (train, predict, explain, CSV append) | tests/test_ml_scorer_live.py |
| Labeled training data (one row per completed live session) | fraud_model/data/training_data_face_scan_live.csv |
| Trained XGBoost model (absent until first training run) | fraud_model/models/ml_scorer_face_scan_live.pkl |
