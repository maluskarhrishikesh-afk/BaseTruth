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

1. The operator goes to the Face Scan screen and picks the challenges they want: Blink, Turn Left, Nod, Turn Right, or Look Straight. The default set is blink, turn left, and nod.
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
3. **Face area ≥ 3% of frame** — the person must be close enough to the camera to fill a meaningful part of the oval. A tiny or distant face fails this check.
4. **Nose within the centre band** (35%–65% of frame width) — the person's face must be roughly centred before challenges begin.
5. **Geometry invariants pass** — the face must look like a real human face (eyes above nose, mouth below eyes, proportions within normal ranges).

If any condition fails, the counter resets to zero. The counter must reach **8 consecutive qualifying frames** (approximately 1 second at 8 FPS) before challenge history starts accumulating.

While the counter is below 8, the server responds with:
- `face_stable: false`
- `face_stable_progress` (current count)
- `face_stable_required` (8)
- A human-readable feedback message explaining what the person needs to fix (e.g. "Move closer to the camera — your face is too far away.")

Once the counter reaches 8, `face_stable: true` is set on all subsequent status messages, and the normal challenge-accumulation logic begins.

### Constants (in `src/basetruth/kyc/liveness.py`)

| Constant | Value | Meaning |
|---|---|---|
| `FACE_STABLE_FRAMES_REQUIRED` | 8 | Consecutive qualifying frames before challenges start |
| `FACE_STABILITY_CONFIDENCE_MIN` | 0.80 | Minimum detection confidence for the stability gate |
| `FACE_STABILITY_X_MIN` / `X_MAX` | 0.35 / 0.65 | Horizontal centering band |
| `MIN_FACE_DETECTION_CONFIDENCE` | 0.55 | Minimum confidence during active challenges |
| `MIN_FACE_AREA_RATIO` | 0.03 | Minimum face area (3% of frame) at all times |

---

## How Each Challenge Works

All measurements are done on the mirrored frame (the same orientation the person sees in their selfie preview).
So "the person turns to their own left" means the nose moves toward the left side of the image.

### Look Straight

This challenge asks the person to face the camera directly and hold still.
It is optional but useful — when it passes, the server saves the current frame as the best face capture for the session.

How it detects this:
It looks at the nose's horizontal position within the face bounding box across the last 3 consecutive frames.
The nose must stay between 40% and 60% of the face width (roughly centred) for all 3 of those frames.
3 frames at 8 FPS takes about 375 milliseconds.

If the nose is too far left, the feedback says "Move slightly to YOUR right to centre."
If the nose is too far right, the feedback says "Move slightly to YOUR left to centre."

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

The minimum frame count for the blink check is 3 frames (about 375 ms at 8 FPS).
This is enough to capture the baseline, dip, and reopen.

### Nod

This challenge asks the person to nod their head down and back up.

How it detects this:
It watches the pitch measurement (how far the nose is below the eye midpoint).
When you nod your chin down, the nose drops further below the eyes, increasing the pitch.
When you bring your head back up, the pitch decreases.

The system checks the last 4 frames and measures the range: (maximum pitch minus minimum pitch).
If that range is at least 0.12 (meaning the nose moved at least 12% of the eye-to-eye distance up and down), the nod is detected.
4 frames at 8 FPS takes about 500 milliseconds — long enough to catch a real nod.

### Turn Left

This challenge asks the person to slowly turn their head to their own left.

How it detects this:
It watches the yaw measurement.
When you turn left, the nose moves toward the left side of the image (away from the eye midpoint), making yaw negative.

The check passes if the most negative yaw value seen in the last 20 frames is at or below -0.16
(meaning the nose has swung at least 16% of the eye-to-eye distance to the left).

There is also a fallback: if the person started centred and the nose shifted left by at least 10% of the face width AND the yaw changed by at least 0.09 from the starting position, that also counts as a turn.
This fallback helps people who cannot do a large turn due to physical constraints.

### Turn Right

Exactly the same logic as Turn Left, but in the opposite direction.
Yaw must reach at least +0.16 (nose swung to the right).

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

### Check 2 — Movement Consistency (weight: 25%)

This checks whether the head movements during the challenges look smooth and natural, or jerky and unnatural.

How it works:
It looks at the yaw (left-right) and pitch (up-down) values across each challenge frame group.
For each group, it calculates the "jerk" — the change-in-change of the movement.
If your speed changes smoothly, that is natural human motion. If your speed changes abruptly in every frame, something is wrong.

Formally: jerk = average of |change in (change in yaw)| across consecutive frames.

A real person moving their head: the jerk will be low, because human muscles produce smooth, continuous movements.
A static photo being physically moved or a looped video: the detector measurements jump erratically because the face tracking loses and re-acquires the face at random positions, producing high jerk.

Important: this check only uses frames from still challenges (Look Straight and Blink).
During turn and nod challenges, the person is intentionally accelerating and decelerating, which naturally produces high jerk even for a real person.
Using those frames for this check would give false alarms.
Restricting to still challenges means we are only measuring jitter in frames where a real face should be almost completely motionless.

### Check 3 — Image Quality (weight: 10%)

This checks whether the live frames were sharp enough and well-lit enough to trust the other checks.

How it works:
It averages the blur score, brightness balance, and face size ratio across all frames.
- Blur risk: same as the static check — low edge variance = blurry = high blur risk.
- Brightness risk: same as static — too dark or too bright = high risk.
- Face size risk: same as static — face too small in the frame = high risk.

    Quality risk = 50% x blur risk + 30% x brightness risk + 20% x face size risk

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

### Check 5 — Screen Frequency Pattern (weight: 5%)

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

### Check 6 — Frame Delivery Timing (weight: 3%)

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

### Check 7 — 3D Depth Consistency (weight: 2%)

This checks whether the face has real 3D depth by watching how the eye separation changes during head turns.

How it works:
When a real 3D head turns to one side, the far eye gradually moves behind the nose (perspective effect).
The visible distance between the two eyes gets smaller as the head turns further.

A flat 2D object — a printed photo, a mask, or a screen — has no depth.
If you physically rotate a flat object sideways, both "eyes" are still on the same flat surface, so their apparent separation stays constant no matter how much it is tilted.

The system looks at all the frames captured during turn challenges.
For each frame, it records how much the head has turned (yaw) and the eye separation ratio (distance between eyes divided by face width).
It then computes the Pearson correlation between these two series.

- Strong negative correlation (below -0.30): as the head turned more, eye separation shrank — real 3D face — low risk.
- Correlation near zero or positive: eye separation did not decrease even when the head turned — flat source — high risk.
- If the eye separation never changes at all across all turn frames — that is extremely suspicious — risk score of 80 immediately.

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

Once all challenges are done and all seven checks have run, the risk score is:

    Risk score = (50% x replay) + (25% x consistency) + (10% x quality risk) + (5% x eye jitter) + (5% x screen) + (3% x timing) + (2% x depth)

All scores are 0–100. The total is clamped between 0 and 100.

Weights are empirically tuned based on internal testing and observed fraud patterns. Replay carries the largest share because repeated-frame injection is both the most common attack vector and the most directly measurable signal. The four newer signals (eye jitter, screen frequency, timing, depth) carry small shares because sessions that lack the required challenges score 0 on those checks — keeping the scale fair for shorter sessions.

Replay is weighted most heavily because it is the most reliable and direct signal of a fraud attempt.
The four newer signals (eye jitter, screen, timing, depth) each contribute a small share so that sessions without those specific challenges do not get unfairly penalised.
For example, a session with no turn challenges scores 0 on depth, which adds zero risk.

### Confidence Score (Live)

    Confidence = 100 - (35% x blur risk) - (25% x brightness risk) - (20% x frame count penalty) - (20% x face tracking dropout)

Frame count penalty: if fewer than 30 frames were captured, confidence drops.
Face tracking dropout: if the face was not found in more than 25% of frames, confidence drops.
Confidence is clamped between 5 and 99.

### Verdict Thresholds (Live)

| Condition checked in order | Verdict |
|---|---|
| Challenges were not all completed | LIVENESS_FAILED |
| Confidence below 35 | INCONCLUSIVE |
| Replay score above 80 | DEEPFAKE |
| Replay score OR consistency score is 50 or above | SUSPICIOUS |
| Everything else | GENUINE |

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
| Session state (face_stable_frames counter, challenge history) | src/basetruth/kyc/session.py |
| Face Scan live session state (face_stable_frames counter) | src/basetruth/face_scan/live.py |
| Session state (face_stable_frames counter, challenge history) | src/basetruth/kyc/session.py |
| Face detection (InsightFace + MediaPipe fallback) | src/basetruth/vision/face.py |
| API routes (session create, WebSocket, result fetch) | src/basetruth/api.py |
| Streamlit UI page (two tabs, challenge picker, result display) | src/basetruth/ui/pages/face_scan.py |
| Unit tests for live checks | tests/test_face_scan_live.py |
| Unit tests for challenge detection | tests/test_kyc_liveness.py |
| Unit tests for static scan | tests/test_face_scan_service.py |
| Unit tests for API routes | tests/test_face_scan_api.py |
