# Identity Verification & Fraud Detection

BaseTruth incorporates a robust, **offline-first** computer vision pipeline specifically engineered for identity fraud detection. This allows the system to verify that the person on an ID document (Aadhaar, PAN, Passport) matches a provided live Selfie, mathematically scoring the likelihood of fraud.

## Core Technology Stack

The pipeline runs locally without making any external API calls, ensuring high privacy, reducing costs, and keeping latency strictly bounded by the edge hardware.

| Component | Responsibility | Purpose in Flow |
| --- | --- | --- |
| **OpenCV** (`cv2`) | Image Processing | Decodes raw byte streams from the UI, manages RGB/BGR color space conversions, heavily resizes images for memory safety, draws visual forensic evidence (bounding boxes), and detects/decodes QR codes on Aadhaar cards. |
| **MediaPipe FaceLandmarker** | Face Detection (primary) | Google's on-device model that detects 468 facial landmarks and outputs blendshape scores (e.g. `eyeBlinkLeft`, `eyeBlinkRight`). Used as the default face detector on Python 3.13+. Model file: `your_data/models/face_landmarker.task` (auto-downloaded on first run). |
| **InsightFace (RetinaFace + ArcFace)** | Face Detection + Identity Recognition (optional) | Deep learning models that detect faces and produce 512-dimensional identity embeddings. Required for face-match scoring. Installs cleanly on Linux (Docker) or Windows with Python ≤ 3.12. |
| **ONNX Runtime** | Inference Engine | Executes InsightFace models locally on the CPU using the `buffalo_l` model pack. Required only when InsightFace is available. |
| **PaddleOCR** | OCR | Extracts recovery text from PAN card images when Gemma4 misses a field, so PAN validation stays on the same OCR engine used elsewhere in BaseTruth. |
| **Gemma4 via Ollama** | Vision extraction | Extracts PAN number, full name, father's name, and DOB from PAN images; used as the primary PAN extraction path. |

## Workflow

BaseTruth offers two variants of Identity Verification:

### 1. Multi-Document Identity Verification (Aadhaar + PAN + Selfie)

The primary identity verification flow mandates **three document inputs**:

1. **Aadhaar Card Upload** — The operator uploads a photo of the applicant's Aadhaar card.
   - OpenCV `QRCodeDetector` scans the card for the QR code.
   - If an old-format (pre-2018) QR is found, the XML payload is parsed to extract: full name, DOB/year of birth, gender, district, state.
   - If a new secure QR (2018+) is found, the system notes it is present but cannot reveal the encrypted payload offline.
2. **PAN Card Upload** — The operator uploads a photo of the applicant's PAN card.
   - Gemma4 (via Ollama) extracts the PAN number, cardholder full name, father's name, and date of birth from the card image.
   - PaddleOCR supplies recovery text when Gemma4 misses a field, so PAN extraction stays on the same OCR stack used for marksheets and scanned documents.
   - The PAN number is validated against the standard format (`ABCDE1234F`).
   - The 4th character is decoded to identify the entity type (Individual / Company / HUF etc.).
   - The extracted PAN full name is compared to the Aadhaar QR name using exact first-name and last-name matching with middle-name tolerance.
3. **Selfie or Camera Capture** — The operator uploads a selfie or triggers the browser camera.
   - If no selfie file is uploaded, `st.camera_input()` opens automatically.
4. **Deterministic Cross-Checks**
   - **First Name & Last Name Match** — Aadhaar QR name vs. PAN full name.
   - **DOB Match** — Aadhaar DOB or year of birth vs. PAN DOB.
   - **PAN Format & Entity Type** — Regex validation plus entity-type decoding from the 4th PAN character.
5. **Face Match** — ArcFace cosine similarity between the face on the Aadhaar card and the selfie (threshold > 0.40).
6. **Auto-fill Entity Form** — Extracted PAN name is used as a fallback when Aadhaar QR name is unavailable. PAN number and masked Aadhaar UID are pre-populated in the "Applicant Details" form, while father's name and DOB remain visible for operator review. The operator only needs to enter phone and email manually.
7. **Persistence** — Identity Verification saves one current row per entity in the `identity_checks` database table. The row stores structured Aadhaar and PAN payloads, similarity outcomes, MinIO object keys for the Aadhaar image, PAN image, selfie, PAN signature crop, and the generated PDF report. Identity Verification does **not** write Aadhaar or PAN payloads into `document_extractions`. In parallel, the save flow upserts section-level evidence into `layered_analysis_entries` for:
   - `Identity Verification` / `Aadhaar`
   - `Identity Verification` / `PAN Card`
   - `Identity Verification` / `Photo Upload`
   - `Identity Verification` / `Run Verification`
   - Aadhaar and selfie uploads also store shared upload-authenticity evidence so auditors can review format/structural validation and image-tampering checks consistently across uploads.
8. **Explainability Review** — Detailed audit evidence is shown on the dedicated Layered Analysis screen rather than inline on the main Identity Verification page.
9. **Final Report Locking** — If a final layered-analysis report has already been generated for the current evidence set, the Layered Analysis screen disables regeneration. Saving fresh identity evidence for the same entity automatically resets that flag.

### 2. Video KYC (WebSocket Liveness Challenge)

Designed to prevent impersonation and photo-spoofing fraud.  
The customer opens a link on **their own device** — no app or plugin needed.

```
Operator dashboard  ──POST /kyc/sessions──►  FastAPI server
                                               │ creates session in memory
                                               │ returns shareable URL
Operator shares URL with customer
                    ◄── Customer opens URL in browser
                         (served by GET /kyc/{session_id})
Customer opens camera
Customer's browser ──WS /kyc/ws/{session_id}──► FastAPI server
                    ◄── liveness instructions ──
                    ──── camera frame (JPEG/B64) ──►
                    ◄──── result / pass / fail ────
```

**Step-by-step:**

1. **Create session** — Operator clicks "Create Secure KYC Session" on the Video KYC page.
   - Uploads the customer's reference identity document; the system extracts a face embedding (ArcFace/InsightFace) to use as a reference for later matching.
   - Uploads a permanent address-proof document. Allowed address-proof documents are Aadhaar or Passport.
   - The backend extracts the address-proof payload before session creation and stores it in the in-memory KYC session.
   - `POST /kyc/sessions` is called on the FastAPI server; a 30-minute session is created.
   - A shareable URL like `http://your-server:8000/kyc/<session_id>` is returned.

2. **Customer opens the link** — The browser loads the self-contained HTML page served directly by the API server.
   - Works on any modern mobile or desktop browser; no install required.
   - Page shows Aadhaar upload, optional PAN upload, address proof, and GPS steps first. When the customer reaches liveness, BaseTruth redirects them into the shared Face Scan live page so Video KYC and Face Scan use the exact same live challenge UI.
   - PAN upload stores the raw image immediately and completes PAN field extraction plus signature cropping in the background, so slow Gemma4/OCR work does not block the browser or the operator status poller.

3. **Live challenges** — Customer clicks "Start Verification":
   - Browser requests camera permission.
   - The KYC page first calls `POST /api/v1/face-scan/sessions`, then redirects the browser into `/face-scan/live/<face_scan_session_id>?autostart=1&result_mode=verification&callback=...`.
   - The shared Face Scan live page opens the WebSocket connection to `/api/v1/face-scan/ws/<face_scan_session_id>` and captures a JPEG frame every ~100 ms.
   - The dedicated Face Scan live engine runs the challenge loop, computes the canonical live-authenticity result payload, and the shared page posts that JSON back to `POST /kyc/sessions/<session_id>/liveness-result` so the Video KYC session stores the exact Face Scan result.
   - A random set of 2–4 **active-liveness challenges** are assigned (configurable):

    | Challenge | What the server looks for |
       |---|---|
    | `blink` | The eyes start open, close clearly, then open again |
       | `turn_left` | The nose shifts left compared with the eye midpoint in the mirrored selfie-style frame |
       | `turn_right` | The nose shifts right compared with the eye midpoint in the mirrored selfie-style frame |
       | `nod` | The nose moves down and back up compared with the eyes over a short run of frames |

   - After each challenge passes, the server advances to the next one and sends a progress update.

    **How this works in simple language**

    - BaseTruth does not trust one frame. It watches a short recent sequence of frames and checks whether the movement really happened.
    - The browser preview is mirrored and the captured frame is mirrored too, so `turn left` means **your real left**, not the opposite direction.
    - The face must stay visible enough for the detector to see the eyes, nose, and mouth area.
    - For `look_straight`, the face must stay near the centre for **10 steady frames** (~1 second).
    - For `blink`, the main signal is **EAR (Eye Aspect Ratio)**. The engine looks for open eyes, then a dip when the eyelids close, then a reopen signal. It is tuned to still work on low-FPS webcams.
   - For `turn_left` and `turn_right`, the main signal is **yaw**. This is the nose position compared with the middle point between the eyes. The engine first checks for an absolute turn of about `0.16` in either direction, and if that is not reached it also checks whether the nose and yaw moved clearly away from the starting pose for that challenge.
    - For `nod`, the main signal is **pitch**. This is how far the nose moves down relative to the eyes. The server records a neutral-pitch baseline from the first 3 frames, then looks for a sustained deviation of at least **0.08** (about 8% of the eye-to-eye distance) held for **6 consecutive frames** (~0.6 s). A gentle chin-dip is enough — the user does not need to bend far down. Direction is accepted in either direction (some camera heights flip which way pitch goes for the same chin-down movement).
    - Distance from the camera matters less because these values are normalised using face size or eye distance.

4. **Current-location capture** — During the customer flow:
   - The browser requests geolocation permission.
   - Latitude, longitude, accuracy, and timestamp are sent to the backend and stored in the in-memory session.
   - The backend uses the latest reliable reverse-geocoding provider available in the deployment environment to derive a human-readable current address.
   - If reverse geocoding is unavailable, the raw coordinates are still preserved and the address comparison is marked `inconclusive`.

5. **Face match and evidence retention** — Once all liveness challenges pass:
   - BaseTruth keeps the best frontal live frame and one best frame per completed challenge.
   - If a reference embedding was provided in step 1, **ArcFace cosine similarity** is computed between the best live face frame and the reference embedding.
   - Threshold: similarity > 0.40 → PASS.
   - If no reference was provided, the session completes as a liveness-only check.
   - The backend also compares the current address against the extracted address-proof payload and calculates an address-distance metric when both sides can be normalized.

6. **Result** — Face Scan live sends its canonical `{"type":"result","verdict":...,"risk_score_0_100":...,"confidence_0_100":..., ...}` message via WebSocket.
   - The shared Face Scan page posts that payload to the KYC callback and then shows a verification-style PASS ✅ or FAIL ❌ result card using the bridged KYC response.
   - Operator dashboard polls `GET /kyc/sessions/{session_id}` for the outcome and can render the full Face Scan live review payload from the stored `face_scan_result` field.

**Face detection strategy:**

| Environment | Face Detector | Liveness | Face Match |
|---|---|---|---|
| Docker (Linux) | InsightFace (RetinaFace) | EAR from MediaPipe landmarks | ArcFace cosine similarity |
| Windows Python ≤ 3.12 | InsightFace (RetinaFace) | EAR from MediaPipe landmarks | ArcFace cosine similarity |
| Windows Python 3.13+ | **MediaPipe FaceLandmarker** | EAR via blendshapes | Skipped (liveness-only) |

The Docker deployment is intentionally pinned to Python 3.12 so the production
face-match stack can keep using the published InsightFace wheel builds.
Python 3.13 remains supported for local development through the MediaPipe
fallback path, while PaddleOCR handles marksheet OCR in both environments.

On Python 3.13+, `insightface` cannot be installed (native extension build fails). The server automatically falls back to MediaPipe — all liveness challenges work fully, and the face-match step is skipped with a clear message instead of failing silently.

## Cross-Document Checks

| Check | Input | Method | Fail Condition |
|---|---|---|---|
| First/last name match | Aadhaar QR ↔ PAN extraction | Exact first-name and last-name comparison | Either first or last name differs |
| DOB match | Aadhaar QR ↔ PAN extraction | Exact DOB match, with year fallback when Aadhaar only provides YOB | Full DOB or year mismatch |
| PAN format | PAN number | Regex `[A-Z]{5}[0-9]{4}[A-Z]` | Non-matching pattern |
| PAN entity type | PAN[3] | Lookup table | Unexpected entity type |
| Face match | Aadhaar face ↔ Selfie | ArcFace cosine similarity | Score < 0.40 (< 40% confidence) |

## Data Persistence

Identity Verification results are persisted in `identity_checks`:

| Field | Description |
| --- | --- |
| `entity_id` | Links to the `entities` table |
| `status` | `pass`, `fail`, or `inconclusive` |
| `cosine_similarity` / `display_score` / `threshold` / `is_match` | Face-match decision fields |
| `aadhar_dtls` | Aadhaar QR payload |
| `pan_dtls` | PAN extraction payload |
| `selfie_pic` / `aadhaar_pic` / `pan_pic` / `signature_pic` | MinIO object keys for the saved images |
| `report_json` | Full identity-verification payload |
| `pdf_report` | MinIO object key for the generated PDF report |

Video KYC results are persisted in `video_kyc_checks`:

| Field | Description |
| --- | --- |
| `entity_id` | Links to the `entities` table |
| `status` | `pass`, `fail`, or `inconclusive` |
| `cosine_similarity` / `display_score` / `threshold` / `is_match` | Live-face match decision fields |
| `liveness_state` / `liveness_passed` | Liveness result |
| `aadhar_dtls` / `pan_dtls` | Structured Aadhaar and PAN identity-proof payloads |
| `address_dtls` | Permanent address-proof payload |
| `current_location` | Reverse-geocoded current address text from the browser GPS fix |
| `isAddressMatch` / `address_distance_meters` / `kyc_comments` | Address-comparison result fields |
| `video_kyc_pic` / `address_proof_pic` / `aadhaar_pic` / `pan_pic` / `signature_pic` | MinIO object keys for the retained proof images |
| `challenge_snapshots_json` | Metadata for one best retained frame per completed challenge |
| `report_json` | Full Video KYC payload, including the canonical `face_scan_result` JSON from the Face Scan live engine |
| `pdf_report` | MinIO object key for the generated PDF report |

`document_extractions` is reserved for Scan Document and Bulk Scan. Identity Verification and Video KYC do not create rows there.

Results are viewable in **Document Intelligence** under the selected entity, alongside document scan history.
The same verification event also updates `layered_analysis_entries`, which is the dedicated source used by the **Layered Analysis** screen to show extracted fields, deterministic checks, model metrics, and raw evidence for audit review.
Layered Analysis now also carries upload-authenticity evidence for Aadhaar, selfie, Video KYC captures, and saved scan entries so auditors can see the strongest available authenticity check per uploaded asset.

## PDF Reports

Both face match and Video KYC results generate a professional PDF report containing:
- Subject information and entity reference
- Verdict box (PASS/FAIL) with colour coding
- Face match analysis (cosine similarity, confidence score, threshold)
- Liveness detection results (Video KYC only)
- Summary table of all checks
- Disclaimer noting offline AI processing

The rendered PDF files are uploaded to MinIO. The owning DB row stores the current object key in `identity_checks.pdf_report` or `video_kyc_checks.pdf_report`.

## Why Not External APIs?

Many competitive products rely on AWS Rekognition or Azure Face API. BaseTruth uses this local ONNX stack because:
* PII (Personally Identifiable Information) never leaves the server.
* Fraud operators can scan thousands of historical case documents per minute without incurring massive per-call cloud API bills.
* The system elegantly degrades; if the internet goes down, fraud checking continues seamlessly.

## Location in Codebase

| File | Purpose |
| --- | --- |
| `src/basetruth/vision/face.py` | Core face detection and comparison (RetinaFace + ArcFace) |
| `src/basetruth/kyc/session.py` | In-memory session store with TTL and challenge sequencing |
| `src/basetruth/kyc/liveness.py` | Per-frame feature extraction and challenge pass/fail logic |
| `src/basetruth/api.py` | REST + WebSocket endpoints (`POST /kyc/sessions`, `WS /kyc/ws/{id}`, etc.) |
| `src/basetruth/ui/pages/video_kyc.py` | Streamlit operator UI (create session, schedule, in-person verify) |
| `src/basetruth/db.py` | `IdentityCheck` and `VideoKYCCheck` ORM models |
| `src/basetruth/store.py` | identity/video-KYC persistence helpers and `get_entity_identity_checks()` compatibility reads |
| `src/basetruth/reporting/pdf.py` | `render_identity_check_pdf()` |
