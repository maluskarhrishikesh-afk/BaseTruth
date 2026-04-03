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
| **pytesseract** | OCR | Extracts text from PAN card images to read the PAN number and cardholder name for cross-document verification. |

## Workflow

BaseTruth offers two variants of Identity Verification:

### 1. Multi-Document Identity Verification (Aadhaar + PAN + Selfie)

The primary identity verification flow mandates **three document inputs**:

1. **Aadhaar Card Upload** — The operator uploads a photo of the applicant's Aadhaar card.
   - OpenCV `QRCodeDetector` scans the card for the QR code.
   - If an old-format (pre-2018) QR is found, the XML payload is parsed to extract: full name, DOB/year of birth, gender, district, state.
   - If a new secure QR (2018+) is found, the system notes it is present but cannot reveal the encrypted payload offline.
2. **PAN Card Upload** — The operator uploads a photo of the applicant's PAN card.
   - pytesseract OCR extracts the PAN number and cardholder name.
   - The PAN number is validated against the standard format (`ABCDE1234F`).
   - The 4th character is decoded to identify the entity type (Individual / Company / HUF etc.).
   - The 5th character (surname initial) is cross-checked against the Aadhaar QR name.
3. **Selfie or Camera Capture** — The operator uploads a selfie or triggers the browser camera.
   - If no selfie file is uploaded, `st.camera_input()` opens automatically.
4. **Name Cross-Check** — The full name from the Aadhaar QR is compared (normalised, word-level overlap ≥ 50%) against the name extracted from the PAN card OCR.
5. **Face Match** — ArcFace cosine similarity between the face on the Aadhaar card and the selfie (threshold > 0.40).
6. **Auto-fill Entity Form** — Extracted name, PAN number, and masked Aadhaar UID are pre-populated in the "Applicant Details" form. The operator only needs to enter phone and email manually.
7. **Persistence** — Results are stored in the `identity_checks` database table, linked to the entity. A PDF report is generated.

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
   - Optionally uploads the customer's ID document; the system extracts a face embedding (ArcFace/InsightFace) to use as a reference for later matching.
   - `POST /kyc/sessions` is called on the FastAPI server; a 30-minute session is created.
   - A shareable URL like `http://your-server:8000/kyc/<session_id>` is returned.

2. **Customer opens the link** — The browser loads the self-contained HTML page served directly by the API server.
   - Works on any modern mobile or desktop browser; no install required.
   - Page shows: customer name, challenge count, and a "Start Verification" button.

3. **Live challenges** — Customer clicks "Start Verification":
   - Browser requests camera permission.
   - A WebSocket connection opens to `/kyc/ws/<session_id>`.
   - The browser captures a JPEG frame every ~310 ms and sends it as base64 over the WebSocket.
   - The server runs **MediaPipe FaceLandmarker** (or InsightFace if available) to locate the face in each frame and extracts 5 key landmarks (eyes, nose, mouth corners).
   - A random set of 2–4 **active-liveness challenges** are assigned (configurable):

     | Challenge | What the server looks for |
     |---|---|
     | `blink` | EAR (Eye Aspect Ratio) dips below 0.15 (eyes close), then recovers above 0.18 (eyes open); fallback: InsightFace detection confidence dip |
     | `turn_left` | Nose `x` position (relative to face width) moves right past 0.62 |
     | `turn_right` | Nose `x` position moves left below 0.38 |
     | `nod` | Pitch (nose height relative to eye midpoint) range exceeds 0.28 across recent frames |

   - After each challenge passes, the server advances to the next one and sends a progress update.

4. **Face match (optional)** — Once all liveness challenges pass:
   - If a reference embedding was provided in step 1, **ArcFace cosine similarity** is computed between the live face and the reference embedding.
   - Threshold: similarity > 0.40 → PASS.
   - If no reference was provided, the session completes as a liveness-only check.

5. **Result** — Server sends a `{"type":"result","passed":true/false,...}` message via WebSocket.
   - Browser shows a full-screen PASS ✅ or FAIL ❌ card.
   - Operator dashboard polls `GET /kyc/sessions/{session_id}` for the outcome.

**Face detection strategy:**

| Environment | Face Detector | Liveness | Face Match |
|---|---|---|---|
| Docker (Linux) | InsightFace (RetinaFace) | EAR + det_score fallback | ArcFace cosine similarity |
| Windows Python ≤ 3.12 | InsightFace (RetinaFace) | EAR + det_score fallback | ArcFace cosine similarity |
| Windows Python 3.13+ | **MediaPipe FaceLandmarker** | EAR via blendshapes | Skipped (liveness-only) |

On Python 3.13+, `insightface` cannot be installed (native extension build fails). The server automatically falls back to MediaPipe — all liveness challenges work fully, and the face-match step is skipped with a clear message instead of failing silently.

## Cross-Document Checks

| Check | Input | Method | Fail Condition |
|---|---|---|---|
| Name match | Aadhaar QR ↔ PAN OCR | Word-level Jaccard similarity | Similarity < 50% |
| PAN format | PAN number | Regex `[A-Z]{5}[0-9]{4}[A-Z]` | Non-matching pattern |
| PAN surname initial | PAN[4] ↔ Aadhaar surname | Character comparison | Mismatch |
| PAN entity type | PAN[3] | Lookup table | Unexpected entity type |
| Face match | Aadhaar face ↔ Selfie | ArcFace cosine similarity | Score < 0.40 (< 40% confidence) |

## Data Persistence

All identity verification results are now persisted in the `identity_checks` PostgreSQL table:

| Field | Description |
| --- | --- |
| `check_type` | `face_match` or `video_kyc` |
| `entity_id` | Links to the `entities` table |
| `cosine_similarity` | Raw ArcFace similarity score |
| `is_match` | Boolean match result |
| `liveness_passed` | Liveness check result (Video KYC) |
| `verdict` | Overall PASS/FAIL |
| `report_json` | Full result payload |
| `pdf_report` | Generated PDF audit report |

Results are viewable in the **Records** page under each entity's detail panel, alongside document scan history.

## PDF Reports

Both face match and Video KYC results generate a professional PDF report containing:
- Subject information and entity reference
- Verdict box (PASS/FAIL) with colour coding
- Face match analysis (cosine similarity, confidence score, threshold)
- Liveness detection results (Video KYC only)
- Summary table of all checks
- Disclaimer noting offline AI processing

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
| `src/basetruth/db.py` | `IdentityCheck` ORM model |
| `src/basetruth/store.py` | `save_identity_check()`, `get_entity_identity_checks()` |
| `src/basetruth/reporting/pdf.py` | `render_identity_check_pdf()` |
