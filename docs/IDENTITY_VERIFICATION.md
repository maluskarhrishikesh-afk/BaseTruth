# Identity Verification & Fraud Detection

BaseTruth incorporates a robust, **offline-first** computer vision pipeline specifically engineered for identity fraud detection. This allows the system to verify that the person on an ID document (Aadhaar, PAN, Passport) matches a provided live Selfie, mathematically scoring the likelihood of fraud.

## Core Technology Stack

The pipeline runs locally without making any external API calls, ensuring high privacy, reducing costs, and keeping latency strictly bounded by the edge hardware.

| Component | Responsibility | Purpose in Flow |
| --- | --- | --- |
| **OpenCV** (`cv2`) | Image Processing | Decodes raw byte streams from the UI, manages RGB/BGR color space conversions, heavily resizes images for memory safety, draws visual forensic evidence (bounding boxes), and detects/decodes QR codes on Aadhaar cards. |
| **RetinaFace** | Face Detection | Deep learning model that acts as the "eyes". It aggressively searches the document to locate the face, cropping it securely and mapping 5 key facial landmarks (eyes, nose, mouth) required to properly align the face angle. |
| **ArcFace** | Identity Recognition | Deep learning model that acts as the "brain". It takes the aligned face from RetinaFace and translates it into a 512-dimensional vector (an embedding). |
| **ONNX Runtime** | Inference Engine | Executes both RetinaFace and ArcFace locally on the CPU (or GPU if available) using the `buffalo_l` pre-trained model pack, bypassing heavy dependencies like PyTorch. |
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

### 2. Video KYC (Live Camera)
Designed specifically to prevent impersonation or photo-spoofing fraud.

1. **Reference Extraction**: The user provides the Aadhaar card. The AI extracts the face embedding.
2. **Live Camera Capture**: The system uses Streamlit's native `st.camera_input()` to capture a live photo directly from the browser.
3. **Liveness Detection**: The system calculates the lateral projection angle between the nose and eyes (using the RetinaFace spatial ratio). By asking the user to turn their head `Left` or `Right`, the system validates physical 3D presence rather than a flat, static printed photo.
4. **Identity Matching**: ArcFace verifies the live capture against the Reference ID document.
5. **Persistence**: Results (identity match + liveness) are stored in `identity_checks` and a PDF report is generated.

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
| `src/basetruth/vision/video_kyc.py` | Video KYC processor with liveness detection |
| `src/basetruth/db.py` | `IdentityCheck` ORM model |
| `src/basetruth/store.py` | `save_identity_check()`, `get_entity_identity_checks()` |
| `src/basetruth/reporting/pdf.py` | `render_identity_check_pdf()` |
| `src/basetruth/ui/app.py` | Identity and Video KYC page UI |
