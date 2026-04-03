# Architecture

## System Overview

BaseTruth runs a micro-DAG style pipeline where each detector contributes signals to a final truth score.

```mermaid
flowchart TD
  A[Input Document] --> B[PDF Metadata Inspector]
  A --> C[LiteParse Extraction]
  A --> IMG[Image File Branch]
  IMG --> IMGOCR[Direct OCR pytesseract]
  IMG --> ELA[ELA Error Level Analysis]
  IMG --> EXIF[EXIF Metadata Inspector]
  IMG --> NOISE[Noise Consistency Analysis]
  X[Client Datasource Connectors] --> Y[Snapshot Workspace]
  Y --> A
  C --> D[Structured Summary Builder]
  D --> E[Semantic And Arithmetic Checks]
  B --> F[Signature And Producer Checks]
  D --> G[Cross Document Comparator]
  IMGOCR --> D
  ELA --> H[Signal Aggregator]
  EXIF --> H
  NOISE --> H
  E --> H
  F --> H
  G --> H
  H --> I[Truth Score And Verdict]
  I --> J[JSON And Markdown And PDF Reports]
  J --> K[Operator UI]
  J --> DB[(PostgreSQL — Scans + PDF Reports)]
  DB --> API[REST API — Entity and Scan Retrieval]
  API --> AUD[Auditor Download PDF]
```

## Layers

### 1. Ingestion Layer

- accepts PDF files directly
- accepts raw image files (.jpg, .jpeg, .png, .tiff, .bmp, .webp) directly — no PDF wrapper needed
- accepts LiteParse JSON outputs directly
- supports datasource connectors such as folder sync and manifest-driven ingest
- snapshots client documents into a BaseTruth-managed workspace before scanning
- produces deterministic artifact directories for each scan

### 1A. Operator UI Layer

- supports single-file upload and immediate scan
- supports bulk upload and folder-driven scan workflows
- supports datasource registration, sync, and scan operations
- supports report review without requiring analysts to browse the filesystem manually
- supports case-centric review by grouping related verification reports

### 1B. Connector Layer

- supports local folder and manifest-based ingest today
- now supports enterprise pull connectors for S3, Google Drive, and SharePoint
- keeps connectors separate from the forensic engine so ingest can evolve independently
- snapshots remote content into the same BaseTruth evidence workspace as local content

### 2. Parsing Layer

- uses LiteParse when available for structure-preserving extraction
- builds normalized label-value pairs and domain summaries
- for raw image files: uses pytesseract OCR directly (no Poppler required) then feeds the same normalisation pipeline
- is intentionally separate from fraud scoring so parsing can be reused elsewhere

### 3. Metadata Layer

- inspects PDF producer and creator fields
- captures creation and modification timestamps when available
- scans for signature markers such as `/Sig`, `/FT /Sig`, `/ByteRange`, and `/Contents`
- for raw image files: inspects EXIF tags (Make, Model, Software, DateTimeOriginal, etc.) via Pillow and exifread

### 4. Logic Layer (Validation Packs)

The logic layer is organised around industry-specific validation packs housed in
`src/basetruth/analysis/packs/`.  Each pack is a self-contained Python module that
inherits from `BaseValidationPack` and declares its own required fields and
domain rules.  Adding a new industry requires only three steps: create the module,
declare the pack, and register it in `packs/__init__.py` — no changes to any
existing file (Open/Closed Principle).

Registered packs:

| Document Type     | Pack Class                  | Industry                        |
|-------------------|-----------------------------|---------------------------------|
| `payslip`         | `PayrollValidationPack`     | Payroll and HR operations       |
| `bank_statement`  | `BankingValidationPack`     | Banking and lending             |
| `payment_receipt` | `PaymentsValidationPack`    | Payments and fintech            |
| `insurance`       | `InsuranceValidationPack`   | Insurance claims                |
| `healthcare`      | `HealthcareValidationPack`  | Hospitals and healthcare        |
| `invoice`         | `InvoiceValidationPack`     | Commercial and GST invoices     |
| `compliance`      | `ComplianceValidationPack`  | Compliance teams and audit      |
| `mortgage`        | `MortgageValidationPack`    | Home-loan / mortgage bundles    |
| `employment_letter` | `MortgageValidationPack`  | Employment verification letters |
| `form16`          | `MortgageValidationPack`    | TDS certificates (Form 16)      |
| `utility_bill`    | `MortgageValidationPack`    | Utility bills (residency proof) |
| `gift_letter`     | `MortgageValidationPack`    | Gift declaration letters        |
| `property_agreement` | `MortgageValidationPack` | Property sale agreements        |

Each pack:
- validates arithmetic consistency (gross vs net, balance identity, subtotal + tax = total)
- validates required field presence
- validates domain-specific formats (IFSC, UAN, GSTIN, UPI ID, policy numbers)
- validates amount and date plausibility

### 5. Comparison Layer

- compares structured summaries across a document series
- currently optimized for monthly payslip analysis
- designed to expand to invoices, claims, statements, and KYC documents

### 6. Image Forensics Layer (`src/basetruth/analysis/image_forensics.py`)

Activated automatically when the input is a raw image file (.jpg, .png, .tiff, etc.).

| Check | Tool | What it catches |
|---|---|---|
| EXIF suspicious tool detection | Pillow + exifread | Photoshop, GIMP, Canva, AI generators in Software/CreatorTool tags |
| Missing camera EXIF | Pillow | Screenshots and generated images lacking Make/Model metadata |
| Timestamp inconsistency | Pillow EXIF | Backdated capture timestamps |
| Error Level Analysis (ELA) | Pillow + NumPy | Copy-paste, text replacement, region editing |
| Noise consistency CV | OpenCV + NumPy | Local editing leaving mismatched noise patterns |

ELA works by resaving the image at a known JPEG quality (95 %) and measuring per-pixel differences.  Genuine unedited images show uniform error; edited regions re-compress differently and emit abnormally high pixel deltas.

**Risk scoring thresholds:**
- ELA score < 8 → no penalty
- ELA score 8–20 → low penalty (15 pts) — mild artefacts
- ELA score ≥ 20 → high penalty (40 pts) — strong editing signature
- Suspicious tool in EXIF → 45 pts
- Noise CV > 1.5 → 25 pts

### 6.1 Identity Verification Layer (`src/basetruth/vision/face.py`)

A standalone offline deep-learning engine dedicated to verifying the identity of individuals across multiple documents (e.g., Aadhaar card vs. Live Selfie) and running real-time liveness challenges over WebSocket.

**Face detector selection (automatic):**

| Environment | Detector | Notes |
|---|---|---|
| Docker / Python ≤ 3.12 | InsightFace (RetinaFace + ArcFace, ONNX) | Full identity embedding + face match |
| Python 3.13+ (local dev) | **MediaPipe FaceLandmarker** | Liveness-only; face match skipped |

| Component | Purpose |
|---|---|
| MediaPipe FaceLandmarker | Detects 468 facial landmarks and outputs blendshape scores (e.g. `eyeBlinkLeft`). Used as default on Python 3.13+. Model: `your_data/models/face_landmarker.task`. |
| InsightFace RetinaFace (ONNX) | Detects facial boundaries and extracts 5-point alignment landmarks. Available on Linux/Python ≤ 3.12. |
| InsightFace ArcFace (ONNX) | Encodes the aligned face into a 512-dimensional identity vector. Required for face-match scoring. |
| OpenCV (`cv2`) | Handles bounding box tracing, BGR/RGB mapping, and image byte decoding prior to analysis. |

*Detailed workflow, challenge thresholds, and face-match scoring are documented in [Identity Verification](IDENTITY_VERIFICATION.md).*

### 7. Reporting Layer

- emits JSON for machines
- emits Markdown for humans and audit trails
- emits PDF audit reports (FPDF2) for loan officers and non-technical reviewers
- PDF reports are stored as binary blobs in the `scans.pdf_report` PostgreSQL column
- auditors can retrieve any historical PDF via `GET /api/v1/scans/{id}/report.pdf`

### 8. Persistence Layer (PostgreSQL)

| Table | Purpose |
|---|---|
| `entities` | One row per verified person/organisation; searchable by name, PAN, Aadhaar, email, phone |
| `scans` | One row per document scan; stores `report_json` (JSONB) + `pdf_report` (LargeBinary) |
| `cases` | Case-management workflow record linked to an entity |
| `case_notes` | Timestamped analyst notes on a case |

The application degrades gracefully to file-only mode when `DATABASE_URL` is not set.

### 9. REST API Layer (`src/basetruth/api.py`)

Key endpoints for auditor workflows:

| Endpoint | Description |
|---|---|
| `GET /api/v1/entities?q=…` | Search entity registry by name / PAN / Aadhaar |
| `GET /api/v1/entities/{ref}` | Entity detail with all linked scans |
| `GET /api/v1/entities/{ref}/scans` | Full scan history with signals for one entity |
| `GET /api/v1/scans/{id}/report.pdf` | Download the PDF audit report for a specific scan |
| `GET /api/v1/scans/recent` | Most-recent scans across all entities |
| `GET /api/v1/db/stats` | Entity / scan / high-risk counts for dashboards |
| `POST /kyc/sessions` | Create a Video KYC session (returns a shareable URL) |
| `GET /kyc/{session_id}` | Customer-facing Video KYC page (served in browser) |
| `GET /kyc/sessions/{session_id}` | Poll session status from the operator dashboard |
| `WS /kyc/ws/{session_id}` | WebSocket: browser streams frames; server sends liveness results |

## 10. Operator UI — Page Routing

The UI is a **single-entry Streamlit app** at `src/basetruth/ui/app.py`.

Navigation is driven entirely by `st.session_state["page"]` — not by Streamlit's native page routing.  This gives full control over the sidebar and prevents Streamlit from auto-discovering `pages/` files.

```text
app.py  →  main()  →  session_state["page"]
                            │
       ┌────────────────────┼────────────────────────┐
       │                    │                        │
 pages/dashboard.py   pages/identity.py   pages/scan.py  …
```

Streamlit auto-discovers any `.py` file in a `pages/` directory and adds it to the sidebar.  We suppress this two ways:

1. **Config** — `.streamlit/config.toml`: `hideSidebarNav = true` (supported in most Streamlit versions)
2. **CSS fallback** — `_CSS` in `app.py`: `[data-testid="stSidebarNav"] { display: none !important; }`

Both are applied so the nav items are hidden regardless of Streamlit version.

## 11. Identity Verification UI

The Identity Verification page (`pages/identity.py`) accepts documents in two modes, selectable via tabs:

| Tab | How it works |
|---|---|
| **📁 Upload Documents** | Three drag-and-drop uploaders — Aadhaar Card, PAN Card, Selfie. Aadhaar QR is decoded and PAN OCR runs immediately on upload, results shown inline in the same column. |
| **📷 Capture with Camera** | Per-document "Open Camera" buttons. Camera only opens on click. The native shutter button takes the photo. Photos are stored in session state and persist across re-renders. A tips banner guides the user to get a sharp, well-lit capture. |

Camera captures are wrapped in a `_DocumentCapture` class that matches the `UploadedFile` API (`.size`, `.name`, `.getvalue()`) so all downstream processing is source-agnostic.

### Image Quality Pipeline for Camera Captures

Camera images often suffer from glare, shadows, or lower resolution. Both the QR decoder and PAN OCR apply a multi-strategy preprocessing cascade before analysis:

**Aadhaar QR (`_parse_aadhaar_qr`)**

The function tries the following in order, stopping as soon as the QR code decodes:

1. **WeChatQRCode** (OpenCV contrib, deep-learning based) — best for blurry, perspective-distorted, or low-resolution camera captures
2. Classic `QRCodeDetector` with a preprocessing cascade:
   - Original colour → grayscale → denoised → CLAHE → adaptive Gaussian threshold → adaptive mean threshold → Otsu → sharpened
3. WeChatQRCode again on each **2×, 3×, 4× upscale** of the image
4. Classic detector on each upscaled variant

`opencv-contrib-python` (replaces `opencv-python` in `requirements.txt`) provides the WeChatQRCode model.

**PAN Card OCR (`_extract_pan_info`)**

- Image resized to max **2 400 px wide** (raised from 1 200) and upscaled up to **2.5×** (raised from 1.5×) for small camera captures
- Preprocessing variants: plain gray → denoised (`fastNlMeansDenoising`) → Otsu → CLAHE → sharpened → adaptive Gaussian threshold
- Multiple Tesseract PSM modes tried; first one to return a valid PAN format wins

**PDF report** — `render_identity_check_pdf()` embeds the ID document image and selfie as a Photo Evidence section alongside the match verdict and similarity scores.

## 12. Video KYC Workflow

The Video KYC page (`pages/video_kyc.py`) has **three tabs**.

### Why a Dedicated Video SDK?

Third-party platforms (Zoom, Teams, Google Meet) do not expose raw frame data to server-side code. That makes reliable AI liveness checks and face-matching impossible over those channels. BaseTruth instead runs its own **WebSocket-based video layer**:

- Customer opens a URL in their browser — no app, no plugin, no account needed.
- `getUserMedia` captures the camera at ~3 fps and streams JPEG frames through a WebSocket.
- The server runs **RetinaFace** (face detection) + **active liveness challenge analysis** + **ArcFace** (face match) on every frame.
- The result is sent back in real time as JSON.

### Frame Processing Pipeline

```
Browser                          FastAPI server (port 8000)
──────                           ─────────────────────────
getUserMedia → canvas.toBlob     base64 decode → cv2.imdecode
→ base64 → WebSocket JSON  ───→  RetinaFace detect → extract_features()
                                 analyze_challenge() per frame
                            ←──  {type:"status", face_detected, challenge,
                                  challenges_completed, feedback}
[all challenges passed]     ←──  run_face_match() → ArcFace cosine sim
                            ←──  {type:"result", passed, display_score, message}
```

Thread safety: InsightFace's `face_app.get()` is not thread-safe. A `threading.Lock()` (`_kyc_face_lock`) wraps every call inside `_process_kyc_frame`, which itself runs in asyncio's thread-pool executor so the WebSocket handler never blocks the event loop.

### Session Management (`src/basetruth/kyc/session.py`)

- **`KYCSession`** dataclass: holds challenges list, per-challenge frame history, reference embedding (base64 float32 bytes), status, and the last live frame bytes (for PDF evidence).
- **`SessionStore`**: thread-safe in-memory dict with 30-minute TTL.
- Statuses: `waiting → active → completed | failed | expired`.

### Active Liveness Detection (`src/basetruth/kyc/liveness.py`)

Uses InsightFace 5-point keypoints (kps[0]=left_eye, kps[1]=right_eye, kps[2]=nose, kps[3]=left_mouth, kps[4]=right_mouth). All features are normalized by the face bounding-box width.

| Challenge     | Detection method                                       | Threshold              |
|---------------|--------------------------------------------------------|------------------------|
| `turn_left`   | `nose_rel_x = (nose_x − bbox_x1) / bbox_w`           | `> 0.62`               |
| `turn_right`  | Same `nose_rel_x`                                      | `< 0.38`               |
| `nod`         | `pitch = (nose_y − eye_mid_y) / interocular_px` range | range `> 0.28` over 6+ frames |
| `blink`       | Face detection confidence (`det_score`) dip + recovery | baseline `> 0.88`, dip `< 0.84`, recovery `> 0.90` |

By default 2 challenges are selected at random per session (the agent may override in the dashboard).

### API Endpoints

| Method      | Path                          | Description                               |
|-------------|-------------------------------|-------------------------------------------|
| `POST`      | `/kyc/sessions`               | Create session; returns `session_id` + URL |
| `GET`       | `/kyc/{session_id}`           | Serve customer HTML page                  |
| `WebSocket` | `/kyc/ws/{session_id}`        | Frame stream in, status/result JSON out   |
| `GET`       | `/kyc/sessions/{session_id}`  | Agent polls for status and result         |

### Tab 1 — 🔗 Start KYC Session

Agent workflow:
1. Upload reference ID → BaseTruth extracts the ArcFace embedding locally.
2. Enter customer name / entity ref, select challenges (or leave empty for 2 random).
3. Click **Create Secure KYC Session** → `POST /kyc/sessions` via internal Docker DNS.
4. The session URL (`http://<host>:8000/kyc/<id>`) is shown and can be shared with the customer.
5. Streamlit polls `/kyc/sessions/<id}` every 2 s and shows live challenge progress.
6. On completion, the result (pass/fail + match score) is saved to PostgreSQL and a PDF is generated.

### Tab 2 — 📅 Schedule Appointment

The agent generates a `.ics` calendar invite. The **Meeting Link** field is pre-populated with the BaseTruth KYC URL from Tab 1 (if a session was created), eliminating any need for Zoom or Teams. No external API keys required.

### Tab 3 — 📷 In-Person Verify

Static webcam check (`st.camera_input`) for face-to-face on-site KYC. Runs entirely within the Streamlit container; no session API needed. Uses the same RetinaFace + ArcFace pipeline and saves to the same DB + PDF path.



## Why This Shape

This architecture lets BaseTruth scale from a local analyst tool into an enterprise service without replacing the core reasoning model.

The key product decision is to keep client data sources read-only and pull from them into BaseTruth snapshots. That is safer than treating a single mutable shared folder as the system of record.

PDF reports are stored in PostgreSQL alongside the JSON so auditors can retrieve the full explanation for any historical flag without needing filesystem access. This is the foundation for the chain-of-custody export planned in Phase 4.
