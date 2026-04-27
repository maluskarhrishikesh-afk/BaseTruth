# BaseTruth — Screen-by-Screen Functionality Guide

> **Purpose:** This document is the single source of truth for what each screen does, what every button triggers, and what the expected result is. It is used by GitHub Copilot and developers to prevent regressions and avoid reopening the same bugs.
>
> **Update policy:** Every time a screen is changed, this document must be updated in the same commit.

---

## Navigation (Sidebar)

| Label | Icon | Page Key | Page Title |
|---|---|---|---|
| Dashboard | 🏠 | `dashboard` | Dashboard |
| Identity Verification | 🧑‍💻 | `identity` | Identity Verification |
| Video KYC | 🎥 | `video_kyc` | Video KYC |
| Scan Document | 🔍 | `scan` | Scan Document |
| Forensic Scan | 🧪 | `forensic_scan` | Forensic Scan |
| Bulk Scan | 📦 | `bulk` | Bulk Scan |
| Review Scans | 🔬 | `scans` | Review Scans |
| Document Intelligence | 🧠 | `document_intelligence` | Document Intelligence |
| Review Reports | 📋 | `review_reports` | Review Reports |
| Reports | 📊 | `reports` | Reports |
| Log Analyzer | 🪵 | `logs` | Log Analyzer |
| Database Viewer | 🗄️ | `database` | Database Viewer |
| ML Training Pipeline | 🧠 | `ml_training` | ML Training Pipeline |
| Swagger | 📘 | `swagger` | Swagger |
| BaseTruth AI Copilot | 💬 | `gemma_chat` | BaseTruth AI Copilot |

**Rules:**
- The nav label, icon, and page title must always match.
- Icons are set in the `_PAGES` dict in `app.py` and in `_page_title(icon, name)` calls in each page file.
- Navigation is session-state-driven (`st.session_state["page"]`), not Streamlit's built-in routing.
- The app uses a left sidebar with one button per page.

---

## 🏠 Dashboard

**Purpose:** Shows a live summary of all verification activity in the system.

| Element | Action | Expected Result |
|---|---|---|
| Metrics row | Auto-renders on page load | Shows entities, scanned documents, pending review count, high-risk count, auto-approved count, and average score from PostgreSQL |
| "Recent Scans" table | Auto-renders | Lists the 20 most recent scans with entity name, document type, risk level, verdict, and date |
| Shortcut buttons (Document Intelligence, Reports, Review Scans, etc.) | Click | Navigates to the corresponding screen |
| If DB is offline | Page load | Shows "Database is offline" warning; no metrics or tables shown |

---

## 🧑‍💻 Identity Verification

**Purpose:** Run a clean operator-facing identity flow using Aadhaar, PAN, and selfie inputs, and save the verification result to the database.

### Step 1 & 2 — Upload or Capture Documents

**Parallel processing:** When the user uploads documents in all three slots simultaneously, the page fires all analysis pipelines (document-type check, QR decode, PAN OCR, signature crop) concurrently in a thread pool. A single "Analysing documents…" spinner is shown; the total wait equals the slowest pipeline, not the sum of all three.

**Document-type validation:** Each slot only accepts the matching document type. If the wrong document is uploaded (e.g. Aadhaar card in the PAN slot) and Gemma4 is confident about the mismatch, a red error banner is shown and the extraction for that slot is skipped. If Ollama is unavailable or Gemma4 is uncertain, the upload is always allowed through so legitimate documents are never blocked.

| Element | Action | Expected Result |
|---|---|---|
| Upload Aadhaar card | Drag-and-drop or file picker | Doc-type check runs first; if passed, QR decoded automatically; success message shows Name, DOB/YOB, Gender, District, and "Aadhaar looks good"; uploading a PAN or other document here shows a red error banner |
| Upload PAN card | Drag-and-drop or file picker | Doc-type check runs first; if passed, a **single** Gemma4 call extracts PAN number, full name, father's name, and DOB **and** returns the signature bounding-box in one response; success message shows PAN, entity type, full name, "Care of (typically Father or Husband's name): S/O:" label with father's name, DOB; signature strip is cropped from the card using the precomputed box and shown as a small preview at ≤ 300 px wide; uploading Aadhaar or other document here shows a red error banner |
| Upload Selfie | Drag-and-drop or file picker | Doc-type check runs first; if passed, selfie stored for face-match; face detection preview shown; uploading an ID document here shows a red error banner |
| Upload all three at once | Drop files into all three slots | All three pipelines run in parallel in background threads; all results appear after a single combined spinner |
| "Capture with Camera" tab | Switch tab | Camera opens per-document; shutter button takes the photo; doc-type check runs immediately after capture and shows error if wrong document |

### Step 3 — Applicant Details

| Element | Action | Expected Result |
|---|---|---|
| Text fields (First name, Last name, PAN, Aadhaar, Email, Phone) | Type or auto-filled | Used to create/find the entity in the database |
| "Link to existing entity" expander | Open + search | Shows matching entities from DB; selecting one links the result to that record |

### Step 4 — Run Identity Verification

| Element | Action | Expected Result |
|---|---|---|
| "Run Identity Verification 🔍" button | Click | Runs deterministic checks for first-name/last-name match, DOB match, PAN format, and photo match; then shows annotated images, confidence score, and MATCH/MISMATCH verdict |
| "💾 Save to Database" button | Click after a successful run | Saves the current face-match result to `identity_checks`; saves Aadhaar QR fields and PAN extracted fields as separate rows in `document_extractions` with their uploaded `file_name`; uploads the Aadhaar image + selfie + PAN image + PAN signature (if successfully cropped) to MinIO under the entity reference; stores the MinIO key of the signature image in the `pan_card` `document_extractions` row under `pan_signature_minio_key`; PDF report download becomes available |
| | If save fails (DB error) | Red error message: "Result could not be saved to the database. Check the Logs screen." |
| | If DB is offline | Warning: "Database is offline — connect PostgreSQL to save results." |
| Previous checks table | Auto-renders after save / for linked entity | Shows all previous face-match records for the linked entity |

**Stored evidence captured on save:**
- Aadhaar QR extraction payload
- PAN extraction payload, including Gemma4/OCR extraction source
- Two `document_extractions` rows when both documents are available: one `aadhaar` row and one `pan_card` row, each keyed by the uploaded `file_name`
- `pan_card` row includes `pan_signature_minio_key` pointing to the cropped signature image in MinIO (when signature extraction succeeds)
- Exact first-name/last-name comparison result
- DOB comparison result
- PAN format and entity-type interpretation
- Photo-match result and similarity metrics
- PAN layered analysis limited to meaningful validation layers
- Aadhaar upload authenticity checks
- Selfie upload authenticity checks

**Upsert rule:**
- Saving Identity Verification again for the same entity updates the current `face_match` record and replaces the current `face_match_report.pdf` object instead of creating a second current record.

**Important rules that must NOT be broken:**
- `_draw_face()` in `vision/face.py` must always be defined before it is called from `compare_faces()`.
- `save_identity_check()` must always show an error (not silent failure) when it returns `None`.
- `init_db()` must be retried on each app load until it succeeds (not just first attempt).
- The Identity Verification page must stay clean; heavy explainability content is accessible via the Review Scans screen, not inline here.

---

## 🎥 Video KYC

**Purpose:** Create a remote identity verification session where the customer performs live liveness challenges on their own device.

### Tab 1 — Start KYC Session

| Element | Action | Expected Result |
|---|---|---|
| Entity selector | Search/select | Links KYC session to an entity record |
| "Upload ID Document" uploader | Upload | Extracts reference face embedding and stores in session state; shows "ID subject successfully extracted" |
| "Create KYC Session" button | Click | Calls `POST /kyc/sessions`; shows shareable customer URL |
| Customer URL | Copy/share | Customer opens URL on their phone; runs blink, turn left/right, nod challenges |
| "Refresh" / poll | Click | Calls `GET /kyc/sessions/{id}`; shows live session status + liveness results |
| "💾 Save to Database" button | Click after a completed KYC result | Saves the completed Video KYC result to `identity_checks`; uploads the uploaded reference ID document to MinIO; PDF report download appears |

### Tab 2 — Schedule Appointment

| Element | Action | Expected Result |
|---|---|---|
| Date picker, time picker, duration | Select | Sets appointment time |
| "Generate Calendar Invite" button | Click | Creates `.ics` file download; shows info message with 📧 emoji (not a string like "info") |
| Download `.ics` | Click | Gets the calendar invite file |

**Important rules:**
- Page title must be `_page_title("🎥", "Video KYC")` — only one 🎥, no duplication.
- `st.info(..., icon=...)` must use a real unicode emoji character, not a shortcode string like `"info"`.

### Tab 3 — In-Person Verify

| Element | Action | Expected Result |
|---|---|---|
| Camera capture + liveness UI | Real-time | Runs blink, turn, and nod challenges locally |
| Face-match section | After liveness passes | Compares live face to uploaded reference ID |
| "💾 Save to Database" button | Click after result is shown | Saves the in-person Video KYC result and uploads both the reference ID and captured live image to MinIO |

### Liveness Challenges

| Challenge | Detector | Pass Condition |
|---|---|---|
| `blink` | MediaPipe EAR (always, even with InsightFace active) | EAR drops below 0.15 then recovers above 0.18 |
| `turn_left` | Nose x-position relative to bbox | `nose_rel_x > 0.62` |
| `turn_right` | Nose x-position relative to bbox | `nose_rel_x < 0.38` |
| `nod` | Nose-to-eye pitch range | Range > 0.28 over ≥ 6 frames |

---

## 🔍 Scan Document

**Purpose:** Instantly classify and extract structured fields from a single document. This screen does not run forensics and does not save to the database.

| Element | Action | Expected Result |
|---|---|---|
| File uploader | Upload PDF or image | Document is processed immediately for extraction |
| AI extraction pipeline | Auto-runs | Gemma classifies + extracts in one pass (`doc_type="generic"`) |
| Classification banner | Auto-renders | Shows document type, scan mode (Scanned/Image-based or Digital/Structured), and file name |
| JSON result panel | Auto-renders | Shows response JSON including `document_type`, `is_image_based`, and `extracted_fields` |
| "⬇ Download extracted data as JSON" | Click | Downloads the same response payload shown in the panel |
| "🔄 Reset" | Click | Clears current result and upload state |

**Important rules:**
- Confidence is not shown in the screen banner.
- The user-visible JSON output must include `document_type`.
- No write calls are allowed from this screen (no save to PostgreSQL/MinIO).

---

## 🧪 Forensic Scan

**Purpose:** Run forensic tamper analysis on one uploaded document and show verdict + evidence + visual intelligence from Gemma4.

| Element | Action | Expected Result |
|---|---|---|
| File uploader | Upload PDF or image | Forensic pipeline starts immediately |
| Forensic routing | Auto-runs | Image/scanned files use image forensics; structured PDFs use PDF forensics |
| Verdict banner | Auto-renders | Shows forensic verdict (ORIGINAL / ORIGINAL-DERIVED / TAMPERED / TAMPERED-DERIVED / UNCERTAIN / LIKELY TAMPERED), score, type, scan mode, scoring source badge (ML or heuristic), and file name |
| Honest Review card | Auto-renders | Plain-English LLM explanation of the forensic findings written for a non-technical reviewer |
| Feature Contributions chart | Auto-renders (ML only) | Top-10 SHAP features showing which signals drove the score |
| Visual Intelligence card | Auto-renders | Gemma4 "Logical Detective" report: one card per visual fraud clue, each with area, observation, suspicion level (HIGH/MEDIUM/LOW), and plain-English reason; if Ollama is offline an info banner is shown instead |
| Forensic Result (JSON) | Auto-renders | Shows `filename`, `document_type`, `is_image_based`, verdict, score, explanation, and evidence |
| Layer expanders | Expand | Shows layer-level forensic outputs and full layered-analysis JSON |
| "⬇ Download forensic result as JSON" | Click | Downloads user-facing forensic response JSON |

**Visual Intelligence — Gemma4 Logical Detective:**
- A single combined Gemma4 call identifies the document type AND scans the document image for visual fraud clues — font inconsistencies, cut-and-paste halos, colour patches, mis-aligned fields, stamp/seal anomalies, signature irregularities, date/number formatting breaks, and background texture changes.
- Combining both tasks into one call avoids a second image-upload round-trip.
- The detective report is displayed as colour-coded finding cards (🔴 HIGH, 🟠 MEDIUM, 🟡 LOW) above the JSON panel, making it easy for non-technical reviewers to know exactly what to inspect.
- If Gemma4 finds nothing suspicious it shows "✅ No visual fraud clues detected."
- If Ollama is offline the card shows a soft info message; the forensic verdict and score are unaffected.

**Important rules:**
- This screen is non-persistent (no DB/MinIO writes).
- It reuses shared forensic utility logic used by API/UI flows.
- The verdict banner must show whether the final score came from the ML model or the heuristic path.
- ELA heat maps, Composite views, Annotated overlays, Region Details tables, and Zoom panels are intentionally removed — they produced misleading results on lossless formats (PNG/BMP/TIFF) and required forensic training to interpret. Gemma4 visual clues replace them with plain-language findings any reviewer can act on.

---

## 📦 Bulk Scan

**Purpose:** Scan many documents at once from a folder or a file list. Captures two things per document: (1) tamper assessment — truth score and risk level; (2) rich document analysis — document type, extracted fields, fraud signals, and authenticity verdict.

> **Extraction Architecture:**
> - **Gemma4 (free-form batch classification):** At the start of each bulk scan, Gemma4 receives all document images in a **single LLM call**. It identifies each document type by observing visual content, layout, headings, and structure — **without a fixed list to pick from**, so it can classify any document type correctly. It also determines whether each file is image-based or text-based.
> - **Per-document worker processes after classification:** Once Gemma4 returns the classification list, each document's forensics and field extraction run in a separate worker process. This keeps heavy OCR/CV state isolated per file and lets the batch finish close to the slowest few documents instead of the full serial sum.
> - **OCR + layout-family extraction (per-document field extraction):** After forensics, `src/basetruth/integrations/document_extract.py` runs PaddleOCR, reconstructs the document in top-to-bottom and left-to-right order, classifies the document layout family, and uses deterministic parsers for layouts that are table-heavy and stable. Maharashtra HSC transposed marksheets and BE/B.Tech `MAX / MIN / OBT` marksheets are now read this way. For marksheets, PaddleOCR is mandatory for the OCR pass and the raw OCR output is written to `artifacts/<source_stem>/<source_stem>_ocr_scan.md` with one detected row per line plus approximate horizontal spacing so operators can inspect the sheet structure before any LLM interpretation. The active marksheet prompt is generic across all marksheet families: it anchors on reliable fields first, rebuilds rows and columns from OCR structure, and treats printed-total mismatches as generic quality checks rather than SSC/HSC-specific rules. When a deterministic parser successfully rebuilds the marks table, that parsed structure is passed into Gemma4 as a strong hint, but the LLM call still runs so the final result is always image-verified by the model. For structured PDFs (payslips, offer letters, Form 16, etc.), PaddleOCR is skipped but the page image is ALWAYS sent to the LLM alongside the embedded text — the text gives exact numbers, the image gives visual layout context. Skipped for photographs, signatures, cancelled cheques. When Ollama is offline, deterministic extraction still runs where possible; otherwise a forensics-summary fallback row (document type, source name, forensic verdict, forgery score) is saved to `document_extractions` so the table is never empty after a save.
> - **Extracted field schemas:** Field extraction prompts (`document_extract_prompts.md`) now cover richer schemas per document type. Payslips extract `pay_period_month`, `pay_period_year`, `working_days`, `paid_days`, `leave_days_taken`, `esic_number`, `bank_name`, `ctc_per_annum`, `employer_pf_contribution`, and a full `allowances` + `deductions` dict (conveyance, medical, LTA, food allowance, variable pay, bonus, arrears, ESIC, loan/advance recovery). Form 16 extracts both Part A and Part B fields including standard deduction, HRA exemption, Chapter VI-A deductions (80C/80D/80E/80G), net taxable income, surcharge, education cess, and total TDS. Offer Letters extract `employment_type`, `grade_or_band`, `notice_period`, `bond_period`, `reporting_manager`, and a detailed `ctc_breakdown` dict. Experience Letters (formerly "Employment Letter") extract `total_experience`, `last_drawn_salary`, and `performance_note`. Relieving Letters are now a distinct document type with `last_working_date`, `clearance_confirmed`, `assets_returned`, and `dues_cleared` fields. Bank Statements now include `total_credits`, `total_debits`, `average_monthly_balance`, `micr_code`, and `account_type`.
> - **Entity auto-creation from Gemma4 extraction:** When a bulk scan is saved without an explicit entity link, Gemma4-extracted name fields (e.g. `employee_name`, `candidate_name`) are used to auto-create or match an entity record. PAN numbers and email addresses extracted by Gemma4 are also used for deduplication.
> - **LiteParse (text-based PDFs / Word / Excel):** For digitally-created structured documents, LiteParse (Node.js) extracts named fields, tables, and metadata accurately and feeds the tamper pipeline.
> - **PaddleOCR row reconstruction (image-based files):** For scanned/photographed documents (JPEG, PNG, TIFF, scanned PDF), PaddleOCR extracts row-aware text before field extraction. Marksheets use this OCR text to classify the layout family and route to a deterministic parser when the table is stable. Each OCR row is written on a separate line in the markdown artifact and keeps approximate column spacing so later review can see the original sheet structure directly.
> - **Document Intelligence data:** When no per-doc Gemma4 analysis exists, a synthetic analysis is automatically produced from LiteParse key_fields + tamper_assessment signals so Document Intelligence always shows rich data.
> - **Target accuracy:** ≥ 95% document type classification accuracy by letting Gemma4 use its own knowledge rather than constraining it to a fixed list.

> **Forensic Pipeline (per file type):**
> - **Image files (.jpg, .png, .tiff, …) and scanned PDFs:** 11-layer *image* forensic engine (`image_forensics_detect.py`) — ELA, metadata, noise, DCT, clone detection, colour anomaly, edge analysis, saturation, font consistency (including baseline alignment jitter), AI-artefact detection, file entropy.
> - **Digitally-created PDFs (payslip, offer letter, bank statement, form16, etc.):** 11-layer *PDF* forensic engine (`pdf_forensics_detect.py`) — incremental-update detection, metadata fingerprinting, font consistency, hidden-text & shadow-attack detection, suspicious-object detection (JavaScript/XFA/OpenAction), content-structure analysis, digital-signature integrity, page-render ELA, embedded-image noise, file entropy, and object/xref integrity. Each forensic layer shows a plain-English explanation of what was found and why it matters.
> - **Routing logic:** Gemma4 classification determines which pipeline runs. If Gemma4 is unavailable or the confidence < 0.5 for structured-PDF classification, the image pipeline is used as a safe fallback.

| Element | Action | Expected Result |
|---|---|---|
| File uploader or folder path | Upload files or enter path | Files queued for scanning; any results from a previous scan are cleared immediately |
| Entity link widget | Fill in PAN / email / phone | Links every document in the batch to the same applicant profile |
| "Run bulk scan →" button | Click | Shows live status: (1) "🧠 Classifying N document(s) with Gemma4…"; (2) "⚙️ Processing N document(s) in parallel worker process(es)…"; (3) completion updates like "✅ Completed X/N: filename" with a progress bar |
| Unsaved-changes navigation guard | Click another page before saving | Shows a modal asking the operator to continue editing or discard the current batch; discard clears uploaded files, results, and the temporary batch workspace |
| Document Results section | After scan | Expandable card per file showing: (1) forensic verdict and 11-layer breakdown; (2) field extraction status — either a "📋 Extracted Fields (Gemma4)" sub-expander with the extracted data, or an info/warning message explaining why extraction was skipped/failed (Ollama offline, exception, all-null result). For marksheets, the card also shows the path to the raw OCR markdown artifact written during extraction. |
| "💾 Save to Database" button | Click after batch completes | Saves each forensic result to `scans`; upserts extracted fields to `document_extractions` using `(entity_id, file_name)`; uploads source documents to MinIO; sets scan `approved` to NULL (pending review); shows success or visible failure |

**Important rules:**
- Uploading a new set of files immediately clears all previous scan results from the UI. The previously saved database records are never affected.
- Starting a new scan with "Run bulk scan →" also immediately clears any previous results.
- No summary table is shown. The Document Results section (expandable cards) is the only result view.
- Bulk Scan does **not** auto-save to PostgreSQL during the scan run; results remain review-only until the operator clicks **Save to Database**.
- Saving a batch must always show a clear success, warning, or error outcome.
- If Ollama is unavailable during classification or extraction, both fall back gracefully — no hard failure. When Ollama is offline, `document_extractions` still receives a forensics-summary stub row (document type, source name, forensic verdict, forgery score, `_extraction_unavailable: true`) so bulk saves always produce a record per document. The UI card for that document shows an info message: "Field extraction skipped — Gemma4 (Ollama) is not running."
- The `_has_gemma4_data` logic in `save_scan_to_db` uses `bool(_bulk_ext)` (non-empty AND no error/unavailable) — never a key-count heuristic — so that even Gemma4 results with mostly-null fields are correctly identified as real extractions and saved as-is.
- `_extraction_unavailable` in the `document_extractions` fallback row is **always `true`** — it is never `false` in the fallback path. A `false` value would be a bug indicator (empty dict arrived from bulk.py's silent exception handler).
- All saved scans start with `approved = NULL` (pending). They only appear in Document Intelligence after being approved on the **🔬 Review Scans** screen.

**Fraud signals in Document Intelligence**: Only tamper signals that **failed** the check (`passed == False`) are shown as fraud signals. Passed checks (no issue found) are suppressed. Signal fields map from `models.Signal`: `name` → type, `summary` → description.

---

## 🔬 Review Scans

**Purpose:** Review saved scans in two approval levels before they are treated as fully approved.

**Workflow position:** After **Bulk Scan → Save to Database**, scans land here first. A first reviewer decides, then a second reviewer signs off if needed.

| Element | Action | Expected Result |
|---|---|---|
| "⏳ Pending" tab | Auto-renders | Lists all scans where `approved IS NULL`, ordered newest-first |
| "🔄 Awaiting 2nd Review" tab | Auto-renders | Lists scans that passed 1st-level review and are waiting for 2nd-level review |
| "✅ Fully Approved" tab | Auto-renders | Lists scans approved at both levels |
| "❌ Rejected" tab | Auto-renders | Lists all rejected scans |
| "📋 All" tab | Auto-renders | Lists all scans in all states |
| Scan card | Auto-renders per scan | Shows: source file name, document type, entity ref, entity name, risk level badge, forensic verdict badge, forgery score, and scan timestamp |
| "🔬 Forensic Details" expander (per card) | Click | Shows: forensic verdict, forgery score metric, plain-English summary, evidence list, and all 11 layers with status icon + plain-English explanation each |
| "📋 Raw JSON" sub-expander | Click | Shows full `layered_analysis_json` as formatted JSON |
| 1st-level comment field | Type | Optional comment saved with the 1st-level decision |
| "✅ 1st Approve" button | Click | Marks the scan as passed at 1st level; shows success; reruns page |
| "❌ 1st Reject" button | Click | Marks the scan as rejected at 1st level; shows warning; reruns page |
| 2nd-level comment field | Type | Optional comment saved with the 2nd-level decision |
| "✅ 2nd Approve" button | Click | Fully approves the scan; shows success; reruns page |
| "❌ 2nd Reject" button | Click | Rejects the scan at 2nd level; shows warning; reruns page |
| If DB is offline | Page load | Shows warning; no scan list rendered |

**Forensic verdict colour mapping:**
- ORIGINAL → 🟢 (green — confirmed genuine, no re-save)
- ORIGINAL-DERIVED → 🔵 (blue — save-as copy of genuine; still authentic)
- UNCERTAIN → 🟡 (yellow — heuristic fallback, inconclusive)
- LIKELY TAMPERED → 🟠 (orange — heuristic fallback, suspicious)
- TAMPERED → 🔴 (red — confirmed direct forgery)
- TAMPERED-DERIVED → 🟣 (purple — laundered forgery; save-as of tampered doc)

**Important rules:**
- Every approve/reject action must show a visible outcome (success or error). Silent failures are not allowed.
- The page title must be `_page_title("🔬", "Review Scans")` matching the `_PAGES` entry `"🔬  Review Scans": "scans"`.
- 2nd-level approval buttons only appear after a scan has passed 1st-level review.
- Approved and Rejected scan cards show the reviewer name, timestamp, and comment (read-only).
- After approval, the scan immediately appears in Document Intelligence on the next page load.

---

## 🧠 Document Intelligence

**Purpose:** Per-entity, per-document AI analysis viewer. Shows the forensic analysis results and extracted fields for every scanned document, grouped by applicant. Also lets analysts generate a **Final Verification Report** comparing all documents for one person.

| Element | Action | Expected Result |
|---|---|---|
| Search bar + field selector | Type name, PAN, email, BT-ref | Filters the entity list |
| Applicant selector | Select | Shows compact applicant strip (name, ref, PAN, Aadhaar) |
| "✅ Approved" tab | Auto-renders | Lists only fully approved scans (both 1st and 2nd level `Y`) |
| "📋 All Scans" tab | Auto-renders | Lists all scans regardless of approval state |
| Scan card | Auto-renders per scan | Shows source filename, document type, approval label, forensic verdict, forgery score, and scanned date |
| "🔬 Forensic Details" expander | Click | Shows the full 11-layer forensic breakdown for that document |
| Pending/rejected scan warning | Auto-renders | Warning banner tells analyst how many scans need approval on the Review Scans screen |
| "🎯 Generate Final Report" button | Click | Runs cross-document analysis across all documents for this applicant; saves result to `entity_reports` as `BTR-XXXXXX`; shows success message with report reference |
| Re-generate before approval | Click again | Refreshes the pending report in-place (same BTR-XXXXXX reference, updated payload) |
| Saved Reports section | Auto-renders after generation | Shows each saved BTR-XXXXXX report with a pass/fail row per check, approval trail, collapsible full JSON, and approval status badge |

**Cross-document checks performed by Generate Final Report:**
- **Name** — checks if the person’s name is the same across all documents
- **Address** — checks if addresses across all documents match
- **PAN** — checks if the same PAN number appears on all documents
- **Aadhaar** — checks if the same Aadhaar number appears across all documents
- **Salary** — compares payslip net salary vs offer/increment letter CTC (30% tolerance for deductions and increments)
- **Forensics** — flags any document with a TAMPERED or TAMPERED-DERIVED verdict

**Important rules:**
- **Only fully approved scans appear in the "✅ Approved" tab.** A scan is fully approved when both `first_level_approval = 'Y'` AND `second_level_approval = 'Y'`. All scans are always visible in the "📋 All Scans" tab.
- A warning banner appears when pending or rejected scans exist for the selected entity, directing the analyst to the **🔬 Review Scans** screen.
- Generate Final Report reads ALL scans for the entity (not just approved ones) so the cross-document check reflects the full picture.
- A pending report (not yet approved by anyone) is refreshed in-place when re-generated. Approved or rejected reports are never overwritten — a new BTR-XXXXXX is created instead so the audit trail stays intact.
- The page title must remain `_page_title("🧠", "Document Intelligence")`.

---

## 📋 Review Reports

**Purpose:** Review Final Verification Reports (`BTR-XXXXXX`) in two approval levels.

| Element | Action | Expected Result |
|---|---|---|
| "⏳ Pending" tab | Auto-renders | Lists reports waiting for 1st-level review |
| "🔄 Awaiting 2nd Review" tab | Auto-renders | Lists reports approved at 1st level and waiting for senior review |
| "✅ Fully Approved" tab | Auto-renders | Lists reports approved at both levels |
| "❌ Rejected" tab | Auto-renders | Lists rejected reports |
| "📋 All" tab | Auto-renders | Lists every report |
| Report card | Auto-renders | Shows report ref, entity details, current approval state, and report summary |
| "✅ 1st Approve" / "❌ 1st Reject" | Click | Saves the 1st-level decision and reruns the page |
| "✅ 2nd Approve" / "❌ 2nd Reject" | Click | Saves the 2nd-level decision and reruns the page |
| If DB is offline | Page load | Shows warning; no report list rendered |

**Important rules:**
- Every approve/reject action must show a visible success or error toast. Silent failures are not allowed.
- Second-level approval is blocked until first-level approval is `'Y'`.
- The page title must be `_page_title("📋", "Review Reports")`.

---

## 📊 Reports

**Purpose:** One consolidated report per applicant, plus source-document ZIP export and entity-level Final Verification Report downloads.

| Element | Action | Expected Result |
|---|---|---|
| Applicant search | Type | Filters entities by name, PAN, email, or BT-reference |
| Applicant cards | Auto-renders | One expandable card per entity with counts for Face Match, Video KYC, and Document Scans |
| "📄 Generate / Refresh Consolidated Report" | Click | Builds one consolidated PDF for that entity from all saved activities; deletes the older consolidated PDF in MinIO first, then uploads the new one |
| "⬇ Download Consolidated Report (PDF)" | Click | Downloads the latest consolidated report |
| "⬇ Download Final Layered Report (PDF)" | Click (appears if exists) | Downloads the Layered Analysis final report for this entity |
| "📦 Download All Source Documents (ZIP)" | Click | Bundles only the uploaded source documents for that entity into a ZIP; generated report PDFs are excluded |
| "📑 Final Verification Reports (BTR-XXXXXX)" section | Auto-renders | Lists every saved entity-level cross-document report with overall verdict badge, approval status, and date |
| "⬇ JSON" per report | Click | Downloads the full `report_json` payload for that BTR-XXXXXX report as a pretty-printed JSON file |

**Scope note:**
- The Reports page remains the concise applicant-summary view.
- Detailed explainability and raw stored evidence are available on the Scans and Document Intelligence pages.
- BTR-XXXXXX entity reports are separate from the consolidated audit PDF. The JSON download gives auditors the raw cross-document findings.

---

## 📘 Swagger

**Purpose:** Provide operators and integrators a direct entry point to the live OpenAPI documentation.

| Element | Action | Expected Result |
|---|---|---|
| Swagger UI link | Click | Opens `http://localhost:8000/docs` |
| OpenAPI JSON link | Click | Opens `http://localhost:8000/openapi.json` |
| ReDoc link | Click | Opens `http://localhost:8000/redoc` |
| Available endpoints panel | Expand | Shows key scan endpoints including `POST /api/v1/extract` and `POST /api/v1/forensic-scan` |

**Important rules:**
- Page title must remain `_page_title("📘", "Swagger")`.
- The page is informational only; it does not execute scans itself.

---

## 💬 BaseTruth AI Copilot

**Purpose:** Immersive, data-aware chat with a locally hosted LLM (Gemma4 via Ollama). The chatbot acts as an intelligent assistant that can answer general questions *and* seamlessly query the application's PostgreSQL database and MinIO storage—without exposing the underlying SQL or commands to the user.

**Capabilities:**
- **Text-to-SQL (Invisible UX):** When asked about data (e.g., "How many documents are high risk?"), the LLM generates a SQL query. The system intercepts it, safely runs the query, and feeds the results back to the LLM to summarize. The user only sees the natural language answer.
- **Storage Exploration:** The LLM can list MinIO objects (globally or by entity) using internal commands, allowing users to ask "What files are stored for BT-000001?" and get a clean summary.

| Element | Action | Expected Result |
|---|---|---|
| Suggestion chips | Click | Sends a predefined query (e.g. data queries) seamlessly into the chat |
| Chat input | Type message + send | Sends message; if the LLM generates SQL/MinIO commands, they are executed silently before streaming the final response |
| Invisible Guardrails | Background processing | Only SELECT SQL is allowed. Strictly bounded limits (max 100 rows). 10-second query timeouts. Rolled-back transactions. |

**Response formatting:**
All responses follow a structured ChatGPT-style layout with section headers on their own lines using `## Markdown headings`:
- `## 📊 Executive Summary` — one short paragraph
- `## 📋 Key Findings` — bullet list with each item on its own line
- `## ✅ Recommended Actions` — bullet list with each action on its own line

Section headers are **never** placed inline with their content. Each bullet point starts on a fresh line. This is enforced via the system prompt and also via the hidden query-result injection message used in Phase 2 of the two-pass LLM architecture.

**Important rules:**
- The page must **not** display "Database Connected" indicators or status bars. The experience should feel magical; if the DB is offline, the LLM simply states it doesn't have the information right now.
- The page title must remain `_page_title("💬", "BaseTruth AI Copilot")`.

---

## 🪵 Log Analyzer

**Purpose:** View and filter application log output for debugging.

| Element | Action | Expected Result |
|---|---|---|
| Log viewer | Auto-renders | Shows log entries in a CloudWatch-style dark terminal view, **oldest entry first** (scroll down to see newest) |
| Severity filter | Select | Filters to ERROR, WARNING, INFO, DEBUG, or ALL |
| Module filter | Select | Filters by logger/module name |
| Search messages | Type keyword | Live-filters the visible log entries by message text |
| Quick-filter buttons | Click (🔴 Errors, 🟡 Warnings, etc.) | Sets the severity filter instantly |
| Recent Errors section | Auto-renders at bottom | Shows up to 20 most recent ERROR entries with full traceback in collapsible expanders |
| Sidebar link | Click | Opens Log Analyzer in the same tab (implemented as `st.link_button` — same alignment as all other sidebar buttons) |

**Log ordering:** Entries are shown **oldest first** (top = earliest, bottom = most recent). This matches the natural reading direction of a terminal or `tail -f` output.

**Marksheet OCR logging:**
- When a deterministic marksheet OCR parser runs, INFO logs include one `marksheet_ocr_structure` JSON payload showing the exact OCR-derived table structure used by the parser. The payload includes the detected layout family and the parsed rows or numeric arrays, so operators can inspect marksheet extraction decisions directly in Log Analyzer.

---

## 🗄️ Database Viewer

**Purpose:** Inspect raw database tables, edit rows in development mode, inspect storage buckets, and run destructive resets during testing.

**Important rule:** DB and MinIO availability checks MUST use `_db_available_cached()` / `_minio_available_cached()` (30-second TTL) — **never** call `db_available()` or `minio_available()` directly in any UI render path, as they make live network calls and will cause the UI to freeze on every tab click or widget interaction.

### PostgreSQL Tab

| Element | Action | Expected Result |
|---|---|---|
| Metrics row | Auto-renders (cached) | Shows row counts for Entities, Scans, Document Extractions, Identity Checks, and Entity Reports |
| "🔄 Refresh" button | Click | Clears the page's cached DB/MinIO data queries and reloads the latest counts / object list |
| Table selector | Select | Loads up to 500 rows from the chosen table, including `identity_checks`, `document_extractions`, and `entity_reports` |
| Data table | Auto-renders | Shows table columns in a wide dataframe. Large JSON fields are shortened for readability, and large binary fields like `identity_checks.pdf_report` are excluded from the cached table query so the page stays fast |
| Row selection | Click a row in the dataframe | Selects that row immediately and reruns the page |
| Row inspector | After a row is selected | Shows the full selected row payload below the dataframe so JSON-heavy tables stay readable |
| Row Operations panel | Shown only when `BASETRUTH_ENABLE_DB_VIEWER_CRUD=true` | Lets developers create, edit, duplicate, or delete the selected row |
| Delete confirmation | Type `DELETE` + click confirm | Deletes the selected row only after exact confirmation |

### MinIO Storage Tab

| Element | Action | Expected Result |
|---|---|---|
| Main bucket stats row | Auto-renders (cached) | Shows bucket name, object count, and total size |
| Main bucket object list | Auto-renders | Lists PDF/image objects with key, size, and date |
| Docs bucket stats row | Auto-renders (cached) | Shows the separate docs bucket name, object count, and total size |
| Docs bucket object list | Auto-renders | Lists docs bucket objects with key, size, and date |

### Danger Zone Tab

| Element | Action | Expected Result |
|---|---|---|
| "Empty Database" button | Type RESET + Click | Shows spinner "Truncating all tables…"; runs `TRUNCATE TABLE … CASCADE`; shows "✅ Database reset" on success or error message on failure |
| "Empty MinIO Bucket" button | Type RESET + Click | Shows spinner "Deleting all objects…"; batch-deletes all objects; shows "✅ MinIO bucket cleared" on success or error on failure |

---

## 🧠 ML Training Pipeline

**Purpose:** Build training data, train the image and PDF fraud models, and explain the forensic signals in plain language.

| Element | Action | Expected Result |
|---|---|---|
| Status cards | Auto-renders | Shows whether the Image and PDF models exist, how many samples they use, and when they were last updated |
| "📦 Data Extraction" tab | Open | Shows sample-folder browser, extraction controls, and extraction results |
| Start extraction button | Click | Opens the extraction WebSocket and starts building the training CSVs from the sample folders |
| "⏹ Stop" button | Click during extraction | Stops the current extraction run early and keeps the rows already written |
| Extraction charts | Auto-renders after extraction | Shows folder sizes, verdict distribution, score distribution, and per-folder verdict breakdown |
| "🤖 Model Training" tab | Open | Shows model-training controls and live training output |
| Start training button | Click | Opens the training WebSocket and trains the selected model(s) |
| Training metrics/cards | Auto-renders after training | Shows accuracy, F1, ROC AUC, and sample counts |
| Training charts | Auto-renders after training | Shows feature importance, confusion view, PCA scatter, and a decision-tree view |
| "🔍 Signal Reference" tab | Open | Shows separate Image and PDF signal guides in simple language |
| Signal cards | Auto-renders | Grey out signals that are not present in the currently saved model |

**Important rules:**
- Image and PDF signals are documented separately. They are different feature sets.
- The PDF signal guide must match the actual `PDF_FEATURE_NAMES` used by the model.
- The signal guide must grey out cards by feature name, not by list position.

---

## Global Rules (Apply Everywhere)

1. **DB availability checks** — always use `_db_available_cached()` (30-second TTL). Never call `db_available()` directly in the render path.
2. **MinIO availability checks** — always use `_minio_available_cached()`. Never call `minio_available()` directly.
3. **Streamlit `st.info/warning/error(..., icon=...)`: the `icon` parameter must be a real unicode emoji string (e.g. `"📧"`), not an emoji shortcode like `"info"` or `":email:"`.
4. **Page titles** — use `_page_title(emoji, "Title Text")`. The emoji in the sidebar `_PAGES` dict must match the emoji in the `_page_title` call. Both must match the heading the user sees.
5. **Silent failures** — every database write must show either a success message or a user-visible error. Never swallow exceptions without feedback.
6. **`init_db()`** — must be retried on each app load until it succeeds. Do not set `db_init_done = True` if `init_db()` returned False.
7. **`_draw_face()`** — must be a properly defined function in `vision/face.py` before `compare_faces()` references it.
