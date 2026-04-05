# BaseTruth — Screen-by-Screen Functionality Guide

> **Purpose:** This document is the single source of truth for what each screen does, what every button triggers, and what the expected result is. It is used by GitHub Copilot and developers to prevent regressions and avoid reopening the same bugs.
>
> **Update policy:** Every time a screen is changed, this document must be updated in the same commit.

---

## Navigation (Left Sidebar)

| Sidebar Label | Icon | Page Key | Page Title |
|---|---|---|---|
| Dashboard | 🏠 | `dashboard` | Dashboard |
| Identity Verification | 🧑‍💻 | `identity` | Identity Verification |
| Layered Analysis | 🧾 | `layered_analysis` | Layered Analysis |
| Video KYC | 🎥 | `video_kyc` | Video KYC |
| Scan Document | 🔍 | `scan` | Scan Document |
| Bulk Scan | 📦 | `bulk` | Bulk Scan |
| Cases | 📁 | `cases` | Cases |
| Records | 🗂️ | `records` | Records |
| Reports | 📊 | `reports` | Reports |
| Datasources | 🔗 | `datasources` | Datasources |
| Log Analyzer | 📋 | `logs` | Log Analyzer |
| Database Viewer | 🗄️ | `database` | Database Viewer |
| Settings | ⚙️ | `settings` | Settings |
| Gemma4 Chat | 🤖 | `gemma_chat` | Gemma4 Chat |

**Rules:**
- The sidebar label, icon, and page title must always match.
- Icons are set in the `_PAGES` dict in `app.py` and in `_page_title(icon, name)` calls in each page file.
- Navigation is session-state-driven (`st.session_state["page"]`), not Streamlit's built-in routing.
- The DB connection status pill at the bottom of the sidebar uses `_db_available_cached()` (30-second TTL) to avoid a live `SELECT 1` on every render.

---

## 🏠 Dashboard

**Purpose:** Shows a live summary of all verification activity in the system.

| Element | Action | Expected Result |
|---|---|---|
| Metrics row | Auto-renders on page load | Shows total entities, scans, high-risk count, and open cases from PostgreSQL |
| "Recent Scans" table | Auto-renders | Lists the 20 most recent scans with entity name, document type, risk level, verdict, and date |
| Shortcut buttons (View Records, View Reports, etc.) | Click | Navigates to the corresponding screen |
| If DB is offline | Page load | Shows "Database is offline" warning; no metrics or tables shown |

---

## 🧑‍💻 Identity Verification

**Purpose:** Run a clean operator-facing identity flow using Aadhaar, PAN, and selfie inputs, then save the underlying evidence for later auditor review on the Layered Analysis screen.

### Step 1 & 2 — Upload or Capture Documents

| Element | Action | Expected Result |
|---|---|---|
| Upload Aadhaar card | Drag-and-drop or file picker | QR decoded automatically; success message shows Name, DOB/YOB, Gender, District, and "Aadhaar looks good" |
| Upload PAN card | Drag-and-drop or file picker | Gemma4 extracts PAN number, full name, father's name, and DOB; success message shows PAN, entity type, full name, DOB, and "PAN looks good" |
| Upload Selfie | Drag-and-drop or file picker | Selfie stored for face-match; face detection preview shown |
| "Capture with Camera" tab | Switch tab | Camera opens per-document; shutter button takes the photo |

### Step 3 — Applicant Details

| Element | Action | Expected Result |
|---|---|---|
| Text fields (First name, Last name, PAN, Aadhaar, Email, Phone) | Type or auto-filled | Used to create/find the entity in the database |
| "Link to existing entity" expander | Open + search | Shows matching entities from DB; selecting one links the result to that record |

### Step 4 — Run Identity Verification

| Element | Action | Expected Result |
|---|---|---|
| "Run Identity Verification 🔍" button | Click | Runs deterministic checks for first-name/last-name match, DOB match, PAN format, and photo match; then shows annotated images, confidence score, and MATCH/MISMATCH verdict |
| "💾 Save to Database" button | Click after a successful run | Saves the current face-match result to `identity_checks`; uploads the Aadhaar image + selfie to MinIO under the entity reference; PDF report download becomes available |
| | If save fails (DB error) | Red error message: "Result could not be saved to the database. Check the Logs screen." |
| | If DB is offline | Warning: "Database is offline — connect PostgreSQL to save results." |
| Previous checks table | Auto-renders after save / for linked entity | Shows all previous face-match records for the linked entity |

**Stored evidence captured on save:**
- Aadhaar QR extraction payload
- PAN extraction payload, including Gemma4/OCR extraction source
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
- The Identity Verification page must stay clean; heavy explainability content belongs on the Layered Analysis page, not inline here.

---

## 🧾 Layered Analysis

**Purpose:** Provide a regulator- and auditor-facing explainable-AI screen showing the stored evidence trail behind Identity Verification, Video KYC, Scan Document, and Bulk Scan decisions.

| Element | Action | Expected Result |
|---|---|---|
| Applicant filter | Type | Filters saved entities by name, PAN, email, or BT-reference |
| Applicant expanders | Open | Shows one audit view per entity with section counts for Identity Verification, Video KYC, Scan Document, and Bulk Scan evidence |
| Identity Verification section | Expand | Reads only from `layered_analysis_entries`; shows four separate section views for Aadhaar, PAN Card, Photo Upload, and Run Verification, including saved upload-authenticity checks where applicable |
| Video KYC section | Expand | Reads only from `layered_analysis_entries`; shows liveness outcome, match metrics, upload-authenticity checks for the reference document and live capture, and raw stored payload |
| Scan Document / Bulk Scan sections | Expand | Read only from `layered_analysis_entries`; show key extracted fields, truth score, risk level, upload-authenticity checks, top forensic signals, and raw stored payload |
| "📄 Generate Final Report" | Click | Builds a detailed layered-analysis PDF for the selected entity from `layered_analysis_entries`; uploads it to MinIO and enables download |
| "⬇ Download Final Report (PDF)" | Click after generation | Downloads the latest layered-analysis PDF for that entity |

**Important rules:**
- This page is intentionally detailed and audit-heavy; it is the place for explainability, not the primary workflow pages.
- The page title must remain `_page_title("🧾", "Layered Analysis")`.
- Report generation must show a visible success, warning, or error outcome.
- Final report generation is locked once the current evidence set has been reported. The button is re-enabled only after fresh layered-analysis data is saved for that entity.
- Layered Analysis entries must be upserted by `(entity_id, screen_name, section_name)` so each entity always has the latest evidence snapshot per section.
- The Database Viewer must expose `identity_checks` and `layered_analysis_entries` in the table browser.

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

**Purpose:** Scan a single document for fraud signals.

| Element | Action | Expected Result |
|---|---|---|
| File uploader | Upload PDF or image | Document stored temporarily; scan starts |
| Document type selector | Select | Activates the relevant validation pack |
| "Start Scan" button | Click | Runs all fraud detectors; shows Truth Score, risk level, and detailed signal report |
| Entity link widget | Search/select | Links scan result to a person record |
| "💾 Save to Database" button | Click after scan completes | Saves the scan to `scans`; uploads the original source document to MinIO; shows success or visible failure |

---

## 📦 Bulk Scan

**Purpose:** Scan many documents at once from a folder or a file list.

| Element | Action | Expected Result |
|---|---|---|
| Folder path input or file uploader | Enter path or upload zip | Queues all documents |
| "Start Bulk Scan" button | Click | Scans each file in sequence; shows progress bar + per-file results |
| "💾 Save Batch to Database" button | Click after the scan batch completes | Saves each report to `scans`; uploads the original source documents to MinIO under the resolved entity reference; shows success or visible failures |
| Summary table | After completion | Shows all results with pass/fail verdict and download links |

---

## 📁 Cases

**Purpose:** Case management — group related documents/scans under a single investigation case.

| Element | Action | Expected Result |
|---|---|---|
| "New Case" button | Click | Opens form to create a case with title, description, and linked entity |
| Case list | Auto-renders | Shows all open/closed cases with status badge |
| Case row | Click | Opens case detail with linked scans, notes, and timeline |
| "Add Note" button | Click | Adds timestamped analyst note to the case |
| Status dropdown | Select | Updates case status (New → Triage → Investigating → Closed) |

---

## 🗂️ Records

**Purpose:** Browse and search the entity registry.

| Element | Action | Expected Result |
|---|---|---|
| Search bar | Type | Filters entities by name, PAN, Aadhaar, email, or BT-ref |
| Entity row | Click | Opens entity detail with all linked scans, identity checks, and cases |
| "Download PDF" per scan | Click | Downloads the stored PDF report for that scan |

---

## 📊 Reports

**Purpose:** One consolidated report per applicant, plus audit download of uploaded source documents.

| Element | Action | Expected Result |
|---|---|---|
| Applicant search | Type | Filters entities by name, PAN, email, or BT-reference |
| Applicant cards | Auto-renders | One expandable card per entity with counts for Face Match, Video KYC, and Document Scans |
| "📄 Generate / Refresh Consolidated Report" | Click | Builds one consolidated PDF for that entity from all saved activities; deletes the older consolidated PDF in MinIO first, then uploads the new one |
| "⬇ Download Consolidated Report (PDF)" | Click | Downloads the latest consolidated report |
| "📦 Download All Source Documents (ZIP)" | Click | Bundles only the uploaded source documents for that entity into a ZIP; generated report PDFs are excluded |

**Scope note:**
- The Reports page remains the concise applicant-summary view.
- Detailed explainability and raw stored evidence live on the Layered Analysis page.
- If a layered-analysis final report exists in MinIO for the entity, it must also be downloadable from the Reports page.

---

## 🔗 Datasources

**Purpose:** Register and sync external document sources (local folders, S3, SharePoint, Google Drive).

| Element | Action | Expected Result |
|---|---|---|
| "Add Datasource" form | Fill + Submit | Registers new connector; saved to `config/datasources.json` |
| "Sync" button per source | Click | Pulls documents from the source into the BaseTruth workspace |
| Source health indicator | Auto-renders | Shows "Connected" or "Error" for each registered source |

---

## 🤖 Gemma4 Chat

**Purpose:** Chat with a locally hosted Gemma model through Ollama from within the BaseTruth UI.

| Element | Action | Expected Result |
|---|---|---|
| Endpoint auto-detection | Page load | Tries the configured Ollama URL first, then the local fallback addresses appropriate for local or Docker execution |
| Model selector | Select | Switches the chat target to another Ollama model visible from the current runtime |
| Chat input | Type message + send | Sends the conversation to Ollama and streams the assistant response into the chat window |
| Clear conversation | Click | Clears the current in-memory chat history from Streamlit session state |
| Retry connection | Click when disconnected | Re-runs endpoint discovery and reconnects without leaving the page |

**Important rules:**
- The page title must remain `_page_title("🤖", "Gemma4 Chat")`.
- When the UI runs in Docker, the page must not assume `localhost:11434`; it must use a host-reachable Ollama endpoint.

---

## 📋 Log Analyzer

**Purpose:** View and filter application log output for debugging.

| Element | Action | Expected Result |
|---|---|---|
| Log viewer | Auto-renders | Shows last N lines of the application log file |
| Severity filter | Select | Filters to ERROR, WARNING, INFO, DEBUG, or ALL |
| "Refresh" button | Click | Reloads the log file |

---

## 🗄️ Database Viewer

**Purpose:** Inspect raw database tables and perform destructive resets during testing.

**Important rule:** DB and MinIO availability checks MUST use `_db_available_cached()` / `_minio_available_cached()` (30-second TTL) — **never** call `db_available()` or `minio_available()` directly in any UI render path, as they make live network calls and will cause the UI to freeze on every tab click or widget interaction.

### PostgreSQL Tab

| Element | Action | Expected Result |
|---|---|---|
| Metrics row | Auto-renders (cached) | Shows row counts for Entities, Scans, Document Extractions, Identity Checks, Layered Analysis Entries, Cases, and Case Notes |
| "🔄 Refresh" button | Click | Clears the page's cached DB/MinIO data queries and reloads the latest counts / object list |
| Table selector | Select | Loads up to 500 rows from the chosen table, including `identity_checks` and `layered_analysis_entries` |
| Data table | Auto-renders | Shows all table columns in a wide dataframe, including JSON and binary fields rendered into readable summaries |
| Row inspector | Select a row | Shows the full selected row payload below the dataframe so JSON-heavy tables such as `layered_analysis_entries` remain readable |

### MinIO Storage Tab

| Element | Action | Expected Result |
|---|---|---|
| Stats row | Auto-renders (cached) | Shows bucket name, object count, total size |
| Object list | Auto-renders | Lists PDF/image objects with key, size, and date |

### Danger Zone Tab

| Element | Action | Expected Result |
|---|---|---|
| "Empty Database" button | Type RESET + Click | Shows spinner "Truncating all tables…"; runs `TRUNCATE TABLE … CASCADE`; shows "✅ Database reset" on success or error message on failure |
| "Empty MinIO Bucket" button | Type RESET + Click | Shows spinner "Deleting all objects…"; batch-deletes all objects; shows "✅ MinIO bucket cleared" on success or error on failure |

---

## ⚙️ Settings

**Purpose:** Configure BaseTruth runtime settings.

| Element | Action | Expected Result |
|---|---|---|
| Artifact root path | Edit + Save | Changes the local folder where reports are stored |
| Product info section | Auto-renders | Shows version, Python version, and API endpoint |
| Quick-start commands | Auto-renders | Copy-paste terminal commands for starting/stopping services |

---

## Global Rules (Apply Everywhere)

1. **DB availability checks** — always use `_db_available_cached()` (30-second TTL). Never call `db_available()` directly in the render path.
2. **MinIO availability checks** — always use `_minio_available_cached()`. Never call `minio_available()` directly.
3. **Streamlit `st.info/warning/error(..., icon=...)`: the `icon` parameter must be a real unicode emoji string (e.g. `"📧"`), not an emoji shortcode like `"info"` or `":email:"`.
4. **Page titles** — use `_page_title(emoji, "Title Text")`. The emoji in the sidebar `_PAGES` dict must match the emoji in the `_page_title` call. Both must match the heading the user sees.
5. **Silent failures** — every database write must show either a success message or a user-visible error. Never swallow exceptions without feedback.
6. **`init_db()`** — must be retried on each app load until it succeeds. Do not set `db_init_done = True` if `init_db()` returned False.
7. **`_draw_face()`** — must be a properly defined function in `vision/face.py` before `compare_faces()` references it.
