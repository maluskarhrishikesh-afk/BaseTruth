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

**Purpose:** Verify that the face on an ID document (Aadhaar card) matches a selfie using AI face matching.

### Step 1 & 2 — Upload or Capture Documents

| Element | Action | Expected Result |
|---|---|---|
| Upload Aadhaar card | Drag-and-drop or file picker | QR decoded automatically; name and Aadhaar number auto-fill the fields below |
| Upload PAN card | Drag-and-drop or file picker | PAN number and name auto-fill the fields below |
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
| "Run Identity Verification 🔍" button | Click | Runs InsightFace face-match (ArcFace) between Aadhaar and selfie; shows annotated images, confidence score, and MATCH/MISMATCH verdict |
| | If Aadhaar QR was decoded AND entity fields are filled | Result is saved to `identity_checks` table; entity created or updated; PDF report generated and download button appears |
| | If save fails (DB error) | Red error message: "Result could not be saved to the database. Check the Logs screen." |
| | If DB is offline | Warning: "Database is offline — result not persisted." |
| Previous checks table | Auto-renders after save | Shows all previous face-match records for the linked entity |

**Important rules that must NOT be broken:**
- `_draw_face()` in `vision/face.py` must always be defined before it is called from `compare_faces()`.
- `save_identity_check()` must always show an error (not silent failure) when it returns `None`.
- `init_db()` must be retried on each app load until it succeeds (not just first attempt).

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
| "Save to Database" | Auto on scan | Scan saved to `scans` table; PDF report generated and stored |

---

## 📦 Bulk Scan

**Purpose:** Scan many documents at once from a folder or a file list.

| Element | Action | Expected Result |
|---|---|---|
| Folder path input or file uploader | Enter path or upload zip | Queues all documents |
| "Start Bulk Scan" button | Click | Scans each file in sequence; shows progress bar + per-file results |
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

**Purpose:** Aggregated fraud analytics across all scans.

| Element | Action | Expected Result |
|---|---|---|
| Risk level breakdown chart | Auto-renders | Bar chart of High/Medium/Low/Review scan counts |
| Document type breakdown | Auto-renders | Most scanned document types |
| Time-series chart | Auto-renders | Scan volume over time |

---

## 🔗 Datasources

**Purpose:** Register and sync external document sources (local folders, S3, SharePoint, Google Drive).

| Element | Action | Expected Result |
|---|---|---|
| "Add Datasource" form | Fill + Submit | Registers new connector; saved to `config/datasources.json` |
| "Sync" button per source | Click | Pulls documents from the source into the BaseTruth workspace |
| Source health indicator | Auto-renders | Shows "Connected" or "Error" for each registered source |

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
| Metrics row | Auto-renders (cached) | Shows row counts for Entities, Scans, Document Extractions, Cases, Case Notes |
| Table selector | Select | Loads up to 500 rows from the chosen table |
| Data table | Auto-renders | Shows rows in a paginated dataframe |

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
