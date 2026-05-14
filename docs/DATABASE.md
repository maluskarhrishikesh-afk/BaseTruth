# BaseTruth — Database & Storage Guide

This document explains what data gets stored where. It has two parts:
- **Part 1 (Plain English):** What each table is for and what it holds — written so any team member can understand without knowing SQL.
- **Part 2 (Technical Reference):** Column-level detail for developers.

---

## Part 1 — Plain English Guide

### What kind of storage does BaseTruth use?

| Storage | What it holds |
|---|---|
| **PostgreSQL** | Structured data — text, numbers, verdicts, JSON blobs |
| **MinIO** | Raw files — the actual uploaded PDFs, photos, and generated reports |

Think of PostgreSQL as the filing cabinet *index*, and MinIO as the physical filing room where the actual documents sit.

---

### `entities` — One row per person

Every person you check gets one row in this table. This row acts as the "master record" that links all their documents, scans, and identity checks together.

**What gets stored:** A unique reference code (e.g. `BT-000042`), their name, email, phone, Aadhaar number, PAN number (all optional), and audit timestamps.

**Created when:** You type a person's name in any screen that asks for Entity Details (Bulk Scan, Scan Document, Identity Verification) and click Save.

---

### `scans` — One row per uploaded document

Every time a document is scanned and saved, it gets a row here. If the same person uploads three payslips, that's three scan rows, all linked to their one entity row.

**What gets stored:** The document filename, a SHA-256 fingerprint, the document type (e.g. `payslip`), the forensic verdict (`GENUINE`, `SUSPICIOUS`, `TAMPERED`), a forgery score 0–100, the full 11-layer forensic JSON, and the Gemma4 classification result. Also tracks a two-level human approval workflow.

**Created when:** You click "Save Results" on the Bulk Scan screen, or "Save Scan" on the Scan Document screen.

---

### `document_extractions` — The actual data pulled out of a document

This table stores the readable **fields** extracted from a document — things like "employee name: Ravi Kumar", "net salary: ₹55,000", "university: Pune University". It answers: *"What does this document actually say?"*

**`file_name` stores the uploaded source filename** and BaseTruth upserts this table by `(entity_id, file_name)` so the latest extraction for the same applicant and same file replaces the earlier one.

**What gets stored by document type:**
- **Payslip:** employee name, employee ID, company, pay period, basic/gross/net salary, allowances, deductions
- **Marksheet:** candidate name, board/university, school, all subjects with marks, total, percentage, result
- **Degree Certificate:** candidate name, university, degree name, specialization, class, year, certificate number
- **Offer Letter:** candidate name, company, designation, joining date, CTC per annum, gross monthly
- **Increment Letter:** employee name, company, previous salary, new salary, increment amount and percentage
- **Bank Statement:** account holder, bank name, account number, IFSC, statement period, opening/closing balances
- **Form 16:** employee name, PAN, employer name, financial year, gross salary, total tax deducted

Identity Verification and Video KYC do **not** store their extracted Aadhaar, PAN, or address-proof payloads here. Those live in the dedicated biometric-check tables.

**`source_screen` field tells you where the extraction came from:**
- `scan_document` — full OCR pipeline on the Scan Document screen
- `bulk_scan` — Gemma4 multimodal extraction after forensics in the Bulk Scan screen

---

### `identity_checks` — Current Identity Verification result

Stores the current saved Identity Verification outcome for one applicant.

**What gets stored:** The face-match status and score, the structured Aadhaar payload, the structured PAN payload, MinIO object keys for the Aadhaar image, PAN image, selfie, PAN signature crop, the full evidence JSON, and the generated PDF-report object key.

**Created when:** You click "Save to Database" on the Identity Verification screen.

**Update rule:** One current row is kept per entity. Re-saving the same applicant updates the same row instead of creating repeated history entries.

---

### `video_kyc_checks` — Current Video KYC result

Stores the current saved Video KYC outcome for one applicant.

**What gets stored:** The liveness result, the live-face match score, the extracted identity-proof payload, the extracted address-proof payload, current-location data from the customer's browser, the resolved current address, the address-comparison result, the address-distance metric, MinIO object keys for the reference document, address proof, best live frame, and best challenge frames, the full evidence JSON, and the generated PDF-report object key.

**Created when:** You click "Save to Database" on the Video KYC screen after a completed session or in-person verification.

**Update rule:** One current row is kept per entity. Re-saving the same applicant updates the same row instead of creating repeated history entries.

---

### `entity_reports` — Final cross-document verification reports (BTR-XXXXXX)

After all of an applicant’s documents have been scanned and approved, an analyst can generate a **Final Verification Report**. This report compares every document the applicant submitted and checks whether the name, address, PAN, Aadhaar, and salary are consistent across all of them. It also summarises whether any documents failed the forensic check.

The report goes through the same two-level human approval process as individual document scans: a first reviewer approves or rejects it, then a senior reviewer makes the final call.

**What gets stored:** A unique reference (e.g. `BTR-000001`), a link to the applicant, the full cross-document findings as JSON (one entry per check — name, address, PAN, Aadhaar, salary, forensics), and the two-level approval trail.

**Created when:** An analyst clicks "🎯 Generate Final Report" on the **Document Intelligence** screen after the applicant's documents have been scanned.

**Reference format:** `BTR-XXXXXX` (as opposed to entity references which are `BT-XXXXXX` and scan approval is on individual scans).

---

## Part 2 — Technical Reference

**Engine:** PostgreSQL 16  
**ORM:** SQLAlchemy 2.x  
**Object Storage:** MinIO (S3-compatible), bucket `basetruth-reports`

---

## Design Principles

1. `entities` is the canonical applicant table.
2. `scans`, `document_extractions`, `identity_checks`, `video_kyc_checks`, `entity_reports`, and `face_scan_live_results` use operational UPSERT behaviour so the UI shows one current record per natural entity-scoped key.
3. Final reports (`entity_reports`) for a pending entity are refreshed in-place; once approved or rejected a new `BTR-XXXXXX` row is created so the audit trail is preserved.
4. Deleting an entity cascades to all related rows across every table.

---

## Core Tables

### `entities`

**What is it used for?**
Stores the basic profile of every person or organisation in the system (like their first name, last name, email, PAN, and Aadhaar). Think of this as the master record for an applicant. Every document or check they do is linked back to this identity.

| Column | Type | Notes |
|---|---|---|
| `id` | SERIAL | Primary key |
| `entity_ref` | VARCHAR(20) | Unique human-readable reference, e.g. `BT-000001` |
| `first_name` / `last_name` | VARCHAR | Searchable identity fields |
| `email` / `phone` | VARCHAR | Contact fields |
| `pan_number` / `aadhar_number` | VARCHAR | Strong identity keys |
| `created_at` / `updated_at` | TIMESTAMPTZ | Audit timestamps |

**Entity matching order:**
1. PAN exact match
2. Aadhaar exact match
3. Case-insensitive `(first_name, last_name)` match
4. Else create a new entity

---

### `scans`

**What is it used for?**
Stores the final result of every document uploaded and scanned by the system. All document intelligence is captured in `layered_analysis_json` — a full 11-layer forensic result. Each scan goes through a **two-level human approval process** before it is considered fully verified.

| Column | Type | Notes |
|---|---|---|
| `id` | SERIAL | Primary key |
| `entity_id` | FK → `entities.id` | Nullable when no entity could be resolved |
| `source_name` | VARCHAR(500) | Original filename |
| `source_sha256` | VARCHAR(64) | Source hash |
| `document_type` | VARCHAR(100) | Document type e.g. payslip, pan_card, aadhaar |
| `layered_analysis_json` | JSONB | Full 11-layer forensic result (ELA, noise, clone, etc.) |
| `first_level_approval` | VARCHAR(1) | `Y` = approved, `N` = rejected, `NULL` = pending (initial reviewer) |
| `first_level_approved_by` | VARCHAR(255) | Who made the 1st-level decision |
| `first_level_approved_at` | TIMESTAMPTZ | When 1st-level decision was made |
| `first_level_approval_comment` | TEXT | Optional 1st-level reviewer comment |
| `second_level_approval` | VARCHAR(1) | `Y` = approved, `N` = rejected, `NULL` = pending (senior reviewer) |
| `second_level_approved_by` | VARCHAR(255) | Who made the 2nd-level decision |
| `second_level_approved_at` | TIMESTAMPTZ | When 2nd-level decision was made |
| `second_level_approval_comment` | TEXT | Optional 2nd-level reviewer comment |
| `approved` | VARCHAR(10) | Legacy column — derived value: `approved` / `rejected` / `NULL` |
| `approved_by` | VARCHAR(255) | Legacy — mirrors 1st-level approved_by |
| `approved_at` | TIMESTAMPTZ | Legacy — mirrors 1st-level approved_at |
| `approval_comment` | TEXT | Legacy — mirrors 1st-level comment |
| `generated_at` | TIMESTAMPTZ | Save timestamp |
| `updated_at` | TIMESTAMPTZ | Last update timestamp |

**Two-level approval flow:**
1. A new scan begins with `first_level_approval = NULL`, `second_level_approval = NULL` → **Pending**
2. Initial reviewer approves → `first_level_approval = 'Y'` → **Awaiting 2nd Review**
3. Senior reviewer approves → `second_level_approval = 'Y'` → **Fully Approved**
4. Either reviewer rejects → their level = `'N'` → **Rejected**

**Operational UPSERT key:**
- `(entity_id, source_name, document_type)`

---

### `identity_checks`

**What is it used for?**
Stores the current Identity Verification result for one entity. This tells you whether the Aadhaar face, PAN details, and selfie verification passed for the applicant.

| Column | Type | Notes |
|---|---|---|
| `id` | SERIAL | Primary key |
| `entity_id` | FK → `entities.id` | Nullable if no entity could be resolved |
| `status` | VARCHAR(20) | `pass`, `fail`, `inconclusive` |
| `cosine_similarity` | FLOAT | Face-match score |
| `display_score` | FLOAT | 0–100 presentation score |
| `threshold` | FLOAT | Applied threshold |
| `is_match` | BOOLEAN | Face-match result |
| `aadhar_dtls` | JSONB | Aadhaar QR payload |
| `pan_dtls` | JSONB | PAN extraction payload |
| `selfie_pic` | VARCHAR(500) | MinIO object key for the saved selfie |
| `signature_pic` | VARCHAR(500) | MinIO object key for the PAN signature crop |
| `aadhaar_pic` | VARCHAR(500) | MinIO object key for the Aadhaar source image |
| `pan_pic` | VARCHAR(500) | MinIO object key for the PAN source image |
| `report_json` | JSONB | Full result payload |
| `pdf_report` | VARCHAR(500) | MinIO object key for the current PDF report |
| `created_at` | TIMESTAMPTZ | Save timestamp |
| `updated_at` | TIMESTAMPTZ | Last update timestamp |

**Operational UPSERT key:**
- `(entity_id)`
- Re-saving Identity Verification for the same entity updates the current row and replaces the current `face_match_report.pdf` object in MinIO.

---

### `video_kyc_checks`

**What is it used for?**
Stores the current Video KYC result for one entity. This tells you whether the applicant passed liveness, matched the live face against the uploaded proof document, and matched the current location against the address proof. The 2026 enriched schema stores Aadhaar QR, PAN extraction, and separate image columns — mirroring the `identity_checks` structure.

| Column | Type | Notes |
|---|---|---|
| `id` | SERIAL | Primary key |
| `entity_id` | FK → `entities.id` | Nullable if no entity could be resolved |
| `status` | VARCHAR(20) | `pass`, `fail`, `inconclusive` |
| `cosine_similarity` | FLOAT | Live-face match score |
| `display_score` | FLOAT | 0–100 presentation score |
| `threshold` | FLOAT | Applied threshold |
| `is_match` | BOOLEAN | Live-face match result |
| `liveness_state` | VARCHAR(50) | Final / latest liveness state |
| `liveness_passed` | BOOLEAN | Liveness result |
| `aadhar_dtls` | JSONB | Aadhaar QR decoded payload (name, dob, gender, uid, state, pc) |
| `pan_dtls` | JSONB | PAN card extraction payload (pan_number, full_name, father_name, dob) |
| `address_dtls` | JSONB | Address-proof payload |
| `isAddressMatch` | VARCHAR(20) | `match`, `mismatch`, `partial`, or `skipped` |
| `kyc_comments` | VARCHAR(500) | System/operator note for mismatch or distance context |
| `current_location_json` | JSONB | Browser latitude/longitude/accuracy/timestamp payload |
| `current_address_text` | TEXT | Reverse-geocoded current address |
| `address_distance_meters` | FLOAT | Distance between current location and proof address |
| `video_kyc_pic` | VARCHAR(500) | MinIO object key for the best live frame |
| `address_proof_pic` | VARCHAR(500) | MinIO object key for the address-proof upload |
| `aadhaar_pic` | VARCHAR(500) | MinIO object key for the Aadhaar card image |
| `pan_pic` | VARCHAR(500) | MinIO object key for the PAN card image |
| `signature_pic` | VARCHAR(500) | MinIO object key for the PAN signature crop |
| `challenge_snapshots_json` | JSONB | Metadata for one best retained frame per completed challenge |
| `report_json` | JSONB | Full Video KYC payload |
| `pdf_report` | VARCHAR(500) | MinIO object key for the current PDF report |
| `created_at` | TIMESTAMPTZ | Save timestamp |
| `updated_at` | TIMESTAMPTZ | Last update timestamp |
| `identity_dtls` | JSONB | Legacy: alias for `aadhar_dtls` (kept for backward compat) |
| `reference_doc_pic` | VARCHAR(500) | Legacy: superseded by `aadhaar_pic` / `pan_pic` |

**Operational UPSERT key:**
- `(entity_id)`
- Re-saving Video KYC for the same entity updates the current row and replaces the current `video_kyc_report.pdf` object in MinIO.

---

### `document_extractions`

**What is it used for?**
Stores the rich extracted JSON for a scanned document so operators and auditors can review the normalised document profile without re-parsing the source file. Populated from two sources: the Scan Document screen (OCR path) and the Bulk Scan screen (Gemma4 AI extraction after forensics).

| Column | Type | Notes |
|---|---|---|
| `id` | SERIAL | Primary key |
| `entity_id` | FK → `entities.id` | Required |
| `scan_id` | FK → `scans.id` | Nullable when a save path does not create a `scans` row |
| `file_name` | VARCHAR(500) | Uploaded filename; per-entity UPSERT key |
| `document_type` | VARCHAR(100) | Normalized document type (e.g. `payslip`, `aadhaar`) |
| `extracted_data` | JSONB | Rich extracted JSON payload |
| `source_screen` | VARCHAR(100) | `scan_document` or `bulk_scan` |
| `created_at` | TIMESTAMPTZ | Insert timestamp |

**Operational UPSERT key:**
- `(entity_id, file_name)`

**`extracted_data` JSON shape varies by document type (see Part 1 above for field list per type).**

> **Migration note:** This table was renamed from `document_information` to `document_extractions`. The `init_db()` function runs the rename migration automatically on startup if the old table still exists.

---

### `entity_reports`

**What is it used for?**
Stores the output of the cross-document consistency analysis that an analyst runs from the Document Intelligence screen. Each row represents one final analysis for one applicant, tracking whether all their submitted documents are consistent with each other.

| Column | Type | Notes |
|---|---|---|
| `id` | SERIAL | Primary key |
| `entity_id` | FK → `entities.id` (CASCADE DELETE) | Which applicant this report belongs to |
| `report_ref` | VARCHAR(20), UNIQUE | Human-readable ID e.g. `BTR-000001` |
| `report_json` | JSONB | Full cross-document analysis payload (checks for name, address, PAN, Aadhaar, salary, forensics) |
| `first_level_approval` | VARCHAR(1) | `Y` = approved, `N` = rejected, `NULL` = pending (initial reviewer) |
| `first_level_approved_by` | VARCHAR(255) | Who made the 1st-level decision |
| `first_level_approved_at` | TIMESTAMPTZ | When 1st-level decision was made |
| `first_level_approval_comment` | TEXT | Optional 1st-level reviewer note |
| `second_level_approval` | VARCHAR(1) | `Y` = approved, `N` = rejected, `NULL` = pending (senior reviewer) |
| `second_level_approved_by` | VARCHAR(255) | Who made the 2nd-level decision |
| `second_level_approved_at` | TIMESTAMPTZ | When 2nd-level decision was made |
| `second_level_approval_comment` | TEXT | Optional 2nd-level reviewer note |
| `generated_at` | TIMESTAMPTZ | When the report was first generated |
| `updated_at` | TIMESTAMPTZ | When the report was last updated |

**Behaviour rules:**
- A pending (unapproved) report for an entity is refreshed in-place when the analyst generates again — no duplicate is created.
- Once approved or rejected, a re-generation creates a new row with a new BTR-XXXXXX reference so the audit trail of past decisions is never lost.
- Deleting an entity cascades to delete all their entity reports automatically.

**`report_json` structure:**
```json
{
  "entity_ref": "BT-000001",
  "entity_name": "Ravi Kumar",
  "overall_verdict": "PASS | FAIL",
  "documents_analysed": 5,
  "scans_reviewed": 5,
  "checks": {
    "name":     { "status": "PASS | MISMATCH",   "detail": "..." },
    "address":  { "status": "PASS | MISMATCH",   "detail": "..." },
    "pan":      { "status": "PASS | MISMATCH",   "detail": "..." },
    "aadhaar":  { "status": "PASS | MISMATCH",   "detail": "..." },
    "salary":   { "status": "PASS | MISMATCH | SKIP", "detail": "..." },
    "forensics":{ "status": "CLEAR | TAMPERED",  "detail": "..." }
  },
  "per_document_evidence": [ ... ]
}
```

---

### `face_scan_live_results`

**What is it used for?**
Stores the durable result of every completed Live Face Scan session. The in-memory session objects expire after 20 minutes; this table is the permanent audit trail. Operators can review past verdicts, download the best still frame, and watch the recorded challenge video long after the session has ended.

| Column | Type | Notes |
|---|---|---|
| `id` | SERIAL | Primary key |
| `session_id` | VARCHAR(50), UNIQUE | URL-safe token matching the in-memory `FaceScanLiveSession` |
| `verdict` | VARCHAR(20) | `GENUINE` \| `SUSPICIOUS` \| `DEEPFAKE` \| `INCONCLUSIVE` \| `LIVENESS_FAILED` |
| `risk_score` | FLOAT | 0–100 risk score from the ML model or heuristic fallback |
| `confidence` | FLOAT | 0–1 model confidence value |
| `best_frame_key` | VARCHAR(500) | MinIO object key for the best captured still frame (e.g. `face-scan-frames/{session_id}.jpg`) |
| `video_key` | VARCHAR(500) | MinIO object key for the recorded MP4 video — `NULL` when not recorded |
| `report_json` | JSONB | Full live scan result payload for audit and re-display |
| `created_at` | TIMESTAMPTZ | Row creation timestamp |
| `updated_at` | TIMESTAMPTZ | Last update timestamp |

**`video_key` is NULL when:**
- `FACE_SCAN_RECORD_VIDEO` env var is `false` (recording disabled), or
- the MinIO upload failed (soft failure — the session result is never blocked), or
- the captured frame buffer was empty.

**No entity link:** `face_scan_live_results` rows are not linked to `entities` by a foreign key. The session is identified by `session_id` only. Deleting an entity does **not** cascade to this table.

---

## Entity Report Generation Rules

1. The Document Intelligence screen generates a final `entity_reports` row by reading the entity's `document_extractions`, `identity_checks`, `video_kyc_checks`, and `scans`.
2. A pending (unapproved) report is refreshed in-place — no duplicate row is created.
3. Once a report is approved or rejected, re-generation creates a new row with a new `BTR-XXXXXX` reference, preserving the audit trail of past decisions.
4. The rendered PDF is uploaded to MinIO under the `BTR-reports/{entity_ref}/` prefix.

---

## MinIO Layout

Objects are stored under the entity reference prefix:

```text
{entity_ref}/
  {source_document}
  {scan_or_identity_report}.pdf
  face_match_report.pdf
  video_kyc_report.pdf
BTR-reports/{entity_ref}/
  {BTR-XXXXXX}.pdf
```

`face_match_report.pdf` and `video_kyc_report.pdf` are the current identity PDFs under the entity prefix. Their owning rows store those object keys in `identity_checks.pdf_report` and `video_kyc_checks.pdf_report`. Re-saving either flow replaces the current object in-place.

Final verification reports (`BTR-XXXXXX.pdf`) are stored under the `BTR-reports/` prefix and are downloadable from the Document Intelligence and Reports screens.

---

## Reset Behaviour

Database reset truncates all seven tables in dependency order:
- `entity_reports`
- `video_kyc_checks`
- `document_extractions`
- `identity_checks`
- `scans`
- `entities`
- `face_scan_live_results`

MinIO reset clears all objects in the configured bucket.