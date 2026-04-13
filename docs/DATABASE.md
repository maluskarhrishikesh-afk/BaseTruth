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
- **Aadhaar:** name, UID (masked), date of birth, address, gender
- **PAN Card:** PAN number, full name, father's name, date of birth

**`source_screen` field tells you where the extraction came from:**
- `scan_document` — full OCR pipeline on the Scan Document screen
- `bulk_scan` — Gemma4 multimodal extraction after forensics in the Bulk Scan screen
- `identity_verification` — Aadhaar/PAN fields captured during face matching

**Identity Verification save behavior:**
- If Aadhaar QR data is present, one `aadhaar` row is written
- If PAN extraction data is present, one `pan_card` row is written
- If both are present, the same save creates or updates two separate rows

**Note:** For identity verification extractions, `scan_id` is empty (NULL) because there is no scan row for an identity check — only an identity_check row.

---

### `identity_checks` — Results of face matching and Video KYC

Stores the outcome every time someone's face is compared to their ID document, or a Video KYC session is completed.

**What gets stored:** The check type (`face_match` or `video_kyc`), the verdict (`PASS`/`FAIL`), the cosine similarity score (how closely the two faces matched), whether liveness was detected (for Video KYC), and the generated PDF report.

**Created when:** You click "Save Check" on the Identity Verification screen, or a Video KYC session completes.

---

### `layered_analysis_entries` — Detailed audit trail

A very detailed logbook. For every major action (each scan, each identity check), detailed notes are written here — one row per screen section. This powers the Layered Analysis view and the final PDF report.

**Created automatically** whenever a scan or identity check is saved.

---

### `cases` and `case_notes` — Manual review workflow

When a document is flagged as risky and needs manual analysis, you open a Case (like a support ticket). Case Notes are the comment thread on a case.

---

## Part 2 — Technical Reference

**Engine:** PostgreSQL 16  
**ORM:** SQLAlchemy 2.x  
**Object Storage:** MinIO (S3-compatible), bucket `basetruth-reports`

---

## Design Principles

1. `entities` is the canonical applicant table.
2. `scans`, `identity_checks`, and `layered_analysis_entries` use operational UPSERT behavior so the UI shows one current record per natural entity-scoped key.
3. `layered_analysis_entries` stores the latest explainability snapshot per entity/screen/section using UPSERT semantics.
4. Final layered-analysis report generation is controlled at the entity level so the same evidence set cannot be reported twice.
5. Saving fresh evidence for an entity automatically invalidates the previously generated final report and re-enables generation.

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
| `layered_report_generated` | BOOLEAN | `true` only when the latest evidence has already been reported |
| `layered_report_generated_at` | TIMESTAMPTZ | When the current final report was generated |
| `layered_analysis_updated_at` | TIMESTAMPTZ | When any layered-analysis section was last upserted |
| `layered_report_minio_key` | VARCHAR(500) | MinIO object key for the current final report |
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
Stores the results of biometric checks like Face Matching (comparing a selfie to an ID) and Video KYC (live video verification). This tells you if a person successfully proved they are who they claim to be.

| Column | Type | Notes |
|---|---|---|
| `id` | SERIAL | Primary key |
| `entity_id` | FK → `entities.id` | Nullable if no entity could be resolved |
| `check_type` | VARCHAR(30) | `face_match` or `video_kyc` |
| `status` | VARCHAR(20) | `pass`, `fail`, `inconclusive` |
| `cosine_similarity` | FLOAT | Face-match score |
| `display_score` | FLOAT | 0–100 presentation score |
| `threshold` | FLOAT | Applied threshold |
| `is_match` | BOOLEAN | Face-match result |
| `liveness_state` / `liveness_passed` | VARCHAR / BOOLEAN | Video KYC fields |
| `verdict` | VARCHAR(20) | `PASS` / `FAIL` |
| `doc_filename` / `selfie_filename` | VARCHAR(500) | Source filenames |
| `report_json` | JSONB | Full result payload |
| `pdf_report` | BYTEA | Optional PDF report bytes |
| `created_at` | TIMESTAMPTZ | Save timestamp |

**Operational UPSERT key:**
- `(entity_id, check_type)`
- Re-saving Identity Verification for the same entity updates the current `face_match` row and replaces the current `face_match_report.pdf` object in MinIO.

---

### `layered_analysis_entries`

**What is it used for?**
A very detailed logbook that records every tiny piece of information captured during a user's verification journey (like the exact text pulled from their Aadhaar scan or details from their identity check). This table's main job is to collect all evidence to magically generate the final PDF "Layered Analysis" report.

| Column | Type | Notes |
|---|---|---|
| `id` | SERIAL | Primary key |
| `entity_id` | FK → `entities.id` | Required |
| `screen_name` | VARCHAR(100) | e.g. `Identity Verification`, `Video KYC`, `Scan Document`, `Bulk Scan` |
| `section_name` | VARCHAR(255) | e.g. `Aadhaar`, `PAN Card`, `Run Verification`, or source filename |
| `details_captured_json` | JSONB | Structured section payload |
| `created_at` | TIMESTAMPTZ | First insert timestamp |
| `updated_at` | TIMESTAMPTZ | Latest UPSERT timestamp |

**Uniqueness rule:**
- Unique on `(entity_id, screen_name, section_name)`

**UPSERT behavior:**
- If the entity/screen/section tuple already exists, the row is updated.
- Otherwise, a new row is inserted.
- Every UPSERT also resets `entities.layered_report_generated = false` and clears the stored final-report pointer so the report can be generated again for fresh evidence.

---

### `document_extractions`, `cases`, `case_notes`

**What are they used for?**
- **`document_extractions`** (formerly `document_information`): Stores the rich extracted JSON for a scanned or verified document so operators and auditors can review the normalized document profile without re-parsing the source file. This table is populated from three sources: the Scan Document screen (OCR path), the Bulk Scan screen (Gemma4 AI extraction after forensics), and the Identity Verification screen (Aadhaar/PAN fields from face matching).
- **`cases`**: Like a to-do list for human analysts. When a document is flagged as risky and needs manual review, a "case" is created here to track its status (open, closed, under review).
- **`case_notes`**: The comment section for cases. Analysts can leave text notes explaining why they approved or rejected a document.

#### `document_extractions`

| Column | Type | Notes |
|---|---|---|
| `id` | SERIAL | Primary key |
| `entity_id` | FK → `entities.id` | Required |
| `scan_id` | FK → `scans.id` | Nullable — NULL for identity_verification extractions |
| `file_name` | VARCHAR(500) | Uploaded filename; per-entity UPSERT key |
| `document_type` | VARCHAR(100) | Normalized document type (e.g. `payslip`, `aadhaar`) |
| `extracted_data` | JSONB | Rich extracted JSON payload |
| `source_screen` | VARCHAR(100) | `scan_document`, `bulk_scan`, or `identity_verification` |
| `created_at` | TIMESTAMPTZ | Insert timestamp |

**Operational UPSERT key:**
- `(entity_id, file_name)`

**`extracted_data` JSON shape varies by document type (see Part 1 above for field list per type).**

> **Migration note:** This table was renamed from `document_information` to `document_extractions`. The `init_db()` function runs the rename migration automatically on startup if the old table still exists.

---

## Layered Analysis Capture Model

Examples of current section capture:

### Identity Verification
- `Aadhaar`
- `PAN Card`
- `Photo Upload`
- `Run Verification`

### Video KYC
- `Remote Session`
- `In-Person Session`

### Scan Document / Bulk Scan
- One section per saved source document, keyed by filename

---

## Final Report Generation Rules

1. The Layered Analysis final report is generated only from `layered_analysis_entries`.
2. When a final report is generated successfully:
   - the PDF is uploaded to MinIO under the entity-specific key
   - `entities.layered_report_generated` becomes `true`
   - `entities.layered_report_generated_at` is set
   - `entities.layered_report_minio_key` stores the active object key
3. While `layered_report_generated = true` for the current evidence set, the Layered Analysis screen must not allow regeneration.
4. Any fresh UPSERT into `layered_analysis_entries` for that entity resets the flag and unlocks report generation again.

---

## MinIO Layout

Objects are stored under the entity reference prefix:

```text
{entity_ref}/
  {source_document}
  {scan_or_identity_report}.pdf
  consolidated_report.pdf
  layered_analysis_report.pdf
  case_reports/
    {timestamp}_case_report.pdf
```

`layered_analysis_report.pdf` is the current final explainability report and is downloadable from both the Layered Analysis screen and the Reports screen.

`face_match_report.pdf` and `video_kyc_report.pdf` are stored as the current identity PDFs under the entity prefix. Legacy timestamped identity report objects are cleaned up by the new upsert flow.

---

## Reset Behaviour

Database reset truncates:
- `layered_analysis_entries`
- `case_notes`
- `cases`
- `document_extractions`
- `identity_checks`
- `scans`
- `entities`

MinIO reset clears all objects in the configured bucket.