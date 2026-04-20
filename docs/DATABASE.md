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
2. `scans`, `identity_checks`, `document_extractions`, and `entity_reports` use operational UPSERT behaviour so the UI shows one current record per natural entity-scoped key.
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

### `document_extractions`

**What is it used for?**
Stores the rich extracted JSON for a scanned or verified document so operators and auditors can review the normalised document profile without re-parsing the source file. Populated from three sources: the Scan Document screen (OCR path), the Bulk Scan screen (Gemma4 AI extraction after forensics), and the Identity Verification screen (Aadhaar/PAN fields from face matching).

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

## Entity Report Generation Rules

1. The Document Intelligence screen generates a final `entity_reports` row by reading the entity's `document_extractions` and `scans`.
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

`face_match_report.pdf` and `video_kyc_report.pdf` are the current identity PDFs under the entity prefix. Re-saving Identity Verification replaces the current object in-place.

Final verification reports (`BTR-XXXXXX.pdf`) are stored under the `BTR-reports/` prefix and are downloadable from the Document Intelligence and Reports screens.

---

## Reset Behaviour

Database reset truncates all five tables in dependency order:
- `entity_reports`
- `document_extractions`
- `identity_checks`
- `scans`
- `entities`

MinIO reset clears all objects in the configured bucket.