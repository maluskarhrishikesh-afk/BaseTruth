# BaseTruth — Database Design

**Engine:** PostgreSQL 16 (running as the `db` service in `docker-compose.yml`)  
**ORM:** SQLAlchemy 2.x  
**Connection string:** `postgresql://basetruth:basetruth_secret@db:5432/basetruth`  
**Object Storage:** MinIO (S3-compatible) — files stored at `http://minio:9000`, bucket `basetruth-reports`

---

## Design Principles

1. **One person → one entity.** The `entities` table is the canonical Person record. It is created (or matched) when documents are scanned. All documents for the same person share a single entity row.
2. **One row = one document scan.** Each file scan inserts one row in `scans`, linked back to the entity. Re-scanning appends a new row for a full audit trail.
3. **Cases are workflow state only.** A `cases` row records the analyst's decision (approve / reject) for a *document-type group per entity*. Document counts and risk are computed live from `scans`.
4. **Notes are append-only.** Analyst observations attach to a case via `case_notes`. Notes are never deleted.
5. **Full JSON preserved.** `scans.report_json` (JSONB) stores the complete verification report so nothing is lost.
6. **MinIO mirrors the DB.** Every scan auto-uploads the source file and the PDF report to MinIO under `{entity_ref}/{filename}`. Case bundle PDFs go to `{entity_ref}/case_reports/{timestamp}_case_report.pdf`. The DB is the source of truth; MinIO is the file-storage layer.

---

## Entity-Relationship Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        entities                                 │
│  id (PK)  entity_ref (UNIQUE)  first_name  last_name           │
│  email    phone    pan_number  aadhar_number                    │
│  created_at  updated_at                                         │
└───────────────┬────────────────────────┬────────────────────────┘
                │ 1                      │ 1
                │                        │
          ┌─────▼──────┐          ┌──────▼──────┐
          │   scans    │          │   cases     │
          │  id (PK)   │          │  id (PK)    │
          │  entity_id │          │  entity_id  │
          │  source_name│         │  case_key   │
          │  document_type│       │  document_type│
          │  truth_score│         │  status     │
          │  risk_level │         │  disposition│
          │  verdict    │         │  priority   │
          │  report_json│         │  assignee   │
          │  pdf_report │         │  labels[]   │
          │  generated_at│        │  max_risk_level│
          └────────────┘         │  document_count│
                                  │  created_at │
                                  │  updated_at │
                                  └──────┬──────┘
                                         │ 1
                                   ┌─────▼──────┐
                                   │ case_notes │
                                   │  id (PK)   │
                                   │  case_id   │
                                   │  author    │
                                   │  text      │
                                   │  created_at│
                                   └────────────┘
```

**MinIO Path Convention:**
```
{entity_ref}/
  {source_document.pdf}          ← original uploaded file
  {stem}_report.pdf              ← per-scan PDF report
  case_reports/
    {timestamp}_case_report.pdf  ← full case bundle PDF
```

---

## Table Definitions

### `entities` — Persons / Applicants

One row per person being verified. Created automatically on first document scan, or explicitly via the "Associate documents with a person" widget. Matched by PAN, Aadhaar, or (first_name, last_name) pair.

| Column         | Type          | Constraints     | Description                         |
|----------------|---------------|-----------------|-------------------------------------|
| `id`           | SERIAL        | PK              | Internal surrogate key              |
| `entity_ref`   | VARCHAR(20)   | UNIQUE NOT NULL | Human reference: `BT-000001`        |
| `first_name`   | VARCHAR(255)  |                 | Given name                          |
| `last_name`    | VARCHAR(255)  |                 | Family name                         |
| `email`        | VARCHAR(255)  |                 | Email address                       |
| `phone`        | VARCHAR(50)   |                 | Contact phone                       |
| `pan_number`   | VARCHAR(20)   |                 | Indian PAN (strong unique ID)       |
| `aadhar_number`| VARCHAR(20)   |                 | Aadhaar UID (strong unique ID)      |
| `created_at`   | TIMESTAMPTZ   | default now()   | Row creation time                   |
| `updated_at`   | TIMESTAMPTZ   | default now()   | Last field update                   |

**Identity matching order (first match wins):**
1. `pan_number` exact match
2. `aadhar_number` exact match
3. `(first_name, last_name)` case-insensitive match
4. If no match → new entity created

---

### `scans` — Document Verification Results

One row per document scan. Never updated after creation (immutable audit log).

| Column          | Type        | Constraints             | Description                        |
|-----------------|-------------|-------------------------|------------------------------------|
| `id`            | SERIAL      | PK                      | Surrogate key                      |
| `entity_id`     | INTEGER     | FK → entities(id) NULL  | Owning person (NULL if undetected) |
| `source_name`   | VARCHAR(500)| NOT NULL                | Original filename                  |
| `source_sha256` | VARCHAR(64) |                         | File SHA-256 hash                  |
| `document_type` | VARCHAR(100)|                         | `payslip`, `bank_statement`, etc.  |
| `truth_score`   | INTEGER     | nullable                | 0–100 integrity score              |
| `risk_level`    | VARCHAR(20) |                         | `low`, `medium`, `high`            |
| `verdict`       | TEXT        |                         | Plain-English verdict string       |
| `parse_method`  | VARCHAR(100)|                         | How the document was parsed        |
| `report_json`   | JSONB       | NOT NULL                | Full verification report           |
| `pdf_report`    | BYTEA       | nullable                | Inline PDF report bytes            |
| `generated_at`  | TIMESTAMPTZ | default now()           | Scan timestamp                     |

---

### `cases` — Workflow State

Stores analyst decisions per (entity × document_type) group. The case_key format is `{document_type}::{entity_ref}` (e.g. `payslip::BT-000001`). Document counts and risk levels are computed live from `scans`.

| Column           | Type        | Constraints             | Description                        |
|------------------|-------------|-------------------------|------------------------------------|
| `id`             | SERIAL      | PK                      | Surrogate key                      |
| `case_key`       | VARCHAR(500)| UNIQUE NOT NULL         | `payslip::BT-000001`               |
| `entity_id`      | INTEGER     | FK → entities(id) NULL  | Linked person                      |
| `document_type`  | VARCHAR(100)|                         | Document category                  |
| `status`         | VARCHAR(50) | default 'new'           | `new`, `triage`, `investigating`, `closed` |
| `disposition`    | VARCHAR(50) | default 'open'          | `open`, `cleared`, `fraud_confirmed`, etc. |
| `priority`       | VARCHAR(20) | default 'normal'        | `low`, `normal`, `high`, `critical`|
| `assignee`       | VARCHAR(255)|                         | Analyst name or email              |
| `labels`         | TEXT[]      | default {}              | Free-form tags                     |
| `max_risk_level` | VARCHAR(20) |                         | Cached highest risk in group       |
| `document_count` | INTEGER     |                         | Cached document count              |
| `created_at`     | TIMESTAMPTZ | default now()           | Case creation time                 |
| `updated_at`     | TIMESTAMPTZ | default now()           | Last update time                   |

---

### `case_notes` — Analyst Notes (Append-Only)

| Column      | Type        | Constraints          | Description               |
|-------------|-------------|----------------------|---------------------------|
| `id`        | SERIAL      | PK                   | Surrogate key             |
| `case_id`   | INTEGER     | FK → cases(id) CASCADE| Parent case               |
| `author`    | VARCHAR(255)| default 'analyst'    | Note author               |
| `text`      | TEXT        |                      | Note body                 |
| `created_at`| TIMESTAMPTZ | default now()        | Note timestamp            |

---

## MinIO Storage Layout

Files are organised under the MinIO bucket (`basetruth-reports` by default):

```
BT-000001/
  bank_statement.pdf            ← original source document
  bank_statement_report.pdf     ← per-scan PDF report
  payslip_2026_jan.pdf
  payslip_2026_jan_report.pdf
  case_reports/
    20260328_115546_case_report.pdf   ← bulk-scan bundle PDF

BT-000002/
  employment_letter.pdf
  employment_letter_report.pdf
  ...
```

All files are downloadable from:
1. **Reports page** — per-entity listing with per-scan download buttons
2. **Database → MinIO Storage tab** — full object list with sizes
3. **MinIO console** at `http://localhost:9001` (admin/admin)

---

## Case Key Convention

Cases are identified by `{document_type}::{entity_ref}`:

| Case Key                      | Meaning                                          |
|-------------------------------|--------------------------------------------------|
| `payslip::BT-000001`          | All payslips for person BT-000001                |
| `bank_statement::BT-000002`   | All bank statements for person BT-000002         |
| `form16::BT-000001`           | All Form 16s for person BT-000001                |

This grouping means **one case per document type per person**, regardless of how many documents were uploaded.

---

## Reset Behaviour

| Operation          | Effect                                                    |
|--------------------|-----------------------------------------------------------|
| **RESET Database** | Truncates `case_notes`, `cases`, `scans`, `entities` with CASCADE. Sequences restarted. |
| **RESET MinIO**    | Deletes all objects in the `basetruth-reports` bucket.    |

Both operations are available in **Database → Danger Zone** and require typing `RESET` to confirm.

`pan_card`, `aadhar`, `utility_bill`, `gift_letter`, `property_agreement`,
`offer_letter`, `increment_letter`, `generic`

---

### `scans`  *(core table)*

One row per document scan.

| Column                   | Type           | Constraints           | Description                                                        |
|--------------------------|----------------|-----------------------|--------------------------------------------------------------------|
| `id`                     | UUID           | PK, DEFAULT gen_uuid  | Globally unique scan ID                                            |
| `file_name`              | VARCHAR(512)   | NOT NULL              | Original file name (e.g. `payslip_2026_mar.pdf`)                   |
| `file_sha256`            | CHAR(64)       |                       | SHA-256 hex of the uploaded file                                   |
| `file_size_bytes`        | BIGINT         |                       | File size in bytes                                                 |
| `document_type`          | VARCHAR(50)    | FK → document_types   | Detected document type                                             |
| `type_confidence`        | NUMERIC(5,4)   |                       | Classifier confidence 0.0–1.0                                      |
| `parse_method`           | VARCHAR(50)    |                       | `liteparse`, `pypdf_fallback`, `ocr_pytesseract`, etc.             |
| `truth_score`            | SMALLINT       | CHECK 0–100           | Composite integrity score (100 = clean)                            |
| `risk_level`             | VARCHAR(20)    |                       | `low`, `review`, `medium`, `high`, `critical`                      |
| `verdict`                | VARCHAR(100)   |                       | Human-readable verdict string                                      |
| `scanned_at`             | TIMESTAMPTZ    | NOT NULL DEFAULT NOW  | UTC timestamp of scan                                              |
| `employee_name`          | VARCHAR(255)   |                       | Extracted (may be NULL for non-payslip docs)                       |
| `employer_name`          | VARCHAR(255)   |                       | Extracted employer / company name                                  |
| `gross_monthly_salary`   | NUMERIC(14,2)  |                       | Extracted gross monthly salary in INR                              |
| `annual_ctc`             | NUMERIC(14,2)  |                       | Extracted annual CTC in INR                                        |
| `bank_name`              | VARCHAR(255)   |                       | Extracted bank name                                                |
| `account_number_masked`  | VARCHAR(20)    |                       | Last 4 digits only (e.g. `xxxx1234`)                               |
| `ifsc_code`              | VARCHAR(11)    |                       | Extracted IFSC code                                                |
| `artifact_dir`           | TEXT           |                       | Absolute path to the artifact directory on disk                    |
| `pdf_report_path`        | TEXT           |                       | Path to the generated plain-English PDF report                     |
| `report_json`            | JSONB          | NOT NULL              | Full `_verification.json` content (for ad-hoc querying)            |
| `created_at`             | TIMESTAMPTZ    | DEFAULT NOW           | Row creation timestamp                                             |
| `deleted_at`             | TIMESTAMPTZ    |                       | Soft-delete timestamp (NULL = active)                              |

**Indexes:**
- `(file_sha256)` — detect duplicate uploads
- `(risk_level, scanned_at DESC)` — dashboard "recent high-risk" queries
- `(document_type, scanned_at DESC)` — type-filtered history
- `(employee_name)` — look up all docs for a specific person
- GIN index on `report_json` — flexible JSONB querying

---

### `signals`  *(one row per fraud check per scan)*

| Column        | Type         | Constraints              | Description                                           |
|---------------|--------------|--------------------------|-------------------------------------------------------|
| `id`          | UUID         | PK, DEFAULT gen_uuid     | Signal row ID                                         |
| `scan_id`     | UUID         | NOT NULL FK → scans(id) ON DELETE CASCADE | Parent scan               |
| `rule`        | VARCHAR(100) | NOT NULL                 | Rule code (e.g. `basic_gross_proportion`)             |
| `passed`      | BOOLEAN      | NOT NULL                 | True = check passed, False = discrepancy found        |
| `severity`    | VARCHAR(20)  |                          | `low`, `medium`, `high`, `critical`                   |
| `score`       | SMALLINT     | CHECK 0–100              | Per-signal confidence score                           |
| `summary`     | TEXT         |                          | One-line machine summary                              |
| `details`     | JSONB        |                          | Arbitrary key-value details from the check            |
| `created_at`  | TIMESTAMPTZ  | DEFAULT NOW              |                                                       |

**Indexes:**
- `(rule, passed)` — "all scans that failed rule X"
- `(scan_id)` — all signals for a given scan

---

### `cases`  *(mortgage application grouping)*

A case corresponds to one mortgage loan application.  It groups together all
documents submitted by the same applicant.

| Column           | Type         | Constraints       | Description                                                    |
|------------------|--------------|-------------------|----------------------------------------------------------------|
| `id`             | UUID         | PK gen_uuid       | Internal case ID                                               |
| `case_key`       | VARCHAR(200) | UNIQUE NOT NULL   | Human-readable key, e.g. `payslip::john_doe`                   |
| `applicant_name` | VARCHAR(255) |                   | Name extracted or entered manually                             |
| `status`         | VARCHAR(30)  | DEFAULT `new`     | `new`, `triage`, `investigating`, `pending_client`, `closed`   |
| `disposition`    | VARCHAR(30)  | DEFAULT `open`    | `open`, `monitor`, `escalate`, `cleared`, `fraud_confirmed`    |
| `priority`       | VARCHAR(20)  | DEFAULT `normal`  | `low`, `normal`, `high`, `urgent`                              |
| `assignee`       | VARCHAR(255) |                   | Analyst username / email                                       |
| `labels`         | TEXT[]       | DEFAULT `{}`      | Free-form tags (e.g. `{"self-employed", "mortgage"}`)          |
| `risk_summary`   | VARCHAR(20)  |                   | Highest risk level across all linked scans                     |
| `min_truth_score`| SMALLINT     |                   | Lowest truth score across all linked scans                     |
| `created_at`     | TIMESTAMPTZ  | DEFAULT NOW       |                                                                |
| `updated_at`     | TIMESTAMPTZ  | DEFAULT NOW       |                                                                |
| `deleted_at`     | TIMESTAMPTZ  |                   | Soft-delete                                                    |

---

### `case_scans`  *(many-to-many: case ↔ scans)*

| Column       | Type        | Constraints                  | Description                    |
|--------------|-------------|------------------------------|--------------------------------|
| `case_id`    | UUID        | NOT NULL FK → cases(id)      |                                |
| `scan_id`    | UUID        | NOT NULL FK → scans(id)      |                                |
| `linked_at`  | TIMESTAMPTZ | DEFAULT NOW                  |                                |

**PK:** `(case_id, scan_id)`

---

### `case_notes`  *(audit trail of analyst comments)*

| Column       | Type         | Constraints                  | Description                              |
|--------------|--------------|------------------------------|------------------------------------------|
| `id`         | UUID         | PK gen_uuid                  |                                          |
| `case_id`    | UUID         | NOT NULL FK → cases(id) ON DELETE CASCADE |                            |
| `author`     | VARCHAR(255) | NOT NULL DEFAULT `system`    | Analyst name / username                  |
| `text`       | TEXT         | NOT NULL                     | Free-form note                           |
| `created_at` | TIMESTAMPTZ  | DEFAULT NOW                  |                                          |

---

## Key Queries

```sql
-- All high/critical risk scans in the last 7 days
SELECT file_name, truth_score, risk_level, scanned_at
FROM   scans
WHERE  risk_level IN ('high', 'critical')
  AND  scanned_at > NOW() - INTERVAL '7 days'
ORDER  BY scanned_at DESC;

-- Scans where the basic_gross_proportion check failed
SELECT s.file_name, s.truth_score, sig.score, sig.details
FROM   scans s
JOIN   signals sig ON sig.scan_id = s.id
WHERE  sig.rule = 'basic_gross_proportion'
  AND  sig.passed = false;

-- Average truth score per document type (fraud frequency by type)
SELECT document_type, COUNT(*) AS total,
       ROUND(AVG(truth_score), 1) AS avg_score,
       COUNT(*) FILTER (WHERE risk_level IN ('high','critical')) AS high_risk_count
FROM   scans
GROUP  BY document_type
ORDER  BY avg_score;

-- All scans belonging to a specific case
SELECT s.file_name, s.document_type, s.truth_score, s.risk_level
FROM   scans s
JOIN   case_scans cs ON cs.scan_id = s.id
WHERE  cs.case_id = '<case-uuid>';

-- Documents with duplicate SHA-256 (same file re-submitted)
SELECT file_sha256, COUNT(*) AS uploads, ARRAY_AGG(file_name) AS names
FROM   scans
GROUP  BY file_sha256
HAVING COUNT(*) > 1;
```

---

## Migration Plan

| Phase | What                                             | Complexity |
|-------|--------------------------------------------------|------------|
| 1     | Create tables + indexes (DDL only)               | Low        |
| 2     | Insert into `scans` + `signals` on every scan    | Low        |
| 3     | Wire `cases` CRUD to the existing case_records   | Medium     |
| 4     | Migrate existing `case_records.json` into DB     | Low        |
| 5     | Dashboard queries against DB instead of JSON     | Medium     |
| 6     | API endpoints for case/scan history              | Medium     |

Phase 1–2 can be implemented without changing the UI at all.  Phase 3 onwards
replaces the current `case_records.json` flat-file store.

---

## Notes for Expert Review

- **UUID primary keys** are preferred over SERIAL integers to support future
  multi-region or distributed deployments.
- **JSONB for `report_json`** is intentional — it stores the full scan output
  for ad-hoc querying with `->>`/`@>` operators, allowing future analytics
  without schema migrations.
- **Soft deletes** (`deleted_at`) rather than hard deletes to preserve the
  audit trail required in fraud investigations.
- **No PII encryption at rest** in this design — if required by your
  compliance framework (e.g. DPDP Act), `employee_name`, `account_number_masked`
  and `applicant_name` should be encrypted at the application layer before
  storage (pgcrypto or application-level AES-256).
- **Password in docker-compose.yml** (`basetruth_secret`) is for local
  development only.  Production deployments should use Docker secrets or a
  secrets manager (AWS Secrets Manager, HashiCorp Vault).
