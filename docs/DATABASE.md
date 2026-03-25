# BaseTruth — Database Design

**Engine:** PostgreSQL 16 (running as the `db` service in `docker-compose.yml`)  
**ORM:** SQLAlchemy 2.x (async-compatible)  
**Connection string:** `postgresql://basetruth:basetruth_secret@db:5432/basetruth`

---

## Design Principles

1. **One row = one document scan.** Each time a file is scanned a row is
   inserted into `scans`.  Re-scanning the same file appends a new row (full
   audit trail).
2. **Signals stored relationally.** Each fraud signal (pass/fail check) is its
   own row in `signals`, FK'd to its parent scan.  This lets you query
   "all scans that failed rule X" without parsing JSON.
3. **Cases are separate from scans.** A `case` groups one or more scans that
   belong to the same mortgage application.  The link is a many-to-many join
   table `case_scans`.
4. **Original JSON preserved.** A `report_json` JSONB column on `scans` stores
   the full `_verification.json` so nothing is lost.
5. **Soft deletes only.** Rows are never hard-deleted; `deleted_at` timestamps
   are used instead.

---

## Entity-Relationship Overview

```
cases  ─────< case_scans >───── scans ─────< signals
  │                               │
  └──< case_notes              document_types (lookup)
```

---

## Table Definitions

### `document_types`  *(lookup / seed table)*

Enumerated set of document types BaseTruth can classify.

| Column        | Type        | Constraints          | Description                          |
|---------------|-------------|----------------------|--------------------------------------|
| `id`          | SERIAL      | PK                   | Surrogate key                        |
| `code`        | VARCHAR(50) | UNIQUE NOT NULL      | Machine name, e.g. `payslip`         |
| `label`       | VARCHAR(100)| NOT NULL             | Human label, e.g. `Pay Slip`         |
| `description` | TEXT        |                      | One-line description                 |

**Seed values:** `payslip`, `bank_statement`, `employment_letter`, `form16`,
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
