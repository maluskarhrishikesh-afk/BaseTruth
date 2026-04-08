# BaseTruth — Database Design

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

### `document_information`, `cases`, `case_notes`

**What are they used for?**
- **`document_information`**: Stores the rich extracted JSON for a scanned document so operators and auditors can review the normalized document profile without reparsing the source file. The payload is intentionally broader than a few key fields and includes document typing, parser hints, authenticity checks, extracted fields, and top fraud signals.
- **`cases`**: Like a to-do list for human analysts. When a document is flagged as risky and needs manual review, a "case" is created here to track its status (open, closed, under review).
- **`case_notes`**: The comment section for cases. Analysts can leave text notes explaining why they approved or rejected a document.

#### `document_information`

| Column | Type | Notes |
|---|---|---|
| `id` | SERIAL | Primary key |
| `entity_id` | FK → `entities.id` | Required |
| `scan_id` | FK → `scans.id` | Required; one operational row per saved scan |
| `document_type` | VARCHAR(100) | Normalized document type |
| `extracted_data` | JSONB | Rich extracted JSON payload |
| `created_at` | TIMESTAMPTZ | Insert timestamp |

**Operational UPSERT key:**
- `scan_id`

**`extracted_data` JSON shape (current):**
- `document_type`
- `document`
- `key_fields`
- `named_fields`
- `source`
- `authenticity_checks`
- `tamper_assessment`
- `signals`
- `gemma4_analysis` — present when Ollama/Gemma4 was reachable at scan time; contains:
  - `document_type` — Gemma4's own classification (may differ from heuristic classifier)
  - `document_subtype`
  - `confidence` — float 0–1 for Gemma4's document-type certainty
  - `extracted_fields` — flat key-value dict of all fields Gemma4 found on the document
  - `fraud_signals` — list of `{type, severity, description}` objects
  - `authenticity_assessment` — `{verdict, confidence, reasons}` where verdict is one of `authentic / suspicious / tampered / unknown`
  - `summary` — 2–3 sentence plain-English narrative
  - `engine` — always `"gemma4_ollama"`
  - `model` — Ollama model name used (e.g. `"gemma4:latest"`)
  - `raw_response` — unprocessed Gemma4 output (for debugging)

This lets Bulk Scan and Scan Document save the latest structured facts, forensic evidence, and parser context in one place in `layered_analysis_json` on the scan row.

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
- `document_information`
- `identity_checks`
- `scans`
- `entities`

MinIO reset clears all objects in the configured bucket.