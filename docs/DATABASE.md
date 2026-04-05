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

One row per applicant / person / organisation.

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

Current operational scan record per entity/source document.

| Column | Type | Notes |
|---|---|---|
| `id` | SERIAL | Primary key |
| `entity_id` | FK → `entities.id` | Nullable when no entity could be resolved |
| `source_name` | VARCHAR(500) | Original filename |
| `source_sha256` | VARCHAR(64) | Source hash |
| `document_type` | VARCHAR(100) | Detected/selected document type |
| `truth_score` | INTEGER | Integrity score |
| `risk_level` | VARCHAR(20) | Low/medium/high/etc. |
| `verdict` | TEXT | Human-readable outcome |
| `parse_method` | VARCHAR(100) | Parser used |
| `report_json` | JSONB | Full verification payload |
| `pdf_report` | BYTEA | Optional PDF report bytes |
| `generated_at` | TIMESTAMPTZ | Save timestamp |

**Operational UPSERT key:**
- `(entity_id, source_name, document_type)`

---

### `identity_checks`

Current operational identity-check record per entity and check type.

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

Latest explainability data per entity, screen, and section. This is the dedicated source for the Layered Analysis screen and final explainability report.

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

These tables continue to behave as before:
- `document_information` stores extracted document key fields.
- `cases` stores workflow state.
- `case_notes` stores append-only analyst notes.

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