# Implementation Plan — Identity Verification / Video KYC Storage Split

## Goal

Implement the new requirement from `docs/new_features.txt` so that:

1. Identity Verification and Video KYC use separate database tables.
2. Identity Verification no longer writes Aadhaar or PAN payloads to `document_extractions`.
3. Video KYC supports permanent address-proof upload, live location capture, address comparison, and live-image capture during liveness.
4. All downstream screens, reports, database tools, and docs are updated to read from the new storage model.

This document is a review-first plan only. No code behavior is changed by this file.

---

## Current State

Today the codebase behaves like this:

- Both Identity Verification and Video KYC are stored in a single table: `identity_checks`.
- `save_identity_check()` in `src/basetruth/store.py` handles both `face_match` and `video_kyc`.
- Identity Verification also writes Aadhaar and PAN payloads into `document_extractions`.
- Video KYC stores only the combined check result plus liveness and face-match fields. It does not store address-proof extraction, browser location, address comparison, or challenge snapshots as first-class DB fields.
- Final report generation currently expects Aadhaar and PAN identity data to exist in `document_extractions`.
- AI Copilot, Database Viewer, and DB query prompts currently only know these tables: `entities`, `scans`, `document_extractions`, `identity_checks`, `entity_reports`.

Because of that, the new requirement is not a small screen-only change. It is a storage-contract change that touches DB schema, save flows, reporting, database viewer, AI Copilot prompts, and product docs.

---

## Target State

After implementation:

- `identity_checks` will store only Identity Verification results.
- `video_kyc_checks` will store only Video KYC results.
- Identity Verification will store Aadhaar and PAN payloads inside `identity_checks` only.
- Video KYC will store identity-proof details, address-proof details, live location payload, address-match result, address-distance calculation, challenge snapshots, and report metadata inside `video_kyc_checks` only.
- `document_extractions` will continue to be used by Scan Document and Bulk Scan, but not by Identity Verification or Video KYC.
- Final report generation and Document Intelligence will read identity data from the dedicated check tables instead of depending on `document_extractions` rows created by Identity Verification.

---

## Scope

### In scope

- DB schema split and migration.
- Store-layer refactor.
- Identity Verification save-flow refactor.
- Video KYC UI/API/session changes for address proof and live location.
- Address comparison and distance calculation pipeline.
- Reporting, Database Viewer, AI Copilot, and documentation updates.
- Unit tests and migration tests.

### Out of scope

- Changing Bulk Scan or Scan Document extraction behavior.
- Replacing MinIO with a different object store.
- Reworking the two-level review flow for `scans` or `entity_reports`.

---

## Proposed Database Design

## Storage Principles

- Store structured verification payloads in PostgreSQL.
- Store raw uploaded images and generated PDFs in MinIO.
- Store the MinIO object keys in PostgreSQL, not raw binary blobs.
- Keep one dedicated table per screen flow so query logic and future reporting stay simple.
- Keep the persistence model simple: one current row per entity per screen flow, updated in place rather than preserving repeated history for now.

---

## Table Plan

### 1. `identity_checks` — Identity Verification only

This table remains, but its role becomes narrower.

| Column | Type | Store | Notes |
|---|---|---|---|
| `id` | SERIAL | PK | Existing table kept |
| `entity_id` | FK → `entities.id` | applicant link | Nullable when unresolved |
| `status` | VARCHAR(20) | pass / fail / inconclusive | As required |
| `cosine_similarity` | FLOAT | raw similarity | Face-match score |
| `display_score` | FLOAT | 0–100 UI score | As required |
| `threshold` | FLOAT | applied threshold | Usually `0.40` |
| `is_match` | BOOLEAN | final face-match boolean | As required |
| `aadhar_dtls` | JSONB | Aadhaar payload | Move from `document_extractions` into this table |
| `pan_dtls` | JSONB | PAN payload | Move from `document_extractions` into this table |
| `selfie_pic` | VARCHAR(500) | MinIO key | Selfie image object key |
| `signature_pic` | VARCHAR(500) | MinIO key | PAN signature crop object key |
| `aadhaar_pic` | VARCHAR(500) | MinIO key | Recommended extra column for the Aadhaar source image |
| `pan_pic` | VARCHAR(500) | MinIO key | Recommended extra column for the PAN source image |
| `report_json` | JSONB | full verification payload | Deterministic checks, layered analysis, evidence |
| `pdf_report` | VARCHAR(500) | MinIO key | MinIO path only |
| `created_at` | TIMESTAMPTZ | audit | Existing/retained |
| `updated_at` | TIMESTAMPTZ | audit | Recommended add if missing |

**Operational behavior**

- One current Identity Verification row per entity.
- A re-save updates the same entity-linked row instead of creating history rows.

### 2. `video_kyc_checks` — Video KYC only

This is a new table.

| Column | Type | Store | Notes |
|---|---|---|---|
| `id` | SERIAL | PK | New table |
| `entity_id` | FK → `entities.id` | applicant link | Nullable when unresolved |
| `status` | VARCHAR(20) | pass / fail / inconclusive | Final check status |
| `cosine_similarity` | FLOAT | raw similarity | Live frame vs Aadhaar/reference image |
| `display_score` | FLOAT | 0–100 UI score | Same presentation logic as today |
| `threshold` | FLOAT | applied threshold | Usually `0.40` |
| `is_match` | BOOLEAN | face-match result | As required |
| `liveness_state` | VARCHAR(50) | latest liveness state | Current / last state |
| `liveness_passed` | BOOLEAN | liveness verdict | Required |
| `identity_dtls` | JSONB | identity-proof payload | Aadhaar / PAN / reference-document extraction details |
| `address_dtls` | JSONB | address-proof payload | Extracted address, normalized address parts, proof doc type |
| `video_kyc_pic` | VARCHAR(500) | MinIO key | Best live-capture image or final frame |
| `address_proof_pic` | VARCHAR(500) | MinIO key | Recommended extra column for uploaded address proof |
| `reference_doc_pic` | VARCHAR(500) | MinIO key | Recommended extra column for the uploaded identity reference doc |
| `isAddressMatch` | VARCHAR(20) | yes / no / inconclusive | Keep field name as required for now |
| `kyc_comments` | VARCHAR(500) | operator/system note | Example: distance or mismatch explanation |
| `current_location_json` | JSONB | browser geolocation payload | Recommended extra field: lat/lon/accuracy/timestamp |
| `current_address_text` | TEXT | reverse-geocoded or composed address | Recommended extra field for human-readable comparison |
| `address_distance_meters` | FLOAT | geo distance | Recommended extra numeric field |
| `challenge_snapshots_json` | JSONB | captured challenge images metadata | Recommended extra field for per-challenge images |
| `report_json` | JSONB | full Video KYC payload | Final result, progress, evidence |
| `pdf_report` | VARCHAR(500) | MinIO key | Path to generated PDF |
| `created_at` | TIMESTAMPTZ | audit | New |
| `updated_at` | TIMESTAMPTZ | audit | Recommended |

**Operational behavior**

- One current Video KYC row per entity.
- A re-save updates the same entity-linked row instead of creating history rows.

### 3. `document_extractions` — no writes from Identity Verification or Video KYC

This table remains for:

- Scan Document
- Bulk Scan
- Any future generic extraction workflow

It must stop receiving rows from:

- Identity Verification Aadhaar save
- Identity Verification PAN save
- Video KYC address-proof upload

That means the `source_screen = 'identity_verification'` usage must be removed from the save path.

---

## What Will Be Stored Where

### Identity Verification

| Data | Table | Column |
|---|---|---|
| Entity link | `entities` | existing entity row |
| Match status and score | `identity_checks` | `status`, `cosine_similarity`, `display_score`, `threshold`, `is_match` |
| Aadhaar payload | `identity_checks` | `aadhar_dtls` |
| PAN payload | `identity_checks` | `pan_dtls` |
| Selfie image path | `identity_checks` | `selfie_pic` |
| PAN signature image path | `identity_checks` | `signature_pic` |
| Aadhaar source image path | `identity_checks` | `aadhaar_pic` |
| PAN source image path | `identity_checks` | `pan_pic` |
| Full evidence JSON | `identity_checks` | `report_json` |
| PDF report path | `identity_checks` | `pdf_report` |

### Video KYC

| Data | Table | Column |
|---|---|---|
| Entity link | `entities` | existing entity row |
| Match status and score | `video_kyc_checks` | `status`, `cosine_similarity`, `display_score`, `threshold`, `is_match` |
| Liveness result | `video_kyc_checks` | `liveness_state`, `liveness_passed` |
| Identity-proof payload | `video_kyc_checks` | `identity_dtls` |
| Address-proof payload | `video_kyc_checks` | `address_dtls` |
| Live geolocation | `video_kyc_checks` | `current_location_json` |
| Human-readable current address | `video_kyc_checks` | `current_address_text` |
| Address match result | `video_kyc_checks` | `isAddressMatch` |
| Address distance | `video_kyc_checks` | `address_distance_meters` |
| KYC comment | `video_kyc_checks` | `kyc_comments` |
| Live capture image | `video_kyc_checks` | `video_kyc_pic` |
| Address proof image path | `video_kyc_checks` | `address_proof_pic` |
| Reference identity document path | `video_kyc_checks` | `reference_doc_pic` |
| Challenge snapshot metadata | `video_kyc_checks` | `challenge_snapshots_json` |
| Full evidence JSON | `video_kyc_checks` | `report_json` |
| PDF report path | `video_kyc_checks` | `pdf_report` |

---

## Implementation Phases

## Phase 1 — DB Schema and Migration

### Changes

- Add new ORM model: `VideoKYCCheck` in `src/basetruth/db.py`.
- Narrow `IdentityCheck` model to Identity Verification fields.
- Add new columns to `identity_checks`.
- Add `video_kyc_checks` table.
- Add schema migration logic in `init_db()`.
- Update `Entity` relationships to include `video_kyc_checks`.

### Migration steps

1. Create `video_kyc_checks` if missing.
2. Add new `identity_checks` columns if missing: `aadhar_dtls`, `pan_dtls`, `selfie_pic`, `signature_pic`, `aadhaar_pic`, `pan_pic`, `updated_at`.
3. Replace `identity_checks.pdf_report` binary storage with MinIO-path string storage.
4. Drop the `check_type` dependency from `identity_checks` logic and do not add `check_type` to `video_kyc_checks`.
5. Copy existing `identity_checks` rows where `check_type = 'video_kyc'` into `video_kyc_checks`.
6. Remove migrated `video_kyc` rows from `identity_checks` once the backfill succeeds.
7. Stop expecting Aadhaar/PAN `document_extractions` rows for identity history.

### Simplification rule

- The new storage model does not keep repeated check history.
- For both tables, the save path updates the current row for the entity instead of inserting a new one each time.

### Migration rule

- Existing Bulk Scan and Scan Document rows in `document_extractions` remain untouched.
- Existing Identity Verification data in `document_extractions` can either:
  - stay as historical legacy rows but no longer be written, or
  - be backfilled into `identity_checks.aadhar_dtls` / `identity_checks.pan_dtls` and then optionally deleted later.

Recommended approach: keep legacy rows for one migration cycle, but stop writing new ones immediately.

---

## Phase 2 — Store-Layer Refactor

### Current anchor

- `save_identity_check()` in `src/basetruth/store.py` currently writes both screen types.

### Planned refactor

- Split `save_identity_check()` into two dedicated persistence paths:
  - `save_identity_verification_check(...)`
  - `save_video_kyc_check(...)`
- Keep a thin compatibility wrapper only during migration, then remove it.
- Remove `_upsert_identity_document_extraction()` calls from Identity Verification saves.
- Add dedicated serializers for:
  - identity payloads
  - address proof payloads
  - live location payloads
  - challenge snapshot payloads
- Implement one-row-per-entity UPSERT behavior for both tables.

### Why this is needed

The current store function mixes two separate workflows and assumes a shared table. That will make future changes harder and will hide bugs during migration.

---

## Phase 3 — Identity Verification Flow Changes

### Changes

- Keep the current UI flow and operator behavior.
- Change only the persistence contract.
- Save Aadhaar details into `identity_checks.aadhar_dtls`.
- Save PAN details into `identity_checks.pan_dtls`.
- Save MinIO keys for selfie, Aadhaar image, PAN image, and PAN signature into dedicated columns.
- Keep layered-analysis and report JSON in `report_json`.

### Code areas to update

- `src/basetruth/ui/pages/identity.py`
- `src/basetruth/store.py`
- `src/basetruth/db.py`
- `src/basetruth/reporting/pdf.py`

### Expected behavior after change

- Clicking Save on Identity Verification will not create any `document_extractions` rows.
- Previous Identity Checks history still works from `identity_checks` alone.

---

## Phase 4 — Video KYC Feature Expansion

## 4A. Start-session UI changes

Add two uploads on the Video KYC start tab:

1. Reference identity proof with face
2. Permanent address proof

### Allowed address-proof documents

- Aadhaar
- Passport

### Address-proof extraction path

- Reuse existing `POST /api/v1/document-extract` pipeline or call the same underlying extractor directly.
- Extract the address-proof JSON before the KYC session is created.
- Store the extracted address payload in session state and pass it into the KYC session creation request.

### Required session-store changes

Add new transient fields to `KYCSession` in `src/basetruth/kyc/session.py`:

- `identity_dtls`
- `address_dtls`
- `reference_doc_filename`
- `address_proof_filename`
- `reference_doc_minio_key` or raw bytes pending save
- `address_proof_minio_key` or raw bytes pending save
- `current_location_json`
- `current_address_text`
- `address_match_result`
- `address_distance_meters`
- `challenge_snapshots`
- `best_live_frame_bytes`

## 4B. Customer-page changes

Add a new action on the customer KYC page:

- `Send Current Location`

### Browser behavior

- Use the browser Geolocation API.
- Send `latitude`, `longitude`, `accuracy`, and `captured_at` to the backend.
- Show success/failure status in the KYC page.

### Backend behavior

- Accept the location payload through:
  - a new WebSocket message type, or
  - a new REST endpoint such as `POST /kyc/sessions/{id}/location`

Recommended approach: use a new REST endpoint for location capture so location and frame-stream logic stay separate.

## 4C. Address matching

### Planned comparison pipeline

1. Extract structured address from uploaded address proof.
2. Normalize address text into comparable parts: line1, line2, locality, city, district, state, pincode.
3. Capture live browser geolocation.
4. Convert live coordinates into a human-readable address if a geocoder is configured.
5. Compare proof address vs live address.
6. Compute geo-distance if both sides can be resolved to coordinates.
7. Save the result into `video_kyc_checks`.

### Matching outputs to store

- exact/partial/inconclusive match status
- matched and mismatched address components
- distance in meters or kilometers
- operator-readable explanation string for `kyc_comments`

### Important design note

The browser Geolocation API returns coordinates, not a postal address. A reverse-geocoding step is required if we want a human-readable current address.

Approved implementation direction:

- Use the latest reliable geocoding provider available in the deployment environment.
- Keep the geocoding integration behind a provider abstraction so the backend can swap providers without UI changes.
- If geocoding is unavailable at runtime, still store raw coordinates in `current_location_json`, mark the address comparison as `inconclusive`, and preserve enough data for later review.

This keeps the product practical while still degrading gracefully.

## 4D. Liveness image capture

### Changes

- Capture and keep more than the final frame.
- Save at least:
  - best frontal live frame
  - one frame per completed challenge where the challenge passed
- Run face match against the Aadhaar/reference identity document using the best frontal frame.

### Retention policy

- Keep only the best frame per completed challenge.
- Do not keep every intermediate frame from the liveness stream.

### Storage

- Save the best live frame MinIO key in `video_kyc_checks.video_kyc_pic`.
- Save challenge snapshot metadata and MinIO keys in `video_kyc_checks.challenge_snapshots_json`.

---

## Phase 5 — Downstream Consumer Updates

These areas will need changes because they currently assume the old storage model.

### Reporting

- `src/basetruth/reporting/final_report_builder.py`
  - currently extracts identity summary from Aadhaar/PAN rows in `document_extractions`
  - must switch to latest `identity_checks` row for identity fields
  - may also consume latest `video_kyc_checks` row for current-address comparison evidence

- `src/basetruth/reporting/pdf.py`
  - must read split-table payloads correctly
  - must handle MinIO-path based report references if `pdf_report` stops being binary

### Database Viewer

- Add `video_kyc_checks` to visible tables.
- Revise `identity_checks` schema display to remove `check_type` and show the new identity-specific JSON and MinIO-path columns.
- Add full schema display for `video_kyc_checks`, including address and snapshot fields.
- Update table schema cards.
- Update CRUD metadata, filters, reset logic, and safe truncation ordering.
- Update any row-detail / JSON-expander logic so `pdf_report` is treated as a MinIO object key, not binary content.
- Update any copy, labels, or helper text that still says Video KYC data is in `identity_checks`.

### AI Copilot / DB Query Layer

- `src/basetruth/integrations/qna_prompts.md`
- `src/basetruth/integrations/db_query.py`
- `src/basetruth/ui/pages/gemma_chat.py`

These currently only know about `identity_checks`. They must be updated so:

- `identity_checks` means Identity Verification only
- `video_kyc_checks` means Video KYC only
- Identity data is not assumed to live in `document_extractions`

### Store helpers

- `get_entity_identity_checks(...)` should either:
  - merge rows from both tables into one unified response for UI/report compatibility, or
  - be split into two functions plus one compatibility aggregator.

Recommended approach: keep one compatibility aggregator for screens that want a combined history.

Because history is intentionally simple for now, that compatibility aggregator should return at most one current Identity Verification row and one current Video KYC row per entity.

---

## Phase 6 — Documentation Updates

These docs must be updated in the same implementation change:

- `docs/ARCHITECTURE.md`
- `docs/FUNCTIONALITY.md`
- `docs/IDENTITY_VERIFICATION.md`
- `docs/DATABASE.md`
- `docs/TESTING.md`

### Specific doc changes

- Identity Verification no longer writes to `document_extractions`.
- Video KYC gets permanent address-proof upload and current-location capture.
- Database guide gains `video_kyc_checks` and revises `identity_checks`.
- Final report / Document Intelligence data sources are updated.

---

## Test Plan

## Unit tests

- schema / migration helper tests for split-table creation and backfill
- store-layer tests for `save_identity_verification_check()`
- store-layer tests for `save_video_kyc_check()`
- address normalization and match-result tests
- distance-calculation tests
- session-store tests for new KYC session fields
- API tests for location submission endpoint or message handler

## Integration tests

- Video KYC start flow with address-proof extraction mocked
- Identity Verification save flow asserting no `document_extractions` write
- final report builder reading identity data from dedicated check tables

## Manual verification

- Identity Verification save and reload
- Video KYC flow with address proof + customer location share
- Database Viewer visibility of both tables
- AI Copilot queries for face match and video KYC data

---

## Rollout Order

1. Add DB schema and migration code.
2. Add new persistence functions.
3. Switch Identity Verification save flow.
4. Add Video KYC address-proof and location capture.
5. Update downstream readers.
6. Update docs.
7. Run focused tests, then full suite.

This order minimizes breakage because downstream readers can temporarily use a compatibility path while the new tables are introduced.

---

## Review Questions

These points are now confirmed for implementation:

1. `pdf_report` is stored only as a MinIO path string.
2. `check_type` is not required in either `identity_checks` or `video_kyc_checks`.
3. Keep storage simple: one current row per entity per screen flow.
4. Allowed address-proof documents for Video KYC are Aadhaar or Passport.
5. Use the latest reliable geocoding strategy/provider available in the deployment environment, with graceful degradation when unavailable.
6. Keep only the best frame per challenge.

---

## Recommended Implementation Decision

If we follow the requirement exactly while keeping the migration risk low, the best path is:

- keep `identity_checks` for Identity Verification only
- add `video_kyc_checks` for Video KYC only
- remove `check_type` from both tables
- stop writing identity rows into `document_extractions`
- keep `document_extractions` for Scan Document and Bulk Scan only
- move PDF references to MinIO-path storage
- keep one current row per entity per flow instead of full history
- allow Aadhaar or Passport as Video KYC address-proof inputs
- keep best-frame-per-challenge snapshots only
- update Database Viewer alongside the schema change
- add a compatibility read layer so reports and history screens do not break during migration

That gives a clean storage model without forcing a risky all-at-once rewrite.