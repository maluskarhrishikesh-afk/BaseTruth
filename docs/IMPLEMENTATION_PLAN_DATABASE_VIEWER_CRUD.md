# Implementation Plan — Database Viewer CRUD Operations

**Status:** Draft v1 — awaiting review  
**Date:** April 27, 2026  
**Scope:** Add development-focused row-level CRUD operations to the Database Viewer screen

---

## 1. Summary

The current Database Viewer already supports:

- viewing row counts for core PostgreSQL tables
- browsing recent rows table-by-table
- inspecting a full row as JSON
- truncating one table or resetting the whole database

The missing part is row-level editing. For development and testing, the screen should also let an operator:

- create a row in a selected table
- edit a row in a selected table
- delete a single row in a selected table
- optionally duplicate an existing row as a starting point for test data

This should be implemented as a **development-only admin tool**, not as an unrestricted production data editor.

---

## 2. Goals

1. Make it easy to seed and tweak data without opening psql or writing one-off scripts.
2. Keep CRUD actions local to the existing Database Viewer screen so the workflow stays discoverable.
3. Preserve BaseTruth audit safety rules: every write must show clear success or visible failure.
4. Avoid schema drift by deriving editable column metadata from the real database model or a single store-layer source of truth.
5. Keep dangerous operations explicit and harder to trigger than simple browsing.

---

## 3. Non-Goals

This feature should **not** try to become a full database admin console.

Out of scope for the first version:

- arbitrary SQL execution
- schema changes from the UI
- bulk CSV import/export
- editing MinIO objects directly from this screen
- multi-row batch updates
- cross-table transaction builders

---

## 4. Current Behaviour

### Screen today

File: `src/basetruth/ui/pages/database.py`

The screen currently provides:

- cached PostgreSQL availability checks
- row counts using `db_table_counts()`
- row browsing using `db_table_rows(table, limit)`
- a schema reference card using `_TABLE_SCHEMA`
- full-row JSON inspection
- whole-database reset
- per-table truncate

### Store helpers today

File: `src/basetruth/store.py`

The store layer currently exposes:

- `db_table_counts()`
- `db_table_rows()`
- `reset_db()`
- `truncate_table()`

There are **no row-level insert, update, delete helpers** for the Database Viewer today.

### Known risk in the current viewer

The viewer already has one drift risk: schema definitions shown in the UI and columns fetched by `db_table_rows()` can get out of sync. The CRUD design should fix that instead of adding more duplicated table knowledge.

---

## 5. Proposed Feature Set

## 5.1 New CRUD workspace inside PostgreSQL tab

Add a second section below the existing table browser:

- `Browse` view stays as-is
- new `Create Row` action
- new `Edit Row` action
- new `Delete Row` action
- optional `Duplicate Row` action for development convenience

Suggested layout:

1. Table selector and row limit
2. Existing dataframe and row JSON inspector
3. New action bar:
   - `➕ Create`
   - `✏️ Edit selected row`
   - `🧬 Duplicate selected row`
   - `🗑️ Delete selected row`
4. Context-sensitive form panel below the action bar

---

## 5.2 Table capability matrix

First version should support only the same allowlisted tables already shown in Database Viewer:

| Table | Create | Edit | Delete | Notes |
|---|---|---|---|---|
| `entities` | Yes | Yes | Yes | Main dev seeding table |
| `scans` | Yes | Yes | Yes | Useful for approval and forensic test states |
| `document_extractions` | Yes | Yes | Yes | JSON editing is important here |
| `identity_checks` | Yes | Yes | Yes | Useful for face-match and KYC test data |
| `entity_reports` | Yes | Yes | Yes | Mainly for approval-state and report JSON testing |

The allowlist should remain explicit. No arbitrary table names should be accepted from the UI.

---

## 5.3 Column behaviour rules

The form should treat columns in one of four ways.

### A. System-managed and read-only

Examples:

- `id`
- auto timestamps like `created_at`, `updated_at`, `generated_at`

Behaviour:

- shown in edit mode as read-only informational fields when useful
- never editable in create mode unless there is a very strong technical reason

### B. Editable scalar fields

Examples:

- `name`
- `email`
- `document_type`
- `verdict`
- `check_type`

Behaviour:

- use `st.text_input`, `st.text_area`, `st.number_input`, `st.checkbox`, or `st.selectbox` depending on type and known enum-like values

### C. Foreign-key fields

Examples:

- `entity_id`
- `scan_id`

Behaviour:

- show a selectbox populated from the parent table where practical
- display both ID and a human-readable label when available
- allow clearing nullable foreign keys

### D. JSON fields

Examples:

- `layered_analysis_json`
- `extracted_data`
- `report_json`
- `report_json` / `details_json`-style payloads

Behaviour:

- use a large text area prefilled with formatted JSON
- validate JSON before save
- show a visible parse error and block save if the payload is invalid

---

## 5.4 Row selection behaviour

The current row inspector already lets the user choose one row. That selection should become the anchor for edit, duplicate, and delete.

Expected behaviour:

- when no row is selected, `Edit`, `Duplicate`, and `Delete` stay disabled
- when a row is selected, actions apply only to that row
- selected row summary should remain visible above the form so the user knows exactly what is being changed

Suggested summary format:

`Editing row id=42 in document_extractions · file_name=PAN_1.jpg · entity_id=7`

---

## 5.5 Create row behaviour

Expected flow:

1. User selects a table
2. User clicks `Create`
3. Empty form appears with table-specific fields
4. User fills values
5. User clicks `Save new row`
6. UI validates input
7. Store layer inserts the row
8. Screen shows success or visible error
9. Cached table rows and counts are refreshed

Development-friendly enhancements:

- `Start from selected row` toggle to prefill a create form from an existing row
- `Pretty format JSON` helper button for JSON fields

---

## 5.6 Edit row behaviour

Expected flow:

1. User selects a row from the current table
2. User clicks `Edit selected row`
3. Form appears prefilled with current values
4. User changes one or more fields
5. User clicks `Save changes`
6. Store layer updates only editable columns
7. UI shows success or visible error
8. Dataframe and JSON inspector refresh immediately

Important rule:

- updates should target the row by primary key only
- the primary key itself must not be editable

---

## 5.7 Duplicate row behaviour

This is optional but strongly useful in development.

Expected flow:

1. User selects a row
2. User clicks `Duplicate selected row`
3. Create form opens with the selected row prefilled
4. System-managed fields are cleared automatically:
   - `id`
   - timestamps
5. User changes only the few fields they care about
6. User clicks `Save new row`

This avoids repetitive manual entry when testing approval states, report JSON variants, or multiple extractions for one entity.

---

## 5.8 Delete row behaviour

Expected flow:

1. User selects a row
2. User clicks `Delete selected row`
3. A confirmation panel appears with the row summary
4. User types a strict confirmation word such as `DELETE`
5. User clicks `Confirm Delete`
6. Store layer deletes that single row
7. UI shows success or visible error
8. Cached table rows and counts refresh

Delete should be harder to trigger than create or edit.

---

## 5.9 Refresh behaviour after writes

After every successful create, edit, duplicate, or delete:

- clear `_cached_db_table_counts()`
- clear `_cached_db_table_rows()`
- rerun the page

This keeps the screen consistent with existing refresh behaviour.

---

## 6. Guardrails and Safety Rules

## 6.1 Development-only feature flag

Because this is mainly for development, CRUD should be hidden unless explicitly enabled.

Recommended options:

- environment variable: `BASETRUTH_ENABLE_DB_VIEWER_CRUD=true`
- or a Streamlit/admin setting that resolves to the same boolean

Recommended default:

- disabled by default

When disabled, the page should remain read-only exactly as it behaves today.

---

## 6.2 Visible write outcomes

Every write operation must show one of these outcomes:

- success message
- validation error message
- database failure message

No write failure should be silent.

---

## 6.3 Cached availability checks only in render path

The page must continue using:

- `_db_available_cached()`
- `_minio_available_cached()`

No live DB availability probe should be introduced into the render path.

---

## 6.4 Strong allowlist for writable tables

Store-layer CRUD helpers must reject:

- unknown table names
- non-allowlisted table names
- attempts to write to columns not declared editable for that table

This keeps the feature narrow and predictable.

---

## 6.5 JSON validation before save

For any JSON column, the UI should:

- parse the JSON client-side in Python before calling the store helper
- show the parse error inline
- not attempt the DB write until JSON is valid

---

## 6.6 Foreign-key validation

Before saving:

- if `entity_id` is provided, verify that entity exists
- if `scan_id` is provided, verify that scan exists

If invalid, show a clear error instead of relying on a raw DB exception.

---

## 7. Technical Design

## 7.1 Store-layer helpers to add

File: `src/basetruth/store.py`

Add a small CRUD helper set specifically for Database Viewer:

```python
def db_viewer_table_meta(table: str) -> dict:
    ...

def db_viewer_get_row(table: str, row_id: int) -> dict | None:
    ...

def db_viewer_create_row(table: str, payload: dict) -> tuple[bool, str, dict | None]:
    ...

def db_viewer_update_row(table: str, row_id: int, payload: dict) -> tuple[bool, str, dict | None]:
    ...

def db_viewer_delete_row(table: str, row_id: int) -> tuple[bool, str]:
    ...
```

Why store-layer helpers are preferred:

- keeps SQL and validation out of the Streamlit page
- keeps allowlists and editable-column rules in one place
- makes the feature testable without the UI
- fixes the existing schema-drift problem if metadata and row fetches share the same source

---

## 7.2 Metadata source of truth

The current `_TABLE_SCHEMA` dict in `database.py` is useful for display, but it should not remain the only schema description once CRUD is added.

Recommended direction:

- move table metadata into a single store-layer constant or helper
- include, per table:
  - display label
  - primary key
  - editable columns
  - read-only columns
  - JSON columns
  - foreign-key columns
  - optional enum-like choices

Then use the same metadata for:

- schema reference card
- form generation
- row validation
- insert/update allowlisting

---

## 7.3 UI changes in Database Viewer

File: `src/basetruth/ui/pages/database.py`

Add a new CRUD section inside the PostgreSQL tab.

Suggested helper functions inside the page module:

```python
def _render_db_viewer_action_bar(...):
    ...

def _render_db_viewer_form(...):
    ...

def _render_db_viewer_delete_confirm(...):
    ...
```

The page should stay structured like this:

1. metrics
2. table browser
3. row inspector
4. CRUD actions
5. form / confirmation area
6. existing danger zone

This preserves the screen's current mental model.

---

## 7.4 Form generation strategy

The form should be generated from table metadata rather than hand-written per table.

Mapping suggestion:

| Column type / role | UI control |
|---|---|
| text / varchar | `st.text_input` |
| long text | `st.text_area` |
| integer / float | `st.number_input` |
| boolean | `st.checkbox` |
| timestamp read-only | text display only |
| JSONB | large `st.text_area` with JSON validation |
| FK | `st.selectbox` + optional blank value |

This keeps the implementation small while still covering the needed tables.

---

## 7.5 Delete implementation strategy

Delete should use a primary-key-based delete against the allowlisted table.

Example rule:

- only rows with integer primary key `id` are supported in v1

That is already true for the five current Database Viewer tables, so this keeps the first version simple.

---

## 8. Table-Specific Notes

## 8.1 `entities`

Useful editable fields:

- `name` or split name fields, depending on current real model
- `pan_number`
- `aadhaar_uid` or Aadhaar-equivalent field
- `email`
- `phone`

Recommended behaviour:

- if `entity_ref` is system-generated in the model/save path, do not expose it for editing in v1

---

## 8.2 `scans`

Useful editable fields in development:

- `entity_id`
- `source_name`
- `source_sha256`
- `document_type`
- approval fields
- `layered_analysis_json`

Recommended caution:

- if some fields are derived elsewhere in normal app flows, still allow editing for dev use, but label them clearly as advanced fields

---

## 8.3 `document_extractions`

This table is one of the main reasons CRUD is valuable.

Useful editable fields:

- `entity_id`
- `scan_id`
- `file_name`
- `document_type`
- `source_screen`
- `extracted_data`

Recommended enhancement:

- preset `source_screen` choices: `bulk_scan`, `scan_document`, `identity_verification`

---

## 8.4 `identity_checks`

Useful editable fields:

- `entity_id`
- `check_type`
- `status`
- `cosine_similarity`
- `display_score`
- `threshold`
- `is_match`
- `liveness_state`
- `liveness_passed`
- `verdict`
- `doc_filename`
- `selfie_filename`
- `report_json`

Recommended enhancement:

- preset `check_type` choices: `face_match`, `video_kyc`

---

## 8.5 `entity_reports`

Useful editable fields:

- `entity_id`
- `report_ref`
- `report_json`
- approval fields
- `report_minio_key` if present in the real model

Recommended caution:

- if there is logic elsewhere that treats approved reports as immutable, note that this screen is a dev override tool

---

## 9. User Experience Details

## 9.1 Suggested top-of-section warning

When CRUD is enabled, show:

> Development Mode: Database Viewer row editing is enabled. These tools can create inconsistent data if used carelessly.

This communicates intent without blocking development.

---

## 9.2 Success messages

Examples:

- `Row created in document_extractions with id=154.`
- `Row 154 updated successfully.`
- `Row 154 deleted from document_extractions.`

---

## 9.3 Error messages

Examples:

- `Save failed: extracted_data is not valid JSON.`
- `Save failed: entity_id 9999 does not exist.`
- `Delete failed: row 42 no longer exists.`

Messages should be short and concrete.

---

## 10. Implementation Steps

## Phase 1 — Foundation

1. Add shared Database Viewer table metadata in `store.py`.
2. Update the existing row-fetch logic to use that metadata where practical.
3. Add row-level store helpers for get/create/update/delete.
4. Add validation helpers for JSON and foreign keys.

## Phase 2 — UI

1. Add CRUD action bar to the PostgreSQL tab.
2. Add generated create/edit form.
3. Add duplicate-row flow.
4. Add strict delete confirmation flow.
5. Refresh caches and rerun after successful writes.

## Phase 3 — Polish

1. Add enum-style selectboxes for known columns.
2. Improve row summary labels.
3. Add a collapsible advanced section for large JSON fields.
4. Add better empty-state and validation copy.

---

## 11. Testing Plan

### Unit / store-layer tests

Add tests for:

- allowlist rejection for unknown tables
- create row success per table
- update row success per table
- delete row success per table
- JSON validation failures
- foreign-key validation failures

### UI behaviour checks

Manual checks:

- CRUD controls hidden when feature flag is off
- CRUD controls visible when feature flag is on
- create updates counts and row list immediately
- edit updates JSON inspector immediately
- delete removes row immediately
- failure states show visible messages

### Regression checks

- existing browse-only flow still works
- refresh button still clears caches
- danger zone truncate/reset still works with spinner and visible outcome

### Required project test run after implementation

Run:

```bash
python -m pytest tests/ -q --tb=short
```

---

## 12. Documentation Updates Needed When Implemented

When the code is actually built, update these files in the same change:

- `docs/FUNCTIONALITY.md` — Database Viewer section
- `docs/DATABASE.md` — note dev CRUD tooling if it affects table conventions
- `docs/ARCHITECTURE.md` — only if the screen responsibilities materially expand

---

## 13. Recommended First Version

The safest first implementation is:

1. keep the existing read-only browser unchanged
2. add a feature-flagged CRUD panel below it
3. support the five current allowlisted tables only
4. allow create, edit, delete, and duplicate
5. validate JSON and foreign keys before every write
6. keep system-managed fields read-only

This gives a large development productivity win without turning the Database Viewer into an unrestricted admin console.

---

## 14. Open Questions

1. Should CRUD be enabled only in local development, or also in staging?
2. Should `entity_ref` and `report_ref` remain system-generated only, or be editable for test setup?
3. Should delete be allowed for approved `entity_reports`, or should those rows be read-only even in dev mode?
4. Should the first version include a `Duplicate Row` action, or can that wait for phase 2?

---

## 15. Recommendation

Yes, this feature is worth adding.

For BaseTruth's current development workflow, row-level CRUD in Database Viewer will remove a lot of friction when testing approval states, identity checks, extracted document payloads, and report flows. The right way to build it is as a feature-flagged, allowlisted, metadata-driven extension of the existing screen, with store-layer validation and clear success/error messages on every write.