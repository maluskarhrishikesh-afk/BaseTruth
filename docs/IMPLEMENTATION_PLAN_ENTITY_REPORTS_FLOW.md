# Implementation Plan — Entity Reports Full Flow Revamp

> Historical implementation note. The live approval screen is now **Review Reports**, not **Cases**. Use `docs/FUNCTIONALITY.md` for the current workflow.

**Status:** Draft v2 — awaiting review  
**Based on:** `docs/new_features.txt` (April 17, 2026)  
**Scope:** Document Intelligence → Cases → Reports end-to-end flow for BTR-XXXXXX reports

---

## 1. Summary of Required Changes

| # | Area | Current Behaviour | Required Behaviour |
|---|------|-------------------|-------------------|
| 1 | `reporting/markdown.py` | Only renders per-scan reports | Add `render_entity_report_markdown()` for cross-doc final reports |
| 2 | `reporting/pdf.py` | Only renders per-scan PDFs | Add `render_entity_report_pdf()` that converts the final report to A4 PDF |
| 3 | DB Schema | `entity_reports` has no MinIO column | Add `report_minio_key VARCHAR(500)` column |
| 4 | `store.py` | `save_entity_report()` only writes JSON to DB | Also render PDF, upload to MinIO, and store the key in the row |
| 5 | Document Intelligence | Shows detailed check results + "Saved Reports" inline | Show only a brief status strip; no check details; point to Cases for approval |
| 6 | Scans screen | Has a "📑 Final Reports" tab | Remove this tab entirely |
| 7 | Cases screen | "📑 Entity Reports" is a flat list; no PDF preview | Add 5 filter sub-tabs + inline PDF preview / download before approve buttons |
| 8 | Reports screen | Already gated behind full approval ✅ | Add PDF download button (currently only JSON download) |
| 9 | Database Viewer | `entity_reports` schema lacks `report_minio_key` | Add column to `_TABLE_SCHEMA` |

---

## 2. Detailed Changes

### 2.1 — New: `render_entity_report_markdown()` in `reporting/markdown.py`

**File:** `src/basetruth/reporting/markdown.py`

Add a new function that accepts the `report_json` dict produced by `_run_cross_doc_analysis()` and returns a Markdown string.

**Structure of the Markdown document:**

```
# BaseTruth Final Verification Report — {report_ref}

**Entity:** {entity_name} ({entity_ref})  
**Overall Verdict:** PASS / FAIL  
**Generated:** {timestamp}

---

## Identity Summary

| Field | Value |
|---|---|
| Name | … |
| PAN | … |
| Aadhaar | … |
| Address | … |

---

## Cross-Document Consistency Checks

| Check | Status | Detail |
|---|---|---|
| Name | ✅ PASS / ❌ MISMATCH | … |
| PAN | … | … |
| Aadhaar | … | … |
| Address | … | … |
| Salary | … | … |
| Forensics | … | … |

---

## Document Inventory

| # | File | Type | Forensic Verdict | Forgery Score |
|---|---|---|---|---|
| 1 | Passport-Front.pdf | Passport | 🔴 TAMPERED | 78.4 |
| 2 | Payslip_Oct.pdf | Payslip | 🟢 ORIGINAL | 4.2 |
…

---

## Per-Document Evidence

(One sub-section per document with the extracted identity fields found in it)

### 1. Passport-Front.pdf
- Name: Hrishikesh Maluskar
- Address: …
…

---

## Forensic Summary

N document(s) flagged as TAMPERED: [list]
All M document(s) forensically clean.

---

*This report is generated automatically by BaseTruth. It supports — but does not
replace — a human review and sign-off.*
```

**Input shape** (`report_json` keys used):

| Key | Used for |
|-----|---------|
| `entity_ref` | Header |
| `entity_name` | Header |
| `overall_verdict` | Overall badge |
| `checks.name / pan / aadhaar / address / salary / forensics` | Checks table |
| `per_document_evidence` | Document inventory + per-doc section |
| `scans_reviewed` | Forensic summary counts |

---

### 2.2 — New: `render_entity_report_pdf()` in `reporting/pdf.py`

**File:** `src/basetruth/reporting/pdf.py`

Add a new public function alongside `render_scan_report_pdf()`:

```python
def render_entity_report_pdf(report_json: dict, report_ref: str = "") -> bytes:
    """Render the cross-document final verification report as an A4 PDF.

    Uses the same _ReportPDF class and colour palette as render_scan_report_pdf()
    so the output looks consistent. The function calls render_entity_report_markdown()
    to produce the human-readable structure, then uses fpdf2 to lay it out as PDF.
    """
```

**PDF layout sections (in order):**

1. **Header bar** — `BaseTruth — Final Verification Report` (same `draw_header_bar()` helper)
2. **Overall verdict box** — green `PASS` or red `FAIL` using `verdict_box()`
3. **Entity identity table** — name, entity ref, PAN, Aadhaar, generated date
4. **Consistency checks table** — 6 rows (Name, PAN, Aadhaar, Address, Salary, Forensics); each row shows ✅/❌ status and the detail text; uses `checks_table()`-style layout adapted for string checks instead of signals
5. **Document inventory table** — one row per document: file name, type, forensic verdict badge, forgery score
6. **Per-document evidence** — brief sub-section per doc listing extracted identity fields (name/PAN/address found in that doc)
7. **Footer** — standard disclaimer + page number (same `footer()`)

Implementation uses the existing `_ReportPDF`, `_safe()`, `_wrap_pdf_text()`, colour constants, and fpdf2 — no new dependencies.

---

### 2.3 — DB Schema: Add `report_minio_key` to `EntityReport`

**File:** `src/basetruth/db.py`

#### EntityReport model — add after `report_json`:

```python
# MinIO object key for the generated PDF, e.g.
# "BTR-reports/BT-000001/BTR-000002.pdf"
report_minio_key = Column(String(500), default="")
```

#### `init_db()` migration block — add alongside existing ALTER TABLE statements:

```sql
ALTER TABLE entity_reports
  ADD COLUMN IF NOT EXISTS report_minio_key VARCHAR(500) DEFAULT '';
```

---

### 2.4 — `store.py`: Render PDF + Upload to MinIO in `save_entity_report()`

**File:** `src/basetruth/store.py`

#### Updated `save_entity_report()` steps (after the DB row is created/updated):

1. Call `render_entity_report_pdf(report_json, report_ref)` → `pdf_bytes`
2. Build MinIO key: `f"BTR-reports/{entity_ref}/{report_ref}.pdf"`
3. Call `minio_upload(minio_key, pdf_bytes, "application/pdf")` → `bool`
4. If upload succeeded: set `entity_report.report_minio_key = minio_key` and commit
5. If upload failed: log a warning; `report_minio_key` stays `""` — DB save still succeeds

Return value updated:

```python
return {"entity_ref": entity_ref, "report_ref": report_ref, "minio_key": minio_key}
```

#### Update dict helpers to expose `report_minio_key`:

In the inline dict construction inside `get_entity_reports()` and `list_all_entity_reports()`, add:

```python
"report_minio_key": r.report_minio_key or "",
```

#### Update `db_table_rows()` entity_reports SELECT:

Add `report_minio_key` to the SELECT column list.

---

### 2.5 — Document Intelligence: Clean Up Post-Generation UI

**File:** `src/basetruth/ui/pages/document_intelligence.py`

#### 2.5.1 — Change success message

Replace current message (pointing to Scans screen) with:

> ✅ Report **BTR-XXXXXX** generated, PDF saved to MinIO. Go to **📁 Cases → 📑 Entity Reports** for the 2-level approval.

If MinIO upload failed, show a softer message:

> ✅ Report **BTR-XXXXXX** generated (MinIO unavailable — PDF not stored). Go to **📁 Cases → 📑 Entity Reports** for the 2-level approval.

#### 2.5.2 — Replace `_render_entity_reports_section()` with a status-only strip

Replace the function with `_render_entity_reports_status()` that shows a minimal one-line per report:

```
BTR-000002  ·  ⏳ Pending Review  ·  Generated 2026-04-17
```

No check breakdown. No JSON expander. No approval trail. A brief caption below pointing to Cases.

#### 2.5.3 — Replace confusing "pending report exists" info box

Replace the `st.info("A pending report already exists…")` block with a quiet `st.caption()`:

> ℹ️ Regenerating will refresh the pending report with the latest document data.

---

### 2.6 — Scans Screen: Remove "📑 Final Reports" Tab

**File:** `src/basetruth/ui/pages/scans.py`

1. Change 6-tab declaration back to 5 tabs — remove `"📑 Final Reports"` from the tab list.
2. Remove `tab_reports` from the tuple unpacking.
3. Remove `with tab_reports: _render_entity_reports_approval()` block.
4. Remove the `_render_entity_reports_approval()` function definition.
5. Remove the 5 imports used only by that function:
   `first_level_approve_entity_report`, `first_level_reject_entity_report`,
   `list_all_entity_reports`, `second_level_approve_entity_report`,
   `second_level_reject_entity_report`

---

### 2.7 — Cases Screen: 5 Filter Sub-tabs + PDF Preview Before Approval

**File:** `src/basetruth/ui/pages/cases.py`

#### 2.7.1 — Add 5 sub-tabs inside `_page_entity_reports_tab()`

| Tab | Filter condition |
|-----|-----------------|
| ⏳ Pending | `first_level_approval IS NULL` |
| 🔄 Awaiting 2nd Review | `first_level_approval == "Y"` AND `second_level_approval IS NULL` |
| ✅ Fully Approved | `second_level_approval == "Y"` |
| ❌ Rejected | `first_level_approval == "N"` OR `second_level_approval == "N"` |
| 📋 All | All reports |

Tab labels include counts, e.g. `"⏳ Pending (3)"`.

#### 2.7.2 — PDF preview / download inside each report expander

Inside each report's `st.expander()`, **before** the approve/reject buttons, add:

```python
# Fetch and display the stored PDF from MinIO
minio_key = rpt.get("report_minio_key", "")
if minio_key and _minio_available_cached():
    pdf_bytes = minio_get_object(minio_key)
    if pdf_bytes:
        st.download_button(
            "📄 Download Report PDF",
            data=pdf_bytes,
            file_name=f"{report_ref}.pdf",
            mime="application/pdf",
            key=f"cases_pdf_dl_{report_ref}",
            use_container_width=True,
            type="primary",
        )
        st.caption(f"MinIO key: `{minio_key}`")
    else:
        st.warning("PDF not found in storage — the report may need to be regenerated.")
else:
    st.caption("PDF not stored (MinIO unavailable at generation time).")
```

The check-by-check summary (name/PAN/Aadhaar/etc.) remains below the download button so the reviewer can read the findings without downloading if preferred.

> **Why download-only, not inline render?**  
> Streamlit has no native PDF viewer. The `st.download_button` is the standard pattern used across the application (same as scan PDFs on the Reports screen). Reviewers can open the PDF in their browser/OS viewer with one click.

#### 2.7.3 — New imports needed in `cases.py`

```python
from basetruth.ui.components import (
    ...
    minio_get_object,          # fetch PDF bytes
    _minio_available_cached,   # cached availability check
)
```

---

### 2.8 — Reports Screen: Add PDF Download for Fully-Approved BTR Reports

**File:** `src/basetruth/ui/pages/reports.py`

The existing JSON download button remains. Add a **PDF download button** alongside it when `report_minio_key` is non-empty:

```python
if minio_key and _minio_available_cached():
    pdf_bytes = minio_get_object(minio_key)
    if pdf_bytes:
        rc2.download_button("⬇ PDF", data=pdf_bytes,
                            file_name=f"{report_ref}_{entity_ref}.pdf",
                            mime="application/pdf",
                            key=f"reports_dl_pdf_{report_ref}",
                            use_container_width=True)
```

Change the column split from `[3, 1]` to `[3, 1, 1]` to fit both buttons.

---

### 2.9 — Database Viewer: Add `report_minio_key` to Schema Display

**File:** `src/basetruth/ui/pages/database.py`

Add to `_TABLE_SCHEMA["entity_reports"]`:

```python
{"name": "report_minio_key", "type": "VARCHAR(500)",
 "description": "MinIO object key for the generated PDF (BTR-reports/{entity_ref}/{report_ref}.pdf)"},
```

---

## 3. Files Changed

| File | Change type |
|------|-------------|
| `src/basetruth/reporting/markdown.py` | Add `render_entity_report_markdown()` |
| `src/basetruth/reporting/pdf.py` | Add `render_entity_report_pdf()` |
| `src/basetruth/db.py` | Add `report_minio_key` column to model + migration |
| `src/basetruth/store.py` | Render PDF + upload in `save_entity_report()`; update dict helpers + DB viewer SELECT |
| `src/basetruth/ui/pages/document_intelligence.py` | Simplify post-generation UI |
| `src/basetruth/ui/pages/scans.py` | Remove "📑 Final Reports" tab + function + imports |
| `src/basetruth/ui/pages/cases.py` | Add 5 filter sub-tabs + PDF download in report cards |
| `src/basetruth/ui/pages/reports.py` | Add PDF download button alongside existing JSON download |
| `src/basetruth/ui/pages/database.py` | Add `report_minio_key` to entity_reports schema |

---

## 4. No-change Items

| Item | Reason |
|------|--------|
| `_run_cross_doc_analysis()` | Logic is correct; report payload structure unchanged |
| `first/second_level_approve_entity_report()` in `store.py` | Correct; used by Cases screen |
| `_ReportPDF` class in `pdf.py` | Reused as-is; no modification needed |
| 67 existing tests | Must all pass after each edit |

---

## 5. Data Flow After Changes

```
[Document Intelligence]
  → Analyst clicks "Generate Final Report"
  → _run_cross_doc_analysis() builds report_json dict
  → save_entity_report():
      1. Write report_json to entity_reports table (DB)
      2. render_entity_report_markdown(report_json) → markdown_str
      3. render_entity_report_pdf(report_json, report_ref) → pdf_bytes
      4. minio_upload("BTR-reports/{entity_ref}/{report_ref}.pdf", pdf_bytes)
      5. entity_reports.report_minio_key = minio_key  (update DB row)
  → Screen shows: "✅ BTR-000002 generated, PDF saved. Go to 📁 Cases → 📑 Entity Reports."
  → _render_entity_reports_status() shows: "BTR-000002 · ⏳ Pending Review · 2026-04-17"

[Cases screen → 📑 Entity Reports → ⏳ Pending tab]
  → list_all_entity_reports() loads pending BTR rows
  → Each expander shows:
      [📄 Download Report PDF] button  ← fetched from MinIO via minio_get_object()
      Check summary (name/PAN/Aadhaar/salary/forensics)
      [✅ 1st Approve] [❌ 1st Reject]
  → After 1st approval → moves to 🔄 Awaiting 2nd Review tab
  → After 2nd approval → second_level_approval = "Y" → moves to ✅ Fully Approved

[Reports screen → entity selected]
  → Only BTR reports with second_level_approval = "Y" shown
  → Each row has: [⬇ PDF] [⬇ JSON] download buttons
  → Pending/rejected shown as info/warning — no download
```

---

## 6. Risk & Notes

- **MinIO optional at all stages**: `minio_upload()` returns `bool`. DB save always succeeds; PDF is "best effort". If MinIO is down, `report_minio_key` stays `""` and the Cases screen shows a caption instead of the download button.
- **`init_db()` migration is additive**: `ADD COLUMN IF NOT EXISTS` is safe on existing DBs with data.
- **Scans screen import cleanup**: Verify none of the 5 removed imports are used elsewhere in `scans.py` before deleting.
- **fpdf2 already a dependency**: `render_entity_report_pdf()` uses `fpdf2` which is pinned in `requirements.txt` — no new package needed.
- **Streamlit PDF rendering**: Streamlit has no native inline PDF viewer. `st.download_button` is the correct pattern (same as existing scan PDF downloads).
- **Tests**: Run `python -m pytest tests/ -q --tb=short` after each file edit. All 67 tests must pass throughout.
