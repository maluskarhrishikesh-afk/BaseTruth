# Tracker

## Completed

### Phase 1 — MVP
- [x] standalone BaseTruth repo created
- [x] scan pipeline implemented
- [x] LiteParse integration implemented
- [x] structured summary generation implemented
- [x] PDF metadata inspection implemented
- [x] digital-signature marker detection implemented
- [x] heuristic tamper scoring implemented
- [x] cross-month payslip comparison implemented
- [x] JSON and Markdown reporting implemented
- [x] unit tests passing (21 tests)
- [x] operator UI implemented (Streamlit)
- [x] datasource registry and snapshot sync implemented
- [x] case-oriented UI and report index implemented
- [x] first enterprise datasource connector layer implemented (S3, Google Drive, SharePoint)

### Phase 2 — Stronger Forensic Layer
- [x] issuer-specific domain validation packs (`src/basetruth/analysis/validators.py`)
  - Payroll: gross≥net, UAN format, paid days range, basic proportion
  - Banking: balance arithmetic (opening+credits−debits=closing)
  - Invoice: subtotal+tax=amount_due
  - Insurance / Healthcare: required-fields checks
- [x] metadata date consistency signal (ModDate should not precede CreationDate)
- [x] domain validator signals wired into `evaluate_tamper_risk()`

### Phase 3 — Cross-Document Intelligence
- [x] identity drift detection across payslip series (employee_id, employee_name)
- [x] net pay drop spike detection (>30 % single-month drop)
- [x] period gap detection (missing months in the series)

### Phase 4 — Productization
- [x] professional sidebar-navigation Streamlit UI (Dashboard, Scan, Bulk, Cases, Reports, Datasources, Settings)
- [x] FastAPI REST layer (`src/basetruth/api.py`) with scan, reports, and case endpoints
- [x] enriched Markdown report renderer (signal icons, structured tables)
- [x] PDF audit report generation (FPDF2) — stored in PostgreSQL `scans.pdf_report` (LargeBinary)
- [x] `GET /api/v1/scans/{id}/report.pdf` — auditor PDF download from database
- [x] `GET /api/v1/entities` — entity registry search (name / PAN / Aadhaar / email / phone)
- [x] `GET /api/v1/entities/{ref}` — entity detail with scan history
- [x] `GET /api/v1/entities/{ref}/scans` — all scans for one entity (full JSON + signals)
- [x] `GET /api/v1/scans/recent` — monitoring feed of latest scans
- [x] `GET /api/v1/db/stats` — entity / scan / high-risk counts

### Phase 5 — SOLID Architecture and Multi-Industry Coverage
- [x] validation packs refactored into individual industry modules (`src/basetruth/analysis/packs/`)
  - `base.py`        : BaseValidationPack, ValidationSignal (shared protocol)
  - `payroll.py`     : PayrollValidationPack (payslip / HR operations)
  - `banking.py`     : BankingValidationPack (bank statements; IFSC check, overdraft flag)
  - `payments.py`    : PaymentsValidationPack (UPI / NEFT / fintech receipts)
  - `insurance.py`   : InsuranceValidationPack (policy docs, claim letters)
  - `healthcare.py`  : HealthcareValidationPack (hospital bills, discharge summaries)
  - `invoice.py`     : InvoiceValidationPack (GST invoices; GSTIN format check)
  - `compliance.py`  : ComplianceValidationPack (audit reports, KYC/AML certificates)
  - `__init__.py`    : central REGISTRY dict; get_pack(), validate_document() API
- [x] `validators.py` reduced to a backwards-compatible import shim
- [x] document-type detection in `structured.py` extended to insurance, healthcare,
  payment_receipt, and compliance keywords
- [x] synthetic sample documents created at `C:/Hrishikesh/Documents/`
  (Banking, Payments, Insurance, Healthcare, Payroll, Compliance -- valid + tampered pairs)
- [x] LiteParse graceful fallback for image-only PDFs (Aadhaar, PAN)
- [x] ImageMagick-free pypdf text-extraction fallback with `parse_fallback` flag
- [x] coded pushed to GitHub: https://github.com/maluskarhrishikesh-afk/BaseTruth

### Phase 6 — Mortgage Fraud Module
- [x] `MortgageValidationPack` implemented (`src/basetruth/analysis/packs/mortgage.py`)
  - Payslip: `basic_within_gross`, `net_pay_arithmetic`, `pf_rate_validity`, `pt_slab_validity`, `tds_plausibility`, `hra_proportion`
  - Employment Letter: `employer_cin_format`, `employer_cin_present`, `cin_age_vs_join_date`, `ctc_monthly_gross_consistency`
  - Form 16: `tan_format_validity`, `form16_tds_plausibility`
  - Utility Bill: `utility_amount_plausibility`
  - Bank Statement (via routing): `circular_funds_detection`, `duplicate_txn_reference`, `salary_credit_regularity`
- [x] `BankingValidationPack` extended with three new mortgage-relevant checks:
  - `circular_funds_detection` — large round-trip debit+credit same day (≥ ₹1 lakh)
  - `duplicate_txn_reference` — same TXN reference appearing twice
  - `salary_credit_regularity` — salary credited more than once in same calendar month
- [x] `structured.py` extended with mortgage document type patterns:
  - `employment_letter`, `form16`, `utility_bill`, `gift_letter`, `property_agreement`, `mortgage`
- [x] `packs/__init__.py` extended — mortgage pack registered for all sub-types
- [x] Synthetic mortgage corpus: 50 cases × 8–9 PDFs = 426 PDFs with `income_inflated`, `employer_fake`, `circular_funds`, `backdated_employment` labels
- [x] `data/mortgage_docs/metadata.json` + `labels.csv` (ML-ready)
- [x] `docs/Mortgage_Fraud.md` — full mortgage industry knowledge base (fraud typology, cross-document checks, rules engine, ML features)
- [x] 21 new mortgage-specific tests — 42 total passing

### Phase 7 — Image Document Scanning and Visual Forensics
- [x] `src/basetruth/analysis/image_forensics.py` — complete multi-layer image forensics module:
  - `_extract_exif_pillow()` — EXIF via Pillow (always available)
  - `_extract_exif_exifread()` — richer tag set via exifread library (optional)
  - `extract_image_metadata()` — merged EXIF + image dimensions
  - `detect_suspicious_tool()` — matches 25+ suspicious software names including AI generators
  - `run_ela()` — Error Level Analysis via Pillow + NumPy; returns (ela_score, high_error_frac)
  - `run_noise_analysis()` — Laplacian CV across 16×16 blocks via OpenCV (optional)
  - `analyse_image()` — combined entry point returning signals + forensics summary
- [x] `integrations/pdf.py` extended with image helpers:
  - `is_image_file()` — extension check
  - `ocr_image_directly()` — pytesseract OCR on raw images (no Poppler dependency)
  - `extract_image_file_metadata()` — dimensions, format, SHA-256 for raw images
- [x] `service.scan_document()` — new `elif is_image_file(path):` branch for `.jpg/.png/.tiff` etc.
  - OCR → structured summary → image forensics → tamper assessment → PDF report → DB persist
- [x] `evaluate_tamper_risk()` signature updated: accepts optional `image_forensics` dict
  - image forensics signals merged into signal list before scoring
  - `image_forensics_summary` key added to return value when forensics ran
- [x] All 42 existing tests still pass

### Phase 8 — UI Reliability and Correctness

- [x] DB stats strip removed from all pages (was consuming vertical space on every screen)
- [x] Dashboard metrics section compacted to a single 6-column row for cleaner layout
- [x] Cases page: applicants now grouped by entity with expandable cards per entity
- [x] Reports page: search / filter bar added (name, PAN, email, BT-reference)
- [x] Reports page: MinIO / object-storage PDF fallback — if `pdf_report` not in DB, attempt to fetch from configured object store before showing "not available"
- [x] Records page: entity field force-overwrite fixed — editing a field via the Records form now persists correctly and no longer silently ignores updates when the value was previously set
- [x] Records page: empty placeholder `<div>` (between the How-to expander and the search inputs) removed — was appearing as a blank white box occupying vertical space
- [x] Cases page: text filter added — filter by entity name, BT-reference, case key, or document type; live filtering across all tabs (Needs Review / Resolved / Auto-Approved)
- [x] Cases page: auto-approve guard strengthened — was auto-approving newly uploaded LOW-risk scans even when the entity already had an open HIGH/MEDIUM-risk case in PostgreSQL; fixed by checking `case_exists_in_db()` before auto-approve fires
- [x] All pages: latest records always shown first — entities ordered by `id DESC`, scans ordered by `generated_at DESC`, cases ordered by `(needs_review, risk_level, -latest_scan_timestamp)`
- [x] `store.py — update_entity()`: now correctly stamps `updated_at` on each entity edit
- [x] `reporting/pdf.py`: critical crash fixed — `pdf.multi_cell(0, ...)` after `pdf.cell(5, ...)` produced zero/negative remaining width, causing ALL PDF reports to fail silently; fixed to `set_x(l_margin)` and explicit `page_width` multi_cell call

## Immediate Next Milestones

1. Cross-document reconciliation engine (`src/basetruth/analysis/cross_doc.py`)
   - payslip net ↔ bank salary credit income reconciliation
   - employer name consistency across payslip + employment letter
   - join date plausibility: letter join date ≤ first payslip period
2. Case bundle grouping (`src/basetruth/analysis/case_bundle.py`)
   - group all mortgage documents for one applicant
   - produce aggregate case-level fraud risk score
3. FAISS-backed fraud template fingerprint library
4. MCA21 CIN registry API stub (`src/basetruth/integrations/mca21.py`)
5. CERSAI charge lookup stub (`src/basetruth/integrations/cersai.py`)
6. Add real cryptographic signature verification (`pdfsig` / `qpdf`)
7. Template matching for image documents (known-good passport/PAN/Aadhaar layout comparison)
8. GAN / AI-generation artefact detector using pretrained CNN (deepfake document detection)
9. Perceptual hash drift check across document series (ImageHash)

## Out Of Scope For Now

- hosted multi-tenant SaaS deployment
- biometric signature verification

## Product Discipline

- every detector emits evidence
- every score is explainable from its signals
- every new industry pack must define required fields, validation rules, and failure modes
