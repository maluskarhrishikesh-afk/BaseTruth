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

## Immediate Next Milestones

1. add real cryptographic signature verification through `pdfsig` or `qpdf`
2. add image manipulation detectors and region-level artifact analysis
3. harden connector authentication flows and secret rotation guidance
4. add asynchronous job queue for large-batch scans (Celery or RQ)
5. evidence export package (PDF report with chain-of-custody)
6. FAISS-backed fraud template library for known-bad document recall

## Out Of Scope For Now

- hosted multi-tenant SaaS deployment
- biometric signature verification

## Product Discipline

- every detector emits evidence
- every score is explainable from its signals
- every new industry pack must define required fields, validation rules, and failure modes
