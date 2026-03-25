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
- [x] `docs/Mortgage_Fraud.md` — full mortgage industry knowledge base:
  - Indian market structure, underwriting parameters, fraud typology (16 fraud types)
  - Cross-document consistency checks table
  - Document-specific forensic checks (payslip, bank statement, employment letter, Form 16)
  - Risk scoring architecture, rules engine (hard + soft rules), ML features
  - Regulatory references (RBI, NHB, PMLA, CERSAI, RERA)
  - Implementation status tracker
- [x] 21 new mortgage-specific tests — 42 total passing

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
7. Evidence export package (PDF report with chain-of-custody)

## Out Of Scope For Now

- hosted multi-tenant SaaS deployment
- biometric signature verification

## Product Discipline

- every detector emits evidence
- every score is explainable from its signals
- every new industry pack must define required fields, validation rules, and failure modes
