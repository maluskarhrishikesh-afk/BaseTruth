# Roadmap

## Phase 1: MVP In This Repository

- standalone repo and product identity
- scan a document and create explainable reports
- build structured summaries from LiteParse output
- add PDF metadata and signature marker checks
- compare payslips across months

## Phase 2: Stronger Forensic Layer ✅

- ~~integrate `pypdf` deeply for signature inspection~~ (stub present)
- `pdfsig` / `qpdf` cryptographic wrappers (pending)
- image artifact checks and region-level comparison (pending)
- **[DONE]** issuer-specific domain validation packs (payroll, banking, invoice, insurance, healthcare)
- **[DONE]** metadata date consistency signal

## Phase 3: Cross-Document Intelligence ✅ (partial)

- **[DONE]** identity drift detection across payslip series
- **[DONE]** net pay drop spike and period gap detection
- reconcile identity and amounts across different document types (pending)
- FAISS-backed fraud template recall (pending)

## Phase 4: Productization ✅

- **[DONE]** FastAPI REST layer (`/api/v1/` endpoints)
- **[DONE]** professional sidebar-navigation operator UI
- **[DONE]** enriched Markdown report renderer
- **[DONE]** PDF audit report generation (FPDF2) — stored in PostgreSQL `scans.pdf_report`
- **[DONE]** `GET /api/v1/scans/{id}/report.pdf` — auditor PDF download endpoint
- **[DONE]** `GET /api/v1/entities` / `/{ref}` / `/{ref}/scans` — entity registry REST API
- **[DONE]** `GET /api/v1/scans/recent` + `GET /api/v1/db/stats` — monitoring endpoints
- asynchronous job queue for large-batch scans (pending)
- evidence export package with chain-of-custody PDF (pending)
- enterprise compliance workflow templates (pending)

## Phase 6: Mortgage Fraud Module

- **[DONE]** `MortgageValidationPack` — payslip arithmetic, PF/PT/TDS slab checks, HRA proportion, CIN format + age validation, CTC vs monthly gross consistency
- **[DONE]** `BankingValidationPack` extended — circular funds detection, duplicate transaction reference flag, salary credit regularity check
- **[DONE]** Mortgage document type detection in `structured.py` (employment_letter, form16, utility_bill, gift_letter, property_agreement, mortgage)
- **[DONE]** Mortgage document type aliases registered in `packs/__init__.py`
- **[DONE]** Synthetic mortgage corpus — 50 cases, 426 PDFs, labels.csv (`scripts/generate_mortgage_docs.py`)
- **[DONE]** Comprehensive `docs/Mortgage_Fraud.md` — full Indian mortgage industry knowledge base, fraud typology, cross-document checks, rules engine, ML features, implementation tracker
- **[DONE]** 21 new mortgage-specific tests (42 total)
- cross-document reconciliation engine (`src/basetruth/analysis/cross_doc.py`) — pending
- case bundle grouping + multi-document scan (`src/basetruth/analysis/case_bundle.py`) — pending
- FAISS-backed fraud template fingerprint library — pending
- MCA21 CIN registry API stub — pending
- CERSAI charge lookup stub — pending
- Graph ring detection engine — pending

## Phase 8: UI Reliability and Correctness ✅

- **[DONE]** Dashboard: DB stats strip removed; metrics compacted to single 6-column row
- **[DONE]** Cases: entities grouped into expandable cards per applicant
- **[DONE]** Reports: search / filter bar (name, PAN, email, BT-reference)
- **[DONE]** Reports: MinIO PDF fallback when `scans.pdf_report` is NULL
- **[DONE]** Records: entity fields force-overwrite fix; `update_entity` now sets `updated_at`
- **[DONE]** Records: stray empty `<div>` placeholder removed from search area
- **[DONE]** Cases: text filter (entity name, reference, case key, document type)
- **[DONE]** Cases: auto-approve guard — checks `case_exists_in_db()` before clearing a LOW-risk scan when a prior HIGH/MEDIUM case exists in PostgreSQL
- **[DONE]** All screens: latest records shown first (entities by id DESC, scans by generated_at DESC, cases by risk + recency)
- **[DONE]** PDF reports: `multi_cell` crash fixed — all reports now generate without crashes

## Phase 7: Image Document Scanning and Visual Forensics ✅

- **[DONE]** Raw image file scanning pipeline (`.jpg`, `.jpeg`, `.png`, `.tiff`, `.bmp`, `.webp`) in `scan_document()`
- **[DONE]** Direct OCR on image files via pytesseract (no Poppler dependency)
- **[DONE]** `src/basetruth/analysis/image_forensics.py` — multi-layer image forensics:
  - EXIF suspicious tool detection (Photoshop, GIMP, Canva, Stable Diffusion, Midjourney, etc.)
  - Missing camera EXIF signal
  - Timestamp inconsistency detection
  - Error Level Analysis (ELA) — quantitative editing-artefact score
  - Noise consistency analysis (OpenCV Laplacian CV across image blocks)
- **[DONE]** Image forensics signals wired into `evaluate_tamper_risk()` scoring
- **[DONE]** `image_forensics_summary` attached to every tamper assessment for image inputs
- Template matching against known-good originals (pending — requires template library)
- GAN / AI-generation artefact detectors (pending — deep learning models)
- Perceptual hash drift check across document series (pending)

- every detector must emit explicit evidence
- every score must be explainable from emitted signals
- every domain pack must define what counts as suspicious and why
- every new mortgage check must reference a rule ID from `docs/Mortgage_Fraud.md`
