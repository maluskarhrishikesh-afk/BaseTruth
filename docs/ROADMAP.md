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
- asynchronous job queue for large-batch scans (pending)
- evidence export package with chain-of-custody PDF (pending)
- enterprise compliance workflow templates (pending)

## Tracking Rules

- every detector must emit explicit evidence
- every score must be explainable from emitted signals
- every domain pack must define what counts as suspicious and why
