# Tooling

## What LiteParse Handles Well

LiteParse is well suited for:

- PDF parsing with layout preserved
- OCR-backed text extraction
- tables and spatial text recovery
- generation of structured raw JSON that other detectors can consume

## What LiteParse Does Not Fully Cover

BaseTruth needs more than parsing. The following capabilities require additional tools or modules:

## Core Tools For MVP

- LiteParse CLI for extraction
- Python standard library for hashing, scoring, reporting, and orchestration
- optional `pypdf` for richer metadata access and form-field inspection

## Recommended Next Tools

- `pypdf` for deeper PDF metadata and form analysis
- `qpdf` or `pdfsig` for stronger signature verification workflows
- `exiftool` for richer document and image metadata inspection
- `opencv-python` for image-level manipulation checks and template alignment
- `pillow` for image preprocessing and region analysis
- `ocrmypdf` when scan normalization becomes important
- `pytesseract` or domain-specific OCR engines where LiteParse is insufficient
- embedding or vector tooling when comparing against known fraud template libraries

## Product-Specific Detector Families

### Banking and payments

- IBAN and account number checksum validation
- merchant statement reconciliation
- invoice and receipt duplicate detection

### Insurance

- claim chronology validation
- signature and stamp region checks
- cross-document claimant identity reconciliation

### Healthcare

- hospital name and provider identifier verification
- inconsistent terminology detection
- treatment date and billing date reconciliation

### Employers and payroll

- payslip and offer-letter template drift analysis
- compensation anomaly detection
- employee identifier reconciliation across months
