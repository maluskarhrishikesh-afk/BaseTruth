# BaseTruth

BaseTruth is a standalone document integrity product focused on explainable fraud detection across industries such as banking, payments, insurance, healthcare, and enterprise operations.

The first MVP in this repository provides:

- LiteParse-backed document parsing when the local CLI is installed
- structured JSON summaries that normalize key fields from parsed documents
- PDF metadata and digital-signature marker inspection
- heuristic fraud and tamper scoring with inspectable signals
- cross-month payslip comparison to surface anomalies automatically
- Markdown and JSON verification reports for auditability

## Product Direction

BaseTruth is designed as a multi-layer forensic pipeline rather than a single OCR script.

Core layers:

- document ingestion and parsing
- metadata and signature verification
- semantic and arithmetic validation
- cross-document reconciliation
- explainable reporting and evidence trails

## Repository Layout

- `src/basetruth/` application code
- `tests/` unit tests
- `docs/` product and architecture documents

## Quick Start

```powershell
cd c:\Hrishikesh\OctaMind\BaseTruth
python -m pytest
python -m basetruth.cli scan --input C:\path\to\document.pdf
python -m basetruth.cli compare-payslips --input-dir C:\path\to\payslips
```

## Outputs

By default BaseTruth writes artifacts under `artifacts/`:

- raw LiteParse output when available
- structured summary JSON
- verification report JSON
- verification report Markdown
- cross-month comparison JSON and Markdown

## LiteParse and Other Tools

LiteParse is a strong fit for document parsing and layout-preserving extraction, but it is not enough for a full fraud product on its own.

BaseTruth treats LiteParse as the parsing layer. Additional tools you will want as the product matures are documented in `docs/TOOLING.md`.

## Current MVP Scope

- payslip-friendly structured extraction
- PDF metadata and signature marker checks
- anomaly scoring with transparent evidence
- cross-month payslip drift detection

## Roadmap

See `docs/ROADMAP.md` for the phased build plan.
