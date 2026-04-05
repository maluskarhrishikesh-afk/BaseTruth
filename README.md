# BaseTruth

BaseTruth is a standalone document integrity product for explainable fraud detection across banking, payroll, insurance, healthcare, and enterprise operations.

## What It Does

| Layer | Capability |
|---|---|
| Parsing | LiteParse-backed document parsing with PDF metadata fallback |
| Structuring | Normalised JSON summaries with key-field extraction |
| Forensics | Heuristic tamper scoring (editor mismatch, metadata anomalies, arithmetic checks) |
| Validators | Domain-specific validation packs (payroll, banking, invoice, insurance, healthcare) |
| Cross-document | Cross-month payslip comparison — identity drift, net-pay drop, period gaps |
| API | FastAPI REST layer (`/api/v1/`) |
| UI | Professional Streamlit operator UI — Dashboard, Scan, Bulk, Cases, Reports, Datasources, Settings |
| Reporting | Markdown and JSON verification reports for auditability |
| Case management | Workflow status, disposition, assignee, labels, analyst notes |
| Connectors | S3, SharePoint, Google Drive datasource registration and sync |

## Quick Start

```powershell
cd c:\Hrishikesh\BaseTruth

# Install (UI, API, and PDF support)
pip install -e ".[ui,api,pdf]"

# Run tests
python -m pytest

# Run Semgrep security and code-quality checks
semgrep scan --config .semgrep.yml src tests

# Scan one document
python -m basetruth.cli scan --input C:\path\to\document.pdf

# Compare payslips across months
python -m basetruth.cli compare-payslips --input-dir C:\path\to\payslips

# Start the Streamlit UI
streamlit run src\basetruth\ui\app.py

# Start the REST API  (requires: pip install -e ".[api]")
uvicorn basetruth.api:app --host 0.0.0.0 --port 8502
```

Windows launcher executables are also provided:

| Program | Action |
|---|---|
| `start.exe` | Starts the BaseTruth UI and records PID to `.runtime/` |
| `stop.exe` | Stops the running BaseTruth UI using the recorded PID |

## REST API

When the `api` extra is installed the REST server is available at `http://localhost:8502`.

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/v1/health` | Product info and artifact root |
| `POST` | `/api/v1/scan` | Scan a document by path |
| `POST` | `/api/v1/scan/upload` | Scan a multipart file upload |
| `GET` | `/api/v1/reports` | List all reports |
| `GET` | `/api/v1/cases` | List all cases |
| `GET` | `/api/v1/cases/{key}` | Get case detail |
| `PATCH` | `/api/v1/cases/{key}` | Update case workflow |

Interactive docs: `http://localhost:8502/api/docs`

## Domain Validation Packs

`src/basetruth/analysis/validators.py` ships a validation pack for each supported industry. Each pack runs a set of arithmetic and format rules and emits a `ValidationSignal` for every check. Signals are wired directly into `evaluate_tamper_risk()` so they contribute to the overall truth score.

| Pack | Key Rules |
|---|---|
| Payroll | gross ≥ net, UAN 12-digit, paid days 0–31, basic ≥ 20 % of gross |
| Banking | opening + credits − debits = closing balance |
| Invoice | subtotal + tax = amount due |
| Insurance | required fields: policy_number, insured_name, insurer_name |
| Healthcare | required fields: patient_name, provider_name, visit_date |

## Repository Layout

```
src/basetruth/
  analysis/       structured.py, tamper.py, payslip.py, validators.py
  integrations/   liteparse.py, pdf.py
  reporting/      markdown.py
  ui/             app.py
  api.py          FastAPI REST layer
  service.py      BaseTruthService orchestrator
  datasources.py  DatasourceRegistry and connector adapters
  models.py       shared data models
  cli.py          CLI entry point
tests/            21 unit tests
docs/             ARCHITECTURE, PRODUCT_VISION, ROADMAP, TRACKER, TOOLING
artifacts/        scan outputs (JSON, Markdown)
```

## Product Direction

BaseTruth is designed as a multi-layer forensic pipeline rather than a single OCR script. Every signal emitted has a name, severity, score contribution, and explanation — nothing is a black box.

Upcoming: cryptographic signature verification, image-region artefact detection, asynchronous job queue, chain-of-custody evidence export.
- scan the synced snapshot and preserve provenance

That keeps client source systems read-only while giving BaseTruth deterministic evidence trails.

Supported datasource types now include:

- local folders
- manifest files
- Amazon S3 prefixes
- Google Drive folders
- SharePoint document libraries

Connector setup is handled in the Datasources tab. BaseTruth stores connector configuration with the datasource record and uses the underlying platform credentials at sync time:

- S3: AWS profile or standard AWS credential environment variables
- Google Drive: local application-default credentials or a service-account JSON path
- SharePoint: a Microsoft Graph bearer token supplied through a configurable environment variable

## Investigation Workflow

Cases are grouped automatically from verification reports and now support persisted workflow state:

- investigator assignment
- priority and status tracking
- disposition decisions
- labels for routing and triage
- time-stamped analyst notes

## Current MVP Scope

- payslip-friendly structured extraction
- PDF metadata and signature marker checks
- anomaly scoring with transparent evidence
- cross-month payslip drift detection

## Roadmap

See `docs/ROADMAP.md` for the phased build plan.

## Developer Checks

BaseTruth includes a local Semgrep configuration at `.semgrep.yml` for high-signal Python security and code-quality checks.

```powershell
semgrep scan --config .semgrep.yml src tests
```
