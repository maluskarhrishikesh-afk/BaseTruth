# BaseTruth

BaseTruth is a standalone document integrity product focused on explainable fraud detection across industries such as banking, payments, insurance, healthcare, and enterprise operations.

The first MVP in this repository provides:

- LiteParse-backed document parsing when the local CLI is installed
- structured JSON summaries that normalize key fields from parsed documents
- PDF metadata and digital-signature marker inspection
- heuristic fraud and tamper scoring with inspectable signals
- cross-month payslip comparison to surface anomalies automatically
- Markdown and JSON verification reports for auditability
- a Streamlit UI for single scans, bulk scans, datasource sync, and report review
- case-oriented investigation view and report index
- persisted case workflow management with assignee, labels, disposition, and analyst notes
- first enterprise datasource connectors for S3, SharePoint, and Google Drive

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
cd c:\Hrishikesh\BaseTruth
python -m pytest
python -m basetruth.cli scan --input C:\path\to\document.pdf
python -m basetruth.cli compare-payslips --input-dir C:\path\to\payslips
streamlit run src\basetruth\ui\app.py
```

Windows launchers are also supported:

- `start.exe` starts the BaseTruth UI and writes runtime state to `.runtime/`
- `stop.exe` stops the running BaseTruth UI process tree using the recorded PID

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

## UI And Datasource Model

BaseTruth now includes a lightweight UI designed around four operator workflows:

- single document scan
- bulk document scan
- datasource registration and sync
- report review
- case review

Instead of asking clients to move files into one shared folder manually, the better operating model is:

- register a datasource such as a shared folder or manifest file
- sync it into a BaseTruth-managed snapshot workspace
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
