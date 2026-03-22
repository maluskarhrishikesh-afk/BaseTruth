# Roadmap

## Phase 1: MVP In This Repository

- standalone repo and product identity
- scan a document and create explainable reports
- build structured summaries from LiteParse output
- add PDF metadata and signature marker checks
- compare payslips across months

## Phase 2: Stronger Forensic Layer

- integrate `pypdf` deeply for signature inspection
- add `pdfsig` or `qpdf` wrappers
- add image artifact checks and region-level comparison
- add issuer-specific validation packs

## Phase 3: Cross-Document Intelligence

- reconcile identity, amounts, and dates across uploaded files
- detect entity drift across statements, IDs, and forms
- support case-level risk scoring

## Phase 4: Productization

- REST API and job queue
- dashboard and investigator UI
- template registry and fraud knowledge base
- enterprise evidence export and compliance workflows

## Tracking Rules

- every detector must emit explicit evidence
- every score must be explainable from emitted signals
- every domain pack must define what counts as suspicious and why
