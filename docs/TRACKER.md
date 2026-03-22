# Tracker

## Current MVP Status

- standalone BaseTruth repo created
- scan pipeline implemented
- LiteParse integration implemented
- structured summary generation implemented
- PDF metadata inspection implemented
- digital-signature marker detection implemented
- heuristic tamper scoring implemented
- cross-month payslip comparison implemented
- JSON and Markdown reporting implemented
- unit tests passing
- operator UI implemented
- datasource registry and snapshot sync implemented
- case-oriented UI and report index implemented
- first enterprise datasource connector layer implemented

## Immediate Next Milestones

1. add real cryptographic signature verification through `pdfsig` or `qpdf`
2. add image manipulation detectors and region-level artifact analysis
3. add issuer validation packs for banks, insurers, and payroll providers
4. add API and asynchronous job orchestration
5. add enterprise datasource connectors such as S3, SharePoint, Drive, and database-backed manifests
6. harden connector authentication flows and secret rotation guidance

## Out Of Scope For This MVP

- hosted multi-tenant SaaS deployment
- investigator case management
- biometric signature verification
- FAISS template library and fraud template recall

## Product Discipline

- every detector emits evidence
- every score is explainable
- every new industry pack must define required fields, validation rules, and failure modes
