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

## Immediate Next Milestones

1. add real cryptographic signature verification through `pdfsig` or `qpdf`
2. add image manipulation detectors and region-level artifact analysis
3. add issuer validation packs for banks, insurers, and payroll providers
4. add API and asynchronous job orchestration

## Out Of Scope For This MVP

- hosted multi-tenant SaaS deployment
- dashboard UI
- investigator case management
- biometric signature verification
- FAISS template library and fraud template recall

## Product Discipline

- every detector emits evidence
- every score is explainable
- every new industry pack must define required fields, validation rules, and failure modes
