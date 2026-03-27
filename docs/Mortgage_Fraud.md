# Mortgage Fraud: Industry Knowledge Base & BaseTruth Product Roadmap

> **Living document.** Updated continuously as BaseTruth implementation progresses.
> Every section maps to implemented or planned platform features.
> See [TRACKER.md](TRACKER.md) for build status and [ROADMAP.md](ROADMAP.md) for milestone planning.

---

## 1. Indian Mortgage Industry — Context

### 1.1 Market Structure

India's home-loan market is dominated by:

| Category | Key Players | Loan Book Size (approx.) |
|---|---|---|
| Public Sector Banks | SBI, Bank of Baroda, Punjab National Bank | ₹10–15 lakh crore combined |
| Private Banks | HDFC Bank, ICICI Bank, Axis Bank, Kotak | ₹5–8 lakh crore combined |
| Housing Finance Companies (HFCs) | HDFC Ltd, LIC HFL, PNB Housing | ₹3–6 lakh crore combined |
| NBFCs | Bajaj Housing Finance, Indiabulls | ₹1–2 lakh crore combined |

**RBI** regulates banks; **NHB (National Housing Bank)** regulates HFCs.
**CERSAI (Central Registry of Securitisation Asset Reconstruction and Security Interest)** is the central repository for mortgage charge registration — critical for detecting duplicate mortgages on the same property.

### 1.2 Standard Home Loan Process (Origination)

```
Borrower Inquiry
    → Pre-Qualification (income, CIBIL score check)
    → Formal Application + Document Submission
    → Document Verification (income, identity, property)
    → Credit Appraisal (LTV, FOIR, CIBIL)
    → Legal & Technical Due Diligence on Property
    → Sanction Letter
    → Disbursement (staged for under-construction; lump-sum for resale)
    → Post-Disbursement Monitoring
```

### 1.3 Key Ratios & Underwriting Parameters

| Parameter | Typical Threshold | What Fraud Manipulates |
|---|---|---|
| **LTV (Loan-to-Value)** | Max 75–90% depending on loan value | Inflated property appraisals raise LTV head-room |
| **FOIR (Fixed Obligation to Income Ratio)** | Max 50–55% | Inflated income lowers FOIR artificially |
| **CIBIL Score** | Min 700–750 | Synthetic identities, straw buyers |
| **Minimum Net Monthly Income** | ₹20,000–₹25,000 | Fake/inflated payslips |
| **Vintage of Employment** | Min 1–2 years continuous | Backdated employment letters |
| **Bank Statement Vintage** | Min 6 months | Manufactured/fabricated statements |

---

## 2. Fraud Typology

### 2.1 Fraud for Housing (Retail / Borrower Fraud)

The borrower lies to secure a loan they cannot afford or would not otherwise qualify for.

#### 2.1.1 Income Fraud

**What happens:**
- Salary slips are photoshopped to inflate gross salary.
- Income from informal employment is fabricated entirely.
- Multiple income streams are double-counted.

**Red flags:**
- Declared monthly income does not match salary credit amounts in bank statements.
- Net-pay on payslip significantly higher than daily-balance average suggests.
- YTD (year-to-date) cumulative on the last payslip of the year doesn't match Form 16.
- Gross-to-net deduction ratio is atypical (too low = deductions removed digitally).
- Font inconsistencies within a single payslip (especially in amount fields).

**BaseTruth checks (implemented / planned):**
- ✅ `gross_gte_net_pay` — arithmetic validation in `PayrollValidationPack`
- ✅ `basic_minimum_proportion` — basic salary ≥ 20% of gross
- 🔲 `payslip_bank_income_reconciliation` — payslip net ≈ bank salary credit ± 5%
- 🔲 `ytd_form16_reconciliation` — cumulative YTD on payslip matches Form 16 gross

#### 2.1.2 Employment Fraud

**What happens:**
- Employment letter issued by a shell or non-existent company.
- Actual employment period is shorter; letter backdates the join date.
- CIN (Company Identification Number) is absent, fabricated, or belongs to a different company.
- HR email domains are free-mail providers (gmail.com, yahoo.com) instead of corporate domains.

**Red flags:**
- CIN not findable in MCA21 (Ministry of Corporate Affairs) registry.
- Join date on employment letter predates the earliest payslip by more than 30 days with no explanation.
- Company name on employment letter differs from company name on payslips.
- Employer address on letter matches a residential address.

**BaseTruth checks (implemented / planned):**
- 🔲 `employer_cin_format` — CIN must match `[LU][0-9]{5}[A-Z]{2}[0-9]{4}[A-Z]{3}[0-9]{6}` pattern
- 🔲 `employment_join_date_vs_payslip` — join date ≤ earliest payslip period
- 🔲 `employer_name_consistency` — employer name identical across payslips and employment letter

#### 2.1.3 Bank Statement Fraud

**What happens:**
- PDF bank statements are edited digitally to add fictitious credits.
- Balances are inflated by deleting debits or adding phantom credits.
- Circular transactions: large amounts moved out and back in same day to create an illusion of cash flow.
- Statement period is truncated to hide low-balance months.

**Red flags:**
- Opening + credits − debits ≠ closing balance (arithmetic fails).
- Running balance column contains value jumps inconsistent with adjacent transactions.
- Large round-trip debit+credit on same date with vague descriptions ("NEFT/SELF TRANSFER").
- Statement PDF producer metadata shows editing tools (Photoshop, Canva, LibreOffice Draw).
- Identical transaction reference numbers appear more than once.

**BaseTruth checks (implemented / planned):**
- ✅ `balance_arithmetic` — opening + credits − debits = closing (BankingValidationPack)
- ✅ `ifsc_format` — IFSC code format validation
- 🔲 `circular_funds_detection` — large same-day matching debit+credit pair
- 🔲 `running_balance_consistency` — each row balance = prior balance ± transaction
- 🔲 `duplicate_reference_number` — flag repeated transaction reference numbers

#### 2.1.4 Identity Fraud

**What happens:**
- PAN card / Aadhaar details belong to another person (stolen identity).
- Synthetic identity: combination of a real PAN with a fake name and address.
- Photo substitution on scanned ID documents.

**Red flags:**
- PAN format invalid (`AAAAA9999A` — 5 letters, 4 digits, 1 letter).
- Name on PAN card differs from name on payslip / bank statement.
- Date of birth on Aadhaar inconsistent with age field in loan application.
- Aadhaar number fails Verhoeff checksum (used by NPCI / UIDAI).

**BaseTruth checks (implemented / planned):**
- 🔲 `pan_format_validation` — 10-character PAN regex check
- 🔲 `name_consistency_across_documents` — name fuzzy-match: payslip ↔ bank ↔ ID
- 🔲 `dob_age_consistency` — age derived from DOB must match declared age ± 1 year

#### 2.1.5 Gift Letter Fraud

**What happens:**
- Down payment is actually a loan from a relative or third party but is declared as a gift.
- Gift letter claims no repayment is required, but simultaneous EMI-like outgoing transfers appear in bank statements.
- Donor does not have sufficient funds to make the gift (no matching debit in donor's bank).

**Red flags:**
- Gift amount does not appear as an incoming credit in the borrower's bank statement near the letter date.
- Post-gift-letter, regular outgoing transfers to the donor appear in statements.
- Donor's PAN is not provided.

**BaseTruth checks (planned):**
- 🔲 `gift_credit_in_statement` — gift amount should appear as a credit within ±7 days of letter date
- 🔲 `post_gift_regular_repayment` — outgoing transfers to same party after gift date = undisclosed loan signal

### 2.2 Fraud for Profit (Professional / Organised Fraud)

More sophisticated, involves multiple colluding parties. Higher financial impact.

#### 2.2.1 Appraisal Fraud / Property Valuation Inflation

**What happens:**
- Appraiser issues a certificate valuing a property far above market rates, enabling a larger loan.
- Developer inflates sale price in agreement; actual cash consideration is lower (agreement-to-sale ≠ actual transaction).
- Same property appraised multiple times with rising values in quick succession.

**Red flags:**
- Property sale consideration > comparable recent sales in the area by > 20–30%.
- Appraiser appears in multiple high-LTV applications within a short period.
- Sale agreement date and registration date are far apart with large price differential.

**BaseTruth checks (planned):**
- 🔲 `property_price_deviation` — compare against median market rate for locality/property type
- 🔲 `appraiser_recurrence_flag` — same appraiser across multiple suspicious applications

#### 2.2.2 Broker / DSA Fraud

**What happens:**
- Direct Selling Agent (DSA) fabricates or inflates borrower documents to earn commission.
- Identical document templates used across multiple unrelated borrowers (same font, same template unique only in numbers changed).
- DSA submits the same borrower to multiple banks simultaneously (multiple disbursements).

**Red flags:**
- Applicants using same phone number or email across applications (same DSA handling).
- Multiple applications from same city with identical employer/address formats.
- Same DSA code appearing across high-default-rate applications.

**BaseTruth checks (planned):**
- 🔲 `shared_contact_across_applications` — same phone/email across multiple cases
- 🔲 `template_fingerprint_match` — FAISS pixel-hash comparison to known fraud templates

#### 2.2.3 Duplicate / Second Mortgage Fraud

**What happens:**
- Property already mortgaged with one lender; borrower suppresses this and mortgages again with another.
- CERSAI registration was not done (legal obligation in India from 2012).

**Red flags:**
- Property not found in CERSAI charge registry.
- Title chain shows existing encumbrance in a prior deed of hypothecation.

**BaseTruth checks (planned):**
- 🔲 `cersai_charge_check_stub` — API stub for CERSAI property search

#### 2.2.4 Builder / Phantom Project Fraud

**What happens:**
- Loans taken against under-construction projects that are non-existent or stalled.
- Multiple borrowers show loans against same unit (same flat number in different applications).
- Builder's RERA registration is absent or lapsed.

**Red flags:**
- RERA registration number absent in sale agreement.
- Same unit number / floor / block appearing in multiple independent loan applications.
- Builder's legal entity (company) is newly incorporated (< 2 years CIN age).

**BaseTruth checks (planned):**
- 🔲 `rera_registration_format_check` — RERA number format validation per state
- 🔲 `unit_duplication_across_cases` — same property coordinates across multiple cases

### 2.3 Advanced & Emerging Fraud

#### 2.3.1 Synthetic Identity Fraud

Real PAN + fabricated address + real mobile → passes many automated checks. Surfaces only when cross-referencing income patterns with credit bureau history.

#### 2.3.2 Deepfake Documents

AI-generated payslips and bank statements generated from publicly available templates.

**Detection approach:**
- Hash known legitimate templates (pixel-hash + font fingerprint) in a FAISS library.
- Flag documents that match no known legitimate issuer template.
- Check PDF creation tool metadata — synthetically generated documents typically have unusual producer strings.

#### 2.3.3 Mule Account / Circular Funds

**Third-party bank accounts used to generate apparent salary credits:**

Funds flow: Fraudster → Friend's account → Credit to borrower's account as "salary". This inflates the apparent income trajectory.

**Circular funds in a single account:**
Large debit and matching credit same-day creates an artificial high-balance snapshot.

#### 2.3.4 Straw Buyer Fraud

A financially unqualified borrower uses a creditworthy friend/relative as the primary applicant. The actual occupant/payer does not appear on the application.

**Detection:**
- Behavioral inconsistency: declared profession vs. typical income profile.
- Property address ≠ residence address with no plausible explanation.

---

## 3. Cross-Document Consistency Checks (Core of BaseTruth)

These are the most powerful fraud signals — they require comparing multiple documents in the same case bundle.

| Check | Documents Compared | Signal |
|---|---|---|
| Income reconciliation | Payslip net pay ↔ Bank statement salary credit | Payslip inflated if credit < 90% of net |
| YTD reconciliation | Final payslip YTD ↔ Form 16 gross salary | If >5% difference, likely tampered |
| Employer consistency | Payslip employer ↔ Employment letter employer | Any name mismatch = flag |
| Join date plausibility | Employment letter join date ↔ Payslip series start | Letter claims earlier than payslips support |
| Gift vs bank credit | Gift letter amount + date ↔ Bank statement credit | Gift must appear as inward credit |
| Utility address | Utility bill address ↔ Loan application address | Address fraud = occupancy fraud indicator |
| Property buyer name | Property agreement purchaser ↔ Loan applicant name | Must match exactly |
| PAN consistency | PAN on payslip ↔ PAN on Form 16 ↔ PAN on ID | Any mismatch = identity/synthetic ID risk |
| Balance trajectory | 6-month bank statement closing balance trend | Sharp single-month spike before application |

---

## 4. Document-Specific Forensic Checks

### 4.1 Payslip Forensics

| Check | Method | Risk if Failed |
|---|---|---|
| Gross arithmetic | Basic + HRA + Allowances = Gross | High — likely digit manipulation |
| Deductions arithmetic | PF + PT + TDS + Other = Total Deductions | High |
| Net pay arithmetic | Gross − Total Deductions = Net Pay | Critical — core tamper signal |
| PF ceiling | PF computed on ≤ ₹15,000 basic (statutory ceiling) | Medium |
| HRA proportion | HRA typically 20–50% of basic | Medium |
| PT slab validity | Professional Tax follows state-specific slabs (max ₹200/month) | Low |
| TDS plausibility | TDS > 0 for annual gross > ₹5 lakh | Medium |
| Font consistency | Uniform font across all amount fields | High — visual tamper signal |
| PDF modification date | ModDate ≥ CreationDate | High |
| Producer field | No Photoshop / Canva / GIMP in producer | High |

### 4.2 Bank Statement Forensics

| Check | Method | Risk if Failed |
|---|---|---|
| Balance identity | Opening + Credits − Debits = Closing | Critical |
| Running balance per row | Balance[n] = Balance[n-1] ± Txn[n] | Critical |
| Circular funds | Large round-trip debit+credit same day, vague description | High |
| Duplicate reference numbers | Same TXN ID appearing twice | High |
| Velocity spike | Single-month salary credit 2× typical month | Medium |
| Salary credit regularity | Salary should credit once per month consistently | Medium |
| Minimum balance maintenance | Account not falling below minimum for savings | Low |
| IFSC format | 11-char IFSC code per RBI specification | Low |

### 4.3 Employment Letter Forensics

| Check | Method | Risk if Failed |
|---|---|---|
| CIN format | `[LU][0-9]{5}[A-Z]{2}[0-9]{4}[A-Z]{3}[0-9]{6}` | High — fake company |
| Company age vs join date | Incorporation date (from CIN year) must predate join date | High |
| HR email domain | Must match company's registered domain (not Gmail/Yahoo) | Medium |
| Salary on letter vs payslip | CTC on letter × (1/12) should ≈ gross on payslip | High |
| Designation consistency | Same designation on letter and payslips | Medium |

### 4.4 Form 16 Forensics

| Check | Method | Risk if Failed |
|---|---|---|
| TAN format | 10-char: `[A-Z]{4}[0-9]{5}[A-Z]` | Medium |
| TDS arithmetic | Tax on income slab must equal declared TDS | High |
| Gross on Form 16 vs payslips | Form 16 gross ÷ 12 ≈ monthly payslip gross | High |
| Standard deduction | ₹50,000 standard deduction (current AY) | Low |
| Employer in Part A matches payslip | Employer name + TAN consistent | Medium |

---

## 5. Risk Scoring Architecture

### 5.1 Signal Taxonomy

```
Severity     Weight    Description
----------   ------    -------------------------------------------
critical       40      Arithmetic failure — certain manipulation
high           25      Strong pattern mismatch
medium         10      Suspicious but explainable
low             5      Informational anomaly only
info            0      Check passed — no risk contribution
```

### 5.2 Truth Score Calculation

```
risk_sum = Σ(signal.score for all failed signals)
base_score = 100 − risk_sum
truth_score = clamp(base_score, 0, 100)

risk_level:
  truth_score >= 75  →  LOW
  truth_score >= 50  →  MEDIUM
  truth_score  < 50  →  HIGH
```

### 5.3 Cross-Document Risk Multipliers (Planned)

A document that fails an internal check scores risk from that single document.
When multiple documents in the same case bundle send conflicting signals, the
risk is amplified:

| Conflict Pattern | Multiplier |
|---|---|
| Payslip income ≠ bank credits | +20 to risk |
| Employer name mismatch across 2+ documents | +15 to risk |
| PAN mismatch across documents | +30 to risk |
| Join date inconsistency | +15 to risk |

---

## 6. Fraud Rules Engine (Implemented & Planned)

### 6.1 Hard Rules (Binary — fail = flag immediately)

| Rule ID | Condition | Disposition |
|---|---|---|
| `HR-001` | Payslip Gross − Deductions ≠ Net Pay (>1% tolerance) | HIGH risk |
| `HR-002` | Bank statement: Opening + Credits − Debits ≠ Closing | HIGH risk |
| `HR-003` | CIN field absent on employment letter | HIGH risk |
| `HR-004` | PAN format invalid | HIGH risk |
| `HR-005` | PDF modified by Photoshop/Canva/GIMP/Inkscape | HIGH risk |
| `HR-006` | PDF ModDate < CreationDate | HIGH risk |

### 6.2 Soft Rules (Probabilistic — scored and weighted)

| Rule ID | Condition | Score Contribution |
|---|---|---|
| `SR-001` | Payslip net pay > bank salary credit by > 10% | 30 |
| `SR-002` | Circular debit+credit pair on same day, same amount ≥ ₹1 lakh | 40 |
| `SR-003` | Employment letter join date > earliest payslip period start | 25 |
| `SR-004` | HRA > 50% of basic | 10 |
| `SR-005` | TDS = 0 but annual gross > ₹7 lakh | 20 |
| `SR-006` | PF computed at rate > 12% of basic | 15 |
| `SR-007` | Same phone number or email across 2+ case bundles | 35 |
| `SR-008` | Salary credit velocity spike: one month > 2× median of series | 20 |

---

## 7. ML Feature Engineering

### 7.1 Per-Document Features

| Feature | Source Document | Purpose |
|---|---|---|
| `gross_to_net_ratio` | Payslip | Deduction rate — unusual = tampered |
| `basic_to_gross_ratio` | Payslip | Should be 35–50% for Indian payslips |
| `tds_rate` | Payslip / Form 16 | Derive implied tax slab |
| `num_debit_credits_per_month` | Bank statement | Low count = manufactured statement |
| `max_balance_to_avg_balance` | Bank statement | Spike detection |
| `days_since_last_salary_credit` | Bank statement | Irregular salary = contract / informal |
| `pdf_producer_risk_flag` | All PDFs | Binary: known editing tool in metadata |
| `has_digital_signature` | All PDFs | Signed documents are lower fraud risk |

### 7.2 Cross-Document Features

| Feature | Source Documents | Purpose |
|---|---|---|
| `income_delta_payslip_vs_bank` | Payslip + Bank | Core inflation signal |
| `employer_name_match_score` | Payslip + Employment Letter | Fuzzy string similarity 0–1 |
| `join_date_consistency_flag` | Employment Letter + Payslip | Binary: join < first payslip |
| `ytd_vs_form16_delta` | Payslip + Form 16 | ≥5% mismatch = tampered |
| `gift_credit_present_flag` | Gift Letter + Bank | Binary: credit found within 7 days |

### 7.3 Case-Level Features (for Fraud Ring Detection)

| Feature | Purpose |
|---|---|
| `shared_phone_across_cases` | DSA fraud ring signal |
| `shared_employer_across_cases` | Shell company employer signal |
| `appraiser_recurrence_rate` | Appraisal fraud ring |
| `template_similarity_to_known_fraud` | FAISS-based template match |

---

## 8. BaseTruth Product Architecture — Mortgage Module

### 8.1 New Components to Build

```
src/basetruth/analysis/packs/mortgage.py    ← MortgageValidationPack
src/basetruth/analysis/cross_doc.py         ← Cross-document reconciliation engine
src/basetruth/analysis/case_bundle.py       ← Case bundle: group + reconcile documents
```

### 8.2 MortgageValidationPack Checks

The `MortgageValidationPack` handles documents explicitly typed as `mortgage_payslip`,
`mortgage_bank_statement`, `mortgage_employment_letter`, `mortgage_form16`, and
`mortgage_utility_bill`. It wraps the base domain packs and adds mortgage-specific
cross-checks.

| Check | Rule ID | Implemented |
|---|---|---|
| Payslip net ≈ bank salary credit | `payslip_bank_reconciliation` | 🔲 |
| CIN format validation | `employer_cin_format` | 🔲 |
| CIN company age vs join date | `cin_age_vs_join_date` | 🔲 |
| Employment join date vs payslip | `join_date_plausibility` | 🔲 |
| Circular funds in bank statement | `circular_funds_detection` | 🔲 |
| Duplicate transaction references | `duplicate_txn_reference` | 🔲 |
| Salary credit regularity | `salary_credit_regularity` | 🔲 |
| Form 16 vs payslip YTD | `ytd_form16_reconciliation` | 🔲 |
| HRA overclaim | `hra_overclaim` | 🔲 |
| TDS plausibility | `tds_plausibility` | 🔲 |
| PF rate validity | `pf_rate_validity` | 🔲 |
| PT slab validity | `pt_slab_validity` | 🔲 |

### 8.3 Case Bundle Reconciliation

A `CaseBundle` groups all documents submitted for one loan application:

```python
CaseBundle(
    case_id: str,
    payslips: list[StructuredSummary],          # 3 months
    bank_statement: StructuredSummary,
    employment_letter: StructuredSummary,
    form16: StructuredSummary | None,
    utility_bill: StructuredSummary | None,
    gift_letter: StructuredSummary | None,
    property_agreement: StructuredSummary | None,
)
```

Cross-document signals produced by the bundle reconciler:

1. `income_reconciliation` — compare all payslip net pay values against bank salary credits
2. `employer_consistency` — employer name fuzzy-match across payslip and employment letter
3. `join_date_plausibility` — join date on employment letter ≤ first payslip period
4. `ytd_form16_reconciliation` — last payslip YTD gross ÷ months = Form 16 gross ÷ 12
5. `gift_credit_tracing` — gift letter amount matches an inward credit near letter date

---

## 9. Explainability Requirements (Banking-Grade)

Every fraud flag must be accompanied by:

1. **Rule ID** — e.g. `SR-001`
2. **Evidence values** — e.g. `{payslip_net: 85000, bank_credit: 52000, delta_pct: 38.6}`
3. **Narrative reason code** — e.g. `"Payslip net pay exceeds salary bank credit by 38.6% — income inflation suspected"`
4. **Severity** — `critical | high | medium | low`
5. **Suggested action** — e.g. `"Request bank statement from alternate source (bank-issued PDF with watermark)"`

This is non-negotiable for banking-sector adoption. Regulators (RBI, NHB) require audit trails.

---

## 10. Data Sources Required

| Data Source | Type | Used For | Status |
|---|---|---|---|
| MCA21 / MCA Portal | External API / Scrape | CIN company registry validation | 🔲 Planned |
| CERSAI | External API | Duplicate mortgage detection | 🔲 Planned |
| RERA State Portals | External API / Scrape | Builder/project registration | 🔲 Planned |
| NSDL / TRACES | External API | PAN validation, TDS reconciliation | 🔲 Planned |
| CIBIL / Experian / Equifax | External API | Credit score + existing loan check | 🔲 Planned |
| Income Tax e-Filing Portal | External | Form 26AS / AIS reconciliation | 🔲 Planned |
| Bank API (Account Aggregator) | External API | Direct statement pull | 🔲 Planned |
| Synthetic corpus (BaseTruth) | Internal | Model training + rule testing | ✅ Done |

---

## 11. Synthetic Training Corpus

Generated at `data/mortgage_docs/` using `scripts/generate_mortgage_docs.py`.

| Attribute | Value |
|---|---|
| Total cases | 50 (configurable via `--n-cases`) |
| Documents per case | 8–9 PDFs |
| Total PDFs | ~426 |
| Tampered cases (30%) | 15 |
| Clean cases (70%) | 35 |
| Document types | Payslip (×3), Bank Statement, Employment Letter, Form 16, Utility Bill, Gift Letter (40% of cases), Property Agreement |
| Tamper variants | `income_inflated`, `employer_fake`, `circular_funds`, `backdated_employment` |
| Labels | `data/mortgage_docs/labels.csv`, `data/mortgage_docs/metadata.json` |

**To regenerate:**
```bash
python scripts/generate_mortgage_docs.py --out data/mortgage_docs --n-cases 100 --tamper-ratio 0.30
```

---

## 12. Physical & Material Forensics (Advanced Layer)

### 12.1 PDF Layer Analysis

- **Floating image layers**: Detect signature or stamp images that were placed as a transparent PDF image layer on top of a text-only page (not embedded in the original scan).
- **Tools**: `pypdf` (existing in BaseTruth), `pdfminer.six` for object-tree inspection.
- **Signal**: `floating_signature_layer` — High risk if signature appears as an isolated image XObject.

### 12.2 Scan Consistency

- **Noise uniformity**: A forged area (digitally inserted content) in an otherwise scanned document will show visually different noise texture.
- **Tool**: OpenCV Laplacian variance across page regions.
- **Signal**: `scan_noise_variance_anomaly` — High risk if one region shows zero noise in noisy document.

### 12.3 PDF Metadata & Producer Forensics

- **pyExifTool / pypdf**: Extract `Producer`, `Creator`, `CreationDate`, `ModDate` fields.
- **Suspicious editors**: Photoshop, Illustrator, Canva, GIMP, CorelDraw, Inkscape, LibreOffice Draw.
- **Compression markers**: Web-optimised JPEG artifacts in a supposedly scanned document.
- **Signal**: `suspicious_editor_in_metadata` — already implemented ✅

### 12.4 Font Fingerprinting

- PDFs created from the same legitimate payroll software consistently use the same fonts.
- A forged payslip that replaces only the salary amounts will often show font inconsistencies in those fields.
- **Tool**: `pdfminer.six` font name extraction per text block.
- **Signal**: `font_inconsistency_in_amount_fields` — Medium risk.

---

## 13. Graph-Based Fraud Ring Detection (Future Phase)

### 13.1 Entity Graph

Nodes: `Applicant`, `Employer`, `Property`, `BankAccount`, `PhoneNumber`, `EmailAddress`, `Appraiser`, `DSA`

Edges: `works_at`, `has_account`, `applied_for_loan`, `owns_property`, `contacted_by`, `appraised_by`

### 13.2 Ring Detection Patterns

| Pattern | Fraud Ring Type | Detection |
|---|---|---|
| N applicants → same employer CIN | Shell employer ring | Degree centrality on employer node |
| N applicants → same property address | Same unit sold multiple times | Duplicate edge detection |
| N applicants → same DSA code, all high-risk | DSA fraud ring | DSA node risk score |
| Same phone/email across applications | Straw buyer network | Shared attribute clustering |

### 13.3 Template Recurrence (FAISS)

- Compute perceptual hash (pHash) of each scanned payslip.
- Index hashes in a FAISS flat index.
- At scan time: nearest-neighbour query — if distance < threshold, flag as matching a known fraud template.
- **Signal**: `known_fraud_template_match` — Critical if distance < 0.05.

---

## 14. Regulatory & Compliance References (India)

| Regulation | Relevance |
|---|---|
| **RBI Master Circular on KYC** | Identity verification requirements for all borrowers |
| **NHB Directions 2010** (amended) | HFC fraud reporting obligations (Fraud Monitoring Returns) |
| **PMLA 2002** | Money laundering via mortgage — reporting of suspicious transactions to FIU-IND |
| **CERSAI Act 2002** | Mandatory charge registration; non-registration = duplicate mortgage risk |
| **IT Act Section 66D** | Cyber fraud via impersonation — applicable to synthetic identity fraud |
| **IPC Section 420** | Cheating and dishonestly inducing delivery of property (classic mortgage fraud charge) |
| **RERA 2016** | Builder/project registration — project-level fraud detection |
| **Income Tax Act Section 197A** | TDS certificate authenticity (Form 16) |

---

## 15. Implementation Status

> See [TRACKER.md](TRACKER.md) for full status. This section shows mortgage-specific build progress.

| Feature | Status | File / Module |
|---|---|---|
| Synthetic mortgage document corpus | ✅ Done | `scripts/generate_mortgage_docs.py` |
| PayrollValidationPack (payslip checks) | ✅ Done | `src/basetruth/analysis/packs/payroll.py` |
| BankingValidationPack (statement checks) | ✅ Done | `src/basetruth/analysis/packs/banking.py` |
| MortgageValidationPack (mortgage-specific checks) | 🔲 Next | `src/basetruth/analysis/packs/mortgage.py` |
| Cross-document reconciliation engine | 🔲 Next | `src/basetruth/analysis/cross_doc.py` |
| Circular funds detection | 🔲 Next | `BankingValidationPack` extension |
| CIN format + age validation | 🔲 Next | `MortgageValidationPack` |
| Payslip ↔ bank income reconciliation | 🔲 Next | `cross_doc.py` |
| Case bundle grouping + scan | 🔲 Next | `src/basetruth/analysis/case_bundle.py` |
| FAISS template fingerprint library | 🔲 Future | Separate module |
| MCA21 CIN API stub | 🔲 Future | `src/basetruth/integrations/mca21.py` |
| CERSAI charge lookup stub | 🔲 Future | `src/basetruth/integrations/cersai.py` |
| Graph ring detection | 🔲 Future | `src/basetruth/analysis/graph.py` |

---

## 16. KPIs & Success Metrics

| KPI | Target |
|---|---|
| Fraud Detection Rate (on labelled corpus) | > 85% |
| False Positive Rate | < 10% |
| Mean Investigation Time (manual review) | < 15 minutes per case |
| Evidence completeness per flag | 100% (every signal has evidence dict) |
| API p95 latency for single-document scan | < 3 seconds |

---

*Last updated: March 2026. Document is authoritative reference for BaseTruth mortgage module.*
