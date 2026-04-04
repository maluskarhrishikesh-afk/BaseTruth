# BaseTruth — Product Usage Guide

## Who this is for

This guide is for anyone setting up a verification workflow using BaseTruth:
loan officers, KYC analysts, compliance teams, and developers integrating
BaseTruth into a banking or fintech product.

---

## The Big Picture

BaseTruth answers three questions about every applicant:

1. **Who are they?** — Identity verified from government-issued documents
2. **Are their documents real?** — Tamper and forgery detection
3. **Do their documents agree with each other?** — Cross-document consistency

Every step feeds into a case that an analyst can approve or reject.
The final output is a signed PDF report that can be stored or shared.

---

## Correct Sequence of Steps

### STEP 1 — Register the Applicant (Entity)

**Where:** Records → New Entity  (or it is created automatically in Step 2)

**Capture once:**
| Field | Why it matters |
|-------|---------------|
| First name + Last name | Matched against every document |
| PAN number | Cross-validated across payslips, Form 16, bank statements |
| Aadhaar number | Cross-validated against Aadhaar QR data |
| Email | Used to pull documents from email or notify the applicant |
| Phone | Used for Video KYC session link delivery |
| Reference number | Auto-assigned (BT-000001, BT-000002, …) |

**What happens:**  A unique entity record is created.  All future documents
and identity checks are linked to this record.

**Auto-population:**  Once you select this person on any screen, they become
the *active applicant* for your entire session.  Every other screen
automatically pre-selects them — you do not need to search again.

---

### STEP 2 — Verify Identity Documents

**Where:** Identity Verification screen

**Upload:**
- PAN card (JPG, PNG, or PDF)
- Aadhaar card (JPG, PNG, or PDF)

**What BaseTruth captures automatically:**

| Document | Fields extracted | Saved to entity |
|----------|-----------------|----------------|
| PAN card | PAN number, Name, Father's name, DOB, Entity type | PAN number |
| Aadhaar card QR | Full name, DOB, Gender, Full address, Last 4 digits | Aadhaar number |

**What gets checked:**
- Deskewing + perspective correction before OCR (automatic)
- PaddleOCR extracts text; falls back to Tesseract if needed
- If OCR confidence is low, Gemma VLM is called as a final fallback
- Image forensics: ELA tampering, copy-move detection, GAN detection,
  EXIF metadata, noise consistency
- PAN format validation (5 letters + 4 digits + 1 letter)

**Face photo extraction:**  The applicant's face is extracted from the
PAN / Aadhaar photo and stored as a reference embedding for face matching
in the next step.

**Output stored in database:**
- Identity document scan with full forensic report
- Reference face embedding linked to the entity
- Extracted fields saved to entity (PAN, Aadhaar, name)

---

### STEP 3 — Face Match (Static Selfie)

**Where:** Identity Verification screen → Face Match section (same page)

**Upload:** A live selfie photo of the applicant

**What gets checked:**
- ArcFace embedding comparison (cosine similarity)
- Match threshold: 40% similarity = pass
- Result shown as a percentage score (0–100)

**Output stored in database:**
- IdentityCheck record (type: face_match)
- Status: pass / fail
- Match score

**Tip:** This works best with a clear, well-lit, front-facing photo.
The selfie does not need to be a government document.

---

### STEP 4 — Video KYC (Live Liveness Verification)

**Where:** Video KYC screen

**When to use:**  When you need to prove the person is live — not using a
photograph of a photograph.  Required for most RBI / banking compliance flows.

**The applicant receives a link** (by phone or email) to a secure browser page
that works on any device with a camera.  No app download required.

**Liveness challenges presented (random 2 of 4):**
- Blink
- Turn head left
- Turn head right
- Nod

**What gets checked:**
- Real-time face detection (RetinaFace)
- Active liveness: challenge pass/fail
- Face match against the reference embedding captured in Step 2

**Transport options:**
- WebSocket (default) — works everywhere, JPEG frames
- WebRTC (optional, install aiortc) — lower latency, better for slow connections

**Output stored in database:**
- IdentityCheck record (type: video_kyc)
- Liveness pass/fail
- Face match score
- Session recording reference

---

### STEP 5 — Scan Supporting Documents

**Where:** Bulk Scan or Scan screen

**Upload all at once:**
- Payslips (multiple months)
- Bank statements
- Employment letter
- Form 16
- Gift letter (if applicable)
- Any other documents

**Active applicant is already selected** — no need to enter details again.
All documents are automatically linked to the entity from Step 1.

**What gets checked per document:**
- Document type classification (auto-detected)
- Field extraction (employer, salary, account number, etc.)
- Arithmetic consistency (gross = basic + HRA + allowances)
- PAN/name cross-check against entity from Step 2
- Image forensics (ELA, copy-move, GAN, metadata)
- Employer domain validation (email domain vs company name)

**Cross-document reconciliation (automatic after bulk scan):**
- Payslip income vs Form 16 income — should agree within 10%
- Payslip employer vs employment letter employer — must match
- Payslip name vs PAN name — must match
- Bank credits vs declared salary — flagged if significantly inconsistent

**Output stored in database:**
- One Scan record per document
- Extracted fields in DocumentInformation table
- All linked to the entity
- Case automatically created or updated

---

### STEP 6 — Review the Case

**Where:** Cases screen

**What you see:**
- Combined risk score across all documents
- All forensic signals (passed / failed)
- Cross-document mismatches highlighted
- Identity verification status (face match + video KYC)
- Timeline of all events

**Analyst actions:**

| Action | When to use |
|--------|------------|
| Approve | Everything checks out — low risk |
| Reject | Clear fraud or forgery detected |
| Request re-submission | Document quality too low to assess |
| Escalate | Needs a senior analyst or legal review |
| Add note | Record reasoning for audit trail |

**Priority levels:** Normal → High → Critical (auto-set based on risk score)

---

### STEP 7 — Generate and Download Report

**Where:** Reports screen (or directly from Cases screen)

**Report contains:**
- Executive summary (one-line verdict)
- Identity verification results (PAN, Aadhaar, face match, video KYC)
- Each document with: type, extracted fields, truth score, forensic signals
- Cross-document reconciliation findings
- Analyst notes and disposition history
- Timestamp and case reference

**Format:** PDF (signed, tamper-evident)

**Storage:** Saved to MinIO / S3; also downloadable from the dashboard.

---

## What NOT to Do

| Common mistake | Why it's a problem | Correct approach |
|---------------|-------------------|-----------------|
| Scanning documents without linking to an entity | Creates duplicate "unknown" entity records, breaks cross-document matching | Always search for the entity first (or it is auto-created from PAN/Aadhaar) |
| Running Video KYC before Identity Verification | No reference embedding to match against — liveness passes but face match skipped | Always do Step 2 before Step 4 |
| Uploading documents one at a time via Scan screen | Misses cross-document income reconciliation | Use Bulk Scan when you have more than one document |
| Approving a case without noting the reason | No audit trail for regulators | Always add a case note when approving or rejecting |

---

## Database Capture Summary

Everything is captured once and reused everywhere:

```
Entity (created in Step 1 or auto-created in Step 2)
  ├── first_name, last_name          ← from PAN / Aadhaar / manual entry
  ├── pan_number                     ← from PAN card OCR
  ├── aadhar_number                  ← from Aadhaar QR
  ├── email, phone                   ← from manual entry / datasource
  │
  ├── Scans[]                        ← one per document (Steps 2 + 5)
  │   ├── document_type, truth_score
  │   ├── extracted fields (salary, employer, account no., …)
  │   └── full forensic report JSON
  │
  ├── IdentityChecks[]               ← one per face match / video KYC (Steps 3 + 4)
  │   ├── check_type (face_match | video_kyc)
  │   ├── status (pass | fail)
  │   └── match score
  │
  └── Cases[]                        ← one per verification workflow (auto-created)
      ├── status, disposition, priority
      ├── risk_level (auto-computed)
      └── CaseNotes[] (analyst notes)
```

---

## Environment Setup Checklist

### Core (required)
- [ ] PostgreSQL running + `DATABASE_URL` environment variable set
- [ ] MinIO / S3 running + `MINIO_*` or `AWS_*` environment variables set
- [ ] Tesseract OCR installed on PATH

### For better ID card OCR (strongly recommended)
- [ ] `pip install paddlepaddle paddleocr`
- [ ] Uncomment `paddlepaddle` and `paddleocr` in requirements.txt

### For VLM OCR fallback (when OCR confidence is low)
- [ ] Option A — Local Gemma: `pip install transformers torch accelerate`  
     Set `GEMMA_MODEL_PATH` env var  
- [ ] Option B — Gemini API: `pip install google-generativeai`  
     Set `GEMINI_API_KEY` env var

### For WebRTC video KYC (lower latency)
- [ ] `pip install aiortc`
- [ ] Uncomment `aiortc` in requirements.txt

### Engine selection (optional via environment variables)
```
BASETRUTH_OCR_ENGINE=auto          # auto | paddleocr | pytesseract
BASETRUTH_FACE_ENGINE=auto         # auto | insightface | mediapipe
BASETRUTH_VLM_ENGINE=auto          # auto | gemma_local | gemini_api | none
BASETRUTH_OCR_CONF_THRESHOLD=0.70  # float 0–1 — VLM fallback trigger
```

---

## Summary: Right Order, Every Time

```
1. Entity Registration      → one record per applicant, done once
2. Identity Documents       → PAN + Aadhaar → auto-fills entity fields
3. Face Match               → selfie vs ID photo
4. Video KYC (if needed)    → liveness + live face match
5. Supporting Documents     → payslips, bank, employment letter (bulk scan)
6. Case Review              → analyst approves / rejects with notes
7. Report Download          → PDF with everything for the file
```
