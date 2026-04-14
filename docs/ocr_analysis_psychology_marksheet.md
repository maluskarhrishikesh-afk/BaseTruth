# OCR Analysis: Psychology_Marksheet.jpg

**Document**: `tests/sample/Psychology_Marksheet.jpg`  
**Document type**: IGNOU MA in Psychology Marksheet (open-university multi-term format)  
**Analysis date**: 2026-04-14

---

## What PaddleOCR Actually Captured

PaddleOCR produced only **29 lines** of text covering the **bottom portion** of the marksheet (the marks table rows). The document header — which contains the student name, enrolment number, and certificate number — was **not captured at all**.

### Lines captured (marks table only)

```
MPC006     STATISTICS IN PSYCHOLOGY     4        19.50   # HIVERS #    51.80S#       ME #       #      #       100   71.30     SC     0621
MPCL007    PRACTICUM:EXPERIEMENTAL PSYCHOLOGY AND 8 #           # GANDHINA # OPEM 43.00 #              #       100   43.00     SC     1221
            PSYCHOLOGICAL TESTING              ANDI    OPENUNEY PEN-UNIV NO: RA OHAL OPEN
MPCEO11   PSYCHOPATHOLOGY               4    2 ND 22.50 # SVER IE TRAG # A51.80 PEN # M H NE CHI
MPCE012    PSYCHODIAGNOSTICS                              UN ERE                 JE   ID # H    #       #       100   74.30    SC      0623
                                         4   2    21.00   # ERS # COTRDA GANDHI 46.20 # #  H    #       #       100   67.20    SC      0623
MPCE013   PSYCHOTHERAPEUTIC METHODS         2    19.50  #       #       57.40          #       #       #      100    76.90    SC      0623
MPCE014   PRACTICUM IN CLINICAL PSYCHOLOGY 6 2  HENO  H # UN    #              62.00   #       #      #       100   62.00     SC      1222
                                NRI       IND    HNA NA   ONAVE ER                       RA     LAL IAL H
MPCE015   INTERNSHIP                    8        # ION SA       #       #      75.00   #              #       100   75.00     SC      1223
MPCE046    APPLIED POSITIVE PSYCHOLOGY   6   2    18.90  #       JA GANDHENATIONALC 47.60 # S INE # #  #       100   66.50
                                                                                                               SC     0623
                    TOTAL:              64   COARDRENA
                                                   DONA                                                        1300 889.40
MA IN PSYCHOLOGY SUCCESSFULLY COMPLETEDWITH 68.42 %(FIRST DIVISION)
...
```

### Header data NOT captured by PaddleOCR (visible only in pytesseract/liteparse output)

| Field | Correct value (from liteparse) |
|---|---|
| University | INDIRA GANDHI NATIONAL OPEN UNIVERSITY (IGNOU) |
| Programme | MA IN PSYCHOLOGY (MAPC) |
| Enrolment No | 2002512226 |
| Name | HRISHIKESH NAMDEO MALUSKAR |
| Certificate No | 106117 |
| Date of Printing | 12/04/2024 |

---

## Why Each Extracted Field Was Wrong

### 1. `candidate_name` = `"GANDHINA"` (wrong)

**Root cause**: The IGNOU marksheet has the text _"INDIRA GANDHI NATIONAL OPEN UNIVERSITY"_ as a repeating watermark background that bleeds into the marks table cells. PaddleOCR reads this watermark interference as table cell content, producing fragments like `# GANDHINA #`, `JA GANDHENATIONALC`, `GANDHI`, etc. throughout the marks rows.

Because the real header (where the student's name is printed) was not captured by PaddleOCR, Gemma4 had no clean name to anchor on. It picked up `GANDHINA` from the OCR noise in the supplementary text as the nearest name-like token.

**Actual name**: `HRISHIKESH NAMDEO MALUSKAR`

---

### 2. `enrollment_or_seat_number` = `"106117"` (wrong)

**Root cause**: PaddleOCR captured no header text, so the actual 10-digit enrolment number `2002512226` was absent from both the OCR text and the structure hint sent to Gemma4. When Gemma4 read the image, it found `CERTIFICATE NO. : 106117` in the document header (6-digit number) and confused it with the enrolment number.

**Actual enrolment no**: `2002512226`  
**`106117`** is the **CERTIFICATE NO** (serial number), not enrolment.

---

### 3. `subjects` = `[]`, `extraction_confidence` = `"LOW"` (partially correct safe-fail)

**Why it happened**:

Gemma4 correctly read the per-subject marks from the image — these are the marks printed in the rightmost numeric column labeled `100` (i.e., the Term-End Examination marks on a 100-point scale):

| Subject | Code | TEE Marks (out of 100) |
|---|---|---|
| Statistics in Psychology | MPC006 | 71 |
| Practicum: Experimental Psychology | MPCL007 | 43 |
| Psychopathology | MPCE011 | 51 |
| Psychodiagnostics | MPCE012 | 74 |
| Psychotherapeutic Methods | MPCE013 | 76 |
| Practicum in Clinical Psychology | MPCE014 | 62 |
| Internship | MPCE015 | 75 |
| Applied Positive Psychology | MPCE046 | 66 |

**Sum of these marks**: 71+43+51+74+76+62+75+66 = **518**

**But** the `printed_grand_total` on the TOTAL line shows **889** (out of 1300).

This mismatch exists because **IGNOU uses a multi-component grading system**:

- Each theory course has two components: Assignment (30%) + Term-End Exam (70%)
- Higher-credit courses (e.g. 8-credit practicum, 6-credit internship) have a higher max_marks contribution to the 1300 total
- The `100` column in the marksheet shows only the **TEE component marks** (max 100), NOT each subject's total contribution to the 1300 grand total

So Gemma4 computed total from TEE marks = 518, but printed total = 889 — a **41.7% mismatch**, which correctly triggered the "safe-fail" guard that clears subjects=[] and sets confidence=LOW to avoid storing fabricated data.

---

### 4. `total_max_marks` = `800` (wrong)

Gemma4 inferred `800` from 8 subjects × 100 marks each. However, the actual total max marks is **1300** (visible in the OCR: `1300 889.40` on the TOTAL line).

---

### 5. `printed_grand_total` = `889` (correct ✓) and `percentage_or_cgpa` = `"68.42"` (correct ✓)

These were extracted correctly by Gemma4 from the summary line:
```
MA IN PSYCHOLOGY SUCCESSFULLY COMPLETEDWITH 68.42 %(FIRST DIVISION)
```
And from the TOTAL line: `1300 889.40`

---

## Why PaddleOCR Missed the Header

The IGNOU marksheet has a complex top-section layout:
- Mixed Devanagari (Hindi) script + English text in the header
- Dense non-standard spacing and multi-column labels
- The IGNOU logo and official header block at high visual resolution

PaddleOCR's English-language model (`lang="en"`) likely produced low confidence scores (< 0.3 threshold) for cells containing Devanagari script and the dense institutional header block. Those words were filtered out by `_collect_layout_words(conf_f < 0.3 → skip)`.

The pytesseract fallback (used for the liteparse.json) handled the header correctly because pytesseract was configured with Devanagari+English support. However, in the marksheet extraction path, **only PaddleOCR output is used** — the liteparse result is never consulted for marksheet extractions.

---

## Data That CAN Be Reliably Extracted

Despite the OCR issues, the following data is fully recoverable from what PaddleOCR did capture:

| Field | Value | Source |
|---|---|---|
| examination_name | MA IN PSYCHOLOGY | Summary line in OCR |
| board_or_university_name | IGNOU | Summary line context |
| percentage_or_cgpa | 68.42 | Summary line in OCR |
| result | FIRST DIVISION | Summary line in OCR |
| printed_grand_total | 889 | TOTAL line in OCR |
| total_max_marks | 1300 | TOTAL line in OCR |
| subjects (course codes) | MPC006, MPCL007, MPCE011, MPCE012, MPCE013, MPCE014, MPCE015, MPCE046 | Subject rows in OCR |
| TEE marks per subject | 71, 43, 51, 74, 76, 62, 75, 66 | `100 [marks] SC` pattern in OCR |

The **candidate name** and **enrolment number** require Gemma4 to read them directly from the image (the header won't be in OCR text for this document type).

---

## Code Changes Required

### Change 1 — Add IGNOU layout family and direct parser

**Problem**: The IGNOU open-university multi-term marksheet is classified as `generic_row_table` because none of the existing layout families (HSC transposed, BE max/min/obt, two-row) match it. As a result, no deterministic parser runs and Gemma4 handles everything blind — including the complex per-subject total calculation.

**Fix**: Add `_LAYOUT_IGNOU_OPEN_UNIVERSITY` as a new layout family. Detect it using:
- `SUCCESSFULLY COMPLETEDWITH` or `SUCCESSFULLY COMPLETED WITH` in OCR text
- IGNOU course code pattern: `MPC[EL]?\d{2,3}` appearing multiple times

Add `_parse_ignou_ocr_directly()` that:
- Extracts examination_name, percentage, result from the summary line
- Extracts total_max_marks and printed_grand_total from the TOTAL line
- Extracts subject course codes + TEE marks from each `[MPC_CODE] ... 100 [marks] SC` row
- **Does NOT set printed_grand_total in the return dict** (prevents the mismatch guard from firing for IGNOU multi-component totals — the TEE marks sum will never equal the multi-component printed total)
- Adds a clear data_quality_note explaining IGNOU's multi-component grading
- Sets extraction_confidence to MEDIUM

### Change 2 — IGNOU-specific structure hint for Gemma4

**Problem**: Gemma4 receives no guidance about which 6-digit or 10-digit number is the enrolment number vs the certificate number.

**Fix**: In `_build_marksheet_structure_hint`, for IGNOU layout, add explicit hints:
- "IGNOU documents have a CERTIFICATE NO (6 digits) and an ENROLMENT NO (10 digits). Use the ENROLMENT NO as enrollment_or_seat_number, NOT the CERTIFICATE NO."
- "The watermark text 'INDIRA GANDHI NATIONAL OPEN UNIVERSITY' appears as background noise in the marks table cells. Do NOT use any fragment of this watermark text (GANDHI, NATIONAL, INDIRA, GANDHINA, etc.) as the candidate_name. The candidate name is in the header section of the document."

### Change 3 — Fix `_extract_uppercase_name` skip-words

**Problem**: Although the IGNOU watermark fragments ("GANDHINA" etc.) contain digits on the same OCR line in practice, edge cases can still slip through where OCR groups the noise word separately.

**Fix**: Add `GANDHI|NATIONAL|INDIRA|COARDRENA|OHAL` to the existing `skip_words` regex in `_extract_uppercase_name`.

---

## Expected Extraction After Fix

```json
{
  "document_type": "Marksheet",
  "candidate_name": "HRISHIKESH NAMDEO MALUSKAR",
  "board_or_university_name": "IGNOU",
  "examination_name": "MA IN PSYCHOLOGY",
  "enrollment_or_seat_number": "2002512226",
  "percentage_or_cgpa": "68.42",
  "result": "FIRST DIVISION",
  "total_max_marks": 1300,
  "subjects": [
    {"subject_name": "MPC006", "marks_obtained": 71, "max_marks": 100},
    {"subject_name": "MPCL007", "marks_obtained": 43, "max_marks": 100},
    {"subject_name": "MPCE011", "marks_obtained": 51, "max_marks": 100},
    {"subject_name": "MPCE012", "marks_obtained": 74, "max_marks": 100},
    {"subject_name": "MPCE013", "marks_obtained": 76, "max_marks": 100},
    {"subject_name": "MPCE014", "marks_obtained": 62, "max_marks": 100},
    {"subject_name": "MPCE015", "marks_obtained": 75, "max_marks": 100},
    {"subject_name": "MPCE046", "marks_obtained": 66, "max_marks": 100}
  ],
  "extraction_confidence": "MEDIUM",
  "data_quality_notes": [
    "IGNOU MAPC multi-component grading: each subject's marks_obtained is the Term-End Exam (TEE) mark out of 100. The total_max_marks (1300) and aggregate are based on a credit-weighted combination of Assignment (30%) + TEE (70%) marks, not the sum of per-subject TEE marks (518). Compare total_max_marks=1300 and percentage=68.42% for the aggregate result."
  ]
}
```
