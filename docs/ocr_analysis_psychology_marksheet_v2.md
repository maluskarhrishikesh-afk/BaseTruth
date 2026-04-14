# OCR Analysis — Psychology_Marksheet.jpg (v2, post-IGNOU-fix)

## Document

**File:** `tests/sample/Psychology_Marksheet.jpg` (uploaded via Bulk Scan)  
**Board:** IGNOU — Indira Gandhi National Open University  
**Programme:** MA IN PSYCHOLOGY (Code: MAPC)  
**Term-End Exam:** DECEMBER 2023

---

## 1. What PaddleOCR Captured

PaddleOCR ran with `PP-OCRv4` (`lang="en"`). The document is a JPG image, so pypdf/pytesseract also ran separately for the liteparse artifact.

Full PaddleOCR OCR text (from `artifacts/Psychology_Marksheet/Psychology_Marksheet_ocr_scan.md`):

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

**What PaddleOCR DID pick up:**
- All 8 course rows (IGNOU course codes MPC006, MPCL007, MPCEO11, MPCE012, MPCE013, MPCE014, MPCE015, MPCE046)
- Final marks column (the `100  XX.XX  SC  MMYY` pattern at the end of each row)
- Summary line: `MA IN PSYCHOLOGY SUCCESSFULLY COMPLETEDWITH 68.42 %(FIRST DIVISION)`
- Total row: `TOTAL: 1300 889.40`

**What PaddleOCR MISSED:**
- The header section (ENROLMENT NO, NAME) — these are in the Devanagari-font header area which PaddleOCR's English model ignores entirely
- The actual year of admission, certificate number, date of printing — again in the watermark-noisy header

---

## 2. What liteparse (pypdf text extraction) Captured

The liteparse ran **pypdf_text_extraction** on the JPG (via pytesseract fallback). It successfully decoded the embedded header text:

```
ENROLMENT NO : 2002512226
NAME : HRISHIKESH NAMDEO MALUSKAR
C/o NAMDEO DAVALA MAUSKAR
CERTIFICATE NO. : 106117
DATE OF PRINTING: 12/04/2024
TERM-END EXAM. : DECEMBER 2023
MTH. & YR. OF ADM.: JULY 2020
MEDIUM : ENGLISH
```

This is the **ground-truth** source for name and enrollment number. PaddleOCR cannot read this area due to the Devanagari-script labels and IGNOU watermark background.

---

## 3. Ground Truth (Expected Correct Extraction)

| Field | Correct Value |
|---|---|
| Candidate Name | HRISHIKESH NAMDEO MALUSKAR |
| Board | INDIRA GANDHI NATIONAL OPEN UNIVERSITY |
| Programme | MA IN PSYCHOLOGY |
| **Enrollment No** | **2002512226** (10-digit IGNOU enrollment) |
| Certificate No | 106117 (document serial — NOT the enrollment ID) |
| Term-End Exam | DECEMBER 2023 |
| Percentage | 68.42% |
| Result | FIRST DIVISION |

### Subject marks (per-subject Term-End Exam marks out of 100):

| Course Code | Subject | Marks Obtained |
|---|---|---|
| MPC006  | Statistics in Psychology | 71 |
| MPCL007 | Practicum: Experimental Psychology and Psychological Testing | 43 |
| MPCE011 | Psychopathology | (marks unclear in OCR — see Bug 4) |
| MPCE012 | **Psychodiagnostics** | 74 |
| MPCE013 | Psychotherapeutic Methods | 77 |
| MPCE014 | Practicum in Clinical Psychology | 62 |
| MPCE015 | Internship | 75 |
| MPCE046 | Applied Positive Psychology | 67 |

Grand total: 889.40 (credit-weighted across all IGNOU programme courses; NOT the sum of marks shown above).

---

## 4. What Was Extracted (Current, After IGNOU Fix)

```json
{
  "candidate_name": "RAHISHKEH NAMDEO MALUSKAR",
  "enrollment_or_seat_number": "2001223",
  "subjects": [
    {"subject_name": "Statistics In Psychology", "marks_obtained": 71, "max_marks": 100},
    {"subject_name": "Practicum:Experiemental Psychology And", "marks_obtained": 43, "max_marks": 100},
    {"subject_name": "MPCE012", "marks_obtained": 74, "max_marks": 100},
    {"subject_name": "Psychotherapeutic Methods", "marks_obtained": 77, "max_marks": 100},
    {"subject_name": "Practicum In Clinical Psychology", "marks_obtained": 62, "max_marks": 100},
    {"subject_name": "Internship", "marks_obtained": 75, "max_marks": 100},
    {"subject_name": "Applied Positive Psychology", "marks_obtained": 66, "max_marks": 100}
  ],
  "result": "FIRST DIVISION",
  "percentage_or_cgpa": "68.42",
  "extraction_confidence": "HIGH"
}
```

Problems:
- `candidate_name` = "RAHISHKEH NAMDEO MALUSKAR" — garbled OCR noise, should be "HRISHIKESH NAMDEO MALUSKAR"
- `enrollment_or_seat_number` = "2001223" — hallucinated/misread, should be "2002512226"
- `subject_name` = "MPCE012" for the Psychodiagnostics course — should be "Psychodiagnostics"
- MPCE011 (Psychopathology) is missing from subjects entirely

---

## 5. Bug Analysis

### Bug 1 — Candidate Name Garbled (`"RAHISHKEH"`)

**Root cause**: PaddleOCR's OCR scan does NOT contain the candidate's name. The name `HRISHIKESH NAMDEO MALUSKAR` is in the document header which PaddleOCR's English-only model skips entirely (Devanagari fonts + IGNOU watermark noise in that region).

Gemma4 reads the image directly and tries to read the name from the header area. Due to the IGNOU watermark densely printed across the header, Gemma4 misreads "HRISHIKESH" as "RAHISHKEH".

The liteparse artifact (`Psychology_Marksheet_liteparse.json`) correctly has:
```
NAME : HRISHIKESH NAMDEO MALUSKAR
```

But the IGNOU structure hint currently passes only the subject rows and examination data to Gemma4 — it does NOT pass the liteparse-extracted name.

**Fix**: In `_build_marksheet_structure_hint`, for IGNOU layout, also include the verified header values (`IGNOU_VERIFIED_NAME`, `IGNOU_VERIFIED_ENROLLMENT`) extracted from the liteparse file on disk. This gives Gemma4 the ground-truth text to copy rather than having to guess from a watermarked image.

---

### Bug 2 — Enrollment Number Wrong (`"2001223"`)

**Root cause**: Same as Bug 1. PaddleOCR misses the header area. Gemma4 tries to read the enrollment from the image header but misreads `2002512226` (10-digit) as `2001223` (7-digit). The IGNOU watermark severely degrades readability in that area.

The liteparse artifact has `ENROLMENT NO : 2002512226` clearly, but this is not currently included in the hints given to Gemma4.

The IGNOU structure hint already warns Gemma4:
- "The document header contains two numbered fields"
- "CERTIFICATE NO (6 digits) is a document serial number — NOT the enrollment ID"
- "ENROLMENT NO (10 digits) is the student enrollment identifier"

But without the actual value being provided as a hint, Gemma4 can still misread the number from the noisy image.

**Fix**: Read `ENROLMENT NO : 2002512226` from the liteparse file and add `IGNOU_VERIFIED_ENROLLMENT: 2002512226` to the structure hint. Gemma4 will then copy this value rather than attempt optical character recognition of the noisy header.

---

### Bug 3 — Subject Named `"MPCE012"` Instead of `"Psychodiagnostics"`

**Root cause**: In `_parse_ignou_ocr_directly`, subject title extraction uses this regex:

```python
title_match = re.search(
    r"^\s*MPC[EL]?\d{2,3}\s+([A-Z][A-Z0-9 :&',./-]{3,60}?)\s+\d",
    line,
    re.IGNORECASE,
)
subject_label = title_match.group(1).strip().title() if title_match else code
```

The OCR line for MPCE012 is:
```
MPCE012    PSYCHODIAGNOSTICS                              UN ERE                 JE   ID # H    #       #       100   74.30    SC      0623
```

After "PSYCHODIAGNOSTICS" there are ~46 whitespace characters and watermark noise before the first digit "100" appears. The `{3,60}?` non-greedy limit means the regex can only match 60 characters. Within those 60 characters (which span "PSYCHODIAGNOSTICS" + lots of noise), there is no `\s+\d` terminator. So the regex **fails**, and the code falls back to using the raw course code `"MPCE012"` as the subject name.

**Fix**: Strip watermark noise (`#` characters and surrounding whitespace) from the line before running the title regex. The IGNOU marks table uses `#` as separator characters between components, so cutting at the first `#` gives a clean title region:
```python
clean_line = re.sub(r"\s*#.*$", "", line)  # remove everything from first '#' onwards
```
After cleanup, the line for MPCE012 becomes:
```
MPCE012    PSYCHODIAGNOSTICS
```
The title regex then matches `PSYCHODIAGNOSTICS` reliably.

---

### Bug 4 — MPCE011 (Psychopathology) Missing From Subjects

**Root cause**: The MPCEO11 line has extreme watermark noise:
```
 MPCEO11   PSYCHOPATHOLOGY               4    2 ND 22.50 # SVER IE TRAG # A51.80 PEN # M H NE CHI
```

The `_parse_ignou_ocr_directly` function looks for a final marks column matching `100  [marks]  SC` at the end of the row. This line does NOT have that pattern — the `SC  MMYY` terminator is completely overwritten by watermark noise (`M H NE CHI`).

The continuation line that follows:
```
                                         4   2    21.00   # ERS # COTRDA GANDHI 46.20 # #  H    #       #       100   67.20    SC      0623
```
Has `100   67.20    SC      0623` which would match — but it has no course code prefix, so `_parse_ignou_ocr_directly` skips it (requires lines to begin with an IGNOU course code).

The actual MPCE011 mark (67.20, i.e., 67) is on the unlabeled continuation row and is currently lost.

**Potential fix**: After parsing all code-anchored lines, also scan for "orphan" continuation rows (no code prefix, but has `100  [marks]  SC  [mmyy]`) and associate them with the previous course. However, this is complex logic and may introduce false matches. The safer fix is to pass the `ignou_subject_rows` hint from `_parse_ignou_ocr_directly` clearly to Gemma4 and let Gemma4 read MPCE011 from the image directly — which with the `IGNOU_VERIFIED_NAME` fix in place should now work better overall.

---

## 6. Layout Detection (Correct)

```json
{
  "engine": "paddleocr",
  "layout_family": "ignou_open_university",
  "layout_markers": ["SUCCESSFULLY_COMPLETED", "ignou_code_count=7"],
  "layout_score": 88,
  "direct_score": 0,
  "score": 92.8
}
```

IGNOU layout is correctly detected. `direct_score=0` because IGNOU intentionally does NOT use the direct parser path — `_parse_ignou_ocr_directly` is used to seed structure hints to Gemma4, which then reads name/enrollment from the image and validates subjects.

`_extraction_attempts=1` confirms Gemma4 was called (expected for IGNOU).

---

## 7. Fixes Applied

| Bug | Location | Fix |
|---|---|---|
| Candidate name garbled | `_build_marksheet_structure_hint` + `extract_document_fields` | Read liteparse file; add `IGNOU_VERIFIED_NAME` to structure hint |
| Enrollment number wrong | Same | Read liteparse file; add `IGNOU_VERIFIED_ENROLLMENT` to hint |
| Subject name = course code ("MPCE012") | `_parse_ignou_ocr_directly` title extraction | Strip `#`-delimited watermark noise before running title regex |
| MPCE011 missing | `_parse_ignou_ocr_directly` | Link orphan continuation rows (no code prefix + `100 marks SC mmyy`) to the preceding course |
