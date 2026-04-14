# OCR Analysis — HSC-Marksheet.pdf

## Document

**File:** `artifacts/HSC-Marksheet/HSC-Marksheet.pdf`  
**Board:** Maharashtra State Board of Secondary and Higher Secondary Education (MSBSHSE)  
**Exam:** Higher Secondary School Certificate (HSC)  
**Year:** FEB-2001 (Stream: SCIENCE)

---

## 1. What PaddleOCR Captured

PaddleOCR ran on the HSC marksheet image with `PP-OCRv4` (`lang="en"`). The document is a scanned image PDF, so pytesseract was also run as a fallback via the liteparse path. Below is the full 61-line OCR output from `artifacts/HSC-Marksheet/HSC-Marksheet_ocr_scan.md`:

```
Line  0: '     -'
Line  2: '                               Aaharashtra State Board f'
Line  3: '                Secondarg and        Higher Secondary Lducation,Hhme'
Line  4: '                             PUNE          DIVISIONAL BOARD fHT g'
Line  7: '    STATEMENT OF MARKS OF THE HIGHER SECONDARY SCHOOL CERTIFICATE EXAMINATION'
Line 10: '     STREAM       SEAT NO    CENTRENODIST.&HR.SEC.SCHOOLNO MONTH &YEAR OF EXAM SR.NO.OF STATEMENT'
Line 12: ' SCIENCE         B006795       041         11.007          FEB-2001           005392'
Line 14: "          GARI YT T 3TST CANDIDATE'SFULLNAME SURNAME FIRST)"
Line 16: '   MALUSKAR HRISHIKESH NAMDED'
Line 19: 'RAa                COMPULSORY                yay                         fauu'
Line 20: 'SUBJECTS           LANGUAGES               OPTIONALSUBJECTS           VOCATIONAL'
Line 21: '                                                                       SUBJECT'
Line 22: '                                                                                  TOTAL'
Line 23: ' 194412                                                                           MARKS'
Line 24: '                    01             40     54      55                    A2'
Line 25: '*SUBJECT CODE'
Line 27: 'MAXIMUM MARKS       100    100     100    100    100     100    100       200     600/700'
Line 28: '9IST JTOE'
Line 29: ' MARKS OBTAINED    070            082    090    091                   171         504'
Line 31: ' TOTAL MARKS'
Line 32: ' IN WORDS          FIVE HUNDRED AND FOUR'
Line 33: '                       84.00            frhTet       PASS'
Line 34: ' PERCENTAGE OFMARKS                     RESULT'
Line 55: '  (Grade I with Distinction  (Grade1)           (Grade II)             Grade Pass'
Line 59: '                           60% and above      45% and above     All other successtulcandidates'
Line 60: '      75% and above        but below 75%      but below 60%       (Including the.exempted)'
```

---

## 2. What the Ground Truth Should Be

| Field | Correct Value |
|---|---|
| Candidate Name | MALUSKAR HRISHIKESH NAMDEO |
| Board | Maharashtra State Board of Secondary and Higher Secondary Education |
| Examination | HSC |
| Stream | SCIENCE |
| **Seat No (SEAT NO)** | **B006795** |
| Month & Year | FEB-2001 |
| SR.NO.OF STATEMENT | 005392 (not the seat number!) |
| Percentage | 84.00 |
| Result | **PASS** |
| Grand Total | **504** (confirmed in words: FIVE HUNDRED AND FOUR) |

### Subject breakdown (transposed column layout)

| Subject Code | Max Marks | Marks Obtained |
|---|---|---|
| 01 | 100 | 70 |
| 40 | 100 | 82 |
| 54 | 100 | 90 |
| 55 | 100 | 91 |
| **A2** (Vocational) | **200** | **171** |
| **Total** | **600** | **504** |

Formula check: 70 + 82 + 90 + 91 + 171 = **504** ✓

---

## 3. What Was Actually Extracted (Wrong)

```json
{
  "subjects": [
    {"subject_name": "01", "marks_obtained": 70,  "max_marks": 100},
    {"subject_name": "40", "marks_obtained": 82,  "max_marks": 100},
    {"subject_name": "54", "marks_obtained": 90,  "max_marks": 100},
    {"subject_name": "55", "marks_obtained": 91,  "max_marks": 100}
  ],
  "computed_total": 333,
  "printed_grand_total": 504,
  "total_max_marks": 400,
  "result": "Distinction",
  "enrollment_or_seat_number": "005392",
  "extraction_confidence": "MEDIUM"
}
```

Problems:
- Only 4 subjects instead of 5 (missing vocational subject A2 worth 170 marks)
- `computed_total = 333` instead of `504` (33.9% mismatch)
- `total_max_marks = 400` instead of `600`
- `result = "Distinction"` instead of `"Pass"`
- `enrollment_or_seat_number = "005392"` instead of `"B006795"`

---

## 4. Root Cause Analysis

### Bug 1 — Vocational Subject Code `A2` Not Recognized (Root Cause)

**Impact:** Missing 5th subject, computed_total=333 instead of 504

**Location:** `_parse_hsc_ocr_directly()` and `_reformat_hsc_ocr_table()` in `document_extract.py`

The HSC layout classifier correctly detects `hsc_transposed` (score=95). The direct parser then runs `_parse_hsc_ocr_directly()` which looks for the subject code line (line 24):

```
                    01             40     54      55                    A2
```

Subject codes are extracted with this regex:
```python
code_list = re.findall(r"\bA[Zz]\b|\b[0-9]{2}\b", code_line)
```

**The pattern `\bA[Zz]\b` matches `AZ` or `Az` only.** The actual OCR text shows `A2` (letter A + digit 2). This is the Maharashtra Board's code for the vocational subject (e.g., "A2" = vocational language).

Result: `code_list = ['01', '40', '54', '55']` — only 4 codes.

Meanwhile, the obtained marks line correctly produces 5 values:
```
obt_subjects_raw = ['070', '082', '090', '091', '171']
```

But `zip(['01','40','54','55'], ['070','082','090','091','171'])` stops at 4 pairs — the 171 (vocational subject marks) is silently dropped.

The same `[Aa][Zz]` pattern is also used in the `std_max` assignment, meaning even if A2 were found, it would get `std_max=100` instead of the correct 200.

**Fix:** Change pattern to `\bA[A-Za-z0-9]\b` to match `AZ`, `Az`, `A2`, `A3`, etc.

---

### Bug 2 — False `"Distinction"` Result (Footer Table False Match)

**Impact:** result="Distinction" instead of "Pass"

**Location:** `_parse_hsc_ocr_directly()` result detection loop

The parser scans the full OCR text for result keywords:
```python
for result_word in ("DISTINCTION", "ATKT", "PASS", "FAIL"):
    if result_word in upper:
        result = result_word.capitalize()
        break
```

Since DISTINCTION is checked first and found in the **grade description footer table** (line 55):
```
(Grade I with Distinction  (Grade1)   (Grade II)   Grade Pass
```

…it matches before PASS is ever checked. This is a false positive. The actual RESULT field (line 33-34) says:
```
84.00   PASS
PERCENTAGE OF MARKS   RESULT
```

The student scored 84% and PASSED — DISTINCTION in line 55 is merely a grade label in the footer chart explaining what grades mean.

**Fix:** Restrict the result keyword search to the first 40 OCR lines (the marks section), which avoids the footer grade chart that starts around line 50.

---

### Bug 3 — Seat Number `"005392"` Instead of `"B006795"`

**Impact:** enrollment_or_seat_number returns SR.NO.OF STATEMENT instead of SEAT NO

**Location:** `_parse_hsc_ocr_directly()` seat number extraction

The parser uses:
```python
seat_match = re.search(r"\b(\d{6})\b", ocr_text)
seat_number = seat_match.group(1) if seat_match else ""
```

Header data row (line 12):
```
SCIENCE   B006795   041   11.007   FEB-2001   005392
```

`B006795` is a 7-character alphanumeric token (`B` + 6 digits). The pattern `\b(\d{6})\b` requires **pure digits** with word boundaries. Since `B006795` is a single alphanumeric word, there is **no word boundary** between `B` and `0` — so the regex cannot match `006795` inside it. However, `005392` is a pure 6-digit token and matches immediately.

The regex returns `005392` which is the **SR.NO.OF STATEMENT** (a record reference), not the exam seat number.

**Fix:** Prefer patterns that start with a letter (`[A-Z]\d{5,7}`), which correctly matches `B006795`. Fall back to the pure-digit search only if no letter-prefixed number is found.

---

### Bug 4 — `total_max_marks=400` Instead of 600 (Downstream of Bug 1)

**Impact:** Maximum marks total is understated

This is a direct consequence of Bug 1. With only 4 subjects (4 × 100 = 400), the `total_max_marks` is 400. Once Bug 1 is fixed and A2 is included with `std_max=200`, the correct total becomes 400 + 200 = 600 which matches the OCR's `600/700` total column.

---

## 5. Layout Detection

The HSC transposed column layout is correctly identified:

```json
{"family": "hsc_transposed", "score": 95, "markers": ["MAXIMUM MARKS", "MARKS OBTAINED", "SUBJECT CODE"]}
```

The `_parse_hsc_ocr_directly()` function is invoked (bypassing Gemma4). The bugs are all within the Python deterministic parser, not in the LLM.

---

## 6. OCR Quality Notes

- PaddleOCR OCR missed several lines due to decorative fonts (line 3 reads "Lducation" instead of "Education")
- The name field OCR reads "NAMDED" instead of "NAMDEO" (OCR misread of 'O' as 'D')
- The subject code column has a visual gap at position 2 of the MARKS OBTAINED row (between 070 and 082) — the blank is because subject 40's obtained mark (082) is in column 3, not column 2 (the layout has an empty column between subject 01 and subject 40). The regex numerical extraction still correctly finds all 5 numeric values from the row.
- The OCR max marks row (`MAXIMUM MARKS 100 100 100 100 100 100 100 200 600/700`) has 7 × 100 values and 1 × 200 value, which is more columns than the 5 subject codes. This is because the document header row spans across both compulsory and optional subject columns — the extra 100-mark slots belong to columns without codes in the code line.

---

## 7. Fixes Applied

All four bugs were fixed in `src/basetruth/integrations/document_extract.py`:

| Bug | Function | Fix |
|---|---|---|
| A2 code not matched | `_parse_hsc_ocr_directly`, `_reformat_hsc_ocr_table` | Changed `\bA[Zz]\b` → `\bA[A-Za-z0-9]\b`; `std_max` pattern `[Aa][Zz]` → `[Aa][A-Za-z0-9]` |
| result="Distinction" | `_parse_hsc_ocr_directly` | Restricted result keyword search to lines[:40] only |
| Seat number = SR.NO. | `_parse_hsc_ocr_directly` | Prefer `[A-Z]\d{5,7}` letter-prefixed match before pure `\d{6}` fallback |
| total_max_marks=400 | Downstream of A2 fix | Fixed automatically when A2 is included with std_max=200 |
