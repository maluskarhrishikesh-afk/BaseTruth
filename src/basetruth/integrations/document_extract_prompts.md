# Document Extract Prompts

This file stores the prompt and rule text used by `document_extract.py`.
The Python module loads these sections on demand and caches them for the
current process.

## system
```text
You are an expert document OCR AI that extracts data from Indian documents. Return ONLY strict JSON — no explanation, no markdown fences.
```

## marksheet
```text
Analyse this marksheet image and extract all data into JSON.

FIRST: Is the image clear enough to read? If too blurry or dark, return ONLY:
{"document_type":"Unreadable","message":"Image is not clear enough to extract data."}

Return ONLY this JSON:
{
    "document_type": "Marksheet",
    "candidate_name": "",
    "board_or_university_name": "",
    "school_or_college_name": "",
    "examination_name": "",
    "month_year_of_passing": "",
    "enrollment_or_seat_number": "",
    "subjects": [
        {"subject_name": "", "marks_obtained": null, "max_marks": null}
    ],
    "printed_grand_total": null,
    "total_max_marks": null,
    "percentage_or_cgpa": "",
    "result": "",
    "extraction_confidence": "HIGH",
    "data_quality_notes": []
}

RULES:
1. Use the OCR structure hints and OCR text as the primary source of truth for row order, anchors, totals, and subject alignment.
2. Extract reliable anchors first: candidate_name, board_or_university_name, enrollment_or_seat_number, month_year_of_passing, result, printed_grand_total, and percentage_or_cgpa.
3. Ignore obvious OCR junk tokens that are not connected to a reliable anchor, subject header, numeric column, or result field.
4. Reconstruct the table by rows and columns, not by flat token order. Keep subject columns aligned with the corresponding max-marks and marks-obtained columns.
5. Extract only real subject rows. Do not create subjects from section headers like LANGUAGES, SCIENCES, COMPULSORY, OPTIONAL, TOTAL, or RESULT.
6. marks_obtained and max_marks must always be plain integers.
7. If the marksheet uses a two-row pattern, read the max-marks row and obtained-marks row column-by-column.
8. If the marksheet uses a transposed column-per-subject layout, use the printed subject code or printed subject label exactly as the subject_name.
9. If the marksheet uses multiple component rows per course, keep each printed component row as its own subject entry.
10. Compute sum(subjects[].marks_obtained) and compare it with printed_grand_total. If they differ, do NOT change marks to force a match. Add a data_quality_notes entry and lower extraction_confidence.
11. percentage_or_cgpa is the decimal percentage field, not the integer grand total.
12. If a field is unclear, leave it empty or null and explain the uncertainty in data_quality_notes. Never guess.
13. If the table itself is not recoverable, return subjects=[] and set extraction_confidence to LOW.
14. IGNOU open-university marksheets: When the structure hints indicate layout_family_hint=ignou_open_university, the printed TOTAL line shows a CREDIT-WEIGHTED AGGREGATE of assignment + term-end marks. This aggregate does NOT equal the sum of individual subject marks. Set printed_grand_total=null so the mismatch guard does not fire. Use ignou_subject_rows from the hints as your subjects list. Read candidate_name from the header (NAME: field) and enrollment_or_seat_number from the ENROLMENT NO field (10-digit number) — NOT from CERTIFICATE NO (6-digit serial number).
```

## educational
```text
Analyse this educational document image and extract all data into JSON.

FIRST: Is the image clear enough to read? If too blurry or dark, return ONLY:
{"document_type":"Unreadable","message":"Image is not clear enough to extract data."}

THEN determine type: "Marksheet" or "Degree Certificate".

FOR MARKSHEETS return:
{
  "document_type": "Marksheet",
  "candidate_name": "",
  "board_or_university_name": "",
  "school_or_college_name": "",
  "examination_name": "",
  "month_year_of_passing": "",
  "enrollment_or_seat_number": "",
  "subjects": [
    {"subject_name": "", "marks_obtained": null, "max_marks": null}
  ],
  "printed_grand_total": null,
  "total_max_marks": null,
  "percentage_or_cgpa": "",
  "result": "",
  "extraction_confidence": "HIGH",
  "data_quality_notes": []
}

FOR DEGREE CERTIFICATES return:
{
  "document_type": "Degree Certificate",
  "candidate_name": "",
  "university_name": "",
  "degree_name": "",
  "specialization_or_major": "",
  "division_or_class": "",
  "year_or_date_of_passing": "",
  "enrollment_number": "",
  "certificate_number": "",
  "date_of_issue": "",
  "data_quality_notes": []
}

CRITICAL RULES FOR MARKSHEETS:

1. SUBJECTS TABLE — REAL ROWS ONLY:
   Extract every INDIVIDUAL SUBJECT row from the marks table into the "subjects"
   array. Each entry must have:
   - "subject_name": the exact subject name (e.g. "English", "Mathematics",
     "Physics", "History"). This must be a real academic subject.
   - "marks_obtained": integer marks scored (whole number, never a decimal).
   - "max_marks": integer maximum marks for that row (e.g. 100, 150, 200).
   Do NOT put subject data in data_quality_notes — it must go in subjects.

2. REJECT SECTION HEADERS AS SUBJECTS:
   Indian marksheets often have SECTION LABELS printed above groups of subjects
   (e.g. "LANGUAGES", "COMPULSORY", "OPTIONAL", "SCIENCES", "VOCATIONAL").
   These are NOT subjects. Do NOT add them as rows in the subjects array.
   Only add rows that have an actual subject name AND its own marks column.

3. SAFE FAIL — TABLE NOT READABLE:
   If the marks table is not clearly visible, corrupted, or you cannot confidently
   identify individual subject rows with their marks, return:
     "subjects": []
     "extraction_confidence": "LOW"
   and explain why in data_quality_notes.
   *** NEVER guess marks or invent numbers just to fill the subjects array. ***
   *** Returning an empty subjects list is the correct and safe behaviour.  ***

4. DO NOT ADJUST MARKS TO MATCH TOTAL:
   After extracting, compute sum(marks_obtained). If it does not equal
   printed_grand_total, DO NOT modify any marks value.
   Instead, add a note to data_quality_notes describing the mismatch and set
   "extraction_confidence": "LOW". Fake consistency is worse than a known gap.

5. EXAMINATION NAME: Fill examination_name with the exam level as written —
   "SSC", "HSC", "10th Standard", "12th Standard", etc.
   SSC / 10th = Secondary School Certificate (age ~16).
   HSC / 12th = Higher Secondary Certificate (age ~18).
   Do NOT confuse the two.

6. NAME RECONSTRUCTION: If the candidate name is split across two print lines
   (e.g. "MALUSKAR HR" / "ISHIKESH NAMDEO"), join them: "MALUSKAR HRISHIKESH NAMDEO".

7. MONTH AND YEAR: Fill month_year_of_passing with both month and year when
   visible — e.g. "March 1999", not just "1999".

8. RESULT: Fill result with "PASS", "FAIL", "ATKT", or "DISTINCTION" as printed.

9. NUMBERS ONLY: marks_obtained and max_marks must be plain integers. Never use
   decimals or strings like "73 out of 100".

10. TOTAL MAX MARKS: Sum all max_marks values and store in total_max_marks.

11. CONFIDENCE: Set extraction_confidence to "HIGH" when you can clearly read the
    marks table, "MEDIUM" when some values are uncertain, "LOW" when the table is
    not clearly readable or you had to guess any marks.

12. TWO-ROW TABLE STRUCTURE (Most Important Rule for Indian Marksheets):
    Many Indian marksheets — especially Maharashtra Board SSC/HSC — print the
    marks table with TWO separate rows per subject group:
      Row A: Maximum marks  — what the student COULD score (e.g. 100, 150)
      Row B: Marks obtained — what the student ACTUALLY scored (e.g. 73, 134)

    You MUST read BOTH rows and map each column correctly:
      marks_obtained (Row B) → the SMALLER number in that column
      max_marks      (Row A) → the LARGER number in that column (the ceiling)

    WARNING: Do NOT use the max-marks row values as marks_obtained.
    This is the single most common mistake and produces impossibly large totals
    (e.g. 100+100+100+150+150+150 = 750 instead of 73+73+77+134+145+123 = 625).

    Example layout (read column by column):
      Subject:          English  Marathi  Hindi  Maths  Science  Soc.Sci
      Max marks (Row A):   100      100     100    150      150      150
      Obtained  (Row B):    73       73      77    134      145      123

    Correct extraction:
      {"subject_name": "English",   "marks_obtained": 73,  "max_marks": 100},
      {"subject_name": "Marathi",   "marks_obtained": 73,  "max_marks": 100},
      {"subject_name": "Hindi",     "marks_obtained": 77,  "max_marks": 100},
      {"subject_name": "Mathematics","marks_obtained": 134, "max_marks": 150},
      {"subject_name": "Science",   "marks_obtained": 145, "max_marks": 150},
      {"subject_name": "Social Sciences","marks_obtained": 123, "max_marks": 150}

13. PERCENTAGE VS GRAND TOTAL:
    The marks table on Indian marksheets usually contains two different summary
    numbers that look similar but mean different things:
      - GRAND TOTAL  : a plain INTEGER  (e.g. 627, 450, 360).  This is the
                       sum of all marks_obtained for all subjects.
                       → Store in printed_grand_total.
      - PERCENTAGE   : a DECIMAL number (e.g. 83.60, 75.20, 91.4 %).  This is
                       the grand total divided by total max marks × 100.
                       → Store in percentage_or_cgpa (as a string like "83.60").

    RULE: If a number has a decimal point (83.60), it is the PERCENTAGE — not the
    grand total.  Never put a decimal value in printed_grand_total.
    If you see only "83.60" and no integer total, look again — the integer total
    is almost certainly also printed on the marksheet (usually in the same row).

14. BOARD NAME — ENGLISH PREFERRED:
    If the board or university name is printed in a regional script (e.g. Marathi
    Devanāgarī), also include the standard English name in the same field so the
    record is searchable.  Example:
      "Maharashtra State Board of Secondary and Higher Secondary Education"
    If only the regional script version is visible, reproduce it exactly — do not
    leave the field blank.

15. HSC MARKSHEET — COLUMN-PER-SUBJECT (TRANSPOSED) TABLE:
    Maharashtra Board HSC / Higher Secondary marksheets use a TRANSPOSED table
    where SUBJECTS ARE COLUMNS, not rows.  The left side of the table has row
    labels in Marathi and English:
      विषयाचा सांकेतिक क्रमांक  / SUBJECT CODE
      कमाल गुण                 / MAXIMUM MARKS
      प्राप्त गुण               / MARKS OBTAINED

    To extract subjects, read LEFT-TO-RIGHT across the table:
      - Each numbered column (01, 40, 54, 55, AZ, etc.) is one subject.
      - subject_name: use the SUBJECT CODE value (e.g. "01", "40", "54", "55",
        "AZ") because full subject names are listed on the back of the marksheet.
      - max_marks   : the value in the MAXIMUM MARKS row for that column.
      - marks_obtained: the value in the MARKS OBTAINED row for that column.
      - SKIP any column where MARKS OBTAINED is blank or "--"
        (those are unused optional subject slots).
      - The RIGHTMOST column is the grand TOTAL column — do NOT treat it as
        a subject.

    Example layout (each column is a subject):
      SUBJECT CODE:     01   --   40   54   55  (blank) (blank)  AZ   TOTAL
      MAXIMUM MARKS:   100  100  100  100  100    100     100    200  600/700
      MARKS OBTAINED:  070   --  082  090  091     --      --    171   504

    Correct extraction:
      {"subject_name": "01", "marks_obtained": 70,  "max_marks": 100},
      {"subject_name": "40", "marks_obtained": 82,  "max_marks": 100},
      {"subject_name": "54", "marks_obtained": 90,  "max_marks": 100},
      {"subject_name": "55", "marks_obtained": 91,  "max_marks": 100},
      {"subject_name": "AZ", "marks_obtained": 171, "max_marks": 200}

    printed_grand_total: the numeric value in the TOTAL cell of MARKS OBTAINED
                         row (e.g. 504), NOT the 600/700 shown in MAXIMUM MARKS.
    total_max_marks    : sum of max_marks for subjects that were actually taken
                         (i.e. those with a marks_obtained value), e.g. 600.
    percentage_or_cgpa : value from the PERCENTAGE OF MARKS / गुणांची टक्केवारी
                         row (e.g. "84.00").

16. BE / ENGINEERING MARKSHEET — MULTI-COMPONENT ROWS PER COURSE:
    University of Pune (and similar) B.E./B.Tech marksheets list each course
    as MULTIPLE ROWS — one per assessment component.  The columns are:
      COURSE NAME   |  MAX  |  MIN  |  OBT

    Component abbreviations printed after the course name:
      PP  = Paper / Theory exam
      TW  = Term Work (internal assessment)
      OR  = Oral examination
      PR  = Practical examination

    Each component row is a SEPARATE subject entry.  Append the component type
    in parentheses to make the subject name unambiguous:
      subject_name  : course name + component, e.g.
                      "Computer Networks (PP)", "Computer Networks (TW)",
                      "Computer Networks (OR)"
      max_marks     : the MAX column value for that row
      marks_obtained: the OBT column value for that row

    The course code prefix (e.g. "010.", "05B.") may be included in subject_name
    or omitted — either is acceptable.

    GRAND TOTAL line at the bottom (e.g. "925/1500") gives:
      printed_grand_total: 925 (the obtained total)
      total_max_marks    : 1500 (the maximum total from the GRAND TOTAL line,
                           not recomputed from subjects — use the printed figure)

    Example for one course with three components:
      COURSE NAME                       MAX   MIN   OBT
      010. COMPUTER NETWORKS   PP       100    40    54
      010. COMPUTER NETWORKS   TW        25    10    15
      010. COMPUTER NETWORKS   OR        50    20    32

    Correct extraction:
      {"subject_name": "Computer Networks (PP)", "marks_obtained": 54, "max_marks": 100},
      {"subject_name": "Computer Networks (TW)", "marks_obtained": 15, "max_marks": 25},
      {"subject_name": "Computer Networks (OR)", "marks_obtained": 32, "max_marks": 50}
```

## financial
```text
Analyse this financial document image and extract all data into JSON.

FIRST: Is the image clear enough to read? If too blurry or dark, return ONLY:
{"document_type":"Unreadable","message":"Image is not clear enough to extract data."}

THEN determine the document type: "Payslip", "Bank Statement", "Form16",
"Increment Letter", "Gift Letter", or "Financial Document".

FOR PAYSLIPS return:
{
  "document_type": "Payslip",
  "employee_name": "",
  "employee_id": "",
  "designation": "",
  "department": "",
  "company_name": "",
  "location": "",
  "joining_date": "",
  "pan_number": "",
  "pf_number": "",
  "uan_number": "",
  "bank_account_last4": "",
  "pay_period": "",
  "basic_salary": null,
  "gross_salary": null,
  "total_deductions": null,
  "net_salary": null,
  "allowances": {
    "basic": null,
    "hra": null,
    "special_allowance": null
  },
  "deductions": {
    "provident_fund": null,
    "income_tax": null,
    "professional_tax": null
  },
  "data_quality_notes": []
}

CRITICAL RULES FOR PAYSLIPS:
1. EARNINGS: The "allowances" dict must contain EVERY individual earnings component
   you see in the Earnings/Income section (e.g. Basic, HRA, Special Allowance,
   Conveyance, Medical, LTA, Bonus, etc.). Add one key per component. Use lowercase
   snake_case keys (e.g. "special_allowance", "conveyance_allowance").
2. DEDUCTIONS: The "deductions" dict must contain EVERY individual deduction item
   you see (e.g. Provident Fund / PF, Income Tax / ITAX, Professional Tax / Prof Tax,
   ESIC, Loan EMI, etc.). Add one key per item. Use lowercase snake_case keys.
3. NUMBERS ONLY: All numeric values must be plain numbers with no currency symbols
   and no commas. Write 129791 not "₹1,29,791" or "1,29,791".
4. basic_salary is the "Basic" or "Basic Pay" earnings component value.
5. gross_salary is the Grand Total / Gross Pay of ALL earnings BEFORE any deduction.
6. total_deductions is the sum total of ALL deductions shown on the slip.
7. net_salary is the final take-home / Net Pay (gross_salary minus total_deductions).
8. CROSS-CHECK: After extracting, verify that the sum of all values in "allowances"
   equals gross_salary. If they differ, re-read the payslip carefully and fix it.
9. CROSS-CHECK: Verify that gross_salary - total_deductions equals net_salary.
   If they differ, re-check your values.
10. For Indian payslips, check for a separate Income Tax / Tax Worksheet section
    and always populate income_tax in deductions from there.
11. If any field is genuinely not present in the document, set it to null (not "N/A").

FOR BANK STATEMENTS return:
{
  "document_type": "Bank Statement",
  "account_holder_name": "",
  "account_number": "",
  "bank_name": "",
  "branch": "",
  "ifsc_code": "",
  "statement_period": "",
  "opening_balance": null,
  "closing_balance": null,
  "data_quality_notes": []
}

FOR FORM16 return:
{
  "document_type": "Form16",
  "employee_name": "",
  "pan_number": "",
  "employer_name": "",
  "employer_tan": "",
  "financial_year": "",
  "gross_salary": null,
  "total_tax_deducted": null,
  "data_quality_notes": []
}

FOR INCREMENT LETTERS return:
{
  "document_type": "Increment Letter",
  "employee_name": "",
  "company_name": "",
  "effective_date": "",
  "previous_salary": null,
  "new_salary": null,
  "increment_amount": null,
  "increment_percentage": null,
  "data_quality_notes": []
}

FOR GIFT LETTERS return:
{
  "document_type": "Gift Letter",
  "donor_name": "",
  "recipient_name": "",
  "relationship": "",
  "gift_amount": null,
  "gift_date": "",
  "purpose": "",
  "data_quality_notes": []
}

Return the most appropriate structure for the document type you identify.
```

## employment
```text
Analyse this employment document image and extract all data into JSON.

FIRST: Is the image clear enough to read? If too blurry or dark, return ONLY:
{"document_type":"Unreadable","message":"Image is not clear enough to extract data."}

THEN determine type: "Offer Letter" or "Employment Letter".

FOR OFFER LETTERS return:
{
  "document_type": "Offer Letter",
  "candidate_name": "",
  "company_name": "",
  "designation": "",
  "department": "",
  "joining_date": "",
  "ctc_per_annum": null,
  "gross_monthly": null,
  "offer_date": "",
  "location": "",
  "probation_period": "",
  "data_quality_notes": []
}

FOR EMPLOYMENT LETTERS return:
{
  "document_type": "Employment Letter",
  "employee_name": "",
  "company_name": "",
  "designation": "",
  "employment_start_date": "",
  "employment_end_date": "",
  "letter_date": "",
  "purpose": "",
  "data_quality_notes": []
}
```

## generic
```text
Analyse this document image and extract all readable fields into JSON.

FIRST: Is the image clear enough to read? If too blurry or dark, return ONLY:
{"document_type":"Unreadable","message":"Image is not clear enough to extract data."}

Extract all visible text fields into this structure:
{
  "document_type": "<describe the document type you see>",
  "extracted_fields": {
    "<field_name>": "<value>"
  },
  "data_quality_notes": []
}

Use descriptive keys for extracted_fields (e.g. "document_number", "issue_date", "name").
```