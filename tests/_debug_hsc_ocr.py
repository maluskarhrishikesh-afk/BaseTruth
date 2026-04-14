"""Quick debug script to test _reformat_hsc_ocr_table."""
import sys
sys.path.insert(0, "src")
from basetruth.integrations.document_extract import _reformat_hsc_ocr_table

# Use the actual OCR output from the HSC marksheet run
sample_ocr = (
    "PUNE\nJed RTTATT\n"
    "STATEMENT OF MARKS OFTHE HIGHER SECONDARY SCHOOL CEKTIFICATE EXAMINATION\n"
    "SEAT NO:\n005392 FEB-2001 SCIENCE 041 11. 007\n"
    "7       CANDIDATE'S FULL NAME (SURNAME First)\nMALUSKAR HRISHIKESH NAMDED\n"
    "fru Fqa\nOPTIONAL SUBJECTS VOCATIONAL SUBJECTS LANGUAGES\n"
    "SUBJECT\nTOTAL\nMARKS\n"
    "01 40 54 55 Az\nSUBJECICODE\n"
    "600 /700 200 1001 100 400 100 100\nMAXIMUM MARKS\n"
    "051 070 082 070 171 504\nMARKS OBTAINED\n"
    "TOTALMARKS\n# FOUR FIVE HUNDRED AND\n(NWORDS)\nFTa\n"
    "84_ 0o\nPass\nRESULT"
)

result = _reformat_hsc_ocr_table(sample_ocr)
print("=== RESULT ===")
marker = result.find("Reformatted")
if marker >= 0:
    print(result[marker:])
else:
    print("NO REFORMATTING DONE")
