#!/usr/bin/env python3
"""generate_fraud_cases.py – Targeted fraud scenario generator for BaseTruth.

Creates synthetic loan-document bundles (cases 051–060) where each case tests
a specific, real-world fraud pattern that is NOT covered by the original 50
cases.  These are intended as precise regression test cases.

Fraud scenarios
---------------
  case_051  ifsc_account_mismatch
              IFSC code on the bank statement header (e.g. HDFC0001234) does not
              match the IFSC on the payslip (ICIC0001234).  Suggests the applicant
              submitted a statement from a different account than the one to which
              salary is credited.

  case_052  salary_cross_doc_mismatch
              Payslip gross (₹1,44,000) is inflated ~75 % above actual bank salary
              credit (₹82,000/month) and well above what the employment letter CTC
              implies (₹72,000/month).  Classic income-inflation fraud.

  case_053  bank_date_range_fabricated
              Statement header says "01-Oct-2025 to 31-Mar-2026" (6 months) but
              all transactions fall within Jan–Mar 2026 (3 months).  The wide date
              range was added manually to look like a longer financial history.

  case_054  bank_address_mismatch
              IFSC prefix identifies the branch as HDFC Bank but the statement
              header names the bank as "State Bank of India, Delhi Main Branch."
              Impossible — the IFSC belongs to a different bank.

  case_055  bank_arithmetic_error
              One row in the running balance is deliberately wrong: the stated
              balance does not equal previous_balance + credit − debit.  Indicates
              a specific row was altered after the statement was generated.

  case_056  salary_structure_low_basic
              Basic salary is only 10 % of gross instead of the expected 40 %.
              The HRA component has been inflated to 80 % of gross, which is illegal
              under Section 10(13A) of the Income Tax Act.

  case_057  hra_exceeds_basic
              HRA (₹18,000) exceeds 50 % of basic (₹20,000 → max exempt = ₹10,000),
              inflating tax-exempt income.

  case_058  extreme_backdating
              Employment letter dated 15 March 2026 but join date is 01 January 2000
              (26 years earlier).  Extreme version of backdated employment.

  case_059  pan_mismatch
              PAN number on payslip (ABCDE1234F) differs from PAN on Form 16
              (XYZAB9876G).  Documents belong to different individuals.

  case_060  gift_letter_loan_disguise
              Gift letter amount (₹18,00,000) is 90 % of the property value
              (₹20,00,000).  This is suspicious — a valid down-payment gift should
              come from legitimate savings, not be the majority of the purchase price.
              Also, the same amount appears as an incoming transfer in the bank
              statement (possible round-trip loan disguised as a gift).

Usage
-----
  python scripts/generate_fraud_cases.py --out data/mortgage_docs
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from copy import deepcopy
from datetime import date, timedelta
from pathlib import Path

# Add the scripts directory to path so we can import the base generator
sys.path.insert(0, str(Path(__file__).parent))
from generate_mortgage_docs import (
    BANKS,
    CITIES,
    COMPANIES_REAL,
    DESIGNATIONS,
    LoanCase,
    MONTHS,
    _amount_words,
    _base_doc,
    _build_transactions,
    _last_day,
    _styles,
    _tbl_style,
    fake,
    fmt_inr,
    gen_bank_statement,
    gen_employment_letter,
    gen_form16,
    gen_gift_letter,
    gen_payslip,
    gen_property_agreement,
    gen_utility_bill,
    masked_account,
    random_account_no,
    random_pan,
    random_uan,
)

from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

random.seed(99)  # Deterministic so that test expectations are stable

# ---------------------------------------------------------------------------
# IFSC-to-bank mapping (subset of real Indian IFSC prefixes)
# Used to verify whether the bank name matches the IFSC in case_054.
# ---------------------------------------------------------------------------
IFSC_BANK_MAP = {
    "HDFC": "HDFC Bank",
    "ICIC": "ICICI Bank",
    "SBIN": "State Bank of India",
    "UTIB": "Axis Bank",
    "KKBK": "Kotak Mahindra Bank",
    "PUNB": "Punjab National Bank",
    "BARB": "Bank of Baroda",
    "CNRB": "Canara Bank",
}


# ---------------------------------------------------------------------------
# Helpers — bank statement generators for tampered variants
# ---------------------------------------------------------------------------

def gen_bank_statement_ifsc_mismatch(
    case: LoanCase,
    wrong_bank_name: str,
    wrong_ifsc: str,
    out_path: Path,
) -> None:
    """Bank statement where the IFSC/bank name in the header is DIFFERENT from
    what the payslip records.  The salary is still credited from the correct
    employer, but the statement itself belongs to a different account.

    Fraud signal: ifsc_account_mismatch
    """
    ss = _styles()
    doc = _base_doc(out_path)
    story = []

    # Header names a DIFFERENT bank than the payslip records
    story.append(Paragraph(wrong_bank_name.upper(), ss["CenterBold"]))
    story.append(Paragraph(
        f"Branch: {case.bank_branch}  |  IFSC: {wrong_ifsc}",
        ss["Center"]))
    story.append(HRFlowable(width="100%", thickness=1.5,
                             color=colors.HexColor("#1F3864")))
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph("ACCOUNT STATEMENT", ss["CenterBold"]))
    story.append(Spacer(1, 3 * mm))

    acc_info = [
        ["Account Holder", case.name, "Account No.", masked_account(case.account_no)],
        ["Account Type",   "Savings Account", "IFSC Code", wrong_ifsc],
        ["Branch",         case.bank_branch, "Statement Period", "01-Oct-2025 to 31-Mar-2026"],
    ]
    acc_tbl = Table(acc_info, colWidths=[3.8*cm, 6.2*cm, 3.8*cm, 5.7*cm])
    acc_tbl.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F5F8FE")),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CCCCCC")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(acc_tbl)
    story.append(Spacer(1, 5 * mm))

    txns = _build_transactions(case)
    txn_data = [["Date", "Description", "Ref No.", "Debit (₹)", "Credit (₹)", "Balance (₹)"]]
    for t in txns:
        txn_data.append([
            t["date"], t["desc"], t["ref"],
            f"{t['debit']:,}" if t["debit"] else "",
            f"{t['credit']:,}" if t["credit"] else "",
            f"{t['balance']:,}",
        ])
    txn_tbl = Table(txn_data, colWidths=[2.3*cm, 7.5*cm, 2.2*cm, 2.3*cm, 2.3*cm, 2.9*cm])
    txn_tbl.setStyle(_tbl_style())
    story.append(txn_tbl)
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph(
        "This is a digitally generated statement. For queries contact your branch or call 1800-XXX-XXXX.",
        ss["Small"]))
    doc.build(story)


def gen_bank_statement_short_range(case: LoanCase, out_path: Path) -> None:
    """Bank statement with a FABRICATED 6-month date range in the header but
    transactions that only span 3 months (Jan–Mar 2026).

    The header says "01-Oct-2025 to 31-Mar-2026" — a lie.
    Banks always cover the stated period fully; any gap means the Oct–Dec
    2025 section was removed (possibly to hide unfavourable transactions).

    Fraud signal: bank_date_range_fabricated / bank_statement_date_range
    """
    ss = _styles()
    doc = _base_doc(out_path)
    story = []

    story.append(Paragraph(case.bank_name.upper(), ss["CenterBold"]))
    story.append(Paragraph(
        f"Branch: {case.bank_branch}  |  IFSC: {case.ifsc}",
        ss["Center"]))
    story.append(HRFlowable(width="100%", thickness=1.5,
                             color=colors.HexColor("#1F3864")))
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph("ACCOUNT STATEMENT", ss["CenterBold"]))
    story.append(Spacer(1, 3 * mm))

    acc_info = [
        ["Account Holder", case.name, "Account No.", masked_account(case.account_no)],
        ["Account Type",   "Savings Account", "IFSC Code", case.ifsc],
        # Fabricated: header claims 6 months but transactions only cover 3
        ["Branch", case.bank_branch, "Statement Period", "01-Oct-2025 to 31-Mar-2026"],
    ]
    acc_tbl = Table(acc_info, colWidths=[3.8*cm, 6.2*cm, 3.8*cm, 5.7*cm])
    acc_tbl.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F5F8FE")),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CCCCCC")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(acc_tbl)
    story.append(Spacer(1, 5 * mm))

    # Only generate Jan–Mar 2026 transactions (3 months instead of 6)
    balance = random.randint(50_000, 200_000)
    txns = []
    months_data = [(2026, 1), (2026, 2), (2026, 3)]  # omits Oct-Dec 2025
    for yr, mo in months_data:
        last = _last_day(mo, yr)
        salary_credit = case.monthly_net
        balance += salary_credit
        txns.append({
            "date": f"01-{mo:02d}-{yr}",
            "desc": f"SALARY CREDIT - {case.employer_name[:30]}",
            "ref": f"NEFT{random.randint(100000, 999999)}",
            "debit": 0, "credit": salary_credit, "balance": balance,
        })
        debit_types = [
            ("NACH/EMI DEBIT", random.randint(8_000, 35_000)),
            ("UPI/GROCERY", random.randint(3_000, 12_000)),
            ("NEFT/RENT PAYMENT", random.randint(10_000, 25_000)),
        ]
        for _ in range(random.randint(2, 4)):
            if balance < 5_000:
                break
            desc, amt = random.choice(debit_types)
            amt = min(amt, balance - 2_000)
            day = random.randint(2, last)
            balance -= amt
            txns.append({
                "date": f"{day:02d}-{mo:02d}-{yr}",
                "desc": desc,
                "ref": f"TXN{random.randint(100000, 999999)}",
                "debit": amt, "credit": 0, "balance": balance,
            })
    txns.sort(key=lambda x: x["date"])

    txn_data = [["Date", "Description", "Ref No.", "Debit (₹)", "Credit (₹)", "Balance (₹)"]]
    for t in txns:
        txn_data.append([
            t["date"], t["desc"], t["ref"],
            f"{t['debit']:,}" if t["debit"] else "",
            f"{t['credit']:,}" if t["credit"] else "",
            f"{t['balance']:,}",
        ])
    txn_tbl = Table(txn_data, colWidths=[2.3*cm, 7.5*cm, 2.2*cm, 2.3*cm, 2.3*cm, 2.9*cm])
    txn_tbl.setStyle(_tbl_style())
    story.append(txn_tbl)
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph(
        "This is a digitally generated statement. For queries contact your branch or call 1800-XXX-XXXX.",
        ss["Small"]))
    doc.build(story)


def gen_bank_statement_arithmetic_error(case: LoanCase, out_path: Path) -> None:
    """Bank statement where one running-balance entry is deliberately wrong.

    Row 5 (approx.) will show a balance inflated by ₹1,00,000 to make the
    account look healthier.  All subsequent rows carry the error forward.

    Fraud signal: bank_debit_credit_arithmetic / bank_arithmetic_error
    """
    ss = _styles()
    doc = _base_doc(out_path)
    story = []

    story.append(Paragraph(case.bank_name.upper(), ss["CenterBold"]))
    story.append(Paragraph(
        f"Branch: {case.bank_branch}  |  IFSC: {case.ifsc}",
        ss["Center"]))
    story.append(HRFlowable(width="100%", thickness=1.5,
                             color=colors.HexColor("#1F3864")))
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph("ACCOUNT STATEMENT", ss["CenterBold"]))
    story.append(Spacer(1, 3 * mm))

    acc_info = [
        ["Account Holder", case.name, "Account No.", masked_account(case.account_no)],
        ["Account Type",   "Savings Account", "IFSC Code", case.ifsc],
        ["Branch", case.bank_branch, "Statement Period", "01-Oct-2025 to 31-Mar-2026"],
    ]
    acc_tbl = Table(acc_info, colWidths=[3.8*cm, 6.2*cm, 3.8*cm, 5.7*cm])
    acc_tbl.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F5F8FE")),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CCCCCC")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(acc_tbl)
    story.append(Spacer(1, 5 * mm))

    txns = _build_transactions(case)
    TAMPER_ROW = 4   # 0-indexed; tamper the 5th transaction
    TAMPER_AMOUNT = 100_000  # add ₹1 lakh phantom credit

    txn_data = [["Date", "Description", "Ref No.", "Debit (₹)", "Credit (₹)", "Balance (₹)"]]
    for idx, t in enumerate(txns):
        balance_display = t["balance"]
        if idx == TAMPER_ROW:
            # Inflate the balance on this row — the arithmetic is now broken
            balance_display = t["balance"] + TAMPER_AMOUNT
        txn_data.append([
            t["date"], t["desc"], t["ref"],
            f"{t['debit']:,}" if t["debit"] else "",
            f"{t['credit']:,}" if t["credit"] else "",
            f"{balance_display:,}",
        ])

    txn_tbl = Table(txn_data, colWidths=[2.3*cm, 7.5*cm, 2.2*cm, 2.3*cm, 2.3*cm, 2.9*cm])
    txn_tbl.setStyle(_tbl_style())
    story.append(txn_tbl)
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph(
        "This is a digitally generated statement. For queries contact your branch or call 1800-XXX-XXXX.",
        ss["Small"]))
    doc.build(story)


def gen_bank_statement_address_mismatch(
    case: LoanCase,
    correct_ifsc: str,
    wrong_bank_name: str,
    out_path: Path,
) -> None:
    """Bank statement where the BANK NAME in the header does not match the IFSC.

    Example: IFSC is "HDFC0001234" (HDFC Bank) but the header says
    "State Bank of India" — an impossible combination.  This happens when
    someone manually edits the bank name on a statement to match a different
    institution.

    Fraud signal: bank_address_mismatch
    """
    ss = _styles()
    doc = _base_doc(out_path)
    story = []

    # Wrong bank name in the header, but correct IFSC
    story.append(Paragraph(wrong_bank_name.upper(), ss["CenterBold"]))
    story.append(Paragraph(
        f"Branch: {case.bank_branch}  |  IFSC: {correct_ifsc}",
        ss["Center"]))
    story.append(HRFlowable(width="100%", thickness=1.5,
                             color=colors.HexColor("#1F3864")))
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph("ACCOUNT STATEMENT", ss["CenterBold"]))
    story.append(Spacer(1, 3 * mm))

    acc_info = [
        ["Account Holder", case.name, "Account No.", masked_account(case.account_no)],
        ["Account Type",   "Savings Account", "IFSC Code", correct_ifsc],
        # Wrong branch city — HDFC IFSC but branch says "Delhi Main Branch"
        ["Branch",         "Delhi Main Branch", "Statement Period", "01-Oct-2025 to 31-Mar-2026"],
    ]
    acc_tbl = Table(acc_info, colWidths=[3.8*cm, 6.2*cm, 3.8*cm, 5.7*cm])
    acc_tbl.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F5F8FE")),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CCCCCC")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(acc_tbl)
    story.append(Spacer(1, 5 * mm))

    txns = _build_transactions(case)
    txn_data = [["Date", "Description", "Ref No.", "Debit (₹)", "Credit (₹)", "Balance (₹)"]]
    for t in txns:
        txn_data.append([
            t["date"], t["desc"], t["ref"],
            f"{t['debit']:,}" if t["debit"] else "",
            f"{t['credit']:,}" if t["credit"] else "",
            f"{t['balance']:,}",
        ])
    txn_tbl = Table(txn_data, colWidths=[2.3*cm, 7.5*cm, 2.2*cm, 2.3*cm, 2.3*cm, 2.9*cm])
    txn_tbl.setStyle(_tbl_style())
    story.append(txn_tbl)
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph(
        "This is a digitally generated statement. For queries contact your branch or call 1800-XXX-XXXX.",
        ss["Small"]))
    doc.build(story)


def gen_payslip_inflated_cross_doc(
    case: LoanCase,
    month: int,
    year: int,
    inflated_gross: int,
    out_path: Path,
) -> None:
    """Payslip with gross salary inflated far above the bank salary credit.

    The payslip claims inflated_gross but the bank only shows case.monthly_net
    as salary credit — the cross-document mismatch is the fraud signal.

    Fraud signal: salary_cross_doc_mismatch
    """
    # gen_payslip already respects case.payslip_gross_override; use a shallow copy
    patched = deepcopy(case)
    patched.payslip_gross_override = inflated_gross
    gen_payslip(patched, month, year, out_path)


def gen_payslip_low_basic(
    case: LoanCase,
    month: int,
    year: int,
    out_path: Path,
) -> None:
    """Payslip where basic salary is only 10% of gross (should be ~40%).

    Pattern: employer inflates non-basic allowances (Special Allowance, HRA) to
    reduce PF liability and inflate tax-free income.

    Fraud signal: basic_gross_proportion, hra_basic_proportion
    """
    ss = _styles()
    doc = _base_doc(out_path, topmargin=1.5)
    story = []

    gross = case.monthly_gross
    # Fraudulent salary structure: basic only 10% (legal minimum is 30-50%)
    basic   = int(gross * 0.10)
    hra     = int(gross * 0.70)   # HRA at 70% of gross — vastly overstated
    conv    = 1_600
    medical = 1_250
    special = gross - basic - hra - conv - medical
    if special < 0:
        special = 0
    pf_emp  = int(basic * 0.12)   # PF on artificially low basic = tiny PF liability
    pt      = 200
    tds     = max(0, int(gross * 0.08) - 1_000)
    total_ded = pf_emp + pt + tds
    net_pay = gross - total_ded

    emp_name = case.employer_name
    emp_cin  = case.employer_cin

    story.append(Paragraph(emp_name.upper(), ss["CenterBold"]))
    story.append(Paragraph(f"CIN: {emp_cin or 'N/A'}  |  {case.employer_city}", ss["Center"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1F3864")))
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph(
        f"PAY SLIP FOR THE MONTH OF {MONTHS[month-1].upper()} {year}", ss["CenterBold"]))
    story.append(Spacer(1, 4 * mm))

    info = [
        ["Employee Name", case.name, "Employee ID", case.employee_id],
        ["Designation", case.designation, "Department", case.department],
        ["Date of Joining", case.join_date.strftime("%d-%b-%Y"), "PAN", case.pan],
        ["Bank", case.bank_name, "Account No.", masked_account(case.account_no)],
        ["UAN", case.uan, "Pay Period",
         f"01-{month:02d}-{year} to {_last_day(month, year):02d}-{month:02d}-{year}"],
    ]
    info_tbl = Table(info, colWidths=[3.8*cm, 6.2*cm, 3.8*cm, 5.7*cm])
    info_tbl.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F5F8FE")),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CCCCCC")),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(info_tbl)
    story.append(Spacer(1, 5 * mm))

    ed_header = [["EARNINGS", "AMOUNT (₹)", "DEDUCTIONS", "AMOUNT (₹)"]]
    ed_rows = [
        ["Basic Salary",         f"{basic:,}",   "Provident Fund (Emp)", f"{pf_emp:,}"],
        ["House Rent Allowance", f"{hra:,}",      "Professional Tax",    f"{pt:,}"],
        ["Conveyance Allowance", f"{conv:,}",     "Income Tax (TDS)",    f"{tds:,}"],
        ["Medical Allowance",    f"{medical:,}",  "",                    ""],
        ["Special Allowance",    f"{special:,}",  "",                    ""],
    ]
    ed_data = ed_header + ed_rows + [
        ["GROSS EARNINGS", f"{gross:,}", "TOTAL DEDUCTIONS", f"{total_ded:,}"],
    ]
    ed_tbl = Table(ed_data, colWidths=[5.5*cm, 3.2*cm, 5.5*cm, 3.3*cm])
    ed_style = _tbl_style(header_color=colors.HexColor("#2E4057"))
    ed_style.add("FONTNAME", (0, len(ed_data)-1), (-1, len(ed_data)-1), "Helvetica-Bold")
    ed_style.add("BACKGROUND", (0, len(ed_data)-1), (-1, len(ed_data)-1),
                 colors.HexColor("#D6E4FF"))
    ed_tbl.setStyle(ed_style)
    story.append(ed_tbl)
    story.append(Spacer(1, 4 * mm))

    net_data = [["NET PAY (TAKE HOME)", f"₹ {net_pay:,}",
                 f"Rupees: {_amount_words(net_pay)} Only"]]
    net_tbl = Table(net_data, colWidths=[5.5*cm, 3.2*cm, 9.0*cm])
    net_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#1F3864")),
        ("TEXTCOLOR",  (0, 0), (-1, -1), colors.white),
        ("FONTNAME",   (0, 0), (-1, -1), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(net_tbl)
    story.append(Spacer(1, 6 * mm))
    story.append(Paragraph(
        "This is a computer-generated payslip and does not require a signature.", ss["Small"]))
    doc.build(story)


def gen_form16_pan_mismatch(case: LoanCase, wrong_pan: str, out_path: Path) -> None:
    """Form 16 where the employee PAN is different from what appears on payslips.

    The payslip has case.pan; this Form 16 shows wrong_pan.  In a legitimate
    bundle, all documents must share the same PAN for the same individual.

    Fraud signal: pan_mismatch (cross-document check)
    """
    import random as _random

    ss = _styles()
    doc = _base_doc(out_path)
    story = []

    gross_py = case.monthly_gross * 12
    hra_py   = case.monthly_hra   * 12
    tds_py   = case.monthly_tds   * 12
    std_ded  = 50_000
    net_tax  = max(0, gross_py - std_ded - hra_py)
    emp_name = case.employer_name

    story.append(Paragraph("FORM 16 — CERTIFICATE OF TAX DEDUCTED AT SOURCE", ss["CenterBold"]))
    story.append(Paragraph("[Under section 203 of the Income-tax Act, 1961]", ss["Center"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1F3864")))
    story.append(Spacer(1, 4 * mm))

    meta = [
        ["Assessment Year", "2025-26", "TAN of Employer", f"PUNE{_random.randint(10000,99999)}E"],
        ["Name of Employer", emp_name, "CIN", case.employer_cin or "N/A"],
        # PAN here is WRONG — does not match case.pan on payslips
        ["Name of Employee", case.name, "PAN of Employee", wrong_pan],
        ["Period", "01-Apr-2025 to 31-Mar-2026", "Employee ID", case.employee_id],
    ]
    meta_tbl = Table(meta, colWidths=[4.2*cm, 7.5*cm, 3.8*cm, 4.0*cm])
    meta_tbl.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F5F8FE")),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CCCCCC")),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(meta_tbl)
    story.append(Spacer(1, 5 * mm))
    story.append(Paragraph("PART A — DETAILS OF TAX DEDUCTED AND DEPOSITED", ss["DocH2"]))

    earn_data = [
        ["Description", "Amount (₹)"],
        ["Gross Salary (Total)", f"{gross_py:,}"],
        ["  Less: House Rent Allowance (HRA)", f"({hra_py:,})"],
        ["  Less: Standard Deduction", f"({std_ded:,})"],
        ["Net Taxable Salary", f"{net_tax:,}"],
        ["Tax Deducted at Source (TDS)", f"{tds_py:,}"],
    ]
    earn_tbl = Table(earn_data, colWidths=[12*cm, 4*cm])
    earn_tbl.setStyle(_tbl_style())
    story.append(earn_tbl)
    story.append(Spacer(1, 5 * mm))
    story.append(Paragraph(
        f"Certified that the tax mentioned above has been deducted from the salary "
        f"of {case.name} (PAN: {wrong_pan}) and deposited to the credit of the "
        f"Central Government as per the provisions of the Income-tax Act.",
        ss["Body"]))
    story.append(Spacer(1, 10 * mm))
    story.append(Paragraph("Authorised Signatory — Accounts & Finance", ss["Body"]))
    story.append(Paragraph(emp_name, ss["Label"]))
    doc.build(story)


def gen_employment_letter_extreme_backdate(
    case: LoanCase,
    backdated_join: date,
    out_path: Path,
) -> None:
    """Employment letter with an extreme backdated join date (25+ years ago).

    Letter issue date is fixed at 15 March 2026; join date is backdated_join
    which is decades in the past.  This tests the employment_backdating_signal check.

    Fraud signal: employment_backdating_signal
    """
    ss = _styles()
    doc = _base_doc(out_path)
    story = []

    emp_name = case.employer_name
    cin      = case.employer_cin

    story.append(Paragraph(emp_name.upper(), ss["CenterBold"]))
    story.append(Paragraph(
        f"CIN: {cin or 'N/A'}  |  Human Resources Department  |  {case.employer_city}",
        ss["Center"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1F3864")))
    story.append(Spacer(1, 5 * mm))
    # Letter is issued TODAY (March 2026) — this is the letter_issue_date
    story.append(Paragraph(f"Date: {date(2026, 3, 15).strftime('%d %B %Y')}", ss["Right"]))
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph("TO WHOMSOEVER IT MAY CONCERN", ss["Label"]))
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph(
        f"This is to certify that <b>{case.name}</b>, bearing PAN <b>{case.pan}</b>, "
        f"is a confirmed, full-time employee of <b>{emp_name}</b> "
        f"(Employee ID: <b>{case.employee_id}</b>).",
        ss["Body"]))
    story.append(Spacer(1, 3 * mm))
    # Join date is backdated_join — far in the past relative to letter issue date
    story.append(Paragraph(
        f"He / She is currently serving as <b>{case.designation}</b> "
        f"in the <b>{case.department}</b> department, "
        f"with effect from <b>{backdated_join.strftime('%d %B %Y')}</b>.",
        ss["Body"]))
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph(
        f"His / Her current annual Cost-to-Company (CTC) is "
        f"<b>{fmt_inr(case.annual_ctc)}</b> per annum "
        f"(Gross Monthly: <b>{fmt_inr(case.monthly_gross)}</b>).",
        ss["Body"]))
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph(
        "This letter is issued at the request of the employee for the purpose of "
        "Home Loan Application and should not be construed as a guarantee of "
        "continuity of employment.",
        ss["Body"]))
    story.append(Spacer(1, 10 * mm))
    story.append(Paragraph("Authorised Signatory", ss["Body"]))
    story.append(Paragraph(f"Head – Human Resources, {emp_name}", ss["Body"]))
    story.append(Spacer(1, 15 * mm))
    story.append(Paragraph(
        "<i>Company Seal</i>",
        ParagraphStyle("seal", parent=ss["Body"], textColor=colors.lightgrey)))
    doc.build(story)


def gen_gift_letter_suspicious(
    case: LoanCase,
    gift_amount: int,
    property_value: int,
    out_path: Path,
) -> None:
    """Gift letter where the gift amount is ~ 90% of the property value.

    Legitimate gifts cover a portion of the down payment.  A gift covering
    almost the entire purchase price is suspicious: it may be a loan disguised
    as a gift to avoid repayment declaration, or circular money.

    Fraud signal: gift_amount_vs_property_value
    """
    ss = _styles()
    doc = _base_doc(out_path)
    story = []

    donor = fake.name()
    relationship = random.choice(["Father", "Mother", "Spouse"])

    story.append(Paragraph("GIFT DECLARATION LETTER", ss["CenterBold"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1F3864")))
    story.append(Spacer(1, 5 * mm))
    story.append(Paragraph(f"Date: {date(2026, 3, 10).strftime('%d %B %Y')}", ss["Right"]))
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph(
        f"I, <b>{donor}</b>, {relationship} of <b>{case.name}</b>, "
        f"do hereby solemnly declare that I have gifted a sum of "
        f"<b>{fmt_inr(gift_amount)}</b> "
        f"(Rupees {_amount_words(gift_amount)} only) to my "
        f"{relationship.lower()} for the purpose of home loan down payment.",
        ss["Body"]))
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph(
        f"Note: The property value is {fmt_inr(property_value)}. "
        f"The gift amount represents {gift_amount*100//property_value}% of the property value.",
        ss["Body"]))
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph(
        "I further declare that this amount is a gift and not a loan and "
        "no repayment is expected or required from the recipient.",
        ss["Body"]))
    story.append(Spacer(1, 10 * mm))
    story.append(Paragraph(f"Donor Name: {donor}", ss["Body"]))
    story.append(Paragraph(f"Relationship: {relationship}", ss["Body"]))
    story.append(Paragraph(f"PAN: {random_pan()}", ss["Body"]))
    story.append(Spacer(1, 10 * mm))
    story.append(Paragraph("Signature: _________________________", ss["Body"]))
    doc.build(story)


# ---------------------------------------------------------------------------
# Case builders — each returns a metadata dict for the case
# ---------------------------------------------------------------------------

def _base_loan_case(case_id: str) -> LoanCase:
    """Build a clean (non-tampered) base case for the fraud case to build on."""
    comp    = random.choice(COMPANIES_REAL)
    bank    = random.choice(BANKS)
    city    = random.choice(CITIES)
    desg, dept, ctc_min, ctc_max = random.choice(DESIGNATIONS)
    annual_ctc   = random.randrange(ctc_min, ctc_max, 10_000)
    monthly_gross = annual_ctc // 12
    basic   = int(monthly_gross * 0.40)
    hra     = int(monthly_gross * 0.20)
    conv    = 1_600
    medical = 1_250
    special = monthly_gross - basic - hra - conv - medical
    pf_emp  = int(basic * 0.12)
    pf_er   = int(basic * 0.12)
    pt      = 200
    tds     = max(0, int(monthly_gross * 0.08) - 1_000)
    net     = monthly_gross - pf_emp - pt - tds
    join_date = date(2020, 1, 1) - timedelta(days=random.randint(180, 1825))
    prop_value   = random.randrange(3_000_000, 15_000_000, 50_000)
    loan_amount  = int(prop_value * random.uniform(0.65, 0.80))
    return LoanCase(
        case_id=case_id,
        name=fake.name(),
        age=random.randint(28, 52),
        pan=random_pan(),
        city=city,
        address=fake.street_address() + f", {city}",
        phone=fake.phone_number()[:14],
        email=fake.email(),
        employer_name=comp[0],
        employer_cin=comp[1],
        employer_city=comp[2],
        designation=desg,
        department=dept,
        join_date=join_date,
        employee_id=f"EMP{random.randint(10000, 99999)}",
        uan=random_uan(),
        annual_ctc=annual_ctc,
        monthly_gross=monthly_gross,
        monthly_basic=basic,
        monthly_hra=hra,
        monthly_conv=conv,
        monthly_medical=medical,
        monthly_special=special,
        monthly_pf_emp=pf_emp,
        monthly_pf_er=pf_er,
        monthly_pt=pt,
        monthly_tds=tds,
        monthly_net=net,
        bank_name=bank[0],
        ifsc=bank[1],
        account_no=random_account_no(),
        bank_branch=f"{city} Main Branch",
        property_type="2BHK Apartment",
        property_address=fake.street_address() + f", {city}",
        property_value=prop_value,
        loan_amount=loan_amount,
        is_tampered=True,
    )


def build_case_051(cases_dir: Path) -> dict:
    """case_051 — ifsc_account_mismatch

    The payslip shows 'HDFC Bank, HDFC0001234' as the salary account.
    The bank statement submitted is from 'ICICI Bank, ICIC0001234'.
    The salary credits in the bank statement still match, but the account
    doesn't belong to the employee named on the payslip.
    """
    case_id = "case_051"
    case    = _base_loan_case(case_id)
    case.tamper_types = ["ifsc_account_mismatch"]
    # Force payslip to show HDFC bank
    case.bank_name = "HDFC Bank"
    case.ifsc      = "HDFC0001234"
    # Bank statement will be generated with ICICI IFSC (wrong account)
    wrong_ifsc      = "ICIC0009999"
    wrong_bank_name = "ICICI Bank"

    case_dir = cases_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    docs = {}
    for m, name in [(1, "jan"), (2, "feb"), (3, "mar")]:
        p = case_dir / f"payslip_2026_{name}.pdf"
        gen_payslip(case, m, 2026, p)
        docs[f"payslip_2026_{name}"] = str(p.relative_to(cases_dir.parent))

    p = case_dir / "bank_statement.pdf"
    gen_bank_statement_ifsc_mismatch(case, wrong_bank_name, wrong_ifsc, p)
    docs["bank_statement"] = str(p.relative_to(cases_dir.parent))

    for gen_fn, fname in [
        (lambda c, p: gen_employment_letter(c, p), "employment_letter"),
        (lambda c, p: gen_form16(c, p),            "form16"),
        (lambda c, p: gen_utility_bill(c, p),      "utility_bill"),
        (lambda c, p: gen_property_agreement(c, p), "property_agreement"),
    ]:
        p = case_dir / f"{fname}.pdf"
        gen_fn(case, p)
        docs[fname] = str(p.relative_to(cases_dir.parent))

    return _metadata_record(case, docs, [
        "ifsc_account_mismatch: IFSC on payslip (HDFC0001234) ≠ IFSC "
        "on bank statement (ICIC0009999). Statement belongs to a different account."
    ])


def build_case_052(cases_dir: Path) -> dict:
    """case_052 — salary_cross_doc_mismatch

    Three sources disagree on the applicant's salary:
      Payslips        : ₹1,44,000 gross/month (inflated ~73%)
      Bank statement  : ₹83,000 salary credit (actual net from real employer)
      Employment letter: CTC ₹10,00,000/year → ₹83,333/month gross
    The inconsistency across documents reveals the payslips are fabricated.
    """
    case_id = "case_052"
    case    = _base_loan_case(case_id)
    # Set a realistic base salary
    case.annual_ctc   = 1_000_000
    case.monthly_gross = 83_333
    case.monthly_basic = int(case.monthly_gross * 0.40)
    case.monthly_hra   = int(case.monthly_gross * 0.20)
    case.monthly_net   = 72_000
    case.tamper_types  = ["salary_cross_doc_mismatch"]

    inflated_gross = 144_000  # 73% above actual

    case_dir = cases_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    docs = {}
    for m, name in [(1, "jan"), (2, "feb"), (3, "mar")]:
        p = case_dir / f"payslip_2026_{name}.pdf"
        gen_payslip_inflated_cross_doc(case, m, 2026, inflated_gross, p)
        docs[f"payslip_2026_{name}"] = str(p.relative_to(cases_dir.parent))

    # Bank statement credits at actual monthly_net (not inflated gross)
    p = case_dir / "bank_statement.pdf"
    gen_bank_statement_base(case, p)  # standard statement — salary shows real amount
    docs["bank_statement"] = str(p.relative_to(cases_dir.parent))

    for gen_fn, fname in [
        (lambda c, p: gen_employment_letter(c, p), "employment_letter"),
        (lambda c, p: gen_form16(c, p),            "form16"),
        (lambda c, p: gen_utility_bill(c, p),      "utility_bill"),
        (lambda c, p: gen_property_agreement(c, p), "property_agreement"),
    ]:
        p = case_dir / f"{fname}.pdf"
        gen_fn(case, p)
        docs[fname] = str(p.relative_to(cases_dir.parent))

    return _metadata_record(case, docs, [
        f"salary_cross_doc_mismatch: Payslip gross ₹1,44,000 vs bank salary "
        f"credit ₹{case.monthly_net:,} vs employment letter implied monthly "
        f"₹{case.annual_ctc//12:,}. Payslips are inflated."
    ])


def build_case_053(cases_dir: Path) -> dict:
    """case_053 — bank_date_range_fabricated

    The bank statement header claims '01-Oct-2025 to 31-Mar-2026' (6 months)
    but actual transactions only cover Jan–Mar 2026 (3 months).
    The Oct–Dec 2025 section was removed — possibly to hide adverse entries
    (bounced cheques, low balance, loan repayments, or suspicious transfers).
    """
    case_id = "case_053"
    case    = _base_loan_case(case_id)
    case.tamper_types = ["bank_date_range_fabricated"]

    case_dir = cases_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    docs = {}
    for m, name in [(1, "jan"), (2, "feb"), (3, "mar")]:
        p = case_dir / f"payslip_2026_{name}.pdf"
        gen_payslip(case, m, 2026, p)
        docs[f"payslip_2026_{name}"] = str(p.relative_to(cases_dir.parent))

    p = case_dir / "bank_statement.pdf"
    gen_bank_statement_short_range(case, p)
    docs["bank_statement"] = str(p.relative_to(cases_dir.parent))

    for gen_fn, fname in [
        (lambda c, p: gen_employment_letter(c, p), "employment_letter"),
        (lambda c, p: gen_form16(c, p),            "form16"),
        (lambda c, p: gen_utility_bill(c, p),      "utility_bill"),
        (lambda c, p: gen_property_agreement(c, p), "property_agreement"),
    ]:
        p = case_dir / f"{fname}.pdf"
        gen_fn(case, p)
        docs[fname] = str(p.relative_to(cases_dir.parent))

    return _metadata_record(case, docs, [
        "bank_date_range_fabricated: Statement header says '01-Oct-2025 to "
        "31-Mar-2026' but all transactions are Jan–Mar 2026 only. "
        "Oct–Dec 2025 records were stripped."
    ])


def build_case_054(cases_dir: Path) -> dict:
    """case_054 — bank_address_mismatch

    IFSC code is 'HDFC0001234' (identifies HDFC Bank) but the statement
    header's bank name reads 'State Bank of India, Delhi Main Branch'.
    An IFSC is bank-specific so the combination is impossible.
    """
    case_id  = "case_054"
    case     = _base_loan_case(case_id)
    case.tamper_types = ["bank_address_mismatch"]
    # Use an HDFC IFSC
    correct_ifsc  = "HDFC0001234"
    # Wrong bank name that doesn't match HDFC IFSC prefix
    wrong_bank    = "State Bank of India"

    case_dir = cases_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    docs = {}
    for m, name in [(1, "jan"), (2, "feb"), (3, "mar")]:
        p = case_dir / f"payslip_2026_{name}.pdf"
        gen_payslip(case, m, 2026, p)
        docs[f"payslip_2026_{name}"] = str(p.relative_to(cases_dir.parent))

    p = case_dir / "bank_statement.pdf"
    gen_bank_statement_address_mismatch(case, correct_ifsc, wrong_bank, p)
    docs["bank_statement"] = str(p.relative_to(cases_dir.parent))

    for gen_fn, fname in [
        (lambda c, p: gen_employment_letter(c, p), "employment_letter"),
        (lambda c, p: gen_form16(c, p),            "form16"),
        (lambda c, p: gen_utility_bill(c, p),      "utility_bill"),
        (lambda c, p: gen_property_agreement(c, p), "property_agreement"),
    ]:
        p = case_dir / f"{fname}.pdf"
        gen_fn(case, p)
        docs[fname] = str(p.relative_to(cases_dir.parent))

    return _metadata_record(case, docs, [
        "bank_address_mismatch: IFSC 'HDFC0001234' belongs to HDFC Bank but "
        "the statement header says 'State Bank of India, Delhi Main Branch'. "
        "IFSC prefix and bank name are inconsistent."
    ])


def build_case_055(cases_dir: Path) -> dict:
    """case_055 — bank_arithmetic_error

    Row 5 of the running-balance column is inflated by ₹1,00,000 (phantom
    credit added manually).  All subsequent balances are incorrect as a result.
    Tampered to inflate average monthly balance to meet loan eligibility criteria.
    """
    case_id = "case_055"
    case    = _base_loan_case(case_id)
    case.tamper_types = ["bank_arithmetic_error"]

    case_dir = cases_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    docs = {}
    for m, name in [(1, "jan"), (2, "feb"), (3, "mar")]:
        p = case_dir / f"payslip_2026_{name}.pdf"
        gen_payslip(case, m, 2026, p)
        docs[f"payslip_2026_{name}"] = str(p.relative_to(cases_dir.parent))

    p = case_dir / "bank_statement.pdf"
    gen_bank_statement_arithmetic_error(case, p)
    docs["bank_statement"] = str(p.relative_to(cases_dir.parent))

    for gen_fn, fname in [
        (lambda c, p: gen_employment_letter(c, p), "employment_letter"),
        (lambda c, p: gen_form16(c, p),            "form16"),
        (lambda c, p: gen_utility_bill(c, p),      "utility_bill"),
        (lambda c, p: gen_property_agreement(c, p), "property_agreement"),
    ]:
        p = case_dir / f"{fname}.pdf"
        gen_fn(case, p)
        docs[fname] = str(p.relative_to(cases_dir.parent))

    return _metadata_record(case, docs, [
        "bank_arithmetic_error: Row 5's running balance is inflated by ₹1,00,000. "
        "Prev balance + credit − debit ≠ stated balance. Row was manually edited."
    ])


def build_case_056(cases_dir: Path) -> dict:
    """case_056 — salary_structure_low_basic

    Basic salary is only 10% of gross (₹8,333 on ₹83,333 gross) instead of
    the expected 40%.  HRA is 70% of gross (₹58,333) to maximise tax-exempt
    income.  PF liability is artificially reduced.

    Checks triggered: basic_gross_proportion (basic < 30%), hra_basic_proportion
    (HRA 700% of basic — hugely exceeds the 50% statutory ceiling).
    """
    case_id = "case_056"
    case    = _base_loan_case(case_id)
    case.annual_ctc    = 1_000_000
    case.monthly_gross = 83_333
    # Override component proportions to the fraudulent structure
    case.monthly_basic   = int(case.monthly_gross * 0.10)   # 10%
    case.monthly_hra     = int(case.monthly_gross * 0.70)   # 70%
    case.monthly_pf_emp  = int(case.monthly_basic * 0.12)
    case.monthly_pf_er   = int(case.monthly_basic * 0.12)
    case.monthly_net     = (
        case.monthly_gross
        - case.monthly_pf_emp
        - case.monthly_pt
        - case.monthly_tds
    )
    case.tamper_types = ["salary_structure_low_basic"]

    case_dir = cases_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    docs = {}
    for m, name in [(1, "jan"), (2, "feb"), (3, "mar")]:
        p = case_dir / f"payslip_2026_{name}.pdf"
        gen_payslip_low_basic(case, m, 2026, p)
        docs[f"payslip_2026_{name}"] = str(p.relative_to(cases_dir.parent))

    p = case_dir / "bank_statement.pdf"
    gen_bank_statement_base(case, p)
    docs["bank_statement"] = str(p.relative_to(cases_dir.parent))

    for gen_fn, fname in [
        (lambda c, p: gen_employment_letter(c, p), "employment_letter"),
        (lambda c, p: gen_form16(c, p),            "form16"),
        (lambda c, p: gen_utility_bill(c, p),      "utility_bill"),
        (lambda c, p: gen_property_agreement(c, p), "property_agreement"),
    ]:
        p = case_dir / f"{fname}.pdf"
        gen_fn(case, p)
        docs[fname] = str(p.relative_to(cases_dir.parent))

    return _metadata_record(case, docs, [
        "salary_structure_low_basic: Basic is 10% of gross (should be 30–50%). "
        "HRA is 70% of gross (700% of basic, legal max = 50% of basic). "
        "PF obligation minimised; tax-free income maximised illegally."
    ])


def build_case_057(cases_dir: Path) -> dict:
    """case_057 — hra_exceeds_basic

    HRA (₹15,000) on a basic of ₹20,000 = 75% of basic.
    Under Section 10(13A), the HRA exemption ceiling for metro cities is 50%
    of basic.  This payslip over-states HRA to create artificial tax savings.

    Checks triggered: hra_basic_proportion, hra_proportion
    """
    case_id = "case_057"
    case    = _base_loan_case(case_id)
    case.monthly_gross = 60_000
    case.monthly_basic = 20_000    # 33% of gross — normal
    case.monthly_hra   = 15_000    # 75% of basic — exceeds 50% legal max
    case.monthly_conv  = 1_600
    case.monthly_medical = 1_250
    case.monthly_special = (
        case.monthly_gross
        - case.monthly_basic
        - case.monthly_hra
        - case.monthly_conv
        - case.monthly_medical
    )
    case.annual_ctc    = case.monthly_gross * 12
    case.monthly_pf_emp  = int(case.monthly_basic * 0.12)
    case.monthly_pf_er   = int(case.monthly_basic * 0.12)
    case.monthly_net     = case.monthly_gross - case.monthly_pf_emp - 200 - case.monthly_tds
    case.tamper_types  = ["hra_exceeds_basic"]

    case_dir = cases_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    docs = {}
    for m, name in [(1, "jan"), (2, "feb"), (3, "mar")]:
        p = case_dir / f"payslip_2026_{name}.pdf"
        gen_payslip(case, m, 2026, p)
        docs[f"payslip_2026_{name}"] = str(p.relative_to(cases_dir.parent))

    p = case_dir / "bank_statement.pdf"
    gen_bank_statement_base(case, p)
    docs["bank_statement"] = str(p.relative_to(cases_dir.parent))

    for gen_fn, fname in [
        (lambda c, p: gen_employment_letter(c, p), "employment_letter"),
        (lambda c, p: gen_form16(c, p),            "form16"),
        (lambda c, p: gen_utility_bill(c, p),      "utility_bill"),
        (lambda c, p: gen_property_agreement(c, p), "property_agreement"),
    ]:
        p = case_dir / f"{fname}.pdf"
        gen_fn(case, p)
        docs[fname] = str(p.relative_to(cases_dir.parent))

    return _metadata_record(case, docs, [
        "hra_exceeds_basic: HRA ₹15,000 is 75% of basic ₹20,000. "
        "Legal max exemption under Sec 10(13A) for metro = 50% of basic. "
        "Excess HRA claimed to reduce taxable income."
    ])


def build_case_058(cases_dir: Path) -> dict:
    """case_058 — extreme_backdating

    Employment letter is dated 15 March 2026.
    Date of joining stated: 01 January 2000 — a 26-year gap!
    The applicant actually joined recently; the join date was rolled back
    to appear as a long-tenured employee meeting bank's stability criterion.

    Checks triggered: employment_backdating_signal
    """
    case_id = "case_058"
    case    = _base_loan_case(case_id)
    case.join_date    = date(2022, 6, 1)   # actual join date (4 years ago)
    case.tamper_types = ["extreme_backdating"]

    # The backdated join date applied to the employment letter only
    backdated_join = date(2000, 1, 1)   # 26 years before letter date

    case_dir = cases_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    docs = {}
    for m, name in [(1, "jan"), (2, "feb"), (3, "mar")]:
        p = case_dir / f"payslip_2026_{name}.pdf"
        gen_payslip(case, m, 2026, p)   # payslip shows real join date
        docs[f"payslip_2026_{name}"] = str(p.relative_to(cases_dir.parent))

    p = case_dir / "bank_statement.pdf"
    gen_bank_statement_base(case, p)
    docs["bank_statement"] = str(p.relative_to(cases_dir.parent))

    # Employment letter with extreme backdated join date
    p = case_dir / "employment_letter.pdf"
    gen_employment_letter_extreme_backdate(case, backdated_join, p)
    docs["employment_letter"] = str(p.relative_to(cases_dir.parent))

    for gen_fn, fname in [
        (lambda c, p: gen_form16(c, p),            "form16"),
        (lambda c, p: gen_utility_bill(c, p),      "utility_bill"),
        (lambda c, p: gen_property_agreement(c, p), "property_agreement"),
    ]:
        p = case_dir / f"{fname}.pdf"
        gen_fn(case, p)
        docs[fname] = str(p.relative_to(cases_dir.parent))

    return _metadata_record(case, docs, [
        f"extreme_backdating: Letter dated 15-Mar-2026; join date 01-Jan-2000 "
        f"(26 years gap). Actual join per payslip: {case.join_date}. "
        "Employment backdated to meet bank's vintage-of-employment criterion."
    ])


def build_case_059(cases_dir: Path) -> dict:
    """case_059 — pan_mismatch

    PAN on all three payslips: case.pan (e.g. ABCDE1234F)
    PAN on Form 16:            XYZAB9876G  (a different person's PAN)

    In a legitimate bundle, the PAN on payslips and Form 16 must be identical.
    A mismatch could mean borrowed documents, identity fraud, or clerical tampering.

    Checks triggered: cross-document pan_mismatch (service layer)
    """
    case_id    = "case_059"
    case       = _base_loan_case(case_id)
    wrong_pan  = random_pan()
    # Make sure it's actually different
    while wrong_pan == case.pan:
        wrong_pan = random_pan()
    case.tamper_types = ["pan_mismatch"]

    case_dir = cases_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    docs = {}
    for m, name in [(1, "jan"), (2, "feb"), (3, "mar")]:
        p = case_dir / f"payslip_2026_{name}.pdf"
        gen_payslip(case, m, 2026, p)   # correct PAN
        docs[f"payslip_2026_{name}"] = str(p.relative_to(cases_dir.parent))

    p = case_dir / "bank_statement.pdf"
    gen_bank_statement_base(case, p)
    docs["bank_statement"] = str(p.relative_to(cases_dir.parent))

    for gen_fn, fname in [
        (lambda c, p: gen_employment_letter(c, p), "employment_letter"),
        (lambda c, p: gen_utility_bill(c, p),      "utility_bill"),
        (lambda c, p: gen_property_agreement(c, p), "property_agreement"),
    ]:
        p = case_dir / f"{fname}.pdf"
        gen_fn(case, p)
        docs[fname] = str(p.relative_to(cases_dir.parent))

    # Form 16 with WRONG PAN
    p = case_dir / "form16.pdf"
    gen_form16_pan_mismatch(case, wrong_pan, p)
    docs["form16"] = str(p.relative_to(cases_dir.parent))

    return _metadata_record(case, docs, [
        f"pan_mismatch: Payslip PAN = {case.pan}; Form16 PAN = {wrong_pan}. "
        "Identity fraud — Form 16 belongs to a different individual."
    ])


def build_case_060(cases_dir: Path) -> dict:
    """case_060 — gift_letter_loan_disguise

    Gift amount (₹18,00,000) is 90% of the property value (₹20,00,000).
    The same amount appears as an incoming NEFT transfer in the bank statement.
    This pattern suggests the 'gift' is actually a back-channel loan that the
    applicant must repay — disguised as a gift to avoid EMI obligation declaration.

    Checks triggered: gift_amount_vs_property_value (service layer)
    """
    case_id = "case_060"
    case    = _base_loan_case(case_id)
    case.property_value  = 2_000_000
    case.loan_amount     = 1_600_000   # 80% LTV
    case.has_gift_letter = True
    case.tamper_types    = ["gift_letter_loan_disguise"]

    gift_amount = int(case.property_value * 0.90)  # 90% of property value

    case_dir = cases_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    docs = {}
    for m, name in [(1, "jan"), (2, "feb"), (3, "mar")]:
        p = case_dir / f"payslip_2026_{name}.pdf"
        gen_payslip(case, m, 2026, p)
        docs[f"payslip_2026_{name}"] = str(p.relative_to(cases_dir.parent))

    # Bank statement with a large incoming transfer matching the gift amount
    p = case_dir / "bank_statement.pdf"
    gen_bank_statement_with_large_inflow(case, gift_amount, p)
    docs["bank_statement"] = str(p.relative_to(cases_dir.parent))

    p = case_dir / "gift_letter.pdf"
    gen_gift_letter_suspicious(case, gift_amount, case.property_value, p)
    docs["gift_letter"] = str(p.relative_to(cases_dir.parent))

    for gen_fn, fname in [
        (lambda c, p: gen_employment_letter(c, p), "employment_letter"),
        (lambda c, p: gen_form16(c, p),            "form16"),
        (lambda c, p: gen_utility_bill(c, p),      "utility_bill"),
        (lambda c, p: gen_property_agreement(c, p), "property_agreement"),
    ]:
        p = case_dir / f"{fname}.pdf"
        gen_fn(case, p)
        docs[fname] = str(p.relative_to(cases_dir.parent))

    return _metadata_record(case, docs, [
        f"gift_letter_loan_disguise: Gift ₹{gift_amount:,} = 90% of property "
        f"₹{case.property_value:,}. Same amount appears as large NEFT inflow "
        "in bank statement — possible undisclosed back-channel loan."
    ])


# ---------------------------------------------------------------------------
# Shared base statement generator (clean, no tampering)
# ---------------------------------------------------------------------------

def gen_bank_statement_base(case: LoanCase, out_path: Path) -> None:
    """Standard 6-month bank statement (no tampering) used by fraud cases that
    only tamper one aspect of the bundle (e.g., payslip or employment letter).
    Thin wrapper around the original generator for clarity.
    """
    gen_bank_statement(case, out_path)


def gen_bank_statement_with_large_inflow(
    case: LoanCase,
    inflow_amount: int,
    out_path: Path,
) -> None:
    """Bank statement with a large unexplained incoming transfer matching the
    gift amount.  When this coincides with a gift letter, it signals that the
    'gift' funds are already in the account (loan proceeds, not genuine savings).

    Fraud signal: gift_letter_loan_disguise
    """
    ss = _styles()
    doc = _base_doc(out_path)
    story = []

    story.append(Paragraph(case.bank_name.upper(), ss["CenterBold"]))
    story.append(Paragraph(
        f"Branch: {case.bank_branch}  |  IFSC: {case.ifsc}", ss["Center"]))
    story.append(HRFlowable(width="100%", thickness=1.5,
                             color=colors.HexColor("#1F3864")))
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph("ACCOUNT STATEMENT", ss["CenterBold"]))
    story.append(Spacer(1, 3 * mm))

    acc_info = [
        ["Account Holder", case.name, "Account No.", masked_account(case.account_no)],
        ["Account Type",   "Savings Account", "IFSC Code", case.ifsc],
        ["Branch", case.bank_branch, "Statement Period", "01-Oct-2025 to 31-Mar-2026"],
    ]
    acc_tbl = Table(acc_info, colWidths=[3.8*cm, 6.2*cm, 3.8*cm, 5.7*cm])
    acc_tbl.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F5F8FE")),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CCCCCC")),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(acc_tbl)
    story.append(Spacer(1, 5 * mm))

    txns = _build_transactions(case)
    # Inject the large gift inflow on 10-Feb-2026
    balance = txns[-1]["balance"] if txns else 50_000
    balance += inflow_amount
    gift_txn = {
        "date": "10-02-2026",
        "desc": f"NEFT/GIFT TRANSFER - FAMILY",
        "ref":  f"NEFT{random.randint(100000, 999999)}",
        "debit": 0,
        "credit": inflow_amount,
        "balance": balance,
    }
    txns.append(gift_txn)
    txns.sort(key=lambda x: x["date"])

    txn_data = [["Date", "Description", "Ref No.", "Debit (₹)", "Credit (₹)", "Balance (₹)"]]
    for t in txns:
        txn_data.append([
            t["date"], t["desc"], t["ref"],
            f"{t['debit']:,}" if t["debit"] else "",
            f"{t['credit']:,}" if t["credit"] else "",
            f"{t['balance']:,}",
        ])
    txn_tbl = Table(txn_data, colWidths=[2.3*cm, 7.5*cm, 2.2*cm, 2.3*cm, 2.3*cm, 2.9*cm])
    txn_tbl.setStyle(_tbl_style())
    story.append(txn_tbl)
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph(
        "This is a digitally generated statement. For queries contact your branch.",
        ss["Small"]))
    doc.build(story)


# ---------------------------------------------------------------------------
# Metadata helper
# ---------------------------------------------------------------------------

def _metadata_record(case: LoanCase, docs: dict, notes: list) -> dict:
    """Build the metadata.json record for a fraud case."""
    return {
        "case_id": case.case_id,
        "applicant": {
            "name": case.name,
            "age": case.age,
            "pan": case.pan,
            "city": case.city,
            "phone": case.phone,
            "email": case.email,
        },
        "employer": {
            "name": case.employer_name,
            "cin": case.employer_cin,
            "city": case.employer_city,
            "designation": case.designation,
            "department": case.department,
            "join_date": case.join_date.isoformat(),
            "employee_id": case.employee_id,
            "annual_ctc": case.annual_ctc,
            "monthly_gross": case.monthly_gross,
            "monthly_net": case.monthly_net,
        },
        "bank": {
            "name": case.bank_name,
            "ifsc": case.ifsc,
            "account_no": masked_account(case.account_no),
        },
        "property": {
            "type": case.property_type,
            "address": case.property_address,
            "value": case.property_value,
            "loan_amount": case.loan_amount,
        },
        "is_tampered": True,
        "tamper_types": case.tamper_types,
        "fraud_notes": notes,   # human-readable explanation of the fraud in this case
        "documents": docs,
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

CASE_BUILDERS = [
    build_case_051,
    build_case_052,
    build_case_053,
    build_case_054,
    build_case_055,
    build_case_056,
    build_case_057,
    build_case_058,
    build_case_059,
    build_case_060,
]


def main() -> None:
    p = argparse.ArgumentParser(description="Targeted fraud case generator")
    p.add_argument("--out", default="data/mortgage_docs",
                   help="Output root directory (same as generate_mortgage_docs.py)")
    args = p.parse_args()

    out       = Path(args.out)
    cases_dir = out / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    fraud_metadata  = []
    fraud_csv_rows  = []

    print(f"Generating {len(CASE_BUILDERS)} targeted fraud cases → {out}")
    for builder in CASE_BUILDERS:
        record = builder(cases_dir)
        fraud_metadata.append(record)
        fraud_csv_rows.append({
            "case_id":       record["case_id"],
            "name":          record["applicant"]["name"],
            "city":          record["applicant"]["city"],
            "employer":      record["employer"]["name"],
            "annual_ctc":    record["employer"]["annual_ctc"],
            "property_value": record["property"]["value"],
            "loan_amount":   record["property"]["loan_amount"],
            "is_tampered":   True,
            "tamper_types":  "|".join(record["tamper_types"]),
        })
        print(f"  [{record['case_id']}] {record['tamper_types']} — done")

    # Append to existing metadata.json (or create if absent)
    meta_path = out / "metadata.json"
    if meta_path.exists():
        existing = json.loads(meta_path.read_text(encoding="utf-8"))
        if isinstance(existing, list):
            existing_ids = {r["case_id"] for r in existing}
            for rec in fraud_metadata:
                if rec["case_id"] not in existing_ids:
                    existing.append(rec)
                else:
                    # Replace the existing entry
                    existing = [rec if r["case_id"] == rec["case_id"] else r for r in existing]
            fraud_metadata_final = existing
        else:
            fraud_metadata_final = fraud_metadata
    else:
        fraud_metadata_final = fraud_metadata

    meta_path.write_text(
        json.dumps(fraud_metadata_final, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Append to existing labels.csv
    csv_path = out / "labels.csv"
    csv_fieldnames = [
        "case_id", "name", "city", "employer", "annual_ctc",
        "property_value", "loan_amount", "is_tampered", "tamper_types",
    ]
    mode = "a" if csv_path.exists() else "w"
    with open(csv_path, mode, newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=csv_fieldnames)
        if mode == "w":
            writer.writeheader()
        writer.writerows(fraud_csv_rows)

    print(f"\nDone. {len(CASE_BUILDERS)} fraud cases generated.")
    print(f"  metadata.json → {meta_path}")
    print(f"  labels.csv    → {csv_path}")


if __name__ == "__main__":
    main()
