#!/usr/bin/env python3
"""generate_mortgage_docs.py – Mortgage-grade synthetic document generator.

Generates realistic Indian home-loan supporting documents grouped into
'case' bundles. Each bundle = one loan applicant. ~30% of cases are
synthetically tampered for fraud-detection training/testing.

Document types per case
-----------------------
  payslip_2026_01.pdf   – January salary slip
  payslip_2026_02.pdf   – February salary slip
  payslip_2026_03.pdf   – March salary slip
  bank_statement.pdf    – 6-month bank statement with salary credits
  employment_letter.pdf – HR employment verification letter
  form16.pdf            – TDS certificate (Form 16 style)
  utility_bill.pdf      – Electricity bill for residency proof
  gift_letter.pdf       – Down-payment gift letter (40% of cases)
  property_agreement.pdf– Property sale agreement skeleton

Tamper variants (recorded in metadata.json)
-------------------------------------------
  income_inflated     – payslip gross > actual bank salary credit
  employer_fake       – employer CIN absent / company unregistered
  circular_funds      – large round-trip debit+credit same day in statement
  backdated_employment– employment letter join date predates company records

Usage
-----
  python scripts/generate_mortgage_docs.py --out data/mortgage_docs --n-cases 50
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import List, Optional

from faker import Faker
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

fake = Faker("en_IN")
random.seed(42)

PAGE_W, PAGE_H = A4

# ── Reference data ────────────────────────────────────────────────────────────

BANKS = [
    ("HDFC Bank",           "HDFC0001234", "4XX1"),
    ("ICICI Bank",          "ICIC0001234", "4XX2"),
    ("State Bank of India", "SBIN0001234", "4XX3"),
    ("Axis Bank",           "UTIB0001234", "4XX4"),
    ("Kotak Mahindra Bank", "KKBK0001234", "4XX5"),
]

CITIES = [
    "Mumbai", "Pune", "Bengaluru", "Hyderabad",
    "Chennai", "Delhi", "Ahmedabad", "Kolkata",
]

COMPANIES_REAL = [
    ("Infosys Limited",                  "L85110KA1981PLC013341", "Bengaluru"),
    ("Tata Consultancy Services Ltd",    "L22210MH1995PLC084781", "Mumbai"),
    ("Wipro Technologies Pvt. Ltd.",     "U72200KA1945C000221",   "Bengaluru"),
    ("HCL Technologies Ltd",             "L74140DL1994PLC055219", "Noida"),
    ("Tech Mahindra Limited",            "L64200MH1986PLC041953", "Pune"),
    ("Capgemini India Pvt. Ltd.",        "U72200MH1997FTC106896", "Mumbai"),
    ("Cognizant Technology Solutions",   "U72200TN1994PLC031108", "Chennai"),
    ("Larsen & Toubro Infotech Ltd",     "L72900GJ1997PLC032016", "Mumbai"),
    ("Persistent Systems Limited",       "L72300PN1990PLC056696", "Pune"),
    ("Mphasis Limited",                  "L30007KA2000PLC025294", "Bengaluru"),
]

COMPANIES_FAKE = [
    ("Synergex Solutions Pvt. Ltd.",    None, "Mumbai"),
    ("Brightpath Technologies",         None, "Pune"),
    ("NextGen Infratech Pvt. Ltd.",     None, "Delhi"),
    ("Veloce Software Pvt. Ltd.",       None, "Hyderabad"),
    ("Orbit Business Systems",          None, "Chennai"),
]

DESIGNATIONS = [
    # (title, department, ctc_min, ctc_max)
    ("Software Engineer",         "Engineering",     500_000,   900_000),
    ("Senior Software Engineer",  "Engineering",     800_000, 1_500_000),
    ("Technical Lead",            "Engineering",   1_200_000, 2_000_000),
    ("Product Manager",           "Product",       1_400_000, 2_500_000),
    ("Business Analyst",          "Consulting",      700_000, 1_400_000),
    ("Accounts Manager",          "Finance",         600_000, 1_200_000),
    ("HR Business Partner",       "Human Resources", 600_000, 1_100_000),
    ("Senior Data Analyst",       "Analytics",       900_000, 1_600_000),
    ("DevOps Engineer",           "Engineering",     900_000, 1_800_000),
    ("Project Manager",           "Delivery",      1_300_000, 2_200_000),
]

PROPERTY_TYPES = [
    "2BHK Apartment", "3BHK Apartment",
    "Independent House", "Resale Flat", "Row House",
]

UTILITY_PROVIDERS = [
    "Maharashtra State Electricity Distribution Co. Ltd. (MSEDCL)",
    "Brihanmumbai Electric Supply and Transport (BEST)",
    "Bangalore Electricity Supply Company (BESCOM)",
    "Hyderabad Metropolitan Water Supply & Sewerage Board",
    "Tamil Nadu Generation and Distribution Corporation (TANGEDCO)",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def fmt_inr(amount: int) -> str:
    """Format integer as Indian Rupee string with comma grouping."""
    s = str(abs(amount))
    if len(s) <= 3:
        result = s
    else:
        result = s[-3:]
        s = s[:-3]
        while len(s) > 2:
            result = s[-2:] + "," + result
            s = s[:-2]
        result = s + "," + result
    return ("₹" if amount >= 0 else "-₹") + result


def random_pan() -> str:
    alpha = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return (
        "".join(random.choices(alpha, k=5))
        + "".join(random.choices("0123456789", k=4))
        + random.choice(alpha)
    )


def random_account_no() -> str:
    return "".join(random.choices("0123456789", k=12))


def random_uan() -> str:
    return "".join(random.choices("0123456789", k=12))


def masked_account(acc: str) -> str:
    return "XXXX XXXX " + acc[-4:]


def _base_doc(path: Path, topmargin: float = 2.0) -> SimpleDocTemplate:
    return SimpleDocTemplate(
        str(path),
        pagesize=A4,
        rightMargin=1.8 * cm,
        leftMargin=1.8 * cm,
        topMargin=topmargin * cm,
        bottomMargin=1.5 * cm,
    )


def _styles():
    ss = getSampleStyleSheet()
    extras = [
        ParagraphStyle("DocH1",      parent=ss["Heading1"], fontSize=14, spaceAfter=4),
        ParagraphStyle("DocH2",      parent=ss["Heading2"], fontSize=11, spaceAfter=3),
        ParagraphStyle("Body",       parent=ss["Normal"],   fontSize=9,  spaceAfter=2),
        ParagraphStyle("SmallBold",  parent=ss["Normal"],   fontSize=8,
                       fontName="Helvetica-Bold"),
        ParagraphStyle("Center",     parent=ss["Normal"],   fontSize=9,  alignment=1),
        ParagraphStyle("CenterBold", parent=ss["Normal"],   fontSize=11, alignment=1,
                       fontName="Helvetica-Bold"),
        ParagraphStyle("Right",      parent=ss["Normal"],   fontSize=9,  alignment=2),
        ParagraphStyle("Small",      parent=ss["Normal"],   fontSize=8),
        ParagraphStyle("Label",      parent=ss["Normal"],   fontSize=9,
                       fontName="Helvetica-Bold"),
    ]
    for style in extras:
        ss.add(style)
    return ss


def _tbl_style(header_color=colors.HexColor("#1F3864"),
               row_color=colors.HexColor("#EBF0FA"),
               alt_color=colors.HexColor("#FFFFFF")) -> TableStyle:
    return TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0),  header_color),
        ("TEXTCOLOR",    (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",     (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, 0),  9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [row_color, alt_color]),
        ("FONTSIZE",     (0, 1), (-1, -1), 8),
        ("GRID",         (0, 0), (-1, -1), 0.35, colors.HexColor("#CCCCCC")),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING",   (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 3),
    ])


# ── Case dataclass ────────────────────────────────────────────────────────────

@dataclass
class LoanCase:
    case_id: str
    name: str
    age: int
    pan: str
    city: str
    address: str
    phone: str
    email: str
    # Employment (actual / real)
    employer_name: str
    employer_cin: Optional[str]
    employer_city: str
    designation: str
    department: str
    join_date: date
    employee_id: str
    uan: str
    # Salary (what bank actually shows)
    annual_ctc: int
    monthly_gross: int
    monthly_basic: int
    monthly_hra: int
    monthly_conv: int
    monthly_medical: int
    monthly_special: int
    monthly_pf_emp: int
    monthly_pf_er: int
    monthly_pt: int
    monthly_tds: int
    monthly_net: int
    # Bank
    bank_name: str
    ifsc: str
    account_no: str
    bank_branch: str
    # Property
    property_type: str
    property_address: str
    property_value: int
    loan_amount: int
    # Tamper
    is_tampered: bool = False
    tamper_types: List[str] = field(default_factory=list)
    # Override values used only in tampered documents
    payslip_gross_override: Optional[int] = None
    employer_name_override: Optional[str] = None
    employer_cin_override: Optional[str] = None
    join_date_override: Optional[date] = None
    has_gift_letter: bool = False


def _make_case(case_id: str) -> LoanCase:
    name = fake.name()
    city = random.choice(CITIES)
    desg, dept, ctc_min, ctc_max = random.choice(DESIGNATIONS)
    annual_ctc = random.randrange(ctc_min, ctc_max, 10_000)
    monthly_gross = annual_ctc // 12
    basic = int(monthly_gross * 0.40)
    hra = int(monthly_gross * 0.20)
    conv = 1_600
    medical = 1_250
    special = monthly_gross - basic - hra - conv - medical
    pf_emp = int(basic * 0.12)
    pf_er = int(basic * 0.12)
    pt = 200
    tds = max(0, int(monthly_gross * 0.08) - 1_000)
    net = monthly_gross - pf_emp - pt - tds
    bank = random.choice(BANKS)
    comp = random.choice(COMPANIES_REAL)
    join_date = date(2026, 1, 1) - timedelta(days=random.randint(180, 3650))
    prop_value = random.randrange(3_000_000, 15_000_000, 50_000)
    loan_amount = int(prop_value * random.uniform(0.65, 0.80))
    return LoanCase(
        case_id=case_id,
        name=name,
        age=random.randint(25, 55),
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
        property_type=random.choice(PROPERTY_TYPES),
        property_address=fake.street_address() + f", {city}",
        property_value=prop_value,
        loan_amount=loan_amount,
        has_gift_letter=random.random() < 0.40,
    )


def _apply_tamper(case: LoanCase) -> LoanCase:
    case = deepcopy(case)
    case.is_tampered = True
    pool = ["income_inflated", "employer_fake", "circular_funds", "backdated_employment"]
    chosen = random.sample(pool, random.randint(1, 2))
    case.tamper_types = chosen
    if "income_inflated" in chosen:
        mult = random.uniform(1.40, 1.75)
        case.payslip_gross_override = int(case.monthly_gross * mult)
    if "employer_fake" in chosen:
        fc = random.choice(COMPANIES_FAKE)
        case.employer_name_override = fc[0]
        case.employer_cin_override = None
    if "backdated_employment" in chosen:
        case.join_date_override = case.join_date - timedelta(
            days=random.randint(365, 1825)
        )
    return case


# ── Document generators ───────────────────────────────────────────────────────

MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]


def gen_payslip(case: LoanCase, month: int, year: int, out_path: Path) -> None:
    """Generates an Indian-style payslip PDF."""
    ss = _styles()
    doc = _base_doc(out_path, topmargin=1.5)
    story = []

    gross = case.payslip_gross_override if case.payslip_gross_override else case.monthly_gross
    # Recompute components proportionally if overridden
    scale = gross / case.monthly_gross if case.monthly_gross else 1
    basic   = int(case.monthly_basic  * scale)
    hra     = int(case.monthly_hra    * scale)
    conv    = case.monthly_conv
    medical = case.monthly_medical
    special = gross - basic - hra - conv - medical
    pf_emp  = int(basic * 0.12)
    pt      = case.monthly_pt
    tds     = max(0, int(gross * 0.08) - 1_000)
    total_ded = pf_emp + pt + tds
    net_pay = gross - total_ded

    emp_name = case.employer_name_override or case.employer_name
    emp_cin  = case.employer_cin_override if "employer_fake" in case.tamper_types else case.employer_cin
    join_dt  = case.join_date_override or case.join_date

    # Header
    story.append(Paragraph(emp_name.upper(), ss["CenterBold"]))
    story.append(Paragraph(
        f"CIN: {emp_cin or 'N/A'}  |  {case.employer_city}",
        ss["Center"]))
    story.append(HRFlowable(width="100%", thickness=1.5,
                             color=colors.HexColor("#1F3864")))
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph(
        f"PAY SLIP FOR THE MONTH OF {MONTHS[month-1].upper()} {year}",
        ss["CenterBold"]))
    story.append(Spacer(1, 4 * mm))

    # Employee info table
    info = [
        ["Employee Name", case.name,          "Employee ID",    case.employee_id],
        ["Designation",   case.designation,   "Department",     case.department],
        ["Date of Joining", join_dt.strftime("%d-%b-%Y"),
         "PAN",            case.pan],
        ["Bank",          case.bank_name,     "Account No.",    masked_account(case.account_no)],
        ["UAN",           case.uan,           "Pay Period",
         f"01-{month:02d}-{year} to {_last_day(month, year):02d}-{month:02d}-{year}"],
    ]
    info_tbl = Table(info, colWidths=[3.8 * cm, 6.2 * cm, 3.8 * cm, 5.7 * cm])
    info_tbl.setStyle(TableStyle([
        ("FONTNAME",  (0, 0), (-1, -1), "Helvetica"),
        ("FONTNAME",  (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME",  (2, 0), (2, -1), "Helvetica-Bold"),
        ("FONTSIZE",  (0, 0), (-1, -1), 8),
        ("BACKGROUND",(0, 0), (-1, -1), colors.HexColor("#F5F8FE")),
        ("GRID",      (0, 0), (-1, -1), 0.3, colors.HexColor("#CCCCCC")),
        ("TOPPADDING",(0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(info_tbl)
    story.append(Spacer(1, 5 * mm))

    # Earnings & Deductions
    ed_header = [["EARNINGS", "AMOUNT (₹)", "DEDUCTIONS", "AMOUNT (₹)"]]
    ed_rows = [
        ["Basic Salary",          f"{basic:,}",   "Provident Fund (Emp)",  f"{pf_emp:,}"],
        ["House Rent Allowance",  f"{hra:,}",     "Professional Tax",      f"{pt:,}"],
        ["Conveyance Allowance",  f"{conv:,}",    "Income Tax (TDS)",      f"{tds:,}"],
        ["Medical Allowance",     f"{medical:,}", "",                      ""],
        ["Special Allowance",     f"{special:,}", "",                      ""],
    ]
    ed_data = ed_header + ed_rows + [
        ["GROSS EARNINGS", f"{gross:,}", "TOTAL DEDUCTIONS", f"{total_ded:,}"],
    ]
    ed_tbl = Table(ed_data, colWidths=[5.5 * cm, 3.2 * cm, 5.5 * cm, 3.3 * cm])
    ed_style = _tbl_style(header_color=colors.HexColor("#2E4057"))
    ed_style.add("FONTNAME", (0, len(ed_data)-1), (-1, len(ed_data)-1), "Helvetica-Bold")
    ed_style.add("BACKGROUND", (0, len(ed_data)-1), (-1, len(ed_data)-1),
                 colors.HexColor("#D6E4FF"))
    ed_tbl.setStyle(ed_style)
    story.append(ed_tbl)
    story.append(Spacer(1, 4 * mm))

    # Net pay box
    net_data = [["NET PAY (TAKE HOME)", f"₹ {net_pay:,}",
                 f"Rupees: {_amount_words(net_pay)} Only"]]
    net_tbl = Table(net_data, colWidths=[5.5 * cm, 3.2 * cm, 9.0 * cm])
    net_tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, -1), colors.HexColor("#1F3864")),
        ("TEXTCOLOR",    (0, 0), (-1, -1), colors.white),
        ("FONTNAME",     (0, 0), (-1, -1), "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, -1), 9),
        ("TOPPADDING",   (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
    ]))
    story.append(net_tbl)
    story.append(Spacer(1, 6 * mm))
    story.append(Paragraph(
        "This is a computer-generated payslip and does not require a signature.",
        ss["Small"]))

    doc.build(story)


def gen_bank_statement(case: LoanCase, out_path: Path) -> None:
    """Generates a 6-month bank statement PDF with salary credits."""
    ss = _styles()
    doc = _base_doc(out_path)
    story = []

    # Header
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
        ["Account Holder", case.name,                         "Account No.", masked_account(case.account_no)],
        ["Account Type",   "Savings Account",                 "IFSC Code",   case.ifsc],
        ["Branch",         case.bank_branch,                  "Statement Period",
         "01-Oct-2025 to 31-Mar-2026"],
    ]
    acc_tbl = Table(acc_info, colWidths=[3.8*cm, 6.2*cm, 3.8*cm, 5.7*cm])
    acc_tbl.setStyle(TableStyle([
        ("FONTNAME",  (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME",  (2, 0), (2, -1), "Helvetica-Bold"),
        ("FONTSIZE",  (0, 0), (-1, -1), 8),
        ("BACKGROUND",(0, 0), (-1, -1), colors.HexColor("#F5F8FE")),
        ("GRID",      (0, 0), (-1, -1), 0.3, colors.HexColor("#CCCCCC")),
        ("TOPPADDING",(0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(acc_tbl)
    story.append(Spacer(1, 5 * mm))

    # Generate transactions
    txns = _build_transactions(case)

    txn_data = [["Date", "Description", "Ref No.", "Debit (₹)", "Credit (₹)", "Balance (₹)"]]
    for t in txns:
        txn_data.append([
            t["date"], t["desc"], t["ref"],
            f"{t['debit']:,}" if t["debit"] else "",
            f"{t['credit']:,}" if t["credit"] else "",
            f"{t['balance']:,}",
        ])

    txn_tbl = Table(txn_data,
                    colWidths=[2.3*cm, 7.5*cm, 2.2*cm, 2.3*cm, 2.3*cm, 2.9*cm])
    txn_tbl.setStyle(_tbl_style())
    story.append(txn_tbl)
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph(
        "This is a digitally generated statement. For queries contact your branch or call 1800-XXX-XXXX.",
        ss["Small"]))

    doc.build(story)


def _build_transactions(case: LoanCase):
    """Build 6 months of transactions starting Oct 2025."""
    balance = random.randint(50_000, 200_000)
    txns = []
    months_data = [
        (2025, 10), (2025, 11), (2025, 12),
        (2026, 1),  (2026, 2),  (2026, 3),
    ]
    for yr, mo in months_data:
        last = _last_day(mo, yr)
        # Salary credit on 1st
        salary_credit = case.monthly_net
        balance += salary_credit
        txns.append({
            "date": f"01-{mo:02d}-{yr}",
            "desc": f"SALARY CREDIT - {(case.employer_name_override or case.employer_name)[:30]}",
            "ref": f"NEFT{random.randint(100000,999999)}",
            "debit": 0, "credit": salary_credit, "balance": balance,
        })
        # 3-6 random debits (rent, grocery, EMI, utilities)
        debit_types = [
            ("NACH/EMI DEBIT",          random.randint(8_000, 35_000)),
            ("UPI/GROCERY",             random.randint(3_000, 12_000)),
            ("NEFT/RENT PAYMENT",       random.randint(10_000, 25_000)),
            ("UTILITY BILL PAYMENT",    random.randint(1_500, 5_000)),
            ("ATM WITHDRAWAL",          random.randint(5_000, 20_000)),
            ("POS PURCHASE",            random.randint(500, 8_000)),
        ]
        for _ in range(random.randint(3, 6)):
            if balance < 5_000:
                break
            desc, amt = random.choice(debit_types)
            amt = min(amt, balance - 2_000)
            day = random.randint(2, last)
            balance -= amt
            txns.append({
                "date": f"{day:02d}-{mo:02d}-{yr}",
                "desc": desc,
                "ref": f"TXN{random.randint(100000,999999)}",
                "debit": amt, "credit": 0, "balance": balance,
            })
        # Circular funds tamper: large debit then equal credit same day
        if "circular_funds" in case.tamper_types and mo == 2026:
            big = random.randint(300_000, 800_000)
            day = random.randint(5, 25)
            balance -= big
            txns.append({
                "date": f"{day:02d}-{mo:02d}-{yr}",
                "desc": "NEFT/SELF TRANSFER OUT",
                "ref": f"ST{random.randint(100000,999999)}",
                "debit": big, "credit": 0, "balance": balance,
            })
            balance += big
            txns.append({
                "date": f"{day:02d}-{mo:02d}-{yr}",
                "desc": "NEFT/SELF TRANSFER IN",
                "ref": f"ST{random.randint(100000,999999)}",
                "debit": 0, "credit": big, "balance": balance,
            })
    txns.sort(key=lambda x: x["date"])
    return txns


def gen_employment_letter(case: LoanCase, out_path: Path) -> None:
    """Generates an employment/HR verification letter."""
    ss = _styles()
    doc = _base_doc(out_path)
    story = []

    emp_name = case.employer_name_override or case.employer_name
    cin      = case.employer_cin_override if "employer_fake" in case.tamper_types else case.employer_cin
    join_dt  = case.join_date_override or case.join_date

    story.append(Paragraph(emp_name.upper(), ss["CenterBold"]))
    story.append(Paragraph(
        f"CIN: {cin or 'N/A'}  |  Human Resources Department  |  {case.employer_city}",
        ss["Center"]))
    story.append(HRFlowable(width="100%", thickness=1.5,
                             color=colors.HexColor("#1F3864")))
    story.append(Spacer(1, 5 * mm))
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
    story.append(Paragraph(
        f"He / She is currently serving as <b>{case.designation}</b> "
        f"in the <b>{case.department}</b> department, "
        f"with effect from <b>{join_dt.strftime('%d %B %Y')}</b>.",
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


def gen_form16(case: LoanCase, out_path: Path) -> None:
    """Generates a Form 16 (TDS Certificate) PDF."""
    ss = _styles()
    doc = _base_doc(out_path)
    story = []

    gross_py = case.monthly_gross * 12
    basic_py = case.monthly_basic * 12
    hra_py   = case.monthly_hra   * 12
    tds_py   = case.monthly_tds   * 12
    std_ded  = 50_000
    net_tax  = max(0, gross_py - std_ded - hra_py)
    emp_name = case.employer_name_override or case.employer_name
    cin      = case.employer_cin_override if "employer_fake" in case.tamper_types else case.employer_cin

    story.append(Paragraph("FORM 16 — CERTIFICATE OF TAX DEDUCTED AT SOURCE", ss["CenterBold"]))
    story.append(Paragraph("[Under section 203 of the Income-tax Act, 1961]", ss["Center"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1F3864")))
    story.append(Spacer(1, 4 * mm))

    meta = [
        ["Assessment Year", "2025-26",           "TAN of Employer", f"PUNE{random.randint(10000,99999)}E"],
        ["Name of Employer", emp_name,            "CIN",             cin or "N/A"],
        ["Name of Employee", case.name,           "PAN of Employee", case.pan],
        ["Period",           "01-Apr-2025 to 31-Mar-2026", "Employee ID", case.employee_id],
    ]
    meta_tbl = Table(meta, colWidths=[4.2*cm, 7.5*cm, 3.8*cm, 4.0*cm])
    meta_tbl.setStyle(TableStyle([
        ("FONTNAME",  (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME",  (2, 0), (2, -1), "Helvetica-Bold"),
        ("FONTSIZE",  (0, 0), (-1, -1), 8),
        ("BACKGROUND",(0, 0), (-1, -1), colors.HexColor("#F5F8FE")),
        ("GRID",      (0, 0), (-1, -1), 0.3, colors.HexColor("#CCCCCC")),
        ("TOPPADDING",(0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(meta_tbl)
    story.append(Spacer(1, 5 * mm))
    story.append(Paragraph("PART A — DETAILS OF TAX DEDUCTED AND DEPOSITED", ss["DocH2"]))

    earn_data = [
        ["Description", "Amount (₹)"],
        ["Gross Salary (Total)",               f"{gross_py:,}"],
        ["  Less: House Rent Allowance (HRA)", f"({hra_py:,})"],
        ["  Less: Standard Deduction",         f"({std_ded:,})"],
        ["Net Taxable Salary",                  f"{net_tax:,}"],
        ["Tax Deducted at Source (TDS)",        f"{tds_py:,}"],
    ]
    earn_tbl = Table(earn_data, colWidths=[12*cm, 4*cm])
    earn_tbl.setStyle(_tbl_style())
    story.append(earn_tbl)
    story.append(Spacer(1, 5 * mm))
    story.append(Paragraph(
        f"Certified that the tax mentioned above has been deducted from the salary "
        f"of {case.name} (PAN: {case.pan}) and deposited to the credit of the "
        f"Central Government as per the provisions of the Income-tax Act.",
        ss["Body"]))
    story.append(Spacer(1, 10 * mm))
    story.append(Paragraph("Authorised Signatory — Accounts & Finance", ss["Body"]))
    story.append(Paragraph(emp_name, ss["Label"]))

    doc.build(story)


def gen_utility_bill(case: LoanCase, out_path: Path) -> None:
    """Generates an electricity / utility bill PDF."""
    ss = _styles()
    doc = _base_doc(out_path)
    story = []

    provider = random.choice(UTILITY_PROVIDERS)
    consumer_no = "".join(random.choices("0123456789", k=10))
    units = random.randint(150, 800)
    amount = units * random.uniform(4.5, 7.5)
    due = date(2026, 3, 20)

    story.append(Paragraph(provider.upper(), ss["CenterBold"]))
    story.append(Paragraph("ELECTRICITY BILL / TAX INVOICE", ss["Center"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1F3864")))
    story.append(Spacer(1, 4 * mm))

    info = [
        ["Consumer Name",   case.name,           "Consumer No.",    consumer_no],
        ["Service Address", case.address,         "Bill Month",      "March 2026"],
        ["Connection Type", "Domestic (LT)",      "Due Date",
         due.strftime("%d-%b-%Y")],
    ]
    info_tbl = Table(info, colWidths=[3.8*cm, 7.2*cm, 3.0*cm, 5.5*cm])
    info_tbl.setStyle(TableStyle([
        ("FONTNAME",  (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME",  (2, 0), (2, -1), "Helvetica-Bold"),
        ("FONTSIZE",  (0, 0), (-1, -1), 8),
        ("BACKGROUND",(0, 0), (-1, -1), colors.HexColor("#F5F8FE")),
        ("GRID",      (0, 0), (-1, -1), 0.3, colors.HexColor("#CCCCCC")),
        ("TOPPADDING",(0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(info_tbl)
    story.append(Spacer(1, 5 * mm))

    bill_data = [
        ["Component",              "Units / Details",  "Amount (₹)"],
        ["Energy Charges",         f"{units} kWh",     f"{amount:.2f}"],
        ["Fixed / Demand Charges", "—",                "125.00"],
        ["Fuel Adjustment Charge", f"{units} × 0.10 ₹/u", f"{units*0.10:.2f}"],
        ["Electricity Duty (6%)",  "—",                f"{amount*0.06:.2f}"],
        ["TOTAL AMOUNT DUE",       "",                 f"₹ {(amount + 125 + units*0.10 + amount*0.06):.2f}"],
    ]
    bill_tbl = Table(bill_data, colWidths=[7*cm, 5*cm, 4*cm])
    bill_style = _tbl_style()
    bill_style.add("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold")
    bill_style.add("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#D6E4FF"))
    bill_tbl.setStyle(bill_style)
    story.append(bill_tbl)
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph(
        "Pay online at www.provider-portal.gov.in or visit your nearest collection centre.",
        ss["Small"]))

    doc.build(story)


def gen_gift_letter(case: LoanCase, out_path: Path) -> None:
    """Generates a down-payment gift letter PDF."""
    ss = _styles()
    doc = _base_doc(out_path)
    story = []

    donor = fake.name()
    relationship = random.choice(["Father", "Mother", "Spouse", "Brother", "Sister"])
    gift_amount = random.randint(200_000, 1_500_000)

    story.append(Paragraph("GIFT DECLARATION LETTER", ss["CenterBold"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1F3864")))
    story.append(Spacer(1, 5 * mm))
    story.append(Paragraph(f"Date: {date(2026, 3, 10).strftime('%d %B %Y')}", ss["Right"]))
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph("To,", ss["Body"]))
    story.append(Paragraph("The Branch Manager / Loan Processing Officer,", ss["Body"]))
    story.append(Paragraph(f"{case.bank_name}, {case.bank_branch}", ss["Body"]))
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph("Sub: Declaration that amount provided is a gift and not a loan", ss["Label"]))
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph(
        f"I, <b>{donor}</b>, {relationship} of <b>{case.name}</b>, "
        f"residing at {fake.address().replace(chr(10), ', ')}, do hereby "
        f"solemnly declare that I have gifted a sum of <b>{fmt_inr(gift_amount)}</b> "
        f"(Rupees {_amount_words(gift_amount)} only) to my "
        f"{relationship.lower()} for the purpose of contributing towards the "
        f"down payment of the home loan being availed by him/her.",
        ss["Body"]))
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph(
        "I further declare that this amount is a gift and not a loan and "
        "no repayment is expected or required from the recipient. "
        "This gift has been made out of my own legitimate funds and I "
        "undertake to provide proof of the same if required by the bank.",
        ss["Body"]))
    story.append(Spacer(1, 10 * mm))
    story.append(Paragraph(f"Donor Name: {donor}", ss["Body"]))
    story.append(Paragraph(f"Relationship: {relationship}", ss["Body"]))
    story.append(Paragraph(f"PAN: {random_pan()}", ss["Body"]))
    story.append(Spacer(1, 10 * mm))
    story.append(Paragraph("Signature: _________________________", ss["Body"]))

    doc.build(story)


def gen_property_agreement(case: LoanCase, out_path: Path) -> None:
    """Generates a property sale agreement skeleton PDF."""
    ss = _styles()
    doc = _base_doc(out_path)
    story = []

    vendor = fake.name()
    agmt_date = date(2026, 2, 20)

    story.append(Paragraph("AGREEMENT FOR SALE OF IMMOVABLE PROPERTY", ss["CenterBold"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1F3864")))
    story.append(Spacer(1, 5 * mm))
    story.append(Paragraph(
        f"This Agreement of Sale is entered into on the <b>{agmt_date.strftime('%d day of %B %Y')}</b> "
        f"between:",
        ss["Body"]))
    story.append(Spacer(1, 3 * mm))

    parties = [
        ["Party",            "Name",        "Address"],
        ["VENDOR (Seller)",  vendor,        fake.address().replace("\n", ", ")],
        ["PURCHASER (Buyer)", case.name,    case.address],
    ]
    p_tbl = Table(parties, colWidths=[4*cm, 5*cm, 9*cm])
    p_tbl.setStyle(_tbl_style())
    story.append(p_tbl)
    story.append(Spacer(1, 4 * mm))

    story.append(Paragraph("PROPERTY DETAILS", ss["DocH2"]))
    prop_info = [
        ["Property Type",        case.property_type],
        ["Property Address",     case.property_address],
        ["Sale Consideration",   fmt_inr(case.property_value)],
        ["Advance / Token Paid", fmt_inr(int(case.property_value * 0.10))],
        ["Balance Payable",      fmt_inr(int(case.property_value * 0.90))],
        ["Registration District", case.city],
    ]
    p2_tbl = Table(prop_info, colWidths=[5*cm, 13*cm])
    p2_tbl.setStyle(TableStyle([
        ("FONTNAME",  (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE",  (0, 0), (-1, -1), 8),
        ("BACKGROUND",(0, 0), (-1, -1), colors.HexColor("#F5F8FE")),
        ("GRID",      (0, 0), (-1, -1), 0.3, colors.HexColor("#CCCCCC")),
        ("TOPPADDING",(0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(p2_tbl)
    story.append(Spacer(1, 4 * mm))

    terms = [
        "The Vendor agrees to sell and the Purchaser agrees to purchase the above-described property for the agreed sale consideration.",
        "The Vendor warrants that the property is free from all encumbrances, liens, and liabilities.",
        "The Purchaser shall obtain the home loan from the bank/financial institution and submit the loan sanction letter within 45 days.",
        "Completion / Registration shall be effected within 90 days from the date hereof.",
        "This agreement is subject to the RERA registration and all applicable laws.",
    ]
    story.append(Paragraph("TERMS AND CONDITIONS", ss["DocH2"]))
    for i, t in enumerate(terms, 1):
        story.append(Paragraph(f"{i}. {t}", ss["Body"]))
    story.append(Spacer(1, 8 * mm))

    sig = [
        ["Signature of Vendor", "", "Signature of Purchaser", ""],
        [f"Name: {vendor}", "", f"Name: {case.name}", ""],
        ["Witness 1: _______________", "", "Witness 2: _______________", ""],
    ]
    s_tbl = Table(sig, colWidths=[4*cm, 4*cm, 4*cm, 6*cm])
    s_tbl.setStyle(TableStyle([("FONTSIZE", (0,0), (-1,-1), 8)]))
    story.append(s_tbl)

    doc.build(story)


# ── Utility functions ─────────────────────────────────────────────────────────

def _last_day(month: int, year: int) -> int:
    import calendar
    return calendar.monthrange(year, month)[1]


def _amount_words(amount: int) -> str:
    """Very simple Indian number-to-words (covers up to crores)."""
    ones = ["", "One", "Two", "Three", "Four", "Five", "Six", "Seven",
            "Eight", "Nine", "Ten", "Eleven", "Twelve", "Thirteen",
            "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen"]
    tens = ["", "", "Twenty", "Thirty", "Forty", "Fifty",
            "Sixty", "Seventy", "Eighty", "Ninety"]

    def _two(n):
        if n < 20:
            return ones[n]
        return tens[n // 10] + (" " + ones[n % 10] if n % 10 else "")

    def _three(n):
        if n >= 100:
            return ones[n // 100] + " Hundred" + (" " + _two(n % 100) if n % 100 else "")
        return _two(n)

    if amount == 0:
        return "Zero"
    parts = []
    cr = amount // 10_000_000
    lk = (amount % 10_000_000) // 100_000
    th = (amount % 100_000) // 1000
    hu = amount % 1000
    if cr:
        parts.append(_three(cr) + " Crore")
    if lk:
        parts.append(_three(lk) + " Lakh")
    if th:
        parts.append(_three(th) + " Thousand")
    if hu:
        parts.append(_three(hu))
    return " ".join(parts)


# ── Orchestrator ──────────────────────────────────────────────────────────────

def generate_case(case: LoanCase, out_dir: Path) -> dict:
    case_dir = out_dir / case.case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    docs = {}
    for m, name in [(1, "jan"), (2, "feb"), (3, "mar")]:
        p = case_dir / f"payslip_2026_{name}.pdf"
        gen_payslip(case, m, 2026, p)
        docs[f"payslip_2026_{name}"] = str(p.relative_to(out_dir.parent))

    p = case_dir / "bank_statement.pdf"
    gen_bank_statement(case, p)
    docs["bank_statement"] = str(p.relative_to(out_dir.parent))

    p = case_dir / "employment_letter.pdf"
    gen_employment_letter(case, p)
    docs["employment_letter"] = str(p.relative_to(out_dir.parent))

    p = case_dir / "form16.pdf"
    gen_form16(case, p)
    docs["form16"] = str(p.relative_to(out_dir.parent))

    p = case_dir / "utility_bill.pdf"
    gen_utility_bill(case, p)
    docs["utility_bill"] = str(p.relative_to(out_dir.parent))

    if case.has_gift_letter:
        p = case_dir / "gift_letter.pdf"
        gen_gift_letter(case, p)
        docs["gift_letter"] = str(p.relative_to(out_dir.parent))

    p = case_dir / "property_agreement.pdf"
    gen_property_agreement(case, p)
    docs["property_agreement"] = str(p.relative_to(out_dir.parent))

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
        "is_tampered": case.is_tampered,
        "tamper_types": case.tamper_types,
        "documents": docs,
    }


def main():
    p = argparse.ArgumentParser(description="Mortgage synthetic document generator")
    p.add_argument("--out", default="data/mortgage_docs",
                   help="Output root directory")
    p.add_argument("--n-cases", type=int, default=50,
                   help="Number of loan cases to generate")
    p.add_argument("--tamper-ratio", type=float, default=0.30,
                   help="Fraction of cases that are tampered (default 0.30)")
    args = p.parse_args()

    out = Path(args.out)
    cases_dir = out / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    metadata = []
    csv_rows = []
    n_tamper = int(args.n_cases * args.tamper_ratio)

    print(f"Generating {args.n_cases} cases ({n_tamper} tampered) → {out}")
    for i in range(1, args.n_cases + 1):
        case_id = f"case_{i:03d}"
        case = _make_case(case_id)
        if i <= n_tamper:
            case = _apply_tamper(case)
        record = generate_case(case, cases_dir)
        metadata.append(record)
        csv_rows.append({
            "case_id": case_id,
            "name": case.name,
            "city": case.city,
            "employer": case.employer_name,
            "annual_ctc": case.annual_ctc,
            "property_value": case.property_value,
            "loan_amount": case.loan_amount,
            "is_tampered": case.is_tampered,
            "tamper_types": "|".join(case.tamper_types),
        })
        status = "TAMPERED" if case.is_tampered else "clean"
        print(f"  [{i:03d}/{args.n_cases}] {case_id} – {case.name} – {status}")

    # Write metadata.json
    meta_path = out / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False)

    # Write labels.csv (handy for ML pipelines)
    csv_path = out / "labels.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\nDone. {len(metadata)} cases written.")
    print(f"  metadata.json → {meta_path}")
    print(f"  labels.csv    → {csv_path}")


if __name__ == "__main__":
    main()
