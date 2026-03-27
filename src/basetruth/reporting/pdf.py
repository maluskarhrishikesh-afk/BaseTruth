"""pdf.py — Plain-English PDF report generator for BaseTruth.

Produces a single-page (or multi-page) A4 PDF that a non-technical reviewer
(e.g. a bank loan officer) can read without needing to understand how fraud
detection works.

Design principles
-----------------
- No forensic jargon.  Every finding is described in plain English.
- Traffic-light colours: green = clear, amber = needs review, red = serious issue.
- Summary verdict at the top with a clear call to action.
- All numbers and checks are labelled so a reviewer knows exactly what was checked.

Usage
-----
    from basetruth.reporting.pdf import render_scan_report_pdf

    pdf_bytes = render_scan_report_pdf(report_dict)
    Path("output_report.pdf").write_bytes(pdf_bytes)
"""
from __future__ import annotations

import io
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from fpdf import FPDF

# ---------------------------------------------------------------------------
# Colours (RGB 0-255)
# ---------------------------------------------------------------------------
_C_DARK_BLUE    = (31,  56, 100)   # #1F3864 — brand header
_C_PASS_GREEN   = (39, 174,  96)   # #27AE60
_C_WARN_AMBER   = (243, 156,  18)  # #F39C12
_C_FAIL_RED     = (192,  57,  43)  # #C0392B
_C_LIGHT_GRAY   = (248, 249, 250)  # #F8F9FA
_C_MID_GRAY     = (206, 212, 218)  # #CED4DA
_C_TEXT_DARK    = (33,   37,  41)  # #212529
_C_TEXT_LIGHT   = (255, 255, 255)  # #FFFFFF
_C_WHITE        = (255, 255, 255)

# ---------------------------------------------------------------------------
# Risk level → human-readable verdict and colour
# ---------------------------------------------------------------------------
_VERDICTS = {
    "low": (
        "CLEAR - No significant issues found",
        "These documents appear genuine. Our automated checks did not find "
        "any significant problems. A standard manual review is still recommended.",
        _C_PASS_GREEN,
    ),
    "review": (
        "REVIEW RECOMMENDED",
        "These documents have minor points that need a closer look. "
        "They may be genuine but should be reviewed by a human before approval.",
        _C_WARN_AMBER,
    ),
    "medium": (
        "ISSUES FOUND - Action needed",
        "We found discrepancies that need your attention. Please review the "
        "issues listed below before making a decision on this application.",
        _C_WARN_AMBER,
    ),
    "high": (
        "SERIOUS ISSUES - High risk",
        "These documents contain serious inconsistencies. Further investigation "
        "is strongly recommended before approving this application.",
        _C_FAIL_RED,
    ),
    "critical": (
        "FRAUDULENT INDICATORS DETECTED",
        "These documents show clear signs of tampering or fraud. "
        "Do not approve this application without a full investigation.",
        _C_FAIL_RED,
    ),
}

# ---------------------------------------------------------------------------
# Technical rule name → plain-English description
# Each entry: (short_label, full_explanation)
# ---------------------------------------------------------------------------
_RULE_PLAIN = {
    # Employment letter checks
    "employment_backdating_signal": (
        "Employment start date",
        "The date on the employment letter and the employee's stated joining "
        "date are far apart, suggesting the joining date may have been altered "
        "to make the employment appear longer than it actually is.",
    ),
    "employer_cin_present": (
        "Company registration number",
        "The company registration number (CIN) is missing or invalid. A "
        "legitimate employer will always have a valid registered CIN.",
    ),
    "ctc_gross_match": (
        "Salary figures consistency",
        "The annual salary figure does not match the monthly salary multiplied "
        "by 12. These should always agree.",
    ),

    # Payslip checks
    "basic_gross_proportion": (
        "Basic pay proportion",
        "The basic salary shown is unusually low compared to the total monthly "
        "pay. Employers sometimes manipulate this to reduce Provident Fund "
        "contributions or inflate tax-free allowances.",
    ),
    "hra_proportion_of_gross": (
        "House rent allowance",
        "The house rent allowance (HRA) is higher than what is legally "
        "permitted. This can be used to claim more tax exemption than is "
        "actually allowed.",
    ),
    "hra_proportion": (
        "House rent allowance",
        "The house rent allowance (HRA) exceeds the legally permitted limit "
        "of 50% of basic salary for metro cities.",
    ),
    "salary_credit_vs_payslip": (
        "Salary deposited vs payslip",
        "The salary actually deposited in the bank account is significantly "
        "different from the salary shown on the payslip. This is a common "
        "sign of inflated payslips.",
    ),
    "payslip_vs_bank_credits": (
        "Salary deposited vs payslip",
        "The salary actually deposited into the bank is lower than what the "
        "payslip declares. The declared income may be inflated.",
    ),
    "professional_tax_statutory": (
        "Professional tax deduction",
        "The professional tax deducted exceeds the legal maximum of Rs 200 "
        "per month.",
    ),

    # Bank statement checks
    "circular_funds_roundtrip": (
        "Suspicious fund movement",
        "The bank statement shows large amounts being transferred in and then "
        "transferred out on the same day or within a short period. This pattern "
        "can be used to temporarily inflate the account balance.",
    ),
    "statement_date_range_validity": (
        "Bank statement coverage period",
        "The bank statement header says it covers a certain period, but the "
        "actual transactions only span a shorter time. Part of the statement "
        "history may have been removed.",
    ),
    "debit_credit_total_arithmetic": (
        "Bank statement totals check",
        "The total credits or debits shown in the bank statement do not match "
        "the sum of individual transactions. This suggests the statement may "
        "have been edited.",
    ),
    "bank_ifsc_account_consistency": (
        "Bank code and bank name match",
        "The bank's branch code (IFSC) and the bank name shown on the "
        "statement do not match each other. Every bank branch has a unique "
        "IFSC code and they must agree.",
    ),
    "balance_progression": (
        "Bank balance calculation",
        "The running balance in the bank statement does not follow correctly "
        "from the transactions (opening balance + credits - debits should equal "
        "the closing balance at each row).",
    ),

    # Form 16 / tax checks
    "form16_tds_vs_payslip": (
        "Tax deducted (payslip vs Form 16)",
        "The amount of income tax deducted shown in the Form 16 does not match "
        "what was deducted across the payslips. These should agree.",
    ),
    "form16_gross_matches_payslips": (
        "Annual income (payslip vs Form 16)",
        "The total annual income shown in the Form 16 does not match the sum "
        "of monthly payslips. One of these has likely been altered.",
    ),

    # PDF/forensic checks
    "pdf_modification_markers": (
        "Document editing signs",
        "The PDF file contains technical signs that suggest it was edited "
        "after it was originally created. Legitimate documents from a company "
        "system are generally not edited after creation.",
    ),
    "metadata_creation_modified_delta": (
        "Document creation date",
        "The document's internal creation and modification dates are suspicious. "
        "The modification date appears before the creation date, which is technically impossible.",
    ),
    "producer_creator_mismatch": (
        "PDF software details",
        "The software that created this PDF does not match the software that "
        "claims to have modified it. This can indicate the document was recreated "
        "rather than generated directly from the source system.",
    ),
}


def _safe(text: str) -> str:
    """Replace characters outside Latin-1 with safe ASCII equivalents.

    fpdf2 core fonts (Helvetica, Courier, Times) only support Latin-1.  Smart
    quotes, em/en dashes, ellipses etc. cause a FPDFUnicodeEncodingException
    unless a Unicode font is registered.  This helper lets us avoid that
    dependency while keeping output readable.
    """
    return (
        text
        .replace("\u2014", "--")    # em dash  ->  --
        .replace("\u2013", "-")     # en dash  ->  -
        .replace("\u2018", "'")     # left single quote  ->  '
        .replace("\u2019", "'")     # right single quote / apostrophe ->  '
        .replace("\u201c", '"')     # left double quote  ->  "
        .replace("\u201d", '"')     # right double quote ->  "
        .replace("\u2026", "...")   # ellipsis
        .replace("\u00a0", " ")     # non-breaking space
        .encode("latin-1", errors="replace").decode("latin-1")
    )


def _rule_label_and_desc(rule_name: str, signal: dict) -> Tuple[str, str]:
    """Return (short_label, plain_english_description) for a signal."""
    entry = _RULE_PLAIN.get(rule_name)
    if entry:
        return entry
    # Fall back: convert the rule name to something readable
    label = rule_name.replace("_", " ").replace("::", " / ").title()
    desc = str(signal.get("message") or signal.get("summary") or
               "This check found an inconsistency that needs review.")
    return label, desc


def _severity_colour(severity: str) -> tuple:
    mapping = {
        "critical": _C_FAIL_RED,
        "high":     _C_FAIL_RED,
        "medium":   _C_WARN_AMBER,
        "low":      _C_WARN_AMBER,
        "info":     _C_PASS_GREEN,
    }
    return mapping.get(str(severity).lower(), _C_WARN_AMBER)


# ---------------------------------------------------------------------------
# PDF builder
# ---------------------------------------------------------------------------

class _ReportPDF(FPDF):
    """BaseTruth report PDF with helpers for styled elements."""

    def header(self) -> None:
        # Drawn manually per page so we skip the default FPDF header
        pass

    def footer(self) -> None:
        self.set_y(-12)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(*_C_TEXT_DARK)
        self.cell(
            0, 5,
            "This is an automated fraud-screening report generated by BaseTruth.  "
            "It supports -- but does not replace -- a human review.",
            align="C",
        )
        self.ln(3)
        self.cell(
            0, 4,
            f"Page {self.page_no()}   |   Generated {datetime.now(timezone.utc).strftime('%d %b %Y %H:%M UTC')}",
            align="C",
        )

    # ---- helpers ----

    def draw_header_bar(self) -> None:
        """Full-width brand-coloured header bar at the very top."""
        self.set_fill_color(*_C_DARK_BLUE)
        self.rect(0, 0, 210, 18, "F")
        self.set_y(3)
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(*_C_TEXT_LIGHT)
        self.cell(0, 12, _safe("BaseTruth - Document Integrity Report"), align="C")
        self.ln(18)

    def section_title(self, text: str) -> None:
        self.ln(3)
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*_C_DARK_BLUE)
        self.cell(0, 7, _safe(text).upper(), ln=True)
        self.set_draw_color(*_C_DARK_BLUE)
        self.set_line_width(0.5)
        self.line(self.get_x(), self.get_y(), 195, self.get_y())
        self.ln(2)

    def info_row(self, label: str, value: str) -> None:
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*_C_TEXT_DARK)
        self.cell(45, 6, _safe(label), ln=False)
        self.set_font("Helvetica", "", 9)
        self.cell(0, 6, _safe(str(value)), ln=True)

    def verdict_box(self, title: str, body: str, colour: tuple) -> None:
        """Large coloured box showing the overall verdict."""
        self.set_fill_color(*colour)
        self.set_text_color(*_C_TEXT_LIGHT)
        # Always start from the left margin so multi_cell has room to wrap.
        box_y = self.get_y() + 3
        self.set_xy(self.l_margin, box_y)
        page_w = self.w - self.l_margin - self.r_margin  # usable width
        # Title line
        self.set_font("Helvetica", "B", 14)
        self.multi_cell(page_w, 10, _safe(title), align="C", fill=True)
        # Body line
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "", 9)
        self.multi_cell(page_w, 6, _safe(body), align="C", fill=True)
        self.set_text_color(*_C_TEXT_DARK)
        self.ln(4)

    def checks_table(self, signals: List[dict]) -> None:
        """Traffic-light table of all checks run."""
        col_w_label  = 115
        col_w_result = 25
        col_w_score  = 20
        row_h        = 6.5

        # Table header
        self.set_fill_color(*_C_DARK_BLUE)
        self.set_text_color(*_C_TEXT_LIGHT)
        self.set_font("Helvetica", "B", 8)
        self.cell(col_w_label,  row_h, "  Check", fill=True, border=0)
        self.cell(col_w_result, row_h, "Result",  fill=True, border=0, align="C")
        self.cell(col_w_score,  row_h, "Score",   fill=True, border=0, align="C")
        self.ln(row_h)

        # Alternating rows
        for i, sig in enumerate(signals):
            passed = sig.get("passed", True)
            rule   = str(sig.get("rule") or sig.get("name", ""))
            label, _ = _rule_label_and_desc(rule, sig)

            bg = _C_LIGHT_GRAY if i % 2 == 0 else _C_WHITE
            self.set_fill_color(*bg)
            self.set_text_color(*_C_TEXT_DARK)

            # Row background
            row_y = self.get_y()
            self.rect(self.get_x(), row_y, col_w_label + col_w_result + col_w_score, row_h, "F")

            self.set_font("Helvetica", "", 8)
            self.cell(col_w_label, row_h, _safe(f"  {label}"), fill=False, border=0)

            # Result badge
            if passed:
                self.set_fill_color(*_C_PASS_GREEN)
                self.set_text_color(*_C_TEXT_LIGHT)
                result_text = "  PASS  "
            else:
                sev = str(sig.get("severity", "medium")).lower()
                self.set_fill_color(*_severity_colour(sev))
                self.set_text_color(*_C_TEXT_LIGHT)
                result_text = "  ISSUE  "

            self.set_font("Helvetica", "B", 7)
            self.cell(col_w_result, row_h, result_text, fill=True, border=0, align="C")

            # Score
            score = sig.get("score", 0)
            self.set_fill_color(*bg)
            self.set_text_color(*_C_TEXT_DARK)
            self.set_font("Helvetica", "", 8)
            self.cell(col_w_score, row_h, str(score) if score else "-", fill=False, border=0, align="C")
            self.ln(row_h)

        # Bottom border
        self.set_draw_color(*_C_MID_GRAY)
        self.set_line_width(0.3)
        self.line(self.get_x(), self.get_y(), 195, self.get_y())
        self.ln(2)

    def issue_card(self, index: int, label: str, description: str, severity: str) -> None:
        """A card describing one specific failed check in plain English."""
        colour = _severity_colour(severity)

        # Left accent bar
        card_x = self.get_x()
        card_y = self.get_y()
        self.set_fill_color(*colour)
        self.rect(card_x, card_y, 3, 18, "F")  # 3mm wide colour bar

        # Card background
        self.set_fill_color(*_C_LIGHT_GRAY)
        self.rect(card_x + 3, card_y, 177, 18, "F")

        # Number bubble
        self.set_xy(card_x + 5, card_y + 1)
        self.set_fill_color(*colour)
        self.set_text_color(*_C_TEXT_LIGHT)
        self.set_font("Helvetica", "B", 9)
        self.cell(8, 8, str(index), fill=True, align="C")

        # Label (bold)
        self.set_xy(card_x + 15, card_y + 1)
        self.set_text_color(*_C_TEXT_DARK)
        self.set_font("Helvetica", "B", 9)
        self.cell(0, 5, _safe(label), ln=True)

        # Description (wrapped)
        self.set_xy(card_x + 15, card_y + 7)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(80, 80, 80)
        self.multi_cell(170, 5, _safe(description))

        self.set_y(card_y + 20)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render_scan_report_pdf(report: Dict[str, Any]) -> bytes:
    """Generate a plain-English A4 PDF report and return the raw bytes.

    Parameters
    ----------
    report:
        The dict returned by BaseTruthService.scan_document() (or loaded from
        the ``_verification.json`` artifact file).

    Returns
    -------
    bytes
        Raw PDF content ready to be written to a ``.pdf`` file or sent as an
        HTTP response (Content-Type: application/pdf).
    """
    source      = report.get("source", {})
    summary     = report.get("structured_summary", {})
    tamper      = report.get("tamper_assessment", {})
    key_fields  = summary.get("key_fields", {})
    doc_info    = summary.get("document", {})
    signals     = tamper.get("signals", [])

    risk_level  = str(tamper.get("risk_level", "low")).lower()
    truth_score = tamper.get("truth_score", 100)
    verdict_title, verdict_body, verdict_colour = _VERDICTS.get(
        risk_level, _VERDICTS["low"]
    )

    # Split signals into passed / failed
    failed_signals = [s for s in signals if not s.get("passed", True)]
    passed_signals = [s for s in signals if s.get("passed", True)]
    all_signals    = failed_signals + passed_signals   # show failures first

    # ------------------------------------------------------------------ build
    pdf = _ReportPDF(orientation="P", unit="mm", format="A4")
    pdf.set_margins(left=10, top=5, right=10)
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()
    pdf.set_text_color(*_C_TEXT_DARK)

    # ── Header bar ──────────────────────────────────────────────────────────
    pdf.draw_header_bar()

    # ── Document info ────────────────────────────────────────────────────────
    pdf.section_title("Document information")
    file_name   = source.get("name", "Unknown")
    doc_type    = doc_info.get("type", "unknown").replace("_", " ").title()
    scanned_at  = report.get("generated_at", "")
    if scanned_at:
        try:
            scanned_at = datetime.fromisoformat(scanned_at).strftime("%d %b %Y %H:%M UTC")
        except ValueError:
            pass

    pdf.info_row("File name:",        _safe(file_name))
    pdf.info_row("Document type:",    _safe(doc_type))
    pdf.info_row("Scanned at:",       _safe(scanned_at))
    pdf.info_row("Confidence score:", f"{truth_score} / 100")

    # Key fields that are useful to show
    _SHOW_FIELDS = [
        ("employee_name",      "Employee name"),
        ("employer_name",      "Employer / company"),
        ("date_of_joining",    "Date of joining"),
        ("letter_issue_date",  "Letter issue date"),
        ("annual_ctc",         "Annual CTC"),
        ("gross_monthly_salary", "Gross monthly salary"),
        ("cin",                "Company CIN"),
        ("account_holder",     "Account holder"),
        ("bank_name",          "Bank name"),
        ("ifsc_code",          "IFSC code"),
        ("statement_period_start", "Statement from"),
        ("statement_period_end",   "Statement to"),
    ]
    for field_key, field_label in _SHOW_FIELDS:
        val = key_fields.get(field_key)
        if val is not None and str(val).strip():
            pdf.info_row(f"{field_label}:", _safe(str(val)))

    # ── Verdict box ──────────────────────────────────────────────────────────
    pdf.ln(3)
    pdf.verdict_box(verdict_title, verdict_body, verdict_colour)

    # ── Summary numbers ──────────────────────────────────────────────────────
    total   = len(signals)
    n_fail  = len(failed_signals)
    n_pass  = len(passed_signals)

    pdf.section_title("Summary")
    pdf.set_font("Helvetica", "", 9)

    # Three side-by-side stat boxes
    box_w = 55
    box_h = 14
    stats = [
        ("Checks run", str(total),  _C_DARK_BLUE),
        ("Passed",     str(n_pass), _C_PASS_GREEN),
        ("Issues",     str(n_fail), _C_FAIL_RED if n_fail else _C_PASS_GREEN),
    ]
    for label, value, colour in stats:
        bx = pdf.get_x()
        by = pdf.get_y()
        pdf.set_fill_color(*colour)
        pdf.rect(bx, by, box_w, box_h, "F")
        pdf.set_xy(bx, by + 1)
        pdf.set_text_color(*_C_TEXT_LIGHT)
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(box_w, 7, value, align="C")
        pdf.set_xy(bx, by + 8)
        pdf.set_font("Helvetica", "", 8)
        pdf.cell(box_w, 5, label, align="C")
        pdf.set_xy(bx + box_w + 5, by)

    pdf.set_text_color(*_C_TEXT_DARK)
    pdf.ln(box_h + 5)

    # ── All checks table ─────────────────────────────────────────────────────
    if all_signals:
        pdf.section_title("All checks")
        pdf.checks_table(all_signals)

    # ── Issue detail cards ───────────────────────────────────────────────────
    if failed_signals:
        pdf.section_title(f"Issues that need your attention  ( {n_fail} )")
        for idx, sig in enumerate(failed_signals, start=1):
            rule      = str(sig.get("rule") or sig.get("name", ""))
            severity  = str(sig.get("severity", "medium")).lower()
            label, desc = _rule_label_and_desc(rule, sig)

            # Add extra context from the signal's own details dict if available
            details = sig.get("details") or {}
            if isinstance(details, dict) and details:
                extras = []
                for k, v in details.items():
                    extras.append(f"{k.replace('_', ' ').title()}: {v}")
                if extras:
                    desc = desc + "  |  " + "  |  ".join(extras)

            pdf.issue_card(idx, label, desc, severity)
    else:
        pdf.section_title("Issues")
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(*_C_PASS_GREEN)
        pdf.cell(0, 8, _safe("No issues found. All checks passed."), ln=True)
        pdf.set_text_color(*_C_TEXT_DARK)

    # ── Limitations / caveats ────────────────────────────────────────────────
    limitations = tamper.get("limitations", [])
    if limitations:
        pdf.section_title("Notes for the reviewer")
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(80, 80, 80)
        for lim in limitations:
            pdf.cell(5, 5, "-")
            pdf.multi_cell(0, 5, _safe(str(lim)))
        pdf.set_text_color(*_C_TEXT_DARK)

    # ── Output ───────────────────────────────────────────────────────────────
    buf = io.BytesIO()
    pdf.output(buf)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Plain-English translation for cross-document reconciliation anomaly types
# ---------------------------------------------------------------------------
_ANOMALY_PLAIN: Dict[str, Tuple[str, str]] = {
    "payslip_vs_form16_salary_mismatch": (
        "Salary mismatch: Payslip vs Form 16",
        "The monthly salary shown on the payslips does not match the annual income "
        "recorded on the Form 16 tax document. Income figures should agree across all documents.",
    ),
    "payslip_vs_offer_letter_salary_mismatch": (
        "Salary mismatch: Payslip vs Offer Letter",
        "The salary shown on the payslips is significantly different from the figure "
        "stated in the employment or offer letter. One of these may have been altered.",
    ),
    "offer_letter_vs_form16_salary_mismatch": (
        "Salary mismatch: Offer Letter vs Form 16",
        "The annual salary in the offer letter does not agree with the gross income shown "
        "on the Form 16. These two documents describe the same employer-employee "
        "relationship and should agree.",
    ),
    "payslip_net_vs_bank_salary_credit_mismatch": (
        "Salary deposited vs Payslip net pay",
        "The take-home salary credited to the bank account is different from the net pay "
        "shown on the payslips. Genuine payslips will always match what is actually "
        "deposited in the bank.",
    ),
    "pan_mismatch_payslip_vs_form16": (
        "PAN number mismatch across documents",
        "The PAN (tax identification number) on the payslip does not match the PAN on the "
        "Form 16.  This means the documents may belong to two different people — a serious "
        "fraud indicator.",
    ),
    "bank_name_payslip_vs_statement_mismatch": (
        "Wrong bank statement submitted",
        "The bank name on the payslip (where salary is credited) does not match the bank "
        "shown on the submitted bank statement. The applicant appears to have submitted a "
        "statement from a different bank account.",
    ),
    "gift_amount_matches_bank_credit": (
        "Suspicious gift money in bank account",
        "The exact amount declared in the gift letter also appears as an incoming transfer "
        "in the bank statement. The 'gift' may actually be a loan that has been disguised "
        "to artificially increase the apparent down-payment savings.",
    ),
}


def _overall_risk_from_reports(
    reports: List[Dict[str, Any]],
    reconciliation: Optional[Dict[str, Any]],
) -> str:
    """Return the worst risk level across all documents and reconciliation anomalies."""
    _order = {"critical": 4, "high": 3, "medium": 2, "review": 1, "low": 0}
    worst = "low"
    for r in reports:
        rl = str(r.get("tamper_assessment", {}).get("risk_level", "low")).lower()
        if _order.get(rl, 0) > _order.get(worst, 0):
            worst = rl
    if reconciliation:
        for a in reconciliation.get("anomalies", []):
            sev = str(a.get("severity", "low")).lower()
            mapped = {"high": "high", "medium": "medium", "low": "low"}.get(sev, "low")
            if _order.get(mapped, 0) > _order.get(worst, 0):
                worst = mapped
    return worst


def render_case_bundle_pdf(
    reports: List[Dict[str, Any]],
    reconciliation: Optional[Dict[str, Any]] = None,
    case_title: str = "Mortgage Application Review",
) -> bytes:
    """Generate a plain-English case-bundle PDF and return the raw bytes.

    This report is intended for a loan officer reviewing a complete mortgage
    application.  It summarises all submitted documents and cross-document
    income consistency on a single multi-page PDF without technical jargon.

    Parameters
    ----------
    reports:
        List of dicts returned by BaseTruthService.scan_document().
    reconciliation:
        Optional dict returned by BaseTruthService.reconcile_income_documents().
    case_title:
        Optional case-level heading (e.g. applicant name or loan reference).

    Returns
    -------
    bytes
        Raw PDF content.
    """
    overall_risk = _overall_risk_from_reports(reports, reconciliation)
    verdict_title, verdict_body, verdict_colour = _VERDICTS.get(overall_risk, _VERDICTS["low"])

    # Extract applicant-level information from payslip / bank statement
    applicant_name: str = ""
    employer_name: str = ""
    monthly_salary: str = ""
    bank_name: str = ""
    for r in reports:
        ss = r.get("structured_summary", {})
        doc_type = str(ss.get("document", {}).get("type", "")).lower()
        kf = ss.get("key_fields", {})
        if doc_type == "payslip":
            if not applicant_name:
                applicant_name = str(kf.get("employee_name") or "").strip()
            if not employer_name:
                employer_name = str(kf.get("employer_name") or kf.get("company_name") or "").strip()
            if not monthly_salary:
                g = kf.get("gross_earnings") or kf.get("gross_salary")
                if g:
                    try:
                        monthly_salary = f"Rs {int(g):,} / month"
                    except (ValueError, TypeError):
                        monthly_salary = str(g)
        if doc_type == "bank_statement" and not bank_name:
            bank_name = str(kf.get("bank_name") or "").strip().title()

    # Aggregate stats
    n_docs = len(reports)
    n_issues = sum(
        1 for r in reports
        if str(r.get("tamper_assessment", {}).get("risk_level", "low")).lower()
        in {"high", "critical", "medium"}
    )
    reconciliation_anomalies: List[Dict[str, Any]] = (reconciliation or {}).get("anomalies", [])
    reconciliation_evidence: Dict[str, Any] = (reconciliation or {}).get("evidence", {})

    # ------------------------------------------------------------------ build
    pdf = _ReportPDF(orientation="P", unit="mm", format="A4")
    pdf.set_margins(left=10, top=5, right=10)
    pdf.set_auto_page_break(auto=True, margin=18)

    # ══════════════════════════════════════════════════════════════════
    # PAGE 1 — Cover / Overview
    # ══════════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.set_text_color(*_C_TEXT_DARK)
    pdf.draw_header_bar()

    # Title
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(*_C_DARK_BLUE)
    pdf.ln(2)
    pdf.cell(0, 10, _safe(case_title.upper()), align="C", ln=True)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(80, 80, 80)
    generated_str = datetime.now(timezone.utc).strftime("%d %b %Y  %H:%M UTC")
    pdf.cell(0, 6, _safe(f"Automated fraud-screening report  -  Generated {generated_str}"), align="C", ln=True)
    pdf.ln(4)

    # Applicant info box
    pdf.section_title("Applicant Information")
    if applicant_name:
        pdf.info_row("Applicant name:", applicant_name)
    if employer_name:
        pdf.info_row("Employer:", employer_name)
    if monthly_salary:
        pdf.info_row("Declared monthly salary:", monthly_salary)
    if bank_name:
        pdf.info_row("Bank:", bank_name)
    pdf.info_row("Documents submitted:", str(n_docs))

    # Overall verdict
    pdf.ln(2)
    pdf.verdict_box(verdict_title, verdict_body, verdict_colour)

    # Bundle stats
    pdf.ln(2)
    stats = [
        ("Documents reviewed", str(n_docs), _C_DARK_BLUE),
        ("Documents with issues", str(n_issues), _C_FAIL_RED if n_issues else _C_PASS_GREEN),
        ("Income checks failed", str(len(reconciliation_anomalies)), _C_FAIL_RED if reconciliation_anomalies else _C_PASS_GREEN),
    ]
    box_w = 55
    box_h = 14
    for label, value, colour in stats:
        bx, by = pdf.get_x(), pdf.get_y()
        pdf.set_fill_color(*colour)
        pdf.rect(bx, by, box_w, box_h, "F")
        pdf.set_xy(bx, by + 1)
        pdf.set_text_color(*_C_TEXT_LIGHT)
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(box_w, 7, value, align="C")
        pdf.set_xy(bx, by + 8)
        pdf.set_font("Helvetica", "", 8)
        pdf.cell(box_w, 5, label, align="C")
        pdf.set_xy(bx + box_w + 5, by)
    pdf.set_text_color(*_C_TEXT_DARK)
    pdf.ln(box_h + 6)

    # ── Document overview table ──────────────────────────────────────
    pdf.section_title("Document Overview")
    _col_file   = 70
    _col_type   = 40
    _col_score  = 22
    _col_risk   = 28
    _col_issues = 25
    _row_h      = 7

    # Table header
    pdf.set_fill_color(*_C_DARK_BLUE)
    pdf.set_text_color(*_C_TEXT_LIGHT)
    pdf.set_font("Helvetica", "B", 8)
    for (label, width) in [
        ("  Document file", _col_file),
        ("Type", _col_type),
        ("Score", _col_score),
        ("Risk level", _col_risk),
        ("Issues", _col_issues),
    ]:
        pdf.cell(width, _row_h, label, fill=True, border=0)
    pdf.ln(_row_h)

    for i, r in enumerate(reports):
        ss = r.get("structured_summary", {})
        doc_type_str = str(ss.get("document", {}).get("type", "")).replace("_", " ").title()
        fname = r.get("source", {}).get("name", "")
        score = r.get("tamper_assessment", {}).get("truth_score", 100)
        rl = str(r.get("tamper_assessment", {}).get("risk_level", "low")).lower()
        failed = [s for s in r.get("tamper_assessment", {}).get("signals", []) if not s.get("passed", True)]
        n_doc_issues = len(failed)

        bg = _C_LIGHT_GRAY if i % 2 == 0 else _C_WHITE
        pdf.set_fill_color(*bg)
        row_y = pdf.get_y()
        total_w = _col_file + _col_type + _col_score + _col_risk + _col_issues
        pdf.rect(pdf.get_x(), row_y, total_w, _row_h, "F")

        pdf.set_text_color(*_C_TEXT_DARK)
        pdf.set_font("Helvetica", "", 7)
        pdf.cell(_col_file,  _row_h, _safe("  " + fname[:38]), fill=False, border=0)
        pdf.cell(_col_type,  _row_h, _safe(doc_type_str[:22]), fill=False, border=0)
        pdf.cell(_col_score, _row_h, f"{score}/100", fill=False, border=0, align="C")

        # Risk badge
        risk_colour = {
            "low": _C_PASS_GREEN,
            "medium": _C_WARN_AMBER,
            "review": _C_WARN_AMBER,
            "high": _C_FAIL_RED,
            "critical": _C_FAIL_RED,
        }.get(rl, _C_PASS_GREEN)
        pdf.set_fill_color(*risk_colour)
        pdf.set_text_color(*_C_TEXT_LIGHT)
        pdf.set_font("Helvetica", "B", 7)
        pdf.cell(_col_risk, _row_h, rl.upper(), fill=True, border=0, align="C")

        pdf.set_fill_color(*bg)
        pdf.set_text_color(*_C_TEXT_DARK)
        pdf.set_font("Helvetica", "", 7)
        issues_str = str(n_doc_issues) if n_doc_issues else "-"
        pdf.cell(_col_issues, _row_h, issues_str, fill=False, border=0, align="C")
        pdf.ln(_row_h)

    pdf.set_draw_color(*_C_MID_GRAY)
    pdf.set_line_width(0.3)
    pdf.line(pdf.get_x(), pdf.get_y(), 195, pdf.get_y())
    pdf.ln(3)

    # ══════════════════════════════════════════════════════════════════
    # SECTION — Income Reconciliation (cross-document consistency)
    # ══════════════════════════════════════════════════════════════════
    if reconciliation_evidence or reconciliation_anomalies:
        pdf.add_page()
        pdf.draw_header_bar()
        pdf.section_title("Cross-Document Income Review")
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(60, 60, 60)
        pdf.multi_cell(
            0, 6,
            _safe(
                "We compared the salary figures across all submitted documents "
                "(payslips, Form 16, offer letter and bank statement). "
                "The table below shows what each document claims and whether the figures agree."
            ),
        )
        pdf.ln(3)

        # Evidence table
        if reconciliation_evidence:
            ev_col_src   = 90
            ev_col_month = 45
            ev_col_ann   = 45
            pdf.set_fill_color(*_C_DARK_BLUE)
            pdf.set_text_color(*_C_TEXT_LIGHT)
            pdf.set_font("Helvetica", "B", 8)
            pdf.cell(ev_col_src,   7, "  Source document", fill=True)
            pdf.cell(ev_col_month, 7, "Monthly figure", fill=True, align="C")
            pdf.cell(ev_col_ann,   7, "Annual figure", fill=True, align="C")
            pdf.ln(7)

            ev_rows = []
            if reconciliation_evidence.get("payslip_avg_monthly_gross"):
                ev_rows.append({
                    "src": f"Payslips ({reconciliation_evidence.get('payslip_count', 0)} docs)",
                    "month": f"Rs {reconciliation_evidence['payslip_avg_monthly_gross']:,}",
                    "ann":   f"Rs {reconciliation_evidence.get('payslip_annualised_gross', 0):,}",
                })
            if reconciliation_evidence.get("letter_annual_ctc"):
                m = reconciliation_evidence.get("letter_gross_monthly")
                ev_rows.append({
                    "src": reconciliation_evidence.get("letter_source") or "Offer letter",
                    "month": f"Rs {m:,}" if m else "--",
                    "ann":   f"Rs {reconciliation_evidence['letter_annual_ctc']:,}",
                })
            if reconciliation_evidence.get("form16_annual_gross"):
                ev_rows.append({
                    "src": reconciliation_evidence.get("form16_source") or "Form 16",
                    "month": "--",
                    "ann":   f"Rs {reconciliation_evidence['form16_annual_gross']:,}",
                })
            if reconciliation_evidence.get("bank_avg_salary_credit"):
                ev_rows.append({
                    "src": f"Bank statement ({reconciliation_evidence.get('bank_salary_credit_count', 0)} salary credits)",
                    "month": f"Rs {reconciliation_evidence['bank_avg_salary_credit']:,} (net)",
                    "ann":   "--",
                })

            for idx_ev, row_ev in enumerate(ev_rows):
                bg = _C_LIGHT_GRAY if idx_ev % 2 == 0 else _C_WHITE
                pdf.set_fill_color(*bg)
                pdf.set_text_color(*_C_TEXT_DARK)
                pdf.set_font("Helvetica", "", 8)
                pdf.cell(ev_col_src,   7, _safe("  " + row_ev["src"]), fill=True, border=0)
                pdf.cell(ev_col_month, 7, _safe(row_ev["month"]), fill=True, border=0, align="C")
                pdf.cell(ev_col_ann,   7, _safe(row_ev["ann"]),   fill=True, border=0, align="C")
                pdf.ln(7)
            pdf.ln(3)

        # Reconciliation anomalies
        if reconciliation_anomalies:
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_text_color(*_C_FAIL_RED)
            pdf.cell(0, 7, _safe(f"Income inconsistencies found: {len(reconciliation_anomalies)}"), ln=True)
            pdf.set_text_color(*_C_TEXT_DARK)
            pdf.ln(1)
            for idx_a, anomaly in enumerate(reconciliation_anomalies, start=1):
                atype = str(anomaly.get("type", ""))
                sev   = str(anomaly.get("severity", "medium")).lower()
                label, desc = _ANOMALY_PLAIN.get(atype, (
                    atype.replace("_", " ").title(),
                    str(anomaly.get("details", {}).get("explanation", "Income figures do not agree across documents.")),
                ))
                # Override desc with the signal's own explanation if available
                explanation = anomaly.get("details", {}).get("explanation", "")
                if explanation:
                    desc = explanation
                pdf.issue_card(idx_a, label, desc, sev)
        else:
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(*_C_PASS_GREEN)
            pdf.cell(0, 8, _safe("Income figures are consistent across all submitted documents."), ln=True)
            pdf.set_text_color(*_C_TEXT_DARK)

    # ══════════════════════════════════════════════════════════════════
    # SECTION — Per-document issues (only docs that have problems)
    # ══════════════════════════════════════════════════════════════════
    docs_with_issues = [
        r for r in reports
        if any(not s.get("passed", True) for s in r.get("tamper_assessment", {}).get("signals", []))
    ]

    if docs_with_issues:
        pdf.add_page()
        pdf.draw_header_bar()
        pdf.section_title("Document-Level Issues")
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(60, 60, 60)
        pdf.multi_cell(
            0, 6,
            _safe(
                "The following documents have one or more issues that require your attention. "
                "Each issue is described in plain language below."
            ),
        )
        pdf.ln(3)

        for r in docs_with_issues:
            fname    = r.get("source", {}).get("name", "Unknown document")
            score    = r.get("tamper_assessment", {}).get("truth_score", 100)
            rl       = str(r.get("tamper_assessment", {}).get("risk_level", "low")).lower()
            signals  = r.get("tamper_assessment", {}).get("signals", [])
            failed   = [s for s in signals if not s.get("passed", True)]

            # Document sub-heading
            risk_colour = {"low": _C_PASS_GREEN, "medium": _C_WARN_AMBER, "review": _C_WARN_AMBER,
                           "high": _C_FAIL_RED, "critical": _C_FAIL_RED}.get(rl, _C_PASS_GREEN)
            pdf.set_fill_color(*risk_colour)
            pdf.set_text_color(*_C_TEXT_LIGHT)
            pdf.set_font("Helvetica", "B", 9)
            pdf.cell(0, 8, _safe(f"  {fname}  --  Score: {score}/100  |  Risk: {rl.upper()}"),
                     fill=True, ln=True)
            pdf.set_text_color(*_C_TEXT_DARK)
            pdf.ln(1)

            for idx_s, sig in enumerate(failed, start=1):
                rule     = str(sig.get("rule") or sig.get("name", ""))
                sev      = str(sig.get("severity", "medium")).lower()
                label, desc = _rule_label_and_desc(rule, sig)
                explanation = str(sig.get("message") or "").strip()
                if explanation:
                    desc = explanation
                details = sig.get("details") or {}
                if isinstance(details, dict) and details:
                    extras = [f"{k.replace('_', ' ').title()}: {v}" for k, v in list(details.items())[:4]]
                    if extras:
                        desc = desc + "  |  " + "  |  ".join(extras)
                pdf.issue_card(idx_s, label, desc, sev)

            pdf.ln(3)

    # ══════════════════════════════════════════════════════════════════
    # FINAL PAGE — Key Extracted Fields & Overall Recommendation
    # ══════════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.draw_header_bar()
    pdf.section_title("Extracted Information Summary")
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(60, 60, 60)
    pdf.multi_cell(
        0, 5,
        _safe("Key details automatically extracted from each submitted document:"),
    )
    pdf.ln(2)

    _SHOW = [
        ("employee_name",          "Employee name"),
        ("employer_name",          "Employer"),
        ("company_name",           "Company"),
        ("date_of_joining",        "Date of joining"),
        ("gross_earnings",         "Gross earnings (monthly)"),
        ("net_pay",                "Net pay (monthly)"),
        ("annual_ctc",             "Annual CTC"),
        ("gross_monthly_salary",   "Gross monthly salary"),
        ("account_holder",         "Account holder"),
        ("bank_name",              "Bank name"),
        ("ifsc",                   "IFSC code"),
        ("period_from",            "Statement from"),
        ("period_to",              "Statement to"),
    ]
    for r in reports:
        ss = r.get("structured_summary", {})
        kf = ss.get("key_fields", {})
        doc_type_str = str(ss.get("document", {}).get("type", "")).replace("_", " ").title()
        fname = r.get("source", {}).get("name", "")
        score = r.get("tamper_assessment", {}).get("truth_score", 100)

        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(*_C_DARK_BLUE)
        pdf.cell(0, 6, _safe(f"{fname}  ({doc_type_str})  --  Score: {score}/100"), ln=True)

        had_field = False
        for fk, fl in _SHOW:
            val = kf.get(fk)
            if val is not None and str(val).strip():
                pdf.info_row(f"  {fl}:", str(val))
                had_field = True
        if not had_field:
            pdf.set_font("Helvetica", "I", 8)
            pdf.set_text_color(120, 120, 120)
            pdf.cell(0, 5, "  No key fields extracted.", ln=True)
        pdf.set_text_color(*_C_TEXT_DARK)
        pdf.ln(2)

    # Final verdict
    pdf.ln(4)
    pdf.section_title("Overall Recommendation")
    pdf.verdict_box(verdict_title, verdict_body, verdict_colour)

    if overall_risk in ("low",):
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(60, 60, 60)
        pdf.ln(2)
        pdf.multi_cell(
            0, 6,
            _safe(
                "This automated report is a screening tool to assist, not replace, "
                "human judgment.  Always perform standard due diligence checks "
                "as required by your institution's policies."
            ),
        )

    # ── Output ───────────────────────────────────────────────────────────────
    buf = io.BytesIO()
    pdf.output(buf)
    return buf.getvalue()
