"""final_report_builder.py — Builds the complete evidence bundle for entity final reports.

This module is called when the analyst clicks "Generate Final Report" on the
Document Intelligence screen.  It orchestrates the full report-building pipeline:

1.  Extract identity summary (name, DOB, address, PAN, Aadhaar) from the Aadhaar
    and PAN rows in document_extractions.
2.  Find the candidate's photo in MinIO: prefer a selfie from identity_checks, then
    fall back to any non-signature image uploaded for the entity.
3.  Build a comprehensive evidence markdown document containing every raw
    extracted_data JSON blob and every raw layered_analysis_json blob.  This is
    the "source of truth" that Gemma4 reads to write the final narrative.
4.  Send the evidence markdown to Gemma4 via the existing provider routing
    (_route_vlm_chat) and ask it to write a professional, systematic verification
    report.  Fallback to a deterministic placeholder if Gemma4/Ollama is offline.
5.  Return a complete report_json dict suitable for storage in
    entity_reports.report_json and rendering by render_entity_report_pdf().

The module has NO import dependency on document_intelligence.py, so there is no
circular import risk.  The consistency-check logic is duplicated here intentionally
to keep this module self-contained.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from basetruth.logger import get_logger

log = get_logger(__name__)


# ── Field-name aliases (mirrors document_intelligence.py) ─────────────────
# We duplicate these here so this module does not import from the UI layer.
_NAME_FIELDS    = ("candidate_name", "name", "employee_name", "account_holder_name",
                   "applicant_name", "holder_name")
_DOB_FIELDS     = ("date_of_birth", "dob", "birth_date", "year_of_birth")
_ADDRESS_FIELDS = ("address", "permanent_address", "residential_address", "current_address")
_PAN_FIELDS     = ("pan_number", "pan")
_AADHAR_FIELDS  = ("aadhaar_number", "aadhar_number", "uid", "aadhaar")
_SALARY_FIELDS_PAYSLIP = ("net_salary", "gross_salary", "net_pay", "ctc",
                           "total_compensation", "net_amount")
_SALARY_FIELDS_OFFER   = ("offered_salary", "ctc", "annual_ctc", "gross_salary",
                           "salary", "annual_salary", "package")


# ── Small field-extraction helpers ─────────────────────────────────────────

def _first(d: dict, keys: tuple) -> Optional[str]:
    """Return the first non-empty string value found in d for any of the given keys."""
    for k in keys:
        v = d.get(k)
        if v and str(v).strip():
            return str(v).strip()
    return None


def _normalise_name(name: Optional[str]) -> str:
    """Lower-case and collapse whitespace for fuzzy name comparison."""
    if not name:
        return ""
    return " ".join(name.lower().split())


def _normalise_number(val: Optional[str]) -> str:
    """Remove spaces, dashes, upper-case an ID number for comparison."""
    if not val:
        return ""
    return val.replace(" ", "").replace("-", "").upper()


def _to_float(val: Optional[str]) -> Optional[float]:
    """Convert a salary string like '₹ 1,23,456' to a plain float, or None on failure."""
    if not val:
        return None
    digits = re.sub(r"[^\d.]", "", str(val))
    try:
        return float(digits)
    except ValueError:
        return None


# ── Identity summary extraction ─────────────────────────────────────────────

def _extract_entity_summary(extractions: List[Dict[str, Any]]) -> Dict[str, str]:
    """Extract canonical identity fields from Aadhaar and PAN document rows.

    We walk all extraction rows in order.  Aadhaar rows are preferred for DOB,
    address, and Aadhaar number.  PAN rows are preferred for PAN number.
    The candidate's full name is taken from whichever identity document provides
    it first.

    Returns a dict with: full_name, dob, address, pan_number, aadhaar_number.
    Any field not found is left as an empty string.
    """
    summary: Dict[str, str] = {
        "full_name":     "",
        "dob":           "",
        "address":       "",
        "pan_number":    "",
        "aadhaar_number": "",
    }

    for ext in extractions:
        doc_type   = (ext.get("document_type") or "").lower()
        data       = ext.get("extracted_data") or {}
        # extracted_data is sometimes wrapped in an outer dict — unwrap if needed.
        fields: Dict[str, Any] = data if isinstance(data, dict) else {}

        is_aadhaar = "aadhaar" in doc_type or "aadhar" in doc_type
        is_pan     = "pan" in doc_type

        # Name: take from the first document that provides it.
        if not summary["full_name"]:
            summary["full_name"] = _first(fields, _NAME_FIELDS) or ""

        # DOB: prefer Aadhaar or PAN cards which carry the authoritative DOB.
        if not summary["dob"] and (is_aadhaar or is_pan):
            summary["dob"] = _first(fields, _DOB_FIELDS) or ""

        # Address: Aadhaar is the only document that carries the official address.
        if not summary["address"] and is_aadhaar:
            summary["address"] = _first(fields, _ADDRESS_FIELDS) or ""

        # PAN number: only from PAN documents.
        if not summary["pan_number"] and is_pan:
            summary["pan_number"] = _first(fields, _PAN_FIELDS) or ""

        # Aadhaar number: only from Aadhaar documents.
        if not summary["aadhaar_number"] and is_aadhaar:
            summary["aadhaar_number"] = _first(fields, _AADHAR_FIELDS) or ""

    return summary


# ── Candidate photo discovery ───────────────────────────────────────────────

def _find_candidate_photo_key(entity_ref: str) -> Optional[str]:
    """Find the MinIO object key for the candidate's portrait photo.

    Priority order:
    1. Selfie uploaded during a face-match or Video KYC identity check — this
       is the most reliable portrait image we have.
    2. Any image-format file stored under the entity's MinIO prefix, excluding
       known non-face images (PAN signature, stamps, thumbprints).
    3. None — no usable photo found; the report will omit the photo section.

    We deliberately avoid using PAN card images as the "candidate photo" because
    they show a thumb impression or signature, not the person's face.
    """
    try:
        # Lazy imports so this module can be tested without a live DB.
        from basetruth.store import get_entity_identity_checks, minio_list_entity_objects
    except Exception as imp_err:
        log.warning("_find_candidate_photo_key: import failed — %s", imp_err)
        return None

    # Option 1: selfie recorded during a face-match or Video KYC session.
    checks = get_entity_identity_checks(entity_ref)
    for chk in checks:
        fname = chk.get("selfie_filename") or ""
        if fname:
            key = f"{entity_ref}/{fname}"
            log.debug("_find_candidate_photo_key: using selfie from identity check", extra={"key": key})
            return key

    # Option 2: fall back to any image stored in the entity's MinIO folder.
    # Skip filenames that look like document scans, signatures, or stamps — we only
    # want a portrait-style image of the actual person.
    _IMAGE_EXTS    = {".jpg", ".jpeg", ".png", ".webp"}
    _SKIP_KEYWORDS = {"signature", "pan_signature", "thumb", "stamp", "scan"}
    objects = minio_list_entity_objects(entity_ref)
    for obj in objects:
        fn = obj.get("filename", "").lower()
        ext = "." + fn.rsplit(".", 1)[-1] if "." in fn else ""
        if ext in _IMAGE_EXTS and not any(kw in fn for kw in _SKIP_KEYWORDS):
            log.debug(
                "_find_candidate_photo_key: using entity image as fallback",
                extra={"key": obj["key"]},
            )
            return obj["key"]

    return None


# ── Cross-document consistency analysis ─────────────────────────────────────

def _run_consistency_checks(
    extractions: List[Dict[str, Any]],
    scans: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], str]:
    """Compute cross-document consistency checks and return (checks_dict, per_doc_evidence, overall).

    This is the same algorithm as _run_cross_doc_analysis() in document_intelligence.py
    but lives here so this module has no import dependency on the UI layer.

    Checks: name, address, PAN, Aadhaar, salary (payslip vs offer), forensics.
    Returns a tuple of:
        checks_dict       — {check_name: {"status": str, "detail": str}}
        per_doc_evidence  — list of one dict per extraction with summary fields
        overall_verdict   — 'PASS' or 'FAIL'
    """

    def _forensic_verdict(s: dict) -> str:
        """Pull forensic verdict from the nested layered_analysis_json path."""
        return (s.get("layered_analysis_json") or {}).get("scan_summary", {}).get("forensic_verdict") or ""

    # ── Build a scan lookup keyed by source_name for forensic enrichment ─────
    # Each extraction row has a file_name; each scan row has a source_name.
    # They refer to the same physical document, so we match on the basename.
    scan_by_name: Dict[str, Any] = {}
    for s in scans:
        sname = s.get("source_name") or ""
        if sname:
            scan_by_name[sname] = s

    # ── Build per-document evidence rows ─────────────────────────────────────
    evidence: List[Dict[str, Any]] = []
    for ext in extractions:
        fields   = ext.get("extracted_data") or {}
        doc_type = ext.get("document_type") or "unknown"
        fname    = ext.get("file_name") or ext.get("source_name") or ""

        # Try to find a matching scan row so we can include the forensic verdict
        # and forgery score in the evidence row (used by the Document Inventory table).
        matching_scan = scan_by_name.get(fname) or {}

        evidence.append({
            "file_name":       fname,
            "document_type":   doc_type,
            "name":            _first(fields, _NAME_FIELDS),
            "address":         _first(fields, _ADDRESS_FIELDS),
            "pan":             _first(fields, _PAN_FIELDS),
            "aadhaar":         _first(fields, _AADHAR_FIELDS),
            "salary_payslip":  (
                _first(fields, _SALARY_FIELDS_PAYSLIP) if "payslip" in doc_type.lower() else None
            ),
            "salary_offer":    (
                _first(fields, _SALARY_FIELDS_OFFER)
                if any(kw in doc_type.lower() for kw in ("offer", "increment", "appointment"))
                else None
            ),
            # Forensic fields pulled from the matched scan row (may be empty if no scan exists).
            "forensic_verdict": matching_scan.get("forensic_verdict") or "",
            "forgery_score":    matching_scan.get("forgery_score"),
        })

    # ── Name ─────────────────────────────────────────────────────────────────
    names        = [_normalise_name(r["name"]) for r in evidence if r["name"]]
    unique_names = list(dict.fromkeys(names))
    name_status  = "PASS" if len(unique_names) <= 1 else "MISMATCH"
    name_detail  = (
        f"Values seen: {unique_names}" if len(unique_names) > 1
        else (unique_names[0] if unique_names else "No name found")
    )

    # ── Address ───────────────────────────────────────────────────────────────
    addresses        = [r["address"] for r in evidence if r["address"]]
    unique_addresses = list(dict.fromkeys(addresses))
    address_status   = "PASS" if len(unique_addresses) <= 1 else "MISMATCH"
    address_detail   = (
        f"{len(unique_addresses)} distinct addresses found" if len(unique_addresses) > 1
        else (unique_addresses[0] if unique_addresses else "No address found")
    )

    # ── PAN ───────────────────────────────────────────────────────────────────
    pans       = [_normalise_number(r["pan"]) for r in evidence if r["pan"]]
    unique_pans = list(dict.fromkeys(pans))
    pan_status = "PASS" if len(unique_pans) <= 1 else "MISMATCH"
    pan_detail = (
        f"Values seen: {unique_pans}" if len(unique_pans) > 1
        else (unique_pans[0] if unique_pans else "No PAN found")
    )

    # ── Aadhaar ───────────────────────────────────────────────────────────────
    aadhaars        = [_normalise_number(r["aadhaar"]) for r in evidence if r["aadhaar"]]
    unique_aadhaars = list(dict.fromkeys(aadhaars))
    aadhaar_status  = "PASS" if len(unique_aadhaars) <= 1 else "MISMATCH"
    aadhaar_detail  = (
        f"Values seen: {unique_aadhaars}" if len(unique_aadhaars) > 1
        else (unique_aadhaars[0] if unique_aadhaars else "No Aadhaar found")
    )

    # ── Salary (payslip vs offer) ─────────────────────────────────────────────
    # Allow a 30% tolerance to accommodate deductions and date-of-joining
    # pro-rating differences between payslip net and offer CTC.
    payslip_salaries = [_to_float(r["salary_payslip"]) for r in evidence if r["salary_payslip"]]
    offer_salaries   = [_to_float(r["salary_offer"]) for r in evidence if r["salary_offer"]]
    salary_status    = "SKIP"
    salary_detail    = "No payslip and/or offer salary data available."
    if payslip_salaries and offer_salaries:
        avg_pay = sum(payslip_salaries) / len(payslip_salaries)
        avg_off = sum(offer_salaries)   / len(offer_salaries)
        if avg_off > 0:
            ratio = abs(avg_pay - avg_off) / avg_off
            salary_status = "PASS" if ratio <= 0.30 else "MISMATCH"
            salary_detail = (
                f"Payslip avg Rs {avg_pay:,.0f} vs offer avg Rs {avg_off:,.0f} "
                f"({ratio * 100:.1f}% difference)"
            )

    # ── Forensics ─────────────────────────────────────────────────────────────
    tampered_docs = [
        s.get("source_name", "?") for s in scans
        if "TAMPERED" in _forensic_verdict(s).upper()
    ]
    forensic_status = "CLEAR" if not tampered_docs else "TAMPERED"
    forensic_detail = (
        f"{len(tampered_docs)} document(s) flagged as TAMPERED: {tampered_docs}"
        if tampered_docs
        else f"All {len(scans)} document(s) are forensically clean."
    )

    # ── Overall verdict ────────────────────────────────────────────────────────
    all_checks = [name_status, pan_status, aadhaar_status, salary_status, forensic_status]
    overall    = "FAIL" if any(c in ("MISMATCH", "TAMPERED") for c in all_checks) else "PASS"

    checks = {
        "name":      {"status": name_status,     "detail": name_detail},
        "address":   {"status": address_status,  "detail": address_detail},
        "pan":       {"status": pan_status,       "detail": pan_detail},
        "aadhaar":   {"status": aadhaar_status,   "detail": aadhaar_detail},
        "salary":    {"status": salary_status,    "detail": salary_detail},
        "forensics": {"status": forensic_status,  "detail": forensic_detail},
    }
    return checks, evidence, overall


# ── Evidence markdown builder ────────────────────────────────────────────────

def build_evidence_markdown(
    entity_ref: str,
    entity: Dict[str, Any],
    entity_summary: Dict[str, str],
    extractions: List[Dict[str, Any]],
    scans: List[Dict[str, Any]],
    consistency_checks: Dict[str, Any],
) -> str:
    """Build the comprehensive evidence markdown sent to Gemma4 for analysis.

    The markdown contains every raw extracted_data JSON blob (from document_extractions)
    and every raw layered_analysis_json blob (from scans).  It also includes the
    structured consistency analysis computed locally.

    This document is deliberately verbose — it is input to an AI model, not a
    human-readable report.  The human-readable report is written by Gemma4 based
    on this evidence bundle.

    Returns a single Markdown string ready to be fed into the Gemma4 prompt.
    """
    entity_name  = f"{entity.get('first_name', '')} {entity.get('last_name', '')}".strip()
    generated_at = datetime.now(timezone.utc).strftime("%d %b %Y %H:%M UTC")

    lines: List[str] = [
        "# CANDIDATE EVIDENCE BUNDLE",
        f"> Entity Ref: {entity_ref}  |  Generated: {generated_at}",
        "",
        "---",
        "## CANDIDATE IDENTITY SUMMARY",
        "",
        f"- **Full Name:** {entity_summary.get('full_name') or entity_name or '(not found)'}",
        f"- **Date of Birth:** {entity_summary.get('dob') or '(not found)'}",
        f"- **Address:** {entity_summary.get('address') or '(not found)'}",
        f"- **PAN Number:** {entity_summary.get('pan_number') or '(not found)'}",
        f"- **Aadhaar Number:** {entity_summary.get('aadhaar_number') or '(not found)'}",
        "",
        "---",
        "## ALL DOCUMENT EXTRACTIONS (RAW DATA)",
        "",
        ("Each section below shows the full structured data extracted from one document "
         "by the Gemma4 extraction engine."),
        "",
    ]

    # One sub-section per document extraction — full raw extracted_data JSON blob.
    for idx, ext in enumerate(extractions, start=1):
        fname    = ext.get("file_name") or ext.get("source_name") or f"Document {idx}"
        doc_type = (ext.get("document_type") or "unknown").replace("_", " ").title()
        data     = ext.get("extracted_data") or {}
        lines += [
            f"### Extraction {idx}: {fname}",
            f"*Document type: {doc_type}*",
            "",
            "```json",
            json.dumps(data, indent=2, ensure_ascii=False, default=str),
            "```",
            "",
        ]

    lines += [
        "---",
        "## ALL FORENSIC SCAN RESULTS (RAW DATA)",
        "",
        ("Each section below shows the full 11-layer forensic analysis result "
         "for one scanned document."),
        "",
    ]

    # One sub-section per scan — full raw layered_analysis_json blob.
    for idx, scan in enumerate(scans, start=1):
        src       = scan.get("source_name") or f"Scan {idx}"
        doc_type  = (scan.get("document_type") or "unknown").replace("_", " ").title()
        verdict   = scan.get("forensic_verdict") or "—"
        score     = scan.get("forgery_score")
        la        = scan.get("layered_analysis_json") or {}
        score_str = f"{score:.1f}" if isinstance(score, (int, float)) else "—"
        lines += [
            f"### Scan {idx}: {src}",
            f"*Document type: {doc_type}*  |  *Forensic verdict: {verdict}*  |  *Forgery score: {score_str}*",
            "",
            "```json",
            json.dumps(la, indent=2, ensure_ascii=False, default=str),
            "```",
            "",
        ]

    # ── Cross-document consistency results ────────────────────────────────────
    lines += [
        "---",
        "## CROSS-DOCUMENT CONSISTENCY ANALYSIS",
        "",
        "| Check | Status | Finding |",
        "|---|---|---|",
    ]
    for check_name, chk in (consistency_checks or {}).items():
        status = (chk or {}).get("status", "—")
        detail = (chk or {}).get("detail", "—")
        lines.append(f"| {check_name.capitalize()} | {status} | {detail} |")
    lines.append("")

    return "\n".join(lines)


# ── Focused evidence brief builder ──────────────────────────────────────────

def _build_focused_brief(
    entity_ref: str,
    entity_name: str,
    entity_summary: Dict[str, str],
    per_doc_evidence: List[Dict[str, Any]],
    checks: Dict[str, Any],
    overall_verdict: str,
    extractions: List[Dict[str, Any]],
) -> str:
    """Build a compact, human-readable evidence brief (~2-3K chars) for Gemma4.

    Sending 100K chars of raw JSON to a small model overwhelms it and produces
    generic output.  This function distils the same data into a structured brief
    that clearly names every document, its forensic verdict, its key extracted
    fields, and the cross-document check results.  Gemma4 can then write a
    specific, high-quality narrative from this compact input.
    """
    lines: List[str] = [
        f"CANDIDATE VERIFICATION BRIEF — {entity_ref}",
        f"Subject:  {entity_summary.get('full_name') or entity_name or '(unknown)'}",
        f"DOB:      {entity_summary.get('dob') or '(not found)'}",
        f"PAN:      {entity_summary.get('pan_number') or '(not found)'}",
        f"Aadhaar:  {entity_summary.get('aadhaar_number') or '(not found)'}",
        f"Address:  {entity_summary.get('address') or '(not found)'}",
        f"Total documents reviewed: {len(per_doc_evidence)}",
        "",
        "=== DOCUMENTS REVIEWED ===",
        "",
    ]

    # Build a lookup from file_name → extracted_data for extra field details.
    ext_lookup: Dict[str, Dict] = {
        ext.get("file_name") or ext.get("source_name") or "": ext.get("extracted_data") or {}
        for ext in extractions
    }

    # Group documents by broad category for clarity.
    def _category(doc_type: str) -> str:
        dt = doc_type.lower()
        if any(k in dt for k in ("aadhaar", "pan", "passport", "voter", "dl", "driving")):
            return "GOVERNMENT IDENTITY / TRAVEL"
        if any(k in dt for k in ("payslip", "offer", "appointment", "increment", "salary")):
            return "EMPLOYMENT"
        if any(k in dt for k in ("degree", "marksheet", "certificate", "diploma", "transcript")):
            return "ACADEMIC"
        if any(k in dt for k in ("bank", "statement", "cheque")):
            return "FINANCIAL"
        return "OTHER"

    # Group evidence rows by category.
    by_cat: Dict[str, List[Dict]] = {}
    for ev in per_doc_evidence:
        cat = _category(ev.get("document_type") or "")
        by_cat.setdefault(cat, []).append(ev)

    for cat, evs in by_cat.items():
        lines.append(f"{cat}:")
        for ev in evs:
            fname   = ev.get("file_name") or "—"
            dt      = (ev.get("document_type") or "unknown").replace("_", " ").title()
            fv      = ev.get("forensic_verdict") or "NOT SCANNED"
            score   = ev.get("forgery_score")
            score_s = f" (score: {score:.1f})" if isinstance(score, (int, float)) else ""
            lines.append(f"  • {fname}  [{dt}]  →  Forensic: {fv}{score_s}")

            # Append key extracted values for this document.
            details = []
            if ev.get("name"):
                details.append(f"Name: {ev['name']}")
            if ev.get("pan"):
                details.append(f"PAN: {ev['pan']}")
            if ev.get("aadhaar"):
                details.append(f"Aadhaar: {ev['aadhaar']}")
            if ev.get("address"):
                details.append(f"Address: {ev['address']}")
            if ev.get("salary_payslip"):
                details.append(f"Net salary: {ev['salary_payslip']}")
            if ev.get("salary_offer"):
                details.append(f"Offer CTC/gross: {ev['salary_offer']}")

            # Pull extra notable fields from the raw extraction (e.g. company_name, ctc_per_annum).
            raw = ext_lookup.get(fname) or {}
            for fld in ("company_name", "ctc_per_annum", "gross_monthly", "designation",
                        "institution", "degree_name", "examination_name"):
                v = raw.get(fld)
                if v and str(v).strip():
                    details.append(f"{fld.replace('_', ' ').title()}: {v}")

            if details:
                lines.append("    Fields: " + " | ".join(details))
        lines.append("")

    # Cross-document check results — written clearly so Gemma4 can synthesise them.
    lines += [
        "=== CROSS-DOCUMENT CONSISTENCY CHECKS ===",
        "",
    ]
    for check_name, chk in (checks or {}).items():
        status = (chk or {}).get("status", "—")
        detail = (chk or {}).get("detail", "—")
        icon   = "✅" if status in ("PASS", "CLEAR") else ("➖" if status == "SKIP" else "❌")
        lines.append(f"{icon} {check_name.upper()}: {status} — {detail}")
    lines += ["", f"=== OVERALL VERDICT: {overall_verdict} ===", ""]
    return "\n".join(lines)


# ── Gemma4 narrative writer ─────────────────────────────────────────────────

# System prompt instructs Gemma4 to produce a 10-section structured report that
# matches the format of a professional document verification assessment.
_GEMMA_SYSTEM_PROMPT = """\
You are BaseTruth, a professional document verification intelligence system.

Write a formal 10-section verification report using the evidence brief provided.

STRICT RULES:
- Use ONLY data from the evidence brief — do NOT invent facts, names, or numbers.
- Name every document by its exact filename in your findings.
- Write in plain English. No technical jargon. No raw JSON.
- Use ## for main section headings and ### for sub-headings.
- Every section must be present — do NOT skip any.
- Use ✅ for PASS/clean findings, ❌ for FAIL/tampered/mismatch, ⚠️ for medium risk.

REQUIRED SECTIONS (use exactly these headings with ## prefix):
## 1. EXECUTIVE SUMMARY
(2-3 sentences: what was reviewed, how many documents, overall outcome)

## 2. SUBJECT PROFILE
(A short key-value list: Full Name, DOB, PAN, Aadhaar, Region, Documents Reviewed)

## 3. DOCUMENT REVIEW SUMMARY
(A summary table grouping documents by category: Government Identity, Employment, Academic, Passport)
For each category state the document names and whether they passed or raised concerns.

## 4. KEY FINDINGS
Write one subsection for each category using ### headings:
### A. Identity Validation
### B. Name Consistency Issues
### C. Employment Verification Concerns
### D. Education Verification Concerns
### E. Passport / Address Records
Each subsection must name specific documents and their specific issues.

## 5. FORENSIC RISK SUMMARY
(List the main forensic risk factors observed across the documents and their severity: High / Medium / Low)

## 6. OVERALL RISK CONCLUSION
(2-3 sentences: overall risk level and why)

## 7. FINAL VERDICT
(State: PASS – LOW RISK  or  FAIL – HIGH RISK  with a one-line explanation)

## 8. RECOMMENDED NEXT ACTIONS
(List 4-6 specific, actionable steps for the reviewer)

## 9. DECISION GUIDANCE FOR CLIENTS
(A short guidance table: Use Case | Recommendation — e.g. Hiring Decision, KYC Acceptance, Lending, Compliance)

## 10. CERTIFICATION NOTE
(Standard closing: who prepared this, its status, confidentiality)
"""

_GEMMA_USER_PROMPT_TEMPLATE = """\
Write the BaseTruth verification report for the candidate below.
Follow all 10 sections from your instructions exactly.

{focused_brief}

Write the complete 10-section report now.
"""


def write_gemma_narrative(focused_brief: str) -> Tuple[str, str]:
    """Send the focused evidence brief to Gemma4 and obtain a structured narrative report.

    Uses the provider routing from integrations/ollama.py so it works with
    Ollama (local), GitHub Models, OpenAI, or Anthropic — whichever is configured.

    The focused brief (~2-3K chars) replaces the previous approach of sending the
    entire 100K-char evidence bundle.  A small model (gemma4:e2b) handles a compact,
    human-readable brief much better than raw JSON dumps.

    Returns (narrative_text, source) where:
    - narrative_text: the AI-written 10-section report (or a fallback message)
    - source: 'gemma4 (model_name)' when AI succeeded, 'fallback' otherwise

    This function never raises — any failure produces the fallback text so report
    generation can always succeed even when Gemma4 is offline.
    """
    try:
        from basetruth.integrations.ollama import _route_vlm_chat  # lazy import

        user_prompt = _GEMMA_USER_PROMPT_TEMPLATE.format(focused_brief=focused_brief)

        # Text-only prompt — evidence is already in text form; no images needed.
        content, engine, model, _ = _route_vlm_chat(
            system_prompt=_GEMMA_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            image_bytes_list=[],
        )

        if content and content.strip():
            log.info(
                "write_gemma_narrative: narrative generated successfully",
                extra={"engine": engine, "model": model, "chars": len(content)},
            )
            return content.strip(), f"gemma4 ({model})"

        log.warning("write_gemma_narrative: Gemma4 returned empty response — using fallback")

    except Exception as exc:
        log.warning(
            "write_gemma_narrative: Gemma4 call failed (%s) — using fallback narrative",
            exc,
        )

    # ── Fallback: deterministic placeholder ──────────────────────────────────
    # Gemma4 is offline or returned nothing.  Return a message explaining this
    # so the reviewer knows to check the structured data sections manually.
    fallback = (
        "*Note: AI narrative generation was unavailable at the time this report "
        "was generated.  Please review the Appendix sections for full evidence.*"
    )
    return fallback, "fallback"


# ── Main builder function ────────────────────────────────────────────────────

def build_final_report_json(
    entity_ref: str,
    entity: Dict[str, Any],
    extractions: List[Dict[str, Any]],
    scans: List[Dict[str, Any]],
    cached_narrative: str | None = None,
    cached_narrative_source: str | None = None,
) -> Dict[str, Any]:
    """Build the complete report_json payload for save_entity_report().

    This is the single entry point called from the "Generate Final Report" button
    in document_intelligence.py.  It orchestrates all five pipeline steps and
    returns a structured dict that is:
    - Stored as entity_reports.report_json (JSONB) in PostgreSQL.
    - Used by render_entity_report_pdf() to generate the PDF.
    - Used by render_entity_report_markdown() to generate the Markdown version.

    The returned dict has two layers:
    - Summary-level fields (entity_ref, entity_name, overall_verdict, checks,
      per_document_evidence) — used by the existing PDF/Markdown renderers.
    - Enriched fields (entity_summary, photo_minio_key, document_extractions_raw,
      scan_layered_analysis_raw, evidence_markdown, gemma_narrative) — new in this version.

    Parameters
    ----------
    entity_ref:   BT-XXXXXX reference string.
    entity:       Entity dict from search_entities() / get_entity() — has first_name, last_name, etc.
    extractions:  List of document_information dicts from get_entity_document_information().
    scans:        List of scan dicts from get_entity_scans() — has layered_analysis_json.

    Returns
    -------
    dict  Complete report_json payload ready for save_entity_report().
    """
    log.info(
        "build_final_report_json: starting",
        extra={
            "entity_ref": entity_ref,
            "extractions": len(extractions),
            "scans": len(scans),
        },
    )

    # ── Step 1: Extract canonical identity summary from Aadhaar/PAN rows ─────
    entity_summary = _extract_entity_summary(extractions)
    log.debug(
        "build_final_report_json: identity summary extracted",
        extra={"summary": entity_summary},
    )

    # ── Step 2: Find candidate photo in MinIO ─────────────────────────────────
    photo_minio_key = _find_candidate_photo_key(entity_ref) or ""
    if photo_minio_key:
        log.info("build_final_report_json: photo found", extra={"key": photo_minio_key})
    else:
        log.info("build_final_report_json: no candidate photo found — report will omit photo")

    # ── Step 3: Compute cross-document consistency checks ────────────────────
    checks, per_doc_evidence, overall_verdict = _run_consistency_checks(extractions, scans)
    log.debug(
        "build_final_report_json: consistency checks done",
        extra={"overall": overall_verdict, "check_count": len(checks)},
    )

    # ── Step 4: Build the full evidence markdown (stored in report_json for appendix) ──
    evidence_markdown = build_evidence_markdown(
        entity_ref=entity_ref,
        entity=entity,
        entity_summary=entity_summary,
        extractions=extractions,
        scans=scans,
        consistency_checks=checks,
    )
    log.debug(
        "build_final_report_json: evidence markdown built",
        extra={"chars": len(evidence_markdown)},
    )

    # ── Step 5: Build a compact focused brief and send to Gemma4 ─────────────
    # The focused brief (~2-3K chars) contains the same facts as the full evidence
    # markdown but in a compact, human-readable format that small models handle well.
    entity_name = f"{entity.get('first_name', '')} {entity.get('last_name', '')}".strip()
    focused_brief = _build_focused_brief(
        entity_ref=entity_ref,
        entity_name=entity_name,
        entity_summary=entity_summary,
        per_doc_evidence=per_doc_evidence,
        checks=checks,
        overall_verdict=overall_verdict,
        extractions=extractions,
    )
    # Use a pre-built narrative (e.g. pulled from a previous DB report) to skip
    # the Gemma4 call.  This is used during preview regeneration so we don't
    # have to wait 7 minutes every time we rerender the same report.
    if cached_narrative:
        gemma_narrative = cached_narrative
        gemma_source    = cached_narrative_source or "gemma4 (cached)"
        log.info(
            "build_final_report_json: using cached narrative",
            extra={"gemma_source": gemma_source, "chars": len(gemma_narrative)},
        )
    else:
        gemma_narrative, gemma_source = write_gemma_narrative(focused_brief)
    log.info(
        "build_final_report_json: narrative complete",
        extra={"gemma_source": gemma_source, "chars": len(gemma_narrative)},
    )

    # ── Step 6: Assemble and return the complete report_json payload ──────────
    # entity_name was already computed in step 5 for the focused brief.

    report_json: Dict[str, Any] = {
        # ── Existing summary-level fields (used by current renderers) ─────────
        "entity_ref":             entity_ref,
        "entity_name":            entity_name,
        "overall_verdict":        overall_verdict,
        "documents_analysed":     len(extractions),
        "scans_reviewed":         len(scans),
        "checks":                 checks,
        "per_document_evidence":  per_doc_evidence,
        # ── New enriched fields ───────────────────────────────────────────────
        # entity_summary: identity fields pulled from Aadhaar/PAN rows.
        "entity_summary":         entity_summary,
        # photo_minio_key: MinIO key for the candidate portrait photo (may be empty).
        "photo_minio_key":        photo_minio_key,
        # Full raw blobs for every document and every scan.
        "document_extractions_raw": [
            {
                "file_name":      ext.get("file_name") or ext.get("source_name") or "",
                "document_type":  ext.get("document_type") or "",
                "extracted_data": ext.get("extracted_data") or {},
            }
            for ext in extractions
        ],
        "scan_layered_analysis_raw": [
            {
                "source_name":           scan.get("source_name") or "",
                "document_type":         scan.get("document_type") or "",
                "forensic_verdict":      scan.get("forensic_verdict") or "",
                "forgery_score":         scan.get("forgery_score"),
                "layered_analysis_json": scan.get("layered_analysis_json") or {},
            }
            for scan in scans
        ],
        # The evidence markdown sent to Gemma4.
        "evidence_markdown":      evidence_markdown,
        # The AI-written narrative (or fallback placeholder).
        "gemma_narrative":        gemma_narrative,
        # Which source produced the narrative: 'gemma4 (model)' or 'fallback'.
        "gemma_narrative_source": gemma_source,
        "generated_at":           datetime.now(timezone.utc).isoformat(),
    }

    log.info(
        "build_final_report_json: complete",
        extra={
            "entity_ref": entity_ref,
            "overall":    overall_verdict,
            "gemma":      gemma_source,
            "has_photo":  bool(photo_minio_key),
        },
    )
    return report_json
