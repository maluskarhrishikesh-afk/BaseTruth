from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from basetruth.logger import get_logger, log_timing

log = get_logger(__name__)

from basetruth.analysis.payslip import compare_payslip_summaries
from basetruth.analysis.structured import build_structured_summary
from basetruth.analysis.tamper import evaluate_tamper_risk
from basetruth.integrations.liteparse import check_liteparse_available, parse_document_to_json
from basetruth.integrations.pdf import (
    build_liteparse_json_from_text,
    extract_pdf_metadata,
    extract_text_from_pdf,
    extract_text_via_ocr,
    is_image_file,
    is_image_only_pdf,
    ocr_image_directly,
    extract_image_file_metadata,
)
from basetruth.models import CaseNote, CaseRecord, VerificationReport
from basetruth.reporting.markdown import render_comparison_report, render_scan_report
from basetruth.reporting.pdf import render_scan_report_pdf


class BaseTruthService:
    def __init__(self, artifact_root: Path | str = "artifacts") -> None:
        self.artifact_root = Path(artifact_root)
        self.artifact_root.mkdir(parents=True, exist_ok=True)

    def _document_artifact_dir(self, source_name: str) -> Path:
        stem = Path(source_name).stem.replace(" ", "_")
        target = self.artifact_root / stem
        target.mkdir(parents=True, exist_ok=True)
        return target

    def _config_dir(self) -> Path:
        target = self.artifact_root / "config"
        target.mkdir(parents=True, exist_ok=True)
        return target

    def _case_records_path(self) -> Path:
        return self._config_dir() / "case_records.json"

    def _load_case_records(self) -> Dict[str, CaseRecord]:
        path = self._case_records_path()
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return {}
        items = payload.get("cases", {}) if isinstance(payload, dict) else {}
        records: Dict[str, CaseRecord] = {}
        for case_key, item in items.items():
            if not isinstance(item, dict):
                continue
            notes = [CaseNote(**note) for note in item.get("notes", []) if isinstance(note, dict)]
            records[str(case_key)] = CaseRecord(
                case_key=str(case_key),
                status=str(item.get("status", "new")),
                disposition=str(item.get("disposition", "open")),
                priority=str(item.get("priority", "normal")),
                assignee=str(item.get("assignee", "")),
                labels=[str(label) for label in item.get("labels", [])],
                notes=notes,
                created_at=str(item.get("created_at", "")),
                updated_at=str(item.get("updated_at", "")),
            )
        return records

    def _save_case_records(self, records: Dict[str, CaseRecord]) -> None:
        payload = {
            "schema_version": 1,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "cases": {case_key: record for case_key, record in records.items()},
        }
        self._case_records_path().write_text(json.dumps(payload, default=lambda value: value.__dict__, indent=2, ensure_ascii=False), encoding="utf-8")

    def update_case(
        self,
        case_key: str,
        *,
        status: str | None = None,
        disposition: str | None = None,
        priority: str | None = None,
        assignee: str | None = None,
        labels: List[str] | None = None,
        note_text: str = "",
        note_author: str = "",
    ) -> Dict[str, Any]:
        timestamp = datetime.now(timezone.utc).isoformat()
        records = self._load_case_records()
        record = records.get(case_key) or CaseRecord(case_key=case_key, created_at=timestamp, updated_at=timestamp)
        if status is not None:
            record.status = str(status)
        if disposition is not None:
            record.disposition = str(disposition)
        if priority is not None:
            record.priority = str(priority)
        if assignee is not None:
            record.assignee = str(assignee)
        if labels is not None:
            record.labels = sorted({label.strip() for label in labels if str(label).strip()})
        if note_text.strip():
            record.notes.append(CaseNote(created_at=timestamp, author=str(note_author).strip() or "analyst", text=note_text.strip()))
        record.updated_at = timestamp
        if not record.created_at:
            record.created_at = timestamp
        records[case_key] = record
        self._save_case_records(records)
        return record.__dict__ | {"notes": [note.__dict__ for note in record.notes]}

    def collect_supported_files(self, input_dir: str | Path) -> List[Path]:
        directory = Path(input_dir)
        if not directory.exists() or not directory.is_dir():
            raise NotADirectoryError(directory)
        supported_extensions = {".pdf", ".json", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}
        paths: List[Path] = []
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix.lower() in supported_extensions:
                paths.append(path)
        return paths

    def scan_many(self, input_paths: List[str | Path]) -> List[Dict[str, Any]]:
        return [self.scan_document(path) for path in input_paths]

    def _case_key_for_report(self, report: Dict[str, Any]) -> str:
        summary = report.get("structured_summary", {})
        document_type = summary.get("document", {}).get("type", "generic")
        key_fields = summary.get("key_fields", {})
        parts = [document_type]
        for field_name in ("employee_id", "employee_name", "issuer_name", "account_number"):
            value = key_fields.get(field_name) or summary.get("document", {}).get(field_name)
            if value:
                parts.append(str(value).strip().lower().replace(" ", "_"))
                break
        else:
            parts.append(Path(report.get("source", {}).get("name", "unknown")).stem.lower())
        return "::".join(parts)

    def scan_document(
        self,
        input_path: str | Path,
        forced_entity_ref: Optional[str] = None,
        extra_identity: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        path = Path(input_path)
        if not path.exists():
            log.error("scan_document: file not found", extra={"path": str(path)})
            raise FileNotFoundError(path)

        log.info(
            "scan_document: START",
            extra={"path": str(path), "size_bytes": path.stat().st_size, "suffix": path.suffix.lower()},
        )

        artifact_dir = self._document_artifact_dir(path.name)
        raw_parse_path = artifact_dir / f"{path.stem}_liteparse.json"
        structured_path = artifact_dir / f"{path.stem}_structured.json"
        verification_json_path = artifact_dir / f"{path.stem}_verification.json"
        verification_markdown_path = artifact_dir / f"{path.stem}_verification.md"

        source_path = path
        # Initialise fallback flags here so they are always defined, even when
        # the input is a pre-parsed JSON file that skips the LiteParse step.
        parse_fallback = False
        parse_fallback_reason = ""
        ocr_engine = ""
        image_only_pdf = False
        parse_method = "liteparse"      # updated below as we discover what actually ran
        liteparse_cmd_used: Optional[str] = None
        image_forensics_result: Optional[Dict[str, Any]] = None  # populated for raw image files

        if path.suffix.lower() == ".json" and path.name.endswith("_structured.json"):
            log.info("scan_document: loading pre-built structured JSON", extra={"path": str(path)})
            structured_summary = json.loads(path.read_text(encoding="utf-8"))
            pdf_metadata = {}
            raw_parse_written = ""
            parse_method = "prebuilt_json"

        elif is_image_file(path):
            # ── Raw image file branch (JPEG, PNG, TIFF, BMP, WebP) ──────────
            log.info("scan_document: detected raw image file — running direct OCR", extra={"path": str(path)})
            # 1. OCR the image directly with pytesseract (no Poppler needed)
            ocr_text, ocr_engine = ocr_image_directly(path)
            if ocr_engine == "pytesseract":
                parse_method = "ocr_pytesseract_direct"
                log.info("scan_document: OCR succeeded via pytesseract", extra={"chars": len(ocr_text or "")})
            else:
                parse_method = "no_text_extracted"
                parse_fallback = True
                parse_fallback_reason = (
                    "Direct image OCR unavailable (install pytesseract + Tesseract binary). "
                    "Image forensics signals will still be generated."
                )
                log.warning("scan_document: OCR unavailable for image file", extra={"ocr_engine": ocr_engine})

            # 2. Build the same structured JSON that the rest of the pipeline expects
            fallback_json = build_liteparse_json_from_text(ocr_text or "", path.name)
            raw_parse_path.write_text(
                json.dumps(fallback_json, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            structured_summary = build_structured_summary(raw_parse_path, source_path=source_path)
            structured_summary["parse_method"] = parse_method
            structured_summary["is_image_file"] = True
            if parse_fallback:
                structured_summary["parse_fallback"] = True
                structured_summary["parse_fallback_reason"] = parse_fallback_reason
            if ocr_engine == "pytesseract":
                structured_summary["ocr_engine"] = "pytesseract"
            structured_path.write_text(
                json.dumps(structured_summary, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            raw_parse_written = str(raw_parse_path)

            # 3. File-level metadata (dimensions, format, SHA-256)
            pdf_metadata = extract_image_file_metadata(path)

            # 4. Image forensics (ELA, EXIF, noise analysis)
            try:
                from basetruth.analysis.image_forensics import analyse_image
                image_forensics_result = analyse_image(path)
                log.info("scan_document: image forensics complete", extra={"signals": len(image_forensics_result or {})})
            except Exception:  # noqa: BLE001
                image_forensics_result = None
                log.warning("scan_document: image forensics failed", exc_info=True)

        else:
            # ── PDF / JSON input branch ──────────────────────────────────────
            if path.suffix.lower() == ".json":
                raw_parse_path = path
                parse_method = "json_input"
                log.info("scan_document: JSON input detected", extra={"path": str(path)})
            else:
                log.info("scan_document: checking LiteParse availability")
                liteparse_status = check_liteparse_available()
                if liteparse_status["available"]:
                    log.info("scan_document: running LiteParse", extra={"cmd_source": liteparse_status.get("source", "?")})
                    with log_timing(log, "liteparse", path=str(path)):
                        result = parse_document_to_json(path, raw_parse_path)
                    liteparse_cmd_used = " ".join(str(c) for c in result.get("command", []))
                    if result["status"] != "success":
                        parse_fallback = True
                        parse_fallback_reason = str(result.get("message", "LiteParse parse failed"))
                        parse_method = "pypdf_fallback"
                        log.warning(
                            "scan_document: LiteParse failed — falling back to pypdf",
                            extra={"reason": parse_fallback_reason},
                        )
                    else:
                        parse_method = f"liteparse({result.get('command_source', '?')})"
                        log.info("scan_document: LiteParse succeeded", extra={"parse_method": parse_method})
                else:
                    parse_fallback = True
                    parse_fallback_reason = str(liteparse_status.get("message", "LiteParse not available"))
                    parse_method = "pypdf_fallback"
                    log.warning(
                        "scan_document: LiteParse not available — falling back to pypdf",
                        extra={"reason": parse_fallback_reason},
                    )

                if parse_fallback:
                    log.info("scan_document: extracting text via pypdf fallback")
                    extracted_text = extract_text_from_pdf(path) if path.suffix.lower() == ".pdf" else ""
                    log.debug("scan_document: text extracted", extra={"chars": len(extracted_text)})

                    page_count = 1
                    try:
                        from pypdf import PdfReader  # type: ignore
                        page_count = len(PdfReader(str(path)).pages)
                    except Exception:  # noqa: BLE001
                        pass

                    if is_image_only_pdf(extracted_text, page_count):
                        image_only_pdf = True
                        log.info("scan_document: detected image-only PDF — attempting OCR", extra={"pages": page_count})
                        ocr_text, ocr_engine = extract_text_via_ocr(path)
                        if ocr_text.strip():
                            extracted_text = ocr_text
                            parse_fallback_reason += " | OCR via pytesseract succeeded."
                            parse_method = "ocr_pytesseract"
                            log.info("scan_document: OCR succeeded", extra={"chars": len(ocr_text)})
                        else:
                            if ocr_engine == "unavailable":
                                parse_fallback_reason += " | OCR unavailable (install Tesseract + pdf2image)."
                            elif ocr_engine == "error":
                                parse_fallback_reason += " | OCR attempt failed."
                            else:
                                parse_fallback_reason += " | Image-only PDF: no text layer found."
                            parse_method = "no_text_extracted"
                            log.warning(
                                "scan_document: OCR produced no text",
                                extra={"ocr_engine": ocr_engine, "reason": parse_fallback_reason},
                            )

                    fallback_json = build_liteparse_json_from_text(extracted_text, path.name)
                    raw_parse_path.write_text(
                        json.dumps(fallback_json, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
            structured_summary = build_structured_summary(raw_parse_path, source_path=source_path)
            # Annotate the structured summary so the UI and downstream callers
            # know exactly how the document was parsed and classified.
            structured_summary["parse_method"] = parse_method
            if liteparse_cmd_used:
                structured_summary["liteparse_command"] = liteparse_cmd_used
            if parse_fallback:
                structured_summary["parse_fallback"] = True
                structured_summary["parse_fallback_reason"] = parse_fallback_reason
                if image_only_pdf:
                    structured_summary["is_image_only_pdf"] = True
                if ocr_engine == "pytesseract":
                    structured_summary["ocr_engine"] = "pytesseract"
            structured_path.write_text(json.dumps(structured_summary, indent=2, ensure_ascii=False), encoding="utf-8")
            pdf_metadata = extract_pdf_metadata(source_path) if source_path.suffix.lower() == ".pdf" else {}
            raw_parse_written = str(raw_parse_path)

        if path.suffix.lower() == ".json" and path.name.endswith("_structured.json"):
            structured_path = path
        log.info(
            "scan_document: running tamper risk evaluation",
            extra={
                "doc_type":     structured_summary.get("document", {}).get("type", "generic"),
                "parse_method": parse_method,
                "fallback":     parse_fallback,
            },
        )
        with log_timing(log, "tamper_eval", path=str(path)):
            tamper_assessment = evaluate_tamper_risk(
                structured_summary, pdf_metadata, image_forensics=image_forensics_result
            )
        log.info(
            "scan_document: tamper assessment complete",
            extra={
                "truth_score": tamper_assessment.get("truth_score"),
                "risk_level":  tamper_assessment.get("risk_level"),
                "verdict":     tamper_assessment.get("verdict"),
            },
        )

        source = {
            "path": str(source_path),
            "name": source_path.name,
            "sha256": pdf_metadata.get("sha256", ""),
            "size_bytes": source_path.stat().st_size if source_path.exists() else 0,
        }
        report = VerificationReport(
            schema_version=1,
            generated_at=datetime.now(timezone.utc).isoformat(),
            source=source,
            pdf_metadata=pdf_metadata,
            structured_summary=structured_summary,
            tamper_assessment=tamper_assessment,
            artifacts={
                "raw_parse_path": raw_parse_written,
                "structured_summary_path": str(structured_path),
                "verification_json_path": str(verification_json_path),
                "verification_markdown_path": str(verification_markdown_path),
                "parse_fallback": parse_fallback,
                "parse_fallback_reason": parse_fallback_reason if parse_fallback else "",
                "is_image_only_pdf": image_only_pdf,
                "ocr_engine": ocr_engine,
                "image_forensics": image_forensics_result or {},
            },
        )
        verification_markdown_path.write_text(render_scan_report(report.to_dict()), encoding="utf-8")

        # Plain-English PDF report for loan officers / non-technical reviewers
        pdf_report_path = artifact_dir / f"{path.stem}_report.pdf"
        try:
            with log_timing(log, "pdf_report_gen", path=str(path)):
                pdf_bytes = render_scan_report_pdf(report.to_dict())
            pdf_report_path.write_bytes(pdf_bytes)
            report.artifacts["pdf_report_path"] = str(pdf_report_path)
            log.info("scan_document: PDF report generated", extra={"pdf_path": str(pdf_report_path)})
        except Exception:  # noqa: BLE001 — PDF generation is non-fatal
            report.artifacts["pdf_report_path"] = ""
            log.warning("scan_document: PDF generation failed", exc_info=True)

        # Re-write verification JSON now that pdf_report_path is populated
        report_dict = report.to_dict()
        verification_json_path.write_text(json.dumps(report_dict, indent=2, ensure_ascii=False), encoding="utf-8")

        # Persist to PostgreSQL (non-fatal — file artefacts are always written first)
        try:
            from basetruth.db import init_db
            from basetruth.store import save_scan_to_db
            log.info("scan_document: persisting scan to PostgreSQL")
            init_db()
            db_result = save_scan_to_db(
                report_dict,
                pdf_bytes if "pdf_bytes" in dir() else None,
                forced_entity_ref=forced_entity_ref,
                extra_identity=extra_identity,
            )
            if db_result:
                log.info(
                    "scan_document: DB persist OK",
                    extra={"scan_id": db_result.get("scan_id"), "entity_ref": db_result.get("entity_ref")},
                )
            else:
                log.warning("scan_document: DB persist returned None (DB may be offline)")
        except Exception:  # noqa: BLE001
            log.warning("scan_document: DB persist failed", exc_info=True)

        # Auto-manage case lifecycle based on risk level (non-fatal)
        try:
            _case_key = self._case_key_for_report(report_dict)
            _risk = tamper_assessment.get("risk_level", "low")
            _existing = self._load_case_records()
            _rec = _existing.get(_case_key)
            # Never override a case already closed by an analyst
            if _rec is None or _rec.disposition not in ("cleared", "fraud_confirmed"):
                if _risk in ("high", "medium"):
                    self.update_case(
                        _case_key,
                        status="triage" if _risk == "high" else "new",
                        priority="high" if _risk == "high" else "normal",
                        note_text=(
                            f"Auto-flagged: {_risk.upper()} risk detected in '{path.name}'. "
                            "Please review and Approve or Reject."
                        ),
                        note_author="system",
                    )
                    log.info(
                        "scan_document: case auto-flagged for review",
                        extra={"case_key": _case_key, "risk": _risk},
                    )
                else:
                    # Low risk — auto-approve if no case record exists yet
                    if _rec is None:
                        self.update_case(
                            _case_key,
                            status="closed",
                            disposition="cleared",
                            note_text=f"Auto-approved: LOW risk scan of '{path.name}'.",
                            note_author="system",
                        )
                        log.info(
                            "scan_document: case auto-approved (low risk)",
                            extra={"case_key": _case_key},
                        )
        except Exception:  # noqa: BLE001
            log.warning("scan_document: auto-case management failed", exc_info=True)

        log.info(
            "scan_document: DONE",
            extra={
                "file": path.name,
                "doc_type": structured_summary.get("document", {}).get("type", "generic"),
                "truth_score": tamper_assessment.get("truth_score"),
                "risk_level": tamper_assessment.get("risk_level"),
                "parse_method": parse_method,
            },
        )

        return report_dict

    def compare_payslip_folder(self, input_dir: str | Path) -> Dict[str, Any]:
        directory = Path(input_dir)
        if not directory.exists() or not directory.is_dir():
            raise NotADirectoryError(directory)

        summaries: List[Dict[str, Any]] = []
        for path in sorted(directory.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() == ".json" and path.name.endswith("_structured.json"):
                summaries.append(json.loads(path.read_text(encoding="utf-8")))
            elif path.suffix.lower() == ".pdf":
                scan_report = self.scan_document(path)
                summaries.append(scan_report["structured_summary"])

        comparison = compare_payslip_summaries(summaries)
        comparison_dir = self.artifact_root / "comparisons"
        comparison_dir.mkdir(parents=True, exist_ok=True)
        json_path = comparison_dir / "payslip_comparison.json"
        markdown_path = comparison_dir / "payslip_comparison.md"
        comparison["artifacts"] = {
            "comparison_json_path": str(json_path),
            "comparison_markdown_path": str(markdown_path),
        }
        json_path.write_text(json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8")
        markdown_path.write_text(render_comparison_report(comparison), encoding="utf-8")
        return comparison

    def compare_payslip_summaries_from_reports(self, reports: List[Dict[str, Any]]) -> Dict[str, Any]:
        summaries = [report.get("structured_summary", {}) for report in reports if report.get("structured_summary")]
        return compare_payslip_summaries(summaries)

    def reconcile_income_documents(self, reports: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Cross-document salary reconciliation for a mortgage document bundle.

        Compares salary figures extracted from payslips, Form 16, and employment /
        offer letter.  A large discrepancy (> 15 %) between any two sources is a
        strong indicator of income inflation fraud (tamper type: income_inflated).

        Also checks whether bank-statement salary credits are consistent with
        payslip net pay.

        Returns a dict with:
          anomalies   -- list of anomaly dicts (same schema as compare_payslip_summaries)
          evidence    -- raw extracted figures for transparency
        """
        from basetruth.analysis.packs.base import _parse_int

        payslip_gross_monthly: List[int] = []
        payslip_net_monthly: List[int] = []
        payslip_sources: List[str] = []

        form16_annual_gross: Optional[int] = None
        form16_source: str = ""

        letter_annual_ctc: Optional[int] = None
        letter_gross_monthly: Optional[int] = None
        letter_source: str = ""

        bank_salary_credits: List[int] = []
        bank_source: str = ""

        # Cross-doc mismatch data collection
        payslip_pan: Optional[str] = None
        payslip_pan_source: str = ""
        payslip_bank_name: Optional[str] = None

        form16_pan: Optional[str] = None
        form16_pan_source: str = ""

        bank_statement_bank_name: Optional[str] = None
        bank_statement_ifsc: Optional[str] = None
        bank_statement_all_credits: List[int] = []  # all credit amounts (for gift check)

        gift_amount: Optional[int] = None
        gift_source: str = ""

        for report in reports:
            summary = report.get("structured_summary", {})
            doc_type = str(summary.get("document", {}).get("type", "")).lower()
            key_fields = summary.get("key_fields", {})
            named: Dict[str, Any] = key_fields.get("named_fields") or {}
            source_name = str(report.get("source", {}).get("name", ""))

            if doc_type == "payslip":
                gross = _parse_int(key_fields.get("gross_earnings"))
                if gross:
                    payslip_gross_monthly.append(gross)
                    payslip_sources.append(source_name)
                net = _parse_int(key_fields.get("net_pay"))
                if net:
                    payslip_net_monthly.append(net)
                # PAN for cross-doc check (case_059)
                pan = str(key_fields.get("pan") or "").strip().upper()
                if pan and payslip_pan is None:
                    payslip_pan = pan
                    payslip_pan_source = source_name
                # Bank name on payslip for cross-doc bank-name check (case_051)
                bk = str(key_fields.get("bank") or "").strip()
                if bk and payslip_bank_name is None:
                    payslip_bank_name = bk.upper()

            elif doc_type == "employment_letter" or (
                doc_type == "generic"
                and (
                    key_fields.get("annual_ctc")
                    or key_fields.get("gross_monthly_salary")
                    or "annual_ctc" in named
                    or "gross_monthly_salary" in named
                    or "whomsoever" in str(key_fields.get("title", "")).lower()
                )
            ):
                ctc = _parse_int(
                    key_fields.get("annual_ctc")
                    or key_fields.get("ctc")
                    or named.get("annual_ctc")
                    or named.get("ctc")
                )
                gm = _parse_int(
                    key_fields.get("gross_monthly_salary")
                    or named.get("gross_monthly_salary")
                )
                if ctc and letter_annual_ctc is None:
                    letter_annual_ctc = ctc
                    letter_source = source_name
                if gm and letter_gross_monthly is None:
                    letter_gross_monthly = gm

            elif doc_type == "form16" or (
                doc_type == "generic"
                and (
                    "form 16" in str(key_fields.get("title", "")).lower()
                    or "form-16" in str(key_fields.get("title", "")).lower()
                    or "certificate of tax" in str(key_fields.get("title", "")).lower()
                    or "assessment_year" in named
                    or "gross_salary_total" in named
                    or "tan_of_employer" in named
                )
            ):
                gross = _parse_int(
                    key_fields.get("gross_salary")
                    or key_fields.get("gross_earnings")
                    or named.get("gross_salary_total")
                    or named.get("gross_salary")
                    or named.get("gross_earnings")
                )
                if gross and form16_annual_gross is None:
                    form16_annual_gross = gross
                    form16_source = source_name
                # PAN of employee on Form 16 for cross-doc check (case_059)
                f16_pan = str(named.get("pan_of_employee") or "").strip().upper()
                if f16_pan and form16_pan is None:
                    form16_pan = f16_pan
                    form16_pan_source = source_name

            elif doc_type == "bank_statement":
                if not bank_source:
                    bank_source = source_name
                # Bank name and IFSC from bank statement (for cross-doc bank check, case_051/054)
                if bank_statement_bank_name is None:
                    bank_statement_bank_name = str(key_fields.get("bank_name") or "").upper()
                if bank_statement_ifsc is None:
                    bank_statement_ifsc = str(key_fields.get("ifsc") or "").upper()
                # Collect all bank credit amounts for gift-letter cross-check (case_060)
                for txn in key_fields.get("transactions") or []:
                    cr = _parse_int(txn.get("credit"))
                    if cr and cr > 10_000:
                        bank_statement_all_credits.append(cr)
                # Also collect via named_fields salary-credit path (legacy fallback)
                named_items = list(named.items())
                for idx, (_key, v) in enumerate(named_items):
                    if isinstance(v, str) and "salary credit" in v.lower():
                        for j in range(idx + 1, min(idx + 3, len(named_items))):
                            next_val = named_items[j][1]
                            if isinstance(next_val, str):
                                digits = "".join(ch for ch in next_val if ch.isdigit())
                                if len(digits) >= 4:
                                    amount = int(digits)
                                    if 5_000 <= amount <= 2_000_000:
                                        bank_salary_credits.append(amount)
                                        break

            elif doc_type == "gift_letter":
                g = _parse_int(key_fields.get("gift_amount"))
                if g and gift_amount is None:
                    gift_amount = g
                    gift_source = source_name

        anomalies: List[Dict[str, Any]] = []
        _TOLERANCE = 0.15  # 15 %

        if not payslip_gross_monthly:
            return {"anomalies": anomalies, "evidence": {}}

        avg_monthly_gross = int(sum(payslip_gross_monthly) / len(payslip_gross_monthly))
        annual_from_payslips = avg_monthly_gross * 12

        # --- Check 1: Payslip annualised gross vs Form 16 annual gross ---
        if form16_annual_gross:
            delta_pct = abs(annual_from_payslips - form16_annual_gross) / max(form16_annual_gross, 1)
            if delta_pct > _TOLERANCE:
                anomalies.append({
                    "type": "payslip_vs_form16_salary_mismatch",
                    "severity": "high",
                    "from_period": "payslips",
                    "to_period": form16_source,
                    "details": {
                        "payslip_avg_monthly_gross": avg_monthly_gross,
                        "payslip_annualised_gross": annual_from_payslips,
                        "form16_annual_gross": form16_annual_gross,
                        "discrepancy_pct": round(delta_pct * 100, 1),
                        "payslip_sources": payslip_sources,
                        "form16_source": form16_source,
                        "explanation": (
                            f"Payslips report ₹{annual_from_payslips:,}/yr but Form 16 shows "
                            f"₹{form16_annual_gross:,}/yr — {round(delta_pct*100, 1):.1f}% gap."
                        ),
                    },
                })

        # --- Check 2: Payslip annualised gross vs employment letter CTC ---
        if letter_annual_ctc:
            delta_pct = abs(annual_from_payslips - letter_annual_ctc) / max(letter_annual_ctc, 1)
            if delta_pct > _TOLERANCE:
                anomalies.append({
                    "type": "payslip_vs_offer_letter_salary_mismatch",
                    "severity": "high",
                    "from_period": "payslips",
                    "to_period": letter_source,
                    "details": {
                        "payslip_avg_monthly_gross": avg_monthly_gross,
                        "payslip_annualised_gross": annual_from_payslips,
                        "letter_annual_ctc": letter_annual_ctc,
                        "discrepancy_pct": round(delta_pct * 100, 1),
                        "payslip_sources": payslip_sources,
                        "letter_source": letter_source,
                        "explanation": (
                            f"Payslips report ₹{annual_from_payslips:,}/yr but offer letter "
                            f"states CTC ₹{letter_annual_ctc:,}/yr — {round(delta_pct*100, 1):.1f}% gap."
                        ),
                    },
                })

        # --- Check 3: Employment letter CTC vs Form 16 annual gross ---
        if letter_annual_ctc and form16_annual_gross:
            delta_pct = abs(letter_annual_ctc - form16_annual_gross) / max(form16_annual_gross, 1)
            if delta_pct > _TOLERANCE:
                anomalies.append({
                    "type": "offer_letter_vs_form16_salary_mismatch",
                    "severity": "medium",
                    "from_period": letter_source,
                    "to_period": form16_source,
                    "details": {
                        "letter_annual_ctc": letter_annual_ctc,
                        "form16_annual_gross": form16_annual_gross,
                        "discrepancy_pct": round(delta_pct * 100, 1),
                        "explanation": (
                            f"Offer letter CTC ₹{letter_annual_ctc:,} vs Form 16 gross "
                            f"₹{form16_annual_gross:,} — {round(delta_pct*100, 1):.1f}% gap."
                        ),
                    },
                })

        # --- Check 4: Bank statement salary credits vs payslip net pay ---
        if bank_salary_credits and payslip_net_monthly:
            avg_bank_credit = int(sum(bank_salary_credits) / len(bank_salary_credits))
            avg_net_pay = int(sum(payslip_net_monthly) / len(payslip_net_monthly))
            delta_pct = abs(avg_net_pay - avg_bank_credit) / max(avg_bank_credit, 1)
            if delta_pct > _TOLERANCE:
                anomalies.append({
                    "type": "payslip_net_vs_bank_salary_credit_mismatch",
                    "severity": "high",
                    "from_period": "payslips",
                    "to_period": bank_source,
                    "details": {
                        "payslip_avg_net_pay": avg_net_pay,
                        "bank_avg_salary_credit": avg_bank_credit,
                        "bank_credits_found": len(bank_salary_credits),
                        "discrepancy_pct": round(delta_pct * 100, 1),
                        "bank_source": bank_source,
                        "explanation": (
                            f"Payslips claim net pay ₹{avg_net_pay:,}/mo but bank statement "
                            f"shows salary credits of ₹{avg_bank_credit:,}/mo — "
                            f"{round(delta_pct*100, 1):.1f}% gap."
                        ),
                    },
                })

        # --- Check 5: PAN mismatch between payslip and Form 16 (case_059) ---
        # Each payslip carries the employee's PAN; Form 16 also carries
        # "PAN of Employee".  These must match — a mismatch means one document
        # belongs to a different person and has been swapped into the bundle.
        if payslip_pan and form16_pan:
            pan_match = payslip_pan.upper() == form16_pan.upper()
            if not pan_match:
                anomalies.append({
                    "type": "pan_mismatch_payslip_vs_form16",
                    "severity": "high",
                    "from_period": payslip_pan_source,
                    "to_period": form16_pan_source,
                    "details": {
                        "payslip_pan": payslip_pan,
                        "form16_pan": form16_pan,
                        "explanation": (
                            f"Payslip PAN ({payslip_pan}) does not match Form 16 PAN "
                            f"({form16_pan}). The documents may belong to different people."
                        ),
                    },
                })

        # --- Check 6: Bank name on payslip vs bank statement (case_051) ---
        # The payslip lists the salaried employee's bank (e.g. "HDFC Bank").
        # The bank statement submitted should be from the same bank.
        # A mismatch indicates either the wrong bank statement was submitted or
        # the payslip bank field was tampered to match a real bank statement.
        if payslip_bank_name and bank_statement_bank_name:
            # Normalise: uppercase, remove "Bank" / "Ltd" suffixes for comparison
            def _norm_bank(name: str) -> str:
                return re.sub(r"\b(BANK|LTD|LIMITED|FINANCIAL|OF|INDIA)\b", "", name.upper()).strip()

            norm_payslip = _norm_bank(payslip_bank_name)
            norm_stmt    = _norm_bank(bank_statement_bank_name)
            # Check if any word from payslip bank name appears in the statement bank name
            words_payslip = [w for w in norm_payslip.split() if len(w) >= 3]
            bank_names_consistent = any(w in norm_stmt for w in words_payslip)
            if not bank_names_consistent:
                anomalies.append({
                    "type": "bank_name_payslip_vs_statement_mismatch",
                    "severity": "high",
                    "from_period": payslip_pan_source or "payslip",
                    "to_period": bank_source,
                    "details": {
                        "payslip_bank": payslip_bank_name,
                        "bank_statement_bank": bank_statement_bank_name,
                        "bank_statement_ifsc": bank_statement_ifsc,
                        "explanation": (
                            f"Payslip lists bank as '{payslip_bank_name}' but the submitted "
                            f"bank statement is from '{bank_statement_bank_name}' "
                            f"(IFSC: {bank_statement_ifsc}). Wrong bank statement submitted."
                        ),
                    },
                })

        # --- Check 7: Gift letter amount appearing as bank credit (case_060) ---
        # A gift letter declares a sum that was "gifted" for the home-loan down payment.
        # If that exact amount appears as an incoming bank credit within the same bundle,
        # it suggests the "gift" is actually a loan repayment or circular transfer used
        # to inflate the applicant's apparent savings balance.
        if gift_amount and bank_statement_all_credits:
            _GIFT_TOL = 0.02  # 2% tolerance
            # Check both credits and all transaction amounts (parser may mis-classify
            # gift inflows as debit if the description lacks standard credit keywords)
            _all_txn_amounts = list(bank_statement_all_credits)
            for _rpt in reports:
                _kf = _rpt.get("structured_summary", {}).get("key_fields", {})
                if _rpt.get("structured_summary", {}).get("document", {}).get("type") == "bank_statement":
                    for _t in _kf.get("transactions") or []:
                        _dv = _parse_int(_t.get("debit"))
                        if _dv and _dv > 10_000:
                            _all_txn_amounts.append(_dv)
            matching_credits = [
                c for c in _all_txn_amounts
                if abs(c - gift_amount) / max(gift_amount, 1) <= _GIFT_TOL
            ]
            if matching_credits:
                anomalies.append({
                    "type": "gift_amount_matches_bank_credit",
                    "severity": "high",
                    "from_period": gift_source,
                    "to_period": bank_source,
                    "details": {
                        "gift_amount": gift_amount,
                        "matching_bank_credits": matching_credits[:3],
                        "explanation": (
                            f"Gift letter declares ₹{gift_amount:,} but the same amount "
                            f"appears as an incoming bank credit in the statement. "
                            "The gift may be a disguised loan or circular transfer."
                        ),
                    },
                })

        return {
            "anomalies": anomalies,
            "evidence": {
                "payslip_avg_monthly_gross": avg_monthly_gross,
                "payslip_annualised_gross": annual_from_payslips,
                "payslip_count": len(payslip_gross_monthly),
                "payslip_sources": payslip_sources,
                "form16_annual_gross": form16_annual_gross,
                "form16_source": form16_source,
                "letter_annual_ctc": letter_annual_ctc,
                "letter_gross_monthly": letter_gross_monthly,
                "letter_source": letter_source,
                "bank_avg_salary_credit": int(sum(bank_salary_credits) / len(bank_salary_credits)) if bank_salary_credits else None,
                "bank_salary_credit_count": len(bank_salary_credits),
                "bank_source": bank_source,
            },
        }

    def list_reports(self) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for path in sorted(self.artifact_root.rglob("*_verification.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                continue
            results.append(
                {
                    "kind": "verification",
                    "path": str(path),
                    "source_name": payload.get("source", {}).get("name", ""),
                    "risk_level": payload.get("tamper_assessment", {}).get("risk_level", ""),
                    "truth_score": payload.get("tamper_assessment", {}).get("truth_score", ""),
                    "case_key": self._case_key_for_report(payload),
                    "document_type": payload.get("structured_summary", {}).get("document", {}).get("type", ""),
                    "generated_at": payload.get("generated_at", ""),
                }
            )
        for path in sorted(self.artifact_root.rglob("*_comparison.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                continue
            results.append(
                {
                    "kind": "comparison",
                    "path": str(path),
                    "source_name": payload.get("document_type", "comparison"),
                    "risk_level": "review" if payload.get("anomalies") else "low",
                    "truth_score": "",
                    "case_key": f"comparison::{payload.get('document_type', 'generic')}",
                    "document_type": payload.get("document_type", ""),
                    "generated_at": payload.get("generated_at", ""),
                }
            )
        return results

    def list_cases(self) -> List[Dict[str, Any]]:
        case_records = self._load_case_records()
        cases: Dict[str, Dict[str, Any]] = {}
        for item in self.list_reports():
            if item.get("kind") != "verification":
                continue
            case_key = item.get("case_key", "uncategorized")
            case = cases.setdefault(
                case_key,
                {
                    "case_key": case_key,
                    "document_type": item.get("document_type", "generic"),
                    "documents": [],
                    "max_risk_level": "low",
                    "min_truth_score": 100,
                },
            )
            case["documents"].append(item)
            truth_score = item.get("truth_score")
            if isinstance(truth_score, int):
                case["min_truth_score"] = min(case["min_truth_score"], truth_score)
            risk_level = str(item.get("risk_level", "low"))
            if risk_level == "high" or (risk_level == "medium" and case["max_risk_level"] == "low"):
                case["max_risk_level"] = risk_level

        for case in cases.values():
            record = case_records.get(case["case_key"])
            case["document_count"] = len(case["documents"])
            case["documents"] = sorted(case["documents"], key=lambda item: str(item.get("generated_at", "")), reverse=True)
            if case["min_truth_score"] == 100 and not case["documents"]:
                case["min_truth_score"] = ""
            case["status"] = record.status if record else "new"
            case["disposition"] = record.disposition if record else "open"
            case["priority"] = record.priority if record else "normal"
            case["assignee"] = record.assignee if record else ""
            case["labels"] = list(record.labels) if record else []
            case["note_count"] = len(record.notes) if record else 0
            case["updated_at"] = record.updated_at if record else case["documents"][0].get("generated_at", "")
            # needs_review is True when risk is elevated AND the case is not yet resolved
            case["needs_review"] = (
                case["max_risk_level"] in ("high", "medium")
                and case["disposition"] not in ("cleared", "fraud_confirmed")
            )
        return sorted(cases.values(), key=lambda item: (str(item.get("max_risk_level")), -int(item.get("document_count", 0))), reverse=True)

    def get_case_detail(self, case_key: str) -> Dict[str, Any]:
        case_records = self._load_case_records()
        for case in self.list_cases():
            if case.get("case_key") == case_key:
                details: List[Dict[str, Any]] = []
                for document in case.get("documents", []):
                    payload = json.loads(Path(document["path"]).read_text(encoding="utf-8"))
                    details.append(payload)
                record = case_records.get(case_key)
                return {
                    "case": case,
                    "workflow": (
                        {
                            "case_key": record.case_key,
                            "status": record.status,
                            "disposition": record.disposition,
                            "priority": record.priority,
                            "assignee": record.assignee,
                            "labels": list(record.labels),
                            "notes": [note.__dict__ for note in record.notes],
                            "created_at": record.created_at,
                            "updated_at": record.updated_at,
                        }
                        if record
                        else {
                            "case_key": case_key,
                            "status": "new",
                            "disposition": "open",
                            "priority": "normal",
                            "assignee": "",
                            "labels": [],
                            "notes": [],
                            "created_at": "",
                            "updated_at": "",
                        }
                    ),
                    "reports": details,
                }
        raise KeyError(case_key)
