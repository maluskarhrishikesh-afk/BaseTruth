"""Entity / scan / case store — high-level CRUD over the SQLAlchemy ORM.

All public functions return Python dicts/lists and degrade gracefully
(returning None or []) when the database is unavailable, so the rest of
the application continues to work in file-only mode.
"""
from __future__ import annotations

import os as _os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy import func, or_, text
from sqlalchemy.orm import Session

from basetruth.analysis.upload_authenticity import (
    analyse_upload_authenticity,
    build_format_check,
    build_scan_authenticity_payload,
)
from basetruth.db import (
    DocumentExtraction,
    Entity,
    EntityReport,
    IdentityCheck,
    Scan,
    db_session,
)
from basetruth.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _next_entity_ref(session: Session) -> str:
    """Generate the next BT-XXXXXX reference string.

    We look up the highest existing entity ID and add 1 so each new applicant
    gets a unique, human-readable reference like BT-000042.  Using the existing
    max ID (rather than a database sequence) means the reference stays predictable
    and survives table truncations during development.
    """
    max_id: int = session.query(func.max(Entity.id)).scalar() or 0
    return f"BT-{(max_id + 1):06d}"


def _clean(value: Any) -> str:
    """Return a non-None, stripped string."""
    return str(value).strip() if value else ""


def _normalise_document_file_name(file_name: str, *, fallback: str = "") -> str:
    """Return a safe basename for document_extractions.file_name.

    We store only the leaf filename because the UPSERT key is meant to identify
    one uploaded document for one entity, regardless of the temp folder or local
    machine path used during processing.
    """
    candidate = _clean(file_name)
    if candidate:
        return Path(candidate).name
    fallback_name = _clean(fallback)
    return Path(fallback_name).name if fallback_name else ""


def _upsert_document_extraction(
    session: Session,
    *,
    entity_id: int,
    file_name: str,
    document_type: str,
    extracted_data: Dict[str, Any],
    source_screen: str,
    scan_id: Optional[int],
    fallback_file_name: str = "",
) -> DocumentExtraction:
    """Create or update one extraction row keyed by (entity_id, file_name).

    A document can be rescanned or reclassified later, but operators still think
    of it as the same uploaded file. Using the entity + file name pair as the
    natural key keeps the latest extraction in one stable row.
    """
    clean_file_name = _normalise_document_file_name(file_name, fallback=fallback_file_name)
    if not clean_file_name:
        raise ValueError("document extraction requires a file_name")

    existing_ext = (
        session.query(DocumentExtraction)
        .filter(
            DocumentExtraction.entity_id == entity_id,
            DocumentExtraction.file_name == clean_file_name,
        )
        .first()
    )
    payload = _json_ready(extracted_data)

    if existing_ext is None:
        existing_ext = DocumentExtraction(
            entity_id=entity_id,
            scan_id=scan_id,
            file_name=clean_file_name,
            document_type=document_type,
            extracted_data=payload,
            source_screen=source_screen,
        )
        session.add(existing_ext)
    else:
        existing_ext.scan_id = scan_id
        existing_ext.file_name = clean_file_name
        existing_ext.document_type = document_type
        existing_ext.extracted_data = payload
        existing_ext.source_screen = source_screen

    return existing_ext


def _upsert_identity_document_extraction(
    session: Session,
    *,
    entity_id: int,
    file_name: str,
    document_type: str,
    extracted_data: Dict[str, Any],
) -> None:
    """Create or update one identity-verification extraction row.

    Identity Verification can produce two different document payloads in one run:
    Aadhaar details and PAN details. We store them as two separate rows so the
    Database Viewer and downstream screens can show both documents clearly.
    """
    _upsert_document_extraction(
        session,
        entity_id=entity_id,
        scan_id=None,
        file_name=file_name,
        fallback_file_name=f"identity_verification_{document_type}",
        document_type=document_type,
        extracted_data=extracted_data,
        source_screen="identity_verification",
    )


def extract_identity_fields(report: Dict[str, Any]) -> Dict[str, str]:
    """Pull the searchable identity fields out of a verification report dict."""
    ss = report.get("structured_summary", {})
    kf = ss.get("key_fields", {})
    # named_fields is a common sub-dict used by the generic parser
    nf: Dict[str, Any] = kf.get("named_fields", {}) if isinstance(kf.get("named_fields"), dict) else {}

    def _pick(*keys: str) -> str:
        """Return the first non-empty value from kf or nf."""
        for k in keys:
            for src in (kf, nf):
                v = src.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()
        return ""

    # Name — try many field name variants across doc types
    full_name = _pick(
        "name", "employee_name", "name_of_employee", "applicant_name",
        "account_holder", "employer_name", "donor_name", "beneficiary_name",
        "patient_name",
    )

    # Also try to build from named_fields items that look like full names
    if not full_name:
        for _v in nf.values():
            if isinstance(_v, str) and len(_v.split()) >= 2:
                # Heuristic: looks like a human name (2+ words, mostly letters)
                words = _v.split()
                if all(w[0].isupper() for w in words if w):
                    full_name = _v.strip()
                    break
            elif isinstance(_v, list):
                for item in _v:
                    if isinstance(item, str) and len(item.split()) >= 2:
                        words = item.split()
                        if all(w[0].isupper() for w in words if w):
                            full_name = item.strip()
                            break
                if full_name:
                    break

    name_parts = full_name.split(maxsplit=1)
    first_name = name_parts[0] if name_parts else ""
    last_name = name_parts[1] if len(name_parts) > 1 else ""

    phone = _pick("phone", "mobile", "mobile_number", "contact_number", "phone_number")
    if phone:
        phone = re.sub(r"[^\d+]", "", phone)[:20]

    aadhar = _pick(
        "aadhar_number", "aadhaar_number", "uid_number", "aadhaar", "uid",
        "enrollment_no",
    )
    if not aadhar:
        # Try to find a 12-digit run (Aadhaar format) in named_fields values
        for _v in nf.values():
            if isinstance(_v, str):
                digits = re.sub(r"\s+", "", _v)
                if re.fullmatch(r"\d{12}", digits):
                    aadhar = digits
                    break
    if aadhar:
        aadhar = re.sub(r"\s+", "", aadhar)[:20]

    pan = _pick("pan_number", "pan", "pan_of_employee", "donor_pan")
    if not pan:
        # Try to find a PAN pattern in named_fields
        pan_re = re.compile(r"[A-Z]{5}[0-9]{4}[A-Z]")
        for _v in nf.values():
            if isinstance(_v, str):
                m = pan_re.search(_v.upper())
                if m:
                    pan = m.group()
                    break
    if pan:
        pan = pan.upper()[:20]

    email = _pick("email", "email_id", "email_address")
    if not email:
        # Try to find an email-like string in named_fields
        email_re = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
        for _v in nf.values():
            if isinstance(_v, str):
                m = email_re.search(_v)
                if m:
                    email = m.group()
                    break

    # ── Also try _document_extraction (Gemma4 bulk scan fields) ──────────
    # Bulk scan reports don't have structured_summary — identity fields like
    # employee_name, pan_number, and email come from the Gemma4 extraction
    # payload attached by bulk.py.  We only fall back to this when the
    # structured_summary path returned nothing useful.
    _de = report.get("_document_extraction") or {}
    if _de and not _de.get("_unavailable") and not _de.get("error"):
        if not first_name:
            # Try all common name field variants that Gemma4 might use
            _de_name = (
                _de.get("employee_name") or _de.get("candidate_name")
                or _de.get("account_holder_name") or _de.get("donor_name")
                or _de.get("recipient_name") or ""
            )
            if _de_name:
                _parts = str(_de_name).split(maxsplit=1)
                first_name = _parts[0] if _parts else ""
                last_name = _parts[1] if len(_parts) > 1 else ""
        if not pan:
            _de_pan = str(_de.get("pan_number") or "").strip()
            if _de_pan:
                pan = _de_pan.upper()[:20]
        if not email:
            email = str(_de.get("email") or _de.get("email_id") or "").strip()
        if not phone:
            _de_ph = str(_de.get("phone") or _de.get("mobile") or "").strip()
            if _de_ph:
                phone = re.sub(r"[^\d+]", "", _de_ph)[:20]

    return {
        "first_name": first_name,
        "last_name": last_name,
        "email": email,
        "phone": phone,
        "pan_number": pan,
        "aadhar_number": aadhar,
    }


def _find_or_create_entity(
    session: Session, identity: Dict[str, str]
) -> Optional[Entity]:
    """Look up an existing entity or create a new one.

    Match priority:
      1. PAN number — unique national identifier, most reliable
      2. Aadhaar UID — unique 12-digit identifier, reliable fallback
      3. (first_name, last_name) — last resort; prevents duplicates when no ID number available

    A new entity is only created when at least one identifier is present.
    If the document has no usable fields at all we return None and the scan
    is saved as an unlinked (entity_id = NULL) record.
    """
    entity: Optional[Entity] = None

    if identity.get("pan_number"):
        entity = (
            session.query(Entity)
            .filter(Entity.pan_number == identity["pan_number"])
            .first()
        )

    if entity is None and identity.get("aadhar_number"):
        entity = (
            session.query(Entity)
            .filter(Entity.aadhar_number == identity["aadhar_number"])
            .first()
        )

    # Name-based fallback: match (first_name, last_name) when no unique identifier is available.
    # Prevents duplicate entity rows when the analyst enters a name but no PAN/Aadhaar.
    if entity is None and identity.get("first_name") and identity.get("last_name"):
        fn = identity["first_name"].strip().lower()
        ln = identity["last_name"].strip().lower()
        entity = (
            session.query(Entity)
            .filter(
                func.lower(Entity.first_name) == fn,
                func.lower(Entity.last_name) == ln,
            )
            .first()
        )

    if entity is not None:
        return entity

    # Create a new entity only if we have enough to identify them
    has_identifier = any(
        identity.get(f)
        for f in ("first_name", "pan_number", "aadhar_number", "email", "phone")
    )
    if not has_identifier:
        return None

    entity = Entity(
        entity_ref=_next_entity_ref(session),
        **{k: v for k, v in identity.items() if v},
    )
    session.add(entity)
    session.flush()  # populate id without committing
    return entity


def _entity_to_dict(entity: Entity, session: Session) -> Dict[str, Any]:
    scan_count: int = (
        session.query(func.count(Scan.id)).filter(Scan.entity_id == entity.id).scalar()
        or 0
    )
    latest_scan: Optional[Scan] = (
        session.query(Scan)
        .filter(Scan.entity_id == entity.id)
        .order_by(Scan.generated_at.desc())
        .first()
    )
    return {
        "id": entity.id,
        "entity_ref": entity.entity_ref,
        "first_name": entity.first_name or "",
        "last_name": entity.last_name or "",
        "email": entity.email or "",
        "phone": entity.phone or "",
        "pan_number": entity.pan_number or "",
        "aadhar_number": entity.aadhar_number or "",
        "scan_count": scan_count,
        # risk_level and truth_score columns were removed — derive verdict from layered_analysis_json instead
        "latest_risk": (
            (latest_scan.layered_analysis_json or {}).get("scan_summary", {}).get("forensic_verdict", "").lower()
            if latest_scan else ""
        ),
        "latest_score": (
            (latest_scan.layered_analysis_json or {}).get("scan_summary", {}).get("forgery_score_0_100")
            if latest_scan else None
        ),
        "created_at": (
            entity.created_at.isoformat() if entity.created_at else ""
        ),
    }


def _json_ready(value: Any) -> Any:
    """Recursively convert any value into something PostgreSQL / psycopg2 can store as JSONB.

    PostgreSQL rejects JSON strings that contain null bytes (\\x00) or broken
    Unicode sequences, which often sneak in from EXIF tags inside scanned images.
    This function walks nested dicts / lists and cleans every string.
    NumPy scalars (int64, float32, etc.) are converted to plain Python types
    because psycopg2 does not know how to serialise them.
    """
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, str):
        # PostgreSQL / psycopg2 rejects null bytes (U+0000) and lone surrogate
        # characters inside JSON strings (UntranslatableCharacter error).
        # Strip null bytes; replace any remaining invalid UTF-8 sequences.
        cleaned = value.replace("\x00", "")
        try:
            cleaned.encode("utf-8")
        except UnicodeEncodeError:
            cleaned = cleaned.encode("utf-8", "replace").decode("utf-8")
        return cleaned
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def _build_document_information_payload(report: Dict[str, Any]) -> Dict[str, Any]:
    """Return a rich JSON payload for document_information.extracted_data."""
    structured_summary = report.get("structured_summary") or {}
    key_fields = structured_summary.get("key_fields") or {}
    named_fields = {}
    if isinstance(key_fields.get("named_fields"), dict):
        named_fields = key_fields.get("named_fields") or {}
    gemma4_analysis = report.get("gemma4_analysis") or {}

    # When no per-doc Gemma4 analysis ran (e.g. bulk batch mode), build a
    # synthetic analysis from LiteParse key_fields + tamper_assessment so the
    # Document Intelligence page always has rich data to display.
    if not gemma4_analysis:
        artifacts = report.get("artifacts") or {}
        batch_class = artifacts.get("batch_classification") or {}
        tamper = report.get("tamper_assessment") or {}
        doc = structured_summary.get("document") or {}
        doc_type = (
            batch_class.get("document_type")
            or doc.get("type", "generic")
        )
        confidence = float(
            batch_class.get("confidence")
            or doc.get("type_confidence")
            or 0.0
        )
        truth_score = int(tamper.get("truth_score") or 100)
        risk_level = tamper.get("risk_level") or "unknown"
        parse_method = structured_summary.get("parse_method", "liteparse")

        # Extracted fields from LiteParse (accurate field extraction)
        extracted_fields: Dict[str, Any] = {}
        for _k, _v in key_fields.items():
            if _k == "named_fields" or _v in (None, "", [], {}):
                continue
            extracted_fields[_k] = _v
        for _k, _v in named_fields.items():
            if _v not in (None, "", [], {}):
                extracted_fields[_k] = _v

        # Fraud signals from rule-based tamper assessment.
        # Signal fields are: name, severity, summary, passed, details (see models.Signal).
        # Only include signals that failed (passed == False) with a non-empty name.
        raw_signals = tamper.get("signals") or []
        fraud_signals = [
            {
                "type": s.get("name") or "unknown",
                "severity": s.get("severity", "low"),
                "description": s.get("summary") or "",
            }
            for s in raw_signals
            if s.get("passed") is False and (s.get("name") or s.get("summary"))
        ]

        # Authenticity verdict derived from truth_score
        if truth_score >= 80:
            verdict, assess_conf = "authentic", truth_score / 100
        elif truth_score >= 50:
            verdict, assess_conf = "suspicious", (100 - truth_score) / 100
        else:
            verdict, assess_conf = "tampered", (100 - truth_score) / 100

        reasons = [f"Truth score: {truth_score}/100", f"Risk level: {risk_level}"]
        for _s in raw_signals:
            _reason = _s.get("reason") or _s.get("description") or ""
            if _reason:
                reasons.append(_reason)

        summary = (
            f"{doc_type.replace('_', ' ').title()} — "
            f"truth score {truth_score}/100, risk {risk_level}. "
            f"Extracted via {parse_method}."
        )

        gemma4_analysis = {
            "document_type": doc_type,
            "confidence": confidence,
            "extracted_fields": extracted_fields,
            "fraud_signals": fraud_signals,
            "authenticity_assessment": {
                "verdict": verdict,
                "confidence": assess_conf,
                "reasons": reasons,
            },
            "summary": summary,
            "source": "liteparse+tamper",
            "model": "liteparse+rule_based",
        }

    return _json_ready(
        {
            "document_type": (structured_summary.get("document") or {}).get("type", "generic"),
            "document": structured_summary.get("document") or {},
            "key_fields": key_fields,
            "named_fields": named_fields,
            "parse_method": structured_summary.get("parse_method", ""),
            "parse_fallback": structured_summary.get("parse_fallback", False),
            "parse_fallback_reason": structured_summary.get("parse_fallback_reason", ""),
            "parser": structured_summary.get("parser") or {},
            "source": report.get("source") or {},
            "authenticity_checks": build_scan_authenticity_payload(report),
            "tamper_assessment": report.get("tamper_assessment") or {},
            "signals": report.get("signals") or [],
            "gemma4_analysis": gemma4_analysis,
        }
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def save_scan_to_db(
    report: Dict[str, Any],
    pdf_bytes: Optional[bytes] = None,
    forced_entity_ref: Optional[str] = None,
    extra_identity: Optional[Dict[str, str]] = None,
    layered_screen_name: str = "Scan Document",
) -> Optional[Dict[str, Any]]:
    """Persist a completed verification report (+ optional PDF) to the database.

    Parameters
    ----------
    report:             Full verification report dict.
    pdf_bytes:          PDF report bytes (optional; loaded from artifact path if absent).
    forced_entity_ref:  When provided, the scan is linked to this existing entity
                        and auto-detection is skipped.  Use when the UI operator
                        explicitly selects a person before scanning.
    extra_identity:     Additional identity fields (email, pan_number, phone, …)
                        supplied by the operator. Merged with auto-extracted fields
                        so that even documents without embedded PAN/Aadhaar get
                        linked to the right entity.
    """
    _src_name = report.get("source", {}).get("name", "unknown")
    log.info(
        "save_scan_to_db: BEGIN",
        extra={
            "source": _src_name,
            "forced_entity_ref": forced_entity_ref or "",
            "has_extra_identity": bool(extra_identity),
            "extra_identity_keys": list((extra_identity or {}).keys()),
        },
    )
    try:
        with db_session() as session:
            entity = None

            # ── Force-link to an explicitly chosen entity ─────────────────
            if forced_entity_ref:
                entity = (
                    session.query(Entity)
                    .filter(Entity.entity_ref == forced_entity_ref)
                    .first()
                )
                if entity:
                    log.info(
                        "save_scan_to_db: force-linked to existing entity",
                        extra={"entity_ref": forced_entity_ref, "entity_id": entity.id},
                    )
                else:
                    log.warning(
                        "save_scan_to_db: forced_entity_ref not found — falling back to auto-detect",
                        extra={"entity_ref": forced_entity_ref},
                    )

            # ── Auto-detect / create entity ────────────────────────────────
            if entity is None:
                identity = extract_identity_fields(report)
                log.debug(
                    "save_scan_to_db: auto-extracted identity from document",
                    extra={k: v for k, v in identity.items() if v},
                )
                # Operator-supplied fields always WIN over document-extracted fields.
                if extra_identity:
                    for k, v in extra_identity.items():
                        if v:
                            identity[k] = v
                    log.info(
                        "save_scan_to_db: merged operator-supplied identity",
                        extra={k: v for k, v in identity.items() if v},
                    )
                entity = _find_or_create_entity(session, identity)
                if entity:
                    log.info(
                        "save_scan_to_db: entity resolved",
                        extra={"entity_ref": entity.entity_ref, "entity_id": entity.id, "is_new": entity.id is None},
                    )
                else:
                    log.warning("save_scan_to_db: could not resolve or create entity")

            # When the operator explicitly supplied identity fields, force-update
            # the entity record — even if those fields were already populated from
            # an earlier (possibly incorrect) scan.  This lets analysts correct
            # a wrong name or PAN without having to delete and re-create the entity.
            if extra_identity and entity:
                _updated_fields = []
                for k, v in extra_identity.items():
                    if v:
                        setattr(entity, k, v)
                        _updated_fields.append(k)
                entity.updated_at = datetime.now(timezone.utc)  # type: ignore[assignment]
                log.info(
                    "save_scan_to_db: entity enriched with operator fields",
                    extra={"entity_ref": entity.entity_ref, "updated_fields": _updated_fields},
                )

            # Sanitize the full report dict before storing as JSONB — this strips
            # null bytes (U+0000) and lone surrogates that psycopg2 cannot encode.
            # EXIF fields from image files (e.g. ImageDescription) often contain \x00.
            report = _json_ready(report)

            ss = report.get("structured_summary", {})
            # The document_type comes from the report directly (new forensics-only format)
            # or from structured_summary (old OCR format from single Scan Document screen).
            _doc_type = (
                report.get("document_type")
                or (ss.get("document") or {}).get("type", "generic")
            )

            scan = None
            if entity is not None:
                scan = (
                    session.query(Scan)
                    .filter(
                        Scan.entity_id == entity.id,
                        Scan.source_name == report.get("source", {}).get("name", ""),
                        Scan.document_type == _doc_type,
                    )
                    .order_by(Scan.generated_at.desc())
                    .first()
                )

            if scan is None:
                # Create a new scan row with only the columns that still exist in the schema.
                # Old OCR columns (truth_score, risk_level, verdict, parse_method,
                # report_json, pdf_report) have been dropped from the table.
                scan = Scan(
                    entity_id=entity.id if entity else None,
                    source_name=report.get("source", {}).get("name", ""),
                    source_sha256=report.get("source", {}).get("sha256", ""),
                    document_type=_doc_type,
                )
                session.add(scan)
            else:
                # Update mutable fields on the existing row
                scan.entity_id = entity.id if entity else None
                scan.source_sha256 = report.get("source", {}).get("sha256", "")
                scan.generated_at = datetime.now(timezone.utc)

            if entity is not None:
                stale_scans = (
                    session.query(Scan)
                    .filter(
                        Scan.entity_id == entity.id,
                        Scan.source_name == scan.source_name,
                        Scan.document_type == scan.document_type,
                        Scan.id != scan.id,
                    )
                    .all()
                )
                for stale_scan in stale_scans:
                    session.delete(stale_scan)

            # ── Run image forensics (non-fatal; result stored in JSONB column) ─
            # First check if forensics was already computed at scan time (Bulk Scan
            # pre-runs all 11 layers and stores the result in report["_layered_forensics"]).
            # If that key is present we use it directly — no need to re-run the analysis.
            # If it is missing we fall back to running forensics from the source file path.
            _forensics_result = report.get("_layered_forensics")
            if _forensics_result is None:
                # Fall back: compute forensics now — this happens for documents saved via
                # Scan Document (single-scan), which does not pre-run forensics.
                _source_path_str = report.get("source", {}).get("path", "")
                if _source_path_str and Path(_source_path_str).exists():
                    try:
                        from basetruth.analysis.image_forensics_detect import (  # noqa: PLC0415
                            run_forensics,
                            run_forensics_on_pdf,
                        )
                        _IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
                        _ext = Path(_source_path_str).suffix.lower()
                        if _ext in _IMAGE_EXTS:
                            _forensics_result = run_forensics(_source_path_str)
                        elif _ext == ".pdf":
                            _forensics_result = run_forensics_on_pdf(_source_path_str)
                    except Exception as _fe:
                        log.warning("save_scan_to_db: forensics failed (non-fatal): %s", _fe)
            if _forensics_result:
                scan.layered_analysis_json = _json_ready(_forensics_result)
                log.info(
                    "save_scan_to_db: forensics stored",
                    extra={
                        "source": report.get("source", {}).get("name", ""),
                        "verdict": _forensics_result.get("scan_summary", {}).get("forensic_verdict", ""),
                        "score": _forensics_result.get("scan_summary", {}).get("forgery_score_0_100", 0),
                        "precomputed": report.get("_layered_forensics") is not None,
                    },
                )

            session.flush()
            log.info(
                "save_scan_to_db: scan row upserted",
                extra={"scan_id": scan.id, "document_type": scan.document_type},
            )

            result = {
                "scan_id": scan.id,
                "entity_ref": entity.entity_ref if entity else None,
            }

            # ── Persist DocumentExtraction ────────────────────────────────
            # Two paths to create a DocumentExtraction row:
            #
            # Path A — Single Scan Document screen (OCR): the report has a
            #   'structured_summary' key with the full OCR payload.
            #
            # Path B — Bulk Scan (Gemma4 extraction): the report has a
            #   '_document_extraction' key with fields pulled by Gemma4.
            #   We only save this if the extraction is non-empty and did not
            #   fail (no 'error' key and not '_unavailable').
            if entity and report.get("structured_summary"):
                _doc_info_payload = _build_document_information_payload(report)
                _doc_type = ss.get("document", {}).get("type", "generic")
                try:
                    doc_info = _upsert_document_extraction(
                        session,
                        entity_id=entity.id,
                        scan_id=scan.id,
                        file_name=report.get("source", {}).get("name", ""),
                        document_type=_doc_type,
                        extracted_data=_doc_info_payload,
                        source_screen="scan_document",
                        fallback_file_name=f"scan_document_{_doc_type}",
                    )
                    session.flush()
                    log.info(
                        "save_scan_to_db: DocumentExtraction upserted (OCR path)",
                        extra={
                            "doc_info_id": doc_info.id,
                            "entity_ref": entity.entity_ref,
                            "file_name": doc_info.file_name,
                            "document_type": _doc_type,
                            "key_field_count": len(ss.get("key_fields") or {}),
                        },
                    )
                except Exception as di_exc:
                    log.error(
                        "save_scan_to_db: DocumentExtraction creation FAILED",
                        extra={"error": str(di_exc), "entity_id": entity.id, "scan_id": scan.id},
                        exc_info=True,
                    )
            elif entity and "_document_extraction" in report:
                # Bulk Scan path: always save a document_extractions row so operators
                # can see extracted fields in Document Intelligence even when Gemma4/Ollama
                # is offline.  When extraction succeeded we use the Gemma4 payload;
                # when it failed we fall back to a minimal forensics-summary payload.
                _bulk_ext: Dict = report.get("_document_extraction") or {}
                # _has_error is True when Gemma4 explicitly flagged Ollama as offline or returned
                # a hard error.  These are the only cases where extraction genuinely failed.
                _has_error = bool(_bulk_ext.get("error") or _bulk_ext.get("_unavailable"))
                # Gemma4 data is usable when there is no error flag AND the dict is non-empty.
                # We do NOT gate on key count — even a result with all-null fields (e.g. the model
                # could not read the document) is still valid Gemma4 output and must be saved.
                # An empty dict {} means an exception was silently caught in bulk.py before the
                # extraction call completed; treat that the same as "unavailable".
                _has_gemma4_data = not _has_error and bool(_bulk_ext)

                if _has_gemma4_data:
                    # Gemma4 ran and returned a result — save it directly.
                    # Includes cases where _validation_errors is set (partial extraction).
                    _extraction_payload: Dict[str, Any] = _bulk_ext
                else:
                    # Gemma4 was unreachable, returned an error, or bulk.py's try/except
                    # caught an exception before the extraction completed (leaving an empty
                    # dict).  Save a forensics-summary stub so the row is never empty.
                    _fsummary = (report.get("_layered_forensics") or {}).get("scan_summary", {})
                    _extraction_payload = {
                        "document_type": report.get("document_type", "generic"),
                        "source_name": (report.get("source") or {}).get("name", ""),
                        "forensic_verdict": _fsummary.get("forensic_verdict", ""),
                        "forgery_score_0_100": _fsummary.get("forgery_score_0_100"),
                        # Always True here — this branch only runs when no Gemma4 data is available
                        "_extraction_unavailable": True,
                    }
                    log.warning(
                        "save_scan_to_db: using forensics fallback for DocumentExtraction "
                        "(Gemma4 extraction unavailable or produced an empty result)",
                        extra={
                            "source": _src_name,
                            "has_error": _has_error,
                            "bulk_ext_keys": list(_bulk_ext.keys()),
                        },
                    )

                _bulk_doc_type = report.get("document_type", "generic")
                try:
                    doc_info = _upsert_document_extraction(
                        session,
                        entity_id=entity.id,
                        scan_id=scan.id,
                        file_name=report.get("source", {}).get("name", ""),
                        document_type=_bulk_doc_type,
                        extracted_data=_extraction_payload,
                        source_screen="bulk_scan",
                        fallback_file_name=f"bulk_scan_{_bulk_doc_type}",
                    )
                    session.flush()
                    log.info(
                        "save_scan_to_db: DocumentExtraction upserted (bulk Gemma4 path)",
                        extra={
                            "doc_info_id": doc_info.id,
                            "entity_ref": entity.entity_ref,
                            "file_name": doc_info.file_name,
                            "document_type": _bulk_doc_type,
                            "has_gemma4_data": _has_gemma4_data,
                        },
                    )
                except Exception as di_exc2:
                    log.error(
                        "save_scan_to_db: DocumentExtraction (bulk) creation FAILED",
                        extra={"error": str(di_exc2), "entity_id": entity.id, "scan_id": scan.id},
                        exc_info=True,
                    )

            else:
                log.debug("save_scan_to_db: skipping DocumentExtraction — no entity or unknown report format")

        # ── Upload PDF to MinIO (non-fatal) ────────────────────────────────
        entity_ref_for_minio = result.get("entity_ref")
        if entity_ref_for_minio and pdf_bytes:
            source_name = report.get("source", {}).get("name", "unknown")
            stem = Path(source_name).stem
            minio_key = f"{entity_ref_for_minio}/{stem}_report.pdf"
            minio_upload(minio_key, pdf_bytes, "application/pdf")

        log.info(
            "save_scan_to_db: COMPLETE",
            extra={"scan_id": result.get("scan_id"), "entity_ref": result.get("entity_ref")},
        )
        return result
    except Exception as exc:
        log.error("save_scan_to_db FAILED: %s", exc, exc_info=True)
        return None


def search_entities(
    query: str = "",
    search_field: str = "all",
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """Search entities by name / PAN / Aadhaar / email / phone.

    Parameters
    ----------
    query:        search term (empty → return most-recent ``limit`` entities)
    search_field: ``all`` | ``name`` | ``pan`` | ``aadhar`` | ``email`` | ``phone``
    limit:        max rows returned
    """
    try:
        with db_session() as session:
            q = session.query(Entity)
            q_clean = query.strip()
            if not q_clean:
                results = q.order_by(Entity.id.desc()).limit(limit).all()
            elif search_field == "pan":
                results = (
                    q.filter(Entity.pan_number.ilike(f"%{q_clean}%"))
                    .limit(limit)
                    .all()
                )
            elif search_field == "aadhar":
                results = (
                    q.filter(Entity.aadhar_number.ilike(f"%{q_clean}%"))
                    .limit(limit)
                    .all()
                )
            elif search_field == "email":
                results = (
                    q.filter(Entity.email.ilike(f"%{q_clean}%")).limit(limit).all()
                )
            elif search_field == "phone":
                results = (
                    q.filter(Entity.phone.ilike(f"%{q_clean}%")).limit(limit).all()
                )
            elif search_field == "name":
                results = (
                    q.filter(
                        or_(
                            Entity.first_name.ilike(f"%{q_clean}%"),
                            Entity.last_name.ilike(f"%{q_clean}%"),
                        )
                    )
                    .limit(limit)
                    .all()
                )
            else:  # all fields
                results = (
                    q.filter(
                        or_(
                            Entity.first_name.ilike(f"%{q_clean}%"),
                            Entity.last_name.ilike(f"%{q_clean}%"),
                            Entity.pan_number.ilike(f"%{q_clean}%"),
                            Entity.aadhar_number.ilike(f"%{q_clean}%"),
                            Entity.email.ilike(f"%{q_clean}%"),
                            Entity.phone.ilike(f"%{q_clean}%"),
                            Entity.entity_ref.ilike(f"%{q_clean}%"),
                        )
                    )
                    .limit(limit)
                    .all()
                )
            return [_entity_to_dict(e, session) for e in results]
    except Exception as exc:
        log.warning("search_entities failed: %s", exc)
        return []


def get_entity_scans(entity_ref: str) -> List[Dict[str, Any]]:
    """Return all scans for an entity (most-recent first), without PDF bytes."""
    try:
        with db_session() as session:
            entity = (
                session.query(Entity)
                .filter(Entity.entity_ref == entity_ref)
                .first()
            )
            if not entity:
                return []
            scans = (
                session.query(Scan)
                .filter(Scan.entity_id == entity.id)
                .order_by(Scan.generated_at.desc())
                .all()
            )
            latest_scans: Dict[tuple[str, str], Scan] = {}
            for scan in scans:
                key = (scan.source_name or "", scan.document_type or "generic")
                if key not in latest_scans:
                    latest_scans[key] = scan
            return [
                {
                    "id": s.id,
                    "source_name": s.source_name,
                    "document_type": s.document_type or "generic",
                    # Forensic verdict + score derived from layered_analysis_json (new format)
                    "forensic_verdict": (
                        (s.layered_analysis_json or {}).get("scan_summary", {}).get("forensic_verdict", "")
                    ),
                    "forgery_score": (
                        (s.layered_analysis_json or {}).get("scan_summary", {}).get("forgery_score_0_100")
                    ),
                    "layered_analysis_json": s.layered_analysis_json or {},
                    # Two-level approval workflow fields
                    "first_level_approval": s.first_level_approval,
                    "first_level_approved_by": s.first_level_approved_by or "",
                    "first_level_approved_at": (
                        s.first_level_approved_at.isoformat() if s.first_level_approved_at else ""
                    ),
                    "first_level_approval_comment": s.first_level_approval_comment or "",
                    "second_level_approval": s.second_level_approval,
                    "second_level_approved_by": s.second_level_approved_by or "",
                    "second_level_approved_at": (
                        s.second_level_approved_at.isoformat() if s.second_level_approved_at else ""
                    ),
                    "second_level_approval_comment": s.second_level_approval_comment or "",
                    # Legacy single-level approval kept for backwards compat
                    "approved": s.approved,
                    "generated_at": (
                        s.generated_at.isoformat() if s.generated_at else ""
                    ),
                }
                for s in latest_scans.values()
            ]
    except Exception as exc:
        log.warning("get_entity_scans failed: %s", exc)
        return []


def get_entity_document_information(entity_ref: str) -> List[Dict[str, Any]]:
    """Return all document_information rows for an entity (most-recent first).

    Each dict includes the full ``extracted_data`` JSONB payload (which contains
    the Gemma4 analysis when available) together with the source document name,
    document type, and scan timestamp so the Records page can render them.
    """
    try:
        with db_session() as session:
            entity = (
                session.query(Entity)
                .filter(Entity.entity_ref == entity_ref)
                .first()
            )
            if not entity:
                return []
            rows = (
                session.query(DocumentExtraction)
                .filter(DocumentExtraction.entity_id == entity.id)
                .order_by(DocumentExtraction.created_at.desc())
                .all()
            )
            result = []
            for row in rows:
                scan = (
                    session.query(Scan).filter(Scan.id == row.scan_id).first()
                )
                result.append(
                    {
                        "id": row.id,
                        "scan_id": row.scan_id,
                        "file_name": row.file_name or (scan.source_name if scan else ""),
                        "document_type": row.document_type or "generic",
                        "source_name": row.file_name or (scan.source_name if scan else ""),
                        "scan_generated_at": (
                            scan.generated_at.isoformat()
                            if scan and scan.generated_at
                            else ""
                        ),
                        "scan_approved": scan.approved if scan else None,
                        "extracted_data": row.extracted_data or {},
                        "created_at": (
                            row.created_at.isoformat() if row.created_at else ""
                        ),
                    }
                )
            return result
    except Exception as exc:
        log.warning("get_entity_document_information failed: %s", exc)
        return []


def get_scan_pdf(scan_id: int) -> Optional[bytes]:
    """Fetch the PDF report bytes for a specific scan row."""
    try:
        with db_session() as session:
            scan = session.query(Scan).filter(Scan.id == scan_id).first()
            return bytes(scan.pdf_report) if scan and scan.pdf_report else None
    except Exception as exc:
        log.warning("get_scan_pdf failed: %s", exc)
        return None


def list_recent_scans(limit: int = 200) -> List[Dict[str, Any]]:
    """List the most recent scans with entity info (no PDF bytes)."""
    try:
        with db_session() as session:
            scans = (
                session.query(Scan)
                .order_by(Scan.generated_at.desc())
                .limit(limit)
                .all()
            )
            result = []
            for s in scans:
                entity = (
                    session.query(Entity).filter(Entity.id == s.entity_id).first()
                    if s.entity_id
                    else None
                )
                result.append(
                    {
                        "id": s.id,
                        "source_name": s.source_name,
                        "document_type": s.document_type or "generic",
                        "truth_score": s.truth_score,
                        "risk_level": s.risk_level or "low",
                        "verdict": s.verdict or "",
                        "generated_at": (
                            s.generated_at.isoformat() if s.generated_at else ""
                        ),
                        "entity_ref": entity.entity_ref if entity else "—",
                        "entity_name": (
                            f"{entity.first_name or ''} {entity.last_name or ''}".strip()
                            if entity
                            else "—"
                        ),
                    }
                )
            return result
    except Exception as exc:
        log.warning("list_recent_scans failed: %s", exc)
        return []


def _scan_to_summary_dict(s: "Scan", entity: Optional["Entity"]) -> Dict[str, Any]:
    """Convert a Scan ORM row + optional Entity to a flat summary dict for the UI."""
    # Determine the effective approval status from two-level system, falling back to legacy column
    _fl = s.first_level_approval   # 'Y' | 'N' | None
    _sl = s.second_level_approval  # 'Y' | 'N' | None

    # Derive legacy 'approved' value from the two new columns so old code still works
    if _fl == "Y" and _sl == "Y":
        _effective_approved = "approved"   # both levels approved
    elif _fl == "N" or _sl == "N":
        _effective_approved = "rejected"   # at least one level rejected
    else:
        # Fall back to the old single-level column if the new ones haven't been set yet
        _effective_approved = s.approved   # None = pending (could be old record)

    return {
        "id": s.id,
        "source_name": s.source_name,
        "document_type": s.document_type or "generic",
        "generated_at": s.generated_at.isoformat() if s.generated_at else "",
        # Legacy approval — derived from two-level system above
        "approved": _effective_approved,
        "approved_by": s.approved_by or "",
        "approved_at": s.approved_at.isoformat() if s.approved_at else "",
        "approval_comment": s.approval_comment or "",
        # First-level approval (initial reviewer)
        "first_level_approval": _fl,
        "first_level_approved_by": s.first_level_approved_by or "",
        "first_level_approved_at": s.first_level_approved_at.isoformat() if s.first_level_approved_at else "",
        "first_level_approval_comment": s.first_level_approval_comment or "",
        # Second-level approval (senior reviewer)
        "second_level_approval": _sl,
        "second_level_approved_by": s.second_level_approved_by or "",
        "second_level_approved_at": s.second_level_approved_at.isoformat() if s.second_level_approved_at else "",
        "second_level_approval_comment": s.second_level_approval_comment or "",
        "entity_ref": entity.entity_ref if entity else "—",
        "entity_name": (
            f"{entity.first_name or ''} {entity.last_name or ''}".strip() if entity else "—"
        ),
        "layered_analysis_json": s.layered_analysis_json,
    }


def list_all_scans_with_status(limit: int = 200) -> List[Dict[str, Any]]:
    """Return all scans (pending, approved, rejected) ordered newest-first."""
    try:
        with db_session() as session:
            scans = (
                session.query(Scan).order_by(Scan.generated_at.desc()).limit(limit).all()
            )
            result = []
            for s in scans:
                entity = (
                    session.query(Entity).filter(Entity.id == s.entity_id).first()
                    if s.entity_id
                    else None
                )
                result.append(_scan_to_summary_dict(s, entity))
            return result
    except Exception as exc:
        log.warning("list_all_scans_with_status failed: %s", exc)
        return []


def list_pending_scans(limit: int = 200) -> List[Dict[str, Any]]:
    """Return scans where approval is still pending (approved IS NULL)."""
    try:
        with db_session() as session:
            scans = (
                session.query(Scan)
                .filter(Scan.approved.is_(None))
                .order_by(Scan.generated_at.desc())
                .limit(limit)
                .all()
            )
            result = []
            for s in scans:
                entity = (
                    session.query(Entity).filter(Entity.id == s.entity_id).first()
                    if s.entity_id
                    else None
                )
                result.append(_scan_to_summary_dict(s, entity))
            return result
    except Exception as exc:
        log.warning("list_pending_scans failed: %s", exc)
        return []


def list_approved_scans(entity_ref: Optional[str] = None, limit: int = 200) -> List[Dict[str, Any]]:
    """Return approved scans, optionally filtered by entity_ref."""
    try:
        with db_session() as session:
            q = session.query(Scan).filter(Scan.approved == "approved")
            if entity_ref:
                entity = (
                    session.query(Entity)
                    .filter(Entity.entity_ref == entity_ref)
                    .first()
                )
                if entity:
                    q = q.filter(Scan.entity_id == entity.id)
                else:
                    return []
            scans = q.order_by(Scan.generated_at.desc()).limit(limit).all()
            result = []
            for s in scans:
                ent = (
                    session.query(Entity).filter(Entity.id == s.entity_id).first()
                    if s.entity_id
                    else None
                )
                result.append(_scan_to_summary_dict(s, ent))
            return result
    except Exception as exc:
        log.warning("list_approved_scans failed: %s", exc)
        return []


def get_scan_with_forensics(scan_id: int) -> Optional[Dict[str, Any]]:
    """Return a single scan row including its full layered_analysis_json."""
    try:
        with db_session() as session:
            s = session.query(Scan).filter(Scan.id == scan_id).first()
            if s is None:
                return None
            entity = (
                session.query(Entity).filter(Entity.id == s.entity_id).first()
                if s.entity_id
                else None
            )
            return _scan_to_summary_dict(s, entity)
    except Exception as exc:
        log.warning("get_scan_with_forensics(%d) failed: %s", scan_id, exc)
        return None


def approve_scan(
    scan_id: int,
    approved_by: str = "",
    comment: str = "",
) -> Optional[Dict[str, Any]]:
    """Mark a scan as 1st-level approved.  Returns the updated summary dict or None on failure."""
    try:
        with db_session() as session:
            s = session.query(Scan).filter(Scan.id == scan_id).first()
            if s is None:
                log.warning("approve_scan: scan %d not found", scan_id)
                return None
            # Set first-level approval to Y (Yes)
            s.first_level_approval = "Y"
            s.first_level_approved_by = approved_by or None
            s.first_level_approved_at = datetime.now(timezone.utc)
            s.first_level_approval_comment = comment or None
            # Keep the legacy column in sync for backwards-compat reads
            s.approved = "approved"
            s.approved_by = approved_by or None
            s.approved_at = datetime.now(timezone.utc)
            s.approval_comment = comment or None
            session.flush()
            entity = (
                session.query(Entity).filter(Entity.id == s.entity_id).first()
                if s.entity_id
                else None
            )
            result = _scan_to_summary_dict(s, entity)
            log.info(
                "approve_scan: 1st-level approved",
                extra={"scan_id": scan_id, "approved_by": approved_by, "entity_ref": result.get("entity_ref")},
            )
            return result
    except Exception as exc:
        log.error("approve_scan(%d) failed: %s", scan_id, exc, exc_info=True)
        return None


def reject_scan(
    scan_id: int,
    approved_by: str = "",
    comment: str = "",
) -> Optional[Dict[str, Any]]:
    """Mark a scan as 1st-level rejected.  Returns the updated summary dict or None on failure."""
    try:
        with db_session() as session:
            s = session.query(Scan).filter(Scan.id == scan_id).first()
            if s is None:
                log.warning("reject_scan: scan %d not found", scan_id)
                return None
            # Set first-level approval to N (No)
            s.first_level_approval = "N"
            s.first_level_approved_by = approved_by or None
            s.first_level_approved_at = datetime.now(timezone.utc)
            s.first_level_approval_comment = comment or None
            # Keep legacy column in sync
            s.approved = "rejected"
            s.approved_by = approved_by or None
            s.approved_at = datetime.now(timezone.utc)
            s.approval_comment = comment or None
            session.flush()
            entity = (
                session.query(Entity).filter(Entity.id == s.entity_id).first()
                if s.entity_id
                else None
            )
            result = _scan_to_summary_dict(s, entity)
            log.info(
                "reject_scan: 1st-level rejected",
                extra={"scan_id": scan_id, "approved_by": approved_by, "entity_ref": result.get("entity_ref")},
            )
            return result
    except Exception as exc:
        log.error("reject_scan(%d) failed: %s", scan_id, exc, exc_info=True)
        return None


def second_level_approve_scan(
    scan_id: int,
    approved_by: str = "",
    comment: str = "",
) -> Optional[Dict[str, Any]]:
    """Mark a scan as 2nd-level (senior) approved.  Returns updated summary dict or None."""
    try:
        with db_session() as session:
            s = session.query(Scan).filter(Scan.id == scan_id).first()
            if s is None:
                log.warning("second_level_approve_scan: scan %d not found", scan_id)
                return None
            if s.first_level_approval != "Y":
                # 2nd-level approval only makes sense after 1st-level has passed
                log.warning(
                    "second_level_approve_scan: 1st-level not yet approved",
                    extra={"scan_id": scan_id, "first_level": s.first_level_approval},
                )
                return None
            # Set second-level approval to Y (Yes)
            s.second_level_approval = "Y"
            s.second_level_approved_by = approved_by or None
            s.second_level_approved_at = datetime.now(timezone.utc)
            s.second_level_approval_comment = comment or None
            session.flush()
            entity = (
                session.query(Entity).filter(Entity.id == s.entity_id).first()
                if s.entity_id
                else None
            )
            result = _scan_to_summary_dict(s, entity)
            log.info(
                "second_level_approve_scan: 2nd-level approved",
                extra={"scan_id": scan_id, "approved_by": approved_by},
            )
            return result
    except Exception as exc:
        log.error("second_level_approve_scan(%d) failed: %s", scan_id, exc, exc_info=True)
        return None


def second_level_reject_scan(
    scan_id: int,
    approved_by: str = "",
    comment: str = "",
) -> Optional[Dict[str, Any]]:
    """Mark a scan as 2nd-level (senior) rejected.  Returns updated summary dict or None."""
    try:
        with db_session() as session:
            s = session.query(Scan).filter(Scan.id == scan_id).first()
            if s is None:
                log.warning("second_level_reject_scan: scan %d not found", scan_id)
                return None
            # Set second-level approval to N (No)
            s.second_level_approval = "N"
            s.second_level_approved_by = approved_by or None
            s.second_level_approved_at = datetime.now(timezone.utc)
            s.second_level_approval_comment = comment or None
            session.flush()
            entity = (
                session.query(Entity).filter(Entity.id == s.entity_id).first()
                if s.entity_id
                else None
            )
            result = _scan_to_summary_dict(s, entity)
            log.info(
                "second_level_reject_scan: 2nd-level rejected",
                extra={"scan_id": scan_id, "approved_by": approved_by},
            )
            return result
    except Exception as exc:
        log.error("second_level_reject_scan(%d) failed: %s", scan_id, exc, exc_info=True)
        return None


def update_entity(entity_ref: str, fields: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """Update mutable identity fields on an entity record."""
    allowed = {"first_name", "last_name", "email", "phone", "pan_number", "aadhar_number"}
    try:
        with db_session() as session:
            entity = (
                session.query(Entity)
                .filter(Entity.entity_ref == entity_ref)
                .first()
            )
            if not entity:
                return None
            for k, v in fields.items():
                if k in allowed:
                    setattr(entity, k, _clean(v))
                entity.updated_at = datetime.now(timezone.utc)  # type: ignore[assignment]
            return _entity_to_dict(entity, session)
    except Exception as exc:
        log.warning("update_entity failed: %s", exc)
        return None


def db_stats() -> Dict[str, int]:
    """Return high-level counts for the dashboard."""
    try:
        with db_session() as session:
            return {
                "entities": session.query(func.count(Entity.id)).scalar() or 0,
                "scans": session.query(func.count(Scan.id)).scalar() or 0,
                # risk_level column removed — count scans with TAMPERED forensic verdict instead
                "high_risk": (
                    session.query(func.count(Scan.id))
                    .filter(
                        Scan.layered_analysis_json["scan_summary"]["forensic_verdict"].astext.in_(
                            ["TAMPERED", "LIKELY TAMPERED"]
                        )
                    )
                    .scalar()
                    or 0
                ),
            }
    except Exception as exc:
        log.warning("db_stats failed: %s", exc)
        return {"entities": 0, "scans": 0, "high_risk": 0}


def db_dashboard_stats() -> Dict[str, Any]:
    """Extended stats for the Dashboard — single round-trip query."""
    try:
        with db_session() as session:
            total_scans = session.query(func.count(Scan.id)).scalar() or 0
            # risk_level column is removed — derive counts from layered_analysis_json forensic_verdict
            high_risk = (
                session.query(func.count(Scan.id))
                .filter(
                    Scan.layered_analysis_json["scan_summary"]["forensic_verdict"].astext == "TAMPERED"
                )
                .scalar() or 0
            )
            medium_risk = (
                session.query(func.count(Scan.id))
                .filter(
                    Scan.layered_analysis_json["scan_summary"]["forensic_verdict"].astext == "LIKELY TAMPERED"
                )
                .scalar() or 0
            )
            low_risk = (
                session.query(func.count(Scan.id))
                .filter(
                    Scan.layered_analysis_json["scan_summary"]["forensic_verdict"].astext.in_(["ORIGINAL", "UNCERTAIN"])
                )
                .scalar() or 0
            )
            entities = session.query(func.count(Entity.id)).scalar() or 0
            # truth_score column is removed — derive average from layered_analysis_json forgery_score
            # Use a fallback of None when the column has no data yet
            avg_score_row = session.execute(
                text(
                    "SELECT AVG((layered_analysis_json->'scan_summary'->>'forgery_score_0_100')::float) "
                    "FROM scans WHERE layered_analysis_json IS NOT NULL"
                )
            ).scalar()
            avg_score = round(float(avg_score_row), 1) if avg_score_row is not None else None
            # Risk distribution per entity (for bar chart)
            risk_by_entity = []
            for e in session.query(Entity).order_by(Entity.id.desc()).limit(20).all():
                scan_count = session.query(func.count(Scan.id)).filter(Scan.entity_id == e.id).scalar() or 0
                if scan_count:
                    risk_by_entity.append({
                        "entity_ref": e.entity_ref,
                        "name": f"{e.first_name or ''} {e.last_name or ''}".strip() or e.entity_ref,
                        "scans": scan_count,
                    })
            return {
                "entities": entities,
                "total_scans": total_scans,
                "high_risk": high_risk,
                "medium_risk": medium_risk,
                "low_risk": low_risk,
                "avg_score": avg_score,
                "pending_review": 0,
                "auto_approved": 0,
                "rejected": 0,
                "total_cases": 0,
                "risk_by_entity": risk_by_entity,
            }
    except Exception as exc:
        log.warning("db_dashboard_stats failed: %s", exc)
        return {}


def get_entity_latest_pdf(entity_ref: str) -> tuple[Optional[bytes], Optional[str]]:
    """Return (pdf_bytes, source_name) for the most recent scan with a PDF for this entity."""
    try:
        with db_session() as session:
            entity = (
                session.query(Entity)
                .filter(Entity.entity_ref == entity_ref)
                .first()
            )
            if not entity:
                return None, None
            scan = (
                session.query(Scan)
                .filter(Scan.entity_id == entity.id, Scan.pdf_report.isnot(None))
                .order_by(Scan.generated_at.desc())
                .first()
            )
            if scan and scan.pdf_report:
                return bytes(scan.pdf_report), scan.source_name
            return None, None
    except Exception as exc:
        log.warning("get_entity_latest_pdf failed: %s", exc)
        return None, None


# ---------------------------------------------------------------------------
# Identity check (face match / Video KYC) persistence
# ---------------------------------------------------------------------------


def save_identity_check(
    check_type: str,
    result: Dict[str, Any],
    forced_entity_ref: Optional[str] = None,
    extra_identity: Optional[Dict[str, str]] = None,
    doc_filename: str = "",
    selfie_filename: str = "",
    pdf_bytes: Optional[bytes] = None,
    doc_bytes: Optional[bytes] = None,
    selfie_bytes: Optional[bytes] = None,
    pan_filename: str = "",
    pan_bytes: Optional[bytes] = None,
    pan_signature_bytes: Optional[bytes] = None,
) -> Optional[Dict[str, Any]]:
    """Persist a face-match or Video KYC result to the DB.

    Parameters
    ----------
    check_type:          'face_match' or 'video_kyc'
    result:              The result dict from compare_faces() or the KYC processor.
    entity_ref:          Link to an existing entity (BT-XXXXXX). If None, unlinked.
    doc_filename:        Original ID document filename.
    selfie_filename:     Selfie filename (face_match only).
    pdf_bytes:           Generated PDF report bytes.
    pan_signature_bytes: JPEG bytes of the cropped PAN card signature strip. When
                         provided, the image is uploaded to MinIO as
                         ``{entity_ref}/pan_signature.jpg`` and its key is stored
                         inside the ``pan_card`` ``document_extractions`` row as
                         ``pan_signature_minio_key`` for downstream signature analysis.
    """
    try:
        with db_session() as session:
            entity = None

            if forced_entity_ref:
                entity = (
                    session.query(Entity)
                    .filter(Entity.entity_ref == forced_entity_ref)
                    .first()
                )

            if entity is None and extra_identity:
                entity = _find_or_create_entity(session, extra_identity)

            if extra_identity and entity:
                for k, v in extra_identity.items():
                    if v:
                        setattr(entity, k, v)
                entity.updated_at = func.now()

            entity_id = entity.id if entity else None
            entity_ref = entity.entity_ref if entity else None

            # Determine status and verdict
            if check_type == "face_match":
                is_match = result.get("match", False)
                status = "pass" if is_match else "fail"
                verdict = "PASS" if is_match else "FAIL"
            else:  # video_kyc
                is_match = result.get("is_match", False)
                liveness = result.get("liveness_passed", False)
                if is_match and liveness:
                    status = "pass"
                    verdict = "PASS"
                elif is_match or liveness:
                    status = "inconclusive"
                    verdict = "FAIL"
                else:
                    status = "fail"
                    verdict = "FAIL"

            row = None
            if entity_id is not None:
                row = (
                    session.query(IdentityCheck)
                    .filter(
                        IdentityCheck.entity_id == entity_id,
                        IdentityCheck.check_type == check_type,
                    )
                    .order_by(IdentityCheck.created_at.desc())
                    .first()
                )

            if row is None:
                row = IdentityCheck(
                    entity_id=entity_id,
                    check_type=check_type,
                    status=status,
                    cosine_similarity=result.get("confidence") or result.get("cosine_similarity"),
                    display_score=result.get("display_score"),
                    threshold=result.get("threshold", 0.40),
                    is_match=is_match,
                    liveness_state=result.get("liveness_state"),
                    liveness_passed=result.get("liveness_passed"),
                    verdict=verdict,
                    doc_filename=doc_filename,
                    selfie_filename=selfie_filename,
                    report_json=result,
                    pdf_report=pdf_bytes,
                )
                session.add(row)
            else:
                row.entity_id = entity_id
                row.status = status
                row.cosine_similarity = result.get("confidence") or result.get("cosine_similarity")
                row.display_score = result.get("display_score")
                row.threshold = result.get("threshold", 0.40)
                row.is_match = is_match
                row.liveness_state = result.get("liveness_state")
                row.liveness_passed = result.get("liveness_passed")
                row.verdict = verdict
                row.doc_filename = doc_filename
                row.selfie_filename = selfie_filename
                row.report_json = result
                row.pdf_report = pdf_bytes
                row.created_at = datetime.now(timezone.utc)

            if entity_id is not None:
                stale_rows = (
                    session.query(IdentityCheck)
                    .filter(
                        IdentityCheck.entity_id == entity_id,
                        IdentityCheck.check_type == check_type,
                        IdentityCheck.id != row.id,
                    )
                    .all()
                )
                for stale_row in stale_rows:
                    session.delete(stale_row)
            session.flush()

            if check_type == "video_kyc":
                layered_analysis = result.setdefault("layered_analysis", {})
                upload_authenticity = layered_analysis.setdefault("upload_authenticity", {})
                if doc_filename and doc_bytes and "reference_document" not in upload_authenticity:
                    upload_authenticity["reference_document"] = analyse_upload_authenticity(
                        doc_bytes,
                        doc_filename,
                        format_check=build_format_check(
                            "Reference ID document was uploaded successfully for Video KYC verification.",
                            True,
                        ),
                    )
                if selfie_filename and selfie_bytes and "live_capture" not in upload_authenticity:
                    upload_authenticity["live_capture"] = analyse_upload_authenticity(
                        selfie_bytes,
                        selfie_filename,
                        format_check=build_format_check(
                            "Live capture image was stored successfully for Video KYC verification.",
                            True,
                        ),
                    )

            # ── Save extracted identity fields to document_extractions ───
            # The Identity Verification page stores Aadhaar fields under
            # result['aadhaar_qr'] and PAN fields under result['pan_extraction'].
            # Save both as separate rows so both documents are visible later.
            if entity is not None and check_type == "face_match":
                aadhaar_data = result.get("aadhaar_qr") or {}
                pan_data = result.get("pan_extraction") or {}

                # Older payloads stored these fields at the top level.
                # Keep that fallback so old saves still work.
                if not aadhaar_data:
                    aadhaar_data = {
                        key: result.get(key)
                        for key in (
                            "uid",
                            "aadhaar_number",
                            "name",
                            "dob",
                            "address",
                            "gender",
                            "yob",
                            "dist",
                            "state",
                            "qr_type",
                        )
                        if result.get(key)
                    }
                if not pan_data:
                    pan_data = {
                        key: result.get(key)
                        for key in (
                            "pan_number",
                            "full_name",
                            "father_name",
                            "date_of_birth",
                            "extraction_source",
                            "engine",
                        )
                        if result.get(key)
                    }

                try:
                    if aadhaar_data:
                        _upsert_identity_document_extraction(
                            session,
                            entity_id=entity.id,
                            file_name=doc_filename,
                            document_type="aadhaar",
                            extracted_data=aadhaar_data,
                        )
                        log.info(
                            "save_identity_check: DocumentExtraction saved for aadhaar",
                            extra={"entity_id": entity.id, "doc_type": "aadhaar"},
                        )
                    if pan_data:
                        # ── Upload signature to MinIO before writing the row ──
                        # This allows us to store the MinIO key inside the
                        # document_extractions row so downstream screens can
                        # load the image directly without an additional lookup.
                        if entity_ref and pan_signature_bytes:
                            try:
                                _sig_minio_key = f"{entity_ref}/pan_signature.jpg"
                                minio_upload(_sig_minio_key, pan_signature_bytes, "image/jpeg")
                                pan_data["pan_signature_minio_key"] = _sig_minio_key
                                log.info(
                                    "save_identity_check: PAN signature uploaded to MinIO",
                                    extra={"entity_id": entity.id, "key": _sig_minio_key},
                                )
                            except Exception:
                                log.warning(
                                    "save_identity_check: PAN signature upload failed (non-fatal)",
                                    exc_info=True,
                                )
                        _upsert_identity_document_extraction(
                            session,
                            entity_id=entity.id,
                            file_name=pan_filename,
                            document_type="pan_card",
                            extracted_data=pan_data,
                        )
                        log.info(
                            "save_identity_check: DocumentExtraction saved for pan_card",
                            extra={"entity_id": entity.id, "doc_type": "pan_card"},
                        )
                    if aadhaar_data or pan_data:
                        session.flush()
                except Exception as ext_exc:
                    log.warning(
                        "save_identity_check: DocumentExtraction save failed (non-fatal)",
                        extra={"error": str(ext_exc), "entity_id": entity.id},
                    )

            saved = {
                "id": row.id,
                "entity_ref": entity_ref,
                "check_type": check_type,
                "status": status,
                "verdict": verdict,
            }

            # Upload PDF to MinIO
            if pdf_bytes and entity_ref:
                try:
                    legacy_objects = minio_list_entity_objects(entity_ref)
                    for obj in legacy_objects:
                        key = obj.get("key", "")
                        filename = Path(key).name
                        if filename == f"{check_type}_report.pdf" or filename.endswith(f"_{check_type}_report.pdf"):
                            minio_delete_object(key)
                except Exception:
                    log.warning("save_identity_check: legacy report cleanup failed", exc_info=True)
                minio_key = f"{entity_ref}/{check_type}_report.pdf"
                minio_upload(minio_key, pdf_bytes, "application/pdf")

            if entity_ref and doc_bytes and doc_filename:
                try:
                    minio_upload(
                        f"{entity_ref}/{Path(doc_filename).name}",
                        doc_bytes,
                        "application/octet-stream",
                    )
                except Exception:
                    log.warning("save_identity_check: source document upload failed", exc_info=True)

            if entity_ref and selfie_bytes and selfie_filename:
                try:
                    minio_upload(
                        f"{entity_ref}/{Path(selfie_filename).name}",
                        selfie_bytes,
                        "application/octet-stream",
                    )
                except Exception:
                    log.warning("save_identity_check: selfie upload failed", exc_info=True)

            if entity_ref and pan_bytes and pan_filename:
                try:
                    minio_upload(
                        f"{entity_ref}/{Path(pan_filename).name}",
                        pan_bytes,
                        "application/octet-stream",
                    )
                except Exception:
                    log.warning("save_identity_check: pan document upload failed", exc_info=True)

            log.info("save_identity_check: saved %s check id=%s for entity=%s", check_type, row.id, entity_ref)
            return saved
    except Exception as exc:
        log.error("save_identity_check failed: %s", exc, exc_info=True)
        return None


def get_entity_identity_checks(entity_ref: str) -> List[Dict[str, Any]]:
    """Return all identity checks for an entity (most-recent first), without PDF bytes."""
    try:
        with db_session() as session:
            entity = (
                session.query(Entity)
                .filter(Entity.entity_ref == entity_ref)
                .first()
            )
            if not entity:
                return []
            checks = (
                session.query(IdentityCheck)
                .filter(IdentityCheck.entity_id == entity.id)
                .order_by(IdentityCheck.created_at.desc())
                .all()
            )
            latest_by_type: Dict[str, IdentityCheck] = {}
            for check in checks:
                if check.check_type not in latest_by_type:
                    latest_by_type[check.check_type] = check
            return [
                {
                    "id": c.id,
                    "check_type": c.check_type,
                    "status": c.status,
                    "cosine_similarity": c.cosine_similarity,
                    "display_score": c.display_score,
                    "is_match": c.is_match,
                    "liveness_state": c.liveness_state,
                    "liveness_passed": c.liveness_passed,
                    "verdict": c.verdict,
                    "doc_filename": c.doc_filename or "",
                    "selfie_filename": c.selfie_filename or "",
                    "report_json": c.report_json or {},
                    "has_pdf": c.pdf_report is not None and len(c.pdf_report) > 0,
                    "created_at": c.created_at.isoformat() if c.created_at else "",
                }
                for c in latest_by_type.values()
            ]
    except Exception as exc:
        log.warning("get_entity_identity_checks failed: %s", exc)
        return []


_DB_VIEWER_TABLES = {"entities", "scans", "document_extractions", "identity_checks", "entity_reports"}


def db_table_counts() -> Dict[str, int]:
    """Return row counts for all application tables."""
    _ALL_TABLES = (
        "entities",
        "scans",
        "document_extractions",
        "identity_checks",
        "entity_reports",
    )
    counts: Dict[str, int] = {}
    try:
        with db_session() as session:
            for tbl in _ALL_TABLES:
                counts[tbl] = session.execute(text(f"SELECT COUNT(*) FROM {tbl}")).scalar() or 0  # noqa: S608
    except Exception as exc:
        log.warning("db_table_counts failed: %s", exc)
        counts = {t: 0 for t in _ALL_TABLES}
    return counts


def db_table_rows(table: str, limit: int = 500) -> tuple[List[Dict[str, Any]], int]:
    """Return (rows_as_dicts, total_count) for any application table.

    Large binary columns (pdf_report) and huge JSONB blobs are excluded
    or truncated automatically for display.
    """
    if table not in _DB_VIEWER_TABLES:
        return [], 0
    try:
        with db_session() as session:
            total: int = session.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar() or 0  # noqa: S608
            if table == "scans":
                # Include layered_analysis_json so the Database Viewer can display
                # the full 11-layer forensic breakdown for each scan row.
                # The JSONB column can be large, but it is essential for analysis.
                rows_raw = session.execute(
                    text(
                        "SELECT id, entity_id, source_name, source_sha256, document_type, "
                        "layered_analysis_json, "
                        "approved, approved_by, approved_at, approval_comment, "
                        "first_level_approval, first_level_approved_by, "
                        "first_level_approved_at, first_level_approval_comment, "
                        "second_level_approval, second_level_approved_by, "
                        "second_level_approved_at, second_level_approval_comment, "
                        "generated_at, updated_at "
                        "FROM scans ORDER BY generated_at DESC LIMIT :lim"
                    ),
                    {"lim": limit},
                ).mappings().all()
            elif table == "document_extractions":
                rows_raw = session.execute(
                    text(
                            "SELECT id, entity_id, scan_id, file_name, document_type, source_screen, created_at, extracted_data "
                        "FROM document_extractions ORDER BY id DESC LIMIT :lim"
                    ),
                    {"lim": limit},
                ).mappings().all()
            elif table == "identity_checks":
                rows_raw = session.execute(
                    text(
                        "SELECT id, entity_id, check_type, status, cosine_similarity, "
                        "display_score, threshold, is_match, liveness_state, "
                        "liveness_passed, verdict, doc_filename, selfie_filename, "
                        "report_json, created_at "
                        "FROM identity_checks ORDER BY created_at DESC LIMIT :lim"
                    ),
                    {"lim": limit},
                ).mappings().all()
            elif table == "entity_reports":
                # Exclude report_json (can be a large JSONB blob) for fast display.
                rows_raw = session.execute(
                    text(
                        "SELECT id, entity_id, report_ref, report_minio_key, "
                        "first_level_approval, first_level_approved_by, "
                        "first_level_approved_at, first_level_approval_comment, "
                        "second_level_approval, second_level_approved_by, "
                        "second_level_approved_at, second_level_approval_comment, "
                        "generated_at, updated_at "
                        "FROM entity_reports ORDER BY generated_at DESC LIMIT :lim"
                    ),
                    {"lim": limit},
                ).mappings().all()
            else:
                rows_raw = session.execute(
                    text(f"SELECT * FROM {table} ORDER BY id DESC LIMIT :lim"),  # noqa: S608
                    {"lim": limit},
                ).mappings().all()
            return [dict(r) for r in rows_raw], total
    except Exception as exc:
        log.warning("db_table_rows failed for %s: %s", table, exc)
        return [], 0


def reset_db() -> bool:
    """Truncate all application tables and restart identity sequences.

    This is an irreversible operation — use only in dev / testing.
    """
    try:
        with db_session() as session:
            session.execute(
                text(
                    "TRUNCATE TABLE entity_reports, document_extractions, identity_checks, scans, entities "
                    "RESTART IDENTITY CASCADE"
                )
            )
        log.warning("reset_db: all tables truncated by user request")
        return True
    except Exception as exc:
        log.error("reset_db failed: %s", exc)
        return False


# Allowlist of tables that may be individually truncated via the Danger Zone UI.
# This prevents arbitrary table names from being injected into the SQL query.
_TRUNCATABLE_TABLES = frozenset({
    "entities",
    "scans",
    "document_extractions",
    "identity_checks",
    "entity_reports",
})


def truncate_table(table_name: str) -> bool:
    """Truncate a single table after verifying it is in the allowlist."""
    if table_name not in _TRUNCATABLE_TABLES:
        log.error("truncate_table: table %s is not in the allowlist", table_name)
        return False
    try:
        with db_session() as session:
            # We use CASCADE to drop any linked rows (e.g. document extractions linked to a scan)
            session.execute(text(f"TRUNCATE TABLE {table_name} CASCADE"))
        log.warning("truncate_table: table %s truncated by user request", table_name)
        return True
    except Exception as exc:
        log.error("truncate_table failed for %s: %s", table_name, exc)
        return False


_s3_client: Optional[Any] = None

def _get_minio_s3_client() -> Optional[Any]:
    global _s3_client
    if _s3_client is not None:
        return _s3_client
    endpoint = _os.environ.get("MINIO_ENDPOINT", "localhost:9000")
    access_key = _os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
    secret_key = _os.environ.get("MINIO_SECRET_KEY", "minioadmin")
    
    if not endpoint.startswith("http://") and not endpoint.startswith("https://"):
        endpoint = f"http://{endpoint}"
        
    if not (endpoint and access_key and secret_key):
        return None
    try:
        import boto3
        from botocore.config import Config
        _s3_client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=Config(signature_version="s3v4"),
        )
        return _s3_client
    except ImportError:
        return None
    except Exception as exc:
        log.warning("_get_minio_s3_client failed: %s", exc)
        return None


def minio_available() -> bool:
    """Return True if the MinIO service is reachable."""
    client = _get_minio_s3_client()
    if client is None:
        return False
    try:
        client.list_buckets()
        return True
    except Exception:
        return False


def minio_bucket_stats() -> Dict[str, Any]:
    """Return summary stats for the configured MinIO bucket."""
    bucket = _os.environ.get("MINIO_BUCKET", "basetruth-reports")
    client = _get_minio_s3_client()
    if client is None:
        return {"available": False, "bucket": bucket, "object_count": 0, "total_bytes": 0, "error": "MinIO not configured"}
    try:
        paginator = client.get_paginator("list_objects_v2")
        total_count = 0
        total_bytes = 0
        for page in paginator.paginate(Bucket=bucket):
            for obj in page.get("Contents", []):
                total_count += 1
                total_bytes += obj.get("Size", 0)
        return {
            "available": True,
            "bucket": bucket,
            "object_count": total_count,
            "total_bytes": total_bytes,
            "total_mb": round(total_bytes / 1024 / 1024, 2),
        }
    except Exception as exc:
        log.warning("minio_bucket_stats failed: %s", exc)
        return {"available": False, "bucket": bucket, "object_count": 0, "total_bytes": 0, "error": str(exc)}


def minio_list_objects(limit: int = 500) -> List[Dict[str, Any]]:
    """Return a list of objects in the configured bucket (most-recent first)."""
    bucket = _os.environ.get("MINIO_BUCKET", "basetruth-reports")
    client = _get_minio_s3_client()
    if client is None:
        return []
    try:
        paginator = client.get_paginator("list_objects_v2")
        objects: List[Dict[str, Any]] = []
        for page in paginator.paginate(Bucket=bucket):
            for obj in page.get("Contents", []):
                objects.append({
                    "key": obj["Key"],
                    "size_bytes": obj.get("Size", 0),
                    "size_kb": round(obj.get("Size", 0) / 1024, 1),
                    "last_modified": obj["LastModified"].isoformat() if obj.get("LastModified") else "",
                    "etag": obj.get("ETag", "").strip('"'),
                })
            if len(objects) >= limit:
                break
        # Sort newest first
        objects.sort(key=lambda o: o["last_modified"], reverse=True)
        return objects[:limit]
    except Exception as exc:
        log.warning("minio_list_objects failed: %s", exc)
        return []


def minio_delete_object(key: str) -> bool:
    """Delete a single object from the configured MinIO bucket.

    Returns True on success or if the object did not exist.
    Returns False on unexpected errors.
    """
    bucket = _os.environ.get("MINIO_BUCKET", "basetruth-reports")
    client = _get_minio_s3_client()
    if client is None:
        return False
    try:
        client.delete_object(Bucket=bucket, Key=key)
        return True
    except Exception as exc:
        log.warning("minio_delete_object(%s) failed: %s", key, exc)
        return False


def minio_list_entity_objects(entity_ref: str) -> List[Dict[str, Any]]:
    """List all objects stored under the *entity_ref/* prefix in MinIO.

    Returns a list of dicts with keys: ``key``, ``size_bytes``, ``size_kb``,
    ``last_modified``, ``filename`` (basename only).
    """
    bucket = _os.environ.get("MINIO_BUCKET", "basetruth-reports")
    client = _get_minio_s3_client()
    if client is None:
        return []
    prefix = f"{entity_ref}/"
    try:
        paginator = client.get_paginator("list_objects_v2")
        objects: List[Dict[str, Any]] = []
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                objects.append({
                    "key": obj["Key"],
                    "size_bytes": obj.get("Size", 0),
                    "size_kb": round(obj.get("Size", 0) / 1024, 1),
                    "last_modified": obj["LastModified"].isoformat() if obj.get("LastModified") else "",
                    "filename": obj["Key"].split("/")[-1],
                })
        objects.sort(key=lambda o: o["last_modified"], reverse=True)
        return objects
    except Exception as exc:
        log.warning("minio_list_entity_objects(%s) failed: %s", entity_ref, exc)
        return []


def minio_truncate_bucket() -> bool:
    """Delete all objects in the configured MinIO bucket.

    Returns True when the bucket is empty (either because objects were deleted
    or the bucket did not exist).  Returns False only on unexpected errors.
    """
    bucket = _os.environ.get("MINIO_BUCKET", "basetruth-reports")
    client = _get_minio_s3_client()
    if client is None:
        log.warning("minio_truncate_bucket: no S3 client — MinIO not configured")
        return False
    try:
        # Check bucket existence first; if missing, there is nothing to delete.
        try:
            client.head_bucket(Bucket=bucket)
        except Exception:
            log.info("minio_truncate_bucket: bucket '%s' does not exist — nothing to delete", bucket)
            return True
        paginator = client.get_paginator("list_objects_v2")
        deleted = 0
        for page in paginator.paginate(Bucket=bucket):
            objs = [{"Key": o["Key"]} for o in page.get("Contents", [])]
            if objs:
                client.delete_objects(Bucket=bucket, Delete={"Objects": objs})
                deleted += len(objs)
        log.warning("minio_truncate_bucket: deleted %d objects from bucket '%s'", deleted, bucket)
        return True
    except Exception as exc:
        log.error("minio_truncate_bucket failed: %s", exc)
        return False


def minio_upload(key: str, data: bytes, content_type: str = "application/octet-stream") -> bool:
    """Upload *data* to the configured MinIO bucket under *key*.

    Returns True on success.  Never raises — failures are logged and ignored
    so that the scan pipeline is not blocked by storage issues.
    """
    bucket = _os.environ.get("MINIO_BUCKET", "basetruth-reports")
    client = _get_minio_s3_client()
    if client is None:
        return False
    try:
        import io
        # Ensure the bucket exists (create if missing)
        try:
            client.head_bucket(Bucket=bucket)
        except Exception:  # bucket does not exist or different error
            try:
                client.create_bucket(Bucket=bucket)
                log.info("minio_upload: created bucket '%s'", bucket)
            except Exception:  # already exists on some MinIO versions
                pass
        client.put_object(
            Bucket=bucket,
            Key=key,
            Body=io.BytesIO(data),
            ContentLength=len(data),
            ContentType=content_type,
        )
        log.info("minio_upload: uploaded %d bytes → %s/%s", len(data), bucket, key)
        return True
    except Exception as exc:
        log.warning("minio_upload failed for key '%s': %s", key, exc)
        return False


def minio_get_object(key: str) -> Optional[bytes]:
    """Download *key* from the configured MinIO bucket and return its bytes.

    Returns None when the object does not exist or MinIO is unavailable.
    Never raises.
    """
    bucket = _os.environ.get("MINIO_BUCKET", "basetruth-reports")
    client = _get_minio_s3_client()
    if client is None:
        return None
    try:
        resp = client.get_object(Bucket=bucket, Key=key)
        return resp["Body"].read()
    except Exception as exc:
        log.debug("minio_get_object: key '%s' not found — %s", key, exc)
        return None


# ---------------------------------------------------------------------------
# MinIO docs bucket — stores platform reference files (DATABASE.md, prompts,
# etc.) so that Docker containers can always load the latest versions even
# when the docs/ directory is not mounted into the container filesystem.
# ---------------------------------------------------------------------------

_DOCS_BUCKET = "basetruth-docs"


def minio_docs_put(key: str, data: bytes, content_type: str = "text/plain") -> bool:
    """Upload *data* to the 'basetruth-docs' MinIO bucket under *key*.

    Used at startup to push DATABASE.md and other reference files so that
    Docker containers can load them via minio_docs_get() as a filesystem
    fallback. Creates the bucket automatically if it does not yet exist.

    Returns True on success. Never raises — failures are logged and ignored.
    """
    import io as _io
    client = _get_minio_s3_client()
    if client is None:
        return False
    try:
        # Ensure the docs bucket exists before uploading
        try:
            client.head_bucket(Bucket=_DOCS_BUCKET)
        except Exception:
            try:
                client.create_bucket(Bucket=_DOCS_BUCKET)
                log.info("minio_docs_put: created bucket '%s'", _DOCS_BUCKET)
            except Exception:
                pass  # bucket may already exist on some MinIO versions
        client.put_object(
            Bucket=_DOCS_BUCKET,
            Key=key,
            Body=_io.BytesIO(data),
            ContentLength=len(data),
            ContentType=content_type,
        )
        log.info("minio_docs_put: uploaded %d bytes → %s/%s", len(data), _DOCS_BUCKET, key)
        return True
    except Exception as exc:
        log.warning("minio_docs_put: upload failed for key '%s' — %s", key, exc)
        return False


def minio_docs_get(key: str) -> Optional[bytes]:
    """Download *key* from the 'basetruth-docs' MinIO bucket and return its bytes.

    Used as a fallback in _load_database_md() when the local docs/ directory
    is not available (e.g. inside a Docker container). Returns None when the
    object does not exist or MinIO is unreachable. Never raises.
    """
    client = _get_minio_s3_client()
    if client is None:
        log.warning("minio_docs_get: no S3 client — docs bucket unavailable")
        return None
    try:
        resp = client.get_object(Bucket=_DOCS_BUCKET, Key=key)
        data = resp["Body"].read()
        log.info("minio_docs_get: fetched %d bytes ← %s/%s", len(data), _DOCS_BUCKET, key)
        return data
    except Exception as exc:
        log.warning("minio_docs_get: failed to fetch %s/%s — %s", _DOCS_BUCKET, key, exc)
        return None


def minio_docs_bucket_stats() -> Dict[str, Any]:
    """Return object-count and size metrics for the MinIO docs bucket."""
    client = _get_minio_s3_client()
    if client is None:
        return {"available": False, "bucket": _DOCS_BUCKET, "object_count": 0, "total_mb": 0.0}
    try:
        objects = client.list_objects_v2(Bucket=_DOCS_BUCKET).get("Contents", [])
        total_bytes = sum(int(obj.get("Size", 0)) for obj in objects)
        return {
            "available": True,
            "bucket": _DOCS_BUCKET,
            "object_count": len(objects),
            "total_mb": round(total_bytes / (1024 * 1024), 3),
        }
    except Exception as exc:
        log.warning("minio_docs_bucket_stats failed: %s", exc)
        return {"available": False, "bucket": _DOCS_BUCKET, "object_count": 0, "total_mb": 0.0}


def minio_list_docs_objects(limit: int = 200) -> List[Dict[str, Any]]:
    """List objects from the MinIO docs bucket, newest first."""
    client = _get_minio_s3_client()
    if client is None:
        return []
    try:
        resp = client.list_objects_v2(Bucket=_DOCS_BUCKET)
        objects = resp.get("Contents", [])
        rows = [
            {
                "key": obj.get("Key", ""),
                "size_kb": round(int(obj.get("Size", 0)) / 1024, 2),
                "last_modified": obj.get("LastModified").isoformat() if obj.get("LastModified") else "",
            }
            for obj in objects
        ]
        rows.sort(key=lambda row: row.get("last_modified", ""), reverse=True)
        return rows[:limit]
    except Exception as exc:
        log.warning("minio_list_docs_objects failed: %s", exc)
        return []


def get_all_entities_with_scans(limit: int = 200) -> List[Dict[str, Any]]:
    """Return all entities (most-recent first) with their full scan summaries."""
    try:
        with db_session() as session:
            entities = (
                session.query(Entity)
                .order_by(Entity.id.desc())
                .limit(limit)
                .all()
            )
            result = []
            for e in entities:
                scans = (
                    session.query(Scan)
                    .filter(Scan.entity_id == e.id)
                    .order_by(Scan.generated_at.desc())
                    .all()
                )
                latest_scans: Dict[tuple[str, str], Scan] = {}
                for scan in scans:
                    key = (scan.source_name or "", scan.document_type or "generic")
                    if key not in latest_scans:
                        latest_scans[key] = scan
                result.append({
                    "entity_ref": e.entity_ref,
                    "name": f"{e.first_name or ''} {e.last_name or ''}".strip() or e.entity_ref,
                    "first_name": e.first_name or "",
                    "last_name": e.last_name or "",
                    "pan_number": e.pan_number or "",
                    "email": e.email or "",
                    "scans": [
                        {
                            "id": s.id,
                            "source_name": s.source_name,
                            "document_type": s.document_type or "generic",
                            "truth_score": (s.layered_analysis_json or {}).get("overall_score") if s.layered_analysis_json else None,
                            "risk_level": "low",
                            "verdict": (s.layered_analysis_json or {}).get("overall_verdict", ""),
                            "generated_at": s.generated_at.isoformat() if s.generated_at else "",
                            "has_pdf": False,
                        }
                        for s in latest_scans.values()
                    ],
                })
            return result
    except Exception as exc:
        log.warning("get_all_entities_with_scans failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# EntityReport — final cross-document verification reports
# ---------------------------------------------------------------------------

def _next_report_ref(session: Session) -> str:
    """Generate the next BTR-XXXXXX reference for an EntityReport.

    Uses the current max id from entity_reports so the reference is stable
    and predictable even across table truncations during development.
    """
    max_id: int = session.query(func.max(EntityReport.id)).scalar() or 0
    return f"BTR-{(max_id + 1):06d}"


def _entity_report_to_dict(r: EntityReport, entity: Optional[Entity]) -> Dict[str, Any]:
    """Convert an EntityReport ORM row to a plain serialisable dict."""
    first_name = (entity.first_name or "") if entity else ""
    last_name = (entity.last_name or "") if entity else ""
    return {
        "id": r.id,
        "report_ref": r.report_ref,
        "entity_id": r.entity_id,
        "entity_ref": (entity.entity_ref if entity else ""),
        "entity_name": f"{first_name} {last_name}".strip() or (entity.entity_ref if entity else ""),
        "report_json": r.report_json or {},
        "first_level_approval": r.first_level_approval,
        "first_level_approved_by": r.first_level_approved_by or "",
        "first_level_approved_at": (
            r.first_level_approved_at.isoformat() if r.first_level_approved_at else ""
        ),
        "first_level_approval_comment": r.first_level_approval_comment or "",
        "second_level_approval": r.second_level_approval,
        "second_level_approved_by": r.second_level_approved_by or "",
        "second_level_approved_at": (
            r.second_level_approved_at.isoformat() if r.second_level_approved_at else ""
        ),
        "second_level_approval_comment": r.second_level_approval_comment or "",
        "generated_at": r.generated_at.isoformat() if r.generated_at else "",
        # MinIO key for the PDF — empty string means PDF was not stored (MinIO unavailable).
        "report_minio_key": r.report_minio_key or "",
    }


def save_entity_report(entity_ref: str, report_json: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Create or refresh the final cross-document report for an entity.

    If the entity already has a pending (unapproved) report, it is replaced
    by the new analysis so there is always at most one pending report per entity.
    Approved reports are never overwritten — a new row is created instead so the
    audit trail is preserved.
    """
    try:
        with db_session() as session:
            entity = (
                session.query(Entity).filter(Entity.entity_ref == entity_ref).first()
            )
            if not entity:
                log.warning(
                    "save_entity_report: entity not found",
                    extra={"entity_ref": entity_ref},
                )
                return None

            # Look for an existing pending (level-1 null) report to refresh.
            existing = (
                session.query(EntityReport)
                .filter(
                    EntityReport.entity_id == entity.id,
                    EntityReport.first_level_approval == None,  # noqa: E711
                )
                .order_by(EntityReport.generated_at.desc())
                .first()
            )

            if existing:
                # Refresh in-place — re-running before approval resets the payload.
                existing.report_json = _json_ready(report_json)
                existing.generated_at = datetime.now(timezone.utc)
                existing.updated_at = datetime.now(timezone.utc)
                report_ref = existing.report_ref
                log.info(
                    "save_entity_report: refreshed existing pending report",
                    extra={"entity_ref": entity_ref, "report_ref": report_ref},
                )
            else:
                # Create a new report row (previous ones are approved/rejected).
                report_ref = _next_report_ref(session)
                session.add(EntityReport(
                    entity_id=entity.id,
                    report_ref=report_ref,
                    report_json=_json_ready(report_json),
                ))
                log.info(
                    "save_entity_report: created new report",
                    extra={"entity_ref": entity_ref, "report_ref": report_ref},
                )

            session.flush()

            # ── Generate PDF and upload to MinIO ──────────────────────────────
            # We render the PDF after the DB row is flushed so we have the
            # report_ref available for the header.  The MinIO key is then written
            # back to the row and committed together with the rest of the data.
            try:
                from basetruth.reporting.pdf import render_entity_report_pdf  # lazy import avoids circular deps

                # Fetch the candidate photo bytes from MinIO if a key was stored
                # in the report_json by build_final_report_json().  The photo is
                # embedded in the PDF but NOT stored separately — we just look it
                # up at render time from the existing MinIO object.
                photo_minio_key = report_json.get("photo_minio_key") or ""
                photo_bytes: bytes | None = None
                if photo_minio_key:
                    try:
                        photo_bytes = minio_get_object(photo_minio_key)
                        log.debug(
                            "save_entity_report: fetched candidate photo",
                            extra={"entity_ref": entity_ref, "key": photo_minio_key, "size": len(photo_bytes) if photo_bytes else 0},
                        )
                    except Exception as photo_exc:
                        log.warning(
                            "save_entity_report: could not fetch photo from MinIO (%s) — PDF will have no photo",
                            photo_exc,
                            extra={"entity_ref": entity_ref, "key": photo_minio_key},
                        )

                pdf_bytes   = render_entity_report_pdf(report_json, report_ref, photo_bytes=photo_bytes)
                minio_key   = f"BTR-reports/{entity_ref}/{report_ref}.pdf"
                upload_ok   = minio_upload(minio_key, pdf_bytes, "application/pdf")

                if upload_ok:
                    # Attach the key to the ORM row so it is persisted on commit.
                    target = existing if existing else session.query(EntityReport).filter(
                        EntityReport.report_ref == report_ref
                    ).first()
                    if target:
                        target.report_minio_key = minio_key
                    log.info(
                        "save_entity_report: PDF uploaded to MinIO",
                        extra={"entity_ref": entity_ref, "report_ref": report_ref, "minio_key": minio_key},
                    )
                else:
                    log.warning(
                        "save_entity_report: MinIO upload failed — PDF not stored",
                        extra={"entity_ref": entity_ref, "report_ref": report_ref},
                    )
            except Exception as pdf_exc:
                # PDF generation/upload is best-effort; do not fail the whole save.
                log.warning(
                    "save_entity_report: PDF render/upload error — %s",
                    pdf_exc,
                    extra={"entity_ref": entity_ref, "report_ref": report_ref},
                )

            return {"entity_ref": entity_ref, "report_ref": report_ref}
    except Exception as exc:
        log.error("save_entity_report failed: %s", exc, exc_info=True)
        return None


def get_entity_reports(entity_ref: str) -> List[Dict[str, Any]]:
    """Return all EntityReport rows for an entity, most-recent first."""
    try:
        with db_session() as session:
            entity = (
                session.query(Entity).filter(Entity.entity_ref == entity_ref).first()
            )
            if not entity:
                return []
            rows = (
                session.query(EntityReport)
                .filter(EntityReport.entity_id == entity.id)
                .order_by(EntityReport.generated_at.desc())
                .all()
            )
            return [_entity_report_to_dict(r, entity) for r in rows]
    except Exception as exc:
        log.warning("get_entity_reports failed: %s", exc)
        return []


def list_all_entity_reports(limit: int = 500) -> List[Dict[str, Any]]:
    """Return all EntityReport rows across all entities, most-recent first.

    Used by the Cases screen to show entity reports awaiting approval.
    """
    try:
        with db_session() as session:
            rows = (
                session.query(EntityReport)
                .order_by(EntityReport.generated_at.desc())
                .limit(limit)
                .all()
            )
            result = []
            for r in rows:
                entity = (
                    session.query(Entity).filter(Entity.id == r.entity_id).first()
                )
                result.append(_entity_report_to_dict(r, entity))
            return result
    except Exception as exc:
        log.warning("list_all_entity_reports failed: %s", exc)
        return []


def _set_entity_report_approval(
    report_ref: str,
    *,
    level: int,
    approval: str,
    approved_by: str,
    comment: str,
) -> Optional[Dict[str, Any]]:
    """Internal: set 1st-level or 2nd-level approval on an EntityReport.

    Level 1 can be set freely.  Level 2 is only allowed after level 1 is Y.
    """
    try:
        with db_session() as session:
            r = (
                session.query(EntityReport)
                .filter(EntityReport.report_ref == report_ref)
                .first()
            )
            if r is None:
                log.warning(
                    "_set_entity_report_approval: report not found",
                    extra={"report_ref": report_ref},
                )
                return None
            now = datetime.now(timezone.utc)
            if level == 1:
                r.first_level_approval = approval
                r.first_level_approved_by = approved_by or None
                r.first_level_approved_at = now
                r.first_level_approval_comment = comment or None
            else:
                if r.first_level_approval != "Y":
                    log.warning(
                        "_set_entity_report_approval: 1st-level not yet approved",
                        extra={"report_ref": report_ref},
                    )
                    return None
                r.second_level_approval = approval
                r.second_level_approved_by = approved_by or None
                r.second_level_approved_at = now
                r.second_level_approval_comment = comment or None
            r.updated_at = now
            session.flush()
            entity = session.query(Entity).filter(Entity.id == r.entity_id).first()
            log.info(
                "entity_report approval updated",
                extra={"report_ref": report_ref, "level": level, "approval": approval},
            )
            return _entity_report_to_dict(r, entity)
    except Exception as exc:
        log.error("_set_entity_report_approval failed: %s", exc, exc_info=True)
        return None


def first_level_approve_entity_report(
    report_ref: str, approved_by: str = "", comment: str = ""
) -> Optional[Dict[str, Any]]:
    """Set 1st-level approval = Y on the given EntityReport."""
    return _set_entity_report_approval(
        report_ref, level=1, approval="Y", approved_by=approved_by, comment=comment
    )


def first_level_reject_entity_report(
    report_ref: str, approved_by: str = "", comment: str = ""
) -> Optional[Dict[str, Any]]:
    """Set 1st-level approval = N on the given EntityReport."""
    return _set_entity_report_approval(
        report_ref, level=1, approval="N", approved_by=approved_by, comment=comment
    )


def second_level_approve_entity_report(
    report_ref: str, approved_by: str = "", comment: str = ""
) -> Optional[Dict[str, Any]]:
    """Set 2nd-level approval = Y on the given EntityReport (1st-level must be Y first)."""
    return _set_entity_report_approval(
        report_ref, level=2, approval="Y", approved_by=approved_by, comment=comment
    )


def second_level_reject_entity_report(
    report_ref: str, approved_by: str = "", comment: str = ""
) -> Optional[Dict[str, Any]]:
    """Set 2nd-level approval = N on the given EntityReport."""
    return _set_entity_report_approval(
        report_ref, level=2, approval="N", approved_by=approved_by, comment=comment
    )
