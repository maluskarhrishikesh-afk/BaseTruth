"""Entity / scan / case store — high-level CRUD over the SQLAlchemy ORM.

All public functions return Python dicts/lists and degrade gracefully
(returning None or []) when the database is unavailable, so the rest of
the application continues to work in file-only mode.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy import func, or_, text
from sqlalchemy.orm import Session

from basetruth.db import Case, CaseNote, Entity, Scan, db_session
from basetruth.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _next_entity_ref(session: Session) -> str:
    """Generate the next BT-XXXXXX reference string."""
    max_id: int = session.query(func.max(Entity.id)).scalar() or 0
    return f"BT-{(max_id + 1):06d}"


def _clean(value: Any) -> str:
    """Return a non-None, stripped string."""
    return str(value).strip() if value else ""


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
    """Look up an existing entity by PAN / Aadhaar number, or create a new one."""
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
        # Enrich empty fields from the new scan
        for field_name in ("first_name", "last_name", "email", "phone", "pan_number", "aadhar_number"):
            val = identity.get(field_name, "")
            if val and not getattr(entity, field_name):
                setattr(entity, field_name, val)
        entity.updated_at = func.now()  # type: ignore[assignment]
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
        "latest_risk": latest_scan.risk_level if latest_scan else "",
        "latest_score": latest_scan.truth_score if latest_scan else None,
        "created_at": (
            entity.created_at.isoformat() if entity.created_at else ""
        ),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def save_scan_to_db(
    report: Dict[str, Any],
    pdf_bytes: Optional[bytes] = None,
    forced_entity_ref: Optional[str] = None,
    extra_identity: Optional[Dict[str, str]] = None,
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
                        "save_scan_to_db: force-linked to entity",
                        extra={"entity_ref": forced_entity_ref},
                    )
                else:
                    log.warning(
                        "save_scan_to_db: forced_entity_ref not found, falling back to auto",
                        extra={"entity_ref": forced_entity_ref},
                    )

            # ── Auto-detect / create entity ────────────────────────────────
            if entity is None:
                identity = extract_identity_fields(report)
                # Operator-supplied fields always WIN over document-extracted fields.
                # This ensures that when the user types PAN / email / name on the
                # scan form, those details are used as the authoritative identity
                # even if the document itself contains different values.
                if extra_identity:
                    for k, v in extra_identity.items():
                        if v:
                            identity[k] = v
                entity = _find_or_create_entity(session, identity)

            # When the operator explicitly supplied identity fields, force-update
            # the entity record — even if those fields were already populated from
            # an earlier (possibly incorrect) scan.  This lets analysts correct
            # a wrong name or PAN without having to delete and re-create the entity.
            if extra_identity and entity:
                for k, v in extra_identity.items():
                    if v:
                        setattr(entity, k, v)
                entity.updated_at = func.now()  # type: ignore[assignment]

            ss = report.get("structured_summary", {})
            tamper = report.get("tamper_assessment", {})

            # Load PDF from artifact path if bytes were not supplied
            if pdf_bytes is None:
                pdf_path_str = report.get("artifacts", {}).get("pdf_report_path", "")
                if pdf_path_str and Path(pdf_path_str).exists():
                    try:
                        pdf_bytes = Path(pdf_path_str).read_bytes()
                    except OSError:
                        pass

            scan = Scan(
                entity_id=entity.id if entity else None,
                source_name=report.get("source", {}).get("name", ""),
                source_sha256=report.get("source", {}).get("sha256", ""),
                document_type=ss.get("document", {}).get("type", "generic"),
                truth_score=tamper.get("truth_score"),
                risk_level=tamper.get("risk_level", "low"),
                verdict=tamper.get("verdict", ""),
                parse_method=ss.get("parse_method", ""),
                report_json=report,
                pdf_report=pdf_bytes,
            )
            session.add(scan)
            session.flush()

            result = {
                "scan_id": scan.id,
                "entity_ref": entity.entity_ref if entity else None,
            }

        # ── Upload PDF to MinIO (non-fatal) ────────────────────────────────
        entity_ref_for_minio = result.get("entity_ref")
        if entity_ref_for_minio and pdf_bytes:
            source_name = report.get("source", {}).get("name", "unknown")
            stem = Path(source_name).stem
            minio_key = f"{entity_ref_for_minio}/{stem}_report.pdf"
            minio_upload(minio_key, pdf_bytes, "application/pdf")

        return result
    except Exception as exc:
        log.warning("save_scan_to_db failed: %s", exc)
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
            return [
                {
                    "id": s.id,
                    "source_name": s.source_name,
                    "document_type": s.document_type or "generic",
                    "truth_score": s.truth_score,
                    "risk_level": s.risk_level or "low",
                    "verdict": s.verdict or "",
                    "parse_method": s.parse_method or "",
                    "generated_at": (
                        s.generated_at.isoformat() if s.generated_at else ""
                    ),
                    "has_pdf": s.pdf_report is not None and len(s.pdf_report) > 0,
                    "report_json": s.report_json or {},
                }
                for s in scans
            ]
    except Exception as exc:
        log.warning("get_entity_scans failed: %s", exc)
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
                "high_risk": (
                    session.query(func.count(Scan.id))
                    .filter(Scan.risk_level == "high")
                    .scalar()
                    or 0
                ),
            }
    except Exception as exc:
        log.warning("db_stats failed: %s", exc)
        return {"entities": 0, "scans": 0, "high_risk": 0}


def db_dashboard_stats() -> Dict[str, Any]:
    """Extended stats for the Dashboard — single round-trip query."""
    from basetruth.db import Case  # local import to avoid circular deps at module level
    try:
        with db_session() as session:
            total_scans = session.query(func.count(Scan.id)).scalar() or 0
            high_risk = (
                session.query(func.count(Scan.id)).filter(Scan.risk_level == "high").scalar() or 0
            )
            medium_risk = (
                session.query(func.count(Scan.id)).filter(Scan.risk_level == "medium").scalar() or 0
            )
            low_risk = (
                session.query(func.count(Scan.id)).filter(Scan.risk_level == "low").scalar() or 0
            )
            entities = session.query(func.count(Entity.id)).scalar() or 0
            avg_score_row = session.query(func.avg(Scan.truth_score)).scalar()
            avg_score = round(float(avg_score_row), 1) if avg_score_row is not None else None
            pending = (
                session.query(func.count(Case.id)).filter(Case.disposition == "open").scalar() or 0
            )
            cleared = (
                session.query(func.count(Case.id)).filter(Case.disposition == "cleared").scalar() or 0
            )
            fraud = (
                session.query(func.count(Case.id)).filter(Case.disposition == "fraud_confirmed").scalar() or 0
            )
            total_cases = session.query(func.count(Case.id)).scalar() or 0
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
                "pending_review": pending,
                "auto_approved": cleared,
                "rejected": fraud,
                "total_cases": total_cases,
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


_DB_VIEWER_TABLES = {"entities", "scans", "cases", "case_notes"}


def db_table_counts() -> Dict[str, int]:
    """Return row counts for all four application tables."""
    counts: Dict[str, int] = {}
    try:
        with db_session() as session:
            for tbl in ("entities", "scans", "cases", "case_notes"):
                counts[tbl] = session.execute(text(f"SELECT COUNT(*) FROM {tbl}")).scalar() or 0  # noqa: S608
    except Exception as exc:
        log.warning("db_table_counts failed: %s", exc)
        counts = {t: 0 for t in ("entities", "scans", "cases", "case_notes")}
    return counts


def db_table_rows(table: str, limit: int = 500) -> tuple[List[Dict[str, Any]], int]:
    """Return (rows_as_dicts, total_count) for one of the four app tables.

    Large binary columns (pdf_report) are excluded automatically.
    """
    if table not in _DB_VIEWER_TABLES:
        return [], 0
    try:
        with db_session() as session:
            total: int = session.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar() or 0  # noqa: S608
            if table == "scans":
                # Exclude pdf_report (large binary) from display
                rows_raw = session.execute(
                    text(
                        "SELECT id, entity_id, source_name, source_sha256, document_type, "
                        "truth_score, risk_level, verdict, parse_method, generated_at "
                        "FROM scans ORDER BY generated_at DESC LIMIT :lim"
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
                    "TRUNCATE TABLE case_notes, cases, scans, entities "
                    "RESTART IDENTITY CASCADE"
                )
            )
        log.warning("reset_db: all tables truncated by user request")
        return True
    except Exception as exc:
        log.error("reset_db failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# MinIO / S3 object-storage helpers
# ---------------------------------------------------------------------------

import os as _os


def _get_minio_s3_client():
    """Return a boto3 S3 client pointed at the MinIO endpoint, or None."""
    try:
        import boto3  # type: ignore
        from botocore.config import Config  # type: ignore
        endpoint = _os.environ.get("MINIO_ENDPOINT", "")
        if not endpoint:
            return None
        # Ensure the scheme is present
        if not endpoint.startswith("http"):
            endpoint = f"http://{endpoint}"
        client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=_os.environ.get("MINIO_ACCESS_KEY", "minioadmin"),
            aws_secret_access_key=_os.environ.get("MINIO_SECRET_KEY", "minioadmin"),
            config=Config(signature_version="s3v4"),
            region_name="us-east-1",
        )
        return client
    except Exception as exc:
        log.debug("_get_minio_s3_client: unavailable — %s", exc)
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
# DB-driven case management
# ---------------------------------------------------------------------------


def update_case_in_db(
    case_key: str,
    *,
    entity_ref: Optional[str] = None,
    document_type: Optional[str] = None,
    status: Optional[str] = None,
    disposition: Optional[str] = None,
    priority: Optional[str] = None,
    assignee: Optional[str] = None,
    labels: Optional[List[str]] = None,
    max_risk_level: Optional[str] = None,
    note_text: str = "",
    note_author: str = "system",
) -> Optional[Dict[str, Any]]:
    """Upsert a case record in the DB cases table.

    Returns {case_id, case_key} on success, None on failure.
    """
    try:
        with db_session() as session:
            case = session.query(Case).filter(Case.case_key == case_key).first()
            if case is None:
                # Try to derive entity_id and doc_type from the case_key
                # Expected format: ``doc_type::entity_ref`` (e.g. payslip::BT-000001)
                eid: Optional[int] = None
                dtype = document_type or ""
                if entity_ref:
                    ent = session.query(Entity).filter(Entity.entity_ref == entity_ref).first()
                    eid = ent.id if ent else None
                elif "::" in case_key:
                    parts = case_key.rsplit("::", 1)
                    dtype = dtype or parts[0]
                    potential_ref = parts[1]
                    if potential_ref.startswith("BT-"):
                        ent = session.query(Entity).filter(Entity.entity_ref == potential_ref).first()
                        eid = ent.id if ent else None
                case = Case(
                    case_key=case_key,
                    entity_id=eid,
                    document_type=dtype,
                    status=status or "new",
                    disposition=disposition or "open",
                    priority=priority or "normal",
                    assignee=assignee or "",
                    labels=labels or [],
                    max_risk_level=max_risk_level or "low",
                )
                session.add(case)
                session.flush()
            else:
                if status is not None:
                    case.status = status
                if disposition is not None:
                    case.disposition = disposition
                if priority is not None:
                    case.priority = priority
                if assignee is not None:
                    case.assignee = assignee
                if labels is not None:
                    case.labels = labels
                if max_risk_level is not None:
                    case.max_risk_level = max_risk_level

            if note_text.strip():
                note = CaseNote(  # type: ignore[call-arg]  # ORM model, not dataclass
                    case_id=case.id,
                    author=note_author or "system",
                    text=note_text.strip(),
                )
                session.add(note)

            return {"case_id": case.id, "case_key": case_key}
    except Exception as exc:
        log.warning("update_case_in_db failed for '%s': %s", case_key, exc)
        return None


def list_cases_from_db() -> List[Dict[str, Any]]:
    """Build the case list from DB scans grouped by (entity, document_type).

    Case state (status, disposition, priority, notes) is loaded from the DB
    cases table.  This replaces the file-based list_cases() so counts are
    always accurate and are automatically cleared when the DB is reset.
    """
    try:
        with db_session() as session:
            # Fetch all scans with entity info in a single query
            scans = session.query(Scan).order_by(Scan.generated_at.desc()).all()

            entity_cache: Dict[int, Optional[Entity]] = {}

            # Group scans by (entity_ref, document_type) → case_key
            groups: Dict[str, Dict] = {}
            for s in scans:
                if s.entity_id:
                    if s.entity_id not in entity_cache:
                        entity_cache[s.entity_id] = (
                            session.query(Entity).filter(Entity.id == s.entity_id).first()
                        )
                    entity = entity_cache[s.entity_id]
                else:
                    entity = None

                entity_ref = entity.entity_ref if entity else "unlinked"
                doc_type = s.document_type or "generic"
                case_key = f"{doc_type}::{entity_ref}"

                if case_key not in groups:
                    ename = (
                        f"{entity.first_name or ''} {entity.last_name or ''}".strip()
                        if entity
                        else ""
                    )
                    groups[case_key] = {
                        "case_key": case_key,
                        "entity_ref": entity_ref,
                        "entity_name": ename,
                        "document_type": doc_type,
                        "documents": [],
                        "max_risk_level": "low",
                        "min_truth_score": 100,
                    }

                groups[case_key]["documents"].append(
                    {
                        "source_name": s.source_name,
                        "document_type": doc_type,
                        "truth_score": s.truth_score,
                        "risk_level": s.risk_level or "low",
                        "verdict": s.verdict or "",
                        "generated_at": s.generated_at.isoformat() if s.generated_at else "",
                        "scan_id": s.id,
                    }
                )
                r = s.risk_level or "low"
                cur = groups[case_key]["max_risk_level"]
                if r == "high" or (r == "medium" and cur == "low"):
                    groups[case_key]["max_risk_level"] = r
                if s.truth_score is not None:
                    groups[case_key]["min_truth_score"] = min(
                        groups[case_key]["min_truth_score"], s.truth_score
                    )

            # Load case state from DB cases table
            db_states: Dict[str, Case] = {
                c.case_key: c for c in session.query(Case).all()
            }

            result = []
            for case_key, group in groups.items():
                state = db_states.get(case_key)
                group["document_count"] = len(group["documents"])
                group["status"] = state.status if state else "new"
                group["disposition"] = state.disposition if state else "open"
                group["priority"] = state.priority if state else "normal"
                group["assignee"] = state.assignee if state else ""
                group["labels"] = list(state.labels) if state and state.labels else []
                group["notes"] = []
                group["note_count"] = 0
                if state:
                    db_notes = (
                        session.query(CaseNote)  # type: ignore[attr-defined]
                        .filter(CaseNote.case_id == state.id)  # type: ignore[attr-defined]
                        .order_by(CaseNote.created_at.asc())  # type: ignore[attr-defined]
                        .all()
                    )
                    group["notes"] = [
                        {
                            "created_at": n.created_at.isoformat() if n.created_at else "",
                            "author": n.author,
                            "text": n.text,
                        }
                        for n in db_notes
                    ]
                    group["note_count"] = len(group["notes"])
                group["needs_review"] = (
                    group["max_risk_level"] in ("high", "medium")
                    and group["disposition"] not in ("cleared", "fraud_confirmed")
                )
                if group["min_truth_score"] == 100 and not group["documents"]:
                    group["min_truth_score"] = None
                result.append(group)

            return sorted(
                result,
                key=lambda c: (
                    0 if c["needs_review"] else 1,
                    {"high": 0, "medium": 1, "low": 2}.get(c["max_risk_level"], 3),
                    -c["document_count"],
                ),
            )
    except Exception as exc:
        log.warning("list_cases_from_db failed: %s", exc)
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
                            "truth_score": s.truth_score,
                            "risk_level": s.risk_level or "low",
                            "verdict": s.verdict or "",
                            "generated_at": s.generated_at.isoformat() if s.generated_at else "",
                            "has_pdf": bool(s.pdf_report),
                        }
                        for s in scans
                    ],
                })
            return result
    except Exception as exc:
        log.warning("get_all_entities_with_scans failed: %s", exc)
        return []
