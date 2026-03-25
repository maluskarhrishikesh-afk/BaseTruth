"""Entity / scan / case store — high-level CRUD over the SQLAlchemy ORM.

All public functions return Python dicts/lists and degrade gracefully
(returning None or []) when the database is unavailable, so the rest of
the application continues to work in file-only mode.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from basetruth.db import Case, CaseNote, Entity, Scan, db_session

log = logging.getLogger(__name__)

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
    report: Dict[str, Any], pdf_bytes: Optional[bytes] = None
) -> Optional[Dict[str, Any]]:
    """Persist a completed verification report (+ optional PDF) to the database.

    Extracts identity fields from the report, finds or creates an entity, then
    writes a Scan row.  Returns a dict with ``scan_id`` and ``entity_ref``, or
    None if the DB is unavailable / the save fails.
    """
    try:
        with db_session() as session:
            identity = extract_identity_fields(report)
            entity = _find_or_create_entity(session, identity)

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

            return {
                "scan_id": scan.id,
                "entity_ref": entity.entity_ref if entity else None,
            }
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
