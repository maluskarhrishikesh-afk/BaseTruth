"""Database layer — SQLAlchemy ORM + PostgreSQL.

Tables
------
  entities   — one row per person/organisation being verified (searchable by
               first_name, last_name, email, phone, PAN, Aadhaar, entity_ref).
  scans      — one row per document scan; stores the full JSON report + optional
               PDF report bytes; linked to an entity.
  cases      — case management records (replaces case_records.json).
  case_notes — timestamped analyst notes attached to a case.

All public functions degrade gracefully (return None / empty list) when the
database is unavailable so the file-based fallback still works.
"""
from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Generator, Optional

from basetruth.logger import get_logger

log = get_logger(__name__)

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Text,
    create_engine,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker
from sqlalchemy.sql import func

DATABASE_URL: str = os.environ.get("DATABASE_URL", "")

_engine = None
_SessionLocal = None


# ---------------------------------------------------------------------------
# Engine / session helpers
# ---------------------------------------------------------------------------


def get_engine():
    global _engine
    if _engine is None and DATABASE_URL:
        try:
            _engine = create_engine(
                DATABASE_URL,
                pool_pre_ping=True,
                pool_size=5,
                max_overflow=10,
                connect_args={"connect_timeout": 5},
            )
            log.info("DB engine created", extra={"url": DATABASE_URL.split("@")[-1]})
        except Exception as exc:
            log.warning("Could not create DB engine: %s", exc)
    return _engine


def db_available() -> bool:
    engine = get_engine()
    if engine is None:
        log.debug("db_available: no engine (DATABASE_URL not set?)")
        return False
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        log.warning("db_available: connection check failed", extra={"error": str(exc)})
        return False


@contextmanager
def db_session() -> Generator[Session, None, None]:
    engine = get_engine()
    if engine is None:
        raise RuntimeError("DATABASE_URL not configured or DB unreachable")
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    session: Session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ---------------------------------------------------------------------------
# ORM models
# ---------------------------------------------------------------------------


class Base(DeclarativeBase):
    pass


class Entity(Base):
    """One record per person / organisation being verified."""

    __tablename__ = "entities"

    id = Column(Integer, primary_key=True, autoincrement=True)
    entity_ref = Column(String(20), unique=True, nullable=False)  # BT-000001
    first_name = Column(String(255), default="")
    last_name = Column(String(255), default="")
    email = Column(String(255), default="")
    phone = Column(String(50), default="")
    pan_number = Column(String(20), default="")
    aadhar_number = Column(String(20), default="")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    scans = relationship("Scan", back_populates="entity", cascade="all, delete-orphan")
    cases = relationship("Case", back_populates="entity")
    extracted_info = relationship("DocumentInformation", back_populates="entity", cascade="all, delete-orphan")
    identity_checks = relationship("IdentityCheck", back_populates="entity", cascade="all, delete-orphan")


class Scan(Base):
    """One row per document scan."""

    __tablename__ = "scans"

    id = Column(Integer, primary_key=True, autoincrement=True)
    entity_id = Column(
        Integer, ForeignKey("entities.id", ondelete="SET NULL"), nullable=True
    )
    source_name = Column(String(500), nullable=False)
    source_sha256 = Column(String(64), default="")
    document_type = Column(String(100), default="generic")
    truth_score = Column(Integer, nullable=True)
    risk_level = Column(String(20), default="low")
    verdict = Column(Text, default="")
    parse_method = Column(String(100), default="")
    report_json = Column(JSONB, nullable=False)
    pdf_report = Column(LargeBinary, nullable=True)
    generated_at = Column(DateTime(timezone=True), server_default=func.now())

    entity = relationship("Entity", back_populates="scans")
    extracted_info = relationship("DocumentInformation", back_populates="scan", cascade="all, delete-orphan")


class Case(Base):
    """Case management record (mirrors CaseRecord dataclass)."""

    __tablename__ = "cases"

    id = Column(Integer, primary_key=True, autoincrement=True)
    case_key = Column(String(500), unique=True, nullable=False)
    entity_id = Column(
        Integer, ForeignKey("entities.id", ondelete="SET NULL"), nullable=True
    )
    document_type = Column(String(100), default="")
    status = Column(String(50), default="new")
    disposition = Column(String(50), default="open")
    priority = Column(String(20), default="normal")
    assignee = Column(String(255), default="")
    labels = Column(ARRAY(Text), default=list)
    max_risk_level = Column(String(20), default="low")
    document_count = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    entity = relationship("Entity", back_populates="cases")
    notes = relationship(
        "CaseNote",
        back_populates="case",
        cascade="all, delete-orphan",
        order_by="CaseNote.created_at",
    )


class CaseNote(Base):
    """Analyst note attached to a Case."""

    __tablename__ = "case_notes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    case_id = Column(Integer, ForeignKey("cases.id", ondelete="CASCADE"))
    author = Column(String(255), default="analyst")
    text = Column(Text, default="")
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    case = relationship("Case", back_populates="notes")


class DocumentInformation(Base):
    """Rich extracted metadata parsed from document scans."""

    __tablename__ = "document_information"

    id = Column(Integer, primary_key=True, autoincrement=True)
    entity_id = Column(Integer, ForeignKey("entities.id", ondelete="CASCADE"), nullable=False)
    scan_id = Column(Integer, ForeignKey("scans.id", ondelete="CASCADE"), nullable=False)
    document_type = Column(String(100), default="generic")
    extracted_data = Column(JSONB, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    entity = relationship("Entity", back_populates="extracted_info")
    scan = relationship("Scan", back_populates="extracted_info")

class IdentityCheck(Base):
    """One row per face-match or Video KYC verification event.

    Stores the full result payload so analysts can review identity
    verification history alongside document scans.
    """

    __tablename__ = "identity_checks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    entity_id = Column(
        Integer, ForeignKey("entities.id", ondelete="SET NULL"), nullable=True
    )
    check_type = Column(String(30), nullable=False)       # 'face_match' | 'video_kyc'
    status = Column(String(20), nullable=False)            # 'pass' | 'fail' | 'inconclusive'

    # Face match fields
    cosine_similarity = Column(Float, nullable=True)
    display_score = Column(Float, nullable=True)           # 0-100 percentage
    threshold = Column(Float, nullable=True)
    is_match = Column(Boolean, nullable=True)

    # Video KYC liveness fields
    liveness_state = Column(String(30), nullable=True)     # 'Center' | 'Turned Left' | 'Turned Right'
    liveness_passed = Column(Boolean, nullable=True)

    # Overall verdict
    verdict = Column(String(20), default="")               # 'PASS' | 'FAIL'

    # Audit trail
    doc_filename = Column(String(500), default="")         # original ID document filename
    selfie_filename = Column(String(500), default="")      # selfie filename (face_match only)
    report_json = Column(JSONB, nullable=False)            # full result payload
    pdf_report = Column(LargeBinary, nullable=True)        # generated PDF report

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    entity = relationship("Entity", back_populates="identity_checks")


# ---------------------------------------------------------------------------
# Schema initialisation
# ---------------------------------------------------------------------------


def init_db() -> bool:
    """Create all tables if they do not already exist.

    Returns True on success, False when the DB is unavailable.
    """
    engine = get_engine()
    if engine is None:
        log.warning("init_db: no DATABASE_URL — skipping schema creation")
        return False
    try:
        Base.metadata.create_all(engine)
        log.info("DB schema ready")
        return True
    except Exception as exc:
        log.warning("init_db failed: %s", exc)
        return False
