"""Database layer — SQLAlchemy ORM + PostgreSQL.

Tables
------
  entities              — one row per person/organisation being verified.
  scans                 — one row per document scan; stores forensic JSON + approval status.
  document_extractions  — structured fields extracted from scanned documents.
  identity_checks       — Identity Verification (face-match) results only.
  video_kyc_checks      — Video KYC session results (separate workflow from face-match).
  entity_reports        — final cross-document verification reports.

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
    String,
    Text,
    UniqueConstraint,
    create_engine,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker
from sqlalchemy.sql import func


def _resolve_database_url() -> str:
    """Return the PostgreSQL URL for the current runtime.

    Resolution order is intentionally simple:
    1. Respect an explicit DATABASE_URL from the environment.
    2. For local Windows/macOS/Linux runs outside Docker, fall back to the
       Docker Compose PostgreSQL service exposed on localhost:5432.

    This keeps local `streamlit run ...` sessions working even when the user
    starts PostgreSQL via Docker Compose but forgets to export DATABASE_URL in
    the shell first. Inside Docker, the container already receives DATABASE_URL,
    so this fallback is never used.
    """
    env_url = os.environ.get("DATABASE_URL", "").strip()
    if env_url:
        return env_url
    return "postgresql://basetruth:basetruth_secret@localhost:5432/basetruth"


DATABASE_URL: str = _resolve_database_url()

_engine = None
_SessionLocal = None


# ---------------------------------------------------------------------------
# Engine / session helpers
# ---------------------------------------------------------------------------


def get_engine():
    """Return the shared SQLAlchemy engine, creating it on first call.

    Uses a connection pool of 5 (with up to 10 overflow connections) so
    multiple Streamlit user sessions can share the same pool instead of each
    opening their own connection.  pool_pre_ping=True silently drops dead
    connections before reusing them (important after DB container restarts).
    Returns None if DATABASE_URL is not set — callers must handle that case.
    """
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
    """Quick connectivity test — returns True only when PostgreSQL answers.

    Runs a trivial SELECT 1 query.  Called from the UI to decide whether
    to show the database status badge.  Deliberately kept cheap (one query,
    5-second timeout) because it runs on every page render.
    """
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
    """Yield a database session then commit, or rollback on any exception.

    Usage:
        with db_session() as session:
            session.query(...)

    The session is committed automatically when the 'with' block exits cleanly.
    If any exception is raised inside the block, all changes are rolled back and
    the exception is re-raised so callers can log it as they see fit.
    """
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
    extracted_info = relationship("DocumentExtraction", back_populates="entity", cascade="all, delete-orphan")
    identity_checks = relationship("IdentityCheck", back_populates="entity", cascade="all, delete-orphan")
    # Video KYC checks are stored separately from face-match Identity Verification results.
    video_kyc_checks = relationship("VideoKYCCheck", back_populates="entity", cascade="all, delete-orphan")
    entity_reports = relationship(
        "EntityReport",
        back_populates="entity",
        cascade="all, delete-orphan",
    )


class Scan(Base):
    """One row per document scan.

    Stores the identity of the scanned document, the 11-layer forensic analysis result,
    and the two-level human approval status. Old OCR-based columns (truth_score,
    risk_level, verdict, parse_method, report_json, pdf_report) have been removed —
    all forensic signals are now captured in layered_analysis_json.
    """

    __tablename__ = "scans"

    id = Column(Integer, primary_key=True, autoincrement=True)
    entity_id = Column(
        Integer, ForeignKey("entities.id", ondelete="SET NULL"), nullable=True
    )
    # Identity of the document that was scanned
    source_name = Column(String(500), nullable=False)
    source_sha256 = Column(String(64), default="")
    document_type = Column(String(100), default="generic")
    # All 11 forensic layers (ELA, noise, metadata, clone, etc.) stored as JSON
    layered_analysis_json = Column(JSONB, nullable=True)
    # Legacy single-level approval — kept for backwards compat, derived from first/second levels
    approved = Column(String(10), nullable=True)  # 'approved' | 'rejected' | None
    approved_by = Column(String(255), nullable=True)
    approved_at = Column(DateTime(timezone=True), nullable=True)
    approval_comment = Column(Text, nullable=True)
    # Two-level human-in-the-loop approval — 'Y' = approved, 'N' = rejected, NULL = pending
    first_level_approval = Column(String(1), nullable=True)   # 1st reviewer decision
    first_level_approved_by = Column(String(255), nullable=True)
    first_level_approved_at = Column(DateTime(timezone=True), nullable=True)
    first_level_approval_comment = Column(Text, nullable=True)
    second_level_approval = Column(String(1), nullable=True)  # 2nd reviewer decision
    second_level_approved_by = Column(String(255), nullable=True)
    second_level_approved_at = Column(DateTime(timezone=True), nullable=True)
    second_level_approval_comment = Column(Text, nullable=True)
    generated_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    entity = relationship("Entity", back_populates="scans")
    extracted_info = relationship("DocumentExtraction", back_populates="scan", cascade="all, delete-orphan")


class DocumentExtraction(Base):
    """Structured fields extracted from document scans and identity verifications.

    Each row holds the key-value data pulled out of one document
    (e.g. candidate name + marks from a marksheet, salary + employer from a payslip,
    or PAN number + name from a PAN card).  Linked to the entity and, where
    applicable, the scan that produced the extraction.
    """

    # This table was previously called 'document_information'.
    # It was renamed to 'document_extractions' to better describe its purpose.
    __tablename__ = "document_extractions"
    __table_args__ = (
        UniqueConstraint(
            "entity_id",
            "file_name",
            name="uq_document_extractions_entity_file_name",
        ),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    entity_id = Column(Integer, ForeignKey("entities.id", ondelete="CASCADE"), nullable=False)
    # scan_id is optional — identity-verification extractions have no scan row
    scan_id = Column(Integer, ForeignKey("scans.id", ondelete="CASCADE"), nullable=True)
    # file_name is the natural key for document extractions within one entity.
    # This lets the latest extraction replace the earlier one for the same file.
    file_name = Column(String(500), nullable=False, default="")
    document_type = Column(String(100), default="generic")
    extracted_data = Column(JSONB, nullable=False)
    # Source screen that triggered the extraction — e.g. 'bulk_scan', 'identity_verification'
    source_screen = Column(String(100), default="")
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    entity = relationship("Entity", back_populates="extracted_info")
    scan = relationship("Scan", back_populates="extracted_info")

class IdentityCheck(Base):
    """One row per Identity Verification (face-match) event.

    Stores the ArcFace result together with the Aadhaar QR and PAN
    extraction data so the Final Report can summarise identity details
    without needing to read from document_extractions.
    """

    __tablename__ = "identity_checks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    entity_id = Column(
        Integer, ForeignKey("entities.id", ondelete="SET NULL"), nullable=True
    )
    status = Column(String(20), nullable=False)            # 'pass' | 'fail' | 'inconclusive'

    # ArcFace / face-match scores
    cosine_similarity = Column(Float, nullable=True)
    display_score = Column(Float, nullable=True)           # 0–100 percentage
    threshold = Column(Float, nullable=True)
    is_match = Column(Boolean, nullable=True)

    # Overall verdict
    verdict = Column(String(20), default="")               # 'PASS' | 'FAIL'

    # Full result payload for explainability and re-display in the UI
    report_json = Column(JSONB, nullable=False)

    # MinIO object key for the rendered PDF (e.g. "identity-reports/BT-000001/face_match.pdf").
    # VARCHAR(500) replaces the old LargeBinary column — PDF bytes live in MinIO, not the DB.
    pdf_report = Column(String(500), nullable=True, default="")

    # Aadhaar QR extracted fields stored as JSONB so the Final Report
    # builder can read them without touching document_extractions.
    aadhar_dtls = Column(JSONB, nullable=True)

    # PAN card extracted fields (pan_number, full_name, father_name, etc.)
    pan_dtls = Column(JSONB, nullable=True)

    # MinIO object keys for each uploaded image — replaces the old selfie_filename
    # text column for cross-document reports that need the actual image.
    selfie_pic = Column(String(500), nullable=True, default="")     # selfie image key
    aadhaar_pic = Column(String(500), nullable=True, default="")    # Aadhaar card image key
    pan_pic = Column(String(500), nullable=True, default="")        # PAN card image key
    signature_pic = Column(String(500), nullable=True, default="")  # extracted signature key

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    entity = relationship("Entity", back_populates="identity_checks")


class VideoKYCCheck(Base):
    """One row per Video KYC verification session result.

    Stores everything from the live video session — identity details,
    address proof data, geolocation comparison, challenge snapshots,
    and the PDF report MinIO key.  Completely separate from identity_checks
    (face-match) so each workflow can evolve independently.
    """

    __tablename__ = "video_kyc_checks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    entity_id = Column(
        Integer, ForeignKey("entities.id", ondelete="SET NULL"), nullable=True
    )

    # KYC outcome — 'pass' | 'fail' | 'inconclusive'
    status = Column(String(20), nullable=False, default="inconclusive")

    # ArcFace / face-match scores from the reference document vs live frame
    cosine_similarity = Column(Float, nullable=True)
    display_score = Column(Float, nullable=True)       # 0–100 percentage
    threshold = Column(Float, nullable=True)
    is_match = Column(Boolean, nullable=True)

    # Liveness challenge result (EAR-based blink detection via MediaPipe)
    liveness_state = Column(String(30), nullable=True)
    liveness_passed = Column(Boolean, nullable=True)

    # Identity document extracted fields (Aadhaar QR or Passport) stored as JSONB
    identity_dtls = Column(JSONB, nullable=True)

    # Address proof extracted fields (Aadhaar or Passport) stored as JSONB
    address_dtls = Column(JSONB, nullable=True)

    # MinIO object keys for uploaded images
    video_kyc_pic = Column(String(500), nullable=True, default="")       # best live frame
    address_proof_pic = Column(String(500), nullable=True, default="")   # address proof image
    reference_doc_pic = Column(String(500), nullable=True, default="")   # reference ID document

    # Address comparison result — 'match' | 'mismatch' | 'partial' | 'skipped'
    isAddressMatch = Column(String(20), nullable=True, default="skipped")

    # Free-text comments from the KYC officer or the system
    kyc_comments = Column(String(500), nullable=True, default="")

    # Geolocation captured from the user's browser during the session
    current_location_json = Column(JSONB, nullable=True)   # {lat, lon, accuracy, captured_at}
    current_address_text = Column(Text, nullable=True)     # reverse-geocoded address string
    address_distance_meters = Column(Float, nullable=True) # metres between proof and live location

    # Per-challenge snapshots: list of {challenge_name, frame_key (MinIO), ear_value}
    challenge_snapshots_json = Column(JSONB, nullable=True)

    # Full result payload stored for audit and re-display
    report_json = Column(JSONB, nullable=True)

    # MinIO object key for the rendered PDF report
    pdf_report = Column(String(500), nullable=True, default="")

    # 'PASS' | 'FAIL'
    verdict = Column(String(20), nullable=True, default="")

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    entity = relationship("Entity", back_populates="video_kyc_checks")


class EntityReport(Base):
    """Final cross-document verification report generated for one entity.

    Created from the Document Intelligence screen when an analyst clicks
    'Generate Final Report'.  Stores a structured comparison of names,
    addresses, PAN/Aadhaar numbers, salaries, and forensic verdicts across
    all of the entity's scanned documents.

    Undergoes the same two-level approval workflow as individual document
    scans so senior reviewers can endorse or dispute the findings before
    the report is considered complete.
    """

    __tablename__ = "entity_reports"

    id = Column(Integer, primary_key=True, autoincrement=True)
    # entity_id links this report to the applicant.  CASCADE means all
    # reports are deleted automatically when the entity is removed.
    entity_id = Column(Integer, ForeignKey("entities.id", ondelete="CASCADE"), nullable=False)
    # report_ref is the human-readable identifier shown to analysts, e.g. "BTR-000001".
    report_ref = Column(String(20), unique=True, nullable=False)
    # report_json holds the full cross-document analysis payload as structured JSON.
    report_json = Column(JSONB, nullable=False)
    # MinIO object key for the rendered PDF, e.g. "BTR-reports/BT-000001/BTR-000002.pdf".
    # Empty string means MinIO was unavailable at generation time — PDF not stored.
    report_minio_key = Column(String(500), default="")
    # Two-level approval columns — Y approved / N rejected / NULL pending (same as scans)
    first_level_approval = Column(String(1), nullable=True)
    first_level_approved_by = Column(String(255), default="")
    first_level_approved_at = Column(DateTime(timezone=True), nullable=True)
    first_level_approval_comment = Column(Text, default="")
    second_level_approval = Column(String(1), nullable=True)
    second_level_approved_by = Column(String(255), default="")
    second_level_approved_at = Column(DateTime(timezone=True), nullable=True)
    second_level_approval_comment = Column(Text, default="")
    generated_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    entity = relationship("Entity", back_populates="entity_reports")


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
        with engine.begin() as conn:
            # Forensic analysis JSON stored per scan row
            conn.execute(text(
                "ALTER TABLE scans ADD COLUMN IF NOT EXISTS layered_analysis_json JSONB"
            ))
            # Human-in-the-loop approval columns
            conn.execute(text(
                "ALTER TABLE scans ADD COLUMN IF NOT EXISTS approved VARCHAR(10)"
            ))
            conn.execute(text(
                "ALTER TABLE scans ADD COLUMN IF NOT EXISTS approved_by VARCHAR(255)"
            ))
            conn.execute(text(
                "ALTER TABLE scans ADD COLUMN IF NOT EXISTS approved_at TIMESTAMPTZ"
            ))
            conn.execute(text(
                "ALTER TABLE scans ADD COLUMN IF NOT EXISTS approval_comment TEXT"
            ))
            conn.execute(text(
                "ALTER TABLE scans ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT NOW()"
            ))
            # Two-level approval columns — Y/N/NULL per level
            conn.execute(text(
                "ALTER TABLE scans ADD COLUMN IF NOT EXISTS first_level_approval VARCHAR(1)"
            ))
            conn.execute(text(
                "ALTER TABLE scans ADD COLUMN IF NOT EXISTS first_level_approved_by VARCHAR(255)"
            ))
            conn.execute(text(
                "ALTER TABLE scans ADD COLUMN IF NOT EXISTS first_level_approved_at TIMESTAMPTZ"
            ))
            conn.execute(text(
                "ALTER TABLE scans ADD COLUMN IF NOT EXISTS first_level_approval_comment TEXT"
            ))
            conn.execute(text(
                "ALTER TABLE scans ADD COLUMN IF NOT EXISTS second_level_approval VARCHAR(1)"
            ))
            conn.execute(text(
                "ALTER TABLE scans ADD COLUMN IF NOT EXISTS second_level_approved_by VARCHAR(255)"
            ))
            conn.execute(text(
                "ALTER TABLE scans ADD COLUMN IF NOT EXISTS second_level_approved_at TIMESTAMPTZ"
            ))
            conn.execute(text(
                "ALTER TABLE scans ADD COLUMN IF NOT EXISTS second_level_approval_comment TEXT"
            ))
            # Migrate old table name document_information → document_extractions.
            # We check if the old table exists and the new one doesn't, then rename.
            # This handles existing databases created before the rename.
            conn.execute(text(
                "DO $$ BEGIN "
                "IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'document_information') "
                "AND NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'document_extractions') "
                "THEN ALTER TABLE document_information RENAME TO document_extractions; "
                "END IF; END $$"
            ))
            # Add the new source_screen column if it was not there before the rename.
            conn.execute(text(
                "ALTER TABLE document_extractions "
                "ADD COLUMN IF NOT EXISTS source_screen VARCHAR(100) DEFAULT ''"
            ))
            conn.execute(text(
                "ALTER TABLE document_extractions "
                "ADD COLUMN IF NOT EXISTS file_name VARCHAR(500) DEFAULT ''"
            ))
            # scan_id was NOT NULL in the old schema — relax it so identity extractions
            # (which have no scan row) can also be stored here.
            conn.execute(text(
                "ALTER TABLE document_extractions "
                "ALTER COLUMN scan_id DROP NOT NULL"
            ))
            # Backfill file_name from the linked scan row where possible.
            conn.execute(text(
                "UPDATE document_extractions AS de "
                "SET file_name = s.source_name "
                "FROM scans AS s "
                "WHERE de.scan_id = s.id "
                "AND COALESCE(de.file_name, '') = ''"
            ))
            # Older identity rows have no scan row, so use a deterministic synthetic
            # file name based on the screen and document type to keep them addressable.
            conn.execute(text(
                "UPDATE document_extractions "
                "SET file_name = CONCAT(source_screen, '_', document_type) "
                "WHERE COALESCE(file_name, '') = ''"
            ))
            # Keep the newest row for each (entity_id, file_name) pair before adding
            # the uniqueness guarantee used by the new UPSERT path.
            conn.execute(text(
                "WITH ranked AS ("
                "  SELECT id, ROW_NUMBER() OVER ("
                "    PARTITION BY entity_id, file_name "
                "    ORDER BY created_at DESC NULLS LAST, id DESC"
                "  ) AS rn "
                "  FROM document_extractions"
                ") "
                "DELETE FROM document_extractions AS de "
                "USING ranked "
                "WHERE de.id = ranked.id AND ranked.rn > 1"
            ))
            conn.execute(text(
                "CREATE UNIQUE INDEX IF NOT EXISTS "
                "ux_document_extractions_entity_file_name "
                "ON document_extractions (entity_id, file_name)"
            ))
            # entity_reports table — final cross-document verification reports
            conn.execute(text(
                "CREATE TABLE IF NOT EXISTS entity_reports ("
                "id SERIAL PRIMARY KEY, "
                "entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE, "
                "report_ref VARCHAR(20) UNIQUE NOT NULL, "
                "report_json JSONB NOT NULL, "
                "report_minio_key VARCHAR(500) DEFAULT '', "
                "first_level_approval VARCHAR(1), "
                "first_level_approved_by VARCHAR(255) DEFAULT '', "
                "first_level_approved_at TIMESTAMPTZ, "
                "first_level_approval_comment TEXT DEFAULT '', "
                "second_level_approval VARCHAR(1), "
                "second_level_approved_by VARCHAR(255) DEFAULT '', "
                "second_level_approved_at TIMESTAMPTZ, "
                "second_level_approval_comment TEXT DEFAULT '', "
                "generated_at TIMESTAMPTZ DEFAULT NOW(), "
                "updated_at TIMESTAMPTZ DEFAULT NOW()"
                ")"
            ))
            # Add report_minio_key to existing entity_reports rows created before this column existed.
            conn.execute(text(
                "ALTER TABLE entity_reports "
                "ADD COLUMN IF NOT EXISTS report_minio_key VARCHAR(500) DEFAULT ''"
            ))

            # ── video_kyc_checks table ──────────────────────────────────────────────
            # Stores Video KYC session results separately from face-match identity checks.
            # Created only if it does not already exist so running init_db() repeatedly is safe.
            conn.execute(text(
                "CREATE TABLE IF NOT EXISTS video_kyc_checks ("
                "id SERIAL PRIMARY KEY, "
                "entity_id INTEGER REFERENCES entities(id) ON DELETE SET NULL, "
                "status VARCHAR(20) NOT NULL DEFAULT 'inconclusive', "
                "cosine_similarity DOUBLE PRECISION, "
                "display_score DOUBLE PRECISION, "
                "threshold DOUBLE PRECISION, "
                "is_match BOOLEAN, "
                "liveness_state VARCHAR(30), "
                "liveness_passed BOOLEAN, "
                "identity_dtls JSONB, "
                "address_dtls JSONB, "
                "video_kyc_pic VARCHAR(500) DEFAULT '', "
                "address_proof_pic VARCHAR(500) DEFAULT '', "
                "reference_doc_pic VARCHAR(500) DEFAULT '', "
                "\"isAddressMatch\" VARCHAR(20) DEFAULT 'skipped', "
                "kyc_comments VARCHAR(500) DEFAULT '', "
                "current_location_json JSONB, "
                "current_address_text TEXT, "
                "address_distance_meters DOUBLE PRECISION, "
                "challenge_snapshots_json JSONB, "
                "report_json JSONB, "
                "pdf_report VARCHAR(500) DEFAULT '', "
                "verdict VARCHAR(20) DEFAULT '', "
                "created_at TIMESTAMPTZ DEFAULT NOW(), "
                "updated_at TIMESTAMPTZ DEFAULT NOW()"
                ")"
            ))

            # ── identity_checks new columns ────────────────────────────────────────
            # These columns were added as part of the Identity / Video-KYC split.
            # ADD COLUMN IF NOT EXISTS is idempotent — safe to run on every startup.
            conn.execute(text(
                "ALTER TABLE identity_checks "
                "ADD COLUMN IF NOT EXISTS aadhar_dtls JSONB"
            ))
            conn.execute(text(
                "ALTER TABLE identity_checks "
                "ADD COLUMN IF NOT EXISTS pan_dtls JSONB"
            ))
            conn.execute(text(
                "ALTER TABLE identity_checks "
                "ADD COLUMN IF NOT EXISTS selfie_pic VARCHAR(500) DEFAULT ''"
            ))
            conn.execute(text(
                "ALTER TABLE identity_checks "
                "ADD COLUMN IF NOT EXISTS aadhaar_pic VARCHAR(500) DEFAULT ''"
            ))
            conn.execute(text(
                "ALTER TABLE identity_checks "
                "ADD COLUMN IF NOT EXISTS pan_pic VARCHAR(500) DEFAULT ''"
            ))
            conn.execute(text(
                "ALTER TABLE identity_checks "
                "ADD COLUMN IF NOT EXISTS signature_pic VARCHAR(500) DEFAULT ''"
            ))
            conn.execute(text(
                "ALTER TABLE identity_checks "
                "ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT NOW()"
            ))
            # Drop stale columns that no longer belong on identity_checks.
            # liveness fields live in video_kyc_checks; filename columns are superseded
            # by MinIO key columns (selfie_pic, aadhaar_pic, pan_pic, signature_pic).
            for _col in ("liveness_state", "liveness_passed", "doc_filename", "selfie_filename", "check_type"):
                conn.execute(text(
                    f"ALTER TABLE identity_checks DROP COLUMN IF EXISTS {_col}"
                ))
            # Migrate pdf_report from BYTEA (LargeBinary) to VARCHAR(500) MinIO key.
            # We rename the old binary column so existing data is not lost, then add
            # the new string column.  The DO block is idempotent — it only renames
            # when the column is still of type bytea.
            conn.execute(text(
                """
                DO $$ BEGIN
                  IF EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'identity_checks'
                      AND column_name = 'pdf_report'
                      AND data_type = 'bytea'
                  ) THEN
                    ALTER TABLE identity_checks
                      RENAME COLUMN pdf_report TO pdf_report_bytes;
                    ALTER TABLE identity_checks
                      ADD COLUMN pdf_report VARCHAR(500) DEFAULT '';
                  END IF;
                END $$;
                """
            ))

        log.info("DB schema ready")
        return True
    except Exception as exc:
        log.warning("init_db failed: %s", exc)
        return False
