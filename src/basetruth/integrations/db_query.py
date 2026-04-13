"""Safe query execution engine for the BaseTruth Q&A chatbot."""
from __future__ import annotations

import re
from typing import Any, Dict, List

from sqlalchemy import text

from basetruth.db import db_available, db_session
from basetruth.logger import get_logger
from basetruth.store import (
    minio_available,
    minio_bucket_stats,
    minio_list_entity_objects,
    minio_list_objects,
)

log = get_logger(__name__)

# Very strict regex whitelist for allowed statements
_WHITELIST_REGEX = re.compile(
    r"^\s*(WITH\s+.*)?SELECT\s+.*$", re.IGNORECASE | re.DOTALL
)

# Reject if we see any of these keywords even inside a SELECT
_BLACKLIST_REGEX = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|CREATE|GRANT|REVOKE|COPY|EXECUTE|pg_sleep|pg_terminate_backend|lo_export|lo_import)\b",
    re.IGNORECASE,
)

def get_schema_summary() -> str:
    """Return a compact DDL-like text summarising the relevant SQL tables."""
    return """
TABLE entities (id, entity_ref, first_name, last_name, email, phone, pan_number, aadhar_number, created_at)
  - Tip: For names, always use ILIKE '%name%' to handle middle names or partial matches (e.g. first_name ILIKE '%Hrishikesh%').
TABLE scans (id, entity_id FK->entities, source_name, source_sha256, document_type, layered_analysis_json JSONB, approved, first_level_approval, second_level_approval, generated_at, updated_at)
TABLE identity_checks (id, entity_id FK->entities, check_type, status, verdict, cosine_similarity, doc_filename, created_at)
  - 'check_type' is either 'face_match' or 'video_kyc'.
TABLE cases (id, case_key, entity_id FK->entities, status, disposition, priority, max_risk_level, document_count, created_at)
TABLE case_notes (id, case_id FK->cases, author, text, created_at)
TABLE document_extractions (id, entity_id FK->entities, scan_id FK->scans nullable, file_name, document_type, extracted_data JSONB, source_screen, created_at)
TABLE layered_analysis_entries (id, entity_id FK->entities, screen_name, section_name, details_captured_json JSONB, updated_at)
  - This table contains detailed breakdown of all the scans and verifications that took place.
"""

def execute_safe_query(sql_query: str) -> str:
    """Validate and execute a SQL query safely, returning markdown table results."""
    if not db_available():
        return "I do not have access to the database right now."

    cleaned_sql = sql_query.strip().strip(";")
    
    if not _WHITELIST_REGEX.match(cleaned_sql):
        log.warning("execute_safe_query: Query rejected (not a valid SELECT). SQL: %s", cleaned_sql)
        return "The query could not be executed for security reasons. Only SELECT statements are allowed."
        
    if _BLACKLIST_REGEX.search(cleaned_sql):
        log.warning("execute_safe_query: Query rejected (contains blacklisted keyword). SQL: %s", cleaned_sql)
        return "The query could not be executed for security reasons."

    try:
        with db_session() as session:
            # 10s timeout
            session.execute(text("SET statement_timeout = 10000"))
            
            # Execute query
            result = session.execute(text(cleaned_sql))
            rows = result.fetchmany(100)  # enforce limit at fetch time
            
            if not rows:
                return "The query completed successfully, but returned 0 rows."
            
            columns = list(result.keys())
            
            # Always roll back to ensure no unintended side effects are persisted
            session.rollback()
            
            # Format as Markdown table
            header = "| " + " | ".join(str(c) for c in columns) + " |"
            separator = "| " + " | ".join("---" for _ in columns) + " |"
            
            md_rows = []
            for row in rows:
                # Provide a safe string representation, avoiding huge blobs
                safe_row = []
                for val in row:
                    s_val = str(val).replace("\n", " ").replace("|", "\\|")
                    if len(s_val) > 250:
                        s_val = s_val[:247] + "..."
                    safe_row.append(s_val)
                md_rows.append("| " + " | ".join(safe_row) + " |")
                
            return f"Query returned {len(rows)} rows (limited to 100):\n\n" + "\n".join([header, separator] + md_rows)
            
    except Exception as exc:
        log.warning("execute_safe_query: Execution failed: %s", exc)
        return "The query could not be completed. It may be invalid or refer to columns that don't exist."

def get_minio_summary() -> str:
    """Return a summary of the MinIO object storage state."""
    if not minio_available():
        return ""
    stats = minio_bucket_stats()
    if not stats.get("available"):
        return ""
    return f"MinIO bucket '{stats.get('bucket')}' contains {stats.get('object_count')} objects ({stats.get('total_mb')} MB)."

def query_minio_objects(command: str) -> str:
    """Process a minio command (like 'LIST ALL' or 'LIST ENTITY BT-000001')."""
    if not minio_available():
        return "I do not have access to object storage right now."
        
    cmd = command.strip().upper()
    
    try:
        if cmd == "LIST ALL":
            objects = minio_list_objects(limit=100)
            if not objects:
                return "No objects found in storage."
            
            header = "| Filename/Key | Size (KB) | Last Modified |"
            separator = "|---|---|---|"
            md_rows = [f"| {obj.get('key')} | {obj.get('size_kb')} | {obj.get('last_modified')} |" for obj in objects]
            return "Latest storage objects (limited to 100):\n\n" + "\n".join([header, separator] + md_rows)
            
        elif cmd.startswith("LIST ENTITY"):
            parts = cmd.split(" ", 2)
            if len(parts) < 3:
                return "Invalid command format. Use: LIST ENTITY <entity_ref>"
            entity_ref = parts[2].strip()
            
            objects = minio_list_entity_objects(entity_ref)
            if not objects:
                return f"No objects found for entity {entity_ref}."
                
            header = "| Filename | Size (KB) | Last Modified |"
            separator = "|---|---|---|"
            md_rows = [f"| {obj.get('filename')} | {obj.get('size_kb')} | {obj.get('last_modified')} |" for obj in objects]
            return f"Storage objects for {entity_ref}:\n\n" + "\n".join([header, separator] + md_rows)
            
        return "Unknown storage command. Use LIST ALL or LIST ENTITY <ref>."
    except Exception as exc:
        log.warning("query_minio_objects: Failed: %s", exc)
        return "Could not retrieve storage information."
