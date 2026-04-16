"""Safe query execution engine for the BaseTruth Q&A chatbot."""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Dict

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

# ---------------------------------------------------------------------------
# Prompt loading — all Q&A prompt text lives in qna_prompts.md so that
# operators can tune chatbot behaviour without editing Python.
# ---------------------------------------------------------------------------

_QNA_PROMPTS_PATH = Path(__file__).with_name("qna_prompts.md")

# Matches the same section format used in document_extract_prompts.md:
#   ## section_name
#   ```text
#   ...body...
#   ```
_QNA_SECTION_PATTERN = re.compile(
    r"(?ms)^##\s+(?P<name>[a-z0-9_]+)\s*\n```(?:text|prompt)?\n(?P<body>.*?)\n```"
)

_QNA_REQUIRED_SECTIONS = frozenset({"system_prompt", "db_query_rules", "minio_instructions", "product_knowledge"})


@lru_cache(maxsize=1)
def _load_qna_prompts() -> Dict[str, str]:
    """Load all prompt sections from qna_prompts.md once per process.

    Keeping prompts in a separate markdown file lets analysts and product
    managers read and tune the chatbot behaviour without touching Python.
    The @lru_cache means the file is only read from disk once.
    """
    try:
        raw = _QNA_PROMPTS_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(
            f"Q&A prompt asset missing or unreadable: {_QNA_PROMPTS_PATH}"
        ) from exc

    sections = {
        m.group("name"): m.group("body").strip("\n")
        for m in _QNA_SECTION_PATTERN.finditer(raw)
    }
    missing = sorted(_QNA_REQUIRED_SECTIONS - set(sections))
    if missing:
        raise RuntimeError("qna_prompts.md is missing required sections: " + ", ".join(missing))
    return sections


def get_qna_system_prompt() -> str:
    """Return the base system-prompt text from qna_prompts.md."""
    prompts = _load_qna_prompts()
    return prompts["system_prompt"] + "\n\n" + prompts["product_knowledge"]


def get_qna_db_rules() -> str:
    """Return the database query rules text from qna_prompts.md."""
    return _load_qna_prompts()["db_query_rules"]


def get_qna_minio_instructions() -> str:
    """Return the MinIO query instructions text from qna_prompts.md."""
    return _load_qna_prompts()["minio_instructions"]

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
    """Return a compact schema summary injected into the LLM system prompt.

    This text appears after the base system prompt when DB is available,
    giving the model everything it needs to write correct SQL queries.
    The full query patterns and examples live in qna_prompts.md (db_query_rules section).
    """
    # Load the detailed query rules from the markdown asset so they stay in sync
    # with the qna_prompts.md file rather than being hardcoded here.
    db_rules = get_qna_db_rules()

    return (
        "LIVE DATABASE CONTEXT:\n"
        "The following schema and query patterns describe the PostgreSQL database "
        "connected to this BaseTruth instance. Use them to write accurate SQL.\n\n"
        + db_rules
        + "\n\nNOTE: Always use ILIKE for name searches (partial, case-insensitive). "
        "Always add LIMIT 20 on non-aggregate queries. "
        "Use only SELECT — no writes are permitted."
    )

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
