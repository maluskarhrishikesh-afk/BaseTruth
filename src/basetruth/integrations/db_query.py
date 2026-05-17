"""Safe query execution engine for the BaseTruth Q&A chatbot."""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

from sqlalchemy import text

from basetruth.db import db_available, db_session
from basetruth.logger import get_logger
from basetruth.store import (
    minio_available,
    minio_bucket_stats,
    minio_docs_get,
    minio_docs_put,
    minio_list_entity_objects,
    minio_list_objects,
)

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Prompt loading — all Q&A prompt text lives in qna_prompts.md so that
# operators can tune chatbot behaviour without editing Python.
# ---------------------------------------------------------------------------

_QNA_PROMPTS_PATH = Path(__file__).with_name("qna_prompts.md")

# DATABASE.md lives at the project root's docs/ folder — 4 levels up from this file.
# (src/basetruth/integrations/db_query.py → project root)
# Injecting the full DATABASE.md gives the model the authoritative column-level
# schema so it never guesses wrong column names (e.g. entity_id on entities).
# Matches the same section format used in document_extract_prompts.md:
#   ## section_name
#   ```text
#   ...body...
#   ```
_QNA_SECTION_PATTERN = re.compile(
    r"(?ms)^##\s+(?P<name>[a-z0-9_]+)\s*\n```(?:text|prompt)?\n(?P<body>.*?)\n```"
)

_QNA_REQUIRED_SECTIONS = frozenset({
    "system_prompt", "db_query_rules", "minio_instructions",
    "product_knowledge", "business_rules", "glossary", "training_examples",
})


# Module-level cache for DATABASE.md content.
# None = not yet loaded. We use a sentinel instead of @lru_cache so that:
#   1. A failed load (empty string) is never permanently cached.
#   2. Subsequent calls can retry MinIO after the startup sync has uploaded the file.
_DATABASE_MD_CONTENT: Optional[str] = None


def _load_database_md() -> str:
    """Load DATABASE.md and cache it in the module-level _DATABASE_MD_CONTENT sentinel.

    DATABASE.md is the authoritative column-level schema reference. Loading it
    into the model context prevents wrong column name guesses (e.g. 'entity_id'
    on the 'entities' table) and ensures the model uses the exact column names,
    join keys, and UPSERT behaviour documented by the engineering team.

    Reads only from the MinIO 'basetruth-docs' bucket. The UI uploads the
    authoritative local docs/DATABASE.md into MinIO at startup via
    sync_database_md_to_minio(), and all runtime prompt construction then reads
    the same object back from MinIO. This keeps one canonical runtime source
    of truth instead of mixing filesystem and object-storage reads.

    On success the content is cached so subsequent calls are zero-cost.
    On failure the empty string is returned WITHOUT caching so the next call
    can retry (useful when MinIO finishes its startup upload slightly late).
    """
    global _DATABASE_MD_CONTENT
    if _DATABASE_MD_CONTENT is not None:
        return _DATABASE_MD_CONTENT

    # Runtime reads are MinIO-only. The upload step is handled separately by
    # sync_database_md_to_minio(), which is called from the UI on startup.
    log.info("DATABASE.md fetch: attempting read from MinIO docs bucket")
    try:
        data = minio_docs_get("DATABASE.md")
        if data:
            content = data.decode("utf-8")
            log.info(
                "DATABASE.md fetch: fetched successfully from MinIO (%d chars)",
                len(content),
            )
            _DATABASE_MD_CONTENT = content
            return _DATABASE_MD_CONTENT
        else:
            log.warning(
                "DATABASE.md fetch: minio_docs_get returned empty/None — "
                "falling back to local filesystem"
            )
    except Exception as exc:
        log.warning("DATABASE.md fetch: MinIO read failed (%s) — falling back to filesystem", exc)

    # ── Filesystem fallback ───────────────────────────────────────────────────
    # MinIO was unavailable or had no file — try reading directly from disk.
    # This prevents the catastrophic case where the model gets zero schema
    # context and hallucinates table names like 'document_uploads'.
    candidate_paths = [
        Path(__file__).parents[3] / "docs" / "DATABASE.md",  # src/basetruth/integrations → project root
        Path.cwd() / "docs" / "DATABASE.md",
    ]
    for candidate in candidate_paths:
        try:
            content = candidate.read_text(encoding="utf-8")
            log.info(
                "DATABASE.md fetch: loaded from local filesystem fallback (%s, %d chars)",
                candidate, len(content),
            )
            # Cache the filesystem content so subsequent calls are zero-cost.
            # We do NOT set _DATABASE_MD_CONTENT here because this is the fallback path;
            # the next request should re-attempt MinIO first.
            return content
        except OSError:
            continue

    log.warning(
        "DATABASE.md fetch: file NOT found in MinIO or on filesystem — "
        "schema context will be INCOMPLETE (model may hallucinate table names). "
        "Ensure docs/DATABASE.md exists in your project and MinIO is running."
    )
    return ""  # Do NOT cache the empty-string failure so the next call can retry


def sync_database_md_to_minio() -> bool:
    """Upload the local DATABASE.md to the MinIO 'basetruth-docs' bucket.

    Call this once at startup from the Streamlit UI. The upload ensures that
    all runtime prompt construction reads the authoritative schema reference
    from MinIO via _load_database_md(). If MinIO is unreachable or the file is
    not found locally, this
    function logs a warning and returns False without raising.

    Returns True on a successful upload, False otherwise.
    """
    # Read the local source document that will be uploaded into MinIO.
    candidate_from_file = Path(__file__).parents[3] / "docs" / "DATABASE.md"
    candidate_from_cwd  = Path.cwd() / "docs" / "DATABASE.md"

    content: Optional[str] = None
    for candidate in (candidate_from_file, candidate_from_cwd):
        try:
            content = candidate.read_text(encoding="utf-8")
            break
        except OSError:
            continue

    if content is None:
        log.warning(
            "sync_database_md_to_minio: DATABASE.md not found on filesystem — "
            "skipping MinIO upload (no docs/ directory available locally)"
        )
        return False

    success = minio_docs_put("DATABASE.md", content.encode("utf-8"), "text/markdown")
    if success:
        log.info(
            "sync_database_md_to_minio: Uploaded DATABASE.md to MinIO docs bucket (%d chars)",
            len(content),
        )
        # Clear and refresh the in-memory cache from the MinIO-backed source of truth.
        global _DATABASE_MD_CONTENT
        _DATABASE_MD_CONTENT = None
    else:
        log.warning("sync_database_md_to_minio: MinIO upload failed — docs bucket may be unavailable")
    return success


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
    """Return the full system-prompt for the Q&A LLM.

    Combines the master prompt, platform knowledge, business rules,
    and the terminology glossary so the model understands BaseTruth
    domain language before seeing any user question.
    """
    prompts = _load_qna_prompts()
    # The glossary is prepended to the product knowledge so the model can map
    # informal user terms ("salary slip", "rejected", "risky") to exact DB concepts
    # before it decides which table or column to query.
    return (
        prompts["system_prompt"]
        + "\n\n" + prompts["business_rules"]
        + "\n\n" + prompts["glossary"]
        + "\n\n" + prompts["product_knowledge"]
    )


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

def get_qna_training_examples() -> str:
    """Return the training example Q&A pairs from qna_prompts.md."""
    return _load_qna_prompts()["training_examples"]


def get_schema_summary() -> str:
    """Return the DB context block injected into the LLM system prompt.

    Loads the full DATABASE.md (both Plain English + Technical Reference) from
    filesystem (dev) or MinIO (Docker/prod) via _load_database_md(). Also includes
    db_query_rules (per-table common query patterns) and training_examples
    (multi-table join patterns). With gemma4:e2b's 128k context window there is
    no token budget concern — quality and correctness come first.
    """
    # Load the full DATABASE.md — the single source of truth for table names,
    # column definitions, types, join keys, and operational upsert behaviour.
    database_md = _load_database_md()

    # db_query_rules provides concise per-table common query patterns
    db_rules = get_qna_db_rules()
    # training_examples provides multi-table join patterns
    examples = get_qna_training_examples()

    # Put a compact non-negotiable schema guardrail BEFORE the full DATABASE.md.
    # Small and mid-size models attend strongly to the start of the prompt; this
    # makes the valid table names impossible to miss even when the full schema is long.
    schema_guardrail = (
        "NON-NEGOTIABLE SQL GUARDRAILS:\n"
        "- ONLY these PostgreSQL tables exist: entities, scans, document_extractions, identity_checks, video_kyc_checks, entity_reports, face_scan_live_results.\n"
        "- NEVER use: users, names, documents, uploads, customers, applicants, checks, reports.\n"
        "- Use 'entities.id' as the primary key. NEVER use 'entity_id' on the entities table.\n"
        "- For uploaded documents by applicant, join entities.id to scans.entity_id and/or document_extractions.entity_id.\n"
        "- Video KYC rows live in video_kyc_checks. Live Face Scan session rows live in face_scan_live_results and use session_id rather than entity_id.\n"
        "- Before emitting SQL, verify every table and column name appears in DATABASE.md below."
    )

    parts = ["LIVE DATABASE CONTEXT:"]
    parts.append(
        "The following is the authoritative PostgreSQL schema for this BaseTruth instance. "
        "Use the column definitions and example queries below to write accurate SQL. "
        "Only SELECT is permitted — no writes."
    )
    parts.append("=" * 60)
    parts.append(schema_guardrail)
    parts.append("=" * 60)

    if database_md:
        parts.append("AUTHORITATIVE DATABASE SCHEMA (docs/DATABASE.md):")
        parts.append(database_md)
    else:
        # DATABASE.md could not be loaded — fall back to a minimal inline reference
        # so the model still knows which tables exist rather than guessing wrong names.
        parts.append(
            "AVAILABLE TABLES (use ONLY these — do NOT use users/customers/applicants/documents):\n"
            "  entities             — one row per applicant (PK = id, human ref = entity_ref)\n"
            "  scans                — one row per document scan (FK entity_id → entities.id)\n"
            "  document_extractions — extracted fields from each document (FK entity_id → entities.id)\n"
            "  identity_checks      — Identity Verification face-match results only (FK entity_id → entities.id)\n"
            "  video_kyc_checks     — Video KYC session results only (FK entity_id → entities.id)\n"
            "  entity_reports       — final cross-document reports (FK entity_id → entities.id)\n"
            "  face_scan_live_results — durable live Face Scan session results (keyed by session_id, no entity_id FK)\n"
        )

    parts.append("=" * 60)
    parts.append("COMMON QUERY PATTERNS AND EXAMPLES:")
    parts.append("=" * 60)
    parts.append(db_rules)
    parts.append(examples)
    parts.append(
        "CRITICAL SQL RULES:\n"
        "- 'entities' PK is 'id' (NOT 'entity_id'). Use 'entity_ref' (e.g. BT-000001) for display.\n"
        "- 'entity_id' exists ONLY in child tables (scans, document_extractions, "
        "identity_checks, video_kyc_checks, entity_reports) as a FK to entities.id.\n"
        "- 'face_scan_live_results' does not use entity_id; query it by session_id, verdict, timestamps, or JSON fields.\n"
        "- Always use ILIKE for name searches (partial, case-insensitive).\n"
        "- Always add LIMIT 20 on non-aggregate queries.\n"
        "- Use only SELECT — no writes are permitted."
    )

    return "\n\n".join(parts)



def execute_safe_query(sql_query: str) -> str:
    """Validate and execute a SQL query safely, returning markdown table results."""
    if not db_available():
        log.warning("execute_safe_query: Database not available — skipping query")
        return "I do not have access to the database right now."

    cleaned_sql = sql_query.strip().strip(";")
    log.info("execute_safe_query: Validating SQL query | sql=%s", cleaned_sql)

    if not _WHITELIST_REGEX.match(cleaned_sql):
        log.warning("execute_safe_query: Query rejected — not a valid SELECT | sql=%s", cleaned_sql)
        return "The query could not be executed for security reasons. Only SELECT statements are allowed."

    if _BLACKLIST_REGEX.search(cleaned_sql):
        log.warning("execute_safe_query: Query rejected — contains blacklisted keyword | sql=%s", cleaned_sql)
        return "The query could not be executed for security reasons."

    try:
        with db_session() as session:
            # Limit each query to 10 seconds to prevent slow-query hangs in the UI
            session.execute(text("SET statement_timeout = 10000"))

            log.info("execute_safe_query: Executing SQL | sql=%s", cleaned_sql)
            result = session.execute(text(cleaned_sql))
            rows = result.fetchmany(100)  # hard cap: never return more than 100 rows
            columns = list(result.keys())

            # Always roll back — queries are read-only, rollback prevents any
            # accidental state change if the DB driver auto-opened a transaction
            session.rollback()

            log.info(
                "execute_safe_query: Query completed | rows_returned=%d columns=%s sql=%s",
                len(rows), columns, cleaned_sql,
            )

            if not rows:
                return "The query completed successfully, but returned 0 rows."

            # Format results as a Markdown table for the LLM to interpret
            header    = "| " + " | ".join(str(c) for c in columns) + " |"
            separator = "| " + " | ".join("---" for _ in columns) + " |"

            md_rows = []
            for row in rows:
                safe_row = []
                for val in row:
                    # Truncate large JSON blobs so the LLM context stays manageable
                    s_val = str(val).replace("\n", " ").replace("|", "\\|")
                    if len(s_val) > 250:
                        s_val = s_val[:247] + "..."
                    safe_row.append(s_val)
                md_rows.append("| " + " | ".join(safe_row) + " |")

            return (
                f"Query returned {len(rows)} row(s) (capped at 100):\n\n"
                + "\n".join([header, separator] + md_rows)
            )

    except Exception as exc:
        # Classify the error type to give the LLM a targeted correction hint
        exc_str = str(exc)
        if "UndefinedTable" in type(exc).__name__ or "does not exist" in exc_str.lower() and "relation" in exc_str.lower():
            # The model queried a table that doesn't exist — tell it the valid ones
            correction_hint = (
                "CORRECTION REQUIRED: The SQL referenced a table that does not exist.\n"
                "Valid tables are: entities, scans, document_extractions, identity_checks, entity_reports.\n"
                "Common mistakes: 'users' → use 'entities'; 'documents' → use 'scans'; "
                "'checks' → use 'identity_checks'; 'reports' → use 'entity_reports'."
            )
        elif "UndefinedColumn" in type(exc).__name__ or "column" in exc_str.lower() and "does not exist" in exc_str.lower():
            correction_hint = (
                "CORRECTION REQUIRED: The SQL used a column that does not exist.\n"
                "Reminder: 'entities' PK is 'id' (NOT entity_id or user_id). "
                "'entity_id' exists only in child tables as a FK to entities.id."
            )
        else:
            correction_hint = "Check the database schema and try a different question."

        log.error(
            "execute_safe_query: Execution failed | sql=%s error=%s",
            cleaned_sql, exc,
        )
        return (
            f"The query could not be completed. Error: {exc}\n\n{correction_hint}"
        )

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
