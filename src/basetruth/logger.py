"""BaseTruth — centralized structured logging.

Every module in BaseTruth should obtain its logger via:

    from basetruth.logger import get_logger
    log = get_logger(__name__)

The root logger is configured once (on first import) to write:
  • Coloured plain-text to stderr (human-readable during development).
  • Newline-delimited JSON to  <log_dir>/basetruth.jsonl  (machine-readable,
    easy to ingest into the Log Analyzer UI or any external tool).

Environment variables
---------------------
  BT_LOG_LEVEL   — root log level, default INFO  (DEBUG | INFO | WARNING | ERROR)
  BT_LOG_DIR     — directory for the JSONL log file, default  logs/
  BT_LOG_FILE    — explicit path to JSONL file (overrides BT_LOG_DIR)
"""
from __future__ import annotations

import json
import logging
import os
import sys
import traceback
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

# ── Constants ────────────────────────────────────────────────────────────────

_DEFAULT_LOG_DIR  = Path("logs")
_DEFAULT_FILENAME = "basetruth.jsonl"
_MAX_BYTES        = 10 * 1024 * 1024   # 10 MB per file
_BACKUP_COUNT     = 5                   # keep 5 rotated files
_ROOT_LOGGER_NAME = "basetruth"

# ── Internal state ───────────────────────────────────────────────────────────

_configured = False


# ── JSON formatter ───────────────────────────────────────────────────────────

class _JsonFormatter(logging.Formatter):
    """Emit one JSON object per line with consistent fields."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        data: dict = {
            "ts":      datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level":   record.levelname,
            "logger":  record.name,
            "module":  record.module,
            "func":    record.funcName,
            "line":    record.lineno,
            "msg":     record.getMessage(),
        }
        if record.exc_info:
            data["exc"] = self.formatException(record.exc_info)
        # Extra fields attached via log.info("msg", extra={"doc": "foo"})
        for key, val in record.__dict__.items():
            if key not in (
                "args", "asctime", "created", "exc_info", "exc_text", "filename",
                "funcName", "id", "levelname", "levelno", "lineno", "message",
                "module", "msecs", "msg", "name", "pathname", "process",
                "processName", "relativeCreated", "stack_info", "taskName",
                "thread", "threadName",
            ) and not key.startswith("_"):
                try:
                    json.dumps(val)          # only include JSON-serialisable extras
                    data[key] = val
                except (TypeError, ValueError):
                    data[key] = str(val)
        return json.dumps(data, ensure_ascii=False)


# ── Coloured stderr formatter ────────────────────────────────────────────────

_COLOURS = {
    "DEBUG":    "\033[36m",   # cyan
    "INFO":     "\033[32m",   # green
    "WARNING":  "\033[33m",   # yellow
    "ERROR":    "\033[31m",   # red
    "CRITICAL": "\033[35m",   # magenta
}
_RESET = "\033[0m"


class _PrettyFormatter(logging.Formatter):
    _FMT = "{colour}[{level}]{reset} {ts}  {logger}:{line}  {msg}"

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        colour  = _COLOURS.get(record.levelname, "")
        reset   = _RESET if colour else ""
        ts      = datetime.fromtimestamp(record.created, tz=timezone.utc).strftime("%H:%M:%S")
        # shorten logger name — keep last two segments
        name    = ".".join(record.name.split(".")[-2:]) if "." in record.name else record.name
        text    = self._FMT.format(
            colour=colour, reset=reset, level=record.levelname[:4],
            ts=ts, logger=name, line=record.lineno, msg=record.getMessage(),
        )
        if record.exc_info:
            text += "\n" + self.formatException(record.exc_info)
        return text


# ── Setup ─────────────────────────────────────────────────────────────────────

def _resolve_log_path() -> Path:
    explicit = os.environ.get("BT_LOG_FILE", "")
    if explicit:
        p = Path(explicit)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    log_dir = Path(os.environ.get("BT_LOG_DIR", str(_DEFAULT_LOG_DIR)))
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / _DEFAULT_FILENAME


def _setup() -> None:
    global _configured
    if _configured:
        return
    _configured = True

    level_name = os.environ.get("BT_LOG_LEVEL", "INFO").upper()
    level      = getattr(logging, level_name, logging.INFO)

    root = logging.getLogger(_ROOT_LOGGER_NAME)
    root.setLevel(level)

    # ── 1. Rotating JSONL file handler ───────────────────────────────────────
    log_path = _resolve_log_path()
    try:
        fh = RotatingFileHandler(
            str(log_path),
            maxBytes=_MAX_BYTES,
            backupCount=_BACKUP_COUNT,
            encoding="utf-8",
        )
        fh.setFormatter(_JsonFormatter())
        fh.setLevel(level)
        root.addHandler(fh)
    except OSError:
        pass   # may not have write permission in some environments

    # ── 2. Coloured stderr handler ───────────────────────────────────────────
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(_PrettyFormatter())
    sh.setLevel(level)
    root.addHandler(sh)

    root.propagate = False


# ── Public API ────────────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the 'basetruth' root.

    Usage::

        from basetruth.logger import get_logger
        log = get_logger(__name__)

        log.info("Scanning document", extra={"path": str(path), "size": 12345})
        log.warning("Low text extraction", extra={"chars": 42})
        log.error("DB save failed", exc_info=True)
    """
    _setup()
    # Ensure name is under the basetruth hierarchy for unified control
    if not name.startswith(_ROOT_LOGGER_NAME):
        name = f"{_ROOT_LOGGER_NAME}.{name}"
    return logging.getLogger(name)


def log_path() -> Optional[Path]:
    """Return the current JSONL log file path (None if logging to file is disabled)."""
    _setup()
    root = logging.getLogger(_ROOT_LOGGER_NAME)
    for h in root.handlers:
        if isinstance(h, RotatingFileHandler):
            return Path(h.baseFilename)
    return None
