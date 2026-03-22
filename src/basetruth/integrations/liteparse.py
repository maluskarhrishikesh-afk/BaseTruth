from __future__ import annotations

"""
LiteParse integration -- subprocess wrapper for the @llamaindex/liteparse Node.js CLI.

LiteParse converts PDFs and images to structured JSON (text + layout data per page)
using a combination of pdfjs-dist, Tesseract OCR, and ImageMagick (for rasterising
PDF pages to images before OCR).

Command resolution order
------------------------
1. LITEPARSE_COMMAND environment variable -- set this to an absolute path or full
   command string if you have a custom installation (e.g. LITEPARSE_COMMAND=lit).
2. 'lit' binary found on PATH (installed globally via 'npm i -g @llamaindex/liteparse').
3. 'npx' found on PATH -- runs '@llamaindex/liteparse' via npx, downloading it
   automatically on first use (requires internet access and npm).
4. None -- LiteParse is unavailable; callers should handle this gracefully.

Windows note
------------
ImageMagick must also be installed on Windows for LiteParse to process PDFs.
Without it, exit code 4 and the message "Invalid Parameter - -density" are returned.
BaseTruth's service layer detects this and falls back to pypdf plain-text extraction;
see integrations/pdf.py for the fallback implementations.

Public API
----------
  resolve_liteparse_command()   -- returns (command_list, source_tag) or (None, None)
  check_liteparse_available()   -- returns a status dict with "available" key
  parse_document_to_json()      -- runs the parse, returns a status dict
"""

import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple


# Environment variable that allows the caller to override the LiteParse command.
# When set, its value is split into a shell argument list and used verbatim.
LITEPARSE_COMMAND_ENV = "LITEPARSE_COMMAND"


def resolve_liteparse_command() -> Tuple[Optional[List[str]], Optional[str]]:
    """Determine the command prefix needed to invoke LiteParse.

    Returns a (command_list, source_tag) tuple where source_tag is one of
    'env', 'lit', or 'npx', indicating how the command was found.
    Returns (None, None) when no suitable command can be located.
    """
    env_command = str(os.environ.get(LITEPARSE_COMMAND_ENV, "") or "").strip()
    if env_command:
        return shlex.split(env_command, posix=os.name != "nt"), "env"

    lit_path = shutil.which("lit")
    if lit_path:
        return [lit_path], "lit"

    npx_path = shutil.which("npx")
    if npx_path:
        return [npx_path, "--yes", "@llamaindex/liteparse"], "npx"

    return None, None


def check_liteparse_available() -> dict:
    """Return a status dict indicating whether LiteParse can be invoked.

    The dict always has an 'available' key (bool).  When True it also
    includes 'command' (list) and 'command_source' ('env'|'lit'|'npx').
    When False it includes a human-readable 'message' explaining what to do.
    """
    command, source = resolve_liteparse_command()
    if not command:
        return {
            "available": False,
            "message": "LiteParse is unavailable. Install Node.js and run 'npm i -g @llamaindex/liteparse'.",
        }
    return {
        "available": True,
        "command": command,
        "command_source": source,
    }


def parse_document_to_json(document_path: Path, output_path: Path, timeout_seconds: int = 600) -> dict:
    """Run LiteParse on *document_path* and write the resulting JSON to *output_path*.

    Returns a status dict with 'status' == 'success' on success, or
    'status' == 'error' with a 'message' key describing what went wrong.
    The 'command' and 'command_source' keys are always present on error to
    aid debugging.  A timeout_seconds of 600 is used by default; increase
    this for very large PDFs on slow machines.
    """
    command_prefix, source = resolve_liteparse_command()
    if not command_prefix:
        return {
            "status": "error",
            "message": "LiteParse is unavailable. Install Node.js and run 'npm i -g @llamaindex/liteparse'.",
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    args = [
        *command_prefix,
        "parse",
        str(document_path),
        "--format",
        "json",
        "-o",
        str(output_path),
        "-q",
    ]
    completed = subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=max(timeout_seconds, 1),
        check=False,
    )
    if completed.returncode != 0:
        return {
            "status": "error",
            "message": completed.stderr.strip() or completed.stdout.strip() or "LiteParse failed.",
            "command": args,
            "command_source": source,
        }

    return {
        "status": "success",
        "output_path": str(output_path),
        "command": args,
        "command_source": source,
    }
