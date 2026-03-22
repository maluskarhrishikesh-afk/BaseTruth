from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple


LITEPARSE_COMMAND_ENV = "LITEPARSE_COMMAND"


def resolve_liteparse_command() -> Tuple[Optional[List[str]], Optional[str]]:
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
