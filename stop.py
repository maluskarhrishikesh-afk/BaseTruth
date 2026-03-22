from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def main() -> int:
    root = _repo_root()
    pid_file = root / ".runtime" / "basetruth-ui.pid"
    if not pid_file.exists():
        return 0

    try:
        pid = pid_file.read_text(encoding="utf-8").strip()
    except OSError:
        pid = ""
    if pid:
        subprocess.run(["taskkill", "/PID", pid, "/T", "/F"], check=False, capture_output=True, text=True)
    try:
        pid_file.unlink()
    except OSError:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())