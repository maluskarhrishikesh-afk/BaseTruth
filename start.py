from __future__ import annotations

import os
import subprocess
import sys
import webbrowser
from pathlib import Path


APP_PORT = 8501


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _runtime_dir(root: Path) -> Path:
    target = root / ".runtime"
    target.mkdir(parents=True, exist_ok=True)
    return target


def _python_executable(root: Path) -> str:
    candidates = [
        root / ".venv" / "Scripts" / "python.exe",
        root / ".venv" / "bin" / "python",
        Path(sys.executable),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return sys.executable


def main() -> int:
    root = _repo_root()
    runtime_dir = _runtime_dir(root)
    pid_file = runtime_dir / "basetruth-ui.pid"
    log_file = runtime_dir / "basetruth-ui.log"
    python_exe = _python_executable(root)
    app_path = root / "src" / "basetruth" / "ui" / "app.py"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")

    if pid_file.exists():
        try:
            existing_pid = pid_file.read_text(encoding="utf-8").strip()
            if existing_pid:
                return 0
        except OSError:
            pass

    with log_file.open("a", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            [python_exe, "-m", "streamlit", "run", str(app_path), "--server.port", str(APP_PORT)],
            cwd=str(root),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        )
    pid_file.write_text(str(process.pid), encoding="utf-8")
    try:
        webbrowser.open(f"http://localhost:{APP_PORT}")
    except OSError:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())