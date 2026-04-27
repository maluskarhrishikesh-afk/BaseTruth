from __future__ import annotations

"""
BaseTruth ML Pipeline Launcher — train_model.py
================================================
Double-click  train_model.exe  (or  python train_model.py) to run the
complete ML training pipeline:

  Step 1 — Collect forensic features from ORIGINAL images  (label = 0)
  Step 2 — Collect forensic features from TAMPERED images  (label = 1)
  Step 3 — Collect forensic features from ORIGINAL PDFs    (label = 0)
  Step 4 — Collect forensic features from TAMPERED PDFs    (label = 1)
  Step 5 — Train XGBoost Image Fraud Classifier
  Step 6 — Train XGBoost PDF Fraud Classifier

Input data is read from tests/sample/ inside the repo.
Output models are saved to data/ inside the repo.

This launcher does NOT bundle any ML libraries — it finds the .venv Python
inside the repo and delegates to scripts/run_ml_pipeline.py which has access
to all installed dependencies.
"""

import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    """Return the directory that contains this file (or the running exe).

    When frozen by PyInstaller the exe lives next to docker-compose.yml,
    train_model.py, etc. — so we use the exe path instead of __file__.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def _find_python(root: Path) -> Path | None:
    """Locate the Python interpreter to use.

    Priority order:
      1. .venv\\Scripts\\python.exe  (Windows virtualenv created during setup)
      2. .venv\\bin\\python           (Linux/macOS virtualenv)
      3. The interpreter running this script right now (fallback)
    """
    candidates = [
        root / ".venv" / "Scripts" / "python.exe",   # Windows
        root / ".venv" / "bin" / "python",            # Linux / macOS
    ]
    for c in candidates:
        if c.is_file():
            return c

    # Fall back to whatever Python is currently executing this launcher
    return Path(sys.executable)


def _banner(msg: str) -> None:
    print("\n" + "=" * 64, flush=True)
    print(f"  {msg}", flush=True)
    print("=" * 64, flush=True)


def main() -> int:
    root = _repo_root()
    pipeline_script = root / "fraud_model" / "scripts" / "run_ml_pipeline.py"

    _banner("BaseTruth — ML Training Pipeline Launcher")
    print(f"  Repo root : {root}", flush=True)

    # Validate the pipeline script exists
    if not pipeline_script.is_file():
        print(
            f"\n  ERROR: pipeline script not found:\n    {pipeline_script}\n"
            "  Make sure train_model.exe is in the BaseTruth repo root.",
            flush=True,
        )
        _pause_on_exit(1)
        return 1

    # Find the right Python (must have all dependencies installed)
    python_exe = _find_python(root)
    print(f"  Python    : {python_exe}", flush=True)
    print(f"  Script    : {pipeline_script}", flush=True)

    # Run the pipeline; inherit stdout/stderr so the user sees everything live
    print("\n  Starting pipeline — this may take 30–60 minutes …\n", flush=True)
    result = subprocess.run(
        [str(python_exe), str(pipeline_script)],
        cwd=str(root),   # ensures relative paths inside the script resolve correctly
    )

    exit_code = result.returncode
    if exit_code == 0:
        _banner("SUCCESS — All models trained and saved")
    else:
        _banner("DONE (with warnings — check output above)")

    _pause_on_exit(exit_code)
    return exit_code


def _pause_on_exit(code: int) -> None:
    """Keep the console window open so the user can read the output.

    When double-clicking an exe the window vanishes the instant the process
    ends unless we wait for a keypress.  When running inside an existing
    terminal (IDE, PowerShell, cmd) the input() call is still harmless.
    """
    print("\n" + "-" * 64, flush=True)
    if code == 0:
        print("  Pipeline finished successfully.", flush=True)
    else:
        print("  Pipeline finished with some warnings — see above.", flush=True)
    print("  Press ENTER to close this window …", flush=True)
    try:
        input()
    except (EOFError, KeyboardInterrupt):
        pass


if __name__ == "__main__":
    sys.exit(main())
