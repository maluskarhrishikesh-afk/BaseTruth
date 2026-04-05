"""Run Semgrep against the BaseTruth source tree.

Usage:
    python scripts/run_semgrep.py               # custom rules only
    python scripts/run_semgrep.py --auto        # custom + Semgrep registry
    python scripts/run_semgrep.py --owasp       # OWASP Top-10 rules
    python scripts/run_semgrep.py --fix         # auto-fix where possible
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEMGREP_CONFIG = ROOT / ".semgrep.yml"
SOURCE_DIR = ROOT / "src"


def _check_semgrep() -> None:
    if not shutil.which("semgrep"):
        print(
            "semgrep not found. Install it with:\n"
            "  pip install semgrep\n"
            "  or\n"
            "  brew install semgrep   (macOS)\n"
            "  choco install semgrep  (Windows via Chocolatey)"
        )
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Semgrep for BaseTruth")
    parser.add_argument("--auto", action="store_true", help="Also run semgrep-managed auto ruleset")
    parser.add_argument("--owasp", action="store_true", help="Run OWASP Top-10 ruleset")
    parser.add_argument("--fix", action="store_true", help="Apply auto-fixes where possible")
    parser.add_argument("--json", action="store_true", help="Output findings as JSON")
    args = parser.parse_args()

    _check_semgrep()

    configs: list[str] = [str(SEMGREP_CONFIG)]
    if args.auto:
        configs.append("auto")
    if args.owasp:
        configs.extend([
            "p/owasp-top-ten",
            "p/python",
        ])

    cmd: list[str] = ["semgrep"]
    for cfg in configs:
        cmd += ["--config", cfg]
    cmd += [str(SOURCE_DIR)]

    if args.fix:
        cmd.append("--autofix")
    if args.json:
        cmd += ["--json"]
    else:
        cmd += ["--color"]

    print(f"Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, check=False)  # noqa: S603
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
