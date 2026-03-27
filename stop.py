from __future__ import annotations

"""
BaseTruth stopper — stop.py
============================
Double-click  stop.exe  (or  python stop.py) to:

  1. Run  docker compose down --remove-orphans
     This stops every BaseTruth container and removes them.
     Data volumes (database, artifacts) are kept intact.

To also wipe the database volume:  docker compose down -v
"""

import subprocess
import sys
from pathlib import Path


def _step(msg: str) -> None:
    print(f"\n>>> {msg}", flush=True)


def _repo_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def main() -> int:
    root = _repo_root()

    print("=" * 60)
    print("  BaseTruth -- Stopping all services")
    print("=" * 60)

    _step("Stopping and removing all BaseTruth containers ...")
    result = subprocess.run(
        ["docker", "compose", "down", "--remove-orphans"],
        cwd=str(root),
    )

    if result.returncode == 0:
        print("\n[ok] All BaseTruth containers stopped and removed.")
        print("     Your artifacts and database data are preserved.")
        print("     Run start.exe (or start.py) to start again.")
    else:
        print(
            "\n[WARNING] docker compose down returned a non-zero exit code.\n"
            "Some containers may still be running.\n"
            "You can force-stop with:  docker compose kill",
            file=sys.stderr,
        )

    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())