from __future__ import annotations

"""
BaseTruth builder — build.py
==============================
Double-click  build.exe  (or  python build.py) to:

  1. Verify Docker Desktop is running.
  2. Run  docker compose build  to rebuild all Docker images.
  3. (Optional) Compile start.py, stop.py, and build.py into standalone .exe
     files using PyInstaller (only when PyInstaller is installed).

After building, use  start.exe  (or start.py) to launch the application.
"""

import subprocess
import sys
import time
import platform
from pathlib import Path


def _step(msg: str) -> None:
    print(f"\n>>> {msg}", flush=True)


def _repo_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def _docker_available() -> bool:
    try:
        result = subprocess.run(
            ["docker", "--version"],
            capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _docker_daemon_running() -> bool:
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def main() -> int:
    root = _repo_root()

    print("=" * 60)
    print("  BaseTruth -- Build & Deploy")
    print("=" * 60)

    # ── 1. Check Docker ───────────────────────────────────────────────────
    if not _docker_available():
        print(
            "\n[ERROR] Docker is not installed.\n"
            "Install Docker Desktop from https://www.docker.com/products/docker-desktop/",
            file=sys.stderr,
        )
        sys.exit(1)

    if not _docker_daemon_running():
        print(
            "\n[ERROR] Docker daemon is not running.\n"
            "Please start Docker Desktop and try again.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("[ok] Docker is available and running.")

    # ── 2. Build Docker images ────────────────────────────────────────────
    _step("Building Docker images (docker compose build) ...")
    result = subprocess.run(
        ["docker", "compose", "build"],
        cwd=str(root),
    )
    if result.returncode != 0:
        print("[ERROR] docker compose build failed.", file=sys.stderr)
        sys.exit(result.returncode)
    print("\n[ok] Docker images built successfully.")

    # ── 3. Compile .exe files (optional — requires PyInstaller) ──────────
    if platform.system() == "Windows":
        try:
            import PyInstaller  # type: ignore  # noqa: F401
            pyinstaller_available = True
        except ImportError:
            pyinstaller_available = False

        if pyinstaller_available:
            _step("Compiling .exe launchers with PyInstaller ...")
            for script in ["start.py", "stop.py", "build.py"]:
                script_path = root / script
                if not script_path.exists():
                    print(f"  [skip] {script} not found.")
                    continue
                print(f"  Building {script} → {script.replace('.py', '.exe')} ...")
                result = subprocess.run(
                    [
                        sys.executable, "-m", "PyInstaller",
                        "--onefile", "--console",
                        "--name", script.replace(".py", ""),
                        "--distpath", str(root),
                        "--workpath", str(root / "build"),
                        "--specpath", str(root),
                        "-y",
                        str(script_path),
                    ],
                    cwd=str(root),
                )
                if result.returncode != 0:
                    print(f"  [WARNING] Failed to compile {script}.", file=sys.stderr)
                else:
                    print(f"  [ok] {script.replace('.py', '.exe')} compiled.")
        else:
            print(
                "\n[info] PyInstaller not installed — skipping .exe compilation.\n"
                "       To compile launchers, run:  pip install pyinstaller\n"
                "       Then re-run build.py."
            )

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Build Complete!")
    print("=" * 60)
    print(f"\n  Docker images rebuilt from: {root}")
    print("\n  Next steps:")
    print("    • Run  start.exe  (or python start.py)  to launch the application.")
    print("    • Run  stop.exe   (or python stop.py)   to stop all containers.")
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
