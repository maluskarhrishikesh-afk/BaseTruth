from __future__ import annotations

"""
BaseTruth launcher — start.py
==============================
Double-click  start.exe  (or  python start.py) to:

  1. Verify Docker Desktop is running (install it via winget if missing).
  2. Build the BaseTruth Docker image when no image exists yet.
  3. Start the REST-API container  (basetruth-api  ->  http://localhost:8000).
  4. Open a browser tab to the interactive API docs.

Scan documents via the API (no separate CLI step needed at startup):
  POST  http://localhost:8000/scan
  GET   http://localhost:8000/health

To run a one-shot CLI scan from the command line:
  docker compose run --rm basetruth-cli scan --input /app/your_data/doc.pdf
"""

import platform
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
API_PORT        = 8000
BROWSER_URL     = f"http://localhost:{API_PORT}/docs"
IMAGE_NAME      = "basetruth:latest"
API_SERVICE     = "basetruth-api"
DAEMON_WAIT_SEC = 60   # how long to wait for Docker daemon after install
API_WAIT_SEC    = 60   # how long to wait for the API to become healthy


def _step(msg: str) -> None:
    print(f"\n>>> {msg}", flush=True)


def _repo_root() -> Path:
    """Return the directory that contains docker-compose.yml."""
    if getattr(sys, "frozen", False):
        # Running as compiled exe — exe sits next to docker-compose.yml
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


# ── Docker availability ───────────────────────────────────────────────────────

def _docker_available() -> bool:
    try:
        result = subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _docker_daemon_running() -> bool:
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _install_docker_windows() -> None:
    """Attempt a Docker Desktop install via winget (Windows only)."""
    _step("Docker not found — attempting install via winget ...")
    result = subprocess.run(
        [
            "winget", "install",
            "--id", "Docker.DockerDesktop",
            "--accept-package-agreements",
            "--accept-source-agreements",
            "--silent",
        ],
        text=True,
    )
    if result.returncode != 0:
        print(
            "\n[ERROR] winget install failed.\n"
            "Please install Docker Desktop manually:\n"
            "  https://www.docker.com/products/docker-desktop/\n"
            "Then re-run this launcher.",
            file=sys.stderr,
        )
        sys.exit(1)
    print("Docker Desktop installed.")
    print("Please start Docker Desktop from the Start Menu, then press Enter ...")
    input()


def _ensure_docker(root: Path) -> None:
    if not _docker_available():
        if platform.system() == "Windows":
            _install_docker_windows()
        else:
            print(
                "[ERROR] Docker is not installed.\n"
                "Install it from https://docs.docker.com/get-docker/",
                file=sys.stderr,
            )
            sys.exit(1)

    if not _docker_daemon_running():
        _step("Waiting for Docker daemon to start ...")
        deadline = time.time() + DAEMON_WAIT_SEC
        while time.time() < deadline:
            if _docker_daemon_running():
                break
            time.sleep(2)
        else:
            print(
                f"\n[ERROR] Docker daemon did not start within {DAEMON_WAIT_SEC}s.\n"
                "Please start Docker Desktop and try again.",
                file=sys.stderr,
            )
            sys.exit(1)
        print("Docker daemon is ready.")


# ── Image & service management ────────────────────────────────────────────────

def _image_exists() -> bool:
    result = subprocess.run(
        ["docker", "image", "inspect", IMAGE_NAME],
        capture_output=True,
    )
    return result.returncode == 0


def _build_image(root: Path) -> None:
    _step(f"Building Docker image ({IMAGE_NAME}) — this takes a few minutes on first run ...")
    result = subprocess.run(
        ["docker", "compose", "build"],
        cwd=str(root),
    )
    if result.returncode != 0:
        print("[ERROR] docker compose build failed.", file=sys.stderr)
        sys.exit(result.returncode)
    print("Build complete.")


def _api_is_up() -> bool:
    """Check if the API container is already serving on API_PORT."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", API_PORT)) == 0


def _start_api(root: Path) -> None:
    _step(f"Starting {API_SERVICE} container ...")
    result = subprocess.run(
        ["docker", "compose", "up", "-d", API_SERVICE],
        cwd=str(root),
    )
    if result.returncode != 0:
        print(f"[ERROR] docker compose up {API_SERVICE} failed.", file=sys.stderr)
        sys.exit(result.returncode)


def _wait_for_api() -> bool:
    _step(f"Waiting for API on port {API_PORT} ...")
    deadline = time.time() + API_WAIT_SEC
    while time.time() < deadline:
        if _api_is_up():
            return True
        time.sleep(1)
    return False


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> int:
    root = _repo_root()

    print("=" * 60)
    print("  BaseTruth -- Document Integrity Platform")
    print("=" * 60)

    # 1. Docker present and daemon running
    _ensure_docker(root)

    # 2. Build image only when it does not exist yet
    if _image_exists():
        print(f"\n[ok] Image {IMAGE_NAME!r} already exists -- skipping build.")
        print("     To force a rebuild:  docker compose build")
    else:
        _build_image(root)

    # 3. Start API (skip if already listening)
    if _api_is_up():
        print(f"\n[ok] API already running on port {API_PORT}.")
    else:
        _start_api(root)
        if not _wait_for_api():
            print(
                f"\n[WARNING] API did not become reachable on port {API_PORT} "
                f"within {API_WAIT_SEC}s.\n"
                "Check logs with:  docker compose logs basetruth-api",
                file=sys.stderr,
            )
            return 1

    # 4. Open browser
    print(f"\n[ok] BaseTruth API is live -> {BROWSER_URL}")
    try:
        webbrowser.open(BROWSER_URL)
    except OSError:
        pass

    print("\nTo stop all containers:  docker compose down")
    print("To scan a document via CLI:")
    print("  docker compose run --rm basetruth-cli scan --input /app/your_data/doc.pdf")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())