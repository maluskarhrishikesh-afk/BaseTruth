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
UI_PORT         = 8501
BROWSER_URL     = f"http://localhost:{UI_PORT}"   # Streamlit dashboard
API_DOCS_URL    = f"http://localhost:{API_PORT}/api/docs"
IMAGE_NAME      = "basetruth:latest"
UI_SERVICE      = "basetruth-ui"
API_SERVICE     = "basetruth-api"
DB_SERVICE      = "db"
MINIO_SERVICE   = "minio"
DAEMON_WAIT_SEC = 60   # how long to wait for Docker daemon after install
UI_WAIT_SEC     = 90   # how long to wait for Streamlit to become healthy
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


def _rebuild_if_needed(root: Path) -> None:
    """Rebuild the Docker image only when Dockerfile or requirements have changed.

    Docker's build cache means unchanged layers are reused — this is fast.
    The image is rebuilt when:
      - requirements.txt was modified after the image was last built
      - Dockerfile was modified after the image was last built
      - The image does not exist yet
    """
    if not _image_exists():
        _build_image(root)
        return
    # Get the image creation timestamp from Docker
    ts_result = subprocess.run(
        ["docker", "image", "inspect", IMAGE_NAME, "--format", "{{.Created}}"],
        capture_output=True, text=True,
    )
    if ts_result.returncode != 0:
        # Can't determine age — rebuild to be safe
        _build_image(root)
        return
    try:
        from datetime import datetime, timezone
        image_created = datetime.fromisoformat(
            ts_result.stdout.strip().replace("Z", "+00:00")
        )
        watch_files = [root / "Dockerfile", root / "requirements.txt", root / "pyproject.toml"]
        for watch in watch_files:
            if not watch.exists():
                continue
            mtime = datetime.fromtimestamp(watch.stat().st_mtime, tz=timezone.utc)
            if mtime > image_created:
                print(f"\n[info] {watch.name} changed after last build — rebuilding image ...")
                _build_image(root)
                return
        print(f"\n[ok] Image {IMAGE_NAME!r} is up-to-date — skipping rebuild.")
    except Exception:  # noqa: BLE001
        # Timestamp comparison failed — just proceed without rebuilding
        pass


def _port_is_up(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _start_services(root: Path) -> None:
    """Start (or restart) both services so they always run the latest code.

    --force-recreate ensures the container is rebuilt from its image even if
    it was already running.  Combined with the  ./src:/app/src  volume mount
    in docker-compose.yml this guarantees the running process always sees the
    latest Python source files without requiring a full image rebuild.
    """
    _step(f"Starting {DB_SERVICE}, {MINIO_SERVICE}, {UI_SERVICE}, and {API_SERVICE} with latest code ...")
    result = subprocess.run(
        [
            "docker", "compose", "up", "-d", "--force-recreate",
            DB_SERVICE, MINIO_SERVICE, UI_SERVICE, API_SERVICE,
        ],
        cwd=str(root),
    )
    if result.returncode != 0:
        print("[ERROR] docker compose up failed.", file=sys.stderr)
        sys.exit(result.returncode)


def _wait_for_port(port: int, label: str, timeout_sec: int) -> bool:
    _step(f"Waiting for {label} on port {port} ...")
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if _port_is_up(port):
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

    # 2. Rebuild image only when Dockerfile / requirements changed (uses cache)
    _rebuild_if_needed(root)

    # 3. Always force-recreate containers so the running process picks up
    #    the latest code from the ./src volume mount.
    _start_services(root)

    # 4. Wait for services to become available
    if not _wait_for_port(UI_PORT, "Streamlit dashboard", UI_WAIT_SEC):
        print(
            f"\n[WARNING] Dashboard did not start on port {UI_PORT} "
            f"within {UI_WAIT_SEC}s.\n"
            "Check logs with:  docker compose logs basetruth-ui",
            file=sys.stderr,
        )
        return 1

    if not _wait_for_port(API_PORT, "REST API", API_WAIT_SEC):
        print(
            f"\n[WARNING] API did not start on port {API_PORT} "
            f"within {API_WAIT_SEC}s.\n"
            "Check logs with:  docker compose logs basetruth-api",
            file=sys.stderr,
        )
        # API failure is non-fatal — dashboard is already up

    # 5. Open browser -> Streamlit dashboard
    print(f"\n[ok] BaseTruth Dashboard  ->  {BROWSER_URL}")
    print(f"[ok] REST API docs        ->  {API_DOCS_URL}")
    try:
        webbrowser.open(BROWSER_URL)
    except OSError:
        pass

    print("\nTo stop all containers:  docker compose down")
    print("To scan via CLI:  docker compose run --rm basetruth-cli scan --input /app/your_data/doc.pdf")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())