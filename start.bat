@echo off
REM BaseTruth — quick-launch batch file
REM Double-click this file, or run start.exe for the compiled version.
REM
REM What this does:
REM   1. Builds the Docker image (skipped if already built)
REM   2. Starts the REST API server on http://localhost:8000
REM   3. Opens the API docs in your browser

setlocal enabledelayedexpansion

echo.
echo ============================================================
echo   BaseTruth -- Document Integrity Platform
echo ============================================================

REM ── Check Docker is installed ─────────────────────────────────
where docker >nul 2>&1
if errorlevel 1 (
    echo.
    echo [ERROR] Docker is not installed or not on PATH.
    echo Install Docker Desktop from: https://www.docker.com/products/docker-desktop/
    echo.
    pause
    exit /b 1
)

REM ── Wait for Docker daemon ────────────────────────────────────
echo.
echo >>> Checking Docker daemon ...
docker info >nul 2>&1
if errorlevel 1 (
    echo Docker daemon is not running. Please start Docker Desktop.
    echo Press any key once Docker Desktop is ready ...
    pause >nul
)

REM ── Build image (only if it doesn't exist yet) ─────────────────
docker image inspect basetruth:latest >nul 2>&1
if errorlevel 1 (
    echo.
    echo >>> Building Docker image -- this takes a few minutes on first run ...
    docker compose build
    if errorlevel 1 (
        echo [ERROR] docker compose build failed.
        pause
        exit /b 1
    )
) else (
    echo.
    echo [ok] Image already exists -- skipping build.
    echo      To force a rebuild: docker compose build
)

REM ── Start API ─────────────────────────────────────────────────
echo.
echo >>> Starting basetruth-api container ...
docker compose up -d basetruth-api
if errorlevel 1 (
    echo [ERROR] docker compose up failed.
    pause
    exit /b 1
)

REM ── Open browser ──────────────────────────────────────────────
echo.
echo [ok] BaseTruth API is live ^-^> http://localhost:8000/docs
timeout /t 3 /nobreak >nul
start "" http://localhost:8000/docs

echo.
echo To stop:  docker compose down
echo To scan:  docker compose run --rm basetruth-cli scan --input /app/your_data/doc.pdf
echo.
