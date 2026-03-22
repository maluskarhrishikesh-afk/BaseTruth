"""BaseTruth FastAPI REST layer.

Run with::

    uvicorn basetruth.api:app --host 0.0.0.0 --port 8502

Or from baseTruth root::

    python -m basetruth.api

Install the api extra first::

    pip install basetruth[api]

"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from fastapi import FastAPI, File, HTTPException, Query, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field

    _FASTAPI_AVAILABLE = True

    # ── Request / response models (must be at module level so Pydantic v2
    # can resolve forward references when generating the OpenAPI schema) ──────

    class ScanPathRequest(BaseModel):
        path: str = Field(..., description="Absolute path to the document or structured JSON to scan.")

    class UpdateCaseRequest(BaseModel):
        status: Optional[str] = Field(None, description="new | triage | investigating | pending_client | closed")
        disposition: Optional[str] = Field(None, description="open | monitor | escalate | cleared | fraud_confirmed")
        priority: Optional[str] = Field(None, description="low | normal | high | critical")
        assignee: Optional[str] = Field(None, description="Investigator user name.")
        labels: Optional[List[str]] = Field(None, description="Free-form labels for routing and triage.")
        note_text: str = Field("", description="Text of the new note to append, if any.")
        note_author: str = Field("api", description="Author name for the note.")

except ImportError:
    _FASTAPI_AVAILABLE = False


_DEFAULT_ARTIFACT_ROOT = Path("artifacts")


def _service(artifact_root: str | Path | None = None) -> Any:
    from basetruth.service import BaseTruthService

    return BaseTruthService(artifact_root or _DEFAULT_ARTIFACT_ROOT)


def create_app(artifact_root: str | Path | None = None) -> Any:
    """Create and return the BaseTruth FastAPI application."""
    if not _FASTAPI_AVAILABLE:
        raise ImportError(
            "FastAPI is required for the API server. "
            "Install the BaseTruth api extra: pip install 'basetruth[api]'"
        )

    app = FastAPI(
        title="BaseTruth API",
        description=(
            "Explainable document integrity and fraud detection REST API. "
            "Scan documents, browse cases, update investigator workflows, "
            "and retrieve reports programmatically."
        ),
        version="0.1.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    svc = _service(artifact_root)

    # -------------------------------------------------------------------------
    # Endpoints
    # -------------------------------------------------------------------------

    @app.get("/api/v1/health", tags=["System"])
    def health() -> Dict[str, Any]:
        """Return product and service health information."""
        return {
            "status": "ok",
            "product": "BaseTruth",
            "version": "0.1.0",
            "artifact_root": str(svc.artifact_root),
        }

    @app.post("/api/v1/scan", tags=["Scan"])
    def scan_by_path(request: ScanPathRequest) -> Dict[str, Any]:
        """Scan a document at an existing path on the server file system."""
        path = Path(request.path)
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {request.path}")
        try:
            return svc.scan_document(path)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/api/v1/scan/upload", tags=["Scan"])
    async def scan_upload(file: UploadFile = File(...)) -> Dict[str, Any]:
        """Upload a document and scan it in one request."""
        import tempfile

        suffix = Path(file.filename or "upload").suffix or ".pdf"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = Path(tmp.name)
        try:
            return svc.scan_document(tmp_path)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass

    @app.get("/api/v1/reports", tags=["Reports"])
    def list_reports(
        kind: Optional[str] = Query(None, description="Filter by kind: verification | comparison"),
        risk_level: Optional[str] = Query(None, description="Filter by risk level: high | medium | low"),
    ) -> List[Dict[str, Any]]:
        """List all reports, optionally filtered by kind and risk level."""
        reports = svc.list_reports()
        if kind:
            reports = [r for r in reports if r.get("kind") == kind]
        if risk_level:
            reports = [r for r in reports if r.get("risk_level") == risk_level]
        return reports

    @app.get("/api/v1/cases", tags=["Cases"])
    def list_cases(
        status: Optional[str] = Query(None, description="Filter by workflow status."),
        priority: Optional[str] = Query(None, description="Filter by priority."),
        disposition: Optional[str] = Query(None, description="Filter by disposition."),
    ) -> List[Dict[str, Any]]:
        """List all cases, optionally filtered by workflow state."""
        cases = svc.list_cases()
        if status:
            cases = [c for c in cases if c.get("status") == status]
        if priority:
            cases = [c for c in cases if c.get("priority") == priority]
        if disposition:
            cases = [c for c in cases if c.get("disposition") == disposition]
        return cases

    @app.get("/api/v1/cases/{case_key}", tags=["Cases"])
    def get_case(case_key: str) -> Dict[str, Any]:
        """Get full case detail including workflow state and all linked reports."""
        try:
            return svc.get_case_detail(case_key)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Case not found: {case_key}")

    @app.patch("/api/v1/cases/{case_key}", tags=["Cases"])
    def update_case(case_key: str, request: UpdateCaseRequest) -> Dict[str, Any]:
        """Update the workflow state of a case, optionally appending a note."""
        return svc.update_case(
            case_key,
            status=request.status,
            disposition=request.disposition,
            priority=request.priority,
            assignee=request.assignee,
            labels=request.labels,
            note_text=request.note_text,
            note_author=request.note_author,
        )

    return app


# ---------------------------------------------------------------------------
# Module-level application instance for uvicorn / importability.
# Gracefully sets app=None when FastAPI is not installed.
# ---------------------------------------------------------------------------

try:
    app = create_app()
except ImportError:
    app = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# CLI entry for running the API server directly.
# ---------------------------------------------------------------------------

def _serve(host: str = "0.0.0.0", port: int = 8502, artifact_root: str | None = None) -> None:  # pragma: no cover
    try:
        import uvicorn  # type: ignore
    except ImportError:
        print(
            "uvicorn is required to run the BaseTruth API server. "
            "Install with: pip install 'basetruth[api]'",
            file=sys.stderr,
        )
        sys.exit(1)

    _app = create_app(artifact_root)
    uvicorn.run(_app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(prog="basetruth.api", description="BaseTruth API server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8502)
    parser.add_argument("--artifact-root", default=None)
    args = parser.parse_args()
    _serve(args.host, args.port, args.artifact_root)
