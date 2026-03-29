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

    # ── Video KYC endpoints ─────────────────────────────────────────
    
    import uuid
    from fastapi import WebSocket, WebSocketDisconnect
    from basetruth.vision.video_kyc import VideoKYCProcessor
    import numpy as np
    
    # Global memory cache for in-flight KYC sessions
    _kyc_sessions: Dict[str, Any] = {}

    class KYCSessionRequest(BaseModel):
        embedding: List[float]

    @app.post("/api/v1/kyc/session", tags=["VideoKYC"])
    def create_kyc_session(req: KYCSessionRequest) -> Dict[str, Any]:
        """Creates a secure session referencing a parsed face embedding."""
        session_id = str(uuid.uuid4())
        _kyc_sessions[session_id] = np.array(req.embedding, dtype=np.float32)
        return {"session_id": session_id}

    @app.websocket("/ws/video_kyc/{session_id}")
    async def websocket_video_kyc(websocket: WebSocket, session_id: str):
        """Standard HTTP/TCP WebSocket for 15 FPS Video processing (Bypasses Docker UDP)."""
        await websocket.accept()
        if session_id not in _kyc_sessions:
            await websocket.close(code=1008)
            return
            
        processor = VideoKYCProcessor()
        processor.set_reference_embedding(_kyc_sessions[session_id])
        
        try:
            while True:
                # Receive base64 frame string from JS client
                b64_str = await websocket.receive_text()
                
                # Send frame into OpenCV + ArcFace pipeline
                out_b64 = processor.process_base64_frame(b64_str)
                
                # Echo annotated frame back to the browser
                if out_b64:
                    await websocket.send_text(out_b64)
        except WebSocketDisconnect:
            pass
        finally:
            if session_id in _kyc_sessions:
                del _kyc_sessions[session_id]

    # ── Entity registry endpoints ─────────────────────────────────────────

    @app.get("/api/v1/entities", tags=["Entities"])
    def list_entities(
        q: str = Query("", description="Search term — name, PAN, Aadhaar, email, or phone."),
        field: str = Query("all", description="Field to search: all | name | pan | aadhar | email | phone"),
        limit: int = Query(100, ge=1, le=1000, description="Maximum rows to return."),
    ) -> List[Dict[str, Any]]:
        """Search the entity registry.

        Returns the most-recent entities when no query is supplied.
        Each result includes scan count and latest risk level for quick triage.
        """
        try:
            from basetruth.store import search_entities
            return search_entities(query=q, search_field=field, limit=limit)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"DB unavailable: {exc}") from exc

    @app.get("/api/v1/entities/{entity_ref}", tags=["Entities"])
    def get_entity(entity_ref: str) -> Dict[str, Any]:
        """Return full detail for one entity including all scans summary."""
        try:
            from basetruth.store import get_entity_scans, search_entities
            matches = search_entities(query=entity_ref, search_field="all", limit=1)
            if not matches:
                raise HTTPException(status_code=404, detail=f"Entity not found: {entity_ref}")
            entity = matches[0]
            entity["scans"] = get_entity_scans(entity_ref)
            return entity
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"DB unavailable: {exc}") from exc

    @app.get("/api/v1/entities/{entity_ref}/scans", tags=["Entities"])
    def list_entity_scans(entity_ref: str) -> List[Dict[str, Any]]:
        """List all document scans for a specific entity (most-recent first).

        Each item includes the full JSON report so analysts can review the
        signals that led to a flag without needing to access the filesystem.
        """
        try:
            from basetruth.store import get_entity_scans
            scans = get_entity_scans(entity_ref)
            if not scans:
                # Could be entity not found or just no scans yet — disambiguate
                from basetruth.store import search_entities
                if not search_entities(query=entity_ref, search_field="all", limit=1):
                    raise HTTPException(status_code=404, detail=f"Entity not found: {entity_ref}")
            return scans
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"DB unavailable: {exc}") from exc

    @app.get("/api/v1/scans/{scan_id}/report.pdf", tags=["Reports"])
    def download_scan_pdf(scan_id: int) -> Any:
        """Download the PDF audit report for a specific scan.

        Returns the PDF binary so auditors can save it locally or attach it
        to a case without needing access to the server filesystem.

        Use this endpoint when explaining a fraud flag to an auditor:
        look up the entity, find the scan ID, call this endpoint, share the PDF.
        """
        try:
            from fastapi.responses import Response
            from basetruth.store import get_scan_pdf
            pdf_bytes = get_scan_pdf(scan_id)
            if not pdf_bytes:
                raise HTTPException(
                    status_code=404,
                    detail=f"PDF report not found for scan {scan_id}. "
                           "The scan may not have generated a PDF, or the DB is unavailable.",
                )
            return Response(
                content=pdf_bytes,
                media_type="application/pdf",
                headers={
                    "Content-Disposition": f'attachment; filename="scan_{scan_id}_report.pdf"',
                    "Content-Length": str(len(pdf_bytes)),
                },
            )
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"DB unavailable: {exc}") from exc

    @app.get("/api/v1/scans/recent", tags=["Reports"])
    def list_recent_scans(
        limit: int = Query(50, ge=1, le=500, description="Number of most-recent scans to return."),
    ) -> List[Dict[str, Any]]:
        """Return the most-recent scans across all entities with their risk levels.

        Useful for a fraud-monitoring dashboard.
        """
        try:
            from basetruth.store import list_recent_scans as _list
            return _list(limit=limit)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"DB unavailable: {exc}") from exc

    @app.get("/api/v1/db/stats", tags=["System"])
    def db_stats() -> Dict[str, Any]:
        """Return entity, scan, and high-risk counts from the database."""
        try:
            from basetruth.store import db_stats as _stats
            return _stats()
        except Exception as exc:
            return {"error": str(exc), "entities": 0, "scans": 0, "high_risk": 0}

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
