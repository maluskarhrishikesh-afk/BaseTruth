from __future__ import annotations

import csv
import io
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List


__all__ = ["DatasourceConfig", "DatasourceRegistry", "SUPPORTED_DOCUMENT_EXTENSIONS"]


SUPPORTED_DOCUMENT_EXTENSIONS = {
    ".pdf",
    ".json",
    ".png",
    ".jpg",
    ".jpeg",
    ".tiff",
    ".bmp",
    ".webp",
    ".docx",
    ".xlsx",
    ".pptx",
    ".txt",
}


def _slugify(value: str) -> str:
    chars = []
    for char in str(value or ""):
        if char.isalnum() or char in {"-", "_"}:
            chars.append(char)
        else:
            chars.append("_")
    slug = "".join(chars).strip("._-")
    return slug or "source"


@dataclass
class DatasourceConfig:
    name: str
    kind: str
    path: str
    recursive: bool = True
    extensions: List[str] | None = None
    enabled: bool = True
    description: str = ""
    settings: Dict[str, Any] | None = None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["extensions"] = list(self.extensions or [])
        payload["settings"] = dict(self.settings or {})
        return payload


class DatasourceRegistry:
    def __init__(self, artifact_root: Path | str) -> None:
        self.artifact_root = Path(artifact_root)
        self.config_dir = self.artifact_root / "config"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.config_path = self.config_dir / "datasources.json"

    def list_sources(self) -> List[DatasourceConfig]:
        if not self.config_path.exists():
            return []
        payload = json.loads(self.config_path.read_text(encoding="utf-8"))
        items = payload.get("sources", []) if isinstance(payload, dict) else []
        sources: List[DatasourceConfig] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            sources.append(
                DatasourceConfig(
                    name=str(item.get("name", "")),
                    kind=str(item.get("kind", "folder")),
                    path=str(item.get("path", "")),
                    recursive=bool(item.get("recursive", True)),
                    extensions=[str(ext) for ext in item.get("extensions", [])],
                    enabled=bool(item.get("enabled", True)),
                    description=str(item.get("description", "")),
                    settings=dict(item.get("settings", {}) or {}),
                )
            )
        return sources

    def save_sources(self, sources: Iterable[DatasourceConfig]) -> None:
        payload = {
            "schema_version": 1,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "sources": [source.to_dict() for source in sources],
        }
        self.config_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def upsert_source(self, source: DatasourceConfig) -> None:
        existing = self.list_sources()
        by_name = {_slugify(item.name): item for item in existing}
        by_name[_slugify(source.name)] = source
        self.save_sources(by_name.values())

    def get_source(self, name: str) -> DatasourceConfig:
        target = _slugify(name)
        for source in self.list_sources():
            if _slugify(source.name) == target:
                return source
        raise KeyError(name)

    def _normalize_extensions(self, extensions: List[str] | None) -> set[str]:
        if not extensions:
            return set(SUPPORTED_DOCUMENT_EXTENSIONS)
        normalized = set()
        for ext in extensions:
            ext_text = str(ext).strip().lower()
            if not ext_text:
                continue
            normalized.add(ext_text if ext_text.startswith(".") else f".{ext_text}")
        return normalized or set(SUPPORTED_DOCUMENT_EXTENSIONS)

    def _parse_s3_path(self, path: str) -> tuple[str, str]:
        raw = str(path or "").strip()
        if raw.startswith("s3://"):
            raw = raw[5:]
        bucket, _, prefix = raw.partition("/")
        if not bucket:
            raise ValueError("S3 datasource path must include a bucket name.")
        return bucket, prefix.rstrip("/")

    def _parse_sharepoint_path(self, path: str) -> tuple[str, str, str]:
        parts = [part.strip() for part in str(path or "").split("|")]
        if len(parts) != 3 or not all(parts):
            raise ValueError("SharePoint datasource path must be 'site_id|drive_id|folder_path'.")
        return parts[0], parts[1], parts[2].strip("/")

    def build_path_from_settings(self, kind: str, settings: Dict[str, Any] | None, fallback_path: str = "") -> str:
        config = dict(settings or {})
        if kind == "s3":
            bucket = str(config.get("bucket", "")).strip()
            prefix = str(config.get("prefix", "")).strip().strip("/")
            if bucket:
                return f"s3://{bucket}/{prefix}".rstrip("/")
        if kind == "google_drive":
            folder_id = str(config.get("folder_id", "")).strip()
            if folder_id:
                return folder_id
        if kind == "sharepoint":
            site_id = str(config.get("site_id", "")).strip()
            drive_id = str(config.get("drive_id", "")).strip()
            folder_path = str(config.get("folder_path", "")).strip().strip("/")
            if site_id and drive_id:
                return f"{site_id}|{drive_id}|{folder_path}"
        return fallback_path

    def _safe_write_bytes(self, destination: Path, data: bytes) -> str:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        return str(destination)

    def _sync_folder_source(self, source: DatasourceConfig, target_dir: Path, extensions: set[str]) -> tuple[List[str], List[str]]:
        source_path = Path(source.path)
        copied_files: List[str] = []
        skipped_files: List[str] = []
        iterator = source_path.rglob("*") if source.recursive else source_path.glob("*")
        for candidate in iterator:
            if not candidate.is_file() or candidate.suffix.lower() not in extensions:
                continue
            relative_path = candidate.relative_to(source_path)
            destination = target_dir / relative_path
            if destination.exists():
                skipped_files.append(str(candidate))
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate, destination)
            copied_files.append(str(destination))
        return copied_files, skipped_files

    def _sync_manifest_source(self, source: DatasourceConfig, target_dir: Path, extensions: set[str]) -> tuple[List[str], List[str]]:
        source_path = Path(source.path)
        copied_files: List[str] = []
        skipped_files: List[str] = []
        for candidate in self._read_manifest_paths(source_path):
            if not candidate.exists() or not candidate.is_file() or candidate.suffix.lower() not in extensions:
                continue
            destination = target_dir / candidate.name
            if destination.exists():
                skipped_files.append(str(candidate))
                continue
            shutil.copy2(candidate, destination)
            copied_files.append(str(destination))
        return copied_files, skipped_files

    def _sync_s3_source(self, source: DatasourceConfig, target_dir: Path, extensions: set[str]) -> tuple[List[str], List[str]]:
        try:
            import boto3  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("S3 datasource support requires boto3. Install the BaseTruth connectors extra.") from exc

        bucket, prefix = self._parse_s3_path(source.path)
        settings = dict(source.settings or {})
        session_kwargs: Dict[str, Any] = {}
        profile_name = str(settings.get("profile_name", "")).strip()
        region_name = str(settings.get("region_name", "")).strip()
        if profile_name:
            session_kwargs["profile_name"] = profile_name
        if region_name:
            session_kwargs["region_name"] = region_name
        client = boto3.session.Session(**session_kwargs).client("s3") if session_kwargs else boto3.client("s3")
        paginator = client.get_paginator("list_objects_v2")
        copied_files: List[str] = []
        skipped_files: List[str] = []
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix or None):
            for item in page.get("Contents", []):
                key = str(item.get("Key", ""))
                if not key or key.endswith("/"):
                    continue
                if Path(key).suffix.lower() not in extensions:
                    continue
                relative = Path(key[len(prefix):].lstrip("/")) if prefix and key.startswith(prefix) else Path(key)
                destination = target_dir / relative
                if destination.exists():
                    skipped_files.append(f"s3://{bucket}/{key}")
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                client.download_file(bucket, key, str(destination))
                copied_files.append(str(destination))
        return copied_files, skipped_files

    def _sync_google_drive_source(self, source: DatasourceConfig, target_dir: Path, extensions: set[str]) -> tuple[List[str], List[str]]:
        try:
            from googleapiclient.discovery import build  # type: ignore
            from googleapiclient.http import MediaIoBaseDownload  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Google Drive datasource support requires google-api-python-client and auth dependencies."
            ) from exc

        folder_id = str(source.path or "").replace("drive:", "").strip()
        if not folder_id:
            raise ValueError("Google Drive datasource path must be a folder id or 'drive:<folder_id>'.")

        settings = dict(source.settings or {})
        drive_kwargs: Dict[str, Any] = {}
        service_account_file = str(settings.get("service_account_file", "")).strip()
        if service_account_file:
            try:
                from google.oauth2 import service_account  # type: ignore
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("Google Drive service-account auth requires google-auth.") from exc
            credentials = service_account.Credentials.from_service_account_file(
                service_account_file,
                scopes=["https://www.googleapis.com/auth/drive.readonly"],
            )
            drive_kwargs["credentials"] = credentials

        drive_service: Any = build("drive", "v3", **drive_kwargs)
        copied_files: List[str] = []
        skipped_files: List[str] = []
        queue: List[tuple[str, Path]] = [(folder_id, Path("."))]

        while queue:
            current_folder_id, relative_root = queue.pop(0)
            page_token = None
            while True:
                response = (
                    drive_service.files()
                    .list(
                        q=f"'{current_folder_id}' in parents and trashed = false",
                        fields="nextPageToken, files(id, name, mimeType)",
                        pageToken=page_token,
                    )
                    .execute()
                )
                for item in response.get("files", []):
                    mime_type = str(item.get("mimeType", ""))
                    name = str(item.get("name", ""))
                    if mime_type == "application/vnd.google-apps.folder":
                        queue.append((str(item.get("id", "")), relative_root / name))
                        continue
                    if Path(name).suffix.lower() not in extensions:
                        continue
                    destination = target_dir / relative_root / name
                    if destination.exists():
                        skipped_files.append(f"gdrive:{item.get('id')}")
                        continue
                    request = drive_service.files().get_media(fileId=item["id"])
                    buffer = io.BytesIO()
                    downloader = MediaIoBaseDownload(buffer, request)
                    done = False
                    while not done:
                        _, done = downloader.next_chunk()
                    copied_files.append(self._safe_write_bytes(destination, buffer.getvalue()))
                page_token = response.get("nextPageToken")
                if not page_token:
                    break
        return copied_files, skipped_files

    def _sharepoint_children(
        self,
        session: Any,
        site_id: str,
        drive_id: str,
        folder_path: str,
    ) -> List[Dict[str, Any]]:
        if folder_path:
            url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drives/{drive_id}/root:/{folder_path}:/children"
        else:
            url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drives/{drive_id}/root/children"
        items: List[Dict[str, Any]] = []
        while url:
            response = session.get(url, timeout=60)
            response.raise_for_status()
            payload = response.json()
            items.extend(payload.get("value", []))
            url = payload.get("@odata.nextLink")
        return items

    def _sync_sharepoint_source(self, source: DatasourceConfig, target_dir: Path, extensions: set[str]) -> tuple[List[str], List[str]]:
        try:
            import requests  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("SharePoint datasource support requires requests.") from exc

        import os

        settings = dict(source.settings or {})
        token_env_var = str(settings.get("token_env_var", "BASETRUTH_SHAREPOINT_TOKEN")).strip() or "BASETRUTH_SHAREPOINT_TOKEN"
        token = str(os.environ.get(token_env_var, "") or "").strip()
        if not token:
            raise RuntimeError(f"Set {token_env_var} to use SharePoint datasource sync.")

        site_id, drive_id, folder_path = self._parse_sharepoint_path(source.path)
        session = requests.Session()
        session.headers.update({"Authorization": f"Bearer {token}"})

        copied_files: List[str] = []
        skipped_files: List[str] = []
        queue: List[tuple[str, Path]] = [(folder_path, Path("."))]
        while queue:
            current_path, relative_root = queue.pop(0)
            for item in self._sharepoint_children(session, site_id, drive_id, current_path):
                name = str(item.get("name", ""))
                if "folder" in item:
                    child_path = "/".join(part for part in [current_path.strip("/"), name] if part)
                    queue.append((child_path, relative_root / name))
                    continue
                if Path(name).suffix.lower() not in extensions:
                    continue
                download_url = item.get("@microsoft.graph.downloadUrl")
                if not download_url:
                    continue
                destination = target_dir / relative_root / name
                if destination.exists():
                    skipped_files.append(f"sharepoint:{name}")
                    continue
                response = session.get(download_url, timeout=120)
                response.raise_for_status()
                copied_files.append(self._safe_write_bytes(destination, response.content))
        return copied_files, skipped_files

    def _read_manifest_paths(self, manifest_path: Path) -> List[Path]:
        if manifest_path.suffix.lower() == ".json":
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                items = payload.get("paths", [])
            else:
                items = payload
            return [Path(str(item)) for item in items if str(item).strip()]

        if manifest_path.suffix.lower() == ".csv":
            paths: List[Path] = []
            with manifest_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    value = row.get("path") or row.get("file") or row.get("document")
                    if value:
                        paths.append(Path(value))
            return paths

        lines = manifest_path.read_text(encoding="utf-8").splitlines()
        return [Path(line.strip()) for line in lines if line.strip()]

    def _resolve_source_files(self, source: DatasourceConfig) -> List[Path]:
        source_path = Path(source.path)
        extensions = self._normalize_extensions(source.extensions)
        resolved: List[Path] = []

        if source.kind == "folder":
            iterator = source_path.rglob("*") if source.recursive else source_path.glob("*")
            for candidate in iterator:
                if candidate.is_file() and candidate.suffix.lower() in extensions:
                    resolved.append(candidate)
            return sorted(resolved)

        if source.kind == "manifest":
            for candidate in self._read_manifest_paths(source_path):
                if candidate.exists() and candidate.is_file() and candidate.suffix.lower() in extensions:
                    resolved.append(candidate)
            return resolved

        raise ValueError(f"Unsupported datasource kind: {source.kind}")

    def sync_source(self, source_name: str) -> Dict[str, Any]:
        source = self.get_source(source_name)
        if not source.enabled:
            return {
                "status": "skipped",
                "source": source.to_dict(),
                "message": "Datasource is disabled.",
                "copied_files": [],
            }

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        target_dir = self.artifact_root / "workspace" / "sources" / _slugify(source.name) / "snapshots" / timestamp
        target_dir.mkdir(parents=True, exist_ok=True)

        extensions = self._normalize_extensions(source.extensions)
        if source.kind == "folder":
            copied_files, skipped_files = self._sync_folder_source(source, target_dir, extensions)
        elif source.kind == "manifest":
            copied_files, skipped_files = self._sync_manifest_source(source, target_dir, extensions)
        elif source.kind == "s3":
            copied_files, skipped_files = self._sync_s3_source(source, target_dir, extensions)
        elif source.kind == "google_drive":
            copied_files, skipped_files = self._sync_google_drive_source(source, target_dir, extensions)
        elif source.kind == "sharepoint":
            copied_files, skipped_files = self._sync_sharepoint_source(source, target_dir, extensions)
        else:
            raise ValueError(f"Unsupported datasource kind: {source.kind}")

        manifest_path = target_dir / "sync_manifest.json"
        manifest_payload = {
            "schema_version": 1,
            "synced_at": datetime.now(timezone.utc).isoformat(),
            "source": source.to_dict(),
            "copied_files": copied_files,
            "skipped_files": skipped_files,
        }
        manifest_path.write_text(json.dumps(manifest_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return {
            "status": "success",
            "source": source.to_dict(),
            "snapshot_dir": str(target_dir),
            "copied_files": copied_files,
            "copied_count": len(copied_files),
            "skipped_files": skipped_files,
            "manifest_path": str(manifest_path),
            "message": f"Synced {len(copied_files)} file(s) from datasource '{source.name}'.",
        }