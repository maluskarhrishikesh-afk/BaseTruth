from __future__ import annotations

import json
from pathlib import Path

from basetruth.datasources import DatasourceConfig, DatasourceRegistry


def test_folder_datasource_sync_copies_supported_files(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "one.pdf").write_text("pdf", encoding="utf-8")
    (source_dir / "two.json").write_text("{}", encoding="utf-8")
    (source_dir / "skip.exe").write_text("x", encoding="utf-8")

    registry = DatasourceRegistry(tmp_path / "artifacts")
    registry.upsert_source(
        DatasourceConfig(
            name="Payroll Share",
            kind="folder",
            path=str(source_dir),
            extensions=[".pdf", ".json"],
        )
    )

    result = registry.sync_source("Payroll Share")

    assert result["status"] == "success"
    assert result["copied_count"] == 2
    assert Path(result["manifest_path"]).exists()


def test_manifest_datasource_sync_reads_json_manifest(tmp_path: Path) -> None:
    doc1 = tmp_path / "a.pdf"
    doc1.write_text("pdf", encoding="utf-8")
    doc2 = tmp_path / "b.json"
    doc2.write_text("{}", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([str(doc1), str(doc2)]), encoding="utf-8")

    registry = DatasourceRegistry(tmp_path / "artifacts")
    registry.upsert_source(
        DatasourceConfig(
            name="Manifest Source",
            kind="manifest",
            path=str(manifest),
            extensions=[".pdf", ".json"],
        )
    )

    result = registry.sync_source("Manifest Source")

    assert result["status"] == "success"
    assert result["copied_count"] == 2


def test_connector_settings_are_persisted_and_path_is_derived(tmp_path: Path) -> None:
    registry = DatasourceRegistry(tmp_path / "artifacts")
    settings = {
        "bucket": "fraud-inputs",
        "prefix": "payroll/jan",
        "region_name": "ap-south-1",
        "profile_name": "security-audit",
    }

    registry.upsert_source(
        DatasourceConfig(
            name="S3 Source",
            kind="s3",
            path=registry.build_path_from_settings("s3", settings),
            settings=settings,
        )
    )

    saved = registry.get_source("S3 Source")

    assert saved.path == "s3://fraud-inputs/payroll/jan"
    assert saved.settings == settings