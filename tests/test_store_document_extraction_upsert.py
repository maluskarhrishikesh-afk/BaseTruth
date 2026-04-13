from __future__ import annotations

from contextlib import contextmanager

from basetruth import store
from basetruth.db import DocumentExtraction, Entity, Scan


def _filter_value(expr):
    right = getattr(expr, "right", None)
    if hasattr(right, "value"):
        return right.value
    return right


def _matches_expr(obj, expr) -> bool:
    left = getattr(expr, "left", None)
    field_name = getattr(left, "key", None)
    operator_name = getattr(getattr(expr, "operator", None), "__name__", "")
    expected = _filter_value(expr)
    actual = getattr(obj, field_name)

    if operator_name == "eq":
        return actual == expected
    if operator_name == "ne":
        return actual != expected
    raise AssertionError(f"Unsupported filter operator: {operator_name}")


class _FakeQuery:
    def __init__(self, session: "_FakeSession", model) -> None:
        self._session = session
        self._model = model
        self._filters = []

    def filter(self, *args, **kwargs):
        self._filters.extend(args)
        return self

    def order_by(self, *args, **kwargs):
        return self

    def all(self):
        rows = list(self._session.rows_for(self._model))
        for expr in self._filters:
            rows = [row for row in rows if _matches_expr(row, expr)]
        return rows

    def first(self):
        rows = self.all()
        return rows[0] if rows else None


class _FakeSession:
    def __init__(self, entity: Entity) -> None:
        self.entities = [entity]
        self.scans: list[Scan] = []
        self.document_extractions: list[DocumentExtraction] = []
        self._next_scan_id = 1
        self._next_document_extraction_id = 1

    def rows_for(self, model):
        if model is Entity:
            return self.entities
        if model is Scan:
            return self.scans
        if model is DocumentExtraction:
            return self.document_extractions
        raise AssertionError(f"Unexpected model queried: {model}")

    def query(self, model):
        return _FakeQuery(self, model)

    def add(self, obj) -> None:
        if isinstance(obj, Scan) and getattr(obj, "id", None) is None:
            obj.id = self._next_scan_id
            self._next_scan_id += 1
            self.scans.append(obj)
            return
        if isinstance(obj, DocumentExtraction) and getattr(obj, "id", None) is None:
            obj.id = self._next_document_extraction_id
            self._next_document_extraction_id += 1
            self.document_extractions.append(obj)
            return
        raise AssertionError(f"Unexpected object added: {obj}")

    def flush(self) -> None:
        return None

    def delete(self, obj) -> None:
        if isinstance(obj, Scan):
            self.scans.remove(obj)
            return
        if isinstance(obj, DocumentExtraction):
            self.document_extractions.remove(obj)
            return
        raise AssertionError(f"Unexpected object deleted: {obj}")


class _FakeMappingsResult:
    def __init__(self, rows) -> None:
        self._rows = rows

    def all(self):
        return list(self._rows)


class _FakeExecuteResult:
    def __init__(self, *, scalar_value=None, rows=None) -> None:
        self._scalar_value = scalar_value
        self._rows = rows or []

    def scalar(self):
        return self._scalar_value

    def mappings(self):
        return _FakeMappingsResult(self._rows)


class _FakeTableRowsSession:
    def execute(self, statement, params=None):
        sql = str(statement)
        if "COUNT(*) FROM document_extractions" in sql:
            return _FakeExecuteResult(scalar_value=1)
        if "FROM document_extractions ORDER BY id DESC LIMIT :lim" in sql:
            return _FakeExecuteResult(
                rows=[
                    {
                        "id": 5,
                        "entity_id": 1,
                        "scan_id": 3,
                        "file_name": "HSC-Marksheet.pdf",
                        "document_type": "marksheet",
                        "source_screen": "bulk_scan",
                        "created_at": "2026-04-13T18:42:35Z",
                        "extracted_data": {"candidate_name": "Hrishikesh"},
                    }
                ]
            )
        raise AssertionError(f"Unexpected SQL: {sql}")


def test_save_scan_to_db_upserts_document_extraction_by_entity_and_file_name(monkeypatch) -> None:
    entity = Entity(id=7, entity_ref="BT-000007", first_name="Hrishikesh")
    fake_session = _FakeSession(entity)

    @contextmanager
    def fake_db_session():
        yield fake_session

    monkeypatch.setattr(store, "db_session", fake_db_session)
    monkeypatch.setattr(store, "minio_upload", lambda *args, **kwargs: None)
    monkeypatch.setattr(store, "_persist_scan_layered_analysis", lambda *args, **kwargs: None)

    first_report = {
        "source": {
            "name": "salary_july.pdf",
            "path": r"C:\temp\salary_july.pdf",
            "sha256": "sha-one",
        },
        "document_type": "payslip",
        "_layered_forensics": {"scan_summary": {"forensic_verdict": "ORIGINAL"}},
        "_document_extraction": {"employee_name": "Hrishikesh", "net_salary": 55000},
    }
    second_report = {
        "source": {
            "name": "salary_july.pdf",
            "path": r"C:\temp\salary_july.pdf",
            "sha256": "sha-two",
        },
        "document_type": "bank_statement",
        "_layered_forensics": {"scan_summary": {"forensic_verdict": "UNCERTAIN"}},
        "_document_extraction": {"account_holder": "Hrishikesh", "closing_balance": 81000},
    }

    saved_first = store.save_scan_to_db(
        first_report,
        forced_entity_ref="BT-000007",
        layered_screen_name="Bulk Scan",
    )
    saved_second = store.save_scan_to_db(
        second_report,
        forced_entity_ref="BT-000007",
        layered_screen_name="Bulk Scan",
    )

    assert saved_first is not None
    assert saved_second is not None
    assert len(fake_session.scans) == 2
    assert len(fake_session.document_extractions) == 1

    extraction = fake_session.document_extractions[0]
    assert extraction.entity_id == entity.id
    assert extraction.file_name == "salary_july.pdf"
    assert extraction.document_type == "bank_statement"
    assert extraction.scan_id == fake_session.scans[-1].id
    assert extraction.extracted_data["account_holder"] == "Hrishikesh"
    assert extraction.extracted_data["closing_balance"] == 81000


def test_db_table_rows_document_extractions_includes_file_name(monkeypatch) -> None:
    @contextmanager
    def fake_db_session():
        yield _FakeTableRowsSession()

    monkeypatch.setattr(store, "db_session", fake_db_session)

    rows, total = store.db_table_rows("document_extractions", limit=50)

    assert total == 1
    assert len(rows) == 1
    assert rows[0]["file_name"] == "HSC-Marksheet.pdf"
    assert rows[0]["document_type"] == "marksheet"