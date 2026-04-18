from __future__ import annotations

from contextlib import contextmanager

from basetruth import store
from basetruth.db import DocumentExtraction, Entity, IdentityCheck


class _FakeQuery:
    def __init__(self, model, first_result=None, all_result=None) -> None:
        self._model = model
        self._first_result = first_result
        self._all_result = all_result or []

    def filter(self, *args, **kwargs):
        return self

    def order_by(self, *args, **kwargs):
        return self

    def first(self):
        return self._first_result

    def all(self):
        return list(self._all_result)


class _FakeSession:
    def __init__(self) -> None:
        self.added = []
        self.deleted = []
        self._next_identity_check_id = 1
        self._next_document_extraction_id = 1

    def query(self, model):
        if model is IdentityCheck:
            return _FakeQuery(model, first_result=None, all_result=[])
        if model is DocumentExtraction:
            return _FakeQuery(model, first_result=None, all_result=[])
        raise AssertionError(f"Unexpected model queried: {model}")

    def add(self, obj) -> None:
        # Assign simple IDs so save_identity_check can build its return payload.
        if isinstance(obj, IdentityCheck) and getattr(obj, "id", None) is None:
            obj.id = self._next_identity_check_id
            self._next_identity_check_id += 1
        if isinstance(obj, DocumentExtraction) and getattr(obj, "id", None) is None:
            obj.id = self._next_document_extraction_id
            self._next_document_extraction_id += 1
        self.added.append(obj)

    def flush(self) -> None:
        return None

    def delete(self, obj) -> None:
        self.deleted.append(obj)


def test_save_identity_check_persists_aadhaar_demographics(monkeypatch) -> None:
    fake_session = _FakeSession()
    entity = Entity(id=7, entity_ref="BT-000007", first_name="Hrishikesh")

    @contextmanager
    def fake_db_session():
        yield fake_session

    monkeypatch.setattr(store, "db_session", fake_db_session)
    monkeypatch.setattr(store, "_find_or_create_entity", lambda session, identity: entity)

    saved = store.save_identity_check(
        check_type="face_match",
        result={
            "match": True,
            "aadhaar_qr": {
                "name": "HRISHIKESH NAMDEO MALUSKAR",
                "uid": "123412341234",
                "gender": "M",
                "dist": "Pune",
                "state": "Maharashtra",
                "qr_type": "xml",
            },
        },
        extra_identity={"first_name": "Hrishikesh"},
    )

    assert saved is not None
    assert saved["entity_ref"] == "BT-000007"

    aadhaar_rows = [
        obj
        for obj in fake_session.added
        if isinstance(obj, DocumentExtraction) and obj.document_type == "aadhaar"
    ]

    assert len(aadhaar_rows) == 1
    assert aadhaar_rows[0].source_screen == "identity_verification"
    assert aadhaar_rows[0].extracted_data["gender"] == "M"
    assert aadhaar_rows[0].extracted_data["dist"] == "Pune"
    assert aadhaar_rows[0].extracted_data["state"] == "Maharashtra"