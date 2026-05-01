"""Unit tests for save_video_kyc_check().

These tests use in-memory fakes for db_session and _find_or_create_entity
so no live database or MinIO is required.
"""
from __future__ import annotations

from contextlib import contextmanager

from basetruth import store
from basetruth.db import Entity, VideoKYCCheck


class _FakeQuery:
    def __init__(self, model, first_result=None, all_result=None) -> None:
        self._model = model
        self._first_result = first_result
        self._all_result = all_result or []

    def filter(self, *args, **kwargs):
        return self

    def order_by(self, *args, **kwargs):
        return self

    def limit(self, *args, **kwargs):
        return self

    def first(self):
        return self._first_result

    def all(self):
        return list(self._all_result)


class _FakeSession:
    def __init__(self) -> None:
        self.added = []
        self.deleted = []
        self._next_id = 1

    def query(self, model):
        if model is VideoKYCCheck:
            return _FakeQuery(model, first_result=None, all_result=[])
        raise AssertionError(f"Unexpected model queried in video KYC test: {model}")

    def add(self, obj) -> None:
        if isinstance(obj, VideoKYCCheck) and getattr(obj, "id", None) is None:
            obj.id = self._next_id
            self._next_id += 1
        self.added.append(obj)

    def flush(self) -> None:
        return None

    def delete(self, obj) -> None:
        self.deleted.append(obj)


def test_save_video_kyc_check_creates_video_kyc_check_row(monkeypatch) -> None:
    """save_video_kyc_check must write exactly one VideoKYCCheck row."""
    fake_session = _FakeSession()
    entity = Entity(id=3, entity_ref="BT-000003", first_name="Alice")

    @contextmanager
    def fake_db_session():
        yield fake_session

    monkeypatch.setattr(store, "db_session", fake_db_session)
    monkeypatch.setattr(store, "_find_or_create_entity", lambda sess, ident: entity)

    result = {
        "is_match": True,
        "liveness_passed": True,
        "cosine_similarity": 0.82,
        "display_score": 82.0,
        "threshold": 0.40,
        "liveness_state": "passed",
        "aadhar_dtls": {"name": "Alice", "uid": "111122223333"},
        "address_dtls": {"address": "123 Main St, Pune"},
    }

    saved = store.save_video_kyc_check(
        result=result,
        extra_identity={"first_name": "Alice"},
    )

    assert saved is not None
    assert saved["entity_ref"] == "BT-000003"
    assert saved["check_type"] == "video_kyc"
    assert saved["verdict"] == "PASS"
    assert saved["status"] == "pass"

    # Exactly one VideoKYCCheck row must be written
    vkyc_rows = [o for o in fake_session.added if isinstance(o, VideoKYCCheck)]
    assert len(vkyc_rows) == 1
    row = vkyc_rows[0]
    assert row.is_match is True
    assert row.liveness_passed is True
    assert row.verdict == "PASS"


def test_save_video_kyc_check_stores_aadhar_and_address_dtls(monkeypatch) -> None:
    """aadhar_dtls and address_dtls must be stored on the VideoKYCCheck row."""
    fake_session = _FakeSession()
    entity = Entity(id=4, entity_ref="BT-000004", first_name="Bob")

    @contextmanager
    def fake_db_session():
        yield fake_session

    monkeypatch.setattr(store, "db_session", fake_db_session)
    monkeypatch.setattr(store, "_find_or_create_entity", lambda sess, ident: entity)

    result = {
        "is_match": False,
        "liveness_passed": False,
        "aadhar_dtls": {"uid": "999988887777", "name": "Bob"},
        "address_dtls": {"address": "456 Park Ave, Mumbai"},
    }

    store.save_video_kyc_check(result=result, extra_identity={"first_name": "Bob"})

    vkyc_rows = [o for o in fake_session.added if isinstance(o, VideoKYCCheck)]
    assert len(vkyc_rows) == 1
    row = vkyc_rows[0]
    assert row.aadhar_dtls is not None
    assert row.aadhar_dtls.get("uid") == "999988887777"
    assert row.address_dtls is not None
    assert "Mumbai" in row.address_dtls.get("address", "")


def test_save_video_kyc_check_inconclusive_when_only_match(monkeypatch) -> None:
    """When is_match=True but liveness_passed=False the result must be 'inconclusive'."""
    fake_session = _FakeSession()
    entity = Entity(id=5, entity_ref="BT-000005", first_name="Carol")

    @contextmanager
    def fake_db_session():
        yield fake_session

    monkeypatch.setattr(store, "db_session", fake_db_session)
    monkeypatch.setattr(store, "_find_or_create_entity", lambda sess, ident: entity)

    result = {"is_match": True, "liveness_passed": False}
    saved = store.save_video_kyc_check(result=result, extra_identity={"first_name": "Carol"})

    assert saved is not None
    assert saved["status"] == "inconclusive"
    assert saved["verdict"] == "FAIL"


def test_save_identity_check_shim_routes_video_kyc(monkeypatch) -> None:
    """save_identity_check with check_type='video_kyc' must delegate to save_video_kyc_check."""
    calls = []

    def fake_save_video_kyc_check(**kwargs):
        calls.append(kwargs)
        return {"id": 1, "entity_ref": "BT-000006", "check_type": "video_kyc",
                "status": "pass", "verdict": "PASS"}

    monkeypatch.setattr(store, "save_video_kyc_check", fake_save_video_kyc_check)

    result = store.save_identity_check(
        check_type="video_kyc",
        result={"is_match": True, "liveness_passed": True},
        forced_entity_ref="BT-000006",
    )

    assert len(calls) == 1, "save_video_kyc_check must be called exactly once"
    assert result["check_type"] == "video_kyc"
