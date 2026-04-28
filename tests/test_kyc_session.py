from __future__ import annotations

from datetime import datetime, timedelta, timezone

from basetruth.kyc.session import KYCSession, SessionStore


def test_kyc_session_exposes_current_history_and_advances_cleanly() -> None:
    session = KYCSession(
        session_id="session-1",
        customer_name="Hrishikesh",
        entity_ref="BT-000001",
        challenges=["blink", "turn_left"],
        reference_embedding_b64=None,
    )

    current_history = session.current_frame_history()
    current_history.append({"ear": 0.30})

    assert session.current_challenge == "blink"
    assert session.challenge_frame_history["ch_0"] == [{"ear": 0.30}]

    session.advance_challenge()

    assert session.current_challenge_idx == 1
    assert session.current_challenge == "turn_left"
    assert session.challenge_frame_history["ch_1"] == []


def test_kyc_session_all_done_turns_true_after_last_challenge() -> None:
    session = KYCSession(
        session_id="session-2",
        customer_name="Hrishikesh",
        entity_ref="BT-000002",
        challenges=["blink"],
        reference_embedding_b64=None,
    )

    assert session.all_done is False

    session.advance_challenge()

    assert session.all_done is True
    assert session.current_challenge is None


def test_session_store_marks_active_expired_sessions_as_expired() -> None:
    store = SessionStore()
    session = store.create(challenges=["blink"], customer_name="Test", entity_ref="BT-000003")
    session.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)

    fetched = store.get(session.session_id)

    assert fetched is session
    assert fetched.status == "expired"


def test_session_store_keeps_completed_session_status_after_expiry() -> None:
    store = SessionStore()
    session = store.create(challenges=["blink"], customer_name="Test", entity_ref="BT-000004")
    session.status = "completed"
    session.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)

    fetched = store.get(session.session_id)

    assert fetched is session
    assert fetched.status == "completed"


def test_session_store_cleanup_expired_removes_only_non_terminal_sessions() -> None:
    store = SessionStore()

    expired_waiting = store.create(challenges=["blink"], customer_name="A", entity_ref="BT-1")
    expired_waiting.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)

    expired_completed = store.create(challenges=["blink"], customer_name="B", entity_ref="BT-2")
    expired_completed.status = "completed"
    expired_completed.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)

    active = store.create(challenges=["blink"], customer_name="C", entity_ref="BT-3")
    active.expires_at = datetime.now(timezone.utc) + timedelta(minutes=5)

    removed = store.cleanup_expired()

    assert removed == 1
    assert store.get(expired_waiting.session_id) is None
    assert store.get(expired_completed.session_id) is expired_completed
    assert store.get(active.session_id) is active