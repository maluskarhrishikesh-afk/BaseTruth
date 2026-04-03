"""KYC Session management — in-memory store with TTL, challenge sequencing."""
from __future__ import annotations

import secrets
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

SESSION_TTL = timedelta(minutes=30)

# All supported active-liveness challenges.
ALL_CHALLENGES: List[str] = ["blink", "turn_left", "turn_right", "nod"]

CHALLENGE_LABELS: Dict[str, str] = {
    "blink":       "CLOSE YOUR EYES",
    "turn_left":   "TURN HEAD LEFT",
    "turn_right":  "TURN HEAD RIGHT",
    "nod":         "NOD YOUR HEAD",
}

CHALLENGE_INSTRUCTIONS: Dict[str, str] = {
    "blink":       "Slowly close both eyes fully, then open them again",
    "turn_left":   "Slowly turn your head to YOUR left",
    "turn_right":  "Slowly turn your head to YOUR right",
    "nod":         "Slowly nod your head down and then back up",
}


@dataclass
class KYCSession:
    """Holds the full state for one Video KYC session."""

    session_id: str
    customer_name: str
    entity_ref: str
    challenges: List[str]
    # base64-encoded float32 numpy bytes of the reference face embedding
    reference_embedding_b64: Optional[str]

    # ── mutable state ────────────────────────────────────────────────────
    status: str = "waiting"          # waiting | active | completed | failed | expired
    current_challenge_idx: int = 0
    # Per-challenge frame features: {"ch_0": [...], "ch_1": [...], ...}
    challenge_frame_history: Dict[str, List[Dict[str, float]]] = field(
        default_factory=dict
    )
    # Last captured frame (JPEG bytes) sent by the customer — used for PDF
    last_live_frame_bytes: Optional[bytes] = None
    result: Optional[Dict[str, Any]] = None

    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    expires_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc) + SESSION_TTL
    )

    # ── helpers ──────────────────────────────────────────────────────────
    def is_expired(self) -> bool:
        return datetime.now(timezone.utc) > self.expires_at

    @property
    def current_challenge(self) -> Optional[str]:
        if self.current_challenge_idx < len(self.challenges):
            return self.challenges[self.current_challenge_idx]
        return None

    @property
    def all_done(self) -> bool:
        return self.current_challenge_idx >= len(self.challenges)

    def current_frame_history(self) -> List[Dict[str, float]]:
        key = f"ch_{self.current_challenge_idx}"
        return self.challenge_frame_history.setdefault(key, [])

    def advance_challenge(self) -> None:
        self.current_challenge_idx += 1
        # Reset frame history for the next challenge
        key = f"ch_{self.current_challenge_idx}"
        self.challenge_frame_history[key] = []

    def to_status_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "customer_name": self.customer_name,
            "entity_ref": self.entity_ref,
            "status": self.status,
            "challenges": self.challenges,
            "challenges_completed": self.current_challenge_idx,
            "total_challenges": len(self.challenges),
            "result": self.result,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
        }


class SessionStore:
    """Thread-safe in-memory session store."""

    def __init__(self) -> None:
        self._sessions: Dict[str, KYCSession] = {}
        self._lock = threading.Lock()

    def create(
        self,
        challenges: List[str],
        reference_embedding_b64: Optional[str] = None,
        customer_name: str = "",
        entity_ref: str = "",
    ) -> KYCSession:
        sid = secrets.token_urlsafe(16)
        session = KYCSession(
            session_id=sid,
            customer_name=customer_name,
            entity_ref=entity_ref,
            challenges=challenges,
            reference_embedding_b64=reference_embedding_b64,
        )
        with self._lock:
            self._sessions[sid] = session
        return session

    def get(self, session_id: str) -> Optional[KYCSession]:
        with self._lock:
            s = self._sessions.get(session_id)
            if s and s.is_expired() and s.status not in ("completed", "failed"):
                s.status = "expired"
            return s

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def cleanup_expired(self) -> int:
        with self._lock:
            expired = [
                k for k, v in self._sessions.items()
                if v.is_expired() and v.status not in ("completed", "failed")
            ]
            for k in expired:
                del self._sessions[k]
        return len(expired)
