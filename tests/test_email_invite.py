"""Unit tests for KYCEmailSender and KYCEmailConfig.

All tests run without a live SMTP server — smtplib is patched at the
module level so no network calls are made.
"""
from __future__ import annotations

import os
import smtplib
from typing import Optional
from unittest.mock import MagicMock, call, patch

import pytest

from basetruth.integrations.email_invite import KYCEmailConfig, KYCEmailSender


# ---------------------------------------------------------------------------
# KYCEmailConfig
# ---------------------------------------------------------------------------

class TestKYCEmailConfig:
    def test_from_env_reads_all_variables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BT_SMTP_HOST", "smtp.example.com")
        monkeypatch.setenv("BT_SMTP_PORT", "465")
        monkeypatch.setenv("BT_SMTP_USER", "user@example.com")
        monkeypatch.setenv("BT_SMTP_PASSWORD", "secret")
        monkeypatch.setenv("BT_EMAIL_FROM", "KYC <kyc@example.com>")
        monkeypatch.setenv("BT_SMTP_SSL", "1")

        cfg = KYCEmailConfig.from_env()

        assert cfg.host      == "smtp.example.com"
        assert cfg.port      == 465
        assert cfg.user      == "user@example.com"
        assert cfg.password  == "secret"
        assert cfg.from_addr == "KYC <kyc@example.com>"
        assert cfg.use_ssl   is True

    def test_from_env_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When env vars are absent, host/user/password default to empty string
        and port defaults to 587."""
        for var in ["BT_SMTP_HOST", "BT_SMTP_PORT", "BT_SMTP_USER",
                    "BT_SMTP_PASSWORD", "BT_EMAIL_FROM", "BT_SMTP_SSL"]:
            monkeypatch.delenv(var, raising=False)

        cfg = KYCEmailConfig.from_env()

        assert cfg.host     == ""
        assert cfg.port     == 587
        assert cfg.user     == ""
        assert cfg.password == ""
        assert cfg.use_ssl  is False

    def test_from_addr_falls_back_to_user(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BT_SMTP_USER", "agent@bt.ai")
        monkeypatch.delenv("BT_EMAIL_FROM", raising=False)

        cfg = KYCEmailConfig.from_env()
        assert cfg.from_addr == "agent@bt.ai"

    def test_is_complete_true_when_all_required_set(self) -> None:
        cfg = KYCEmailConfig(
            host="smtp.example.com", port=587,
            user="u", password="p", from_addr="u",
        )
        assert cfg.is_complete is True

    def test_is_complete_false_when_host_missing(self) -> None:
        cfg = KYCEmailConfig(host="", port=587, user="u", password="p", from_addr="u")
        assert cfg.is_complete is False

    def test_is_complete_false_when_user_missing(self) -> None:
        cfg = KYCEmailConfig(host="smtp.example.com", port=587, user="", password="p", from_addr="")
        assert cfg.is_complete is False

    def test_is_complete_false_when_password_missing(self) -> None:
        cfg = KYCEmailConfig(host="smtp.example.com", port=587, user="u", password="", from_addr="u")
        assert cfg.is_complete is False

    @pytest.mark.parametrize("ssl_value", ["1", "true", "True", "TRUE", "yes", "Yes"])
    def test_use_ssl_truthy_values(
        self, ssl_value: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("BT_SMTP_SSL", ssl_value)
        monkeypatch.setenv("BT_SMTP_HOST", "h")
        monkeypatch.setenv("BT_SMTP_USER", "u")
        monkeypatch.setenv("BT_SMTP_PASSWORD", "p")
        monkeypatch.delenv("BT_EMAIL_FROM", raising=False)
        assert KYCEmailConfig.from_env().use_ssl is True

    @pytest.mark.parametrize("ssl_value", ["0", "false", "no", ""])
    def test_use_ssl_falsy_values(
        self, ssl_value: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("BT_SMTP_SSL", ssl_value)
        monkeypatch.setenv("BT_SMTP_HOST", "h")
        monkeypatch.setenv("BT_SMTP_USER", "u")
        monkeypatch.setenv("BT_SMTP_PASSWORD", "p")
        monkeypatch.delenv("BT_EMAIL_FROM", raising=False)
        assert KYCEmailConfig.from_env().use_ssl is False


# ---------------------------------------------------------------------------
# Helper to build a fully-configured sender (no live SMTP needed)
# ---------------------------------------------------------------------------

def _sender(use_ssl: bool = False) -> KYCEmailSender:
    cfg = KYCEmailConfig(
        host="smtp.example.com", port=587,
        user="kyc@example.com", password="pass",
        from_addr="KYC <kyc@example.com>",
        use_ssl=use_ssl,
    )
    return KYCEmailSender(cfg)


def _unconfigured_sender() -> KYCEmailSender:
    cfg = KYCEmailConfig(host="", port=587, user="", password="", from_addr="")
    return KYCEmailSender(cfg)


# ---------------------------------------------------------------------------
# KYCEmailSender.is_configured
# ---------------------------------------------------------------------------

class TestIsConfigured:
    def test_true_when_complete(self) -> None:
        assert _sender().is_configured() is True

    def test_false_when_not_complete(self) -> None:
        assert _unconfigured_sender().is_configured() is False


# ---------------------------------------------------------------------------
# KYCEmailSender.send_kyc_invite — failure cases (no SMTP calls)
# ---------------------------------------------------------------------------

class TestSendKYCInviteFailures:
    """Failure paths that must return (False, message) without touching SMTP."""

    def test_returns_false_when_not_configured(self) -> None:
        ok, err = _unconfigured_sender().send_kyc_invite(
            to_email="c@example.com",
            customer_name="Test", agent_name="Agent",
            session_url="http://url", date_str="01 Jan 2026",
            time_str="10:00 AM", duration_min=30,
        )
        assert ok is False
        assert "BT_SMTP_HOST" in err

    @pytest.mark.parametrize("bad_email", [
        "", "notanemail", "missing-at.com", "@nodomain", "a@b",
    ])
    def test_returns_false_for_invalid_email(self, bad_email: str) -> None:
        ok, err = _sender().send_kyc_invite(
            to_email=bad_email,
            customer_name="Test", agent_name="Agent",
            session_url="http://url", date_str="01 Jan 2026",
            time_str="10:00 AM", duration_min=30,
        )
        assert ok is False
        assert bad_email in err or "Invalid" in err

    def test_returns_false_on_auth_error(self) -> None:
        with patch.object(KYCEmailSender, "_deliver", side_effect=smtplib.SMTPAuthenticationError(535, b"Bad credentials")):
            ok, err = _sender().send_kyc_invite(
                to_email="c@example.com",
                customer_name="Test", agent_name="Agent",
                session_url="http://url", date_str="01 Jan 2026",
                time_str="10:00 AM", duration_min=30,
            )
        assert ok is False
        assert "authentication" in err.lower()

    def test_returns_false_on_smtp_exception(self) -> None:
        with patch.object(KYCEmailSender, "_deliver", side_effect=smtplib.SMTPException("Connection refused")):
            ok, err = _sender().send_kyc_invite(
                to_email="c@example.com",
                customer_name="Test", agent_name="Agent",
                session_url="http://url", date_str="01 Jan 2026",
                time_str="10:00 AM", duration_min=30,
            )
        assert ok is False
        assert "Connection refused" in err

    def test_returns_false_on_os_error(self) -> None:
        with patch.object(KYCEmailSender, "_deliver", side_effect=OSError("Network unreachable")):
            ok, err = _sender().send_kyc_invite(
                to_email="c@example.com",
                customer_name="Test", agent_name="Agent",
                session_url="http://url", date_str="01 Jan 2026",
                time_str="10:00 AM", duration_min=30,
            )
        assert ok is False
        assert "smtp.example.com" in err or "Network" in err


# ---------------------------------------------------------------------------
# KYCEmailSender.send_kyc_invite — success path
# ---------------------------------------------------------------------------

class TestSendKYCInviteSuccess:
    def test_returns_true_on_success(self) -> None:
        with patch.object(KYCEmailSender, "_deliver", return_value=None):
            ok, err = _sender().send_kyc_invite(
                to_email="rahul@example.com",
                customer_name="Rahul Sharma", agent_name="Priya",
                session_url="http://bt.local/kyc/abc123",
                date_str="29 Apr 2026", time_str="10:30 AM", duration_min=30,
            )
        assert ok is True
        assert err == ""

    def test_deliver_is_called_with_correct_recipient(self) -> None:
        with patch.object(KYCEmailSender, "_deliver") as mock_deliver:
            _sender().send_kyc_invite(
                to_email="rahul@example.com",
                customer_name="Rahul", agent_name="Agent",
                session_url="http://url", date_str="01 Jan 2026",
                time_str="09:00 AM", duration_min=30,
            )
        mock_deliver.assert_called_once()
        # Second argument to _deliver is the to_email
        _, call_to_email = mock_deliver.call_args.args
        assert call_to_email == "rahul@example.com"


# ---------------------------------------------------------------------------
# KYCEmailSender._build_email_body
# ---------------------------------------------------------------------------

class TestBuildEmailBody:
    def test_body_contains_all_key_fields(self) -> None:
        body = KYCEmailSender._build_email_body(
            customer_name="Rahul Sharma",
            agent_name="Priya Mehta",
            session_url="http://bt.local/kyc/xyz",
            date_str="29 Apr 2026",
            time_str="10:30 AM",
            duration_min=30,
        )
        assert "Rahul Sharma" in body
        assert "Priya Mehta" in body
        assert "http://bt.local/kyc/xyz" in body
        assert "29 Apr 2026" in body
        assert "10:30 AM" in body
        assert "30 minutes" in body
        # Session TTL notice must be present
        assert "30 minutes" in body and "active" in body

    def test_body_mentions_required_documents(self) -> None:
        body = KYCEmailSender._build_email_body(
            "C", "A", "http://url", "01 Jan 2026", "09:00 AM", 30,
        )
        assert "Aadhaar" in body
        assert "PAN" in body
        assert "address proof" in body.lower()


# ---------------------------------------------------------------------------
# KYCEmailSender._deliver — STARTTLS vs SSL branching
# ---------------------------------------------------------------------------

class TestDeliver:
    def test_uses_starttls_by_default(self) -> None:
        sender = _sender(use_ssl=False)
        mock_smtp = MagicMock()
        mock_smtp.__enter__ = MagicMock(return_value=mock_smtp)
        mock_smtp.__exit__ = MagicMock(return_value=False)

        with patch("smtplib.SMTP", return_value=mock_smtp):
            sender._deliver("raw_msg", "to@example.com")

        mock_smtp.ehlo.assert_called_once()
        mock_smtp.starttls.assert_called_once()
        mock_smtp.login.assert_called_once_with("kyc@example.com", "pass")
        mock_smtp.sendmail.assert_called_once()

    def test_uses_smtp_ssl_when_flag_set(self) -> None:
        sender = _sender(use_ssl=True)
        mock_smtp = MagicMock()
        mock_smtp.__enter__ = MagicMock(return_value=mock_smtp)
        mock_smtp.__exit__ = MagicMock(return_value=False)

        with patch("smtplib.SMTP_SSL", return_value=mock_smtp):
            sender._deliver("raw_msg", "to@example.com")

        # ehlo and starttls must NOT be called for direct SSL
        mock_smtp.ehlo.assert_not_called()
        mock_smtp.starttls.assert_not_called()
        mock_smtp.login.assert_called_once_with("kyc@example.com", "pass")
        mock_smtp.sendmail.assert_called_once()

    def test_ics_attachment_included_when_provided(self) -> None:
        """Verify that ics_bytes results in an attachment in the outgoing message."""
        captured: list = []

        def fake_deliver(raw_message: str, to_email: str) -> None:
            captured.append(raw_message)

        with patch.object(KYCEmailSender, "_deliver", side_effect=fake_deliver):
            _sender().send_kyc_invite(
                to_email="c@example.com",
                customer_name="Test Customer",
                agent_name="Agent",
                session_url="http://url",
                date_str="01 Jan 2026",
                time_str="09:00 AM",
                duration_min=30,
                ics_bytes=b"BEGIN:VCALENDAR\r\nEND:VCALENDAR",
            )

        assert len(captured) == 1
        # The raw MIME message must mention text/calendar for the .ics part
        assert "text/calendar" in captured[0]
        assert "vkyc_Test_Customer.ics" in captured[0]

    def test_no_attachment_when_ics_bytes_none(self) -> None:
        captured: list = []

        def fake_deliver(raw_message: str, to_email: str) -> None:
            captured.append(raw_message)

        with patch.object(KYCEmailSender, "_deliver", side_effect=fake_deliver):
            _sender().send_kyc_invite(
                to_email="c@example.com",
                customer_name="Test", agent_name="Agent",
                session_url="http://url", date_str="01 Jan 2026",
                time_str="09:00 AM", duration_min=30,
                ics_bytes=None,
            )

        assert "text/calendar" not in captured[0]
