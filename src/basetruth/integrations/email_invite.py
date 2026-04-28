"""KYC email invite sender — SMTP-based email for Video KYC session invites.

This module separates all email-sending concerns from the UI layer.
The Streamlit page only calls ``KYCEmailSender.send_kyc_invite()`` and
gets a ``(success: bool, error_message: str)`` tuple back.

Configuration is read from environment variables so no credentials ever
appear in source code:

  BT_SMTP_HOST      SMTP server hostname  (e.g. smtp.gmail.com)
  BT_SMTP_PORT      Port — 587 for STARTTLS (default), 465 for SSL
  BT_SMTP_USER      SMTP login username
  BT_SMTP_PASSWORD  SMTP password or app-specific password
  BT_EMAIL_FROM     Sender address shown to recipient
                    (e.g. "BaseTruth KYC <kyc@basetruth.ai>").
                    Falls back to BT_SMTP_USER when not set.
  BT_SMTP_SSL       Set to '1' or 'true' to use direct SSL (port 465).
                    Defaults to STARTTLS on port 587.

Gmail quick-start:
  1. Enable 2-Step Verification in your Google Account.
  2. Create an App Password (Security → App Passwords).
  3. Set BT_SMTP_HOST=smtp.gmail.com, BT_SMTP_PORT=587,
     BT_SMTP_USER=you@gmail.com, BT_SMTP_PASSWORD=<app_password>.
"""
from __future__ import annotations

import os
import smtplib
from dataclasses import dataclass
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional, Tuple

from basetruth.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KYCEmailConfig:
    """Immutable SMTP configuration.  Loaded once from environment variables
    and passed into ``KYCEmailSender``.  Use ``KYCEmailConfig.from_env()``
    as the normal constructor.
    """

    host: str
    port: int
    user: str
    password: str
    from_addr: str
    use_ssl: bool = False

    @classmethod
    def from_env(cls) -> "KYCEmailConfig":
        """Read SMTP settings from environment variables and return a config
        instance.  Fields default to empty strings / safe defaults so the
        caller can always construct an instance and check ``is_complete``.
        """
        host      = os.getenv("BT_SMTP_HOST", "")
        port      = int(os.getenv("BT_SMTP_PORT", "587"))
        user      = os.getenv("BT_SMTP_USER", "")
        password  = os.getenv("BT_SMTP_PASSWORD", "")
        # Allow operator to customise the sender display name / address.
        from_addr = os.getenv("BT_EMAIL_FROM", user)
        use_ssl   = os.getenv("BT_SMTP_SSL", "").lower() in ("1", "true", "yes")
        return cls(
            host=host, port=port, user=user,
            password=password, from_addr=from_addr, use_ssl=use_ssl,
        )

    @property
    def is_complete(self) -> bool:
        """Returns True only when all three required fields are set.
        ``from_addr`` falls back to ``user``, so it is always populated when
        user is set.
        """
        return bool(self.host and self.user and self.password)


# ---------------------------------------------------------------------------
# Email sender
# ---------------------------------------------------------------------------

class KYCEmailSender:
    """Sends Video KYC session invite emails to customers.

    Responsibilities (single):
      - Build the invite email (plain-text body + .ics attachment)
      - Deliver it via SMTP (STARTTLS or SSL)
      - Return a (success, error_message) result so the UI can show feedback

    This class does NOT touch the database, MinIO, or Streamlit session state.
    Those concerns belong to the calling page.

    Usage::

        sender = KYCEmailSender()
        if sender.is_configured():
            ok, err = sender.send_kyc_invite(
                to_email="rahul@example.com",
                customer_name="Rahul Sharma",
                agent_name="Priya Mehta",
                session_url="http://...",
                date_str="29 Apr 2026",
                time_str="10:30 AM",
                duration_min=30,
                ics_bytes=<bytes>,
            )
    """

    def __init__(self, config: Optional[KYCEmailConfig] = None) -> None:
        # If no config is passed, read from environment.
        self._cfg = config or KYCEmailConfig.from_env()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def is_configured(self) -> bool:
        """Return True when SMTP credentials are fully set in the environment.
        The UI uses this to decide whether to show the 'Send Email' button.
        """
        return self._cfg.is_complete

    def send_kyc_invite(
        self,
        to_email: str,
        customer_name: str,
        agent_name: str,
        session_url: str,
        date_str: str,
        time_str: str,
        duration_min: int,
        ics_bytes: Optional[bytes] = None,
    ) -> Tuple[bool, str]:
        """Send the KYC session invite to the customer.

        Attaches the .ics calendar file when ``ics_bytes`` is provided.
        The session link is active for 30 minutes from the time it was created
        (controlled by ``SESSION_TTL`` in ``basetruth.kyc.session``).

        Returns:
            (True, "")               on success
            (False, error_message)   on any failure — message is safe to show
                                     directly to the operator in the UI.

        Validates the recipient address format before making any network call
        so obviously-invalid inputs are rejected locally without SMTP overhead.
        """
        # Guard: SMTP not configured — tell the operator what env vars to set.
        if not self._cfg.is_complete:
            msg = (
                "SMTP is not configured. "
                "Set BT_SMTP_HOST, BT_SMTP_USER, and BT_SMTP_PASSWORD "
                "environment variables to enable direct email sending."
            )
            log.warning("KYC email skipped — SMTP not configured")
            return False, msg

        # Guard: sanity-check email format to avoid pointless SMTP round-trips.
        if not to_email or "@" not in to_email or "." not in to_email.split("@")[-1]:
            msg = f"Invalid recipient email address: {to_email!r}"
            log.warning(msg)
            return False, msg

        subject    = f"Your Video KYC Session — {date_str} at {time_str} IST"
        body_text  = self._build_email_body(
            customer_name, agent_name, session_url,
            date_str, time_str, duration_min,
        )

        # Build MIME message — mixed to allow the .ics attachment.
        outer = MIMEMultipart("mixed")
        outer["From"]    = self._cfg.from_addr
        outer["To"]      = to_email
        outer["Subject"] = subject
        outer.attach(MIMEText(body_text, "plain", "utf-8"))

        # Attach .ics calendar invite so the customer can add it to their
        # calendar app directly from the email.
        if ics_bytes:
            safe_name  = (customer_name.strip().replace(" ", "_") or "invite")
            attachment = MIMEBase("text", "calendar", method="REQUEST")
            attachment.set_payload(ics_bytes)
            encoders.encode_base64(attachment)
            attachment.add_header(
                "Content-Disposition",
                "attachment",
                filename=f"vkyc_{safe_name}.ics",
            )
            outer.attach(attachment)

        # Deliver the message — prefer STARTTLS; fall back to direct SSL when
        # BT_SMTP_SSL=1 is set (e.g. port 465).
        try:
            self._deliver(outer.as_string(), to_email)
            log.info(
                "KYC email invite sent",
                extra={"to": to_email, "session_url": session_url},
            )
            return True, ""

        except smtplib.SMTPAuthenticationError:
            msg = "SMTP authentication failed — check BT_SMTP_USER and BT_SMTP_PASSWORD."
            log.error(msg, extra={"to": to_email})
            return False, msg

        except smtplib.SMTPException as exc:
            msg = f"Failed to send email: {exc}"
            log.error(msg, extra={"to": to_email, "error": str(exc)})
            return False, msg

        except OSError as exc:
            # Covers connection refused, DNS failure, timeout, etc.
            msg = (
                f"Could not reach SMTP server "
                f"({self._cfg.host}:{self._cfg.port}): {exc}"
            )
            log.error(msg, extra={"to": to_email, "error": str(exc)})
            return False, msg

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _deliver(self, raw_message: str, to_email: str) -> None:
        """Open an SMTP connection and send ``raw_message``.
        Chooses SSL or STARTTLS based on ``KYCEmailConfig.use_ssl``.
        Raises ``smtplib.SMTPException`` or ``OSError`` on failure —
        the public ``send_kyc_invite`` method catches and translates these.
        """
        if self._cfg.use_ssl:
            # Direct SSL — used on port 465.
            with smtplib.SMTP_SSL(self._cfg.host, self._cfg.port) as smtp:
                smtp.login(self._cfg.user, self._cfg.password)
                smtp.sendmail(self._cfg.from_addr, to_email, raw_message)
        else:
            # STARTTLS — the standard for port 587 (Gmail, Outlook, etc.)
            with smtplib.SMTP(self._cfg.host, self._cfg.port) as smtp:
                smtp.ehlo()
                smtp.starttls()
                smtp.login(self._cfg.user, self._cfg.password)
                smtp.sendmail(self._cfg.from_addr, to_email, raw_message)

    @staticmethod
    def _build_email_body(
        customer_name: str,
        agent_name: str,
        session_url: str,
        date_str: str,
        time_str: str,
        duration_min: int,
    ) -> str:
        """Build the plain-text body of the KYC invite email.
        Plain text renders correctly in all email clients without needing HTML,
        and avoids spam filters that flag HTML-only messages.
        """
        return (
            f"Dear {customer_name},\n\n"
            "Your Video KYC session has been scheduled:\n\n"
            f"  Date & Time : {date_str} at {time_str} IST\n"
            f"  Duration    : {duration_min} minutes\n"
            f"  Join Link   : {session_url}\n\n"
            "Steps to complete your verification:\n"
            "  1. Open the Join Link above at the scheduled time\n"
            "  2. Upload your Aadhaar card (front) or PAN card\n"
            "  3. Upload your address proof (Aadhaar back or Passport back)\n"
            "  4. Allow location access so we can verify your current address\n"
            "  5. Follow the on-screen liveness prompts\n\n"
            "The entire process takes under 2 minutes and runs in your browser — "
            "no app or plugin needed.\n\n"
            "Note: The session link is active for 30 minutes from the time it was "
            "created. If the link has expired, please contact your agent.\n\n"
            f"Regards,\n{agent_name}\n\nPowered by BaseTruth AI"
        )
