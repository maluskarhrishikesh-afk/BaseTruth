from __future__ import annotations

from basetruth.ui.pages.video_kyc import _normalize_external_api_url


def test_normalize_external_api_url_rewrites_localhost() -> None:
    assert _normalize_external_api_url("http://localhost:8000") == "http://127.0.0.1:8000"


def test_normalize_external_api_url_rewrites_ipv6_loopback() -> None:
    assert _normalize_external_api_url("http://[::1]:8000") == "http://127.0.0.1:8000"


def test_normalize_external_api_url_preserves_real_host() -> None:
    assert _normalize_external_api_url("https://demo.basetruth.ai") == "https://demo.basetruth.ai"