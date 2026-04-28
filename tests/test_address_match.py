"""Unit tests for src/basetruth/kyc/address_match.py.

All tests are purely computational — no network calls, no DB, no MinIO.
"""
from __future__ import annotations

import math

import pytest

from basetruth.kyc.address_match import (
    calculate_distance,
    compare_addresses,
    normalize_address,
    reverse_geocode,
)


# ---------------------------------------------------------------------------
# normalize_address
# ---------------------------------------------------------------------------

def test_normalize_address_extracts_pin() -> None:
    result = normalize_address("Flat 5, Shivaji Nagar, Pune 411005, Maharashtra")
    assert result["pin"] == "411005"


def test_normalize_address_maps_state_to_abbreviation() -> None:
    result = normalize_address("123 MG Road, Bengaluru 560001, Karnataka")
    assert result["state"] == "ka"


def test_normalize_address_removes_stop_words() -> None:
    result = normalize_address("Flat no 12, Near the temple, Mumbai 400001, Maharashtra")
    # 'near', 'the', 'no' are stop words and must be removed
    assert "near" not in result["tokens"]
    assert "the" not in result["tokens"]
    assert "no" not in result["tokens"]


def test_normalize_address_empty_string_returns_empty_result() -> None:
    result = normalize_address("")
    assert result["pin"] == ""
    assert result["state"] == ""
    assert result["tokens"] == set()


def test_normalize_address_returns_lowercase_tokens() -> None:
    result = normalize_address("Powai, MUMBAI")
    # All tokens must be lowercase
    assert all(t == t.lower() for t in result["tokens"])


# ---------------------------------------------------------------------------
# compare_addresses
# ---------------------------------------------------------------------------

def test_compare_addresses_match_when_same_pin_and_sufficient_overlap() -> None:
    proof = "Flat 5, Shivaji Nagar, Pune 411005, Maharashtra"
    live  = "Shivaji Nagar, Pune 411005, Maharashtra, India"
    result = compare_addresses(proof, live)
    assert result["result"] == "match"
    assert result["pin_match"] is True
    assert result["state_match"] is True
    assert result["jaccard"] > 0.0


def test_compare_addresses_mismatch_when_different_city_and_pin() -> None:
    proof = "Andheri West, Mumbai 400058, Maharashtra"
    live  = "MG Road, Bengaluru 560001, Karnataka"
    result = compare_addresses(proof, live)
    assert result["result"] == "mismatch"
    assert result["pin_match"] is False


def test_compare_addresses_skipped_when_either_is_empty() -> None:
    result = compare_addresses("", "Shivaji Nagar, Pune 411005")
    assert result["result"] == "skipped"

    result2 = compare_addresses("Shivaji Nagar, Pune 411005", "")
    assert result2["result"] == "skipped"


def test_compare_addresses_partial_when_same_state_and_moderate_overlap() -> None:
    # Same city and state, no PIN in live address
    proof = "Koregaon Park, Pune 411001, Maharashtra"
    live  = "Pune, Maharashtra"
    result = compare_addresses(proof, live)
    # Jaccard may be low but state match + some overlap → at least partial
    assert result["result"] in ("partial", "match")
    assert result["state_match"] is True


def test_compare_addresses_returns_jaccard_as_float() -> None:
    result = compare_addresses("Baner, Pune 411045", "Baner Pune 411045")
    assert isinstance(result["jaccard"], float)
    assert 0.0 <= result["jaccard"] <= 1.0


# ---------------------------------------------------------------------------
# calculate_distance
# ---------------------------------------------------------------------------

def test_calculate_distance_same_point_is_zero() -> None:
    dist = calculate_distance(18.5204, 73.8567, 18.5204, 73.8567)
    assert dist == pytest.approx(0.0, abs=1e-6)


def test_calculate_distance_known_reference() -> None:
    # Distance between Pune (18.5204° N, 73.8567° E) and Mumbai (19.0760° N, 72.8777° E)
    # is approximately 120 km.
    dist = calculate_distance(18.5204, 73.8567, 19.0760, 72.8777)
    assert 115_000 < dist < 130_000, f"Expected ~120 km, got {dist:.0f} m"


def test_calculate_distance_is_symmetric() -> None:
    # Distance A→B must equal B→A (Haversine is symmetric)
    d1 = calculate_distance(18.5204, 73.8567, 12.9716, 77.5946)
    d2 = calculate_distance(12.9716, 77.5946, 18.5204, 73.8567)
    assert d1 == pytest.approx(d2, rel=1e-6)


def test_calculate_distance_short_distance_accurate() -> None:
    # ~1 km at equatorial latitudes: longitude shift of ~0.009° ≈ 1 km
    dist = calculate_distance(0.0, 0.0, 0.0, 0.009)
    assert 900 < dist < 1100, f"Expected ~1 km, got {dist:.0f} m"


# ---------------------------------------------------------------------------
# reverse_geocode — must degrade gracefully on network failure
# ---------------------------------------------------------------------------

def test_reverse_geocode_returns_none_on_network_failure(monkeypatch) -> None:
    """reverse_geocode must return None (not raise) when the network is unavailable."""
    import urllib.request

    def fake_urlopen(*args, **kwargs):
        raise OSError("Network unreachable (test)")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    result = reverse_geocode(18.5204, 73.8567)
    assert result is None, "reverse_geocode must return None on network failure, not raise"


def test_reverse_geocode_returns_none_on_bad_json(monkeypatch) -> None:
    """reverse_geocode must return None if the API returns malformed JSON."""
    import io
    import urllib.request

    class _FakeResponse:
        def read(self):
            return b"not valid json {{{"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: _FakeResponse())

    result = reverse_geocode(18.5204, 73.8567)
    assert result is None
