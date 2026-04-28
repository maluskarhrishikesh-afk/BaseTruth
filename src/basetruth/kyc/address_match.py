"""Address matching utilities for Video KYC.

Compares the address on an applicant's address proof document against the
live GPS location captured during the KYC session.  All external lookups
(reverse-geocoding) degrade gracefully so the rest of the KYC flow is never
blocked by a network failure or missing API key.
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, Optional

from basetruth.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Address normalisation
# ---------------------------------------------------------------------------

# State name synonyms used in Indian addresses (full name → abbreviation)
_STATE_MAP: Dict[str, str] = {
    "maharashtra": "mh",
    "karnataka": "ka",
    "delhi": "dl",
    "new delhi": "dl",
    "uttar pradesh": "up",
    "gujarat": "gj",
    "rajasthan": "rj",
    "tamil nadu": "tn",
    "west bengal": "wb",
    "andhra pradesh": "ap",
    "telangana": "tg",
    "kerala": "kl",
    "madhya pradesh": "mp",
    "bihar": "br",
    "haryana": "hr",
    "punjab": "pb",
    "odisha": "or",
    "jharkhand": "jh",
    "uttarakhand": "uk",
    "himachal pradesh": "hp",
    "goa": "ga",
    "assam": "as",
    "chhattisgarh": "cg",
    "jammu and kashmir": "jk",
    "tripura": "tr",
    "manipur": "mn",
    "meghalaya": "ml",
    "nagaland": "nl",
    "arunachal pradesh": "ar",
    "mizoram": "mz",
    "sikkim": "sk",
}

# Words that carry no discriminative power when comparing two addresses
_STOP_WORDS = frozenset({
    "near", "opp", "opposite", "behind", "beside", "next", "to", "the",
    "and", "of", "at", "in", "on", "by", "no", "house", "flat", "room",
    "floor", "building", "block", "wing", "sector", "phase", "street",
    "road", "lane", "marg", "nagar", "colony", "society", "plot", "survey",
    "s", "no", "pincode", "pin", "india",
})


def normalize_address(text: str) -> Dict[str, Any]:
    """Parse and normalise a raw address string into a structured dict.

    Extracts the PIN code, reduces noise words, and maps state names to
    their two-letter codes so two addresses from different sources (Aadhaar
    QR, manual entry, reverse-geocode) can be compared fairly.

    Returns a dict with keys:
      - ``tokens``   : deduplicated meaningful words (set, lowercased)
      - ``pin``      : 6-digit PIN/ZIP code as a string, or ``""``
      - ``state``    : two-letter state abbreviation, or ``""``
      - ``raw``      : cleaned lowercase version of the input
    """
    if not text:
        return {"tokens": set(), "pin": "", "state": "", "raw": ""}

    # Lowercase and strip extra whitespace
    clean = text.lower().strip()

    # Extract PIN code (6 consecutive digits) before we start stripping numbers
    pin_match = re.search(r"\b(\d{6})\b", clean)
    pin = pin_match.group(1) if pin_match else ""

    # Remove punctuation and extra spaces
    clean = re.sub(r"[^a-z0-9 ]", " ", clean)
    clean = re.sub(r"\s+", " ", clean).strip()

    # Tokenise and remove stop words
    raw_tokens = clean.split()
    tokens = {t for t in raw_tokens if t not in _STOP_WORDS and len(t) > 1}

    # Detect state — check if any state name appears as a substring
    detected_state = ""
    for full, abbr in _STATE_MAP.items():
        if full in clean:
            detected_state = abbr
            # Remove the long state name to avoid it dominating the token overlap
            tokens -= set(full.split())
            tokens.add(abbr)
            break
    if not detected_state:
        # Already using abbreviation?
        for abbr in _STATE_MAP.values():
            if abbr in tokens:
                detected_state = abbr
                break

    # Remove the PIN from tokens so it doesn't skew the similarity score
    if pin:
        tokens.discard(pin)

    return {
        "tokens": tokens,
        "pin":    pin,
        "state":  detected_state,
        "raw":    clean,
    }


# ---------------------------------------------------------------------------
# Address comparison
# ---------------------------------------------------------------------------


def compare_addresses(proof_addr: str, live_addr: str) -> Dict[str, Any]:
    """Compare an address from an identity document against a live address.

    The live address typically comes from reverse-geocoding the user's GPS
    coordinates captured during the Video KYC session.  The proof address
    comes from the Aadhaar QR / Passport fields.

    Both strings are normalised then compared by:
      1. PIN code exact match (strong signal)
      2. State code exact match (medium signal)
      3. Jaccard similarity of meaningful address tokens (continuous signal)

    Returns a dict with keys:
      - ``result``      : 'match' | 'partial' | 'mismatch' | 'skipped'
      - ``jaccard``     : float 0–1 token overlap score
      - ``pin_match``   : bool — True if both have the same 6-digit PIN
      - ``state_match`` : bool — True if both resolve to the same state code
      - ``proof_norm``  : normalised proof address string
      - ``live_norm``   : normalised live address string
    """
    if not proof_addr or not live_addr:
        log.debug(
            "compare_addresses: skipped — one or both addresses are empty"
        )
        return {
            "result": "skipped",
            "jaccard": 0.0,
            "pin_match": False,
            "state_match": False,
            "proof_norm": proof_addr or "",
            "live_norm": live_addr or "",
        }

    proof = normalize_address(proof_addr)
    live  = normalize_address(live_addr)

    # PIN code comparison — exact match is a strong location signal
    pin_match = bool(proof["pin"] and live["pin"] and proof["pin"] == live["pin"])

    # State code comparison
    state_match = bool(
        proof["state"] and live["state"] and proof["state"] == live["state"]
    )

    # Jaccard similarity: |intersection| / |union| across meaningful tokens
    intersection = proof["tokens"] & live["tokens"]
    union        = proof["tokens"] | live["tokens"]
    jaccard = len(intersection) / len(union) if union else 0.0

    # Determine overall result from combined signals
    if pin_match and jaccard >= 0.4:
        result = "match"
    elif pin_match or (state_match and jaccard >= 0.3):
        result = "partial"
    elif jaccard >= 0.5:
        # High token overlap even without PIN suggests the same locality
        result = "partial"
    else:
        result = "mismatch"

    log.debug(
        "compare_addresses: result=%s jaccard=%.2f pin=%s state=%s",
        result, jaccard, pin_match, state_match,
    )
    return {
        "result":      result,
        "jaccard":     round(jaccard, 4),
        "pin_match":   pin_match,
        "state_match": state_match,
        "proof_norm":  proof["raw"],
        "live_norm":   live["raw"],
    }


# ---------------------------------------------------------------------------
# Distance calculation (Haversine formula)
# ---------------------------------------------------------------------------

_EARTH_RADIUS_M = 6_371_000  # metres


def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return the great-circle distance between two GPS points in metres.

    Uses the Haversine formula which gives a good approximation for the
    short-to-medium distances typical in address verification (~99.9 % accurate
    within a city, sub-percent error up to ~500 km).

    Parameters
    ----------
    lat1, lon1 : Latitude/longitude of the first point (degrees).
    lat2, lon2 : Latitude/longitude of the second point (degrees).

    Returns
    -------
    Distance in metres (float).
    """
    # Convert degrees to radians — the Haversine formula needs radians
    phi1    = math.radians(lat1)
    phi2    = math.radians(lat2)
    d_phi   = math.radians(lat2 - lat1)   # delta latitude in radians
    d_lam   = math.radians(lon2 - lon1)   # delta longitude in radians

    # Haversine formula: a = sin²(Δφ/2) + cos(φ1)·cos(φ2)·sin²(Δλ/2)
    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lam / 2) ** 2
    )

    # c = 2·atan2(√a, √(1−a)) — the central angle in radians
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return _EARTH_RADIUS_M * c


# ---------------------------------------------------------------------------
# Reverse geocoding (graceful degradation)
# ---------------------------------------------------------------------------


def reverse_geocode(lat: float, lon: float) -> Optional[str]:
    """Convert GPS coordinates to a human-readable address string.

    Uses the free Nominatim API (OpenStreetMap).  No API key is required but
    a reasonable User-Agent is set so OSM can contact us if we abuse the
    service.

    Returns the display_name string from Nominatim on success, or ``None``
    when the request fails (network error, timeout, non-200 response, or
    missing key in the response body).  The caller must handle ``None``
    gracefully — this function must never raise.
    """
    try:
        import urllib.request  # noqa: PLC0415 — lazy import to avoid startup cost
        import json as _json

        url = (
            f"https://nominatim.openstreetmap.org/reverse"
            f"?lat={lat}&lon={lon}&format=json"
        )
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "BaseTruth/1.0 (identity verification platform)"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:  # 5-second timeout
            data = _json.loads(resp.read().decode())
        address = data.get("display_name")
        if address:
            log.debug("reverse_geocode: %.4f, %.4f → %s", lat, lon, address[:80])
        return address
    except Exception as exc:  # network failure, timeout, JSON error, etc.
        log.warning("reverse_geocode failed (non-fatal): %s", exc)
        return None


def geocode_address(address_text: str) -> Optional[tuple]:
    """Forward-geocode a free-text address string to a (lat, lon) tuple.

    Used to calculate the GPS distance between the customer's live location
    and the registered address on their proof document.  The result lets us
    apply the 500 m physical-proximity rule on top of pure text comparison.

    Nominatim (OpenStreetMap) is queried with no API key.  Results for
    Indian addresses (district / taluka specifics) have variable success
    rates, so the caller must always treat None as a graceful skip and fall
    back to text-only comparison.

    Returns a (lat, lon) tuple on success, or None on any failure.  Never
    raises — all exceptions are caught and logged at DEBUG level.
    """
    if not address_text or len(address_text.strip()) < 5:
        return None
    try:
        import json as _json  # noqa: PLC0415
        import urllib.parse  # noqa: PLC0415
        import urllib.request  # noqa: PLC0415

        # Build a Nominatim search URL with a 1-result limit for speed
        params = urllib.parse.urlencode(
            {"q": address_text, "format": "json", "limit": "1"}
        )
        url = f"https://nominatim.openstreetmap.org/search?{params}"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "BaseTruth/1.0 (identity verification platform)"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = _json.loads(resp.read().decode())
        if data:
            lat = float(data[0]["lat"])
            lon = float(data[0]["lon"])
            log.debug("geocode_address: '%s…' → (%.4f, %.4f)", address_text[:40], lat, lon)
            return lat, lon
    except Exception as exc:  # network, timeout, JSON, key error — all non-fatal
        log.debug("geocode_address failed (non-critical): %s", exc)
    return None
