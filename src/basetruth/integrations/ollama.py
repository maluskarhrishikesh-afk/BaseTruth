"""Shared Ollama helpers for Gemma4-powered BaseTruth features."""
from __future__ import annotations

import base64
import json
import os
import re
from typing import Any, Dict, List, Sequence

import requests

DEFAULT_OLLAMA_MODEL = "gemma4:latest"
DEFAULT_OLLAMA_BASES = (
    "http://localhost:11434",
    "http://host.docker.internal:11434",
)
OLLAMA_CONNECT_TIMEOUT_SEC = 5
OLLAMA_READ_TIMEOUT_SEC = 600

_EMPTY_FIELD_MARKERS = {"", "null", "none", "n/a", "na", "unknown", "not visible"}
_PAN_FIELD_NAMES = ("pan_number", "pan", "pan_no", "pan_card_number", "panNumber")
_NAME_FIELD_NAMES = ("full_name", "name", "cardholder_name", "holder_name", "applicant_name")
_FATHER_FIELD_NAMES = (
    "father_name",
    "fathers_name",
    "fatherName",
    "father",
    "parent_name",
)
_DOB_FIELD_NAMES = (
    "date_of_birth",
    "dob",
    "birth_date",
    "dateOfBirth",
    "date_of_birth_or_incorporation",
)

PAN_EXTRACTION_SYSTEM_PROMPT = (
    "You extract structured fields from Indian PAN card images. "
    "Return strict JSON only and do not add commentary."
)

PAN_EXTRACTION_PROMPT = """
Read this PAN card image and return a JSON object with exactly these keys:
{
  "pan_number": "",
  "full_name": "",
  "father_name": "",
  "date_of_birth": ""
}

Rules:
- Preserve the card text as written.
- PAN number must use the 5 letters, 4 digits, 1 letter format if visible.
- Do not guess missing values.
- If a field is not visible, return an empty string.
- Output JSON only.
""".strip()


AADHAAR_EXTRACTION_SYSTEM_PROMPT = (
    "You extract structured fields from Indian Aadhaar card images. "
    "Return strict JSON only and do not add commentary."
)


AADHAAR_EXTRACTION_PROMPT = """
Read this Aadhaar card image and return a JSON object with exactly these keys:
{
  "uid": "",
  "name": "",
  "dob": "",
  "yob": "",
  "gender": ""
}

Rules:
- Preserve the card text as written.
- uid must be exactly 12 digits, strip spaces.
- If a field is not visible, return an empty string.
- Output JSON only.
""".strip()


def candidate_ollama_bases() -> List[str]:
    """Return possible Ollama base URLs in the order most likely to work."""
    env_base = os.getenv("OLLAMA_BASE_URL", "").strip().rstrip("/")
    candidates: List[str] = []
    if env_base:
        candidates.append(env_base)

    if os.path.exists("/.dockerenv"):
        candidates.extend([DEFAULT_OLLAMA_BASES[1], DEFAULT_OLLAMA_BASES[0]])
    else:
        candidates.extend(DEFAULT_OLLAMA_BASES)

    unique_candidates: List[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate and candidate not in seen:
            unique_candidates.append(candidate)
            seen.add(candidate)
    return unique_candidates


def probe_ollama() -> tuple[str | None, List[str], List[str]]:
    """Find a reachable Ollama endpoint and return its models and attempted URLs."""
    attempted = candidate_ollama_bases()
    for base_url in attempted:
        try:
            response = requests.get(
                f"{base_url}/api/tags",
                timeout=OLLAMA_CONNECT_TIMEOUT_SEC,
            )  # nosemgrep: basetruth-ssrf
            response.raise_for_status()
            models = [model["name"] for model in response.json().get("models", [])]
            models.sort(key=lambda name: (0 if "gemma4" in name.lower() else 1, name))
            return base_url, (models or [DEFAULT_OLLAMA_MODEL]), attempted
        except requests.RequestException:
            continue
    return None, [DEFAULT_OLLAMA_MODEL], attempted


def select_ollama_model(
    models: Sequence[str],
    preferred_substring: str = "gemma4",
) -> str:
    """Return the preferred Ollama model, favouring Gemma4 when available."""
    preferred = preferred_substring.lower().strip()
    for name in models:
        if preferred and preferred in name.lower():
            return name
    return models[0] if models else DEFAULT_OLLAMA_MODEL


def _extract_json_object(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end < start:
        return ""
    return stripped[start:end + 1]


def _clean_field(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    normalized = re.sub(r"\s+", " ", text)
    if normalized.lower() in _EMPTY_FIELD_MARKERS:
        return ""
    return normalized


def _normalize_pan(value: Any) -> str:
    candidate = _clean_field(value).upper().replace(" ", "")
    match = re.search(r"[A-Z]{5}[0-9]{4}[A-Z]", candidate)
    return match.group(0) if match else ""


def _candidate_payloads(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    candidates = [payload]
    for key in ("fields", "data", "result"):
        nested = payload.get(key)
        if isinstance(nested, dict):
            candidates.insert(0, nested)
    return candidates


def _pick_field(payloads: Sequence[Dict[str, Any]], names: Sequence[str]) -> str:
    for payload in payloads:
        for name in names:
            if name in payload:
                cleaned = _clean_field(payload.get(name))
                if cleaned:
                    return cleaned
    return ""


def parse_pan_response_content(content: str) -> Dict[str, str]:
    """Parse a Gemma4 PAN extraction response into normalized fields."""
    json_text = _extract_json_object(content)
    if not json_text:
        return {}

    try:
        payload = json.loads(json_text)
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}

    payloads = _candidate_payloads(payload)
    pan_number = ""
    for candidate in payloads:
        pan_number = _normalize_pan(_pick_field([candidate], _PAN_FIELD_NAMES))
        if pan_number:
            break

    full_name = _pick_field(payloads, _NAME_FIELD_NAMES)
    father_name = _pick_field(payloads, _FATHER_FIELD_NAMES)
    date_of_birth = _pick_field(payloads, _DOB_FIELD_NAMES)

    parsed: Dict[str, str] = {}
    if pan_number:
        parsed["pan_number"] = pan_number
    if full_name:
        parsed["full_name"] = full_name
        parsed["name"] = full_name
    if father_name:
        parsed["father_name"] = father_name
    if date_of_birth:
        parsed["date_of_birth"] = date_of_birth
    return parsed


def parse_aadhaar_response_content(content: str) -> Dict[str, str]:
    """Parse a Gemma4 Aadhaar extraction response into normalized fields."""
    json_text = _extract_json_object(content)
    if not json_text:
        return {}

    try:
        payload = json.loads(json_text)
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}

    payloads = _candidate_payloads(payload)

    uid = _pick_field(payloads, ("uid", "aadhaar", "aadhaar_number", "aadhar_number"))
    if uid:
        uid = re.sub(r"[^\d]", "", uid)

    name = _pick_field(payloads, ("name", "full_name"))
    dob = _pick_field(payloads, ("dob", "date_of_birth", "birth_date"))
    yob = _pick_field(payloads, ("yob", "year_of_birth"))
    gender = _pick_field(payloads, ("gender", "sex"))

    parsed: Dict[str, str] = {}
    if uid:
        parsed["uid"] = uid
    if name:
        parsed["name"] = name
    if dob:
        parsed["dob"] = dob
    if yob:
        parsed["yob"] = yob
    if gender:
        parsed["gender"] = gender
    return parsed


def extract_aadhaar_details_with_ollama(
    image_bytes: bytes,
    *,
    model: str | None = None,
    base_url: str | None = None,
) -> Dict[str, Any]:
    """Extract structured Aadhaar fields from an image using Gemma4 via Ollama."""
    if not image_bytes:
        return {}

    resolved_base = base_url
    resolved_model = model
    if not resolved_base:
        resolved_base, models, _ = probe_ollama()
        if not resolved_base:
            return {}
        resolved_model = resolved_model or select_ollama_model(models)
    elif not resolved_model:
        resolved_model = DEFAULT_OLLAMA_MODEL

    payload = {
        "model": resolved_model,
        "messages": [
            {"role": "system", "content": AADHAAR_EXTRACTION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": AADHAAR_EXTRACTION_PROMPT,
                "images": [base64.b64encode(image_bytes).decode("ascii")],
            },
        ],
        "stream": False,
        "options": {"temperature": 0},
    }

    try:
        response = requests.post(
            f"{resolved_base}/api/chat",
            json=payload,
            timeout=(OLLAMA_CONNECT_TIMEOUT_SEC, OLLAMA_READ_TIMEOUT_SEC),
        )  # nosemgrep: basetruth-ssrf
        response.raise_for_status()
    except requests.RequestException:
        return {}

    content = str(response.json().get("message", {}).get("content", "")).strip()
    parsed = parse_aadhaar_response_content(content)
    if not parsed:
        return {}
    parsed["engine"] = "gemma4_ollama"
    parsed["model"] = resolved_model or DEFAULT_OLLAMA_MODEL
    parsed["base_url"] = resolved_base or ""
    parsed["raw_response"] = content
    parsed["qr_type"] = "gemma4"
    return parsed


def extract_pan_details_with_ollama(
    image_bytes: bytes,
    *,
    model: str | None = None,
    base_url: str | None = None,
) -> Dict[str, Any]:
    """Extract structured PAN fields from an image using Gemma4 via Ollama."""
    if not image_bytes:
        return {}

    resolved_base = base_url
    resolved_model = model
    if not resolved_base:
        resolved_base, models, _ = probe_ollama()
        if not resolved_base:
            return {}
        resolved_model = resolved_model or select_ollama_model(models)
    elif not resolved_model:
        resolved_model = DEFAULT_OLLAMA_MODEL

    payload = {
        "model": resolved_model,
        "messages": [
            {"role": "system", "content": PAN_EXTRACTION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": PAN_EXTRACTION_PROMPT,
                "images": [base64.b64encode(image_bytes).decode("ascii")],
            },
        ],
        "stream": False,
        "options": {"temperature": 0},
    }

    try:
        response = requests.post(
            f"{resolved_base}/api/chat",
            json=payload,
            timeout=(OLLAMA_CONNECT_TIMEOUT_SEC, OLLAMA_READ_TIMEOUT_SEC),
        )  # nosemgrep: basetruth-ssrf
        response.raise_for_status()
    except requests.RequestException:
        return {}

    content = str(response.json().get("message", {}).get("content", "")).strip()
    parsed = parse_pan_response_content(content)
    if not parsed:
        return {}
    parsed["engine"] = "gemma4_ollama"
    parsed["model"] = resolved_model or DEFAULT_OLLAMA_MODEL
    parsed["base_url"] = resolved_base or ""
    parsed["raw_response"] = content
    return parsed