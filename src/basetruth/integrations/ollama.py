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
- CRITICAL for names: Copy full_name and father_name letter-by-letter exactly as printed.
  Never substitute visually similar characters (do not change 'Hr' to 'Har', 'H' to 'N',
  or any other character). Indian names may have unusual sequences — do not correct them.
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
- CRITICAL for name: Copy the name character-by-character exactly as printed on the card.
  Never substitute visually similar letters (e.g. do not change 'Hr' to 'Har' or 'H' to 'N').
  Indian names can have unusual combinations — preserve them as-is without correction.
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


# ---------------------------------------------------------------------------
# Universal document analysis
# ---------------------------------------------------------------------------

UNIVERSAL_DOCUMENT_ANALYSIS_SYSTEM_PROMPT = (
    "You are a document fraud detection analyst specialising in Indian identity and "
    "financial documents. Analyse the document image thoroughly and return strict JSON "
    "only. Do not add commentary, explanations, or markdown outside the JSON object."
)

UNIVERSAL_DOCUMENT_ANALYSIS_PROMPT = """
Analyse this document image and return a JSON object with the following structure:
{
  "document_type": "",
  "document_subtype": "",
  "confidence": 0.0,
  "extracted_fields": {},
  "fraud_signals": [],
  "authenticity_assessment": {
    "verdict": "",
    "confidence": 0.0,
    "reasons": []
  },
  "summary": ""
}

Rules:
- document_type: identify the document category. Examples: "payslip", "bank_statement",
  "pan_card", "aadhaar", "offer_letter", "form16", "utility_bill", "property_agreement",
  "gift_letter", "employment_letter", "cancelled_cheque", "hospital_bill", "photograph",
  "passport", "driving_licence", "unknown".
- document_subtype: more specific description when available (e.g. "monthly_payslip",
  "salary_bank_statement", "front_side_pan").
- confidence: float 0.0–1.0 representing certainty about document_type.
- extracted_fields: flat key-value dict of ALL important fields found on the document
  (names, dates, amounts, account numbers, addresses, IDs, etc.).
- CRITICAL for person names: Transcribe every character exactly as it appears on the
  document — letter by letter, with no substitution. Do NOT replace visually similar
  characters (e.g. do not change 'Hr' to 'Har', or 'H' to 'N', or 'Kr' to 'Ka').
  Indian names can have unusual consonant clusters — preserve them precisely without
  correction, normalisation, or "fixing" of perceived spelling errors.
- fraud_signals: list of objects, each with keys "type" (string), "severity"
  ("high"/"medium"/"low"), and "description" (string). Include typography mismatches,
  font inconsistencies, resolution anomalies, missing official elements, impossible
  date combinations, suspicious rounded numbers, etc.
- authenticity_assessment.verdict: one of "authentic", "suspicious", "tampered", "unknown".
- authenticity_assessment.confidence: float 0.0–1.0.
- authenticity_assessment.reasons: list of plain-English strings explaining the verdict.
- summary: 2–3 sentence plain-English summary of the document including any concerns.
- Output strict JSON only — no markdown fences, no leading/trailing text.
""".strip()


def analyze_document_with_ollama(
    image_bytes: bytes,
    *,
    doc_hint: str = "",
    model: str | None = None,
    base_url: str | None = None,
) -> Dict[str, Any]:
    """Analyse any document image with Gemma4 via Ollama.

    Gemma4 classifies the document type, extracts key fields, identifies fraud
    signals, and produces an authentication verdict — all in one vision call.

    Parameters
    ----------
    image_bytes:
        Raw bytes of the image (JPEG, PNG, WebP, etc.).  For PDF sources the
        caller should convert the first page to an image before calling this.
    doc_hint:
        Optional free-text hint (e.g. ``"payslip"`` or ``"PAN card"``) that is
        prepended to the prompt to improve accuracy on known document types.
    model / base_url:
        Override the auto-detected Ollama endpoint/model when needed.

    Returns
    -------
    Dict with the keys documented in UNIVERSAL_DOCUMENT_ANALYSIS_PROMPT, plus
    ``engine``, ``model``, and ``raw_response``.  Returns ``{}`` when Ollama is
    unavailable or the response cannot be parsed.
    """
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

    prompt = UNIVERSAL_DOCUMENT_ANALYSIS_PROMPT
    if doc_hint:
        prompt = f"Document hint: {doc_hint}\n\n{prompt}"

    payload = {
        "model": resolved_model,
        "messages": [
            {"role": "system", "content": UNIVERSAL_DOCUMENT_ANALYSIS_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": prompt,
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
    json_text = _extract_json_object(content)
    if not json_text:
        return {}

    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError:
        return {}

    if not isinstance(parsed, dict):
        return {}

    parsed["engine"] = "gemma4_ollama"
    parsed["model"] = resolved_model or DEFAULT_OLLAMA_MODEL
    parsed["raw_response"] = content
    return parsed


# ---------------------------------------------------------------------------
# Batch document classification (one Ollama call for N files)
# ---------------------------------------------------------------------------

BATCH_CLASSIFICATION_PROMPT = """
You will be shown images of multiple documents. Identify each document type by carefully
observing its visual content, layout, headings, logos, and structure.

Return a JSON ARRAY with one object per document, in EXACTLY the same order as the
images were provided:

[
  {
    "index": 0,
    "document_type": "payslip",
    "is_image_based": false,
    "confidence": 0.95
  },
  ...
]

document_type:
  Identify the type as precisely as possible from the document itself — do NOT rely on
  filenames. Use concise snake_case (e.g. "payslip", "bank_statement", "pan_card",
  "aadhaar", "offer_letter", "form16", "utility_bill", "property_agreement",
  "gift_letter", "employment_letter", "cancelled_cheque", "hospital_bill",
  "photograph", "passport", "driving_licence", "marksheet", "degree_certificate",
  "insurance_policy", "loan_agreement", "tax_return", "salary_certificate").
  If truly unidentifiable, use "unknown".

is_image_based:
  true  = scanned / photographed document (no machine-readable text layer)
  false = digitally created document (PDF with embedded text, Word, Excel)

Output the JSON array ONLY — no markdown fences, no extra text.
""".strip()


def classify_documents_batch(
    image_bytes_list: List[bytes],
    filenames: List[str],
    *,
    model: str | None = None,
    base_url: str | None = None,
) -> List[Dict[str, Any]]:
    """Classify N documents with a SINGLE Gemma4 Ollama call.

    Parameters
    ----------
    image_bytes_list:
        Rasterised preview bytes (JPEG/PNG) for each document.  Entries that
        are empty/None are skipped and replaced with an "unknown" fallback.
    filenames:
        Display names corresponding to each entry (same length).

    Returns
    -------
    List of dicts — one per input in the same order — each containing:
      ``filename``, ``document_type``, ``is_image_based``, ``confidence``.
    On any failure the list is populated with "unknown" fallbacks.
    """
    fallback: List[Dict[str, Any]] = [
        {
            "filename": filenames[i] if i < len(filenames) else f"doc_{i}",
            "document_type": "unknown",
            "is_image_based": False,
            "confidence": 0.0,
        }
        for i in range(len(image_bytes_list))
    ]

    valid_pairs = [(i, img) for i, img in enumerate(image_bytes_list) if img]
    if not valid_pairs:
        return fallback

    resolved_base = base_url
    resolved_model = model
    if not resolved_base:
        resolved_base, models, _ = probe_ollama()
        if not resolved_base:
            return fallback
        resolved_model = resolved_model or select_ollama_model(models)
    elif not resolved_model:
        resolved_model = DEFAULT_OLLAMA_MODEL

    # List file names in the prompt so the model can reference them
    names_block = "\n".join(
        f"{i}. {filenames[i] if i < len(filenames) else f'document_{i}'}"
        for i, _ in valid_pairs
    )
    prompt = (
        f"Documents to classify (in order):\n{names_block}\n\n"
        f"{BATCH_CLASSIFICATION_PROMPT}"
    )

    payload: Dict[str, Any] = {
        "model": resolved_model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [
                    base64.b64encode(img).decode("ascii")
                    for _, img in valid_pairs
                ],
            }
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
        return fallback

    content = str(response.json().get("message", {}).get("content", "")).strip()

    # Parse the returned JSON array
    try:
        stripped = content.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
            stripped = re.sub(r"\s*```$", "", stripped)
        arr_start = stripped.find("[")
        arr_end = stripped.rfind("]")
        if arr_start == -1 or arr_end == -1:
            return fallback
        classifications: Any = json.loads(stripped[arr_start : arr_end + 1])
    except (json.JSONDecodeError, ValueError):
        return fallback

    if not isinstance(classifications, list):
        return fallback

    result = list(fallback)  # start with fallback entries
    valid_indices = [i for i, _ in valid_pairs]
    for item in classifications:
        if not isinstance(item, dict):
            continue
        idx = item.get("index")
        if not isinstance(idx, int) or idx < 0 or idx >= len(valid_indices):
            continue
        real_idx = valid_indices[idx]
        result[real_idx] = {
            "filename": filenames[real_idx] if real_idx < len(filenames) else f"doc_{real_idx}",
            "document_type": str(item.get("document_type", "unknown")),
            "is_image_based": bool(item.get("is_image_based", False)),
            "confidence": float(item.get("confidence", 0.0)),
        }
    return result