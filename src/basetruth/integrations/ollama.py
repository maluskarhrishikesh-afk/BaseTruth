"""Shared Ollama helpers and VLM routing for BaseTruth features.

This module owns the *routing* logic — reading settings.json, resolving which
provider and model to use for each feature, and probing Ollama for a live
endpoint.  The actual HTTP calls to each provider are delegated to VLMClient
in llm_client.py so that provider-specific code stays in one place.
"""
from __future__ import annotations

import json
import os
import pathlib
import re
from typing import Any, Dict, List, Sequence

import requests

from basetruth.logger import get_logger
from basetruth.integrations.llm_client import (
    VLMClient,
    OLLAMA_CONNECT_TIMEOUT_SEC,
    OLLAMA_READ_TIMEOUT_SEC,
)

log = get_logger(__name__)


def _log_vlm_request(
    operation: str,
    *,
    feature: str | None,
    model: str | None,
    base_url: str | None,
    system_prompt: str,
    user_prompt: str,
    image_bytes_list: List[bytes],
) -> None:
    """Log the exact request we are sending to a vision model.

    The log intentionally includes the full system and user prompts so operators
    can reproduce the same call during debugging. Images are logged by count and
    byte size only because dumping binary image data would make the logs unreadable.
    """
    log.info(
        "%s: sending VLM request | feature=%s model=%s base_url=%s image_count=%d image_sizes=%s",
        operation,
        feature or "",
        model or "",
        base_url or "",
        len(image_bytes_list),
        [len(img) for img in image_bytes_list],
    )
    log.info("%s: exact system prompt follows:\n%s", operation, system_prompt)
    log.info("%s: exact user prompt follows:\n%s", operation, user_prompt)


def _log_vlm_response(
    operation: str,
    *,
    engine: str,
    model: str,
    base_url: str,
    content: str,
) -> None:
    """Log the exact response returned by a vision model."""
    log.info(
        "%s: received VLM response | engine=%s model=%s base_url=%s response_len=%d",
        operation,
        engine,
        model,
        base_url,
        len(content),
    )
    log.info("%s: exact raw VLM response follows:\n%s", operation, content)

# Path to the project-level LLM configuration file.  Resolved relative to the
# current working directory so it works when Streamlit launches from the project
# root AND inside Docker where the project root is the container cwd.
_LLM_CONFIG_PATH = pathlib.Path("artifacts/config/settings.json")

# Fallback model used when no model is configured.  "gemma4:e2b" is the efficient
# 2B-parameter variant that runs comfortably on a 16 GB RAM machine.  Switch to
# "gemma4:latest" for higher accuracy when more RAM / VRAM is available.
DEFAULT_OLLAMA_MODEL = "gemma4:e2b"
DEFAULT_OLLAMA_BASES = (
    "http://localhost:11434",
    "http://host.docker.internal:11434",
)

# Shorter timeout used only for the quick document-type classification call.
# 45 s is enough for a small yes/no classification; the full extraction calls
# keep their 1200 s timeout because they return much more data.
_DOC_CLASSIFY_READ_TIMEOUT_SEC = 45

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


def _load_llm_config() -> Dict[str, Any]:
    """Read LLM settings from artifacts/config/settings.json.

    Returns an empty dict when the file is missing or malformed so that every
    caller can safely fall back to its hardcoded defaults without crashing.
    """
    try:
        return json.loads(_LLM_CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


# Reserved key inside the 'providers' block that holds per-feature model routing.
# Keeping it as a constant means a single place to change if the key is ever renamed.
_FEATURE_MODELS_KEY = "feature_models"


def _get_configured_model() -> str:
    """Return the Ollama model name to use as a global fallback, in priority order.

    Priority:
      1. OLLAMA_MODEL environment variable  — highest priority (Docker / CI)
      2. providers.ollama.model in settings.json — operator-editable
      3. DEFAULT_OLLAMA_MODEL — hardcoded fallback
    """
    # 1. Environment variable overrides everything
    env_model = os.getenv("OLLAMA_MODEL", "").strip()
    if env_model:
        return env_model
    # 2. Ollama provider's default model from the providers block
    ollama_provider = _load_llm_config().get("providers", {}).get("ollama", {})
    configured = str(ollama_provider.get("model", "")).strip()
    if configured:
        return configured
    # 3. Hardcoded fallback
    return DEFAULT_OLLAMA_MODEL


def get_model_for_feature(feature: str) -> str:
    """Return the Ollama model configured for a specific application feature.

    Handles both the legacy string format and the new object format in settings.json:
      Legacy:  "document_extraction": "gemma4:e4b"
      Object:  "document_extraction": {"provider": "ollama", "model": "gemma4:e4b"}

    Priority:
      1. settings.json 'models.<feature>' key  — per-feature override
      2. OLLAMA_MODEL env var                   — global env override
      3. settings.json 'ollama_model'           — global settings override
      4. DEFAULT_OLLAMA_MODEL                   — hardcoded fallback
    """
    models_section = _load_llm_config().get("providers", {}).get(_FEATURE_MODELS_KEY, {})
    if isinstance(models_section, dict):
        feature_entry = models_section.get(feature, "")
        # Support both string ("gemma4:e4b") and object ({"provider": "ollama", "model": "..."})
        if isinstance(feature_entry, dict):
            feature_model = str(feature_entry.get("model", "")).strip()
        elif isinstance(feature_entry, str):
            feature_model = feature_entry.strip()
        else:
            feature_model = ""
        if feature_model and not feature_model.startswith("_"):
            log.debug("get_model_for_feature: feature=%s → model=%s (from settings.json)", feature, feature_model)
            return feature_model
    return _get_configured_model()


def get_provider_config_for_feature(feature: str) -> Dict[str, Any]:
    """Return the full provider + model + connection config for a specific feature.

    Each BaseTruth feature can be routed to a different LLM provider. The 'models'
    block in settings.json supports two formats:

      Legacy string (Ollama only):
        "document_extraction": "gemma4:e4b"

      Object format (any provider):
        "qna_copilot": {"provider": "github_models", "model": "gpt-4o-mini"}
        "document_extraction": {"provider": "ollama", "model": "gemma4:e4b"}

    Returns a dict with:
      provider  — 'ollama', 'github_models', 'openai', 'anthropic'
      model     — model name to pass to the API
      base_url  — API base URL (empty for ollama; resolved at probe time)
      api_key   — API key (empty for ollama)

    Configure in artifacts/config/settings.json under 'models'.
    Provider connection details (base_url, api_key) come from 'providers.<name>'.
    """
    cfg = _load_llm_config()
    models_section = cfg.get("providers", {}).get(_FEATURE_MODELS_KEY, {})
    providers_section = cfg.get("providers", {})

    feature_entry = (models_section or {}).get(feature, "") if isinstance(models_section, dict) else ""

    if isinstance(feature_entry, dict) and feature_entry and "provider" in feature_entry:
        # Explicit {provider, model} object — highest priority
        provider_name = str(feature_entry.get("provider", "ollama")).lower().strip()
        model_name = str(feature_entry.get("model", "")).strip()
        model_explicitly_configured = True
    elif isinstance(feature_entry, str) and feature_entry and not feature_entry.startswith("_"):
        # Legacy string — treat as an Ollama model name
        provider_name = "ollama"
        model_name = feature_entry.strip()
        model_explicitly_configured = True
    else:
        # No feature-specific config — fall back to global active_provider.
        # Leave model_name empty so Ollama callers use select_ollama_model()
        # against the probe_ollama() list (which already honours OLLAMA_MODEL
        # env var and providers.ollama.model in settings.json internally).
        provider_name = get_active_provider()
        model_name = ""
        model_explicitly_configured = False

    # Ollama: base_url and api_key are resolved later via probe_ollama()
    if provider_name == "ollama":
        # Use the explicitly configured model when set; otherwise return empty
        # so that _route_vlm_chat falls back to select_ollama_model(probe_list).
        resolved_model = model_name if model_explicitly_configured else ""
        log.debug(
            "get_provider_config_for_feature: feature=%s → provider=ollama model=%s "
            "(explicit=%s)",
            feature, resolved_model or "(probe selection)", model_explicitly_configured,
        )
        return {"provider": "ollama", "model": resolved_model, "base_url": "", "api_key": ""}

    # Cloud provider — look up connection details from the 'providers' section
    provider_cfg = dict(providers_section.get(provider_name, {}))
    resolved_model = model_name or provider_cfg.get("model", "")
    resolved_base_url = provider_cfg.get("base_url", "")
    resolved_api_key = provider_cfg.get("api_key", "")

    log.debug(
        "get_provider_config_for_feature: feature=%s → provider=%s model=%s base_url=%s",
        feature, provider_name, resolved_model, resolved_base_url,
    )
    return {
        "provider": provider_name,
        "model": resolved_model,
        "base_url": resolved_base_url,
        "api_key": resolved_api_key,
    }


def get_active_provider() -> str:
    """Return the currently configured VLM provider from settings.json.

    Possible values: 'ollama' (default/local), 'github_models', 'openai', 'anthropic'.
    Falls back to 'ollama' when the config file is missing or the key is absent.
    Change 'active_provider' in artifacts/config/settings.json to switch providers.
    """
    return str(_load_llm_config().get("active_provider", "ollama") or "ollama").strip().lower()


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
            return base_url, (models or [_get_configured_model()]), attempted
        except requests.RequestException:
            continue
    return None, [_get_configured_model()], attempted


def select_ollama_model(
    models: Sequence[str],
    preferred_substring: str = "gemma4",
) -> str:
    """Return the preferred Ollama model, favouring the configured model above all else.

    Priority:
      1. Exact match on the configured model name (e.g. 'gemma4:e2b')
      2. Any model whose name contains preferred_substring (e.g. 'gemma4')
      3. The first model in the list
      4. The configured model name as a last-resort fallback
    """
    configured = _get_configured_model()
    # 1. Exact match — use the configured model if it is installed on this Ollama instance
    for name in models:
        if name.lower() == configured.lower():
            return name
    # 2. Substring match — any model with 'gemma4' (or the given substring) in its name
    preferred = preferred_substring.lower().strip()
    for name in models:
        if preferred and preferred in name.lower():
            return name
    # 3. Fall back to whatever Ollama has, or the configured name if list is empty
    return models[0] if models else configured


def _ollama_vlm_chat(
    system_prompt: str,
    user_prompt: str,
    image_bytes_list: List[bytes],
    *,
    base_url: str,
    model: str,
    timeout_sec: int = OLLAMA_READ_TIMEOUT_SEC,
) -> tuple[str, str, str, str]:
    """POST a vision chat request to the local Ollama API.

    Thin wrapper around VLMClient._call_ollama that preserves the original
    function signature for backward compatibility with call sites that already
    have a resolved Ollama base_url and model name.
    Returns (content, 'gemma4_ollama', model_name, base_url).
    """
    client = VLMClient("ollama", model, base_url=base_url)
    return client.chat_vision(
        system_prompt, user_prompt, image_bytes_list, timeout_sec=timeout_sec
    )


def _openai_compatible_vlm_chat(
    system_prompt: str,
    user_prompt: str,
    image_bytes_list: List[bytes],
    provider_cfg: Dict[str, Any],
    *,
    timeout_sec: int = OLLAMA_READ_TIMEOUT_SEC,
) -> tuple[str, str, str]:
    """POST a vision chat request to an OpenAI-compatible endpoint.

    Thin wrapper around VLMClient._call_openai_compatible that accepts the
    provider_cfg dict format used throughout the routing layer.  Preserves the
    original 3-tuple return type (content, engine, model) for backward compat.
    Returns (content, 'openai_compatible', model_name).
    """
    client = VLMClient(
        "openai_compatible",
        str(provider_cfg.get("model", "gpt-4o-mini")),
        base_url=str(provider_cfg.get("base_url", "")).rstrip("/"),
        api_key=str(provider_cfg.get("api_key", "")).strip(),
    )
    content, engine, model, _base = client.chat_vision(
        system_prompt, user_prompt, image_bytes_list, timeout_sec=timeout_sec
    )
    return content, engine, model


def _anthropic_vlm_chat(
    system_prompt: str,
    user_prompt: str,
    image_bytes_list: List[bytes],
    provider_cfg: Dict[str, Any],
    *,
    timeout_sec: int = OLLAMA_READ_TIMEOUT_SEC,
) -> tuple[str, str, str]:
    """POST a vision chat request to the Anthropic Messages API.

    Thin wrapper around VLMClient._call_anthropic that accepts the provider_cfg
    dict format used throughout the routing layer.  Preserves the original
    3-tuple return type (content, engine, model) for backward compat.
    Returns (content, 'anthropic', model_name).
    """
    client = VLMClient(
        "anthropic",
        str(provider_cfg.get("model", "claude-3-5-haiku-20241022")),
        api_key=str(provider_cfg.get("api_key", "")).strip(),
    )
    content, engine, model, _base = client.chat_vision(
        system_prompt, user_prompt, image_bytes_list, timeout_sec=timeout_sec
    )
    return content, engine, model


def _route_vlm_chat(
    system_prompt: str,
    user_prompt: str,
    image_bytes_list: List[bytes],
    *,
    timeout_sec: int = OLLAMA_READ_TIMEOUT_SEC,
    # Caller-supplied Ollama overrides skip provider selection entirely so we
    # do not re-probe Ollama on every function call when the endpoint was
    # already resolved by the caller.
    ollama_base_url: str | None = None,
    ollama_model: str | None = None,
    # Per-feature routing: when set, reads feature_models config from settings.json
    # so individual features (pan_extraction, aadhaar_extraction, etc.) can be
    # routed to different providers without changing the global active_provider.
    feature: str | None = None,
) -> tuple[str, str, str, str]:
    """Route a VLM vision request to the configured provider.

    Routing priority (highest to lowest):
      1. Explicit ollama_base_url / ollama_model caller overrides.
      2. Per-feature config from providers.feature_models.<feature> in settings.json
         (when 'feature' is provided).
      3. Global active_provider in settings.json.
      4. Local Ollama (hardcoded fallback).

    Provider credentials are read from the providers block in settings.json with
    environment variable fallbacks (GITHUB_TOKEN, OPENAI_API_KEY, ANTHROPIC_API_KEY).

    Returns (content, engine_label, model_name, base_url_or_empty).
    """
    # If the caller already resolved an Ollama endpoint, use it directly
    if ollama_base_url:
        log.info(
            "_route_vlm_chat: using caller-supplied Ollama endpoint — model=%s base_url=%s",
            ollama_model or _get_configured_model(), ollama_base_url,
        )
        return _ollama_vlm_chat(
            system_prompt, user_prompt, image_bytes_list,
            base_url=ollama_base_url,
            model=ollama_model or _get_configured_model(),
            timeout_sec=timeout_sec,
        )

    # Per-feature routing: consult get_provider_config_for_feature when a feature
    # name is given.  This honours the providers.feature_models.<feature> block in
    # settings.json and is the correct path for pan_extraction / aadhaar_extraction.
    if feature:
        provider_cfg = get_provider_config_for_feature(feature)
        provider_name = provider_cfg.get("provider", "ollama").lower().strip()
        resolved_model = str(provider_cfg.get("model", "")).strip()
        resolved_base_url = str(provider_cfg.get("base_url", "")).strip()
        log.info(
            "_route_vlm_chat: feature=%s → provider=%s model=%s base_url=%s",
            feature, provider_name, resolved_model, resolved_base_url or "(ollama auto-detect)",
        )
        if provider_name in ("github_models", "openai", "openai_compatible"):
            content, engine, model = _openai_compatible_vlm_chat(
                system_prompt, user_prompt, image_bytes_list,
                provider_cfg,
                timeout_sec=timeout_sec,
            )
            return content, engine, model, resolved_base_url
        if provider_name == "anthropic":
            content, engine, model = _anthropic_vlm_chat(
                system_prompt, user_prompt, image_bytes_list,
                provider_cfg,
                timeout_sec=timeout_sec,
            )
            return content, engine, model, "https://api.anthropic.com/v1"
        if provider_name == "google":
            # Google AI Studio: instantiate VLMClient with the resolved credentials
            api_key = str(provider_cfg.get("api_key", "")).strip()
            client = VLMClient(
                "google", resolved_model or provider_cfg.get("model", "gemini-2.0-flash"),
                base_url=resolved_base_url,
                api_key=api_key,
            )
            return client.chat_vision(
                system_prompt, user_prompt, image_bytes_list, timeout_sec=timeout_sec
            )
        # Ollama: auto-detect endpoint and use the feature-configured model
        base_url_o, models, _ = probe_ollama()
        if not base_url_o:
            return "", "gemma4_ollama", resolved_model or _get_configured_model(), ""
        selected_model = resolved_model or select_ollama_model(models)
        log.info(
            "_route_vlm_chat: Ollama endpoint resolved — base_url=%s model=%s",
            base_url_o, selected_model,
        )
        return _ollama_vlm_chat(
            system_prompt, user_prompt, image_bytes_list,
            base_url=base_url_o,
            model=selected_model,
            timeout_sec=timeout_sec,
        )

    # Global routing (no per-feature override): use active_provider from settings.json.
    cfg = _load_llm_config()
    provider = str(cfg.get("active_provider", "ollama") or "ollama").strip().lower()

    # "feature_models" is a routing sentinel, not a real provider name.
    # When it appears as the global active_provider and no feature was specified,
    # fall back to Ollama so callers that do not pass a feature still work.
    if provider == "feature_models":
        log.debug(
            "_route_vlm_chat: active_provider=feature_models but no feature specified — "
            "falling back to Ollama. Pass feature= to use per-feature routing.",
        )
        provider = "ollama"

    provider_cfg = cfg.get("providers", {}).get(provider, {})
    log.info(
        "_route_vlm_chat: global routing — provider=%s model=%s",
        provider, provider_cfg.get("model", _get_configured_model()),
    )

    if provider in ("github_models", "openai", "openai_compatible"):
        content, engine, model = _openai_compatible_vlm_chat(
            system_prompt, user_prompt, image_bytes_list,
            provider_cfg,
            timeout_sec=timeout_sec,
        )
        return content, engine, model, str(provider_cfg.get("base_url", ""))

    if provider == "anthropic":
        content, engine, model = _anthropic_vlm_chat(
            system_prompt, user_prompt, image_bytes_list,
            provider_cfg,
            timeout_sec=timeout_sec,
        )
        return content, engine, model, "https://api.anthropic.com/v1"

    if provider == "google":
        # Google AI Studio: instantiate VLMClient with connection details from providers block
        resolved_google_base = str(provider_cfg.get("base_url", "")).rstrip("/")
        resolved_google_model = str(provider_cfg.get("model", "gemini-2.0-flash"))
        resolved_google_key = str(provider_cfg.get("api_key", "")).strip()
        client = VLMClient(
            "google", resolved_google_model,
            base_url=resolved_google_base,
            api_key=resolved_google_key,
        )
        return client.chat_vision(
            system_prompt, user_prompt, image_bytes_list, timeout_sec=timeout_sec
        )

    # Default path: local Ollama
    base_url, models, _ = probe_ollama()
    if not base_url:
        return "", "gemma4_ollama", _get_configured_model(), ""
    selected_model = select_ollama_model(models)
    log.info(
        "_route_vlm_chat: Ollama endpoint resolved — base_url=%s model=%s",
        base_url, selected_model,
    )
    return _ollama_vlm_chat(
        system_prompt, user_prompt, image_bytes_list,
        base_url=base_url,
        model=selected_model,
        timeout_sec=timeout_sec,
    )


def _extract_json_object(text: str) -> str:
    """Pull the first complete JSON object from a model response string.

    Uses two strategies in order:
    1. If a markdown code fence (```json or ```) appears *anywhere* in the
       text, extract the content inside it — Google/cloud models often add
       preamble text before the fence so a plain startswith("```") check fails.
    2. Fall back to scanning the raw text for the first '{' and walking
       brace-depth to find its matching '}', which is immune to postamble
       text that contains unrelated '}' characters (the old rfind approach
       breaks in that case and returns garbage to json.loads).
    """
    if not text:
        return ""

    # Strategy 1: find a ```json ... ``` or ``` ... ``` fence anywhere.
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if fence_match:
        candidate = fence_match.group(1).strip()
        brace_start = candidate.find("{")
        if brace_start != -1:
            depth = 0
            for i, ch in enumerate(candidate[brace_start:], brace_start):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return candidate[brace_start : i + 1]

    # Strategy 2: brace-depth scan on the raw text.
    brace_start = text.find("{")
    if brace_start == -1:
        return ""
    depth = 0
    for i, ch in enumerate(text[brace_start:], brace_start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[brace_start : i + 1]
    return ""


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
    """Extract structured Aadhaar fields from an image using the configured VLM provider.

    Routes via the 'aadhaar_extraction' feature key in settings.json so the
    operator can point this feature at any supported provider (Ollama, GitHub
    Models, OpenAI, Anthropic) without touching the code.
    """
    if not image_bytes:
        return {}

    log.info(
        "extract_aadhaar_details: starting VLM extraction — feature=aadhaar_extraction",
    )
    _log_vlm_request(
        "extract_aadhaar_details",
        feature="aadhaar_extraction",
        model=model,
        base_url=base_url,
        system_prompt=AADHAAR_EXTRACTION_SYSTEM_PROMPT,
        user_prompt=AADHAAR_EXTRACTION_PROMPT,
        image_bytes_list=[image_bytes],
    )
    content, engine, resolved_model, resolved_base = _route_vlm_chat(
        AADHAAR_EXTRACTION_SYSTEM_PROMPT,
        AADHAAR_EXTRACTION_PROMPT,
        [image_bytes],
        # Honour explicit caller overrides (e.g. from the identity page probe path);
        # otherwise _route_vlm_chat resolves the provider from settings.json.
        ollama_base_url=base_url,
        ollama_model=model,
        feature="aadhaar_extraction",
    )
    if not content:
        log.warning(
            "extract_aadhaar_details: VLM returned empty response — "
            "engine=%s model=%s base_url=%s",
            engine, resolved_model, resolved_base,
        )
        return {}
    _log_vlm_response(
        "extract_aadhaar_details",
        engine=engine,
        model=resolved_model,
        base_url=resolved_base,
        content=content,
    )
    parsed = parse_aadhaar_response_content(content)
    if not parsed:
        log.warning(
            "extract_aadhaar_details: could not parse VLM response — "
            "engine=%s model=%s raw_len=%d",
            engine, resolved_model, len(content),
        )
        return {}
    # Log what the model extracted so operators can verify accuracy at a glance
    log.info(
        "extract_aadhaar_details: extraction complete — engine=%s model=%s "
        "uid=%s name=%s dob=%s gender=%s",
        engine, resolved_model,
        parsed.get("uid", ""),
        parsed.get("name", ""),
        parsed.get("dob", ""),
        parsed.get("gender", ""),
    )
    parsed["engine"] = engine
    parsed["model"] = resolved_model
    parsed["base_url"] = resolved_base
    parsed["raw_response"] = content
    parsed["qr_type"] = "gemma4"
    return parsed


def extract_pan_details_with_ollama(
    image_bytes: bytes,
    *,
    model: str | None = None,
    base_url: str | None = None,
) -> Dict[str, Any]:
    """Extract structured PAN fields from an image using the configured VLM provider.

    Routes via the 'pan_extraction' feature key in settings.json.
    """
    if not image_bytes:
        return {}

    log.info("extract_pan_details: starting VLM extraction — feature=pan_extraction")
    _log_vlm_request(
        "extract_pan_details",
        feature="pan_extraction",
        model=model,
        base_url=base_url,
        system_prompt=PAN_EXTRACTION_SYSTEM_PROMPT,
        user_prompt=PAN_EXTRACTION_PROMPT,
        image_bytes_list=[image_bytes],
    )
    content, engine, resolved_model, resolved_base = _route_vlm_chat(
        PAN_EXTRACTION_SYSTEM_PROMPT,
        PAN_EXTRACTION_PROMPT,
        [image_bytes],
        ollama_base_url=base_url,
        ollama_model=model,
        feature="pan_extraction",
    )
    if not content:
        log.warning(
            "extract_pan_details: VLM returned empty response — engine=%s model=%s",
            engine, resolved_model,
        )
        return {}
    _log_vlm_response(
        "extract_pan_details",
        engine=engine,
        model=resolved_model,
        base_url=resolved_base,
        content=content,
    )
    parsed = parse_pan_response_content(content)
    if not parsed:
        log.warning(
            "extract_pan_details: could not parse VLM response — engine=%s model=%s",
            engine, resolved_model,
        )
        return {}
    log.info(
        "extract_pan_details: extraction complete — engine=%s model=%s pan=%s name=%s",
        engine, resolved_model,
        parsed.get("pan_number", ""),
        parsed.get("full_name", ""),
    )
    parsed["engine"] = engine
    parsed["model"] = resolved_model
    parsed["base_url"] = resolved_base
    parsed["raw_response"] = content
    return parsed


# ---------------------------------------------------------------------------
# Combined PAN extraction + signature bounding-box in one Gemma4 call
# ---------------------------------------------------------------------------

# Combined prompt asks Gemma4 to return both the PAN card fields AND the
# normalised signature bounding box in a single JSON object.  Merging the two
# previously separate Gemma4 calls (one for field extraction, one for the
# signature crop region) into one reduces latency by ~50 % on PAN card upload.

PAN_COMBINED_EXTRACTION_SYSTEM_PROMPT = (
    "You extract structured fields and document layout information from Indian PAN card "
    "images. Return strict JSON only. Do not add commentary or markdown."
)

PAN_COMBINED_EXTRACTION_PROMPT = """
Read this Indian PAN card image and return a JSON object with exactly these keys:
{
  "pan_number": "",
  "full_name": "",
  "father_name": "",
  "date_of_birth": "",
  "signature_box": {
    "x1": 0.0,
    "y1": 0.0,
    "x2": 0.0,
    "y2": 0.0,
    "confidence": 0.0
  }
}

Rules for card text fields:
- Preserve card text exactly as printed — letter by letter, no substitution.
- CRITICAL: Copy full_name and father_name character-by-character.
  Never replace visually similar characters ('Hr' → 'Har', 'H' → 'N', etc.).
  Indian names may have unusual sequences — do not correct them.
- PAN number: 5 letters, 4 digits, 1 letter format if visible.
- Do not guess missing values; return empty string when a field is not visible.

Rules for signature_box (CRITICAL — read carefully):
- Locate ONLY the handwritten applicant signature — the cursive or handwritten ink
  strokes made by a person using a pen. These appear in the lower-left area of the card.
- The signature region contains ONLY handwritten pen strokes. It does NOT contain:
    * The PAN number (printed text like "ABCDE1234F") — this is ABOVE the signature.
    * The printed label "Signature" or "सत्यापन" below the signature area.
    * The photo/hologram on the right side of the card.
    * Any printed text, logos, or government seals.
- IMPORTANT y1 rule: The PAN number is printed text that always appears ABOVE the
  signature strokes. Your y1 value MUST be strictly below the bottom edge of the PAN
  number. Never set y1 high enough to include the PAN number line.
- Return the normalised bounding box as fractions of the FULL image dimensions (0.0-1.0):
  x1=left edge, y1=top edge, x2=right edge, y2=bottom edge.
- x1 < x2 and y1 < y2. All values must be between 0.0 and 1.0.
- IMPORTANT: Signatures often start very close to the left margin.
  Make sure x1 captures the very beginning of the LEFTMOST ink stroke.
  When in doubt, bias x1 slightly to the left — it is better to include
  a few background pixels than to clip the start of the signature.
- Use at most 3% padding around the strokes (tight fit, ink strokes only).
- If the signature is not visible set all four coordinates and confidence to 0.0.

Output JSON only — no extra text.
""".strip()


def extract_pan_details_and_signature_with_ollama(
    image_bytes: bytes,
    *,
    model: str | None = None,
    base_url: str | None = None,
) -> Dict[str, Any]:
    """Extract PAN card text fields AND the signature bounding-box in one VLM call.

    This replaces two separate VLM calls (one for field extraction via
    ``extract_pan_details_with_ollama`` and one for signature detection inside
    ``_crop_pan_signature``) with a single vision call that returns both pieces
    of information in one JSON response.

    Routes via the 'pan_extraction' feature key in settings.json so the operator
    can switch providers without code changes.

    Returns a dict with the same PAN field keys as ``parse_pan_response_content``
    PLUS a ``sig_box`` key containing ``{x1, y1, x2, y2}`` normalised fractions
    when the signature region was confidently located.  Returns ``{}`` when
    the VLM provider is unreachable or the response is unparseable.
    """
    if not image_bytes:
        return {}

    log.info(
        "extract_pan_details_and_signature: starting combined VLM call — "
        "feature=pan_extraction",
    )
    _log_vlm_request(
        "extract_pan_details_and_signature",
        feature="pan_extraction",
        model=model,
        base_url=base_url,
        system_prompt=PAN_COMBINED_EXTRACTION_SYSTEM_PROMPT,
        user_prompt=PAN_COMBINED_EXTRACTION_PROMPT,
        image_bytes_list=[image_bytes],
    )
    content, engine, resolved_model, resolved_base = _route_vlm_chat(
        PAN_COMBINED_EXTRACTION_SYSTEM_PROMPT,
        PAN_COMBINED_EXTRACTION_PROMPT,
        [image_bytes],
        ollama_base_url=base_url,
        ollama_model=model,
        feature="pan_extraction",
    )
    if not content:
        log.warning(
            "extract_pan_details_and_signature: VLM returned empty response — "
            "engine=%s model=%s base_url=%s",
            engine, resolved_model, resolved_base,
        )
        return {}
    _log_vlm_response(
        "extract_pan_details_and_signature",
        engine=engine,
        model=resolved_model,
        base_url=resolved_base,
        content=content,
    )

    # Parse PAN card text fields using the shared parser
    parsed = parse_pan_response_content(content)

    # Also extract the signature_box from the raw response JSON
    json_text = _extract_json_object(content)
    if json_text:
        try:
            raw_data = json.loads(json_text)
            sig_box_raw = raw_data.get("signature_box", {})
            if isinstance(sig_box_raw, dict):
                # Parse and validate bounding-box coordinates
                x1 = float(sig_box_raw.get("x1", 0))
                y1 = float(sig_box_raw.get("y1", 0))
                x2 = float(sig_box_raw.get("x2", 0))
                y2 = float(sig_box_raw.get("y2", 0))
                width_f = x2 - x1
                height_f = y2 - y1
                # Accept the box only when it is in-range, non-degenerate, and
                # not suspiciously large (full-card-spanning bbox is noise)
                if (
                    0 <= x1 < x2 <= 1
                    and 0 <= y1 < y2 <= 1
                    and width_f >= 0.03
                    and height_f >= 0.02
                    and width_f <= 0.7
                    and height_f <= 0.5
                ):
                    parsed["sig_box"] = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
        except (TypeError, ValueError, json.JSONDecodeError):
            pass  # sig_box is optional — fall through to Gemma4-free crop path

    if not parsed:
        log.warning(
            "extract_pan_details_and_signature: could not parse VLM response — "
            "engine=%s model=%s raw_len=%d",
            engine, resolved_model, len(content),
        )
        return {}

    # Log every extracted field so operators can verify accuracy and debug issues
    log.info(
        "extract_pan_details_and_signature: extraction complete — engine=%s model=%s "
        "pan=%s name=%s father=%s dob=%s sig_box=%s",
        engine, resolved_model,
        parsed.get("pan_number", ""),
        parsed.get("full_name", ""),
        parsed.get("father_name", ""),
        parsed.get("date_of_birth", ""),
        "yes" if parsed.get("sig_box") else "not found",
    )
    parsed["engine"] = engine
    parsed["model"] = resolved_model
    parsed["base_url"] = resolved_base
    parsed["raw_response"] = content
    return parsed


# ---------------------------------------------------------------------------
# Fast document-type classifier — used as a pre-flight check before extraction
# ---------------------------------------------------------------------------

# This short, focused prompt only asks "what is this document?" so Gemma4 can
# answer quickly.  It is intentionally much smaller than the full extraction
# prompts so the round-trip is fast (seconds, not tens of seconds).
DOC_CLASSIFY_SYSTEM_PROMPT = (
    "You are a document type classifier for Indian government and financial documents. "
    "Return strict JSON only. Do not add commentary or markdown."
)

DOC_CLASSIFY_PROMPT = """\
Look at this image and identify what type of document it is.

Return ONLY a JSON object with these three keys:
{
  "document_type": "<see allowed values below>",
  "confidence": <0.0–1.0>,
  "reason": "<one short sentence explaining your answer>"
}

Allowed values for document_type:
- "aadhaar_card"      → Indian Aadhaar identity card (has a printed card layout with
                         "AADHAAR" or UIDAI branding, 12-digit UID number, QR code.)
- "pan_card"          → Indian PAN card (has a printed card layout with
                         "INCOME TAX DEPARTMENT" text, a 10-character PAN like ABCDE1234F.)
- "photograph_selfie" → A PLAIN portrait or selfie photo of a person's face with NO card
                         layout, NO printed government text, NO logos, NO card border.
                         IMPORTANT: An Aadhaar or PAN card that happens to have a
                         person's photo printed on it is still "aadhaar_card" / "pan_card",
                         NOT "photograph_selfie".
- "other"             → Anything else (payslip, bank statement, utility bill, etc.)

Key rule: if the image looks like a government-issued identity card (even if a face
photo is visible on it), always use "aadhaar_card" or "pan_card". Only use
"photograph_selfie" for a standalone photo that is NOT an ID card.
Use "other" when the image does not clearly match one of the three specific types above.
Output JSON only — no extra text.\
"""


def classify_document_type_with_ollama(
    image_bytes: bytes,
    *,
    model: str | None = None,
    base_url: str | None = None,
    feature: str | None = None,
) -> Dict[str, Any]:
    """Ask the configured VLM what type of document an image is.

    This is a lightweight, fast call used as a pre-flight check before running
    the full QR decode or OCR extraction.  It catches the most common user
    mistake: uploading a document into the wrong slot (e.g. a PAN card in the
    Aadhaar section).

    Uses a shorter read timeout (_DOC_CLASSIFY_READ_TIMEOUT_SEC = 45 s) than
    full extraction calls (1200 s) because the response is tiny.

    Parameters
    ----------
    image_bytes : raw bytes of the image to classify.
    model / base_url : override the auto-detected Ollama endpoint when needed.
    feature : optional feature key (e.g. "aadhaar_extraction", "pan_extraction")
              used to route this call through the per-feature provider config in
              settings.json instead of the global active_provider.  When None,
              the global provider is used.  Pass the feature key that matches the
              slot being checked so the same provider is used for both the
              classification pre-flight and the full extraction that follows.

    Returns
    -------
    Dict with keys:
        ``available``     — True when the VLM provider was reachable; False otherwise.
        ``document_type`` — one of "aadhaar_card", "pan_card",
                            "photograph_selfie", or "other".
        ``confidence``    — float 0.0–1.0 (how certain the model is).
        ``reason``        — one-sentence plain-English explanation.

    Returns ``{"available": False}`` when the VLM provider is completely
    unreachable so callers can gracefully skip the check rather than blocking
    the upload.
    """
    if not image_bytes:
        return {}

    _log_vlm_request(
        "classify_document_type",
        feature=feature,
        model=model,
        base_url=base_url,
        system_prompt=DOC_CLASSIFY_SYSTEM_PROMPT,
        user_prompt=DOC_CLASSIFY_PROMPT,
        image_bytes_list=[image_bytes],
    )

    content, _engine, _model, _base = _route_vlm_chat(
        DOC_CLASSIFY_SYSTEM_PROMPT,
        DOC_CLASSIFY_PROMPT,
        [image_bytes],
        timeout_sec=_DOC_CLASSIFY_READ_TIMEOUT_SEC,
        ollama_base_url=base_url,
        ollama_model=model,
        # Pass through the caller-supplied feature key so this pre-flight check
        # uses the same provider as the full extraction that follows
        feature=feature,
    )
    if not content:
        # VLM provider is unreachable — signal to caller that the check was skipped
        return {"available": False}
    _log_vlm_response(
        "classify_document_type",
        engine=_engine,
        model=_model,
        base_url=_base,
        content=content,
    )

    json_text = _extract_json_object(content)
    if not json_text:
        # Could not parse the response — treat as inconclusive
        return {
            "available": True,
            "document_type": "other",
            "confidence": 0.0,
            "reason": "Could not parse classification response.",
        }

    try:
        result = json.loads(json_text)
    except json.JSONDecodeError:
        return {
            "available": True,
            "document_type": "other",
            "confidence": 0.0,
            "reason": "Could not parse classification response.",
        }

    if not isinstance(result, dict):
        return {
            "available": True,
            "document_type": "other",
            "confidence": 0.0,
            "reason": "Unexpected response format.",
        }

    # Pull out the three fields; clamp confidence to the valid 0–1 range
    doc_type   = str(result.get("document_type", "other")).lower().strip()
    confidence = max(0.0, min(1.0, float(result.get("confidence", 0.0))))
    reason     = str(result.get("reason", "")).strip()

    return {
        "available":     True,
        "document_type": doc_type,
        "confidence":    confidence,
        "reason":        reason,
    }


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

    # Prepend the optional document type hint to the prompt for better accuracy
    prompt = UNIVERSAL_DOCUMENT_ANALYSIS_PROMPT
    if doc_hint:
        prompt = f"Document hint: {doc_hint}\n\n{prompt}"

    content, engine, resolved_model, _base = _route_vlm_chat(
        UNIVERSAL_DOCUMENT_ANALYSIS_SYSTEM_PROMPT,
        prompt,
        [image_bytes],
        ollama_base_url=base_url,
        ollama_model=model,
    )
    if not content:
        return {}

    json_text = _extract_json_object(content)
    if not json_text:
        return {}

    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError:
        return {}

    if not isinstance(parsed, dict):
        return {}

    parsed["engine"] = engine
    parsed["model"] = resolved_model
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

    # List file names in the prompt so the model knows which document is which
    names_block = "\n".join(
        f"{i}. {filenames[i] if i < len(filenames) else f'document_{i}'}"
        for i, _ in valid_pairs
    )
    prompt = (
        f"Documents to classify (in order):\n{names_block}\n\n"
        f"{BATCH_CLASSIFICATION_PROMPT}"
    )

    # Extract just the valid (non-empty) images to send to the VLM
    valid_images = [img for _, img in valid_pairs]

    # _route_vlm_chat with empty system_prompt sends a user-only message,
    # matching the original Ollama payload structure for batch classification
    content, _engine, _model, _base = _route_vlm_chat(
        "",  # no system prompt for batch — the user message contains all instructions
        prompt,
        valid_images,
        ollama_base_url=base_url,
        ollama_model=model,
    )
    if not content:
        return fallback

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