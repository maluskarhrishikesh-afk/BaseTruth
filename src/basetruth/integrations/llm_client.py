"""Unified VLM provider client for BaseTruth.

VLMClient is the single class that handles all HTTP calls to supported LLM
providers.  Provider selection and feature routing live in ollama.py; this
module is only responsible for building requests and parsing responses.

Supported providers
-------------------
- ollama           : Local Ollama server (Gemma4, Llama, etc.)
- openai_compatible: Any OpenAI-format endpoint (OpenAI, GitHub Models, Azure)
- anthropic        : Anthropic Messages API (Claude models)
- google           : Google AI Studio REST API (Gemini models)

Usage
-----
Instantiate VLMClient with a fully-resolved provider name and connection
details, then call chat_vision() for a single-turn vision request:

    client = VLMClient("google", "gemini-2.0-flash", api_key="AI...")
    content, engine, model, base_url = client.chat_vision(system, user, [img])

Routing (which provider to use for which feature) is handled in ollama.py's
get_provider_config_for_feature() and _route_vlm_chat().  This class never
reads settings.json — it only sends HTTP requests.
"""
from __future__ import annotations

import base64
import os
from typing import Any, Dict, List

import requests

from basetruth.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Timeout constants
# ---------------------------------------------------------------------------
# These are re-exported so that ollama.py and document_extract.py can import
# them from here when they want to share the same defaults.
OLLAMA_CONNECT_TIMEOUT_SEC: int = 5     # seconds to wait for initial TCP connection
OLLAMA_READ_TIMEOUT_SEC: int = 1200     # seconds to wait for the full response body

# Google AI Studio REST API base URL.
# The generateContent endpoint is: {base_url}/models/{model}:generateContent
_GOOGLE_DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


class VLMClient:
    """Unified vision-language model client for BaseTruth.

    Instantiate with a fully-resolved provider name and connection details.
    No settings.json reading happens here — that belongs to the routing layer
    in ollama.py.  This class only sends HTTP requests and parses responses.

    All chat_vision calls return a 4-tuple:
        (content: str, engine_label: str, model_name: str, base_url: str)

    On error, content is "" and the other fields reflect what was attempted, so
    callers can check for empty content without catching exceptions.

    Example
    -------
        client = VLMClient("google", "gemini-2.0-flash", api_key="AI...")
        content, engine, model, base = client.chat_vision(
            "You are a document analyst. Return JSON only.",
            "Extract all fields from this PAN card.",
            [pan_card_image_bytes],
        )
    """

    def __init__(
        self,
        provider: str,
        model: str,
        *,
        base_url: str = "",
        api_key: str = "",
    ) -> None:
        # Normalize provider name to lowercase so routing checks are case-insensitive
        self.provider = provider.lower().strip()
        self.model = model.strip()
        # Strip trailing slash from base_url so we can always safely append paths
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    # ---------------------------------------------------------------------- #
    # Public interface                                                         #
    # ---------------------------------------------------------------------- #

    def chat_vision(
        self,
        system_prompt: str,
        user_prompt: str,
        image_bytes_list: List[bytes],
        *,
        pdf_bytes: bytes | None = None,
        timeout_sec: int = OLLAMA_READ_TIMEOUT_SEC,
    ) -> tuple[str, str, str, str]:
        """Route a single-turn vision request to the configured provider.

        Parameters
        ----------
        system_prompt     : System / role instruction for the model.  May be
                            empty — the call still works without it.
        user_prompt       : The main user message (question / extraction prompt).
        image_bytes_list  : Raw image bytes (JPEG / PNG) for each image to include.
        pdf_bytes         : Optional raw PDF file bytes. When provided and the
                            provider is Google (Gemini), the PDF is sent directly
                            as an inline_data part with mime_type application/pdf,
                            which is better than a JPEG render because Gemini reads
                            all pages with full vector text fidelity and no JPEG
                            compression artifacts. image_bytes_list is ignored when
                            pdf_bytes is given to the Google provider. For all other
                            providers (Ollama, OpenAI-compatible, Anthropic), this
                            parameter is ignored and image_bytes_list is used.
        timeout_sec       : Maximum seconds to wait for the full response.

        Returns (content, engine_label, model_name, base_url_or_empty).
        Returns ("", engine, model, base_url) on any error so callers can check
        for empty content and handle the failure gracefully.
        """
        if self.provider in ("openai_compatible", "openai", "github_models"):
            return self._call_openai_compatible(
                system_prompt, user_prompt, image_bytes_list, timeout_sec=timeout_sec
            )
        if self.provider == "anthropic":
            return self._call_anthropic(
                system_prompt, user_prompt, image_bytes_list, timeout_sec=timeout_sec
            )
        if self.provider == "google":
            return self._call_google(
                system_prompt, user_prompt, image_bytes_list,
                pdf_bytes=pdf_bytes,
                timeout_sec=timeout_sec,
            )
        # Default: Ollama (covers "ollama" and any unrecognised provider name).
        # Ollama does not support native PDF input, so pdf_bytes is ignored here.
        return self._call_ollama(
            system_prompt, user_prompt, image_bytes_list, timeout_sec=timeout_sec
        )

    # ---------------------------------------------------------------------- #
    # Private provider implementations                                        #
    # ---------------------------------------------------------------------- #

    def _call_ollama(
        self,
        system_prompt: str,
        user_prompt: str,
        image_bytes_list: List[bytes],
        *,
        timeout_sec: int = OLLAMA_READ_TIMEOUT_SEC,
    ) -> tuple[str, str, str, str]:
        """POST a vision chat request to the local Ollama API.

        Ollama embeds images as base64 strings directly inside the message object.
        This differs from the OpenAI format that uses 'image_url' content blocks.
        A system message is only added when system_prompt is non-empty so that
        batch classification calls (which have no system prompt) work correctly.

        Returns (content, 'gemma4_ollama', model_name, base_url).
        """
        messages: List[Dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({
            "role": "user",
            "content": user_prompt,
            # Each image is base64-encoded as an ASCII string in the 'images' list
            "images": [base64.b64encode(img).decode("ascii") for img in image_bytes_list],
        })
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0},  # zero temperature = most deterministic output
        }
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=(OLLAMA_CONNECT_TIMEOUT_SEC, timeout_sec),
            )  # nosemgrep: basetruth-ssrf
            response.raise_for_status()
            content = str(
                response.json().get("message", {}).get("content", "")
            ).strip()
            log.info(
                "VLMClient._call_ollama: response received — model=%s base_url=%s "
                "response_chars=%d",
                self.model, self.base_url, len(content),
            )
            return content, "gemma4_ollama", self.model, self.base_url
        except requests.Timeout as exc:
            # Timeout is the most common failure — log it separately so it is not
            # confused with a model error or an empty response
            log.warning(
                "VLMClient._call_ollama: timed out after %ds — model=%s base_url=%s error=%s",
                timeout_sec, self.model, self.base_url, exc,
            )
            return "", "gemma4_ollama", self.model, self.base_url
        except requests.RequestException as exc:
            log.warning(
                "VLMClient._call_ollama: request failed — model=%s base_url=%s error=%s",
                self.model, self.base_url, exc,
            )
            return "", "gemma4_ollama", self.model, self.base_url

    def _call_openai_compatible(
        self,
        system_prompt: str,
        user_prompt: str,
        image_bytes_list: List[bytes],
        *,
        timeout_sec: int = OLLAMA_READ_TIMEOUT_SEC,
    ) -> tuple[str, str, str, str]:
        """POST a vision chat request to an OpenAI-compatible endpoint.

        Works with GitHub Models (Azure inference endpoint), standard OpenAI, and
        any other provider following the OpenAI chat/completions API.  Images are
        sent as base64 data URIs inside 'image_url' content blocks — the standard
        OpenAI multimodal format, different from Ollama's format.

        API key resolution (in priority order):
          1. api_key stored on this instance (from settings.json)
          2. GITHUB_TOKEN env var (for GitHub Models / Azure endpoints)
          3. OPENAI_API_KEY env var (for standard OpenAI endpoints)

        Returns (content, 'openai_compatible', model_name, base_url).
        """
        # Use configured base_url or fall back to the standard OpenAI endpoint
        base_url = self.base_url or "https://api.openai.com/v1"
        api_key = self.api_key
        if not api_key:
            # Determine which env var to try based on which service we're calling
            if "azure.com" in base_url or self.provider == "github_models":
                api_key = os.getenv("GITHUB_TOKEN", "")
            else:
                api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            log.warning(
                "VLMClient._call_openai_compatible: no API key — model=%s base_url=%s "
                "(set api_key in settings.json or GITHUB_TOKEN / OPENAI_API_KEY env var)",
                self.model, base_url,
            )
            return "", "openai_compatible", self.model, base_url

        log.info(
            "VLMClient._call_openai_compatible: sending request — model=%s base_url=%s "
            "images=%d",
            self.model, base_url, len(image_bytes_list),
        )

        # Build the content array: text prompt first, then one image_url block per image.
        # OpenAI requires images to come after the text prompt (unlike Anthropic).
        content_parts: List[Dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        for img_bytes in image_bytes_list:
            b64 = base64.b64encode(img_bytes).decode("ascii")
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })

        messages: List[Dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content_parts})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 4096,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        try:
            response = requests.post(
                f"{base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=(OLLAMA_CONNECT_TIMEOUT_SEC, timeout_sec),
            )  # nosemgrep: basetruth-ssrf
            response.raise_for_status()
            text = str(
                response.json().get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            ).strip()
            log.info(
                "VLMClient._call_openai_compatible: response received — model=%s "
                "base_url=%s response_chars=%d",
                self.model, base_url, len(text),
            )
            return text, "openai_compatible", self.model, base_url
        except (requests.RequestException, KeyError, IndexError) as exc:
            log.warning(
                "VLMClient._call_openai_compatible: request failed — model=%s "
                "base_url=%s error=%s",
                self.model, base_url, exc,
            )
            return "", "openai_compatible", self.model, base_url

    def _call_anthropic(
        self,
        system_prompt: str,
        user_prompt: str,
        image_bytes_list: List[bytes],
        *,
        timeout_sec: int = OLLAMA_READ_TIMEOUT_SEC,
    ) -> tuple[str, str, str, str]:
        """POST a vision chat request to the Anthropic Messages API.

        Anthropic uses a different auth header ('x-api-key') and a different image
        format where images appear as 'source' blocks inside the content array.
        The system prompt is a top-level field, not a messages entry.
        Images must appear before the text prompt in the content array — Anthropic
        requires text to reference images that were already provided.

        Returns (content, 'anthropic', model_name, 'https://api.anthropic.com/v1').
        """
        api_key = self.api_key or os.getenv("ANTHROPIC_API_KEY", "")
        anthropic_url = "https://api.anthropic.com/v1"
        if not api_key:
            log.warning(
                "VLMClient._call_anthropic: no API key — model=%s "
                "(set api_key in settings.json or ANTHROPIC_API_KEY env var)",
                self.model,
            )
            return "", "anthropic", self.model, anthropic_url

        # Anthropic content array: images first, then the text prompt
        content_parts: List[Dict[str, Any]] = []
        for img_bytes in image_bytes_list:
            b64 = base64.b64encode(img_bytes).decode("ascii")
            content_parts.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
            })
        content_parts.append({"type": "text", "text": user_prompt})

        payload: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": content_parts}],
        }
        # Anthropic only accepts a non-empty system string as a top-level field
        if system_prompt:
            payload["system"] = system_prompt

        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        try:
            response = requests.post(
                f"{anthropic_url}/messages",
                json=payload,
                headers=headers,
                timeout=(OLLAMA_CONNECT_TIMEOUT_SEC, timeout_sec),
            )  # nosemgrep: basetruth-ssrf
            response.raise_for_status()
            blocks = response.json().get("content", [])
            # Pull the first text block out of Anthropic's content array
            text = str(
                next((b.get("text", "") for b in blocks if b.get("type") == "text"), "")
            ).strip()
            return text, "anthropic", self.model, anthropic_url
        except (requests.RequestException, KeyError, StopIteration) as exc:
            log.warning(
                "VLMClient._call_anthropic: request failed — model=%s error=%s",
                self.model, exc,
            )
            return "", "anthropic", self.model, anthropic_url

    def _call_google(
        self,
        system_prompt: str,
        user_prompt: str,
        image_bytes_list: List[bytes],
        *,
        pdf_bytes: bytes | None = None,
        timeout_sec: int = OLLAMA_READ_TIMEOUT_SEC,
    ) -> tuple[str, str, str, str]:
        """POST a vision chat request to the Google AI Studio REST API.

        Google AI Studio (generativelanguage.googleapis.com) uses a different
        request format from OpenAI:
          - Images are embedded as 'inline_data' parts inside the contents array
          - The system prompt uses a special 'system_instruction' top-level field
          - Authentication is via the 'x-goog-api-key' header (or '?key=' query param)
          - The endpoint is: {base_url}/models/{model}:generateContent

        When pdf_bytes is supplied, the actual PDF file is sent to Gemini as an
        inline_data part with mime_type 'application/pdf'.  Gemini reads the PDF
        natively — it can see all pages and the exact embedded vector text, which
        is far better than a JPEG render of only page 1.  In this case
        image_bytes_list is NOT sent (the PDF already contains the full document).

        When pdf_bytes is None, the call falls back to sending image_bytes_list as
        JPEG inline_data parts — the original behaviour for image inputs.

        API key resolution (in priority order):
          1. api_key stored on this instance (from settings.json)
          2. GOOGLE_AI_API_KEY environment variable
          3. GOOGLE_API_KEY environment variable (older convention)

        Returns (content, 'google', model_name, base_url).
        """
        base_url = self.base_url or _GOOGLE_DEFAULT_BASE_URL
        api_key = self.api_key
        if not api_key:
            api_key = os.getenv("GOOGLE_AI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
        if not api_key:
            log.warning(
                "VLMClient._call_google: no API key — model=%s "
                "(set api_key in settings.json or GOOGLE_AI_API_KEY env var)",
                self.model,
            )
            return "", "google", self.model, base_url

        if pdf_bytes:
            # Sending the actual PDF is better than a JPEG render: Gemini reads all
            # pages, gets the exact vector text, and avoids JPEG compression artifacts.
            log.info(
                "VLMClient._call_google: sending PDF directly — model=%s base_url=%s "
                "pdf_size_bytes=%d (native PDF input; image render skipped)",
                self.model, base_url, len(pdf_bytes),
            )
        else:
            log.info(
                "VLMClient._call_google: sending JPEG image(s) — model=%s base_url=%s "
                "images=%d",
                self.model, base_url, len(image_bytes_list),
            )

        # Build user parts: text prompt first, then the document (PDF preferred over JPEG)
        user_parts: List[Dict[str, Any]] = [{"text": user_prompt}]
        if pdf_bytes:
            # Send the full PDF — Gemini reads all pages natively from this single part
            b64_pdf = base64.b64encode(pdf_bytes).decode("ascii")
            user_parts.append({
                "inline_data": {"mime_type": "application/pdf", "data": b64_pdf},
            })
        else:
            # Fallback: send each image as a separate JPEG inline_data block
            for img_bytes in image_bytes_list:
                b64 = base64.b64encode(img_bytes).decode("ascii")
                user_parts.append({
                    "inline_data": {"mime_type": "image/jpeg", "data": b64},
                })

        payload: Dict[str, Any] = {
            "contents": [
                {"role": "user", "parts": user_parts},
            ],
            "generationConfig": {
                "temperature": 0,
                "maxOutputTokens": 4096,
            },
        }
        # System instruction is a top-level field in Google's format (not a message role)
        if system_prompt:
            payload["system_instruction"] = {"parts": [{"text": system_prompt}]}

        endpoint = f"{base_url}/models/{self.model}:generateContent"
        headers = {
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
        }
        try:
            response = requests.post(
                endpoint,
                json=payload,
                headers=headers,
                timeout=(OLLAMA_CONNECT_TIMEOUT_SEC, timeout_sec),
            )  # nosemgrep: basetruth-ssrf
            response.raise_for_status()
            candidates = response.json().get("candidates", [])
            # Google returns text inside candidates[0].content.parts[*].text.
            # Gemma 4 (and other thinking models) emit an internal reasoning part
            # first, marked with "thought": true — that is the model's scratchpad
            # and must NOT be parsed as the answer.  We skip all thought parts and
            # use only the first non-thought text part as the actual response.
            text = ""
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                # Filter out thought parts — only keep parts where thought is absent or False
                response_parts = [p for p in parts if not p.get("thought", False)]
                text = str(
                    next((p.get("text", "") for p in response_parts if "text" in p), "")
                ).strip()
            log.info(
                "VLMClient._call_google: response received — model=%s base_url=%s "
                "response_chars=%d",
                self.model, base_url, len(text),
            )
            return text, "google", self.model, base_url
        except (requests.RequestException, KeyError, IndexError, StopIteration) as exc:
            log.warning(
                "VLMClient._call_google: request failed — model=%s base_url=%s error=%s",
                self.model, base_url, exc,
            )
            return "", "google", self.model, base_url
