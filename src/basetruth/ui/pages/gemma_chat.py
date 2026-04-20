"""BaseTruth Q&A page — local LLM chat via Ollama."""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List

import requests
import streamlit as st

from basetruth.integrations.ollama import (
    DEFAULT_OLLAMA_MODEL,
    OLLAMA_CONNECT_TIMEOUT_SEC,
    OLLAMA_READ_TIMEOUT_SEC,
    probe_ollama,
    select_ollama_model,
    get_provider_config_for_feature,
)
from basetruth.integrations.db_query import (
    get_schema_summary,
    get_minio_summary,
    get_qna_system_prompt,
    get_qna_minio_instructions,
    execute_safe_query,
    query_minio_objects,
    sync_database_md_to_minio,
)
from basetruth.ui.components import _page_title, _db_available_cached, _minio_available_cached
from basetruth.logger import get_logger

log = get_logger(__name__)

_DEFAULT_MODEL = DEFAULT_OLLAMA_MODEL


def _build_system_prompt() -> str:
    """Build the full system prompt for the Q&A LLM.

    The base identity and behaviour rules come from qna_prompts.md (system_prompt section)
    so operators can tune them without touching Python. DB schema (loaded from MinIO)
    is appended only when the database is reachable. MinIO instructions are
    appended only when MinIO is reachable. This keeps the prompt tight and avoids
    confusing the model with unavailable resources.
    """
    # Base identity + behaviour rules loaded from the markdown asset
    prompt = get_qna_system_prompt()

    # Use the 30-second TTL cached helper (Rule #1) — never call db_available()
    # directly from a Streamlit render path as it makes a live network round-trip
    # on every re-render and freezes the UI.
    if _db_available_cached():
        prompt += f"\n\n{get_schema_summary()}"
        # Remind the model it should use sql blocks — the detailed rules are in the schema
        prompt += (
            "\n\nWhen you need to query data, emit a SQL SELECT inside a triple-backtick sql block. "
            "The system will run it and return results back to you as context."
        )

    # Similarly guard MinIO with the cached helper to avoid a live MinIO ping every render
    if _minio_available_cached():
        minio_summary = get_minio_summary()
        if minio_summary:
            prompt += f"\n\nCurrent storage state: {minio_summary}"
            prompt += f"\n\n{get_qna_minio_instructions()}"

    # Always append formatting instructions so every response is well-structured
    # and easy to read regardless of the question type. This keeps the presentation
    # consistent without needing the operator to tune it in the markdown asset.
    #
    # The key rule here is: section headers must be on their OWN lines (as ## headings),
    # never inline with content. This is what makes the output look like ChatGPT
    # instead of a wall of text where "Key Findings: • ..." appears on one line.
    prompt += (
        "\n\n## Response Formatting Rules (always follow these)\n"
        "- Use **bold** for important terms, field names, and key figures.\n"
        "- Use relevant emojis as section markers: 📊 Executive Summary, 📋 Key Findings, "
        "✅ Positive Outcomes, ❌ Rejections, ⚠️ Warnings, 🔍 Analysis, 💡 Tips/Recommendations.\n"
        "\n"
        "SECTION HEADERS RULE — CRITICAL:\n"
        "- Every section header (Executive Summary, Key Findings, Recommended Actions, etc.) "
        "MUST be on its OWN separate line using a ## Markdown heading.\n"
        "- NEVER put a section title and its content on the same line.\n"
        "- Always leave a blank line between a ## heading and the content that follows it.\n"
        "- Correct format example:\n"
        "  ## 📊 Executive Summary\n\n"
        "  There are 3 applicants in the system.\n\n"
        "  ## 📋 Key Findings\n\n"
        "  • **Active Volume**: 3 entities have submitted files.\n"
        "  • **Pending Review**: 2 identity documents are awaiting approval.\n\n"
        "  ## ✅ Recommended Actions\n\n"
        "  • Review the pending PAN and Aadhaar cards for BT-000001.\n"
        "\n"
        "BULLET POINTS RULE:\n"
        "- Each bullet point (•) must be on its own new line.\n"
        "- Never put multiple bullet points on the same line.\n"
        "- Indent sub-bullets with two extra spaces.\n"
        "\n"
        "STRUCTURE FOR DATA QUERY RESPONSES:\n"
        "Always use this three-section structure when presenting data results:\n"
        "  1. ## 📊 Executive Summary  — one short paragraph (2-3 sentences max)\n"
        "  2. ## 📋 Key Findings       — bullet list of the most important data points\n"
        "  3. ## ✅ Recommended Actions — bullet list of next steps or observations\n"
        "\n"
        "- Keep answers concise but complete. Avoid padding or filler phrases.\n"
        "- Never show raw SQL, JSON, or table dumps to the user — always describe "
        "the data in plain English."
    )

    return prompt


# ---------------------------------------------------------------------------
# Thinking-block stripping — some models wrap reasoning in
# <think>...</think> or <thought>...</thought> tags that should never be
# shown to end users.
# ---------------------------------------------------------------------------

_THINK_BLOCK_RE = re.compile(
    r"<(?:think|thought)>.*?</(?:think|thought)>",
    re.DOTALL | re.IGNORECASE,
)
_THINK_OPEN_RE = re.compile(r"<(?:think|thought)>", re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r"</(?:think|thought)>", re.IGNORECASE)


def _has_open_thinking_block(text: str) -> bool:
    """Return True when the streamed text is inside an unfinished reasoning block.

    Some providers stream tokens inside <think>...</think> while others use
    <thought>...</thought>. We treat both forms the same so the UI can show a
    simple "Thinking..." indicator instead of leaking the model's internal notes.
    """
    return len(_THINK_OPEN_RE.findall(text)) > len(_THINK_CLOSE_RE.findall(text))


def _last_user_message(messages: List[Dict[str, Any]]) -> str:
    """Return the most recent user message content for logging."""
    return next((str(m.get("content", "")) for m in reversed(messages) if m.get("role") == "user"), "")


def _strip_thinking(text: str) -> tuple[str, str]:
    """Remove reasoning blocks like <think>...</think> from model output.

    Some modern models (Qwen3, Deepseek-R1, etc.) wrap their step-by-step
    reasoning in <think> tags before producing the final answer, while some
    hosted providers return the same content inside <thought> tags. This
    internal monologue is useful for the model but distracting and confusing
    for users.

    Returns a tuple of:
      (thinking_content, clean_response)

    - thinking_content : raw text inside the reasoning blocks (may be empty string)
    - clean_response   : the rest of the response with think blocks removed
    """
    thinking_parts = _THINK_BLOCK_RE.findall(text)
    # Remove the tags and collect the inner content
    inner_thoughts = "\n".join(
        re.sub(r"^<(?:think|thought)>|</(?:think|thought)>$", "", part, flags=re.IGNORECASE).strip()
        for part in thinking_parts
    )
    clean = _THINK_BLOCK_RE.sub("", text).strip()
    return inner_thoughts, clean

# ---------------------------------------------------------------------------
# Custom CSS — modern chat UI inspired by Gemini / WhatsApp
# ---------------------------------------------------------------------------

_CHAT_CSS = """\
<style>
/* ── Chat page container ─────────────────────────────────────────────────── */
.bt-chat-wrapper {
    display: flex;
    flex-direction: column;
    height: calc(100vh - 180px);
    min-height: 500px;
    position: relative;
}

/* ── Welcome hero ────────────────────────────────────────────────────────── */
.bt-chat-hero {
    text-align: center;
    padding: 3.5rem 1.5rem 2rem;
    animation: bt-hero-fade 0.6s ease forwards;
}
@keyframes bt-hero-fade {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
.bt-chat-hero-icon {
    width: 72px;
    height: 72px;
    margin: 0 auto 1rem;
    border-radius: 20px;
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #06b6d4 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 36px;
    box-shadow: 0 8px 32px rgba(99, 102, 241, 0.25);
    animation: bt-icon-pulse 3s ease-in-out infinite;
}
@keyframes bt-icon-pulse {
    0%, 100% { box-shadow: 0 8px 32px rgba(99, 102, 241, 0.25); }
    50%      { box-shadow: 0 8px 48px rgba(99, 102, 241, 0.40); }
}
.bt-chat-hero h2 {
    font-size: 1.75rem !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 60%, #06b6d4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem !important;
    letter-spacing: -0.02em;
}
.bt-chat-hero p {
    color: #64748b;
    font-size: 0.95rem;
    max-width: 480px;
    margin: 0 auto;
    line-height: 1.6;
}

/* ── Suggestion chip buttons ─────────────────────────────────────────────── */
/* Streamlit buttons rendered as elegant pill chips */
[data-testid="stHorizontalBlock"]:has(button[kind="secondary"][key^="bt_chip"]),
.bt-chat-hero + [data-testid="stHorizontalBlock"] {
    justify-content: center !important;
    gap: 0.5rem !important;
    padding: 0 1rem 1.5rem !important;
    animation: bt-hero-fade 0.8s ease forwards;
}
button[kind="secondary"] {
    border-radius: 999px !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    border: 1px solid rgba(99, 102, 241, 0.20) !important;
    background: rgba(99, 102, 241, 0.06) !important;
    color: #6366f1 !important;
    transition: all 0.2s ease !important;
    letter-spacing: -0.01em !important;
    padding: 0.5rem 1rem !important;
}
button[kind="secondary"]:hover {
    background: rgba(99, 102, 241, 0.12) !important;
    border-color: rgba(99, 102, 241, 0.35) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.12) !important;
    color: #4f46e5 !important;
}

/* Dark-mode chip overrides */
@media (prefers-color-scheme: dark) {
    button[kind="secondary"] {
        background: rgba(99, 102, 241, 0.10) !important;
        border-color: rgba(99, 102, 241, 0.25) !important;
        color: #818cf8 !important;
    }
    button[kind="secondary"]:hover {
        background: rgba(99, 102, 241, 0.18) !important;
        color: #a5b4fc !important;
    }
}
[data-testid="stApp"][data-theme="dark"] button[kind="secondary"] {
    background: rgba(99, 102, 241, 0.10) !important;
    border-color: rgba(99, 102, 241, 0.25) !important;
    color: #818cf8 !important;
}
[data-testid="stApp"][data-theme="dark"] button[kind="secondary"]:hover {
    background: rgba(99, 102, 241, 0.18) !important;
    color: #a5b4fc !important;
}

/* ── Typing indicator ────────────────────────────────────────────────────── */
.bt-typing {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 0.5rem 0.75rem;
}
.bt-typing-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #6366f1;
    animation: bt-typing-bounce 1.4s ease-in-out infinite;
}
.bt-typing-dot:nth-child(1) { animation-delay: 0s; }
.bt-typing-dot:nth-child(2) { animation-delay: 0.2s; }
.bt-typing-dot:nth-child(3) { animation-delay: 0.4s; }
@keyframes bt-typing-bounce {
    0%, 60%, 100% { transform: translateY(0); opacity: 0.4; }
    30%           { transform: translateY(-6px); opacity: 1; }
}

/* ── Markdown inside chat messages ───────────────────────────────────────── */
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
    font-size: 0.92rem !important;
    line-height: 1.65 !important;
    margin-bottom: 0.4rem !important;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] code {
    background: rgba(99, 102, 241, 0.08) !important;
    border-radius: 6px !important;
    padding: 1px 6px !important;
    font-size: 0.84rem !important;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] pre {
    border-radius: 12px !important;
    border: 1px solid rgba(99, 102, 241, 0.12) !important;
}

/* ── Chat message bubbles ────────────────────────────────────────────────── */
[data-testid="stChatMessage"] {
    border-radius: 16px !important;
    padding: 0.75rem 1rem !important;
    margin-bottom: 0.5rem !important;
    animation: bt-msg-slide 0.3s ease forwards;
    border: none !important;
    max-width: 85% !important;
}
@keyframes bt-msg-slide {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* User messages — right-aligned, tinted */
[data-testid="stChatMessage"][data-testid-subtype="user"],
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.08) 0%, rgba(139, 92, 246, 0.06) 100%) !important;
    margin-left: auto !important;
    border-bottom-right-radius: 4px !important;
}

/* Assistant messages — left-aligned, clean */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background: rgba(241, 245, 249, 0.6) !important;
    margin-right: auto !important;
    border-bottom-left-radius: 4px !important;
}

/* Dark mode message overrides */
@media (prefers-color-scheme: dark) {
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(139, 92, 246, 0.10) 100%) !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background: rgba(30, 41, 59, 0.7) !important;
    }
    .bt-chat-hero p { color: #94a3b8; }
}
[data-testid="stApp"][data-theme="dark"] [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(139, 92, 246, 0.10) 100%) !important;
}
[data-testid="stApp"][data-theme="dark"] [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background: rgba(30, 41, 59, 0.7) !important;
}
[data-testid="stApp"][data-theme="dark"] .bt-chat-hero p { color: #94a3b8; }

/* Avatar styling */
[data-testid="stChatMessage"] [data-testid^="chatAvatarIcon"] {
    border-radius: 12px !important;
    overflow: hidden !important;
}

/* ── Chat input bar ──────────────────────────────────────────────────────── */
/*
  FIX: On first render Streamlit places the textarea and the send button as
  separate flex children aligned to "stretch" (default).  The button ends up
  vertically offset because the textarea has a taller intrinsic height.
  We override alignment to "center" on the inner wrapper div so the send
  button stays vertically centred beside the text area at every height.

  We also move the visible border + pill shape from the textarea element to the
  wrapping div so both the textarea AND the button sit inside the same rounded
  container — matching the design in the screenshot.
*/

/* Outer stChatInput container — remove any default chrome */
[data-testid="stChatInput"] {
    border-radius: 999px !important;
    padding: 0 !important;
}

/* Inner wrapper div: the direct child that holds both textarea and button */
[data-testid="stChatInput"] > div {
    display: flex !important;
    flex-direction: row !important;
    align-items: center !important;           /* vertically centre button with textarea */
    border-radius: 999px !important;
    border: 1.5px solid rgba(99, 102, 241, 0.20) !important;
    padding: 0 0.5rem 0 0 !important;
    gap: 0 !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}

/* Focus ring on the container, not the textarea */
[data-testid="stChatInput"] > div:focus-within {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.10) !important;
}

/* Textarea: no border (the container provides it); pill only on left side */
[data-testid="stChatInput"] textarea {
    border: none !important;
    background: transparent !important;
    outline: none !important;
    border-radius: 999px 0 0 999px !important;
    padding: 0.75rem 1.25rem !important;
    font-size: 0.92rem !important;
    transition: none !important;
    flex: 1 !important;
}

/* Send button: circle, vertically centred, consistent margin */
[data-testid="stChatInput"] button {
    border-radius: 50% !important;
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    align-self: center !important;            /* explicit centre in case flex-direction changes */
    flex-shrink: 0 !important;
    margin: 0.35rem !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}
[data-testid="stChatInput"] button:hover {
    transform: scale(1.08) !important;
    box-shadow: 0 4px 16px rgba(99, 102, 241, 0.30) !important;
}
</style>
"""


# ---------------------------------------------------------------------------
# Conversation memory — rolling context window
# ---------------------------------------------------------------------------

# Ollama's default context limit if we cannot retrieve it from the model info.
# 8 192 tokens covers most 7-B class models (Gemma-3, Mistral, Llama-3 etc.).
_FALLBACK_CONTEXT_TOKENS = 8_192

# We start trimming when the running conversation consumes more than this
# fraction of the model's context window.  Keeping 30 % headroom lets the
# model breathe and produce long, complete answers without getting cut off.
_CONTEXT_FILL_THRESHOLD = 0.70


# GPT-4o-mini and most hosted models have a 128k token context window.
# We use this as the fixed limit for all non-Ollama (cloud) providers.
_CLOUD_CONTEXT_LIMIT = 128_000


def _stream_chat_openai(
    messages: List[Dict[str, Any]],
    model: str,
    base_url: str,
    api_key: str,
    silent: bool = False,
) -> str:
    """Stream a chat response from an OpenAI-compatible API endpoint.

    GitHub Models, OpenAI, and similar APIs all use the same SSE wire format:
      data: {"choices": [{"delta": {"content": "token"}}]}
      data: [DONE]
    This is different from Ollama's JSON-per-line format handled in _stream_chat().
    The function streams tokens as they arrive and renders them live unless silent=True.
    """
    endpoint = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
    }
    full_response = ""
    placeholder = None if silent else st.empty()
    log.info(
        "BaseTruth AI Copilot: Sending LLM request | backend=openai-compatible provider_url=%s model=%s silent=%s message_count=%d system_chars=%d last_user_message=%r",
        base_url,
        model,
        silent,
        len(messages),
        sum(len(str(m.get("content", ""))) for m in messages if m.get("role") == "system"),
        _last_user_message(messages),
    )
    try:
        with requests.post(
            endpoint,
            json=payload,
            headers=headers,
            stream=True,
            timeout=(OLLAMA_CONNECT_TIMEOUT_SEC, OLLAMA_READ_TIMEOUT_SEC),
        ) as resp:  # nosemgrep: basetruth-ssrf
            resp.raise_for_status()
            chunks: List[str] = []
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                # OpenAI SSE lines are prefixed with "data: "
                line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                if not line.startswith("data: "):
                    continue
                payload_str = line[6:]  # strip the "data: " prefix
                if payload_str.strip() == "[DONE]":
                    break
                try:
                    data = json.loads(payload_str)
                except json.JSONDecodeError:
                    continue
                # Extract the incremental token from the delta
                token = (data.get("choices") or [{}])[0].get("delta", {}).get("content") or ""
                chunks.append(token)
                full_response = "".join(chunks)
                if placeholder is not None:
                    # Strip reasoning blocks and show "Thinking..." while the
                    # model is inside an unfinished <think> or <thought> block.
                    in_think_block = _has_open_thinking_block(full_response)
                    if in_think_block:
                        placeholder.markdown("🤔 *Thinking...*")
                    else:
                        _, _clean = _strip_thinking(full_response)
                        display_text = _clean if _clean else full_response
                        placeholder.markdown(display_text + "◌")
            # Final render with think blocks stripped
            if placeholder is not None:
                _, clean_final = _strip_thinking(full_response)
                placeholder.markdown(clean_final if clean_final else full_response)
            log.info(
                "BaseTruth AI Copilot: Raw LLM response received | backend=openai-compatible model=%s silent=%s response=%r",
                model,
                silent,
                full_response,
            )
            # Strip reasoning blocks from return value so they are not stored in history
            _, full_response = _strip_thinking(full_response)
    except requests.exceptions.ConnectionError:
        full_response = f"⚠️ Could not connect to `{base_url}`. Check the API endpoint and your network."
        if not silent:
            st.error(full_response)
    except requests.exceptions.Timeout:
        full_response = "⚠️ Request timed out. The API did not respond in time."
        if not silent:
            st.error(full_response)
    except requests.exceptions.HTTPError as exc:
        full_response = f"⚠️ API error {exc.response.status_code}: {exc.response.text[:300]}"
        if not silent:
            st.error(full_response)
    except requests.RequestException as exc:
        full_response = f"⚠️ Error: {exc}"
        if not silent:
            st.error(full_response)
    return full_response


def _call_llm(
    messages: List[Dict[str, Any]],
    provider_cfg: Dict[str, Any],
    silent: bool = False,
) -> str:
    """Route an LLM call to the correct backend based on the provider config.

    Dispatches to _stream_chat (Ollama) or _stream_chat_openai (GitHub Models,
    OpenAI, etc.) so the rest of the code never needs to know which API is active.
    """
    provider = provider_cfg.get("provider", "ollama")
    model    = provider_cfg["model"]
    base_url = provider_cfg.get("base_url", "")
    api_key  = provider_cfg.get("api_key", "")
    if provider == "ollama":
        return _stream_chat(messages, model=model, base_url=base_url, silent=silent)
    # All other providers use the OpenAI-compatible streaming format
    return _stream_chat_openai(messages, model=model, base_url=base_url, api_key=api_key, silent=silent)


def _estimate_tokens(text: str) -> int:
    """Rough BPE token count — 4 characters ≈ 1 token for English text.

    This is an intentional under-estimate: it errs on the side of keeping
    slightly more context rather than dropping messages too aggressively.
    """
    return max(1, len(text) // 4)


def _get_model_context_limit(base_url: str, model: str) -> int:
    """Query Ollama's /api/show endpoint for the model's declared context length.

    Different model families store the value under different keys
    (llama.context_length, gemma.context_length, etc.), so we scan every key
    in the modelinfo dict that contains the substring 'context_length'.
    If the endpoint is unreachable or the key is absent we fall back to a
    conservative default so the caller always gets a usable number.
    """
    try:
        resp = requests.post(
            f"{base_url}/api/show",
            json={"name": model},
            timeout=(3, 6),
        )
        if resp.ok:
            model_info: Dict[str, Any] = resp.json().get("modelinfo", {})
            for key, value in model_info.items():
                if "context_length" in key:
                    return int(value)
    except Exception:
        # Network error or model not loaded yet — use the fallback silently
        pass
    return _FALLBACK_CONTEXT_TOKENS


def _apply_rolling_window(
    messages: List[Dict[str, Any]],
    system_prompt: str,
    context_limit: int,
) -> List[Dict[str, Any]]:
    """Return a trimmed copy of *messages* that fits within the context budget.

    Strategy — rolling window:
      1. Estimate total tokens = system_prompt tokens + all message tokens.
      2. If the total exceeds 70 % of *context_limit*, drop the oldest
         user+assistant pair and re-check, repeating until the budget is met.
      3. If only a single message remains (the user's latest question) we stop
         trimming to avoid sending an empty conversation.

    The system prompt is never removed — it defines the model's identity and
    SQL-query rules.  The *messages* list held in session state is NOT mutated
    here; the caller keeps the full history for display and only sends the
    trimmed slice to Ollama.
    """
    target_tokens = int(context_limit * _CONTEXT_FILL_THRESHOLD)

    # Token budget consumed by the fixed system prompt
    system_tokens = _estimate_tokens(system_prompt)

    # Build a mutable copy so we never touch the original session-state list
    trimmed: List[Dict[str, Any]] = list(messages)

    # Running total for the conversation turns
    msg_tokens = sum(_estimate_tokens(m["content"]) for m in trimmed)

    while (system_tokens + msg_tokens) > target_tokens and len(trimmed) > 1:
        # Always drop a full turn (user message + its assistant reply) so the
        # conversation stays coherent — an assistant message without its
        # preceding question would confuse the model.
        if trimmed[0]["role"] == "user":
            dropped = trimmed.pop(0)
            msg_tokens -= _estimate_tokens(dropped["content"])
            # Drop the paired assistant reply if it follows immediately
            if trimmed and trimmed[0]["role"] == "assistant":
                dropped = trimmed.pop(0)
                msg_tokens -= _estimate_tokens(dropped["content"])
        else:
            # Orphaned assistant message at the front (shouldn't happen in
            # normal flow, but handle it gracefully)
            dropped = trimmed.pop(0)
            msg_tokens -= _estimate_tokens(dropped["content"])

    return trimmed


def _stream_chat(messages: List[Dict[str, Any]], model: str, base_url: str, silent: bool = False) -> str:
    """Send messages to Ollama and collect the full response.

    When silent=True tokens are accumulated without rendering to the UI.
    The caller is responsible for deciding how to display the result.
    This is used for the first-pass SQL/MinIO detection call so the user
    only ever sees the final human-readable answer, not the intermediate
    query-generation step.

    For visible calls (silent=False), reasoning blocks (<think>...</think> or
    <thought>...</thought>) are detected as tokens arrive. While the model is
    inside one of those blocks, a "🤔 Thinking..." indicator is shown instead of
    the raw reasoning text. Once the block ends, the actual answer streams
    normally. The final returned string always has the reasoning blocks removed
    so callers store only the clean human-readable answer in session history.
    """
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
    }
    full_response = ""
    # Only create a UI placeholder when we want to stream output to the screen
    placeholder = None if silent else st.empty()
    chat_endpoint = f"{base_url}/api/chat"
    log.info(
        "BaseTruth AI Copilot: Sending LLM request | backend=ollama base_url=%s model=%s silent=%s message_count=%d system_chars=%d last_user_message=%r",
        base_url,
        model,
        silent,
        len(messages),
        sum(len(str(m.get("content", ""))) for m in messages if m.get("role") == "system"),
        _last_user_message(messages),
    )
    try:
        with requests.post(
            chat_endpoint,
            json=payload,
            stream=True,
            timeout=(OLLAMA_CONNECT_TIMEOUT_SEC, OLLAMA_READ_TIMEOUT_SEC),
        ) as resp:  # nosemgrep: basetruth-ssrf
            resp.raise_for_status()
            chunks: List[str] = []
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                try:
                    data: Dict[str, Any] = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue
                token = data.get("message", {}).get("content", "")
                chunks.append(token)
                full_response = "".join(chunks)
                if placeholder is not None:
                    # Check if the model is still inside an unfinished reasoning
                    # block. While that block is open, keep showing a generic
                    # thinking indicator instead of raw internal notes.
                    _thinking_content, _clean_so_far = _strip_thinking(full_response)
                    in_think_block = _has_open_thinking_block(full_response)
                    if in_think_block:
                        # Show a pulsing "Thinking..." indicator while the model reasons
                        placeholder.markdown("🤔 *Thinking...*")
                    else:
                        # Show the clean answer (think blocks stripped) as it streams
                        display_text = _clean_so_far if _clean_so_far else full_response
                        placeholder.markdown(display_text + "▌")
                if data.get("done"):
                    break
            # Final render: strip thinking blocks and show only the clean answer
            if placeholder is not None:
                _, clean_final = _strip_thinking(full_response)
                placeholder.markdown(clean_final if clean_final else full_response)
            log.info(
                "BaseTruth AI Copilot: Raw LLM response received | backend=ollama model=%s silent=%s response=%r",
                model,
                silent,
                full_response,
            )
            # Return the clean response so reasoning blocks are never stored in chat history
            _, full_response = _strip_thinking(full_response)
    except requests.exceptions.ConnectionError:
        full_response = (
            f"⚠️ Could not connect to Ollama at `{base_url}`. "
            "Make sure Ollama is running."
        )
        if not silent:
            st.error(full_response)
    except requests.exceptions.Timeout:
        full_response = "⚠️ Request timed out. The model may be loading — please try again."
        if not silent:
            st.error(full_response)
    except requests.RequestException as exc:
        full_response = f"⚠️ Error: {exc}"
        if not silent:
            st.error(full_response)
    return full_response


def _retry_sql_with_db_error(
    user_question: str,
    original_sql: str,
    db_error_text: str,
    system_prompt: str,
    provider_cfg: Dict[str, Any],
) -> str:
    """Ask the model once to repair a SQL query after a DB schema error.

    The first-pass query generation can still occasionally use a wrong table or
    column name even with DATABASE.md in context. When PostgreSQL returns an
    UndefinedTable/UndefinedColumn error, we send the exact DB error back to the
    model and ask for a corrected SQL query only. This gives the copilot one
    automatic self-correction round before the user sees a failure.
    """
    retry_directive = (
        "\n\n[SQL RETRY AFTER DB ERROR — NOT VISIBLE TO USER]\n"
        "The previous SQL failed against the live PostgreSQL schema.\n"
        "Read the DATABASE.md schema already present in the system prompt and the\n"
        "exact DB error below, then emit ONLY one corrected PostgreSQL SELECT inside\n"
        "a triple-backtick sql block. Do not explain anything. Do not repeat the bad SQL.\n"
        "Never use 'entity_id' on the 'entities' table; its primary key is 'id'."
    )
    retry_messages = [
        {"role": "system", "content": system_prompt + retry_directive},
        {
            "role": "user",
            "content": (
                f"Original user question:\n{user_question}\n\n"
                f"Previous SQL:\n{original_sql}\n\n"
                f"Database error:\n{db_error_text}"
            ),
        },
    ]
    retry_response = _call_llm(retry_messages, provider_cfg, silent=True)
    retry_match = re.search(r"```sql\s*(.*?)\s*```", retry_response, re.IGNORECASE | re.DOTALL)
    if not retry_match:
        log.warning(
            "BaseTruth AI Copilot: SQL retry did not return a sql block | response_preview=%r",
            retry_response[:300],
        )
        return ""
    corrected_sql = retry_match.group(1).strip()
    log.info(
        "BaseTruth AI Copilot: Retrying DB query after schema error | corrected_sql=%s",
        corrected_sql,
    )
    return corrected_sql


# ---------------------------------------------------------------------------
# Defence helpers against lexical hallucination
# ---------------------------------------------------------------------------

# Mapping from informal words the user might type → the hint we inject into
# the question so the model sees the correct concept before it builds SQL.
# Keys are matched as whole words (case-insensitive) in the user message.
# Order matters: longer / more-specific phrases must come first.
_TERM_NORMALISATIONS = [
    # "users have uploaded documents" → "entities have uploaded scans"
    (r"\busers\b",          "entities (applicants)"),
    (r"\bcustomers\b",      "entities (applicants)"),
    (r"\bapplicants\b",     "entities (applicants)"),
    (r"\bpeople\b",         "entities (applicants)"),
    (r"\bpersons\b",        "entities (applicants)"),
    (r"\bindividuals\b",    "entities (applicants)"),
    (r"\bborrowers\b",      "entities (applicants)"),
    (r"\bclients\b",        "entities (applicants)"),
    (r"\bemployees\b",      "entities (applicants)"),
    # document synonyms — must come before generic "uploads"
    (r"\buploaded documents\b",  "scans (uploaded documents)"),
    (r"\bdocuments uploaded\b",  "scans (uploaded documents)"),
    (r"\bdocument uploads\b",    "scans (uploaded documents)"),
    (r"\bfile uploads\b",        "scans (uploaded documents)"),
    (r"\buploads\b",             "scans (uploaded documents)"),
    (r"\bfiles\b",               "scans or document_extractions"),
    (r"\bdocuments\b",           "scans or document_extractions"),
    (r"\bpapers\b",              "scans (uploaded documents)"),
    # check / report synonyms
    (r"\bchecks\b",              "identity_checks"),
    (r"\bfinal reports\b",       "entity_reports"),
    (r"\bverification reports\b","entity_reports"),
    (r"\breports\b",             "entity_reports"),
]

# Compiled once at import time for performance
_COMPILED_NORMALISATIONS = [
    (re.compile(pattern, re.IGNORECASE), replacement)
    for pattern, replacement in _TERM_NORMALISATIONS
]


def _normalise_question_in_messages(
    windowed: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Rewrite informal synonym words in the last visible user message.

    Small models like gemma4:e4b perform *lexical copying* — they take words
    directly from the user's question and use them as SQL table names.  Asking
    "how many users have uploaded documents?" reliably produces
    ``FROM users JOIN documents`` even when the system prompt says those tables
    don't exist.

    This function rewrites the last non-hidden user message in the windowed
    history so that informal words are replaced with the exact BaseTruth
    table/concept names before Phase 1 feeds the messages to the model.
    The original message in session state is NOT mutated — only the copy sent
    to the LLM is changed, so the user still sees their original wording in
    the chat UI.
    """
    # Find the last non-hidden user message index (working backwards)
    last_user_idx: int | None = None
    for i in range(len(windowed) - 1, -1, -1):
        if windowed[i]["role"] == "user" and not windowed[i].get("_hidden"):
            last_user_idx = i
            break

    if last_user_idx is None:
        return windowed  # nothing to rewrite

    original_text = windowed[last_user_idx]["content"]
    normalised_text = original_text
    for pattern, replacement in _COMPILED_NORMALISATIONS:
        normalised_text = pattern.sub(replacement, normalised_text)

    if normalised_text != original_text:
        log.info(
            "BaseTruth AI Copilot: Question normalised for Phase 1"
            " | original=%r normalised=%r",
            original_text[:200],
            normalised_text[:200],
        )
        # Return a shallow copy of the list with the rewritten message
        result = list(windowed)
        result[last_user_idx] = {**windowed[last_user_idx], "content": normalised_text}
        return result

    return windowed


# Deterministic table-name corrections applied to any SQL the LLM emits.
# Maps hallucinated table names → correct BaseTruth table names.
# This is a last-resort safety net after question normalisation.
_SQL_TABLE_CORRECTIONS: Dict[str, str] = {
    "users":            "entities",
    "customers":        "entities",
    "applicants":       "entities",
    "people":           "entities",
    "persons":          "entities",
    "members":          "entities",
    "borrowers":        "entities",
    "document_uploads": "scans",
    "documents":        "scans",
    "uploads":          "scans",
    "files":            "scans",
    "checks":           "identity_checks",
    "reports":          "entity_reports",
    "document_information": "document_extractions",
}


def _sanitise_sql(sql: str) -> str:
    """Replace hallucinated table names in LLM-generated SQL with correct ones.

    Works as a last-resort after _normalise_question_in_messages().  Uses
    whole-word regex replacement so ``entity_reports`` is never mangled into
    ``entity_entity_reports``.  Only applies corrections for table names that
    appear after FROM or JOIN keywords to avoid clobbering column aliases.
    """
    corrected = sql
    for bad, good in _SQL_TABLE_CORRECTIONS.items():
        # Match keyword + whitespace as a capturing group, then the bad table
        # name as a whole word. The replacement preserves the captured keyword
        # so we don't strip FROM/JOIN from the output.
        # NOTE: Cannot use a variable-width lookbehind in Python re — use a
        # capturing group + \1 backreference instead.
        pattern = re.compile(
            r"((?:FROM|JOIN|,)\s+)" + re.escape(bad) + r"\b",
            re.IGNORECASE,
        )
        corrected = pattern.sub(r"\g<1>" + good, corrected)

    if corrected != sql:
        log.warning(
            "BaseTruth AI Copilot: SQL sanitised — hallucinated table name corrected"
            " | before=%r after=%r",
            sql[:300],
            corrected[:300],
        )
    return corrected


def _do_reply(
    messages: List[Dict[str, Any]],
    system_prompt: str,
    provider_cfg: Dict[str, Any],
    context_limit: int,
) -> None:
    """Handle one assistant reply turn, showing the user exactly one chat bubble.

    This function implements a two-phase approach to combine SQL/MinIO execution
    with natural-language explanation into a single visible response bubble:

    Phase 1 (silent): call the LLM without rendering output. The model decides
    whether to emit a SQL block, a MinIO block, both, or neither.

    Phase 2 (visible): after executing any queries silently and appending the
    results to the conversation context, the model is called again. This second
    call is the ONLY one streamed to the UI — giving a single clean answer
    regardless of whether a data query was needed.

    Mixed-mode queries (e.g. "how many rejected PAN cards and why?") are handled
    naturally because the model can emit a SQL block in Phase 1, receive results,
    and then write both the data findings AND the expert explanation in Phase 2.
    """
    # Log the user question so we can trace what triggered this response turn
    model_name = provider_cfg["model"]
    last_user_msg = next((m["content"] for m in reversed(messages) if m["role"] == "user" and not m.get("_hidden")), "")
    log.info(
        "BaseTruth AI Copilot: Processing user question | provider=%s model=%s question=%r",
        provider_cfg.get("provider", "ollama"), model_name, last_user_msg[:300],
    )

    # ── Defence 1: Question normalisation ────────────────────────────────────
    # Small models copy words from the user question directly into SQL table
    # names (e.g. "documents" → FROM documents, "users" → FROM users).
    # We deterministically rewrite informal synonyms in the last user message
    # to the exact BaseTruth table/concept names BEFORE Phase 1 sees the text.
    # This is the single most effective fix for lexical hallucination.
    windowed = _apply_rolling_window(messages, system_prompt, context_limit)
    windowed = _normalise_question_in_messages(windowed)

    # Phase 1 — silent call: ask the model to emit a SQL/MinIO block if data is needed.
    # A directive-enhanced system prompt is injected ONLY for Phase 1 to force the model
    # to emit a ```sql``` or ```minio``` block immediately rather than responding with
    # conversational filler like "Sure, let me check" (which would give has_sql=False).
    _PHASE1_DIRECTIVE = (
        "\n\n[PHASE 1 — INTERNAL QUERY CHECK — NOT VISIBLE TO USER]\n"
        "Examine the LAST user message only. Decide:\n"
        "  A. Does it require fetching data from the database? "
        "→ Emit ONLY a ```sql SELECT...``` block. Nothing else.\n"
        "  B. Does it require listing files in MinIO storage? "
        "→ Emit ONLY a ```minio LIST...``` block. Nothing else.\n"
        "  C. Can it be answered from existing conversation context or general knowledge? "
        "→ Answer directly and concisely.\n"
        "Before emitting SQL, verify that EVERY table name is one of: "
        "entities, scans, document_extractions, identity_checks, entity_reports.\n"
        "Never use users, names, documents, uploads, customers, applicants, checks, or reports.\n"
        "Never use entity_id on the entities table; its primary key is id.\n"
        "NEVER say 'Let me check', 'Sure', 'I\'ll look', or any similar filler "
        "— emit the SQL block or the answer immediately."
    )
    phase1_system = system_prompt + _PHASE1_DIRECTIVE
    phase1_messages = [
        {"role": "system", "content": phase1_system}
    ] + windowed
    log.debug("BaseTruth AI Copilot: Phase 1 — sending silent request to model to detect query intent")
    phase1_indicator = st.empty()
    phase1_indicator.markdown("🤔 *Thinking...*")
    try:
        first_response = _call_llm(phase1_messages, provider_cfg, silent=True)
    finally:
        phase1_indicator.empty()
    log.info(
        "BaseTruth AI Copilot: Phase 1 raw response | response=%r",
        first_response,
    )

    sql_match   = re.search(r"```sql\s*(.*?)\s*```", first_response, re.IGNORECASE | re.DOTALL)
    minio_match = re.search(r"```minio\s*(.*?)\s*```", first_response, re.IGNORECASE | re.DOTALL)

    log.info(
        "BaseTruth AI Copilot: Phase 1 complete | has_sql=%s has_minio=%s response_length=%d",
        bool(sql_match), bool(minio_match), len(first_response),
    )

    # Track whether any data was injected so we know if Phase 2 is required
    data_injected = False

    if sql_match:
        # Model requested a DB query — execute it silently, then stream the
        # human-readable explanation as the sole visible response.
        sql_query = sql_match.group(1).strip()
        # ── Defence 2: SQL sanitisation ──────────────────────────────────────
        # Catch any hallucinated table names the model still emitted despite
        # question normalisation, and remap them to the correct BaseTruth names.
        sql_query = _sanitise_sql(sql_query)
        # ─────────────────────────────────────────────────────────────────────
        log.info(
            "BaseTruth AI Copilot: Executing DB query from silent first pass | sql=%s",
            sql_query,
        )
        result_table = execute_safe_query(sql_query)
        if "CORRECTION REQUIRED:" in result_table:
            corrected_sql = _retry_sql_with_db_error(
                user_question=last_user_msg,
                original_sql=sql_query,
                db_error_text=result_table,
                system_prompt=system_prompt,
                provider_cfg=provider_cfg,
            )
            if corrected_sql:
                retry_result = execute_safe_query(corrected_sql)
                if "CORRECTION REQUIRED:" not in retry_result:
                    sql_query = corrected_sql
                    result_table = retry_result
        log.info(
            "BaseTruth AI Copilot: DB query result ready | result_preview=%r",
            result_table[:300],
        )
        # Store intermediate SQL turn and query results with a hidden flag so
        # the history render loop skips them.  They stay in the conversation
        # for the model's context window but never appear as chat bubbles.
        messages.append({"role": "assistant", "content": first_response, "_hidden": True})
        messages.append({
            "role": "user",
            "_hidden": True,
            "content": (
                "System Query Result — present the findings as a senior compliance analyst.\n\n"
                "Use this EXACT structure with each heading on its own line:\n"
                "## 📊 Executive Summary\n\n"
                "<one short paragraph>\n\n"
                "## 📋 Key Findings\n\n"
                "• Each finding on its own bullet line\n\n"
                "## ✅ Recommended Actions\n\n"
                "• Each action on its own bullet line\n\n"
                "Do NOT put headings and content on the same line. "
                "Do not show raw table data or SQL.\n\n"
                f"{result_table}"
            ),
        })
        data_injected = True

    if minio_match:
        # Execute MinIO storage command silently, inject result as hidden context
        minio_cmd = minio_match.group(1).strip()
        log.info(
            "BaseTruth AI Copilot: Executing MinIO command from silent first pass | cmd=%s",
            minio_cmd,
        )
        result_table = query_minio_objects(minio_cmd)
        log.info(
            "BaseTruth AI Copilot: MinIO result ready | result_preview=%r",
            result_table[:300],
        )
        # Only append the silent SQL turn once (it was already appended above if sql_match hit)
        if not sql_match:
            messages.append({"role": "assistant", "content": first_response, "_hidden": True})
        messages.append({
            "role": "user",
            "_hidden": True,
            "content": (
                "System Storage Result — summarise in plain English:\n\n"
                f"{result_table}"
            ),
        })
        data_injected = True

    if data_injected:
        # Phase 2 — visible: stream the final expert explanation to the user
        log.debug("BaseTruth AI Copilot: Phase 2 — sending visible request with data context")
        windowed2 = _apply_rolling_window(messages, system_prompt, context_limit)
        api_messages2 = [{"role": "system", "content": system_prompt}] + windowed2
        with st.chat_message("assistant", avatar="💬"):
            final_response = _call_llm(api_messages2, provider_cfg, silent=False)
        log.info(
            "BaseTruth AI Copilot: Phase 2 response delivered | response_length=%d",
            len(final_response),
        )
        log.info(
            "BaseTruth AI Copilot: Phase 2 final response | response=%r",
            final_response,
        )
        messages.append({"role": "assistant", "content": final_response})

    else:
        # No data query needed — pure knowledge answer.
        # The silent Phase 1 call already produced the full answer in first_response.
        # Strip any thinking blocks before displaying — models that include
        # <think>...</think> reasoning should not show that to the user.
        _, clean_first_response = _strip_thinking(first_response)
        display_response = clean_first_response if clean_first_response else first_response
        log.info(
            "BaseTruth AI Copilot: Knowledge-only response (no SQL/MinIO) | response_length=%d",
            len(display_response),
        )
        log.info(
            "BaseTruth AI Copilot: Knowledge-only final response | response=%r",
            display_response,
        )
        with st.chat_message("assistant", avatar="💬"):
            st.markdown(display_response)
        messages.append({"role": "assistant", "content": display_response})


def _page_gemma_chat() -> None:
    # ── Inject custom CSS ─────────────────────────────────────────────────
    st.markdown(_CHAT_CSS, unsafe_allow_html=True)

    # ── Page title ────────────────────────────────────────────────────────
    st.markdown(_page_title("💬", "BaseTruth AI Copilot"), unsafe_allow_html=True)

    # ── Resolve provider + model for the AI Copilot feature ─────────────
    # get_provider_config_for_feature reads settings.json["models"]["qna_copilot"]
    # and resolves the full provider config (provider name, model, base_url, api_key).
    # The result is cached in session state so we don't re-read the file on every render.
    if "bt_provider_cfg" not in st.session_state:
        st.session_state["bt_provider_cfg"] = get_provider_config_for_feature("qna_copilot")
    provider_cfg: Dict[str, Any] = st.session_state["bt_provider_cfg"]

    if provider_cfg["provider"] == "ollama":
        # ── Ollama path: probe the local endpoint once and cache the result ──
        if "bt_ollama_probe" not in st.session_state:
            st.session_state["bt_ollama_probe"] = probe_ollama()
        base_url, available_models, attempted_urls = st.session_state["bt_ollama_probe"]

        if not base_url:
            attempted_text = "\n".join(f"- `{url}`" for url in attempted_urls)
            st.markdown(
                '<div class="bt-chat-hero">'
                '<div class="bt-chat-hero-icon">💬</div>'
                "<h2>Connection Required</h2>"
                "<p>Ollama is not reachable. Start the Ollama service to begin chatting.</p>"
                "</div>",
                unsafe_allow_html=True,
            )
            st.error(
                "**Ollama is not reachable from this UI runtime.**\n\n"
                "Attempted endpoints:\n"
                f"{attempted_text}\n\n"
                "If the UI is running in Docker, Ollama must be reachable from the container.",
                icon="🔴",
            )
            if st.button("🔄  Retry Connection", use_container_width=True, type="primary"):
                # Clear both caches so the next render does a fresh probe
                st.session_state.pop("bt_ollama_probe", None)
                st.session_state.pop("bt_provider_cfg", None)
                st.rerun()
            st.stop()

        # Resolve the exact installed model name — prefer the configured one,
        # fall back to any gemma4 variant, then whatever Ollama has available
        preferred_model = provider_cfg["model"]
        model_name = select_ollama_model(available_models, preferred_substring="gemma4") \
            if available_models else preferred_model
        for m in available_models:
            if m.lower() == preferred_model.lower():
                model_name = m
                break

        # Merge the resolved base_url and model back into provider_cfg for _call_llm
        provider_cfg = {**provider_cfg, "model": model_name, "base_url": base_url}

        # Fetch the model's declared context window from Ollama once per session
        context_limit_key = f"bt_ctx_limit_{model_name}"
        if context_limit_key not in st.session_state:
            st.session_state[context_limit_key] = _get_model_context_limit(base_url, model_name)
        context_limit: int = st.session_state[context_limit_key]

    else:
        # ── Cloud provider path (GitHub Models, OpenAI, etc.) ─────────────
        # No Ollama probe needed — base_url and api_key are already in provider_cfg.
        model_name = provider_cfg["model"]
        if not provider_cfg.get("api_key"):
            st.warning(
                f"⚠️ No API key set for provider **{provider_cfg['provider']}**. "
                "Add your API key to `artifacts/config/settings.json` under `providers`.",
                icon="🔑",
            )
        # GPT-4o-mini and similar cloud models support up to 128k tokens context
        context_limit = _CLOUD_CONTEXT_LIMIT

    log.info(
        "BaseTruth AI Copilot: Provider resolved | provider=%s model=%s",
        provider_cfg["provider"], model_name,
    )

    # ── Init session state ────────────────────────────────────────────────
    if "gemma_messages" not in st.session_state:
        st.session_state["gemma_messages"] = []

    # Upload DATABASE.md to MinIO docs bucket once per session so every runtime
    # schema read comes from the same MinIO object instead of mixing sources.
    if _minio_available_cached() and "bt_db_md_synced" not in st.session_state:
        sync_ok = sync_database_md_to_minio()
        if sync_ok:
            log.info("BaseTruth AI Copilot: DATABASE.md synced to MinIO successfully")
        else:
            log.warning("BaseTruth AI Copilot: DATABASE.md sync to MinIO failed")
        st.session_state["bt_db_md_synced"] = True

    # ── Handle suggestion chip clicks ─────────────────────────────────────
    if "bt_qa_pending" in st.session_state:
        pending = st.session_state.pop("bt_qa_pending")
        st.session_state["gemma_messages"].append(
            {"role": "user", "content": pending}
        )

    messages: List[Dict[str, Any]] = st.session_state["gemma_messages"]

    # ── Suggestion chips — cover all three modes: DATA, KNOWLEDGE, and MIXED ──
    # Row 1: data-query chips (trigger a SQL lookup)
    _SUGGESTIONS_ROW1 = [
        ("👥", "How many applicants are in the system?"),
        ("⏳", "How many documents are pending review?"),
        ("🚨", "Show me high-risk applicants"),
        ("❌", "Show all rejected documents"),
    ]
    # Row 2: mixed + knowledge chips
    _SUGGESTIONS_ROW2 = [
        ("🔒", "Which applicants failed face match and why?"),
        ("🏠", "What documents are needed for a home loan?"),
        ("📋", "What documents are required for BGV?"),
        ("🪪", "Explain Aadhaar and PAN verification in India"),
    ]

    if not messages:
        # Welcome hero — shown only when the chat is empty
        st.markdown(
            '<div class="bt-chat-hero">'
            '<div class="bt-chat-hero-icon">💬</div>'
            "<h2>BaseTruth AI Copilot</h2>"
            "<p>Your enterprise compliance analyst. Ask about applicants, approvals, "
            "fraud risk, identity checks — or industry topics like KYC, BGV, and mortgage requirements.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        # Row 1: data-query chips
        chip_cols1 = st.columns(len(_SUGGESTIONS_ROW1))
        for idx, (icon, label) in enumerate(_SUGGESTIONS_ROW1):
            with chip_cols1[idx]:
                if st.button(
                    f"{icon}  {label}",
                    key=f"bt_chip_r1_{idx}",
                    use_container_width=True,
                ):
                    st.session_state["bt_qa_pending"] = label
                    st.rerun()

        # Row 2: general-knowledge chips
        chip_cols2 = st.columns(len(_SUGGESTIONS_ROW2))
        for idx, (icon, label) in enumerate(_SUGGESTIONS_ROW2):
            with chip_cols2[idx]:
                if st.button(
                    f"{icon}  {label}",
                    key=f"bt_chip_r2_{idx}",
                    use_container_width=True,
                ):
                    st.session_state["bt_qa_pending"] = label
                    st.rerun()

    else:
        # Show a "Clear chat" button aligned to the right when conversation exists.
        # This lets operators start fresh without refreshing the browser tab.
        _, _clear_col = st.columns([8, 1])
        with _clear_col:
            if st.button("🗑️ Clear", key="bt_clear_chat", help="Clear this conversation"):
                log.debug("BaseTruth Q&A: User cleared the chat conversation history.")
                st.session_state["gemma_messages"] = []
                st.session_state.pop("bt_qa_replied", None)
                st.rerun()

    # ── Render chat history ───────────────────────────────────────────────
    # Skip hidden messages (intermediate SQL/MinIO turns used only for model
    # context) so they never appear as visible chat bubbles.
    for msg in messages:
        if msg.get("_hidden"):
            continue
        role = msg["role"]
        avatar = "💬" if role == "assistant" else "👤"
        with st.chat_message(role, avatar=avatar):
            st.markdown(msg["content"])

    # ── Build system prompt dynamically ───────────────────────────────────
    # DATABASE.md is loaded from filesystem (dev) or MinIO (Docker/prod) and its
    # Technical Reference section is always included. This is what lets the model
    # know that 'entities' is the correct table (not 'users') and prevents wrong
    # column name guesses. See get_schema_summary() for the token budget analysis.
    system_prompt = _build_system_prompt()

    # ── Auto-send pending suggestion (just appended above) ────────────────
    _needs_reply = (
        messages
        and messages[-1]["role"] == "user"
        and "bt_qa_replied" not in st.session_state
    )
    if _needs_reply:
        st.session_state["bt_qa_replied"] = True
        # Single entry point for all reply logic — silently detects SQL/MinIO
        # and streams only the final human-readable answer to the UI.
        _do_reply(messages, system_prompt, provider_cfg, context_limit)
        st.session_state["gemma_messages"] = messages
    else:
        st.session_state.pop("bt_qa_replied", None)

    # ── Chat input ────────────────────────────────────────────────────────
    if user_input := st.chat_input("Message BaseTruth Q&A…"):
        log.info(
            "BaseTruth AI Copilot: User submitted question | question=%r provider=%s model=%s",
            user_input[:300], provider_cfg.get("provider", "ollama"), model_name,
        )
        messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)

        # Single entry point: silently detects SQL/MinIO and streams only the
        # final human-readable answer so the user sees one clean response.
        _do_reply(messages, system_prompt, provider_cfg, context_limit)
        st.session_state["gemma_messages"] = messages
