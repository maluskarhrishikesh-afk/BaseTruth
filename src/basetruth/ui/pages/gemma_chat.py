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
)
from basetruth.integrations.db_query import (
    get_schema_summary,
    get_minio_summary,
    get_qna_system_prompt,
    get_qna_minio_instructions,
    execute_safe_query,
    query_minio_objects,
)
from basetruth.ui.components import _page_title, _db_available_cached, _minio_available_cached
from basetruth.logger import get_logger

log = get_logger(__name__)

_DEFAULT_MODEL = DEFAULT_OLLAMA_MODEL


def _build_system_prompt() -> str:
    """Build the full system prompt for the Q&A LLM.

    The base identity and behaviour rules come from qna_prompts.md (system_prompt section)
    so operators can tune them without touching Python. DB schema rules (also from the md)
    are appended only when the database is reachable. MinIO instructions are appended
    only when MinIO is reachable. This keeps the prompt tight and avoids confusing
    the model with unavailable resources.
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

    return prompt

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
    messages: List[Dict[str, str]],
    system_prompt: str,
    context_limit: int,
) -> List[Dict[str, str]]:
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
    trimmed: List[Dict[str, str]] = list(messages)

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


def _stream_chat(messages: List[Dict[str, str]], model: str, base_url: str, silent: bool = False) -> str:
    """Send messages to Ollama and collect the full response.

    When silent=True tokens are accumulated without rendering to the UI.
    The caller is responsible for deciding how to display the result.
    This is used for the first-pass SQL/MinIO detection call so the user
    only ever sees the final human-readable answer, not the intermediate
    query-generation step.
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
                    placeholder.markdown(full_response + "▌")
                if data.get("done"):
                    break
            if placeholder is not None:
                placeholder.markdown(full_response)
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

def _do_reply(
    messages: List[Dict[str, str]],
    system_prompt: str,
    model_name: str,
    base_url: str,
    context_limit: int,
) -> None:
    """Handle one assistant reply turn, showing the user exactly one chat bubble.

    Genuinely combining SQL generation and result explanation into a single
    Gemma4 API call is not possible — the model must produce the SQL first,
    the system must execute it, and only then can the model see the results
    to explain them.  This function achieves the same user-facing outcome by:

    Phase 1 (silent): call Gemma4 without rendering output.  If the response
    contains a SQL or MinIO block, execute the query silently and append the
    results to the conversation context.

    Phase 2 (visible): call Gemma4 again with the query results already in
    context.  This is the only call streamed to the UI, so the user sees a
    single, clean answer bubble regardless of whether a query was run.

    For questions that need no data query, Phase 1 returns a plain answer and
    Phase 2 is skipped — the answer is displayed directly.
    """
    windowed = _apply_rolling_window(messages, system_prompt, context_limit)
    api_messages = [{"role": "system", "content": system_prompt}] + windowed

    # Phase 1 — silent call: collect the model output without showing it yet
    first_response = _stream_chat(api_messages, model=model_name, base_url=base_url, silent=True)

    sql_match = re.search(r"```sql\s*(.*?)\s*```", first_response, re.IGNORECASE | re.DOTALL)
    minio_match = re.search(r"```minio\s*(.*?)\s*```", first_response, re.IGNORECASE | re.DOTALL)

    if sql_match:
        # Model requested a DB query — execute it silently, then stream the
        # human-readable explanation as the sole visible response.
        sql_query = sql_match.group(1).strip()
        log.info(
            "BaseTruth Q&A: Executing DB query from silent first pass.",
            extra={"sql_query": sql_query},
        )
        result_table = execute_safe_query(sql_query)
        # Store intermediate SQL turn and query results with a hidden flag so
        # the history render loop skips them.  They stay in the conversation
        # for the model's context window but never appear as chat bubbles.
        messages.append({"role": "assistant", "content": first_response, "_hidden": True})
        messages.append({
            "role": "user",
            "_hidden": True,
            "content": (
                "System Query Result "
                "(present the findings naturally — do not show raw table data):\n\n"
                f"{result_table}"
            ),
        })
        # Phase 2 — visible: stream the final explanation to the user
        windowed2 = _apply_rolling_window(messages, system_prompt, context_limit)
        api_messages2 = [{"role": "system", "content": system_prompt}] + windowed2
        with st.chat_message("assistant", avatar="💬"):
            final_response = _stream_chat(api_messages2, model=model_name, base_url=base_url, silent=False)
        messages.append({"role": "assistant", "content": final_response})

    elif minio_match:
        # Same two-phase pattern for MinIO storage queries
        minio_cmd = minio_match.group(1).strip()
        log.info(
            "BaseTruth Q&A: Executing MinIO command from silent first pass.",
            extra={"minio_cmd": minio_cmd},
        )
        result_table = query_minio_objects(minio_cmd)
        # Same hidden-flag pattern as the SQL path above
        messages.append({"role": "assistant", "content": first_response, "_hidden": True})
        messages.append({
            "role": "user",
            "_hidden": True,
            "content": (
                "System Storage Result "
                "(present the findings naturally — do not show raw table data):\n\n"
                f"{result_table}"
            ),
        })
        windowed2 = _apply_rolling_window(messages, system_prompt, context_limit)
        api_messages2 = [{"role": "system", "content": system_prompt}] + windowed2
        with st.chat_message("assistant", avatar="💬"):
            final_response = _stream_chat(api_messages2, model=model_name, base_url=base_url, silent=False)
        messages.append({"role": "assistant", "content": final_response})

    else:
        # No data query needed — display the first response directly
        with st.chat_message("assistant", avatar="💬"):
            st.markdown(first_response)
        messages.append({"role": "assistant", "content": first_response})


def _page_gemma_chat() -> None:
    # ── Inject custom CSS ─────────────────────────────────────────────────
    st.markdown(_CHAT_CSS, unsafe_allow_html=True)

    # ── Page title ────────────────────────────────────────────────────────
    st.markdown(_page_title("💬", "BaseTruth Q&A"), unsafe_allow_html=True)

    # ── Probe Ollama — cached in session state so we don't re-probe on every
    # Streamlit re-render (probe makes HTTP requests with 5 s timeouts and
    # would add seconds of latency before each response if uncached).
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
            # Clear the cached probe so the next render does a fresh connection attempt
            st.session_state.pop("bt_ollama_probe", None)
            st.rerun()
        st.stop()

    # ── Use first available model ─────────────────────────────────────────
    model_name = available_models[0] if available_models else _DEFAULT_MODEL

    # ── Fetch model context window once per render ────────────────────────
    # We cache it in session state so we only call /api/show on the first
    # render — subsequent re-runs reuse the stored value.
    context_limit_key = f"bt_ctx_limit_{model_name}"
    if context_limit_key not in st.session_state:
        st.session_state[context_limit_key] = _get_model_context_limit(base_url, model_name)
    context_limit: int = st.session_state[context_limit_key]

    # ── Init session state ────────────────────────────────────────────────
    if "gemma_messages" not in st.session_state:
        st.session_state["gemma_messages"] = []

    # ── Handle suggestion chip clicks ─────────────────────────────────────
    if "bt_qa_pending" in st.session_state:
        pending = st.session_state.pop("bt_qa_pending")
        st.session_state["gemma_messages"].append(
            {"role": "user", "content": pending}
        )

    messages: List[Dict[str, str]] = st.session_state["gemma_messages"]

    # ── Suggestion chips — reflect the real Q&A use cases from new_features.txt ──
    # Two rows: first row = data questions that trigger a SQL query,
    # second row = general knowledge questions answered from the model's training.
    _SUGGESTIONS_ROW1 = [
        ("👥", "How many users have uploaded documents?"),
        ("⏳", "How many documents are waiting for approval?"),
        ("📄", "Show me details of all submitted documents"),
        ("🔗", "Show applicants with their submitted document types"),
    ]
    _SUGGESTIONS_ROW2 = [
        ("🏠", "What documents are needed for a home loan?"),
        ("🔒", "What is Background Verification (BGV)?"),
        ("🪪", "Tell me about Aadhaar and PAN verification in India"),
        ("📋", "What documents are required for BGV?"),
    ]

    if not messages:
        # Welcome hero — shown only when the chat is empty
        st.markdown(
            '<div class="bt-chat-hero">'
            '<div class="bt-chat-hero-icon">💬</div>'
            "<h2>How can I help you today?</h2>"
            "<p>Ask me anything about applicants, document approvals, fraud risk, "
            "identity checks — or general topics like KYC, BGV, and mortgage requirements.</p>"
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
        _do_reply(messages, system_prompt, model_name, base_url, context_limit)
        st.session_state["gemma_messages"] = messages
    else:
        st.session_state.pop("bt_qa_replied", None)

    # ── Chat input ────────────────────────────────────────────────────────
    if user_input := st.chat_input("Message BaseTruth Q&A…"):
        messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)

        # Single entry point: silently detects SQL/MinIO and streams only the
        # final human-readable answer so the user sees one clean response.
        _do_reply(messages, system_prompt, model_name, base_url, context_limit)
        st.session_state["gemma_messages"] = messages
