"""BaseTruth Q&A page — local LLM chat via Ollama."""
from __future__ import annotations

import json
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
    execute_safe_query,
    query_minio_objects,
)
from basetruth.ui.components import _page_title
from basetruth.db import db_available
from basetruth.store import minio_available

_DEFAULT_MODEL = DEFAULT_OLLAMA_MODEL

def _build_system_prompt() -> str:
    prompt = (
        "You are BaseTruth Q&A, an intelligent AI assistant powered by a local LLM "
        "running via Ollama. You are embedded in the BaseTruth document fraud detection "
        "platform. Answer clearly, concisely, and helpfully."
    )
    if db_available():
        prompt += f"\n\nYou have access to a PostgreSQL database with these tables:\n{get_schema_summary()}\n"
        prompt += "When the user asks about data, generate a SQL query wrapped in ```sql blocks.\n"
        prompt += "Use ONLY SELECT statements. The system will execute the query and show you the results, then you will summarize them for the user in natural language."
        
    minio_summary = get_minio_summary()
    if minio_summary:
        prompt += f"\n\nYou also have access to MinIO storage:\n{minio_summary}\n"
        prompt += "To list stored files, use ```minio blocks with commands like:\n"
        prompt += "  LIST ALL\n"
        prompt += "  LIST ENTITY BT-000001\n"
        prompt += "Do not generate code blocks for MinIO unless specifically asked to list files."
        
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
[data-testid="stChatInput"] {
    border-radius: 999px !important;
    padding: 0 !important;
}
[data-testid="stChatInput"] textarea {
    border-radius: 999px !important;
    padding: 0.75rem 1.25rem !important;
    font-size: 0.92rem !important;
    border: 1.5px solid rgba(99, 102, 241, 0.20) !important;
    transition: all 0.2s ease !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.10) !important;
}
[data-testid="stChatInput"] button {
    border-radius: 50% !important;
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    transition: all 0.2s ease !important;
}
[data-testid="stChatInput"] button:hover {
    transform: scale(1.05);
    box-shadow: 0 4px 16px rgba(99, 102, 241, 0.30) !important;
}
</style>
"""


def _stream_chat(messages: List[Dict[str, str]], model: str, base_url: str) -> str:
    """Send messages to Ollama and collect the full response."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
    }
    full_response = ""
    placeholder = st.empty()
    chat_endpoint = f"{base_url}/api/chat"
    with placeholder.container():
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
                    placeholder.markdown(full_response + "▌")
                    if data.get("done"):
                        break
                placeholder.markdown(full_response)
        except requests.exceptions.ConnectionError:
            full_response = (
                f"⚠️ Could not connect to Ollama at `{base_url}`. "
                "Make sure Ollama is running."
            )
            st.error(full_response)
        except requests.exceptions.Timeout:
            full_response = "⚠️ Request timed out. The model may be loading — please try again."
            st.error(full_response)
        except requests.RequestException as exc:
            full_response = f"⚠️ Error: {exc}"
            st.error(full_response)
    return full_response

def _process_llm_response(response: str, messages: List[Dict[str, str]], system_prompt: str, model_name: str, base_url: str) -> None:
    """Post-process the response, check for SQL/MinIO commands, execute them silently, and get a natural language summary."""
    import re
    # Check for SQL or MinIO blocks
    sql_match = re.search(r"```sql\s*(.*?)\s*```", response, re.IGNORECASE | re.DOTALL)
    minio_match = re.search(r"```minio\s*(.*?)\s*```", response, re.IGNORECASE | re.DOTALL)
    
    if sql_match:
        sql_query = sql_match.group(1).strip()
        result_table = execute_safe_query(sql_query)
        followup = f"System Query Result (DO NOT SHOW RAW TABLE TO USER, SUMMARIZE IT):\n\n{result_table}"
        messages.append({"role": "user", "content": followup})
        
        with st.chat_message("assistant", avatar="💬"):
            api_messages = [{"role": "system", "content": system_prompt}] + messages
            followup_response = _stream_chat(api_messages, model=model_name, base_url=base_url)
        messages.append({"role": "assistant", "content": followup_response})
        
    elif minio_match:
        minio_cmd = minio_match.group(1).strip()
        result_table = query_minio_objects(minio_cmd)
        followup = f"System Storage Result (DO NOT SHOW RAW TABLE TO USER, SUMMARIZE IT):\n\n{result_table}"
        messages.append({"role": "user", "content": followup})
        
        with st.chat_message("assistant", avatar="💬"):
            api_messages = [{"role": "system", "content": system_prompt}] + messages
            followup_response = _stream_chat(api_messages, model=model_name, base_url=base_url)
        messages.append({"role": "assistant", "content": followup_response})


def _page_gemma_chat() -> None:
    # ── Inject custom CSS ─────────────────────────────────────────────────
    st.markdown(_CHAT_CSS, unsafe_allow_html=True)

    # ── Page title ────────────────────────────────────────────────────────
    st.markdown(_page_title("💬", "BaseTruth Q&A"), unsafe_allow_html=True)

    # ── Probe Ollama ──────────────────────────────────────────────────────
    base_url, available_models, attempted_urls = probe_ollama()

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
            st.rerun()
        st.stop()

    # ── Use first available model ─────────────────────────────────────────
    model_name = available_models[0] if available_models else _DEFAULT_MODEL

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

    # ── Welcome screen when no messages yet ───────────────────────────────
    _SUGGESTIONS = [
        ("📊", "How many documents have been scanned?"),
        ("👤", "Show all registered applicants"),
        ("🔍", "Which entities have high-risk scans?"),
        ("📁", "What files are stored in the system?"),
    ]

    if not messages:
        st.markdown(
            '<div class="bt-chat-hero">'
            '<div class="bt-chat-hero-icon">💬</div>'
            "<h2>How can I help you today?</h2>"
            "<p>Ask me anything about document verification, fraud detection, "
            "identity checks, or any topic you need assistance with.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        # Render clickable suggestion chips as real buttons
        chip_cols = st.columns(len(_SUGGESTIONS))
        for idx, (icon, label) in enumerate(_SUGGESTIONS):
            with chip_cols[idx]:
                if st.button(
                    f"{icon}  {label}",
                    key=f"bt_chip_{idx}",
                    use_container_width=True,
                ):
                    st.session_state["bt_qa_pending"] = f"{label}"
                    st.rerun()

    # ── Render chat history ───────────────────────────────────────────────
    for msg in messages:
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
        api_messages = [{"role": "system", "content": system_prompt}] + messages
        with st.chat_message("assistant", avatar="💬"):
            response = _stream_chat(api_messages, model=model_name, base_url=base_url)
        messages.append({"role": "assistant", "content": response})
        st.session_state["gemma_messages"] = messages
        _process_llm_response(response, messages, system_prompt, model_name, base_url)
        st.session_state["gemma_messages"] = messages
    else:
        st.session_state.pop("bt_qa_replied", None)

    # ── Chat input ────────────────────────────────────────────────────────
    if user_input := st.chat_input("Message BaseTruth Q&A…"):
        messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)

        # Build full message list including system prompt
        api_messages = [{"role": "system", "content": system_prompt}] + messages

        with st.chat_message("assistant", avatar="💬"):
            response = _stream_chat(api_messages, model=model_name, base_url=base_url)

        messages.append({"role": "assistant", "content": response})
        st.session_state["gemma_messages"] = messages
        _process_llm_response(response, messages, system_prompt, model_name, base_url)
        st.session_state["gemma_messages"] = messages
