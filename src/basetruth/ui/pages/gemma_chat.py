"""Gemma4 Chat page — local LLM chat via Ollama."""
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
from basetruth.ui.components import _page_title

_DEFAULT_MODEL = DEFAULT_OLLAMA_MODEL

_SYSTEM_PROMPT = (
    "You are a helpful AI assistant powered by Google's Gemma4 model running locally "
    "via Ollama. You are embedded in the BaseTruth document fraud detection platform. "
    "Answer clearly and concisely."
)
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
                f"⚠️ Could not connect to Ollama at {base_url}. "
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


def _page_gemma_chat() -> None:
    st.markdown(_page_title("🤖", "Gemma4 Chat"), unsafe_allow_html=True)

    base_url, available_models, attempted_urls = probe_ollama()

    if not base_url:
        attempted_text = "\n".join(f"- {url}" for url in attempted_urls)
        st.error(
            "Ollama is not reachable from this UI runtime.\n\n"
            "Attempted endpoints:\n"
            f"{attempted_text}\n\n"
            "If the UI is running in Docker, Ollama must be reachable from the container.",
            icon="🔴",
        )
        if st.button("🔄 Retry connection", use_container_width=True, type="primary"):
            st.rerun()
        st.stop()

    # ── Sidebar-style settings in an expander ────────────────────────────────
    with st.expander("⚙️ Model settings", expanded=False):
        model_name = st.selectbox(
            "Ollama model",
            options=available_models,
            index=0,
            help="Select which local Ollama model to chat with.",
        )
        temperature_note = st.empty()
        temperature_note.caption(
            "Temperature is controlled by Ollama's default for the selected model. "
            "Use `ollama run <model>` in a terminal to change defaults."
        )
        st.caption(f"Connected endpoint: {base_url}")
        if st.button("🗑️ Clear conversation", use_container_width=True):
            st.session_state["gemma_messages"] = []
            st.rerun()

    st.success(f"Ollama connected — using **{model_name}**", icon="🟢")

    # ── Init session state ────────────────────────────────────────────────────
    if "gemma_messages" not in st.session_state:
        st.session_state["gemma_messages"] = []

    messages: List[Dict[str, str]] = st.session_state["gemma_messages"]

    # ── Render chat history ───────────────────────────────────────────────────
    for msg in messages:
        role = msg["role"]
        with st.chat_message(role, avatar="🤖" if role == "assistant" else "🧑"):
            st.markdown(msg["content"])

    # ── Chat input ────────────────────────────────────────────────────────────
    if user_input := st.chat_input("Ask Gemma4 anything…"):
        messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(user_input)

        # Build full message list including system prompt
        api_messages = [{"role": "system", "content": _SYSTEM_PROMPT}] + messages

        with st.chat_message("assistant", avatar="🤖"):
            response = _stream_chat(api_messages, model=model_name, base_url=base_url)

        messages.append({"role": "assistant", "content": response})
        st.session_state["gemma_messages"] = messages
