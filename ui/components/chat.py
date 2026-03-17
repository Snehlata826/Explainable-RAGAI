"""
components/chat.py
Renders the chat header, message history, animated loader,
typing effect, and processes new user questions.
"""

import time
import threading
import streamlit as st
from api.client import query
from styles.theme import get_theme


# ── Loader config ─────────────────────────────────────────────────────────────

LOADER_STAGES = [
    ("🧠", "Thinking"),
    ("🔍", "Searching documents"),
    ("📄", "Reading context"),
    ("🔗", "Connecting knowledge"),
    ("⚡", "Generating answer"),
]

SPIN_FRAMES = ["◐", "◓", "◑", "◒"]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _render_loader_frame(placeholder, stage_idx: int, frame_idx: int, t: dict) -> None:
    """Write one animation frame into the given st.empty() placeholder."""
    icon, label = LOADER_STAGES[stage_idx]
    spin = SPIN_FRAMES[frame_idx % 4]
    dots = "." * ((frame_idx % 3) + 1)
    placeholder.markdown(f"""
    <div class="msg-meta">
      <span class="accent">⚡ RAG</span><span>·</span><span>Processing</span>
    </div>
    <div class="loader-wrap">
      <div class="loader-stage">
        <span style="font-size:1rem;">{icon}</span>
        <span>{spin}</span>
        <span>{label}{dots}</span>
      </div>
      <div class="loader-bar"><div class="loader-bar-fill"></div></div>
    </div>
    """, unsafe_allow_html=True)


def _render_user_bubble(text: str, t: dict) -> None:
    st.markdown(f"""
    <div class="msg-meta" style="justify-content:flex-end; padding:0 2rem;">
      <span>You</span>
    </div>
    <div class="user-bubble">{text}</div>
    """, unsafe_allow_html=True)


def _render_ai_bubble(text: str) -> None:
    st.markdown(f"""
    <div class="msg-meta" style="padding:0 2rem;">
      <span class="accent">⚡ RAG</span><span>·</span><span>Assistant</span>
    </div>
    <div class="ai-bubble">{text}</div>
    """, unsafe_allow_html=True)


def _render_error_bubble(message: str) -> None:
    st.markdown(f"""
    <div class="msg-meta" style="padding:0 2rem;">
      <span style="color:#ff6b6b;">✗ Error</span>
    </div>
    <div class="error-bubble">⚠ {message}</div>
    """, unsafe_allow_html=True)


def _typing_effect(placeholder, text: str, t: dict) -> None:
    """Render text character-by-character into placeholder with a blinking cursor."""
    st.markdown(f"""
    <div class="msg-meta" style="padding:0 2rem;">
      <span class="accent">⚡ RAG</span><span>·</span><span>Assistant</span>
    </div>
    """, unsafe_allow_html=True)

    chunk = 3
    for i in range(0, len(text) + chunk, chunk):
        shown = text[:i]
        cursor = '<span class="typing-cursor"></span>' if i <= len(text) else ""
        placeholder.markdown(
            f'<div class="ai-bubble">{shown}{cursor}</div>',
            unsafe_allow_html=True,
        )
        time.sleep(0.012)

    # Final render without cursor
    placeholder.markdown(f'<div class="ai-bubble">{text}</div>', unsafe_allow_html=True)


# ── Public API ────────────────────────────────────────────────────────────────

def render_chat_header(t: dict) -> None:
    """Render the sticky chat header with title and status indicator."""
    st.markdown(f"""
    <div class="chat-header">
      <div class="chat-title">Neural Document Assistant</div>
      <div class="chat-status">
        <span class="status-dot"></span>
        <span>Ready · Ask anything about your documents</span>
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_empty_state(t: dict) -> None:
    """Render the centred empty-state prompt when no messages exist."""
    st.markdown(f"""
    <div class="empty-state">
      <div class="empty-icon">🧠</div>
      <div class="empty-title">Intelligence Ready</div>
      <div class="empty-sub">
        Upload your documents in the sidebar,<br>
        then ask questions below.<br><br>
        <span style="color:{t['ACCENT']};">⚡ Powered by RAG</span>
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_history(t: dict) -> None:
    """Replay all stored messages from session state."""
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            _render_user_bubble(msg["content"], t)
        elif msg["role"] == "assistant":
            if msg.get("error"):
                _render_error_bubble(msg["content"])
            else:
                _render_ai_bubble(msg["content"])


def handle_user_input(prompt: str, t: dict) -> None:
    """
    Called when the user submits a new question.
    1. Shows user bubble immediately.
    2. Runs animated multi-stage loader while fetching answer in background thread.
    3. Replaces loader with typed answer (or error).
    4. Persists both messages in session state.
    """
    # ── 1. Show user message ──────────────────────────────────────────────────
    _render_user_bubble(prompt, t)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # ── 2. Start background fetch ─────────────────────────────────────────────
    result_holder: dict = {"answer": None, "success": False, "done": False}

    def fetch():
        res = query(prompt)
        result_holder["success"] = res["success"]
        result_holder["answer"]  = res["answer"]
        result_holder["done"]    = True

    thread = threading.Thread(target=fetch, daemon=True)
    thread.start()

    # ── 3. Animate loader until done ──────────────────────────────────────────
    loader_placeholder = st.empty()
    frame, stage_tick = 0, 0
    frames_per_stage = 8

    while not result_holder["done"]:
        stage_idx = min(stage_tick // frames_per_stage, len(LOADER_STAGES) - 1)
        _render_loader_frame(loader_placeholder, stage_idx, frame, t)
        time.sleep(0.18)
        frame      += 1
        stage_tick += 1

    # Final loader frame at last stage, then clear
    _render_loader_frame(loader_placeholder, len(LOADER_STAGES) - 1, frame, t)
    thread.join(timeout=2)
    time.sleep(0.25)
    loader_placeholder.empty()

    # ── 4. Display answer ─────────────────────────────────────────────────────
    answer  = result_holder["answer"] or "No response received."
    success = result_holder["success"]

    if not success:
        _render_error_bubble(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer, "error": True})
    else:
        answer_placeholder = st.empty()
        _typing_effect(answer_placeholder, answer, t)
        st.session_state.messages.append({"role": "assistant", "content": answer, "error": False})