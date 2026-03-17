"""
components/sidebar.py
Renders the left sidebar: logo, theme toggle, file uploader, clear chat.
"""

import streamlit as st
from api.client import upload_file
from styles.theme import get_theme


def render_sidebar() -> None:
    """Render the full sidebar. Mutates st.session_state as needed."""
    t = get_theme()

    with st.sidebar:
        # ── Logo ─────────────────────────────────────────────────────────────
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:10px;
                    padding-bottom:1.2rem;border-bottom:1px solid {t['BORDER']};
                    margin-bottom:0.4rem;">
          <div style="width:36px;height:36px;background:{t['ACCENT']};
                      border-radius:10px;display:flex;align-items:center;
                      justify-content:center;font-size:18px;flex-shrink:0;">⚡</div>
          <div>
            <div style="font-family:'Syne',sans-serif;font-weight:800;
                        font-size:1rem;color:{t['TEXT']};">RAG Intelligence</div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.62rem;
                        color:{t['ACCENT']};letter-spacing:0.08em;">
              v1.0 · Neural Interface
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Theme toggle ──────────────────────────────────────────────────────
        st.markdown('<div class="sidebar-label">⚙ Interface</div>', unsafe_allow_html=True)
        new_dark = st.toggle("Dark Mode", value=st.session_state.dark_mode, key="theme_toggle")
        if new_dark != st.session_state.dark_mode:
            st.session_state.dark_mode = new_dark
            st.rerun()

        st.markdown("---")

        # ── File uploader ─────────────────────────────────────────────────────
        st.markdown('<div class="sidebar-label">📂 Documents</div>', unsafe_allow_html=True)

        uploaded_files = st.file_uploader(
            "Upload PDF or TXT",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key="file_uploader",
        )

        if uploaded_files:
            for f in uploaded_files:
                if f.name not in st.session_state.uploaded_files:
                    with st.spinner(""):
                        result = upload_file(f)
                    if result["success"]:
                        st.session_state.uploaded_files.append(f.name)
                        st.markdown(
                            f'<div class="success-toast">✓ {f.name} indexed</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f'<div class="error-bubble">✗ {result["message"]}</div>',
                            unsafe_allow_html=True,
                        )

        # Indexed file tags
        if st.session_state.uploaded_files:
            st.markdown('<div style="margin-top:0.6rem;">', unsafe_allow_html=True)
            for fname in st.session_state.uploaded_files:
                icon = "📄" if fname.lower().endswith(".txt") else "📕"
                st.markdown(
                    f'<div class="file-tag">{icon} {fname}</div>',
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")

        # ── Clear chat ────────────────────────────────────────────────────────
        st.markdown('<div class="sidebar-label">🗑 Actions</div>', unsafe_allow_html=True)
        if st.button("Clear Chat History", key="clear_btn"):
            st.session_state.messages = []
            st.rerun()

        # ── Stats footer ──────────────────────────────────────────────────────
        doc_count = len(st.session_state.uploaded_files)
        st.markdown(f"""
        <div style="margin-top:2rem;font-family:'JetBrains Mono',monospace;
                    font-size:0.65rem;color:{t['TEXT_MUTED']};line-height:1.9;">
          Backend &rarr; <span style="color:{t['ACCENT']};">localhost:8000</span><br>
          Model &rarr; <span style="color:{t['ACCENT']};">RAG Pipeline</span><br>
          Docs &rarr; <span style="color:{t['ACCENT']};">{doc_count} indexed</span>
        </div>
        """, unsafe_allow_html=True)