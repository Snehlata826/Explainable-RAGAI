"""
app.py
Entry point for the RAG Intelligence Streamlit UI.

Run with:
    streamlit run app.py
"""

import streamlit as st

# Must be the FIRST Streamlit call
st.set_page_config(
    page_title="RAG Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Local imports (after set_page_config)
from styles.theme import get_theme, inject_css
from components.sidebar import render_sidebar
from components.chat import (
    render_chat_header,
    render_empty_state,
    render_history,
    handle_user_input,
)



def _init_session() -> None:
    defaults = {
        "messages":       [],   # list of {role, content, error?}
        "dark_mode":      True,
        "uploaded_files": [],   # list of indexed file name strings
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def main() -> None:
    _init_session()

    t = get_theme()
    inject_css(t)

    render_sidebar()
    render_chat_header(t)

    if not st.session_state.messages:
        render_empty_state(t)
    else:
        render_history(t)

    if prompt := st.chat_input("Ask about your documents…"):
        handle_user_input(prompt, t)


if __name__ == "__main__":
    main()