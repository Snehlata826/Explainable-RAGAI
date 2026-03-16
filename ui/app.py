import streamlit as st

from styles.theme import apply_theme
from components.sidebar import render_sidebar
from components.chat import render_chat


st.set_page_config(
    page_title="RAG Copilot",
    page_icon="📄",
    layout="wide",
)

# -------------------------
# Initialize session state
# -------------------------

if "documents" not in st.session_state:
    st.session_state.documents = []

if "messages" not in st.session_state:
    st.session_state.messages = []

if "theme" not in st.session_state:
    st.session_state.theme = "dark"


# -------------------------
# Theme toggle (top right)
# -------------------------

col1, col2 = st.columns([9, 1])

with col2:
    dark_mode = st.toggle("🌙", value=st.session_state.theme == "dark")

if dark_mode:
    st.session_state.theme = "dark"
else:
    st.session_state.theme = "light"


# -------------------------
# Apply theme
# -------------------------

apply_theme(st.session_state.theme)


# -------------------------
# Sidebar
# -------------------------

render_sidebar()


# -------------------------
# Main header
# -------------------------

st.markdown("### RAG Copilot")
st.caption("Document Intelligence Assistant")

st.markdown(
"""
Ask questions about your uploaded documents.

Responses are generated **only from your document content**.
"""
)

st.divider()


# -------------------------
# Chat section
# -------------------------

render_chat()