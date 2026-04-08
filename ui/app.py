import streamlit as st
import time
import threading
import os
import requests
from requests.exceptions import ConnectionError, Timeout

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="RAG Intelligence v2",
    page_icon="⚡",
    layout="wide",
)

def get_backend_url():
    if os.getenv("SPACE_ID"):
        return "http://127.0.0.1:8005"   # ✅ INTERNAL HF FIX
    return "http://localhost:8005"

API_BASE = get_backend_url()

# ─────────────────────────────────────────────
# SESSION INIT
# ─────────────────────────────────────────────

def _init_session():
    defaults = {
        "messages": [],
        "uploaded_files": [],
        "debug_mode": False,
        "db_reset_done": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# ─────────────────────────────────────────────
# API
# ─────────────────────────────────────────────

def api_query(question: str):
    try:
        r = requests.post(
            f"{API_BASE}/query",
            json={"question": question},
            timeout=120,
        )
        if r.status_code == 200:
            return {"success": True, **r.json()}
        return {"success": False, "answer": r.text}
    except Exception as e:
        return {"success": False, "answer": str(e)}

def api_reset():
    try:
        r = requests.delete(f"{API_BASE}/reset", timeout=20)
        return {"success": r.status_code == 200}
    except:
        return {"success": False}

# ─────────────────────────────────────────────
# SIDEBAR (FIXED)
# ─────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.title("⚡ RAG Intelligence")

        st.markdown("### Upload Documents")

        uploaded_files = st.file_uploader(
            "Upload",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True
        )

        # 🔥 FIXED UPLOAD LOGIC (NO 403)
        if uploaded_files:

            if not st.session_state.db_reset_done:
                with st.spinner("Resetting index..."):
                    res = api_reset()
                if res["success"]:
                    st.session_state.db_reset_done = True
                    st.session_state.uploaded_files = []

            for f in uploaded_files:
                if f.name not in st.session_state.uploaded_files:

                    with st.spinner(f"Uploading {f.name}..."):
                        try:
                            file_bytes = f.read()   # ✅ KEY FIX

                            response = requests.post(
                                f"{API_BASE}/upload",
                                files={"file": (f.name, file_bytes)},
                                timeout=300
                            )

                            if response.status_code == 200:
                                st.session_state.uploaded_files.append(f.name)
                                st.success(f"{f.name} indexed ✅")
                            else:
                                st.error(response.text)

                        except Exception as e:
                            st.error(str(e))

        st.markdown("---")

        # 🔥 FIXED BACKEND DISPLAY
        st.markdown(f"**Backend → {API_BASE}**")

        st.markdown("---")

        if st.button("Reset Index"):
            api_reset()
            st.session_state.uploaded_files = []
            st.session_state.db_reset_done = False
            st.success("Index reset")

# ─────────────────────────────────────────────
# CHAT
# ─────────────────────────────────────────────

def handle_query(prompt):
    with st.spinner("Thinking..."):
        res = api_query(prompt)

    if not res["success"]:
        st.error(res["answer"])
        return

    answer = res.get("answer", "")
    st.chat_message("assistant").write(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    _init_session()
    render_sidebar()

    st.title("🧠 Neural Document Assistant")

    if not st.session_state.uploaded_files:
        st.info("Upload a document to begin")

    # chat history
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Ask something..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        handle_query(prompt)

if __name__ == "__main__":
    main()