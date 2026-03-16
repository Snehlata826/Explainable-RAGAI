import streamlit as st
from api.client import upload_file


def render_sidebar():

    with st.sidebar:

        st.markdown("### RAG Copilot")
        st.caption("AI assistant for document understanding")

        st.divider()

        # Upload section
        st.subheader("Upload Documents")

        files = st.file_uploader(
            "Upload PDF or text files",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
        )

        if files:

            for f in files:

                if f.name not in st.session_state.documents:

                    with st.spinner(f"Indexing {f.name}"):

                        upload_file(f)

                        st.session_state.documents.append(f.name)

                        st.success(f"{f.name} uploaded")

        st.divider()

        # Document list
        st.subheader("Documents")

        docs = st.session_state.get("documents", [])

        if docs:

            for d in docs:
                st.write("•", d)

        else:

            st.caption("No documents uploaded")

        st.divider()

        # Reset conversation
        if st.button("Reset conversation"):

            st.session_state.messages = []

            st.rerun()