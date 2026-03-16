import streamlit as st
from api.client import upload_file


def render_uploader():

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

                    res = upload_file(f)

                    st.session_state.documents.append(f.name)

                    st.success(f"{f.name} uploaded")