import streamlit as st
from api.client import upload_file, reset_index


def render_uploader():

    st.subheader("Upload Documents")

    files = st.file_uploader(
        "Upload PDF or text files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )

    # Track reset state
    if "db_reset_done" not in st.session_state:
        st.session_state.db_reset_done = False

    if files:

        # 🔥 Reset ONLY ONCE per upload batch
        if not st.session_state.db_reset_done:
            with st.spinner("Clearing previous knowledge base..."):
                res = reset_index()

            if res["success"]:
                st.success("Vector DB cleared ✅")
                st.session_state.db_reset_done = True
                st.session_state.documents = []  # clear old docs
            else:
                st.error(res["message"])
                return  # stop if reset fails

        # 🚀 Upload files
        for f in files:

            if f.name not in st.session_state.documents:

                with st.spinner(f"Indexing {f.name}..."):

                    res = upload_file(f)

                if res["success"]:
                    st.session_state.documents.append(f.name)
                    st.success(f"{f.name} uploaded ✅")
                else:
                    st.error(res["message"])