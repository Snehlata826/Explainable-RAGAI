import streamlit as st
from api.client import query
from components.sources import render_sources


def render_chat():

    if not st.session_state.documents:

        st.info("Upload a document to start asking questions.")

        return

    # Display chat history

    for msg in st.session_state.messages:

        with st.chat_message(msg["role"]):

            st.markdown(msg["content"])

    question = st.chat_input(
        "Ask a question about your documents"
    )

    if question:

        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.spinner("Generating answer..."):

            resp = query(question)

        answer = resp["answer"]

        with st.chat_message("assistant"):

            st.markdown(answer)

            render_sources(resp)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )