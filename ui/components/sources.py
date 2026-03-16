import streamlit as st


def render_sources(resp):

    sources = resp.get("sources", [])

    if sources:

        with st.expander("Sources"):

            for s in sources:

                st.markdown(f"**{s['document']}**")

                st.markdown(f"> {s['snippet']}")

                st.divider()