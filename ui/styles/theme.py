import streamlit as st


def apply_theme(mode="dark"):

    if mode == "dark":

        st.markdown("""
        <style>

        /* Main App Background */

        .stApp {
            background-color: #0B0F0C;
            color: #E5E7EB;
        }

        /* Sidebar */

        section[data-testid="stSidebar"] {
            background-color: #111715;
        }

        /* Headings */

        h1, h2, h3 {
            color: #22C55E;
        }

        /* Chat message bubble */

        .stChatMessage {
            background-color: #1A2220;
            border-radius: 10px;
            padding: 12px;
        }

        /* Chat input */

        [data-testid="stChatInput"] {
            border-radius: 10px;
        }

        /* Buttons */

        .stButton>button {
            background-color: #22C55E;
            color: black;
            border-radius: 6px;
            border: none;
        }

        /* File uploader container */

        .stFileUploader {
            background-color: #1A2220;
            border-radius: 10px;
            padding: 10px;
        }

        /* File uploader drop zone */

        [data-testid="stFileUploaderDropzone"] {
            border: 2px dashed #22C55E;
            border-radius: 10px;
        }

        </style>
        """, unsafe_allow_html=True)

    else:

        st.markdown("""
        <style>

        /* Main App Background */

        .stApp {
            background-color: #F0F5F2;
            color: #111827;
        }

        /* Sidebar */

        section[data-testid="stSidebar"] {
            background-color: #E8F1EC;
            border-right: 1px solid #D1D5DB;
        }

        /* Headings */

        h1, h2, h3 {
            color: #16A34A;
        }

        /* Chat message bubble */

        .stChatMessage {
            background-color: #FFFFFF;
            border-radius: 10px;
            padding: 12px;
            border: 1px solid #D1D5DB;
        }

        /* Chat input */

        [data-testid="stChatInput"] {
            border-radius: 10px;
            border: 1px solid #D1D5DB;
        }

        /* Buttons */

        .stButton>button {
            background-color: #16A34A;
            color: white;
            border-radius: 6px;
            border: none;
        }

        /* File uploader */

        .stFileUploader {
            background-color: #FFFFFF;
            border-radius: 10px;
            padding: 12px;
        }

        /* Upload drop zone */

        [data-testid="stFileUploaderDropzone"] {
            border: 2px dashed #16A34A;
            border-radius: 10px;
        }

        </style>
        """, unsafe_allow_html=True)