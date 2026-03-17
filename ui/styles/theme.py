"""
styles/theme.py
All theme variables and CSS injection for the RAG UI.
"""

import streamlit as st


# ── Colour tokens ─────────────────────────────────────────────────────────────

DARK = {
    "BG":          "#0b0b0b",
    "BG2":         "#111111",
    "BG3":         "#1a1a1a",
    "CARD":        "#161616",
    "BORDER":      "#1f1f1f",
    "ACCENT":      "#00ff88",
    "ACCENT2":     "#00cc6a",
    "TEXT":        "#e8e8e8",
    "TEXT_MUTED":  "#555555",
    "USER_BG":     "#0d2b1a",
    "USER_BORDER": "#00ff8830",
    "AI_BG":       "#111111",
    "AI_BORDER":   "#1f1f1f",
    "SHADOW":      "0 4px 24px rgba(0,255,136,0.06)",
    "GLOW":        "0 0 20px rgba(0,255,136,0.15)",
    "INPUT_BG":    "#0f0f0f",
    "SIDEBAR_BG":  "#0d0d0d",
    "ERROR_BG":    "#1a0000",
    "ERROR_BORDER":"#ff444440",
    "ERROR_TEXT":  "#ff6b6b",
}

LIGHT = {
    # Warm off-white — easy on the eyes, no harsh pure white
    "BG":          "#f7f6f3",   # warm parchment base
    "BG2":         "#f0ede8",   # slightly deeper warm surface
    "BG3":         "#e9e5df",   # input/uploader areas
    "CARD":        "#faf9f7",   # card surface
    "BORDER":      "#ddd9d2",   # warm gray border
    "ACCENT":      "#0d7a5f",   # deeper teal-green
    "ACCENT2":     "#0a6650",
    "TEXT":        "#0f0a04",   # warm near-black (not harsh #111)
    "TEXT_MUTED":  "#0d0701",   # warm muted gray
    "USER_BG":     "#edf7f2",   # soft mint bubble
    "USER_BORDER": "#b6dece",   # muted sage border
    "AI_BG":       "#faf9f7",   # warm white AI bubble
    "AI_BORDER":   "#ddd9d2",
    "SHADOW":      "0 4px 20px rgba(0,0,0,0.04)",
    "GLOW":        "0 0 16px rgba(13,122,95,0.07)",
    "INPUT_BG":    "#f7f6f3",
    "SIDEBAR_BG":  "#f0ede8",   # warm sidebar
    "ERROR_BG":    "#fdf2f2",
    "ERROR_BORDER":"#f5c6c6",
    "ERROR_TEXT":  "#b91c1c",
}


def get_theme() -> dict:
    """Return the active colour token dict based on session state."""
    return DARK if st.session_state.get("dark_mode", True) else LIGHT


def inject_css(t: dict) -> None:
    """Inject all global CSS into the Streamlit page."""
    st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600&family=Syne:wght@400;500;600;700;800&display=swap');

  html, body, [data-testid="stApp"] {{
    background: {t['BG']} !important;
    color: {t['TEXT']} !important;
    font-family: 'Syne', sans-serif !important;
  }}

  ::-webkit-scrollbar {{ width: 4px; }}
  ::-webkit-scrollbar-track {{ background: {t['BG']}; }}
  ::-webkit-scrollbar-thumb {{ background: {t['ACCENT']}30; border-radius: 2px; }}
  ::-webkit-scrollbar-thumb:hover {{ background: {t['ACCENT']}60; }}

  [data-testid="stSidebar"] {{
    background: {t['SIDEBAR_BG']} !important;
    border-right: 1px solid {t['BORDER']} !important;
  }}
  [data-testid="stSidebar"] > div {{
    background: {t['SIDEBAR_BG']} !important;
    padding: 1.5rem 1rem !important;
  }}

  .main .block-container {{
    padding: 0 !important;
    max-width: 100% !important;
  }}

  #MainMenu, footer, header {{ visibility: hidden; }}
  [data-testid="stToolbar"] {{ display: none; }}

  /* ── File uploader ── */
  [data-testid="stFileUploader"] {{
    background: {t['BG3']} !important;
    border: 1px dashed {t['ACCENT']}40 !important;
    border-radius: 12px !important;
    padding: 0.5rem !important;
  }}
  [data-testid="stFileUploader"] label {{
    color: {t['TEXT_MUTED']} !important;
    font-size: 0.8rem !important;
    font-family: 'JetBrains Mono', monospace !important;
  }}

  /* ── Buttons ── */
  .stButton > button {{
    background: {t['ACCENT']}15 !important;
    color: {t['ACCENT']} !important;
    border: 1px solid {t['ACCENT']}40 !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.08em !important;
    padding: 0.4rem 1rem !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
    text-transform: uppercase !important;
  }}
  .stButton > button:hover {{
    background: {t['ACCENT']}25 !important;
    border-color: {t['ACCENT']}80 !important;
    box-shadow: {t['GLOW']} !important;
    transform: translateY(-1px) !important;
  }}

  /* ── Chat input ── */
  [data-testid="stChatInput"] textarea {{
    background: {t['INPUT_BG']} !important;
    color: {t['TEXT']} !important;
    border: 1px solid {t['BORDER']} !important;
    border-radius: 12px !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.92rem !important;
    transition: border-color 0.2s ease !important;
  }}
  [data-testid="stChatInput"] textarea:focus {{
    border-color: {t['ACCENT']}60 !important;
    box-shadow: 0 0 0 2px {t['ACCENT']}12 !important;
  }}
  [data-testid="stChatInput"] {{
    background: {t['BG']} !important;
    border-top: 1px solid {t['BORDER']} !important;
    padding: 1rem 2rem !important;
  }}
  [data-testid="stChatInputSubmitButton"] > button {{
    background: {t['ACCENT']} !important;
    color: #000 !important;
    border-radius: 10px !important;
    border: none !important;
    width: auto !important;
  }}

  [data-testid="stChatMessage"] {{
    background: transparent !important;
    border: none !important;
    padding: 0.25rem 2rem !important;
  }}

  
  
  

  hr {{ border-color: {t['BORDER']} !important; margin: 1rem 0 !important; }}

  /* ── Shared component classes ── */
  .sidebar-label {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: {t['ACCENT']};
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
    margin-top: 1rem;
  }}

  .file-tag {{
    background: {t['ACCENT']}10;
    border: 1px solid {t['ACCENT']}30;
    border-radius: 6px;
    padding: 0.3rem 0.6rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: {t['ACCENT']};
    margin: 0.2rem 0;
    display: flex;
    align-items: center;
    gap: 6px;
    word-break: break-all;
  }}

  .success-toast {{
    background: {t['ACCENT']}15;
    border: 1px solid {t['ACCENT']}40;
    border-radius: 8px;
    padding: 0.5rem 0.8rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: {t['ACCENT']};
    margin-top: 0.5rem;
  }}

  .chat-header {{
    padding: 1.5rem 2rem 0.8rem 2rem;
    border-bottom: 1px solid {t['BORDER']};
    background: {t['BG']};
  }}
  .chat-title {{
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1.1rem;
    color: {t['TEXT']};
  }}
  .chat-status {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: {t['ACCENT']};
    display: flex;
    align-items: center;
    gap: 6px;
  }}
  .status-dot {{
    width: 6px; height: 6px;
    background: {t['ACCENT']};
    border-radius: 50%;
    display: inline-block;
    animation: pulse-dot 2s infinite;
  }}
  @keyframes pulse-dot {{
    0%,100% {{ opacity:1; transform:scale(1); }}
    50%      {{ opacity:.4; transform:scale(.8); }}
  }}

  .user-bubble {{
    background: {t['USER_BG']};
    border: 1px solid {t['USER_BORDER']};
    border-radius: 16px 16px 4px 16px;
    padding: 0.9rem 1.2rem;
    margin: 0.4rem 0 0.4rem 3rem;
    font-size: 0.92rem;
    line-height: 1.6;
    box-shadow: {t['SHADOW']};
    word-wrap: break-word;
    animation: slide-in-right 0.25s ease;
  }}
  @keyframes slide-in-right {{
    from {{ opacity:0; transform:translateX(10px); }}
    to   {{ opacity:1; transform:translateX(0); }}
  }}

  .ai-bubble {{
    background: {t['AI_BG']};
    border: 1px solid {t['AI_BORDER']};
    border-radius: 16px 16px 16px 4px;
    padding: 0.9rem 1.2rem 0.9rem 1.5rem;
    margin: 0.4rem 3rem 0.4rem 0;
    font-size: 0.92rem;
    line-height: 1.7;
    box-shadow: {t['SHADOW']};
    word-wrap: break-word;
    position: relative;
    overflow: hidden;
    animation: slide-in-left 0.3s ease;
  }}
  .ai-bubble::before {{
    content: '';
    position: absolute;
    left: 0; top: 0;
    width: 3px; height: 100%;
    background: linear-gradient(to bottom, {t['ACCENT']}, {t['ACCENT2']});
  }}
  @keyframes slide-in-left {{
    from {{ opacity:0; transform:translateX(-8px); }}
    to   {{ opacity:1; transform:translateX(0); }}
  }}

  .msg-meta {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: {t['TEXT_MUTED']};
    margin-bottom: 0.2rem;
    display: flex;
    align-items: center;
    gap: 6px;
  }}
  .msg-meta .accent {{ color: {t['ACCENT']}; }}

  /* ── Loader ── */
  .loader-wrap {{
    background: {t['AI_BG']};
    border: 1px solid {t['AI_BORDER']};
    border-radius: 16px 16px 16px 4px;
    padding: 1rem 1.2rem 1rem 1.5rem;
    margin: 0.4rem 3rem 0.4rem 0;
    position: relative;
    overflow: hidden;
  }}
  .loader-wrap::before {{
    content: '';
    position: absolute;
    left: 0; top: 0;
    width: 3px; height: 100%;
    background: linear-gradient(to bottom, {t['ACCENT']}, {t['ACCENT2']});
  }}
  .loader-stage {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    color: {t['ACCENT']};
    display: flex;
    align-items: center;
    gap: 10px;
    animation: loader-glow 1.5s ease-in-out infinite;
  }}
  @keyframes loader-glow {{
    0%,100% {{ opacity:1; }}
    50%      {{ opacity:.5; }}
  }}
  .loader-bar {{
    height: 2px;
    background: {t['BORDER']};
    border-radius: 1px;
    margin-top: 0.7rem;
    overflow: hidden;
  }}
  .loader-bar-fill {{
    height: 100%;
    background: linear-gradient(90deg, {t['ACCENT']}00, {t['ACCENT']}, {t['ACCENT']}00);
    animation: bar-sweep 1.8s ease-in-out infinite;
  }}
  @keyframes bar-sweep {{
    0%   {{ width:0%;  margin-left:0%; }}
    50%  {{ width:60%; margin-left:20%; }}
    100% {{ width:0%;  margin-left:100%; }}
  }}

  /* ── Typing cursor ── */
  .typing-cursor {{
    display: inline-block;
    width: 2px; height: 1em;
    background: {t['ACCENT']};
    margin-left: 2px;
    animation: blink 0.8s step-end infinite;
    vertical-align: text-bottom;
  }}
  @keyframes blink {{
    0%,100% {{ opacity:1; }}
    50%      {{ opacity:0; }}
  }}

  /* ── Empty state ── */
  .empty-state {{
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 4rem 2rem;
    text-align: center;
  }}
  .empty-icon {{
    font-size: 3rem;
    margin-bottom: 1rem;
    animation: float 3s ease-in-out infinite;
  }}
  @keyframes float {{
    0%,100% {{ transform:translateY(0); }}
    50%      {{ transform:translateY(-8px); }}
  }}
  .empty-title {{
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1.4rem;
    color: {t['TEXT']};
    margin-bottom: 0.5rem;
  }}
  .empty-sub {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: {t['TEXT_MUTED']};
    line-height: 1.7;
    max-width: 340px;
  }}

  /* ── Error bubble ── */
  .error-bubble {{
    background: {t['ERROR_BG']};
    border: 1px solid {t['ERROR_BORDER']};
    border-radius: 12px;
    padding: 0.8rem 1.2rem;
    margin: 0.4rem 3rem 0.4rem 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: {t['ERROR_TEXT']};
  }}

  /* ── Streamlit file uploader overrides ── */
  [data-testid="stFileUploaderFile"] {{
    background: {t['BG3']} !important;
    border: 1px solid {t['BORDER']} !important;
    border-radius: 8px !important;
    padding: 0.4rem 0.6rem !important;
  }}
  [data-testid="stFileUploaderFileName"] {{
    color: {t['TEXT']} !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
  }}
  [data-testid="stFileUploaderFileIcon"] svg {{
    color: {t['TEXT_MUTED']} !important;
    fill: {t['TEXT_MUTED']} !important;
  }}
  /* Hide the red/colored icon dot */
  [data-testid="stFileUploaderFile"] > div:first-child {{
    display: none !important;
  }}
  [data-testid="stBaseButton-secondary"] {{
    background: {t['BG3']} !important;
    color: {t['TEXT']} !important;
    border: 1px solid {t['BORDER']} !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important;
  }}

  /* ── Responsive ── */
  @media (max-width: 768px) {{
    [data-testid="stChatMessage"] {{ padding: 0.25rem 0.5rem !important; }}
    .user-bubble, .ai-bubble, .loader-wrap, .error-bubble {{
      margin-left: 0 !important; margin-right: 0 !important;
    }}
    .chat-header {{ padding: 1rem !important; }}
    [data-testid="stChatInput"] {{ padding: 0.8rem !important; }}
  }}
</style>
""", unsafe_allow_html=True)