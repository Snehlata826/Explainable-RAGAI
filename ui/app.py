"""
app.py — RAG Intelligence UI v2.0
Dark mode: deep black + electric green.
Features: animated loader, feedback loop, evaluation metrics display, multi-doc view.
"""

import streamlit as st

st.set_page_config(
    page_title="RAG Intelligence v2",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

import time
import threading
import requests
from requests.exceptions import ConnectionError, Timeout

API_BASE = "http://localhost:8005"


# ══════════════════════════════════════════════════════════════
# THEME & CSS
# ══════════════════════════════════════════════════════════════

def inject_css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Space+Grotesk:wght@300;400;500;600;700&display=swap');

:root {
  --bg:          #040404;
  --bg2:         #080808;
  --bg3:         #0d0d0d;
  --card:        #0a0a0a;
  --border:      #141414;
  --border2:     #1c1c1c;
  --accent:      #00ff88;
  --accent2:     #00cc6a;
  --accent3:     #00ff8820;
  --text:        #e2e8e4;
  --text-muted:  #404844;
  --user-bg:     #061410;
  --user-border: #00ff8825;
  --ai-bg:       #060606;
  --ai-border:   #161616;
  --error-bg:    #140303;
  --error-border:#ff444430;
  --error-text:  #ff6b6b;
  --warn-bg:     #110d00;
  --warn-text:   #ffa040;
  --shadow:      0 4px 32px rgba(0,255,136,0.05);
  --glow:        0 0 24px rgba(0,255,136,0.2);
  --glow-sm:     0 0 12px rgba(0,255,136,0.12);
}

html, body, [data-testid="stApp"] {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: 'Space Grotesk', sans-serif !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 3px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--accent)25; border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent)50; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: var(--bg2) !important;
  border-right: 1px solid var(--border2) !important;
}
[data-testid="stSidebar"] > div {
  background: var(--bg2) !important;
  padding: 1.4rem 1rem !important;
}

/* ── Main container ── */
.main .block-container {
  padding: 0 !important;
  max-width: 100% !important;
}

#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
  background: var(--bg3) !important;
  border: 1px dashed var(--accent)35 !important;
  border-radius: 10px !important;
  padding: 0.4rem !important;
  transition: border-color 0.3s ease !important;
}
[data-testid="stFileUploader"]:hover {
  border-color: var(--accent)60 !important;
}

/* ── Buttons ── */
.stButton > button {
  background: transparent !important;
  color: var(--accent) !important;
  border: 1px solid var(--accent)35 !important;
  border-radius: 8px !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.72rem !important;
  font-weight: 500 !important;
  letter-spacing: 0.12em !important;
  padding: 0.4rem 1rem !important;
  transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
  width: 100% !important;
  text-transform: uppercase !important;
}
.stButton > button:hover {
  background: var(--accent)10 !important;
  border-color: var(--accent)70 !important;
  box-shadow: var(--glow-sm) !important;
  transform: translateY(-1px) !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] textarea {
  background: var(--bg3) !important;
  color: var(--text) !important;
  border: 1px solid var(--border2) !important;
  border-radius: 12px !important;
  font-family: 'Space Grotesk', sans-serif !important;
  font-size: 0.92rem !important;
}
[data-testid="stChatInput"] textarea:focus {
  border-color: var(--accent)50 !important;
  box-shadow: 0 0 0 2px var(--accent)10 !important;
}
[data-testid="stChatInput"] {
  background: var(--bg) !important;
  border-top: 1px solid var(--border) !important;
  padding: 1rem 2rem !important;
}
[data-testid="stChatInputSubmitButton"] > button {
  background: var(--accent) !important;
  color: #000 !important;
  border-radius: 10px !important;
  border: none !important;
  width: auto !important;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
  background: transparent !important;
  border: none !important;
  padding: 0.2rem 2rem !important;
}

/* ── Toggle ── */
[data-testid="stToggle"] span {
  background-color: var(--accent)40 !important;
}

hr { border-color: var(--border2) !important; margin: 0.8rem 0 !important; }

/* ══════════════════════════════════════════════════════════════
   SHARED COMPONENT CLASSES
   ══════════════════════════════════════════════════════════════ */

.sidebar-label {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.65rem;
  color: var(--accent);
  letter-spacing: 0.18em;
  text-transform: uppercase;
  margin-bottom: 0.5rem;
  margin-top: 1rem;
}

.file-tag {
  background: var(--accent)08;
  border: 1px solid var(--accent)20;
  border-radius: 6px;
  padding: 0.28rem 0.6rem;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.68rem;
  color: var(--accent);
  margin: 0.2rem 0;
  display: flex;
  align-items: center;
  gap: 5px;
  word-break: break-all;
  animation: fadeIn 0.4s ease;
}

.success-toast {
  background: var(--accent)10;
  border: 1px solid var(--accent)35;
  border-radius: 8px;
  padding: 0.45rem 0.8rem;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.7rem;
  color: var(--accent);
  margin-top: 0.4rem;
  animation: slideUp 0.3s ease;
}

/* ── Chat header ── */
.chat-header {
  padding: 1.2rem 2rem 0.7rem 2rem;
  border-bottom: 1px solid var(--border);
  background: var(--bg);
  position: sticky;
  top: 0;
  z-index: 10;
}
.chat-title {
  font-family: 'Space Grotesk', sans-serif;
  font-weight: 700;
  font-size: 1.05rem;
  color: var(--text);
  letter-spacing: -0.01em;
}
.chat-status {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.65rem;
  color: var(--accent);
  display: flex;
  align-items: center;
  gap: 5px;
  margin-top: 2px;
}
.status-dot {
  width: 5px; height: 5px;
  background: var(--accent);
  border-radius: 50%;
  display: inline-block;
  animation: pulse-dot 2s infinite;
  box-shadow: 0 0 6px var(--accent);
}
@keyframes pulse-dot {
  0%,100% { opacity:1; transform:scale(1); box-shadow: 0 0 6px var(--accent); }
  50%      { opacity:.4; transform:scale(.75); box-shadow: 0 0 2px var(--accent); }
}

/* ── Message bubbles ── */
.user-bubble {
  background: var(--user-bg);
  border: 1px solid var(--user-border);
  border-radius: 16px 16px 4px 16px;
  padding: 0.85rem 1.1rem;
  margin: 0.35rem 0 0.35rem 3.5rem;
  font-size: 0.9rem;
  line-height: 1.65;
  box-shadow: var(--shadow);
  word-wrap: break-word;
  animation: slideInRight 0.22s cubic-bezier(0.16, 1, 0.3, 1);
}
@keyframes slideInRight {
  from { opacity:0; transform:translateX(12px); }
  to   { opacity:1; transform:translateX(0); }
}

.ai-bubble {
  background: var(--ai-bg);
  border: 1px solid var(--ai-border);
  border-radius: 16px 16px 16px 4px;
  padding: 0.85rem 1.1rem 0.85rem 1.4rem;
  margin: 0.35rem 3.5rem 0.35rem 0;
  font-size: 0.9rem;
  line-height: 1.75;
  box-shadow: var(--shadow);
  word-wrap: break-word;
  position: relative;
  overflow: hidden;
  animation: slideInLeft 0.28s cubic-bezier(0.16, 1, 0.3, 1);
  white-space: pre-wrap;
}
.ai-bubble::before {
  content: '';
  position: absolute;
  left: 0; top: 0;
  width: 3px; height: 100%;
  background: linear-gradient(180deg, var(--accent), var(--accent2));
  border-radius: 0 0 0 4px;
}
@keyframes slideInLeft {
  from { opacity:0; transform:translateX(-10px); }
  to   { opacity:1; transform:translateX(0); }
}

/* ── Structured response formatting ── */
.ai-bubble .section-label {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.68rem;
  color: var(--accent);
  letter-spacing: 0.15em;
  text-transform: uppercase;
  margin-top: 0.6rem;
  margin-bottom: 0.2rem;
  display: block;
}
.ai-bubble .source-item {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.72rem;
  color: var(--text-muted);
  padding: 0.15rem 0;
}
.ai-bubble .source-item::before { content: "▸ "; color: var(--accent); }

.msg-meta {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.62rem;
  color: var(--text-muted);
  margin-bottom: 0.18rem;
  display: flex;
  align-items: center;
  gap: 5px;
}
.msg-meta .accent { color: var(--accent); }

/* ── Confidence badge ── */
.confidence-badge {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.6rem;
  padding: 0.15rem 0.45rem;
  border-radius: 20px;
  letter-spacing: 0.08em;
}
.conf-high { background: #00ff8815; color: var(--accent); border: 1px solid #00ff8830; }
.conf-med  { background: #ffa04015; color: var(--warn-text); border: 1px solid #ffa04030; }
.conf-low  { background: #ff444415; color: var(--error-text); border: 1px solid #ff444430; }

/* ── Loader ── */
.loader-wrap {
  background: var(--ai-bg);
  border: 1px solid var(--ai-border);
  border-radius: 16px 16px 16px 4px;
  padding: 0.9rem 1.1rem 0.9rem 1.4rem;
  margin: 0.35rem 3.5rem 0.35rem 0;
  position: relative;
  overflow: hidden;
}
.loader-wrap::before {
  content: '';
  position: absolute;
  left: 0; top: 0;
  width: 3px; height: 100%;
  background: linear-gradient(180deg, var(--accent), var(--accent2));
}
.loader-stage {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.78rem;
  color: var(--accent);
  display: flex;
  align-items: center;
  gap: 8px;
  animation: loaderGlow 1.4s ease-in-out infinite;
}
@keyframes loaderGlow {
  0%,100% { opacity:1; }
  50%      { opacity:.4; }
}
.loader-bar {
  height: 1px;
  background: var(--border2);
  border-radius: 1px;
  margin-top: 0.6rem;
  overflow: hidden;
}
.loader-bar-fill {
  height: 100%;
  background: linear-gradient(90deg, transparent, var(--accent), transparent);
  animation: barSweep 1.6s ease-in-out infinite;
}
@keyframes barSweep {
  0%   { width:0%;   margin-left:0%; }
  50%  { width:60%;  margin-left:20%; }
  100% { width:0%;   margin-left:100%; }
}

/* ── Typing cursor ── */
.typing-cursor {
  display: inline-block;
  width: 2px; height: 0.9em;
  background: var(--accent);
  margin-left: 2px;
  animation: blink 0.75s step-end infinite;
  vertical-align: text-bottom;
  box-shadow: 0 0 4px var(--accent);
}
@keyframes blink {
  0%,100% { opacity:1; }
  50%      { opacity:0; }
}

/* ── Empty state ── */
.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 3.5rem 2rem;
  text-align: center;
}
.empty-icon {
  font-size: 2.8rem;
  margin-bottom: 1rem;
  animation: float 3.5s ease-in-out infinite;
  filter: drop-shadow(0 0 12px rgba(0,255,136,0.3));
}
@keyframes float {
  0%,100% { transform:translateY(0); }
  50%      { transform:translateY(-10px); }
}
.empty-title {
  font-family: 'Space Grotesk', sans-serif;
  font-weight: 700;
  font-size: 1.3rem;
  color: var(--text);
  margin-bottom: 0.5rem;
}
.empty-sub {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.74rem;
  color: var(--text-muted);
  line-height: 1.75;
  max-width: 340px;
}

/* ── Error bubble ── */
.error-bubble {
  background: var(--error-bg);
  border: 1px solid var(--error-border);
  border-radius: 12px;
  padding: 0.75rem 1.1rem;
  margin: 0.35rem 3.5rem 0.35rem 0;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.76rem;
  color: var(--error-text);
  animation: slideInLeft 0.25s ease;
}

/* ── Eval metrics strip ── */
.eval-strip {
  display: flex;
  gap: 0.6rem;
  flex-wrap: wrap;
  margin-top: 0.5rem;
  padding-top: 0.5rem;
  border-top: 1px solid var(--border);
}
.eval-pill {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.58rem;
  padding: 0.12rem 0.5rem;
  border-radius: 20px;
  background: var(--bg3);
  border: 1px solid var(--border2);
  color: var(--text-muted);
  letter-spacing: 0.06em;
}
.eval-pill span { color: var(--accent); }

/* ── Feedback buttons ── */
.feedback-row {
  display: flex;
  gap: 0.5rem;
  margin-top: 0.55rem;
  padding-top: 0.5rem;
  border-top: 1px solid var(--border);
}
.fb-btn {
  cursor: pointer;
  padding: 0.2rem 0.6rem;
  border-radius: 6px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.65rem;
  border: 1px solid var(--border2);
  background: transparent;
  color: var(--text-muted);
  transition: all 0.2s ease;
  letter-spacing: 0.06em;
}
.fb-btn:hover { border-color: var(--accent)50; color: var(--accent); background: var(--accent)08; }
.fb-btn.active-up { border-color: var(--accent)60; color: var(--accent); background: var(--accent)10; }
.fb-btn.active-down { border-color: var(--error-text)60; color: var(--error-text); background: #ff444410; }

/* ── Source card ── */
.source-card {
  background: var(--bg3);
  border: 1px solid var(--border2);
  border-radius: 8px;
  padding: 0.55rem 0.8rem;
  margin-top: 0.35rem;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.68rem;
  animation: fadeIn 0.4s ease;
}
.source-doc { color: var(--accent); font-weight: 600; margin-bottom: 0.2rem; }
.source-snippet { color: var(--text-muted); line-height: 1.5; }

/* ── Animations ── */
@keyframes fadeIn {
  from { opacity:0; }
  to   { opacity:1; }
}
@keyframes slideUp {
  from { opacity:0; transform:translateY(8px); }
  to   { opacity:1; transform:translateY(0); }
}

/* ── Scan line effect (cosmetic) ── */
.chat-header::after {
  content: '';
  position: absolute;
  left: 0; right: 0;
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--accent)30, transparent);
  bottom: 0;
  animation: scanLine 4s ease-in-out infinite;
}
@keyframes scanLine {
  0%   { opacity:0; transform:scaleX(0); }
  50%  { opacity:1; transform:scaleX(1); }
  100% { opacity:0; transform:scaleX(0); }
}

/* ── Streamlit overrides ── */
[data-testid="stFileUploaderFile"] {
  background: var(--bg3) !important;
  border: 1px solid var(--border2) !important;
  border-radius: 8px !important;
}
[data-testid="stFileUploaderFileName"] {
  color: var(--text) !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.75rem !important;
}
[data-testid="stBaseButton-secondary"] {
  background: var(--bg3) !important;
  color: var(--text) !important;
  border: 1px solid var(--border2) !important;
  border-radius: 8px !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.72rem !important;
}

/* ── Responsive ── */
@media (max-width: 768px) {
  [data-testid="stChatMessage"] { padding: 0.2rem 0.5rem !important; }
  .user-bubble, .ai-bubble, .loader-wrap, .error-bubble {
    margin-left: 0 !important; margin-right: 0 !important;
  }
  .chat-header { padding: 0.8rem !important; }
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# API CLIENT
# ══════════════════════════════════════════════════════════════

def api_upload(file) -> dict:
    try:
        r = requests.post(
            f"{API_BASE}/upload",
            files={"file": (file.name, file.getvalue(), file.type)},
            timeout=60,
        )
        if r.status_code == 200:
            return {"success": True, "data": r.json()}
        return {"success": False, "message": r.text[:150]}
    except Exception as e:
        return {"success": False, "message": str(e)}


def api_query(question: str, debug: bool = False) -> dict:
    endpoint = "/query-debug" if debug else "/query"
    try:
        r = requests.post(
            f"{API_BASE}{endpoint}",
            json={"question": question},
            timeout=90,
        )
        if r.status_code == 200:
            return {"success": True, **r.json()}
        return {"success": False, "answer": f"Error {r.status_code}: {r.text[:200]}"}
    except ConnectionError:
        return {"success": False, "answer": "Cannot connect to backend (localhost:8005)."}
    except Timeout:
        return {"success": False, "answer": "Request timed out."}
    except Exception as e:
        return {"success": False, "answer": str(e)}


def api_feedback(query: str, answer: str, thumbs_up: bool) -> dict:
    try:
        r = requests.post(
            f"{API_BASE}/feedback",
            json={"query": query, "answer": answer, "rating": 5 if thumbs_up else 1,
                  "thumbs_up": thumbs_up},
            timeout=10,
        )
        return {"success": r.status_code == 200}
    except Exception:
        return {"success": False}


def api_reset() -> dict:
    try:
        r = requests.delete(f"{API_BASE}/reset", timeout=15)
        return {"success": r.status_code == 200}
    except Exception as e:
        return {"success": False, "message": str(e)}


# ══════════════════════════════════════════════════════════════
# LOADER
# ══════════════════════════════════════════════════════════════

LOADER_STAGES = [
    ("🧠", "Thinking"),
    ("🔍", "Searching papers"),
    ("📊", "Hybrid retrieval"),
    ("🔗", "Reranking context"),
    ("⚡", "Generating answer"),
]
SPIN = ["◐", "◓", "◑", "◒"]


def render_loader_frame(placeholder, stage_idx: int, frame: int):
    icon, label = LOADER_STAGES[stage_idx]
    spin = SPIN[frame % 4]
    dots = "." * ((frame % 3) + 1)
    placeholder.markdown(f"""
<div class="msg-meta">
  <span class="accent">⚡ RAG</span><span>·</span><span>Processing</span>
</div>
<div class="loader-wrap">
  <div class="loader-stage">
    <span style="font-size:.95rem">{icon}</span>
    <span>{spin}</span>
    <span>{label}{dots}</span>
  </div>
  <div class="loader-bar"><div class="loader-bar-fill"></div></div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# RENDER HELPERS
# ══════════════════════════════════════════════════════════════

def render_user_bubble(text: str):
    st.markdown(f"""
<div class="msg-meta" style="justify-content:flex-end;padding:0 2rem;">
  <span>You</span>
</div>
<div class="user-bubble">{text}</div>
""", unsafe_allow_html=True)


def _confidence_badge(label: str, score: float) -> str:
    cls = {"HIGH": "conf-high", "MEDIUM": "conf-med", "LOW": "conf-low"}.get(label, "conf-low")
    icon = {"HIGH": "●", "MEDIUM": "◑", "LOW": "○"}.get(label, "○")
    return f'<span class="confidence-badge {cls}">{icon} {label} {score:.2f}</span>'


def _eval_strip(eval_data: dict) -> str:
    if not eval_data:
        return ""
    metrics = [
        ("Grounded", eval_data.get("groundedness_score", 0)),
        ("Halluc↓", eval_data.get("hallucination_rate", 0)),
        ("Relevance", eval_data.get("answer_relevance", 0)),
        ("F1", eval_data.get("f1_retrieval", 0)),
        ("Overall", eval_data.get("overall_score", 0)),
    ]
    pills = "".join(
        f'<span class="eval-pill">{k}: <span>{v:.2f}</span></span>'
        for k, v in metrics
    )
    return f'<div class="eval-strip">{pills}</div>'


def render_ai_bubble(
    answer: str,
    confidence: float = 0.0,
    confidence_label: str = "",
    sources: list = None,
    eval_data: dict = None,
    msg_id: str = "",
):
    badge = _confidence_badge(confidence_label, confidence) if confidence_label else ""
    eval_html = _eval_strip(eval_data or {})

    # Format sources
    sources_html = ""
    if sources:
        items = "".join(
            f'<div class="source-card">'
            f'<div class="source-doc">📄 {s.get("document","")}</div>'
            f'<div class="source-snippet">{s.get("snippet","")[:200]}…</div>'
            f'</div>'
            for s in sources[:3]
        )
        sources_html = f'<details style="margin-top:.5rem;"><summary style="font-family:JetBrains Mono,monospace;font-size:.65rem;color:var(--text-muted);cursor:pointer;letter-spacing:.1em;">▸ VIEW SOURCES ({len(sources)})</summary>{items}</details>'

    # Feedback buttons (use Streamlit buttons below for interactivity)
    feedback_html = f"""
<div class="feedback-row">
  <span style="font-family:JetBrains Mono,monospace;font-size:.6rem;color:var(--text-muted);align-self:center;letter-spacing:.1em;">Was this helpful?</span>
</div>
"""

    st.markdown(f"""
<div class="msg-meta" style="padding:0 2rem;">
  <span class="accent">⚡ RAG</span><span>·</span><span>Assistant</span>
  {badge}
</div>
<div class="ai-bubble">{answer}{eval_html}{sources_html}{feedback_html}</div>
""", unsafe_allow_html=True)

    # Feedback buttons via Streamlit (for state management)
    col1, col2, col3 = st.columns([0.08, 0.08, 0.84])
    with col1:
        if st.button("👍", key=f"up_{msg_id}", help="Helpful"):
            res = api_feedback(
                st.session_state.messages[-2]["content"] if len(st.session_state.messages) >= 2 else "",
                answer, True
            )
            if res["success"]:
                st.toast("Thanks for the feedback!", icon="✅")
    with col2:
        if st.button("👎", key=f"dn_{msg_id}", help="Not helpful"):
            res = api_feedback(
                st.session_state.messages[-2]["content"] if len(st.session_state.messages) >= 2 else "",
                answer, False
            )
            if res["success"]:
                st.toast("Feedback noted.", icon="📝")


def render_error_bubble(message: str):
    st.markdown(f"""
<div class="msg-meta" style="padding:0 2rem;">
  <span style="color:var(--error-text);">✗ Error</span>
</div>
<div class="error-bubble">⚠ {message}</div>
""", unsafe_allow_html=True)


def typing_effect(placeholder, text: str):
    st.markdown("""
<div class="msg-meta" style="padding:0 2rem;">
  <span class="accent">⚡ RAG</span><span>·</span><span>Assistant</span>
</div>
""", unsafe_allow_html=True)
    chunk = 4
    for i in range(0, len(text) + chunk, chunk):
        shown = text[:i]
        cursor = '<span class="typing-cursor"></span>' if i <= len(text) else ""
        placeholder.markdown(
            f'<div class="ai-bubble">{shown}{cursor}</div>',
            unsafe_allow_html=True,
        )
        time.sleep(0.01)
    placeholder.empty()


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        # Logo
        st.markdown("""
<div style="display:flex;align-items:center;gap:10px;
            padding-bottom:1rem;border-bottom:1px solid #141414;margin-bottom:.5rem;">
  <div style="width:34px;height:34px;background:#00ff88;border-radius:9px;
              display:flex;align-items:center;justify-content:center;
              font-size:16px;flex-shrink:0;box-shadow:0 0 16px #00ff8840;">⚡</div>
  <div>
    <div style="font-family:'Space Grotesk',sans-serif;font-weight:700;
                font-size:.95rem;color:#e2e8e4;">RAG Intelligence</div>
    <div style="font-family:'JetBrains Mono',monospace;font-size:.6rem;
                color:#00ff88;letter-spacing:.1em;">v2.0 · EXPLAINABLE</div>
  </div>
</div>
""", unsafe_allow_html=True)

        # Debug mode toggle
        st.markdown('<div class="sidebar-label">⚙ Settings</div>', unsafe_allow_html=True)
        st.session_state.debug_mode = st.toggle(
            "Debug Mode (show metrics)", value=st.session_state.get("debug_mode", False)
        )

        st.markdown("---")

        # Documents
        st.markdown('<div class="sidebar-label">📂 Documents</div>', unsafe_allow_html=True)

        uploaded_files = st.file_uploader(
            "Upload PDF or TXT",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key="file_uploader",
        )

        if uploaded_files:
            if not st.session_state.get("db_reset_done", False):
                with st.spinner("Clearing old index…"):
                    res = api_reset()
                if res["success"]:
                    st.session_state.db_reset_done = True
                    st.session_state.uploaded_files = []

            for f in uploaded_files:
                if f.name not in st.session_state.uploaded_files:
                    with st.spinner(f"Indexing {f.name}…"):
                        res = api_upload(f)
                    if res["success"]:
                        st.session_state.uploaded_files.append(f.name)
                        st.markdown(
                            f'<div class="success-toast">✓ {f.name} indexed</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.error(res.get("message", "Upload failed"))

        # File tags
        if st.session_state.uploaded_files:
            st.markdown('<div style="margin-top:.5rem">', unsafe_allow_html=True)
            for fname in st.session_state.uploaded_files:
                icon = "📄" if fname.lower().endswith(".txt") else "📕"
                st.markdown(f'<div class="file-tag">{icon} {fname}</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")

        # Actions
        st.markdown('<div class="sidebar-label">🗑 Actions</div>', unsafe_allow_html=True)
        if st.button("Clear Chat History", key="clear_btn"):
            st.session_state.messages = []
            st.rerun()
        if st.button("Reset Index", key="reset_btn"):
            with st.spinner("Resetting…"):
                api_reset()
                st.session_state.uploaded_files = []
                st.session_state.db_reset_done = False
            st.success("Index cleared")
            st.rerun()

        # Stats footer
        doc_count = len(st.session_state.uploaded_files)
        st.markdown(f"""
<div style="margin-top:1.8rem;font-family:'JetBrains Mono',monospace;
            font-size:.62rem;color:#404844;line-height:2.0;">
  Backend → <span style="color:#00ff88;">localhost:8005</span><br>
  Retrieval → <span style="color:#00ff88;">FAISS + BM25</span><br>
  Reranker → <span style="color:#00ff88;">cross-encoder</span><br>
  Docs → <span style="color:#00ff88;">{doc_count} indexed</span>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# CHAT HANDLER
# ══════════════════════════════════════════════════════════════

def handle_user_input(prompt: str):
    render_user_bubble(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Background fetch
    result_holder = {"data": None, "done": False}

    def fetch():
        data = api_query(prompt, debug=st.session_state.get("debug_mode", False))
        result_holder["data"] = data
        result_holder["done"] = True

    thread = threading.Thread(target=fetch, daemon=True)
    thread.start()

    # Animated loader
    loader_ph = st.empty()
    frame, tick = 0, 0

    while not result_holder["done"]:
        stage = min(tick // 6, len(LOADER_STAGES) - 1)
        render_loader_frame(loader_ph, stage, frame)
        time.sleep(0.16)
        frame += 1
        tick  += 1

    render_loader_frame(loader_ph, len(LOADER_STAGES) - 1, frame)
    thread.join(timeout=2)
    time.sleep(0.2)
    loader_ph.empty()

    # Display answer
    data = result_holder["data"] or {}
    success = data.get("success", False)

    if not success:
        msg = data.get("answer", "No response received.")
        render_error_bubble(msg)
        st.session_state.messages.append({"role": "assistant", "content": msg, "error": True})
        return

    answer = data.get("answer", "No answer returned.")
    confidence = data.get("confidence", 0.0)
    confidence_label = data.get("confidence_label", "")
    sources = data.get("sources", [])
    eval_data = data.get("evaluation", {})

    # Typing effect then render full bubble
    ph = st.empty()
    typing_effect(ph, answer)

    msg_id = str(len(st.session_state.messages))
    render_ai_bubble(
        answer=answer,
        confidence=confidence,
        confidence_label=confidence_label,
        sources=sources,
        eval_data=eval_data,
        msg_id=msg_id,
    )

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "confidence": confidence,
        "confidence_label": confidence_label,
        "sources": sources,
        "eval": eval_data,
        "error": False,
        "id": msg_id,
    })


# ══════════════════════════════════════════════════════════════
# HISTORY REPLAY
# ══════════════════════════════════════════════════════════════

def render_history():
    for i, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            render_user_bubble(msg["content"])
        elif msg["role"] == "assistant":
            if msg.get("error"):
                render_error_bubble(msg["content"])
            else:
                render_ai_bubble(
                    answer=msg["content"],
                    confidence=msg.get("confidence", 0.0),
                    confidence_label=msg.get("confidence_label", ""),
                    sources=msg.get("sources", []),
                    eval_data=msg.get("eval", {}),
                    msg_id=msg.get("id", str(i)),
                )


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

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


def main():
    _init_session()
    inject_css()
    render_sidebar()

    # Chat header
    st.markdown("""
<div class="chat-header" style="position:relative;">
  <div class="chat-title">Neural Document Assistant</div>
  <div class="chat-status">
    <span class="status-dot"></span>
    <span>Hybrid RAG · Grounded · Explainable · Zero Hallucination</span>
  </div>
</div>
""", unsafe_allow_html=True)

    # Empty state
    if not st.session_state.messages:
        st.markdown("""
<div class="empty-state">
  <div class="empty-icon">🧠</div>
  <div class="empty-title">Intelligence Ready</div>
  <div class="empty-sub">
    Upload research papers in the sidebar,<br>
    then ask questions below.<br><br>
    <span style="color:#00ff88;">⚡ Hybrid BM25 + FAISS · Cross-Encoder Reranking</span>
  </div>
</div>
""", unsafe_allow_html=True)
    else:
        render_history()

    if prompt := st.chat_input("Ask about your research papers…"):
        handle_user_input(prompt)


if __name__ == "__main__":
    main()