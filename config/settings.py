"""
Central configuration for GenAI RAG Copilot.
All settings can be overridden via environment variables.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
print("HF TOKEN:", os.getenv("HF_API_TOKEN"))

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "raw_docs"
FAISS_INDEX_DIR = BASE_DIR / "data" / "faiss_index"
FEEDBACK_DIR = BASE_DIR / "data" / "feedback"
LOG_DIR = BASE_DIR / "logs"

for d in [DATA_DIR, FAISS_INDEX_DIR, FEEDBACK_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Embedding ──────────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = os.getenv(
    "EMBEDDING_MODEL",
    "BAAI/bge-small-en-v1.5"
)

EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", "384"))

# ── Chunking (optimized for research papers) ───────────────────────────────
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "400"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))

# ── Retrieval ──────────────────────────────────────────────────────────────
TOP_K_RETRIEVAL: int = int(os.getenv("TOP_K_RETRIEVAL", "10"))
TOP_K_RERANK: int = int(os.getenv("TOP_K_RERANK", "3"))

BM25_WEIGHT: float = float(os.getenv("BM25_WEIGHT", "0.35"))
VECTOR_WEIGHT: float = float(os.getenv("VECTOR_WEIGHT", "0.65"))

# ── Reranker ───────────────────────────────────────────────────────────────
RERANKER_MODEL: str = os.getenv(
    "RERANKER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# ── LLM (Hugging Face Inference API) ───────────────────────────────────────

HF_API_TOKEN = os.getenv("HF_API_TOKEN")

HF_MODEL = os.getenv(
    "HF_MODEL",
    "mistralai/Mistral-7B-Instruct-v0.2"
)


LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "512"))

# ── API ────────────────────────────────────────────────────────────────────
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8005"))

# ── Confidence thresholds ──────────────────────────────────────────────────
HIGH_CONFIDENCE: float = 0.75
MEDIUM_CONFIDENCE: float = 0.50