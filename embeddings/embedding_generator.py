"""
Embedding generation using Sentence Transformers.

Model: sentence-transformers/all-MiniLM-L6-v2
"""

from __future__ import annotations

from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from config.settings import EMBEDDING_MODEL
from monitoring.logger import get_logger

logger = get_logger(__name__)

# ── Singleton model loader ─────────────────────────────────────────────────

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Lazy-load the embedding model (singleton pattern)."""
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        _model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("Embedding model loaded.")
    return _model


# ── Public API ─────────────────────────────────────────────────────────────

def embed_texts(texts: List[str], batch_size: int = 64) -> np.ndarray:
    """
    Generate L2-normalised embeddings for a list of text strings.

    Args:
        texts: List of input strings.
        batch_size: Number of texts to encode in one forward pass.

    Returns:
        Float32 NumPy array of shape (len(texts), EMBEDDING_DIM).
    """
    model = _get_model()
    logger.debug(f"Embedding {len(texts)} texts (batch_size={batch_size})")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 50,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def embed_query(query: str) -> np.ndarray:
    """
    Generate a query embedding optimized for retrieval.
    BGE models require a special instruction prefix.
    """

    instruction = "Represent this sentence for searching relevant passages: "

    query_text = instruction + query

    return embed_texts([query_text])[0]