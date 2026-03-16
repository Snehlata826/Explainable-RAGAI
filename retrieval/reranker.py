"""
Cross-encoder reranking for retrieved document chunks.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2

Takes (query, passage) pairs and returns relevance scores that are far
more accurate than bi-encoder cosine similarity.
"""

from __future__ import annotations

from typing import List, Tuple

from sentence_transformers import CrossEncoder

from config.settings import RERANKER_MODEL, TOP_K_RERANK
from ingestion.document_processor import DocumentChunk
from monitoring.logger import get_logger

logger = get_logger(__name__)

# ── Singleton loader ───────────────────────────────────────────────────────

_reranker: CrossEncoder | None = None


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        logger.info(f"Loading reranker model: {RERANKER_MODEL}")
        _reranker = CrossEncoder(RERANKER_MODEL)
        logger.info("Reranker model loaded.")
    return _reranker


# ── Public API ─────────────────────────────────────────────────────────────

def rerank(
    query: str,
    candidates: List[Tuple[DocumentChunk, float]],
    top_k: int = TOP_K_RERANK,
) -> List[Tuple[DocumentChunk, float]]:

    if not candidates:
        return []

    reranker = _get_reranker()

    pairs = [(query, chunk.text) for chunk, _ in candidates]

    scores: List[float] = reranker.predict(
        pairs,
        batch_size=32,
        show_progress_bar=False
    ).tolist()

    reranked = sorted(
        zip([c for c, _ in candidates], scores),
        key=lambda x: x[1],
        reverse=True,
    )

    logger.debug(
        f"Reranked {len(candidates)} candidates → top {top_k}: "
        + str([round(s, 3) for _, s in reranked[:top_k]])
    )

    return reranked[:top_k]