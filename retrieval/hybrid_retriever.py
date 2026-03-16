"""
Hybrid retrieval: fuses FAISS vector search with BM25 keyword search.

Scores are combined via weighted linear fusion:
    final_score = VECTOR_WEIGHT * vector_score + BM25_WEIGHT * bm25_score
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

from config.settings import BM25_WEIGHT, TOP_K_RETRIEVAL, VECTOR_WEIGHT
from ingestion.document_processor import DocumentChunk
from monitoring.logger import get_logger
from vector_store.faiss_store import FAISSVectorStore

logger = get_logger(__name__)


def _tokenize(text: str) -> List[str]:
    """Simple whitespace tokenizer for BM25."""
    return text.lower().split()


class HybridRetriever:
    """
    Combines cosine-similarity vector search with BM25 keyword matching.

    The BM25 index is rebuilt lazily from the current FAISS chunk list,
    so it is always in sync without manual management.
    """

    def __init__(self, vector_store: FAISSVectorStore) -> None:
        self.vector_store = vector_store
        self._bm25: BM25Okapi | None = None
        self._bm25_corpus_size: int = 0

    # ── BM25 lifecycle ─────────────────────────────────────────────────────

    def _rebuild_bm25_if_needed(self) -> None:
        """Rebuild the BM25 index if the corpus has grown."""
        current_size = self.vector_store.num_chunks
        if current_size == self._bm25_corpus_size:
            return
        texts = self.vector_store.get_all_texts()
        tokenised = [_tokenize(t) for t in texts]
        self._bm25 = BM25Okapi(tokenised)
        self._bm25_corpus_size = current_size
        logger.debug(f"BM25 index rebuilt ({current_size} docs).")

    # ── Retrieval ──────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        query_embedding: np.ndarray,
        top_k: int = TOP_K_RETRIEVAL,
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Run hybrid retrieval and return fused top-k results.

        Args:
            query: Raw query string (used for BM25).
            query_embedding: Pre-computed query embedding (used for FAISS).
            top_k: Number of results to return.

        Returns:
            List of (DocumentChunk, fused_score) sorted descending.
        """
        if self.vector_store.num_chunks == 0:
            logger.warning("Vector store is empty.")
            return []

        self._rebuild_bm25_if_needed()

        # -- Vector scores -------------------------------------------------
        vec_results = self.vector_store.search(query_embedding, top_k=top_k * 2)
        vec_scores: Dict[str, float] = {
            c.chunk_id: s for c, s in vec_results
        }
        vec_chunk_map: Dict[str, DocumentChunk] = {
            c.chunk_id: c for c, _ in vec_results
        }

        # -- BM25 scores ---------------------------------------------------
        query_tokens = _tokenize(query)
        bm25_raw: np.ndarray = self._bm25.get_scores(query_tokens)  # type: ignore[union-attr]

        # Normalise BM25 scores to [0, 1]
        bm25_max = bm25_raw.max() if bm25_raw.max() > 0 else 1.0
        bm25_norm = bm25_raw / bm25_max

        # Collect top BM25 candidates
        bm25_top_indices = np.argsort(bm25_norm)[::-1][: top_k * 2]
        bm25_scores: Dict[str, float] = {}
        bm25_chunk_map: Dict[str, DocumentChunk] = {}
        for idx in bm25_top_indices:
            chunk = self.vector_store.get_chunk_by_index(int(idx))
            bm25_scores[chunk.chunk_id] = float(bm25_norm[idx])
            bm25_chunk_map[chunk.chunk_id] = chunk

        # -- Fusion --------------------------------------------------------
        all_ids = set(vec_scores) | set(bm25_scores)
        fused: List[Tuple[DocumentChunk, float]] = []

        for cid in all_ids:
            vs = vec_scores.get(cid, 0.0)
            bs = bm25_scores.get(cid, 0.0)
            fused_score = VECTOR_WEIGHT * vs + BM25_WEIGHT * bs
            chunk = vec_chunk_map.get(cid) or bm25_chunk_map[cid]
            fused.append((chunk, fused_score))

        fused.sort(key=lambda x: x[1], reverse=True)
        logger.debug(f"Hybrid retrieval returned {len(fused[:top_k])} results.")
        return fused[:top_k]
