"""
High-level context retriever.

Orchestrates:
  1. Hybrid retrieval (FAISS + BM25)
  2. Cross-encoder reranking
  3. Returns final context with scores
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from config.settings import TOP_K_RERANK, TOP_K_RETRIEVAL
from embeddings.embedding_generator import embed_query
from ingestion.document_processor import DocumentChunk
from monitoring.logger import get_logger
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import rerank
from vector_store.faiss_store import FAISSVectorStore

logger = get_logger(__name__)


@dataclass
class RetrievedContext:
    """Encapsulates a single retrieved and reranked document snippet."""

    chunk: DocumentChunk
    retrieval_score: float
    rerank_score: float

    @property
    def document_name(self) -> str:
        return self.chunk.document_name

    @property
    def text(self) -> str:
        return self.chunk.text


class ContextRetriever:
    """
    End-to-end retrieval pipeline combining hybrid search and reranking.
    """

    def __init__(self, vector_store: FAISSVectorStore) -> None:
        self.vector_store = vector_store
        self.hybrid_retriever = HybridRetriever(vector_store)

    def retrieve(
        self,
        query: str,
        top_k_retrieval: int = 10,   # retrieve more candidates first
        top_k_rerank: int = 5        # final 5 chunks
    ) -> List[RetrievedContext]:

        logger.info(f"Retrieving context for: '{query}'")

        # 1️⃣ Embed query
        q_emb: np.ndarray = embed_query(query)

        # 2️⃣ Hybrid retrieval (FAISS + BM25)
        candidates = self.hybrid_retriever.retrieve(
            query,
            q_emb,
            top_k=top_k_retrieval
        )

        if not candidates:
            logger.warning("No candidates returned from hybrid retrieval.")
            return []

        # Retrieval score map
        retrieval_score_map = {c.chunk_id: s for c, s in candidates}

        # 3️⃣ Cross-encoder reranking
        reranked = rerank(query, candidates, top_k=top_k_rerank)

        contexts: List[RetrievedContext] = []

        for chunk, rerank_score in reranked:

            ctx = RetrievedContext(
                chunk=chunk,
                retrieval_score=retrieval_score_map.get(chunk.chunk_id, 0.0),
                rerank_score=rerank_score,
            )

            contexts.append(ctx)

        logger.info(f"Retrieved {len(contexts)} final context chunks.")

        return contexts


# ─────────────────────────────────────────────────────────────
# Helper function to combine chunks into a single context string
# ─────────────────────────────────────────────────────────────

def combine_context(contexts: List[RetrievedContext]) -> str:
    """
    Combine retrieved chunks into a single context string
    for the LLM prompt.
    """

    if not contexts:
        return ""

    combined = "\n\n".join([ctx.text for ctx in contexts])

    return combined