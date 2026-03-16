"""
FAISS-backed vector store with persistence support.

Handles:
- Adding document chunk embeddings
- Similarity search
- Save / load index to disk
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np

from config.settings import EMBEDDING_DIM, FAISS_INDEX_DIR
from ingestion.document_processor import DocumentChunk
from monitoring.logger import get_logger

logger = get_logger(__name__)

_INDEX_FILE = FAISS_INDEX_DIR / "index.faiss"
_META_FILE = FAISS_INDEX_DIR / "metadata.pkl"


class FAISSVectorStore:
    """
    Persistent FAISS vector store backed by an IndexFlatIP index
    (inner-product on L2-normalised embeddings = cosine similarity).
    """

    def __init__(self) -> None:
        self.index: faiss.IndexFlatIP = faiss.IndexFlatIP(EMBEDDING_DIM)
        self.chunks: List[DocumentChunk] = []  # parallel list to index rows

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self) -> None:
        """Persist index and metadata to disk."""
        FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(_INDEX_FILE))
        with open(_META_FILE, "wb") as f:
            pickle.dump(self.chunks, f)
        logger.info(f"FAISS index saved ({self.index.ntotal} vectors).")

    def load(self) -> bool:
        """
        Load index and metadata from disk.

        Returns:
            True if loaded successfully, False if no saved index found.
        """
        if not _INDEX_FILE.exists() or not _META_FILE.exists():
            logger.info("No existing FAISS index found – starting fresh.")
            return False
        self.index = faiss.read_index(str(_INDEX_FILE))
        with open(_META_FILE, "rb") as f:
            self.chunks = pickle.load(f)
        logger.info(f"FAISS index loaded ({self.index.ntotal} vectors).")
        return True

    # ── CRUD ───────────────────────────────────────────────────────────────

    def add_chunks(
        self, chunks: List[DocumentChunk], embeddings: np.ndarray
    ) -> None:
        """
        Add document chunks and their embeddings to the store.

        Args:
            chunks: List of DocumentChunk objects.
            embeddings: Float32 array of shape (len(chunks), EMBEDDING_DIM).
        """
        if len(chunks) != embeddings.shape[0]:
            raise ValueError("Chunks and embeddings count mismatch.")
        self.index.add(embeddings)
        self.chunks.extend(chunks)
        logger.info(f"Added {len(chunks)} chunks. Total: {self.index.ntotal}")

    def clear(self) -> None:
        """Remove all vectors and metadata."""
        self.index.reset()
        self.chunks.clear()
        logger.info("FAISS store cleared.")

    # ── Search ─────────────────────────────────────────────────────────────

    def search(
        self, query_embedding: np.ndarray, top_k: int = 10
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Return the top-k most similar chunks for a query embedding.

        Args:
            query_embedding: 1-D float32 array of shape (EMBEDDING_DIM,).
            top_k: Number of results to return.

        Returns:
            List of (DocumentChunk, cosine_similarity_score) tuples,
            sorted by descending score.
        """
        if self.index.ntotal == 0:
            logger.warning("Vector store is empty – no results returned.")
            return []

        q = query_embedding.reshape(1, -1).astype(np.float32)
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(q, k)

        results: List[Tuple[DocumentChunk, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score)))

        return results

    # ── Helpers ────────────────────────────────────────────────────────────

    @property
    def num_chunks(self) -> int:
        """Number of indexed chunks."""
        return self.index.ntotal

    def get_all_texts(self) -> List[str]:
        """Return raw text of every stored chunk (needed by BM25)."""
        return [c.text for c in self.chunks]

    def get_chunk_by_index(self, idx: int) -> DocumentChunk:
        """Retrieve a chunk by its position in the parallel list."""
        return self.chunks[idx]
