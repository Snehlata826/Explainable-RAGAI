"""
RAG pipeline orchestrator.

Single entry-point for:
- Document ingestion
- Query processing (retrieval → reranking → generation → explanation)
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import List

from config.settings import DATA_DIR
from embeddings.embedding_generator import embed_texts
from explainability.explanation_engine import ExplainedResponse, build_explained_response
from generation.answer_generator import generate_answer
from ingestion.document_processor import DocumentChunk, process_document
from monitoring.logger import get_logger, log_query_event
from retrieval.context_retriever import ContextRetriever
from vector_store.faiss_store import FAISSVectorStore

logger = get_logger(__name__)


# ───────────────── Query Expansion ─────────────────

def expand_query(question: str) -> list[str]:
    """
    Generate multiple search queries from the original question.
    Improves retrieval coverage.
    """

    base = question.strip()

    queries = [
        base,
        f"Explain {base}",
        f"Definition of {base}",
        f"{base} in machine learning",
    ]

    return queries


# ─────────────── Context Compression ───────────────
def compress_context(question: str, contexts):

    keywords = set(question.lower().split())

    for ctx in contexts:

        sentences = re.split(r'(?<=[.!?]) +', ctx.text)

        filtered = [
            s for s in sentences
            if keywords & set(s.lower().split())
        ]

        if filtered:
            # modify underlying chunk text
            ctx.chunk.text = " ".join(filtered)

    return contexts


# ───────────────── RAG Pipeline ─────────────────

class RAGPipeline:

    def __init__(self) -> None:

        self.vector_store = FAISSVectorStore()
        self.retriever = ContextRetriever(self.vector_store)

        self._load_existing_index()

    # ───────── Initialization ─────────

    def _load_existing_index(self) -> None:
        """Load previously saved FAISS index if available."""
        self.vector_store.load()

    # ───────── Duplicate Document Protection ─────────

    def _document_exists(self, document_name: str) -> bool:

        if not hasattr(self.vector_store, "chunks"):
            return False

        for chunk in self.vector_store.chunks:

            if getattr(chunk, "document_name", None) == document_name:
                return True

        return False

    # ───────── Document Ingestion ─────────

    def ingest(self, file_path: Path) -> int:

        logger.info(f"Starting ingestion: {file_path.name}")

        # Prevent duplicate ingestion
        if self._document_exists(file_path.name):

            logger.warning(f"{file_path.name} already indexed — skipping.")

            return 0

        chunks: List[DocumentChunk] = process_document(file_path)

        if not chunks:

            logger.warning(f"No chunks produced for {file_path.name}")

            return 0

        texts = [c.text for c in chunks]

        embeddings = embed_texts(texts)

        self.vector_store.add_chunks(chunks, embeddings)

        self.vector_store.save()

        logger.info(f"Ingestion complete: {len(chunks)} chunks from {file_path.name}")

        return len(chunks)

    # ───────── Directory Ingestion ─────────

    def ingest_directory(self, directory: Path = DATA_DIR) -> int:

        total = 0

        for fp in sorted(directory.iterdir()):

            if fp.suffix.lower() in (".pdf", ".txt", ".md"):

                try:

                    total += self.ingest(fp)

                except Exception as exc:

                    logger.error(f"Failed to ingest {fp.name}: {exc}")

        return total

    # ───────── Context Deduplication ─────────

    def _deduplicate_contexts(self, contexts):

        seen = set()

        unique_contexts = []

        for ctx in contexts:

            key = (ctx.document_name, ctx.text[:100])

            if key in seen:
                continue

            seen.add(key)

            unique_contexts.append(ctx)

        return unique_contexts

    # ───────── Query Pipeline ─────────

    def query(self, question: str) -> ExplainedResponse:
        question = question.strip().lower()
        start = time.perf_counter()

        logger.info(f"Query received: '{question}'")

        # 1️⃣ Expand query
        queries = expand_query(question)

        # 2️⃣ Retrieve contexts
        all_contexts = []

        for q in queries:

            retrieved = self.retriever.retrieve(q)

            all_contexts.extend(retrieved)

        # 3️⃣ Remove duplicates
        contexts = self._deduplicate_contexts(all_contexts)

        # 4️⃣ Compress context (sentence filtering)
        contexts = compress_context(question, contexts)

        # 5️⃣ Limit context size (faster LLM)
        contexts = contexts[:5]

        # 6️⃣ Generate answer
        answer = generate_answer(question, contexts)

        # 7️⃣ Build explainable response
        response = build_explained_response(answer, contexts)

        latency_ms = (time.perf_counter() - start) * 1000

        log_query_event(
            query=question,
            num_sources=len(response.sources),
            confidence=response.confidence,
            latency_ms=latency_ms,
        )

        logger.info(f"Query handled in {latency_ms:.1f} ms")

        return response

    # ───────── Reset Vector Store ─────────

    def reset(self) -> None:

        self.vector_store.clear()

        self.vector_store.save()

        logger.info("Vector store reset.")

    # ───────── Stats ─────────

    @property
    def num_chunks(self) -> int:

        return self.vector_store.num_chunks