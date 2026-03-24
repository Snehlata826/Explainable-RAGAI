"""
Production-grade RAG Pipeline Orchestrator.

Features:
- Strict hallucination mitigation (context-only grounding)
- Multi-document reasoning with per-paper grouping
- Hybrid retrieval (FAISS + BM25)
- Similarity threshold filtering
- Evaluation metrics integration
- Query expansion
- Context deduplication and compression
"""

from __future__ import annotations

import re
import time
from collections import defaultdict
from pathlib import Path
from typing import List, Dict

from config.settings import DATA_DIR, SIMILARITY_THRESHOLD
from embeddings.embedding_generator import embed_texts
from explainability.explanation_engine import ExplainedResponse, build_explained_response
from generation.answer_generator import generate_answer
from ingestion.document_processor import DocumentChunk, process_document
from monitoring.logger import get_logger, log_query_event
from retrieval.context_retriever import ContextRetriever
from vector_store.faiss_store import FAISSVectorStore

logger = get_logger(__name__)


# ─────────────────────── Query Expansion ───────────────────────

def expand_query(question: str) -> list[str]:
    """
    Generate multiple search queries to improve recall across papers.
    Keeps expansions semantically tight to avoid noise.
    """
    base = question.strip()
    return [
        base,
        f"What is {base}",
        f"{base} methodology approach",
        f"{base} results findings",
    ]


# ─────────────────────── Context Compression ───────────────────────

def compress_context(question: str, contexts) -> list:
    """
    Filter each chunk to only sentences containing query keywords.
    Reduces prompt size and focuses the LLM on relevant sentences.
    """
    keywords = set(re.findall(r"\b\w{4,}\b", question.lower()))
    compressed = []

    for ctx in contexts:
        sentences = re.split(r'(?<=[.!?])\s+', ctx.text)
        relevant = [s for s in sentences if keywords & set(s.lower().split())]

        if relevant:
            ctx.chunk.text = " ".join(relevant)
        # Keep chunk even if no keyword overlap — reranker already scored it

        compressed.append(ctx)

    return compressed


# ─────────────────────── Multi-Document Grouping ───────────────────────

def group_contexts_by_document(contexts) -> Dict[str, list]:
    """
    Group retrieved contexts by source document name.
    Enables per-paper reasoning and comparison.
    """
    groups: Dict[str, list] = defaultdict(list)
    for ctx in contexts:
        groups[ctx.document_name].append(ctx)
    return dict(groups)


# ─────────────────────── RAG Pipeline ───────────────────────

class RAGPipeline:

    def __init__(self) -> None:
        self.vector_store = FAISSVectorStore()
        self.retriever = ContextRetriever(self.vector_store)
        self._load_existing_index()

    # ───────── Initialization ─────────

    def _load_existing_index(self) -> None:
        self.vector_store.load()

    # ───────── Duplicate Protection ─────────

    def _document_exists(self, document_name: str) -> bool:
        if not hasattr(self.vector_store, "chunks"):
            return False
        return any(
            getattr(c, "document_name", None) == document_name
            for c in self.vector_store.chunks
        )

    # ───────── Document Ingestion ─────────

    def ingest(self, file_path: Path) -> int:
        logger.info(f"Starting ingestion: {file_path.name}")

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

    # ───────── Deduplication ─────────

    def _deduplicate_contexts(self, contexts) -> list:
        seen = set()
        unique = []
        for ctx in contexts:
            key = (ctx.document_name, ctx.text[:120])
            if key not in seen:
                seen.add(key)
                unique.append(ctx)
        return unique

    # ───────── Similarity Threshold Filtering ─────────

    def _filter_by_threshold(self, contexts, threshold: float = SIMILARITY_THRESHOLD) -> list:
        """
        Remove contexts with retrieval score below threshold.
        Prevents irrelevant chunks from polluting the prompt.
        """
        filtered = [ctx for ctx in contexts if ctx.retrieval_score >= threshold]
        if not filtered:
            logger.warning("No contexts passed similarity threshold — returning top-3 regardless.")
            return contexts[:3]
        return filtered

    # ───────── Main Query Pipeline ─────────

    def query(self, question: str) -> ExplainedResponse:
        question = question.strip()
        start = time.perf_counter()
        logger.info(f"Query received: '{question}'")

        # 1. Expand query for better recall
        queries = expand_query(question)

        # 2. Retrieve from hybrid search (FAISS + BM25)
        all_contexts = []
        for q in queries:
            retrieved = self.retriever.retrieve(
                q,
                top_k_retrieval=10,
                top_k_rerank=3,
            )
            all_contexts.extend(retrieved)

        # 3. Deduplicate
        contexts = self._deduplicate_contexts(all_contexts)

        # 4. Filter by similarity threshold
        contexts = self._filter_by_threshold(contexts)

        # 5. Compress context
        contexts = compress_context(question, contexts)

        # 6. Limit to top-5 for LLM
        contexts = contexts[:5]

        # 7. Group by document for multi-paper reasoning
        doc_groups = group_contexts_by_document(contexts)
        multi_doc = len(doc_groups) > 1

        # 8. Generate answer (strict grounding enforced in answer_generator)
        answer = generate_answer(question, contexts, doc_groups=doc_groups, multi_doc=multi_doc)

        # 9. Build explainable response
        response = build_explained_response(answer, contexts)

        latency_ms = (time.perf_counter() - start) * 1000
        log_query_event(
            query=question,
            num_sources=len(response.sources),
            confidence=response.confidence,
            latency_ms=latency_ms,
        )

        logger.info(f"Query handled in {latency_ms:.1f} ms | confidence={response.confidence}")
        return response

    # ───────── Reset ─────────

    def reset(self) -> None:
        self.vector_store.clear()
        self.vector_store.save()
        logger.info("Vector store reset.")

    # ───────── Stats ─────────

    @property
    def num_chunks(self) -> int:
        return self.vector_store.num_chunks

    def get_document_list(self) -> List[str]:
        """Return list of unique indexed document names."""
        if not hasattr(self.vector_store, "chunks"):
            return []
        return list({c.document_name for c in self.vector_store.chunks})