"""
RAG Pipeline Orchestrator — Refactored.

Changes vs original:
- Added query rewriting via LLM before retrieval
- Improved multi-document reasoning (explicit doc names in prompt)
- Deduplication, threshold filtering, context compression retained
"""

from __future__ import annotations

import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from config.settings import DATA_DIR, SIMILARITY_THRESHOLD
from embeddings.embedding_generator import embed_texts
from explainability.explanation_engine import ExplainedResponse, build_explained_response
from generation.answer_generator import generate_answer
from generation.llm_client import generate
from ingestion.document_processor import DocumentChunk, process_document
from monitoring.logger import get_logger, log_query_event
from retrieval.context_retriever import ContextRetriever
from vector_store.faiss_store import FAISSVectorStore

logger = get_logger(__name__)


# ─────────────────────── Query Rewriting ───────────────────────

REWRITE_PROMPT = """Rewrite the following user question to be more specific and retrieval-friendly.
Return ONLY the rewritten question, nothing else.

Original question: {question}

Rewritten question:"""


def rewrite_query(question: str) -> str:
    """
    Use the LLM to rewrite the user query for better retrieval.
    Falls back to the original question on any failure.
    """
    try:
        rewritten = generate(REWRITE_PROMPT.format(question=question), max_tokens=80)
        rewritten = rewritten.strip().strip('"').strip("'")
        if rewritten and len(rewritten) > 5:
            logger.info(f"Query rewritten: '{question}' → '{rewritten}'")
            return rewritten
    except Exception as exc:
        logger.warning(f"Query rewrite failed, using original: {exc}")
    return question


# ─────────────────────── Query Expansion ───────────────────────

def expand_query(question: str) -> List[str]:
    """Generate multiple search variants for better recall."""
    base = question.strip()
    return [
        base,
        f"What is {base}",
        f"{base} methodology approach",
        f"{base} results findings",
    ]


# ─────────────────────── Context Compression ───────────────────────

def compress_context(question: str, contexts) -> list:
    keywords = set(re.findall(r"\b\w{4,}\b", question.lower()))
    compressed = []
    for ctx in contexts:
        sentences = re.split(r"(?<=[.!?])\s+", ctx.text)
        relevant = [s for s in sentences if keywords & set(s.lower().split())]
        if relevant:
            ctx.chunk.text = " ".join(relevant)
        compressed.append(ctx)
    return compressed


# ─────────────────────── Multi-Document Grouping ───────────────────────

def group_contexts_by_document(contexts) -> Dict[str, list]:
    groups: Dict[str, list] = defaultdict(list)
    for ctx in contexts:
        groups[ctx.document_name].append(ctx)
    return dict(groups)


# ─────────────────────── RAG Pipeline ───────────────────────

class RAGPipeline:

    def __init__(self) -> None:
        self.vector_store = FAISSVectorStore()
        self.retriever = ContextRetriever(self.vector_store)
        self.vector_store.load()

    def _document_exists(self, document_name: str) -> bool:
        return any(
            getattr(c, "document_name", None) == document_name
            for c in self.vector_store.chunks
        )

    def ingest(self, file_path: Path) -> int:
        logger.info(f"Ingesting: {file_path.name}")
        if self._document_exists(file_path.name):
            logger.warning(f"{file_path.name} already indexed — skipping.")
            return 0
        chunks: List[DocumentChunk] = process_document(file_path)
        if not chunks:
            return 0
        embeddings = embed_texts([c.text for c in chunks])
        self.vector_store.add_chunks(chunks, embeddings)
        self.vector_store.save()
        logger.info(f"Ingested {len(chunks)} chunks from {file_path.name}")
        return len(chunks)

    def ingest_directory(self, directory: Path = DATA_DIR) -> int:
        total = 0
        for fp in sorted(directory.iterdir()):
            if fp.suffix.lower() in (".pdf", ".txt", ".md"):
                try:
                    total += self.ingest(fp)
                except Exception as exc:
                    logger.error(f"Failed to ingest {fp.name}: {exc}")
        return total

    def _deduplicate_contexts(self, contexts) -> list:
        seen = set()
        unique = []
        for ctx in contexts:
            key = (ctx.document_name, ctx.text[:120])
            if key not in seen:
                seen.add(key)
                unique.append(ctx)
        return unique

    def _filter_by_threshold(self, contexts, threshold: float = SIMILARITY_THRESHOLD) -> list:
        filtered = [ctx for ctx in contexts if ctx.retrieval_score >= threshold]
        if not filtered:
            logger.warning("No contexts passed threshold — returning top-3.")
            return contexts[:3]
        return filtered

    def query(self, question: str) -> ExplainedResponse:
        question = question.strip()
        start = time.perf_counter()
        logger.info(f"Query: '{question}'")

        # 1. Rewrite query for better retrieval
        rewritten = rewrite_query(question)

        # 2. Expand into variants
        queries = expand_query(rewritten)

        # 3. Hybrid retrieval
        all_contexts = []
        for q in queries:
            retrieved = self.retriever.retrieve(q, top_k_retrieval=10, top_k_rerank=3)
            all_contexts.extend(retrieved)

        # 4. Deduplicate + threshold filter
        contexts = self._deduplicate_contexts(all_contexts)
        contexts = self._filter_by_threshold(contexts)
        contexts = compress_context(question, contexts)
        contexts = contexts[:5]

        # 5. Group by document for multi-doc reasoning
        doc_groups = group_contexts_by_document(contexts)
        multi_doc = len(doc_groups) > 1

        # 6. Generate answer with explicit document names
        answer = generate_answer(
            question, contexts,
            doc_groups=doc_groups,
            multi_doc=multi_doc,
        )

        # 7. Build explainable response
        response = build_explained_response(answer, contexts)

        latency_ms = (time.perf_counter() - start) * 1000
        log_query_event(
            query=question,
            num_sources=len(response.sources),
            confidence=response.confidence,
            latency_ms=latency_ms,
        )
        logger.info(f"Answered in {latency_ms:.1f}ms | confidence={response.confidence}")
        return response

    def reset(self) -> None:
        self.vector_store.clear()
        self.vector_store.save()
        logger.info("Vector store reset.")

    @property
    def num_chunks(self) -> int:
        return self.vector_store.num_chunks

    def get_document_list(self) -> List[str]:
        return list({c.document_name for c in self.vector_store.chunks})