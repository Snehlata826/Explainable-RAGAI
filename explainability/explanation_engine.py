"""
Explainability engine.

Computes a confidence score and assembles the structured RAG response
including answer, sources, evidence snippets, and similarity scores.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List

from config.settings import HIGH_CONFIDENCE, MEDIUM_CONFIDENCE
from monitoring.logger import get_logger
from retrieval.context_retriever import RetrievedContext

logger = get_logger(__name__)

_SNIPPET_LENGTH = 300
MAX_SOURCES = 3


# ── Response data models ───────────────────────────────────────────────────

@dataclass
class SourceEvidence:
    """Evidence record for a single retrieved chunk."""

    document: str
    score: float
    snippet: str
    chunk_index: int


@dataclass
class ExplainedResponse:
    """Complete explainable RAG response."""

    answer: str
    confidence: float
    confidence_label: str
    hallucination_risk: str
    sources: List[SourceEvidence] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "confidence": round(self.confidence, 4),
            "confidence_label": self.confidence_label,
            "hallucination_risk": self.hallucination_risk,
            "sources": [
                {
                    "document": s.document,
                    "score": round(s.score, 4),
                    "snippet": s.snippet,
                    "chunk_index": s.chunk_index,
                }
                for s in self.sources
            ],
        }


# ── Confidence calculation ─────────────────────────────────────────────────

def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    else:
        z = math.exp(x)
        return z / (1 + z)


def _compute_confidence(contexts: List[RetrievedContext]) -> float:
    """
    Compute a confidence score in [0, 1].

    Strategy:
    - Apply sigmoid to reranker scores
    - Normalize scores
    - Weighted average (top contexts more important)
    """

    if not contexts:
        return 0.0

    scores = [_sigmoid(ctx.rerank_score) for ctx in contexts]

    # Normalize scores
    max_score = max(scores)
    min_score = min(scores)

    if max_score != min_score:
        scores = [(s - min_score) / (max_score - min_score) for s in scores]

    # Weighted average
    weights = [1.0 / (i + 1) for i in range(len(scores))]
    total_weight = sum(weights)

    confidence = sum(s * w for s, w in zip(scores, weights)) / total_weight

    return round(min(confidence, 1.0), 4)


def _label(confidence: float) -> str:
    """Convert confidence score to label."""

    if confidence >= HIGH_CONFIDENCE:
        return "HIGH"
    if confidence >= MEDIUM_CONFIDENCE:
        return "MEDIUM"
    return "LOW"


def _hallucination_risk(confidence: float) -> str:
    """Estimate hallucination risk."""

    if confidence < 0.3:
        return "HIGH"
    if confidence < 0.6:
        return "MEDIUM"
    return "LOW"


def _make_snippet(text: str, length: int = _SNIPPET_LENGTH) -> str:
    """Return a centered snippet from the retrieved chunk."""

    text = text.replace("\n", " ").strip()

    if len(text) <= length:
        return text

    start = max(0, len(text) // 2 - length // 2)
    snippet = text[start : start + length]

    return snippet.rsplit(" ", 1)[0] + " …"


# ── Public API ─────────────────────────────────────────────────────────────

def build_explained_response(
    answer: str,
    contexts: List[RetrievedContext],
) -> ExplainedResponse:
    """
    Assemble the full explainable response.

    Args:
        answer: Generated answer string
        contexts: Reranked context chunks

    Returns:
        ExplainedResponse
    """

    confidence = _compute_confidence(contexts)
    label = _label(confidence)
    risk = _hallucination_risk(confidence)

    sources: List[SourceEvidence] = []

    for ctx in contexts[:MAX_SOURCES]:

        sources.append(
            SourceEvidence(
                document=ctx.document_name,
                score=round(_sigmoid(ctx.rerank_score), 4),
                snippet=_make_snippet(ctx.text),
                chunk_index=ctx.chunk.chunk_index,
            )
        )

    response = ExplainedResponse(
        answer=answer,
        confidence=confidence,
        confidence_label=label,
        hallucination_risk=risk,
        sources=sources,
    )

    logger.info(
        "Explanation built",
        extra={
            "confidence": confidence,
            "confidence_label": label,
            "hallucination_risk": risk,
            "sources": len(sources),
        },
    )

    return response