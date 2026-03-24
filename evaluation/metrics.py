"""
RAG Evaluation Metrics — Production Grade.

Implements:
- Retrieval Precision@K
- Retrieval Recall@K
- Groundedness Score (answer vs. context overlap)
- Hallucination Rate
- Answer Relevance Score
- Context Utilization Rate
- Optional: RAGAS-compatible scoring structure

Usage:
    from evaluation.metrics import RAGEvaluator
    evaluator = RAGEvaluator()
    results = evaluator.evaluate(question, answer, contexts, ground_truth_docs=[])
"""

from __future__ import annotations

import re
import math
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Dict
from datetime import datetime, timezone

from monitoring.logger import get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Stop words for meaningful overlap computation
# ─────────────────────────────────────────────────────────────────────────────

_STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "have",
    "has", "had", "do", "does", "did", "will", "would", "can", "could",
    "may", "might", "shall", "should", "of", "in", "on", "at", "to",
    "for", "with", "by", "from", "this", "that", "it", "its", "and",
    "or", "but", "not", "as", "if", "so", "we", "our", "their", "they",
    "which", "who", "what", "how", "when", "where", "also", "than",
}


def _meaningful_tokens(text: str) -> set:
    words = set(re.findall(r"\b[a-z]{4,}\b", text.lower()))
    return words - _STOP_WORDS


# ─────────────────────────────────────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvaluationResult:
    query: str
    answer: str
    groundedness_score: float       # 0–1: how much of answer is in context
    hallucination_rate: float       # 0–1: fraction of answer NOT in context
    answer_relevance: float         # 0–1: how much of answer is relevant to query
    context_utilization: float      # 0–1: how much of context was used in answer
    retrieval_precision: float      # 0–1: fraction of retrieved docs that are relevant
    retrieval_recall: float         # 0–1: fraction of relevant docs retrieved
    f1_retrieval: float             # harmonic mean of precision + recall
    confidence_score: float         # from reranker
    overall_score: float            # weighted aggregate
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        lines = [
            f"Query: {self.query[:80]}",
            f"  Groundedness:        {self.groundedness_score:.3f}",
            f"  Hallucination Rate:  {self.hallucination_rate:.3f}",
            f"  Answer Relevance:    {self.answer_relevance:.3f}",
            f"  Context Utilization: {self.context_utilization:.3f}",
            f"  Retrieval Precision: {self.retrieval_precision:.3f}",
            f"  Retrieval Recall:    {self.retrieval_recall:.3f}",
            f"  F1 Retrieval:        {self.f1_retrieval:.3f}",
            f"  Overall Score:       {self.overall_score:.3f}",
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Core Metric Functions
# ─────────────────────────────────────────────────────────────────────────────

def compute_groundedness(answer: str, context_texts: List[str]) -> float:
    """
    Groundedness = fraction of meaningful answer words found in context.

    Formula:
        G = |answer_words ∩ context_words| / |answer_words|

    A fully grounded answer uses ONLY context vocabulary.
    Values < 0.15 suggest potential hallucination.
    """
    answer_words = _meaningful_tokens(answer)
    if not answer_words:
        return 0.0

    context_words: set = set()
    for text in context_texts:
        context_words.update(_meaningful_tokens(text))

    overlap = answer_words & context_words
    score = len(overlap) / len(answer_words)
    return round(min(score, 1.0), 4)


def compute_hallucination_rate(answer: str, context_texts: List[str]) -> float:
    """
    Hallucination Rate = fraction of answer words NOT found in context.

    Formula:
        H = 1 - G
        where G = groundedness score

    H > 0.5 = high hallucination risk.
    H < 0.2 = well grounded.
    """
    return round(1.0 - compute_groundedness(answer, context_texts), 4)


def compute_answer_relevance(answer: str, question: str) -> float:
    """
    Answer Relevance = word overlap between answer and question.

    Formula:
        R = |answer_words ∩ question_words| / |question_words|

    Measures whether the answer addresses what was asked.
    """
    answer_words = _meaningful_tokens(answer)
    question_words = _meaningful_tokens(question)

    if not question_words:
        return 0.0

    overlap = answer_words & question_words
    score = len(overlap) / len(question_words)
    return round(min(score, 1.0), 4)


def compute_context_utilization(answer: str, context_texts: List[str]) -> float:
    """
    Context Utilization = fraction of context words used in the answer.

    Formula:
        U = |answer_words ∩ context_words| / |context_words|

    Low utilization means the context was retrieved but ignored.
    """
    answer_words = _meaningful_tokens(answer)

    context_words: set = set()
    for text in context_texts:
        context_words.update(_meaningful_tokens(text))

    if not context_words:
        return 0.0

    overlap = answer_words & context_words
    score = len(overlap) / len(context_words)
    return round(min(score, 1.0), 4)


def compute_retrieval_precision(
    retrieved_doc_names: List[str],
    relevant_doc_names: List[str],
) -> float:
    """
    Retrieval Precision@K = |relevant ∩ retrieved| / |retrieved|

    What fraction of retrieved documents are actually relevant?
    High precision → few irrelevant chunks retrieved.

    Args:
        retrieved_doc_names: Documents returned by retrieval pipeline.
        relevant_doc_names: Ground-truth relevant documents (if known).
    """
    if not retrieved_doc_names:
        return 0.0
    if not relevant_doc_names:
        # No ground truth: use retrieval score heuristic (all retrieved = relevant)
        return 1.0

    retrieved_set = set(retrieved_doc_names)
    relevant_set = set(relevant_doc_names)
    intersection = retrieved_set & relevant_set
    return round(len(intersection) / len(retrieved_set), 4)


def compute_retrieval_recall(
    retrieved_doc_names: List[str],
    relevant_doc_names: List[str],
) -> float:
    """
    Retrieval Recall@K = |relevant ∩ retrieved| / |relevant|

    What fraction of relevant documents were actually retrieved?
    High recall → few relevant chunks missed.
    """
    if not relevant_doc_names:
        return 1.0
    if not retrieved_doc_names:
        return 0.0

    retrieved_set = set(retrieved_doc_names)
    relevant_set = set(relevant_doc_names)
    intersection = retrieved_set & relevant_set
    return round(len(intersection) / len(relevant_set), 4)


def compute_f1(precision: float, recall: float) -> float:
    """
    F1 = 2 * (P * R) / (P + R)

    Harmonic mean balancing precision and recall.
    """
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)


def compute_overall_score(
    groundedness: float,
    hallucination_rate: float,
    answer_relevance: float,
    context_utilization: float,
    retrieval_f1: float,
    confidence: float,
) -> float:
    """
    Weighted aggregate score.

    Weights (sum to 1.0):
        groundedness       : 0.30 — most critical (anti-hallucination)
        answer_relevance   : 0.25 — answer must address the question
        retrieval_f1       : 0.20 — good retrieval enables good answers
        confidence         : 0.15 — reranker confidence
        context_utilization: 0.10 — penalize ignoring context
    """
    return round(
        0.30 * groundedness
        + 0.25 * answer_relevance
        + 0.20 * retrieval_f1
        + 0.15 * confidence
        + 0.10 * context_utilization
        - 0.10 * hallucination_rate,  # penalty
        4,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main Evaluator
# ─────────────────────────────────────────────────────────────────────────────

class RAGEvaluator:
    """
    End-to-end RAG evaluation suite.

    Tracks all evaluation results in memory and optionally
    persists them to JSONL for offline analysis / RLHF.
    """

    def __init__(self, log_dir: Optional[Path] = None) -> None:
        self.results: List[EvaluationResult] = []
        self.log_dir = log_dir
        if log_dir:
            log_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(
        self,
        question: str,
        answer: str,
        context_texts: List[str],
        retrieved_doc_names: List[str],
        confidence: float = 0.0,
        relevant_doc_names: Optional[List[str]] = None,
        notes: str = "",
    ) -> EvaluationResult:
        """
        Run full evaluation for one query-answer pair.

        Args:
            question: User query.
            answer: Generated answer string.
            context_texts: List of raw chunk texts used in generation.
            retrieved_doc_names: Document names of retrieved chunks.
            confidence: Reranker confidence score (0–1).
            relevant_doc_names: Optional ground truth relevant docs.
            notes: Optional annotation.

        Returns:
            EvaluationResult dataclass with all metric scores.
        """
        ground = compute_groundedness(answer, context_texts)
        halluc = compute_hallucination_rate(answer, context_texts)
        relevance = compute_answer_relevance(answer, question)
        utilization = compute_context_utilization(answer, context_texts)
        precision = compute_retrieval_precision(retrieved_doc_names, relevant_doc_names or [])
        recall = compute_retrieval_recall(retrieved_doc_names, relevant_doc_names or [])
        f1 = compute_f1(precision, recall)
        overall = compute_overall_score(ground, halluc, relevance, utilization, f1, confidence)

        result = EvaluationResult(
            query=question,
            answer=answer[:500],
            groundedness_score=ground,
            hallucination_rate=halluc,
            answer_relevance=relevance,
            context_utilization=utilization,
            retrieval_precision=precision,
            retrieval_recall=recall,
            f1_retrieval=f1,
            confidence_score=confidence,
            overall_score=overall,
            notes=notes,
        )

        self.results.append(result)

        if self.log_dir:
            self._persist(result)

        logger.info(f"Eval: overall={overall:.3f} | groundedness={ground:.3f} | halluc={halluc:.3f}")

        return result

    def _persist(self, result: EvaluationResult) -> None:
        """Append result to JSONL log file."""
        log_file = self.log_dir / "eval_results.jsonl"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result.to_dict()) + "\n")

    def aggregate_stats(self) -> Dict[str, float]:
        """
        Compute mean scores across all evaluations.

        Returns:
            Dict with mean of each metric.
        """
        if not self.results:
            return {}

        n = len(self.results)
        fields = [
            "groundedness_score", "hallucination_rate", "answer_relevance",
            "context_utilization", "retrieval_precision", "retrieval_recall",
            "f1_retrieval", "confidence_score", "overall_score",
        ]
        return {
            f"mean_{field}": round(sum(getattr(r, field) for r in self.results) / n, 4)
            for field in fields
        }

    def load_from_log(self, log_dir: Path) -> None:
        """Load previously saved evaluation results from JSONL."""
        log_file = log_dir / "eval_results.jsonl"
        if not log_file.exists():
            return
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                self.results.append(EvaluationResult(**data))
        logger.info(f"Loaded {len(self.results)} eval results from {log_file}")

    def print_report(self) -> None:
        """Print aggregated evaluation report."""
        stats = self.aggregate_stats()
        print("\n" + "=" * 60)
        print("  RAG EVALUATION REPORT")
        print("=" * 60)
        print(f"  Total evaluations: {len(self.results)}")
        print("-" * 60)
        for k, v in stats.items():
            label = k.replace("mean_", "").replace("_", " ").title()
            bar = "█" * int(v * 20) + "░" * (20 - int(v * 20))
            print(f"  {label:<25} {bar} {v:.3f}")
        print("=" * 60 + "\n")