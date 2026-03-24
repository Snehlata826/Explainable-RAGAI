"""
Chunking Strategy Comparison Tool.

Compares different chunk sizes and overlaps on your documents.
Measures: chunk count, avg token count, coverage, and retrieval impact.

Usage:
    python -m evaluation.chunking_comparison --file data/raw_docs/paper.pdf
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

from ingestion.document_processor import build_chunks, extract_text, clean_text, split_sections
from embeddings.embedding_generator import embed_texts
from monitoring.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Chunking Strategies to Compare
# ─────────────────────────────────────────────────────────────────────────────

STRATEGIES: List[Tuple[int, int, str]] = [
    # (chunk_size, overlap, label)
    (200,  50,  "Small (200/50)"),
    (350, 100,  "Medium (350/100) — original"),
    (400, 100,  "Medium+ (400/100)"),
    (500, 150,  "Large (500/150)"),
    (600, 200,  "X-Large (600/200)"),
    (800, 200,  "Research-Optimized (800/200)"),
]

# ─────────────────────────────────────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ChunkingResult:
    label: str
    chunk_size: int
    overlap: int
    num_chunks: int
    avg_tokens: float
    min_tokens: int
    max_tokens: int
    coverage_pct: float         # % of original text covered
    embedding_time_ms: float    # time to embed all chunks
    semantic_density: float     # avg pairwise cosine sim between consecutive chunks

    def summary_row(self) -> str:
        return (
            f"  {self.label:<30} "
            f"chunks={self.num_chunks:<5} "
            f"avg_tok={self.avg_tokens:<7.1f} "
            f"coverage={self.coverage_pct:.1f}% "
            f"density={self.semantic_density:.3f} "
            f"embed_ms={self.embedding_time_ms:.0f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Analysis Functions
# ─────────────────────────────────────────────────────────────────────────────

def _token_counts(chunks: List[str]) -> List[int]:
    """Approximate token count via whitespace split."""
    return [len(c.split()) for c in chunks]


def _coverage(chunks: List[str], original_text: str) -> float:
    """
    Fraction of unique original words covered by chunks.
    Higher = less information loss.
    """
    original_words = set(original_text.lower().split())
    chunk_words: set = set()
    for c in chunks:
        chunk_words.update(c.lower().split())
    if not original_words:
        return 0.0
    return round(len(original_words & chunk_words) / len(original_words) * 100, 2)


def _semantic_density(chunks: List[str]) -> float:
    """
    Average cosine similarity between consecutive chunk pairs.

    High density: chunks are semantically coherent (good overlap).
    Low density: chunks jump topics (too large or bad boundaries).
    """
    if len(chunks) < 2:
        return 0.0

    try:
        # Sample at most 20 consecutive pairs for speed
        sample_chunks = chunks[:21]
        embeddings = embed_texts(sample_chunks)

        sims = []
        for i in range(len(embeddings) - 1):
            a = embeddings[i]
            b = embeddings[i + 1]
            cos_sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
            sims.append(cos_sim)

        return round(float(np.mean(sims)), 4)

    except Exception as exc:
        logger.warning(f"Semantic density computation failed: {exc}")
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Main Comparison Runner
# ─────────────────────────────────────────────────────────────────────────────

def compare_strategies(file_path: Path) -> List[ChunkingResult]:
    """
    Run all chunking strategies on a document and return comparison results.
    """
    logger.info(f"Comparing chunking strategies on: {file_path.name}")

    raw_text = extract_text(file_path)
    cleaned = clean_text(raw_text)
    sections = split_sections(cleaned)

    results = []

    for chunk_size, overlap, label in STRATEGIES:
        all_chunks: List[str] = []
        for section in sections:
            all_chunks.extend(build_chunks(section, chunk_size=chunk_size, chunk_overlap=overlap))

        if not all_chunks:
            continue

        token_counts = _token_counts(all_chunks)
        coverage = _coverage(all_chunks, cleaned)

        # Time embedding
        t0 = time.perf_counter()
        _semantic_density_score = _semantic_density(all_chunks)
        embed_time_ms = (time.perf_counter() - t0) * 1000

        result = ChunkingResult(
            label=label,
            chunk_size=chunk_size,
            overlap=overlap,
            num_chunks=len(all_chunks),
            avg_tokens=round(float(np.mean(token_counts)), 1),
            min_tokens=min(token_counts),
            max_tokens=max(token_counts),
            coverage_pct=coverage,
            embedding_time_ms=round(embed_time_ms, 1),
            semantic_density=_semantic_density_score,
        )
        results.append(result)
        logger.info(f"  Strategy '{label}': {len(all_chunks)} chunks")

    return results


def print_comparison_table(results: List[ChunkingResult]) -> None:
    print("\n" + "=" * 90)
    print("  CHUNKING STRATEGY COMPARISON")
    print("=" * 90)
    print(f"  {'Strategy':<30} {'Chunks':<8} {'Avg Tok':<10} {'Coverage':<12} {'Sem. Density':<14} {'Embed ms'}")
    print("-" * 90)
    for r in results:
        print(r.summary_row())

    # Recommendation
    if results:
        best = max(results, key=lambda x: x.semantic_density * (x.coverage_pct / 100))
        print("\n" + "-" * 90)
        print(f"  ✓ Recommended strategy: {best.label}")
        print(f"    Reasoning: best balance of semantic density ({best.semantic_density:.3f}) "
              f"and coverage ({best.coverage_pct:.1f}%)")
    print("=" * 90 + "\n")


def save_comparison_json(results: List[ChunkingResult], output_path: Path) -> None:
    """Save results to JSON for downstream analysis."""
    data = [
        {
            "label": r.label,
            "chunk_size": r.chunk_size,
            "overlap": r.overlap,
            "num_chunks": r.num_chunks,
            "avg_tokens": r.avg_tokens,
            "min_tokens": r.min_tokens,
            "max_tokens": r.max_tokens,
            "coverage_pct": r.coverage_pct,
            "embedding_time_ms": r.embedding_time_ms,
            "semantic_density": r.semantic_density,
        }
        for r in results
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Comparison saved to {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare chunking strategies")
    parser.add_argument("--file", required=True, help="Path to PDF or TXT file")
    parser.add_argument("--output", default="data/chunking_comparison.json", help="Output JSON path")
    args = parser.parse_args()

    file_path = Path(args.file)
    if not file_path.exists():
        print(f"File not found: {file_path}")
        exit(1)

    results = compare_strategies(file_path)
    print_comparison_table(results)
    save_comparison_json(results, Path(args.output))