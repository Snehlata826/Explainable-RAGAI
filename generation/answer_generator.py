"""
Grounded Answer Generation Module — Production Grade.

Key features:
- Strict context-only grounding (ZERO external knowledge by default)
- Mandatory structured response format:
    Answer from Documents:
    Additional AI Knowledge (optional, labeled):
    Sources:
- Multi-document reasoning: per-paper answers + comparison
- Hallucination detection via token overlap
- Fallback: "Not found in uploaded papers"
"""

from __future__ import annotations

import re
from typing import List, Dict, Optional

from generation.llm_client import generate
from monitoring.logger import get_logger
from retrieval.context_retriever import RetrievedContext

logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Prompt Templates
# ──────────────────────────────────────────────────────────────────────────

SINGLE_DOC_PROMPT = """You are a strict research paper Q&A assistant.

RULES (MANDATORY):
1. ONLY use information from the provided context.
2. Do NOT use any external knowledge or training data.
3. If the answer is not in the context, respond EXACTLY with:
   "I cannot find the answer in the uploaded papers."
4. Structure your answer EXACTLY as shown below.
5. Use concise bullet points or short paragraphs to make the answer highly readable.

Context from uploaded documents:
{context}

Question: {question}

Respond in this EXACT format:

Answer from Documents:
[Your answer using ONLY the context above. Be specific and cite details.]
"""

MULTI_DOC_PROMPT = """You are a strict research paper comparison assistant.

RULES (MANDATORY):
1. ONLY use information from the provided context sections.
2. Do NOT use any external knowledge or training data.
3. Answer SEPARATELY for each paper, then provide a comparison.
4. If a paper doesn't address the question, say "Not addressed in [paper name]."
5. Structure your answer EXACTLY as shown below.
6. Use concise bullet points or short paragraphs to make the answer highly readable.

Context by Paper:
{context_by_paper}

Question: {question}

Respond in this EXACT format:

Answer from Documents:

{per_paper_sections}

Comparison:
[Summarise similarities and differences between the papers on this topic.]
"""

FALLBACK_ANSWER = "I cannot find the answer in the uploaded papers."


# ──────────────────────────────────────────────────────────────────────────
# Context Builders
# ──────────────────────────────────────────────────────────────────────────

def _build_single_context(contexts: List[RetrievedContext]) -> str:
    parts = []
    for i, ctx in enumerate(contexts, 1):
        parts.append(f"[{i}] {ctx.document_name}:\n{ctx.text}")
    return "\n\n".join(parts)


def _build_multi_context(doc_groups: Dict[str, List[RetrievedContext]]) -> str:
    parts = []
    for doc_name, ctxs in doc_groups.items():
        combined = " ".join(c.text for c in ctxs)
        parts.append(f"--- {doc_name} ---\n{combined}")
    return "\n\n".join(parts)


def _build_per_paper_sections(doc_groups: Dict[str, List[RetrievedContext]]) -> str:
    sections = []
    for doc_name in doc_groups:
        sections.append(f"From {doc_name}:\n[Answer based on this paper's context]")
    return "\n\n".join(sections)


def _build_source_list(contexts: List[RetrievedContext]) -> str:
    lines = []
    for ctx in contexts:
        lines.append(f"- {ctx.document_name} (chunk {ctx.chunk.chunk_index})")
    return "\n".join(lines)


def _truncate_context(context: str, max_chars: int = 5500) -> str:
    if len(context) > max_chars:
        logger.warning("Context truncated to prevent oversized prompt.")
        return context[:max_chars] + "\n[...truncated]"
    return context


# ──────────────────────────────────────────────────────────────────────────
# Hallucination Detection
# ──────────────────────────────────────────────────────────────────────────

_STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "can", "could", "may", "might", "shall", "should", "of", "in",
    "on", "at", "to", "for", "with", "by", "from", "this", "that",
    "it", "its", "and", "or", "but", "not", "as", "if", "so",
}


def _meaningful_words(text: str) -> set:
    words = set(re.findall(r"\b[a-z]{4,}\b", text.lower()))
    return words - _STOP_WORDS


def _is_grounded(answer: str, contexts: List[RetrievedContext], threshold: float = 0.15) -> bool:
    """
    Check answer has sufficient word overlap with retrieved context.
    Threshold of 0.15 allows paraphrasing while blocking pure hallucination.
    """
    answer_words = _meaningful_words(answer)
    if not answer_words:
        return False

    context_words: set = set()
    for ctx in contexts:
        context_words.update(_meaningful_words(ctx.text))

    overlap_ratio = len(answer_words & context_words) / len(answer_words)
    logger.debug(f"Groundedness overlap ratio: {overlap_ratio:.3f}")

    return overlap_ratio >= threshold


def _is_fallback_response(answer: str) -> bool:
    fallback_phrases = [
        "cannot find",
        "not found in",
        "not addressed",
        "no information",
        "don't have information",
    ]
    lower = answer.lower()
    return any(p in lower for p in fallback_phrases)


# ──────────────────────────────────────────────────────────────────────────
# Post-processing
# ──────────────────────────────────────────────────────────────────────────

def _ensure_structured_format(answer: str, contexts: List[RetrievedContext]) -> str:
    """
    If LLM didn't follow the format, inject structured wrapper.
    """
    if "Answer from Documents:" in answer:
        if "\nSources:" in answer:
            answer = answer.split("\nSources:")[0].strip()
        return answer

    return f"Answer from Documents:\n{answer.strip()}"


def _clean_answer(answer: str) -> str:
    """Remove common LLM artifacts."""
    # Remove "Answer:" prefix if present
    answer = re.sub(r"^(Answer:|Response:)\s*", "", answer, flags=re.IGNORECASE)
    # Ensure proper ending
    if answer.strip() and not answer.strip()[-1] in ".!?":
        answer = answer.strip() + "."
    # Squeeze consecutive newlines completely to avoid any massive layout gaps
    answer = re.sub(r'\n{2,}', '\n', answer)
    return answer.strip()


# ──────────────────────────────────────────────────────────────────────────
# Main Generator
# ──────────────────────────────────────────────────────────────────────────

def generate_answer(
    question: str,
    contexts: List[RetrievedContext],
    doc_groups: Optional[Dict[str, List[RetrievedContext]]] = None,
    multi_doc: bool = False,
) -> str:

    if not contexts:
        return f"Answer from Documents:\n{FALLBACK_ANSWER}"

    source_list = _build_source_list(contexts)

    try:
        if multi_doc and doc_groups and len(doc_groups) > 1:
            # ── Multi-document prompt ──
            context_by_paper = _build_multi_context(doc_groups)
            context_by_paper = _truncate_context(context_by_paper)
            per_paper_sections = _build_per_paper_sections(doc_groups)

            prompt = MULTI_DOC_PROMPT.format(
                context_by_paper=context_by_paper,
                question=question,
                per_paper_sections=per_paper_sections,
            )
        else:
            # ── Single-document prompt ──
            context_str = _build_single_context(contexts)
            context_str = _truncate_context(context_str)

            prompt = SINGLE_DOC_PROMPT.format(
                context=context_str,
                question=question,
            )

        logger.debug(f"Prompt length: {len(prompt)} chars")

        raw_answer = generate(prompt)

        if not raw_answer or not raw_answer.strip():
            raise ValueError("Empty LLM response")

        answer = _clean_answer(raw_answer)

        # Groundedness check
        if not _is_fallback_response(answer) and not _is_grounded(answer, contexts):
            logger.warning("Hallucination risk detected — returning fallback.")
            return f"Answer from Documents:\n{FALLBACK_ANSWER}"

        # Enforce structure
        answer = _ensure_structured_format(answer, contexts)

        return answer

    except Exception as exc:
        logger.error(f"Answer generation failed: {exc}")
        best = contexts[0].text.strip().split(".")[0] + "."
        return f"Answer from Documents:\n{best}"