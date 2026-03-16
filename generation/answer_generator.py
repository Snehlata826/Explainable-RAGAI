"""
Grounded answer generation module.

Constructs context-grounded prompts and calls the LLM.
Includes hallucination protection and evidence verification.
"""

from __future__ import annotations

from typing import List
import re

from generation.llm_client import generate
from monitoring.logger import get_logger
from retrieval.context_retriever import RetrievedContext

logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────
# Prompt Template
# ──────────────────────────────────────────────────────────────

PROMPT = """
You are an AI assistant answering questions about documents.

Use ONLY the provided context.

Explain clearly in simple language.

Write a COMPLETE answer in 3 to 6 sentences.
Finish the explanation properly.

If the answer is not in the context, say:
"I cannot find the answer in the documents."

Context:
{context}

Question:
{question}

Answer:
"""


# ──────────────────────────────────────────────────────────────
# Context formatter
# ──────────────────────────────────────────────────────────────

def _build_context_str(contexts: List[RetrievedContext]) -> str:
    """
    Format retrieved chunks into numbered context sections.
    """

    parts = []

    for i, ctx in enumerate(contexts, 1):
        parts.append(
            f"[{i}] Source: {ctx.document_name}\n{ctx.text}"
        )

    return "\n\n".join(parts)


# ──────────────────────────────────────────────────────────────
# Context length protection
# ──────────────────────────────────────────────────────────────

def _truncate_context(context: str, max_chars: int = 6000) -> str:
    """
    Prevent prompt from becoming too large.
    """

    if len(context) > max_chars:
        logger.warning("Context truncated to prevent oversized prompt.")
        return context[:max_chars]

    return context


# ──────────────────────────────────────────────────────────────
# Hallucination detector
# ──────────────────────────────────────────────────────────────

def _answer_supported(answer: str, contexts: List[RetrievedContext]) -> bool:
    """
    Check if answer shares meaningful overlap with context.
    Prevent hallucinated answers.
    """

    answer_words = set(re.findall(r"\w+", answer.lower()))
    context_words = set()

    for ctx in contexts:
        context_words.update(re.findall(r"\w+", ctx.text.lower()))

    overlap = answer_words.intersection(context_words)

    if len(answer_words) == 0:
        return False

    overlap_ratio = len(overlap) / len(answer_words)

    # Slightly relaxed threshold to allow paraphrasing
    return overlap_ratio > 0.05


# ──────────────────────────────────────────────────────────────
# Main answer generation
# ──────────────────────────────────────────────────────────────

def generate_answer(question: str, contexts: List[RetrievedContext]) -> str:

    if not contexts:
        return "No relevant information found in the documents."

    # Build context
    context_str = _build_context_str(contexts)

    # Prevent oversized prompts
    context_str = _truncate_context(context_str)

    # Build final prompt
    prompt = PROMPT.format(
        context=context_str,
        question=question
    )

    # DEBUG: show prompt
    print("\n===== PROMPT SENT TO MODEL =====\n")
    print(prompt[:1500])
    print("\n================================\n")

    try:

        answer = generate(prompt)

        if not answer:
            raise ValueError("Empty LLM response")

        answer = answer.strip()

        # Ensure sentence completion
        if not answer.endswith((".", "!", "?")):
            answer += "."

        # Detect hallucinations
        if not _answer_supported(answer, contexts):
            logger.warning("Possible hallucination detected.")
            return "I cannot find the answer in the documents."

        # Protect against extremely short answers
        if len(answer.split()) < 8:
            logger.warning("Answer too short, using fallback.")
            best_context = contexts[0].text.strip()
            return best_context.split(".")[0] + "."

        return answer

    except Exception as e:

        logger.error(f"LLM generation failed: {e}")

        # Fallback answer from best chunk
        best_context = contexts[0].text.strip()

        return best_context.split(".")[0] + "."