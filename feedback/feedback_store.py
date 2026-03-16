"""
Simple JSON-based feedback store.

Persists user ratings and comments for offline analysis and RLHF prep.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from config.settings import FEEDBACK_DIR
from monitoring.logger import get_logger

logger = get_logger(__name__)

_FEEDBACK_FILE = FEEDBACK_DIR / "feedback.jsonl"


def store_feedback(
    query: str,
    answer: str,
    rating: int,
    comment: Optional[str] = None,
) -> str:
    """
    Append a feedback record to the JSONL store.

    Args:
        query: Original user question.
        answer: Answer that was shown to the user.
        rating: Integer score 1-5 (1 = very bad, 5 = very good).
        comment: Optional free-text comment.

    Returns:
        Feedback record ID.
    """
    if not 1 <= rating <= 5:
        raise ValueError(f"Rating must be between 1 and 5, got {rating}")

    record_id = str(uuid.uuid4())
    record = {
        "id": record_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "answer": answer,
        "rating": rating,
        "comment": comment or "",
    }

    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    with open(_FEEDBACK_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    logger.info(f"Feedback stored: id={record_id}, rating={rating}")
    return record_id


def load_all_feedback() -> list:
    """
    Load and return all feedback records.

    Returns:
        List of feedback dicts.
    """
    if not _FEEDBACK_FILE.exists():
        return []
    records = []
    with open(_FEEDBACK_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
