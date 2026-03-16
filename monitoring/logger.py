"""
Centralised logging and monitoring for GenAI RAG Copilot.
"""

import logging
import time
import json
from pathlib import Path
from functools import wraps
from typing import Callable, Any

from config.settings import LOG_DIR

# ── Logger factory ─────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger with both console and rotating file handlers.

    Args:
        name: Module name (use __name__).

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    log_file = LOG_DIR / "rag_copilot.log"
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ── Latency decorator ──────────────────────────────────────────────────────

def log_latency(logger: logging.Logger) -> Callable:
    """
    Decorator that logs the execution time of a function.

    Args:
        logger: Logger to write timing info to.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = (time.perf_counter() - start) * 1000
            logger.info(f"{func.__name__} completed in {elapsed:.1f} ms")
            return result
        return wrapper
    return decorator


# ── Query event logger ─────────────────────────────────────────────────────

_query_logger = get_logger("query_events")


def log_query_event(
    query: str,
    num_sources: int,
    confidence: float,
    latency_ms: float,
) -> None:
    """
    Log a structured query event as JSON for downstream analysis.

    Args:
        query: User question.
        num_sources: Number of sources returned.
        confidence: Model confidence score.
        latency_ms: End-to-end latency in milliseconds.
    """
    event = {
        "query": query,
        "num_sources": num_sources,
        "confidence": confidence,
        "latency_ms": round(latency_ms, 1),
    }
    _query_logger.info(json.dumps(event))
