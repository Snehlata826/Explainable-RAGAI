"""
FastAPI backend for GenAI RAG Copilot.

Endpoints:
  POST /upload   – ingest a document
  POST /query    – ask a question
  POST /feedback – store user feedback
  GET  /health   – health check
  GET  /stats    – index statistics
"""


from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from api.rag_pipeline import RAGPipeline
from config.settings import DATA_DIR
from feedback.feedback_store import store_feedback
from generation.llm_client import is_hf_available
from monitoring.logger import get_logger

logger = get_logger(__name__)

# ── App setup ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="GenAI RAG Copilot",
    description="Explainable RAG system for document Q&A with hybrid retrieval.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singleton pipeline – loaded once on startup
_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    global _pipeline

    if _pipeline is None:
        print("⚡ Initializing pipeline...")
        _pipeline = RAGPipeline()

    return _pipeline


def get_pipeline() -> RAGPipeline:
    if _pipeline is None:
        raise RuntimeError("Pipeline not initialised.")
    return _pipeline


# ── Request / Response schemas ─────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, description="User question")


class SourceModel(BaseModel):
    document: str
    score: float
    snippet: str
    chunk_index: int


class QueryResponse(BaseModel):
    answer: str


class FeedbackRequest(BaseModel):
    query: str
    answer: str
    rating: int = Field(..., ge=1, le=5)
    comment: Optional[str] = None


class FeedbackResponse(BaseModel):
    feedback_id: str
    message: str


class UploadResponse(BaseModel):
    filename: str
    chunks_indexed: int
    message: str


class HealthResponse(BaseModel):
    status: str
    hf_available: bool
    indexed_chunks: int


# ── Endpoints ──────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """Return service health and readiness."""

    pipeline = get_pipeline()

    try:
        hf_ok = is_hf_available()
    except Exception:
        hf_ok = False

    return HealthResponse(
        status="ok",
        hf_available=hf_ok,
        indexed_chunks=pipeline.num_chunks,
    )


@app.get("/stats", tags=["System"])
async def stats() -> dict:
    """Return index statistics."""

    pipeline = get_pipeline()

    return {
        "indexed_chunks": pipeline.num_chunks,
    }


@app.post("/upload", response_model=UploadResponse, tags=["Documents"])
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    """
    Upload and ingest a PDF or TXT document.
    """

    allowed = {".pdf", ".txt", ".md"}

    suffix = Path(file.filename or "").suffix.lower()

    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {allowed}",
        )

    dest = DATA_DIR / (file.filename or "upload")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    try:
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)

    except Exception as exc:
        logger.error(f"File save error: {exc}")
        raise HTTPException(status_code=500, detail="Failed to save file.")

    pipeline = get_pipeline()

    try:
        chunks_added = pipeline.ingest(dest)

    except Exception as exc:
        logger.error(f"Ingestion error: {exc}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")

    return UploadResponse(
        filename=file.filename or "unknown",
        chunks_indexed=chunks_added,
        message=f"Successfully indexed {chunks_added} chunks.",
    )


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query(request: QueryRequest) -> QueryResponse:

    pipeline = get_pipeline()

    if pipeline.num_chunks == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents indexed yet. Please upload documents first.",
        )

    start = time.perf_counter()

    try:
        response = pipeline.query(request.question)

    except RuntimeError as exc:
        logger.error(f"Query error: {exc}")
        raise HTTPException(status_code=503, detail=str(exc))

        latency_ms = (time.perf_counter() - start) * 1000

    return QueryResponse(
        answer=response.answer
    )

@app.post("/query-debug", tags=["RAG"])
async def query_debug(request: QueryRequest):

    pipeline = get_pipeline()

    if pipeline.num_chunks == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents indexed yet. Please upload documents first.",
        )

    start = time.perf_counter()

    try:
        response = pipeline.query(request.question)

    except RuntimeError as exc:
        logger.error(f"Query error: {exc}")
        raise HTTPException(status_code=503, detail=str(exc))

    latency_ms = (time.perf_counter() - start) * 1000

    return {
        "answer": response.answer,
        "confidence": response.confidence,
        "confidence_label": response.confidence_label,
        "sources": [
            {
                "document": s.document,
                "score": s.score,
                "snippet": s.snippet,
                "chunk_index": s.chunk_index,
            }
            for s in response.sources
        ],
        "latency_ms": round(latency_ms, 1),
    }

@app.post("/feedback", response_model=FeedbackResponse, tags=["Feedback"])
async def feedback(request: FeedbackRequest) -> FeedbackResponse:
    """
    Store user feedback.
    """

    try:
        fid = store_feedback(
            query=request.query,
            answer=request.answer,
            rating=request.rating,
            comment=request.comment,
        )

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return FeedbackResponse(
        feedback_id=fid,
        message="Thank you for your feedback!",
    )


@app.delete("/reset", tags=["System"])
async def reset_index() -> dict:
    """Clear the entire vector store."""

    get_pipeline().reset()

    return {
        "message": "Vector store cleared."
    }

@app.get("/")
def root():
    return {"status": "running"}
    
    
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api.main:app", host="0.0.0.0", port=port)