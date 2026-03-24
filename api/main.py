"""
FastAPI backend — Production-Grade RAG Copilot.

Endpoints:
  POST /upload        – ingest a document
  POST /query         – ask a question (structured explainable response)
  POST /query-debug   – query with full metrics + sources
  POST /feedback      – store thumbs-up/down feedback
  POST /evaluate      – run evaluation metrics on a query-answer pair
  GET  /health        – health check
  GET  /stats         – index statistics
  GET  /documents     – list indexed documents
  DELETE /reset       – clear vector store
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from api.rag_pipeline import RAGPipeline
from config.settings import DATA_DIR, EVAL_DIR
from evaluation.metrics import RAGEvaluator
from feedback.feedback_store import store_feedback, load_all_feedback
from generation.llm_client import is_hf_available
from monitoring.logger import get_logger

logger = get_logger(__name__)

# ── App setup ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="GenAI RAG Copilot",
    description="Production-grade explainable RAG for research paper Q&A.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipeline: RAGPipeline | None = None
_evaluator: RAGEvaluator | None = None


@app.on_event("startup")
async def startup_event() -> None:
    global _pipeline, _evaluator
    logger.info("Initialising RAG pipeline v2…")
    try:
        _pipeline = RAGPipeline()
        _evaluator = RAGEvaluator(log_dir=EVAL_DIR)
        logger.info(f"Pipeline ready. Indexed chunks: {_pipeline.num_chunks}")
    except Exception as exc:
        logger.error(f"Pipeline initialization failed: {exc}")
        raise


def get_pipeline() -> RAGPipeline:
    if _pipeline is None:
        raise RuntimeError("Pipeline not initialised.")
    return _pipeline


def get_evaluator() -> RAGEvaluator:
    if _evaluator is None:
        raise RuntimeError("Evaluator not initialised.")
    return _evaluator


# ── Schemas ────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3)


class SourceModel(BaseModel):
    document: str
    score: float
    snippet: str
    chunk_index: int


class QueryResponse(BaseModel):
    answer: str
    confidence: float
    confidence_label: str
    sources: List[SourceModel]


class FeedbackRequest(BaseModel):
    query: str
    answer: str
    rating: int = Field(..., ge=1, le=5)
    comment: Optional[str] = None
    thumbs_up: Optional[bool] = None  # alternative to numeric rating


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
    documents: List[str]


class EvalRequest(BaseModel):
    question: str
    answer: str
    context_texts: List[str]
    retrieved_doc_names: List[str]
    confidence: float = 0.0
    relevant_doc_names: Optional[List[str]] = None


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    pipeline = get_pipeline()
    try:
        hf_ok = is_hf_available()
    except Exception:
        hf_ok = False

    return HealthResponse(
        status="ok",
        hf_available=hf_ok,
        indexed_chunks=pipeline.num_chunks,
        documents=pipeline.get_document_list(),
    )


@app.get("/stats", tags=["System"])
async def stats() -> dict:
    pipeline = get_pipeline()
    evaluator = get_evaluator()
    agg = evaluator.aggregate_stats()
    return {
        "indexed_chunks": pipeline.num_chunks,
        "documents": pipeline.get_document_list(),
        "evaluation_stats": agg,
    }


@app.get("/documents", tags=["Documents"])
async def list_documents() -> dict:
    pipeline = get_pipeline()
    return {"documents": pipeline.get_document_list()}


@app.post("/upload", response_model=UploadResponse, tags=["Documents"])
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
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
    logger.info(f"Query latency: {latency_ms:.1f}ms")

    return QueryResponse(
        answer=response.answer,
        confidence=response.confidence,
        confidence_label=response.confidence_label,
        sources=[
            SourceModel(
                document=s.document,
                score=s.score,
                snippet=s.snippet,
                chunk_index=s.chunk_index,
            )
            for s in response.sources
        ],
    )


@app.post("/query-debug", tags=["RAG"])
async def query_debug(request: QueryRequest) -> dict:
    """Full debug response with evaluation metrics."""
    pipeline = get_pipeline()
    evaluator = get_evaluator()

    if pipeline.num_chunks == 0:
        raise HTTPException(status_code=400, detail="No documents indexed.")

    start = time.perf_counter()

    try:
        response = pipeline.query(request.question)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    latency_ms = (time.perf_counter() - start) * 1000

    # Run evaluation
    context_texts = [s.snippet for s in response.sources]
    retrieved_docs = [s.document for s in response.sources]

    eval_result = evaluator.evaluate(
        question=request.question,
        answer=response.answer,
        context_texts=context_texts,
        retrieved_doc_names=retrieved_docs,
        confidence=response.confidence,
    )

    return {
        "answer": response.answer,
        "confidence": response.confidence,
        "confidence_label": response.confidence_label,
        "hallucination_risk": response.hallucination_risk,
        "sources": [
            {
                "document": s.document,
                "score": s.score,
                "snippet": s.snippet,
                "chunk_index": s.chunk_index,
            }
            for s in response.sources
        ],
        "evaluation": eval_result.to_dict(),
        "latency_ms": round(latency_ms, 1),
    }


@app.post("/evaluate", tags=["Evaluation"])
async def evaluate(request: EvalRequest) -> dict:
    """Standalone evaluation endpoint for any query-answer pair."""
    evaluator = get_evaluator()

    result = evaluator.evaluate(
        question=request.question,
        answer=request.answer,
        context_texts=request.context_texts,
        retrieved_doc_names=request.retrieved_doc_names,
        confidence=request.confidence,
        relevant_doc_names=request.relevant_doc_names,
    )

    return result.to_dict()


@app.get("/evaluation/stats", tags=["Evaluation"])
async def evaluation_stats() -> dict:
    evaluator = get_evaluator()
    return {
        "total_evaluated": len(evaluator.results),
        "aggregate": evaluator.aggregate_stats(),
    }


@app.post("/feedback", response_model=FeedbackResponse, tags=["Feedback"])
async def feedback(request: FeedbackRequest) -> FeedbackResponse:
    # Convert thumbs_up to rating if provided
    rating = request.rating
    if request.thumbs_up is not None:
        rating = 5 if request.thumbs_up else 1

    try:
        fid = store_feedback(
            query=request.query,
            answer=request.answer,
            rating=rating,
            comment=request.comment,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return FeedbackResponse(
        feedback_id=fid,
        message="Thank you for your feedback!",
    )


@app.get("/feedback/all", tags=["Feedback"])
async def get_feedback() -> dict:
    records = load_all_feedback()
    return {
        "total": len(records),
        "records": records[-50:],  # last 50
    }


@app.delete("/reset", tags=["System"])
async def reset_index() -> dict:
    get_pipeline().reset()
    return {"message": "Vector store cleared."}