"""
FastAPI backend — Refactored RAG Copilot.

Endpoints:
  POST /upload        – ingest a document
  POST /query         – returns { answer } only (or full debug payload if DEBUG=True)
  POST /feedback      – store feedback
  GET  /health        – health check
  DELETE /reset       – clear vector store
"""

from __future__ import annotations

import os
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
from feedback.feedback_store import store_feedback
from generation.llm_client import is_hf_available
from monitoring.logger import get_logger

logger = get_logger(__name__)

# ── Debug mode (set DEBUG=true in env to expose sources/metrics) ───────────
DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

app = FastAPI(
    title="GenAI RAG Copilot",
    description="Production-grade explainable RAG for document Q&A.",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipeline: RAGPipeline | None = None
_evaluator: RAGEvaluator | None = None


import asyncio

async def load_models_bg():
    global _pipeline, _evaluator
    logger.info("Initializing RAG pipeline in the background...")
    try:
        loop = asyncio.get_running_loop()
        
        # Preload models to avoid timeout on first request
        from embeddings.embedding_generator import _get_model
        await loop.run_in_executor(None, _get_model)
        
        from retrieval.reranker import _get_reranker
        await loop.run_in_executor(None, _get_reranker)

        _pipeline_temp = await loop.run_in_executor(None, RAGPipeline)
        _evaluator_temp = await loop.run_in_executor(None, lambda: RAGEvaluator(log_dir=EVAL_DIR))
        
        _pipeline = _pipeline_temp
        _evaluator = _evaluator_temp
        logger.info(f"Pipeline ready. Indexed chunks: {_pipeline.num_chunks}")
    except Exception as e:
        logger.error(f"Error initializing pipeline: {e}")

@app.on_event("startup")
async def startup_event() -> None:
    asyncio.create_task(load_models_bg())

def get_pipeline() -> RAGPipeline:
    if _pipeline is None:
        raise RuntimeError("Pipeline is still initializing. Please wait a moment.")
    return _pipeline

def get_evaluator() -> RAGEvaluator:
    if _evaluator is None:
        raise RuntimeError("Evaluator is still initializing. Please wait a moment.")
    return _evaluator

# ── Schemas ────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3)

class FeedbackRequest(BaseModel):
    query: str
    answer: str
    rating: int = Field(..., ge=1, le=5)
    comment: Optional[str] = None
    thumbs_up: Optional[bool] = None

# ── Endpoints ──────────────────────────────────────────────────────────────

from fastapi.responses import RedirectResponse

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")

@app.get("/health", tags=["System"])
async def health_check() -> dict:
    if _pipeline is None:
        return {
            "status": "loading",
            "message": "Models are loading in the background. The API will be fully ready shortly."
        }
    
    pipeline = _pipeline
    try:
        hf_ok = is_hf_available()
    except Exception:
        hf_ok = False
    return {
        "status": "ready",
        "hf_available": hf_ok,
        "indexed_chunks": pipeline.num_chunks,
        "documents": pipeline.get_document_list(),
    }


@app.post("/upload", tags=["Documents"])
async def upload_document(file: UploadFile = File(...)) -> dict:
    allowed = {".pdf", ".txt", ".md"}
    suffix = Path(file.filename or "").suffix.lower()

    if suffix not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{suffix}'.")

    dest = DATA_DIR / (file.filename or "upload")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Clear previously uploaded files to ensure only the new upload exists
    for p in DATA_DIR.iterdir():
        if p.is_file():
            try:
                p.unlink()
            except Exception as e:
                logger.warning(f"Could not remove old file {p.name}: {e}")

    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    pipeline = get_pipeline()
    try:
        # Clear the vector store so no previous data is retained
        pipeline.reset()
        chunks_added = pipeline.ingest(dest)
    except Exception as exc:
        logger.error(f"Ingestion error: {exc}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")

    return {
        "filename": file.filename,
        "chunks_indexed": chunks_added,
        "message": f"Indexed {chunks_added} chunks.",
    }


@app.post("/query", tags=["RAG"])
async def query(request: QueryRequest) -> dict:
    pipeline = get_pipeline()

    if pipeline.num_chunks == 0:
        raise HTTPException(status_code=400, detail="No documents indexed yet.")

    start = time.perf_counter()

    try:
        response = pipeline.query(request.question)
    except RuntimeError as exc:
        logger.error(f"Query error: {exc}")
        raise HTTPException(status_code=503, detail=str(exc))

    latency_ms = round((time.perf_counter() - start) * 1000, 1)

    # ── Always log internally ──────────────────────────────────────────────
    logger.info(
        f"Query answered | confidence={response.confidence} "
        f"| sources={[s.document for s in response.sources]} "
        f"| latency={latency_ms}ms"
    )

    # ── Clean production response ──────────────────────────────────────────
    payload: dict = {"answer": response.answer}

    # ── DEBUG: append full details only when DEBUG=True ───────────────────
    if DEBUG:
        evaluator = get_evaluator()
        context_texts = [s.snippet for s in response.sources]
        retrieved_docs = [s.document for s in response.sources]
        eval_result = evaluator.evaluate(
            question=request.question,
            answer=response.answer,
            context_texts=context_texts,
            retrieved_doc_names=retrieved_docs,
            confidence=response.confidence,
        )
        payload["debug"] = {
            "confidence": response.confidence,
            "confidence_label": response.confidence_label,
            "hallucination_risk": response.hallucination_risk,
            "latency_ms": latency_ms,
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
        }

    return payload


@app.post("/feedback", tags=["Feedback"])
async def feedback(request: FeedbackRequest) -> dict:
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
    return {"feedback_id": fid, "message": "Feedback stored."}


@app.delete("/reset", tags=["System"])
async def reset_index() -> dict:
    get_pipeline().reset()
    return {"message": "Vector store cleared."}