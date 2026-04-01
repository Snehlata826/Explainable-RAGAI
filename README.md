# ⚡ GenAI RAG Copilot

> **Production-grade Explainable RAG system for document Q&A**
> 
> Hybrid FAISS + BM25 retrieval · Cross-encoder reranking · Grounded LLM answers · Confidence scoring · Source citations

**🎯 [Try the Live Demo](https://huggingface.co/spaces/23bced49/explainable-ragai-ui)** ← Click here to explore!

---

## Overview

GenAI RAG Copilot answers questions from your documents using a multi-stage retrieval-augmented generation pipeline. It combines semantic vector search with keyword search, reranks candidates using a cross-encoder, and generates grounded answers via LLM — with full explainability through confidence scores, hallucination risk flags, and source citations.

---

## Architecture

```
User Query
    │
    ▼
Query Rewriting + Expansion (4 semantic variants)
    │
    ▼
Query Embedding (BAAI/bge-small-en-v1.5)
    │
    ├──► FAISS Vector Search  ──────┐
    │                               ├──► Hybrid Fusion (0.65 · vec + 0.35 · BM25)
    └──► BM25 Keyword Search  ──────┘
                    │
                    ▼
        Top-15 Candidates
                    │
                    ▼
    Cross-Encoder Reranking (ms-marco-MiniLM-L-6-v2)
                    │
                    ▼
        Top-3 Context Chunks
        (deduplicated · threshold-filtered · compressed)
                    │
                    ▼
    LLM Answer Generation (Mistral-7B-Instruct)
                    │
                    ▼
    Explainability Engine
    (confidence · hallucination risk · source citations)
                    │
                    ▼
    Response { answer, confidence, sources }
```

---

## Key Features

| Feature | Details |
|---|---|
| **Hybrid Retrieval** | FAISS vector search + BM25 keyword search with intelligent fusion |
| **Query Intelligence** | Query rewriting and semantic expansion for better recall |
| **Smart Reranking** | Cross-encoder reranking ensures most relevant context |
| **Source Attribution** | Every answer includes cited sources with confidence scores |
| **Hallucination Guards** | Word-overlap verification and grounding checks |
| **Explainability** | Confidence labels (HIGH/MEDIUM/LOW) + hallucination risk assessment |
| **Document Ingestion** | Supports PDF, TXT, Markdown with intelligent chunking |
| **Multi-Doc Reasoning** | Compare answers across multiple uploaded documents |
| **Evaluation Metrics** | Groundedness, hallucination rate, context utilization, F1 retrieval |
| **REST API** | FastAPI with async processing and Swagger documentation |
| **Modern UI** | Streamlit with dark/light mode and typing animations |
| **Containerized** | Docker Compose for local deployment |
| **HuggingFace Ready** | Auto-deploy to HuggingFace Spaces via GitHub Actions |
| **Monitoring** | Structured JSON logging and latency tracking |

---

## Tech Stack

| Component | Technology |
|---|---|
| **Backend API** | FastAPI (async, OpenAPI docs) |
| **Frontend UI** | Streamlit (responsive, dark mode) |
| **Embeddings** | BAAI/bge-small-en-v1.5 (384-dim, L2-normalized) |
| **Vector Store** | FAISS IndexFlatIP (disk-persisted) |
| **Keyword Search** | BM25Okapi |
| **Reranking** | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| **LLM** | Mistral-7B-Instruct-v0.2 (via HuggingFace Inference API) |
| **Document Processing** | spaCy, PyPDF2, python-docx |
| **Containerization** | Docker & Docker Compose |
| **Language** | Python 3.10+ |

---

## Project Structure

```
GenAI-RAG-Copilot/
├── api/
│   ├── main.py                  # FastAPI endpoints (upload, query, feedback, health)
│   └── rag_pipeline.py          # RAG orchestration (rewrite → retrieve → generate)
├── config/
│   └── settings.py              # Environment configuration management
├── data/
│   ├── raw_docs/                # Uploaded documents
│   ├── faiss_index/             # Persisted FAISS index + metadata
│   └── feedback/                # User feedback records (JSONL)
├── embeddings/
│   └── embedding_generator.py   # Sentence transformer wrapper
├── ingestion/
│   └── document_processor.py    # PDF/TXT parsing & chunking
├── retrieval/
│   ├── hybrid_retriever.py      # FAISS + BM25 fusion
│   ├── reranker.py              # Cross-encoder reranking
│   └── context_retriever.py     # End-to-end retrieval pipeline
├── vector_store/
│   └── faiss_store.py           # FAISS CRUD & persistence
├── generation/
│   ├── llm_client.py            # HuggingFace InferenceClient
│   └── answer_generator.py      # Prompt engineering & grounding
├── explainability/
│   └── explanation_engine.py    # Confidence scoring & source citation
├── evaluation/
│   └── metrics.py               # RAG evaluation (groundedness, hallucination rate, F1)
├── monitoring/
│   └── logger.py                # Structured logging & performance tracking
├── ui/
│   └── app.py                   # Streamlit frontend
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── start.sh                     # Startup script
└── .env.example
```

---

## Prerequisites

- Python 3.10+
- [HuggingFace](https://huggingface.co) account with API token
- Access to Mistral-7B-Instruct model (or configure alternative LLM)

---

## Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/Snehlata826/Explainable-RAGAI.git
cd Explainable-RAGAI

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Configure Environment

```bash
cp .env.example .env
# Set: HF_API_TOKEN (required)
# Optional: HF_MODEL, CHUNK_SIZE, TOP_K_RETRIEVAL, etc.
```

### 3. Start Backend API

```bash
PYTHONPATH=. uvicorn api.main:app --host 0.0.0.0 --port 8005 --reload
```

**Swagger UI**: [http://localhost:8005/docs](http://localhost:8005/docs)

> Models load asynchronously — check `/health` endpoint for status.

### 4. Start Frontend UI

```bash
PYTHONPATH=. streamlit run ui/app.py
```

**UI**: [http://localhost:8501](http://localhost:8501)

---

## Docker Deployment

Run both services with a single command:

```bash
docker-compose up --build
```

| Service | URL |
|---|---|
| FastAPI | http://localhost:8000 |
| Swagger UI | http://localhost:8000/docs |
| Streamlit | http://localhost:8501 |

---

## API Reference

### `POST /upload`

Upload a document for indexing.

```bash
curl -X POST http://localhost:8005/upload \
  -F "file=@document.pdf"
```

**Response:**
```json
{
  "filename": "document.pdf",
  "chunks_indexed": 42,
  "message": "Indexed 42 chunks."
}
```

---

### `POST /query`

Ask a question about indexed documents.

```bash
curl -X POST http://localhost:8005/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic?"}'
```

**Response:**
```json
{
  "answer": "Based on the documents: ...",
  "confidence": 0.85,
  "sources": [
    {
      "document": "document.pdf",
      "chunk_id": 5,
      "score": 0.92,
      "snippet": "..."
    }
  ]
}
```

Enable `DEBUG=true` in `.env` for full evaluation metrics.

---

### `POST /feedback`

Submit user feedback (rating 1-5).

```bash
curl -X POST http://localhost:8005/feedback \
  -H "Content-Type: application/json" \
  -d '{"query": "...", "answer": "...", "rating": 5}'
```

---

### `GET /health`

Check API and model status.

```bash
curl http://localhost:8005/health
```

---

### `DELETE /reset`

Clear the vector store and all indexed documents.

---

## Configuration

All settings in `config/settings.py` can be overridden via environment variables:

| Variable | Default | Description |
|---|---|---|
| `HF_API_TOKEN` | *(required)* | HuggingFace API token |
| `HF_MODEL` | `mistralai/Mistral-7B-Instruct-v0.2` | Chat model |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Embedding model |
| `CHUNK_SIZE` | `350` | Tokens per chunk |
| `CHUNK_OVERLAP` | `100` | Overlap between chunks |
| `TOP_K_RETRIEVAL` | `15` | Candidates before reranking |
| `TOP_K_RERANK` | `3` | Final contexts after reranking |
| `SIMILARITY_THRESHOLD` | `0.25` | Min retrieval score |
| `VECTOR_WEIGHT` | `0.65` | Weight for vector search in hybrid fusion |
| `BM25_WEIGHT` | `0.35` | Weight for keyword search in hybrid fusion |
| `LLM_TEMPERATURE` | `0.1` | Sampling temperature |
| `DEBUG` | `false` | Expose evaluation metrics |

---

## Explainability Engine

Every response includes:

- **Confidence Score** (0-1) — Normalized relevance with hallucination awareness
- **Confidence Label** — HIGH (≥0.75) · MEDIUM (≥0.50) · LOW (<0.50)
- **Hallucination Risk** — LOW / MEDIUM / HIGH assessment
- **Source Citations** — Per-chunk evidence with document name and relevance score
- **Grounding Verification** — Word-overlap check ensures answer is grounded in retrieved context

---

## Evaluation Metrics

When `DEBUG=true`, responses include:

| Metric | Description |
|---|---|
| **Groundedness** | Fraction of answer words found in retrieved context |
| **Hallucination Rate** | Inverse of groundedness |
| **Answer Relevance** | Similarity between answer and query |
| **Context Utilization** | Fraction of context used in answer |
| **Retrieval F1** | Precision and recall of document retrieval |
| **Overall Score** | Weighted aggregate metric |

---

## Monitoring & Logging

All events logged to `logs/rag_copilot.log`:

```json
{
  "timestamp": "2024-01-15T10:30:45Z",
  "query": "What is RAG?",
  "confidence": 0.87,
  "latency_ms": 1250.5,
  "num_sources": 3
}
```

---

## Extending the Project

| Goal | Where to modify |
|---|---|
| Use different LLM (OpenAI, Ollama) | `generation/llm_client.py` |
| Add document formats (DOCX, HTML) | `ingestion/document_processor.py` |
| Swap FAISS for other vector DB | `vector_store/faiss_store.py` |
| Implement streaming responses | `api/main.py` |
| Build custom evaluation metrics | `evaluation/metrics.py` |

---

## License

MIT License

---

## Support & Contributions

Found a bug? Have a feature request? Open an issue on GitHub!

Contributions welcome — see CONTRIBUTING.md for guidelines.

---

## Disclaimer

This system generates answers based on document content. **Always verify critical information** with authoritative sources. The model may occasionally produce inaccurate or incomplete responses despite grounding mechanisms.
