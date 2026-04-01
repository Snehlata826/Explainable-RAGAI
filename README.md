---
title: Explainable RAG AI
colorFrom: blue
colorTo: purple
sdk: docker
app_file: ui/app.py
pinned: false
---

# ⚡GenAI RAG Copilot

> Production-grade Retrieval-Augmented Generation system for academic document Q&A with grounded answers, source citations, and confidence scoring.

[Try the Live Demo](https://huggingface.co/spaces/23bced49/explainable-ragai-ui)

---

## Quick Navigation

- [Overview](#overview)
- [Installation](#installation)
- [Configuration](#configuration-reference)
- [API Reference](#api-reference)
- [Usage Examples](#usage-examples)
- [Troubleshooting](#troubleshooting)

---

## Overview

GenAI RAG Copilot is a RAG system designed for researchers to efficiently extract information from academic papers. Upload one or more research papers and ask natural language questions to receive precise, grounded answers with source citations and confidence metrics.

The system is designed to answer only from uploaded documents—if information is not available in the papers, it indicates so rather than hallucinating.

---

## Architecture

```
Question Input
    |
    v
Query Enhancement (Rewriting + Expansion)
    |
    v
Hybrid Search
├─ FAISS Semantic Search (BAAI/bge-small-en-v1.5)
└─ BM25 Keyword Search
    |
    v
Cross-Encoder Reranking
    |
    v
LLM Answer Generation (Mistral-7B-Instruct)
    |
    v
Response with Confidence & Citations
```

---

## Features

- Document ingestion (PDF, TXT, Markdown)
- Hybrid retrieval combining semantic and keyword search
- Cross-encoder reranking for relevance
- Source citations and evidence snippets
- Confidence scoring with hallucination detection
- Multi-document comparison mode
- REST API with documentation
- Streamlit web interface
- Docker containerization
- HuggingFace Spaces deployment

---

## Technology Stack

| Component | Technology |
|---|---|
| Backend API | FastAPI |
| Frontend UI | Streamlit |
| Embeddings | BAAI/bge-small-en-v1.5 |
| Vector Store | FAISS |
| Reranking | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM | Mistral-7B-Instruct-v0.2 |
| Language | Python 3.10+ |

---

## Project Structure

```
Explainable-RAGAI/
├── api/
│   ├── main.py
│   └── rag_pipeline.py
├── config/
│   └── settings.py
├── data/
│   ├── raw_docs/
│   └── faiss_index/
├── embeddings/
├── ingestion/
├── retrieval/
├── vector_store/
├── generation/
├── explainability/
├── evaluation/
├── monitoring/
├── ui/
│   └── app.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## Prerequisites

- Python 3.10 or higher
- HuggingFace account with API token
- Docker (for containerized deployment)
- 4GB RAM minimum (8GB recommended)

---

## Installation

### Local Setup

```bash
git clone https://github.com/Snehlata826/Explainable-RAGAI.git
cd Explainable-RAGAI

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Configuration

```bash
cp .env.example .env
# Add HuggingFace API token to .env file
# HF_API_TOKEN=hf_your_token_here
```

### Start Services

Terminal 1 - Backend API:
```bash
PYTHONPATH=. uvicorn api.main:app --host 0.0.0.0 --port 8005 --reload
```

Terminal 2 - Frontend UI:
```bash
PYTHONPATH=. streamlit run ui/app.py
```

### Verify Installation

Check if everything is working:
```bash
curl http://localhost:8005/health
```

Access the application at `http://localhost:8501`

---

## Docker Deployment

```bash
docker-compose up --build
```

| Service | URL |
|---|---|
| FastAPI | http://localhost:8000 |
| API Documentation | http://localhost:8000/docs |
| Streamlit UI | http://localhost:8501 |

---

## HuggingFace Spaces Configuration

Set the following environment variable in Spaces Settings:

- `HF_API_TOKEN` - Your HuggingFace API token

---

## Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `HF_API_TOKEN` | required | HuggingFace API token |
| `HF_MODEL` | mistralai/Mistral-7B-Instruct-v0.2 | LLM model |
| `EMBEDDING_MODEL` | BAAI/bge-small-en-v1.5 | Embedding model |
| `CHUNK_SIZE` | 350 | Tokens per chunk |
| `CHUNK_OVERLAP` | 100 | Overlap between chunks |
| `TOP_K_RETRIEVAL` | 15 | Retrieval candidates |
| `TOP_K_RERANK` | 3 | Final contexts for LLM |
| `VECTOR_WEIGHT` | 0.65 | Semantic search weight |
| `BM25_WEIGHT` | 0.35 | Keyword search weight |
| `LLM_TEMPERATURE` | 0.1 | Temperature for LLM |
| `DEBUG` | false | Enable debug metrics |

---

## API Reference

### Upload Document

**Endpoint:**
```bash
POST /upload
```

**Description:** Upload a research paper for indexing.

**Example:**
```bash
curl -X POST http://localhost:8005/upload \
  -F "file=@research_paper.pdf"
```

**Response:**
```json
{
  "filename": "research_paper.pdf",
  "chunks_indexed": 42,
  "message": "Successfully indexed 42 chunks"
}
```

---

### Query Documents

**Endpoint:**
```bash
POST /query
```

**Description:** Ask a question about indexed documents.

**Example:**
```bash
curl -X POST http://localhost:8005/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main contribution of this paper?"}'
```

**Response:**
```json
{
  "answer": "The paper proposes a novel RAG system that combines semantic and keyword search...",
  "confidence": 0.87,
  "confidence_label": "HIGH",
  "hallucination_risk": "LOW",
  "sources": [
    {
      "document": "research_paper.pdf",
      "relevance_score": 0.92,
      "snippet": "Our system combines FAISS semantic search with BM25 keyword search...",
      "chunk_index": 5
    }
  ]
}
```

**Response includes:**
- Answer text grounded in documents
- Confidence score (0-1)
- Source citations with snippets
- Hallucination risk assessment

---

### Reset Index

**Endpoint:**
```bash
DELETE /reset
```

**Description:** Clear all indexed documents.

**Example:**
```bash
curl -X DELETE http://localhost:8005/reset
```

---

### Health Check

**Endpoint:**
```bash
GET /health
```

**Description:** Check API and model status.

**Example:**
```bash
curl http://localhost:8005/health
```

**Response:**
```json
{
  "status": "ready",
  "api": "operational",
  "models_loaded": true,
  "indexed_documents": 3,
  "indexed_chunks": 156
}
```

---

## Usage Examples

### Single Paper Queries

- "What dataset was used for evaluation?"
- "What are the main limitations of this approach?"
- "Summarize the methodology section."
- "What are the key findings?"
- "How does this compare to previous work?"

### Multi-Paper Queries

- "How do these papers differ in their approach?"
- "Which paper reports higher accuracy and under what conditions?"
- "What do all these papers agree on?"
- "Compare the methodologies used across these papers."
- "What datasets are used in each paper?"

### Example Workflow

```bash
# 1. Upload first paper
curl -X POST http://localhost:8005/upload \
  -F "file=@paper1.pdf"

# 2. Upload second paper
curl -X POST http://localhost:8005/upload \
  -F "file=@paper2.pdf"

# 3. Ask a comparative question
curl -X POST http://localhost:8005/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How do these approaches differ?"}'

# 4. View results with citations and confidence scores
```

---

## Response Format

Each query response includes:
- **Answer** - Grounded answer from documents
- **Confidence** - Score from 0 to 1
- **Confidence Label** - HIGH, MEDIUM, or LOW
- **Sources** - List of cited sources with snippets and chunk indices
- **Hallucination Risk** - Risk assessment level (LOW, MEDIUM, HIGH)

---

## Evaluation Metrics

Available when `DEBUG=true`:
- Groundedness - Fraction of answer grounded in context
- Hallucination Rate - Inverse of groundedness
- Answer Relevance - Similarity to original question
- Context Utilization - Amount of context used
- Retrieval F1 - Precision and recall metrics

---

## Performance Requirements

| Aspect | Details |
|---|---|
| CPU | Dual-core minimum (4-core recommended) |
| RAM | 4GB minimum (8GB recommended) |
| Disk | 5GB for models |
| Upload time | 5-10 seconds per 10-page PDF |
| Query latency | 2-4 seconds (first query may take longer) |
| Model initialization | 30-60 seconds on first run |

---

## Security

- Never commit `.env` files to version control
- Keep `HF_API_TOKEN` confidential
- Use environment variables for all secrets
- Store credentials in Spaces Secrets for HuggingFace deployment
- Use HTTPS in production environments

---

## Troubleshooting

| Issue | Solution |
|---|---|
| HF_API_TOKEN not set | Add token to .env file or Spaces secrets |
| Models loading slowly | First load takes 30-60 seconds; check `/health` endpoint |
| Port already in use | Change port: `--port 8006` in startup command |
| FAISS import error | Run `pip install faiss-cpu` or `pip install faiss-gpu` |
| Connection refused | Ensure both FastAPI and Streamlit services are running |
| Out of memory | Reduce `TOP_K_RETRIEVAL` value or upload fewer documents |

---

## License

MIT License

---

## References

- Live Demo: https://huggingface.co/spaces/23bced49/explainable-ragai-ui
- Repository: https://github.com/Snehlata826/Explainable-RAGAI
- HuggingFace: https://huggingface.co/spaces/23bced49/explainable-ragai-ui

---

## Contributing

We welcome contributions. Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Submit a pull request with a clear description

---

**Important:** Verify critical information from original sources. The system generates answers based on uploaded documents and may produce incomplete responses despite grounding mechanisms.
