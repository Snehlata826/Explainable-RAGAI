---
title: Explainable RAG AI
colorFrom: blue
colorTo: purple
sdk: docker
app_file: ui/app.py
pinned: false
---

# GenAI RAG Copilot

> Production-grade Retrieval-Augmented Generation system for academic document Q&A with grounded answers, source citations, and confidence scoring.

[Try the Live Demo](https://huggingface.co/spaces/23bced49/explainable-ragai-ui)

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
```bash
POST /upload
```
Upload a research paper for indexing.

### Query Documents
```bash
POST /query
```
Ask a question about indexed documents.

**Response includes:**
- Answer text
- Confidence score (0-1)
- Source citations
- Hallucination risk assessment

### Reset Index
```bash
DELETE /reset
```
Clear all indexed documents.

### Health Check
```bash
GET /health
```
Check API and model status.

---

## Response Format

Each query response includes:
- **Answer** - Grounded answer from documents
- **Confidence** - Score from 0 to 1
- **Confidence Label** - HIGH, MEDIUM, or LOW
- **Sources** - List of cited sources with snippets
- **Hallucination Risk** - Risk assessment level

---

## Evaluation Metrics

Available when `DEBUG=true`:
- Groundedness - Fraction of answer grounded in context
- Hallucination Rate - Inverse of groundedness
- Answer Relevance - Similarity to original question
- Context Utilization - Amount of context used
- Retrieval F1 - Precision and recall metrics

---

## Troubleshooting

| Issue | Solution |
|---|---|
| HF_API_TOKEN not set | Add token to .env file or Spaces secrets |
| Models loading slowly | First load takes 30-60 seconds |
| Port already in use | Change port in startup command |
| FAISS import error | Run `pip install faiss-cpu` |

---

## License

MIT License

---

## References

- Live Demo: https://huggingface.co/spaces/23bced49/explainable-ragai-ui
- Repository: https://github.com/Snehlata826/Explainable-RAGAI
- HuggingFace: https://huggingface.co/spaces/23bced49/explainable-ragai-ui

---

**Important:** Verify critical information from original sources. The system generates answers based on uploaded documents and may produce incomplete responses despite grounding mechanisms.
