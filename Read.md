# 🔍 GenAI RAG Copilot

> **Production-quality Explainable RAG system for document Q&A**
> Hybrid retrieval · Cross-encoder reranking · Grounded LLM answers · Confidence scores · Full citations

---

## Architecture

```
User Query
    │
    ▼
Query Embedding (all-MiniLM-L6-v2)
    │
    ├──► FAISS Vector Search ──────┐
    │                              ├──► Hybrid Fusion (0.6·vec + 0.4·BM25)
    └──► BM25 Keyword Search ──────┘
                   │
                   ▼
        Top-K Candidates (10)
                   │
                   ▼
    Cross-Encoder Reranking (ms-marco-MiniLM-L-6-v2)
                   │
                   ▼
        Top-5 Context Chunks
                   │
                   ▼
    LLM Answer Generation (Ollama · Mistral/LLaMA)
                   │
                   ▼
    Explainability Engine
                   │
                   ▼
    Response { answer, confidence, sources[] }
```

---

## Features

| Feature | Details |
|---|---|
| **Document ingestion** | PDF, TXT, Markdown with token-based chunking |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` |
| **Vector store** | FAISS (IndexFlatIP, persisted to disk) |
| **Hybrid retrieval** | FAISS + BM25 with configurable weights |
| **Reranking** | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| **LLM** | Any Ollama model (Mistral, LLaMA 3, etc.) |
| **Explainability** | Confidence score · Source citations · Evidence snippets |
| **API** | FastAPI with async endpoints + Swagger UI |
| **UI** | Streamlit with chat interface + feedback |
| **Feedback** | JSONL feedback store for RLHF |
| **Monitoring** | Structured JSON query logs + latency tracking |

---

## Project Structure

```
GenAI-RAG-Copilot/
├── api/
│   ├── main.py               # FastAPI app (upload / query / feedback endpoints)
│   └── rag_pipeline.py       # Pipeline orchestrator
├── config/
│   └── settings.py           # All config via env vars
├── data/
│   ├── raw_docs/             # Uploaded documents
│   ├── faiss_index/          # Persisted FAISS index + metadata
│   └── feedback/             # JSONL feedback records
├── embeddings/
│   └── embedding_generator.py
├── ingestion/
│   └── document_processor.py # PDF/TXT extraction + chunking
├── retrieval/
│   ├── hybrid_retriever.py   # FAISS + BM25 fusion
│   ├── reranker.py           # Cross-encoder reranking
│   └── context_retriever.py  # Full retrieval pipeline
├── vector_store/
│   └── faiss_store.py        # FAISS CRUD + persistence
├── generation/
│   ├── llm_client.py         # Ollama HTTP client
│   └── answer_generator.py   # Grounded prompt + generation
├── explainability/
│   └── explanation_engine.py # Confidence + source packaging
├── feedback/
│   └── feedback_store.py     # JSONL feedback persistence
├── monitoring/
│   └── logger.py             # Structured logging + latency decorator
├── ui/
│   └── app.py                # Streamlit frontend
├── logs/                     # Runtime log files
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

---

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running locally
- A pulled Ollama model (Mistral recommended)

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/yourname/GenAI-RAG-Copilot.git
cd GenAI-RAG-Copilot

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Start Ollama and pull a model

```bash
# In a separate terminal
ollama serve

# Pull Mistral (recommended, ~4 GB)
ollama pull mistral

# Or LLaMA 3 8B
ollama pull llama3
```

### 3. Configure environment (optional)

```bash
cp .env.example .env
# Edit .env to change model, chunk sizes, weights, etc.
```

### 4. Start the FastAPI backend

```bash
# From project root
PYTHONPATH=. uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

API docs available at: http://localhost:8000/docs

### 5. Start the Streamlit UI

```bash
# In a new terminal (from project root)
PYTHONPATH=. streamlit run ui/app.py
```

UI available at: http://localhost:8501

---

## Docker Compose (recommended)

Runs both services with a single command. Ollama must still run on your host.

```bash
# Build and start
docker-compose up --build

# Stop
docker-compose down
```

| Service | URL |
|---|---|
| FastAPI | http://localhost:8000 |
| Swagger UI | http://localhost:8000/docs |
| Streamlit | http://localhost:8501 |

---

## API Reference

### `POST /upload`
Upload a PDF or TXT document for ingestion.

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@research_paper.pdf"
```

```json
{
  "filename": "research_paper.pdf",
  "chunks_indexed": 42,
  "message": "Successfully indexed 42 chunks."
}
```

---

### `POST /query`
Ask a question against indexed documents.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the transformer architecture?"}'
```

```json
{
  "answer": "The transformer architecture uses self-attention mechanisms ...",
  "confidence": 0.87,
  "confidence_label": "HIGH",
  "sources": [
    {
      "document": "attention_is_all_you_need.pdf",
      "score": 0.92,
      "snippet": "Multi-head attention allows the model to jointly attend ...",
      "chunk_index": 5
    }
  ],
  "latency_ms": 1234.5
}
```

---

### `POST /feedback`
Submit user feedback on an answer (rating 1–5).

```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the transformer architecture?",
    "answer": "The transformer ...",
    "rating": 5,
    "comment": "Very accurate!"
  }'
```

---

### `GET /health`
Check API status, Ollama availability, and index size.

---

## Configuration Reference

All settings live in `config/settings.py` and can be overridden via environment variables:

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_MODEL` | `mistral` | Ollama model name |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `CHUNK_SIZE` | `500` | Tokens per chunk |
| `CHUNK_OVERLAP` | `50` | Overlapping tokens |
| `TOP_K_RETRIEVAL` | `10` | Candidates before reranking |
| `TOP_K_RERANK` | `5` | Final contexts after reranking |
| `VECTOR_WEIGHT` | `0.6` | Weight for vector scores in fusion |
| `BM25_WEIGHT` | `0.4` | Weight for BM25 scores in fusion |
| `LLM_TEMPERATURE` | `0.1` | LLM sampling temperature |
| `LLM_MAX_TOKENS` | `1024` | Max generation tokens |

---

## Explainability

Every response includes:

- **`confidence`** — sigmoid-normalised score from cross-encoder relevance, weighted by rank position (range 0–1)
- **`confidence_label`** — `HIGH` (≥0.75) · `MEDIUM` (≥0.50) · `LOW` (<0.50)
- **`sources[]`** — per-chunk citation with document name, relevance score, evidence snippet, and chunk index
- **Hallucination guard** — LLM is strictly prompted to answer only from context; returns a "not enough evidence" message if no answer is found

---

## Logs

All logs are written to `logs/rag_copilot.log`. Query events are also emitted as structured JSON:

```json
{
  "query": "What is attention?",
  "num_sources": 5,
  "confidence": 0.83,
  "latency_ms": 980.2
}
```

---

## Extending the Project

| Goal | Where to change |
|---|---|
| Use a different LLM provider | `generation/llm_client.py` |
| Add more file types (DOCX, HTML) | `ingestion/document_processor.py` |
| Switch vector store to Chroma/Weaviate | `vector_store/` |
| Add re-retrieval / query expansion | `retrieval/context_retriever.py` |
| Build RLHF pipeline from feedback | `feedback/feedback_store.py` |

---

## License

MIT
