# ⚡ GenAI RAG Copilot

> **Ask questions across your research papers. Get grounded answers with citations and confidence scores.**
> 
> Hybrid retrieval · Cross-encoder reranking · Multi-paper reasoning · Full explainability

**🎯 [Try the Live Demo](https://huggingface.co/spaces/23bced49/explainable-ragai-ui)** ← Click here to explore!

---

## Overview

GenAI RAG Copilot transforms how researchers interact with academic papers. Instead of manually skimming through documents, you can upload papers and ask natural language questions, receiving precise answers with source citations and confidence scores.

The system answers **only from your uploaded papers** — if information isn't in the documents, it says so rather than hallucinating.

---

## How It Works

```
Your Question
      │
      ▼
Query Enhancement (rewriting + semantic expansion)
      │
      ▼
Hybrid Search
├──► FAISS Semantic Search (BAAI/bge-small-en-v1.5)
└──► BM25 Keyword Search
      │
      ▼
Cross-Encoder Reranking (ms-marco-MiniLM-L-6-v2)
      │
      ▼
Grounded Answer Generation (Mistral-7B-Instruct)
      │
      ▼
Response with Confidence & Citations
```

---

## Key Features

| Feature | Details |
|---|---|
| **Multiple formats** | PDF, TXT, Markdown support |
| **Smart chunking** | Sentence-aware 350-token chunks with 100-token overlap |
| **Hybrid retrieval** | FAISS + BM25 with configurable fusion weights |
| **Reranking** | Cross-encoder ensures most relevant results |
| **Multi-paper mode** | Compare answers across multiple documents |
| **Source citations** | Every answer includes document name, relevance score, and evidence snippet |
| **Confidence scoring** | Know how confident the system is in each answer |
| **Hallucination guard** | Grounding verification prevents false information |
| **REST API** | FastAPI with Swagger documentation |
| **Modern UI** | Streamlit with dark/light mode |
| **Docker ready** | One-command deployment |
| **HuggingFace Spaces** | Auto-deploy via GitHub Actions |

---

## Tech Stack

| Component | Technology |
|---|---|
| **Backend** | FastAPI |
| **Frontend** | Streamlit |
| **Embeddings** | BAAI/bge-small-en-v1.5 |
| **Vector Store** | FAISS |
| **Keyword Search** | BM25 |
| **Reranking** | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| **LLM** | Mistral-7B-Instruct-v0.2 (HuggingFace) |
| **Language** | Python 3.10+ |

---

## Project Structure

```
GenAI-RAG-Copilot/
├── api/
│   ├── main.py              # FastAPI endpoints
│   └── rag_pipeline.py      # RAG orchestration
├── config/
│   └── settings.py          # Configuration management
├── data/
│   ├── raw_docs/            # Uploaded papers
│   └── faiss_index/         # Vector store
├── embeddings/
│   └── embedding_generator.py
├── ingestion/
│   └── document_processor.py
├── retrieval/
│   ├── hybrid_retriever.py
│   ├── reranker.py
│   └── context_retriever.py
├── vector_store/
│   └── faiss_store.py
├── generation/
│   ├── llm_client.py
│   └── answer_generator.py
├── explainability/
│   └── explanation_engine.py
├── evaluation/
│   └── metrics.py
├── monitoring/
│   └── logger.py
├── ui/
│   └── app.py
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

---

## Prerequisites

- Python 3.10+
- [HuggingFace](https://huggingface.co) account with API token
- Access to Mistral-7B-Instruct model

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
# Add your HuggingFace token:
# HF_API_TOKEN=hf_your_token_here
```

### 3. Start Backend API

```bash
PYTHONPATH=. uvicorn api.main:app --host 0.0.0.0 --port 8005 --reload
```

**API Documentation**: [http://localhost:8005/docs](http://localhost:8005/docs)

### 4. Start Frontend UI

```bash
PYTHONPATH=. streamlit run ui/app.py
```

**Open**: [http://localhost:8501](http://localhost:8501)

---

## Docker Deployment

```bash
docker-compose up --build
```

| Service | URL |
|---|---|
| FastAPI | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |
| Streamlit UI | http://localhost:8501 |

---

## Usage Examples

**Single Paper**
- "What dataset was used for evaluation?"
- "What are the main limitations?"
- "Summarize the methodology section."

**Multiple Papers**
- "How do these papers differ in their approach?"
- "Which paper reports higher accuracy?"
- "What do all these papers agree on?"

---

## API Reference

### `POST /upload`
Upload a research paper for indexing.

```bash
curl -X POST http://localhost:8005/upload \
  -F "file=@paper.pdf"
```

### `POST /query`
Ask a question about indexed papers.

```bash
curl -X POST http://localhost:8005/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main contribution?"}'
```

**Response includes:**
- Answer grounded in paper content
- Confidence score (0-1)
- Source citations with snippets
- Hallucination risk assessment

### `DELETE /reset`
Clear all indexed papers.

### `GET /health`
Check API and model status.

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `HF_API_TOKEN` | *(required)* | HuggingFace API token |
| `HF_MODEL` | `mistralai/Mistral-7B-Instruct-v0.2` | LLM model |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Embedding model |
| `CHUNK_SIZE` | `350` | Tokens per chunk |
| `CHUNK_OVERLAP` | `100` | Overlap between chunks |
| `TOP_K_RETRIEVAL` | `15` | Candidates before reranking |
| `TOP_K_RERANK` | `3` | Final contexts for LLM |
| `VECTOR_WEIGHT` | `0.65` | Weight for semantic search |
| `BM25_WEIGHT` | `0.35` | Weight for keyword search |
| `LLM_TEMPERATURE` | `0.1` | Controls randomness (lower = more grounded) |
| `DEBUG` | `false` | Enable detailed response metrics |

---

## Explainability

Every response includes:

- **Confidence Score** (0-1) — Relevance score from cross-encoder
- **Confidence Label** — HIGH / MEDIUM / LOW
- **Hallucination Risk** — Risk assessment based on grounding
- **Source Citations** — Document, relevance score, and evidence snippet
- **Grounding Verification** — Ensures answer is grounded in retrieved context

---

## Evaluation Metrics

Available in debug mode:

- **Groundedness** — Fraction of answer grounded in context
- **Hallucination Rate** — Inverse of groundedness
- **Answer Relevance** — Similarity to original question
- **Context Utilization** — How much context was used
- **Retrieval F1** — Precision and recall of document retrieval

---

## Extending the System

| Goal | File to Modify |
|---|---|
| Support new document formats | `ingestion/document_processor.py` |
| Use different LLM | `generation/llm_client.py` |
| Tune retrieval weights | `config/settings.py` |
| Add custom evaluation | `evaluation/metrics.py` |
| Integrate another vector DB | `vector_store/faiss_store.py` |

---

## Troubleshooting

**"HF_API_TOKEN is required"**
→ Set your token in `.env` file or as an environment variable.

**Models loading slowly**
→ Check `/health` endpoint. Models load asynchronously in the background.

**FAISS not found**
→ Run `pip install faiss-cpu` (or `faiss-gpu` for GPU acceleration).

**Port already in use**
→ Change port in startup command: `--port 8006`

---

## License

MIT License

---

## Disclaimer

This system generates answers based solely on your uploaded documents. **Always verify critical information** with original sources. The model may occasionally produce incomplete or inaccurate responses despite grounding mechanisms.
