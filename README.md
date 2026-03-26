---
title: Explainable RAG AI
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: "1.36.0"
app_file: ui/app.py
pinned: false
---

# GenAI RAG Copilot

GenAI RAG Copilot is a Retrieval-Augmented Generation (RAG) system for answering questions from documents using hybrid retrieval, reranking, and grounded LLM responses.

The system retrieves relevant context from documents and generates answers with source citations and confidence scores to reduce hallucinations.

---

## Problem

Large Language Models often hallucinate when answering questions about specific documents.  
Traditional keyword search systems also fail to capture semantic meaning in text.

The goal is to build a system that:

- retrieves relevant document context
- combines semantic search and keyword search
- ranks the most relevant passages
- generates answers grounded in retrieved evidence
- provides explainability through citations and confidence scores

---

## Solution

This project implements a Retrieval-Augmented Generation pipeline that includes:

- document ingestion and chunking
- semantic embeddings for vector search
- BM25 keyword retrieval
- hybrid retrieval combining vector and keyword search
- cross-encoder reranking to improve relevance
- LLM generation using retrieved context
- explainability through source citations and confidence scores

---

## Architecture

```
User Query
    │
    ▼
Query Embedding (all-MiniLM-L6-v2)
    │
    ├── FAISS Vector Search
    └── BM25 Keyword Search
            │
            ▼
Hybrid Score Fusion
            │
            ▼
Top-K Retrieval
            │
            ▼
Cross Encoder Reranking
            │
            ▼
Top Context Chunks
            │
            ▼
LLM Generation (Ollama: Mistral / LLaMA)
            │
            ▼
Response
{ answer, confidence, sources }
```

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend API | FastAPI |
| UI | Streamlit |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Store | FAISS |
| Keyword Search | BM25 |
| Reranking | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM | Ollama (Mistral / LLaMA) |

---

## Run the Project

Clone the repository

```
git clone https://github.com/yourusername/GenAI-RAG-Copilot.git
cd GenAI-RAG-Copilot
```

Create environment

```
python -m venv .venv
source .venv/bin/activate
```

Install dependencies

```
pip install -r requirements.txt
```

Start Ollama and download a model

```
ollama serve
ollama pull mistral
```

Start backend

```
PYTHONPATH=. uvicorn api.main:app --port 8000 --reload
```

Start frontend

```
PYTHONPATH=. streamlit run ui/app.py
```

---

## License

MIT License