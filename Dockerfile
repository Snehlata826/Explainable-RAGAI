# ── Build stage ────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

LABEL maintainer="GenAI RAG Copilot"
LABEL description="Explainable RAG system for document Q&A"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY . .

# Create required directories
RUN mkdir -p data/raw_docs data/faiss_index data/feedback logs

# ── Environment ─────────────────────────────────────────────────────────────
ENV PYTHONPATH=/app
ENV OLLAMA_BASE_URL=http://host.docker.internal:11434
ENV OLLAMA_MODEL=mistral
ENV API_HOST=0.0.0.0
ENV API_PORT=8000

# ── Expose ports ────────────────────────────────────────────────────────────
# FastAPI
EXPOSE 8000
# Streamlit
EXPOSE 8501

# ── Default command: start FastAPI ──────────────────────────────────────────
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
