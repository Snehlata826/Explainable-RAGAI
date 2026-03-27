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

# Install Python dependencies (both backend and UI)
COPY requirements.txt .
COPY requirements-ui.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements-ui.txt

# Copy project source
COPY . .

# Ensure start script is executable
RUN chmod +x start.sh

# Create required directories and give permissions for Hugging Face (which runs as non-root)
RUN mkdir -p data/raw_docs data/faiss_index data/feedback logs && \
    chmod -R 777 /app/data /app/logs

# ── Environment ─────────────────────────────────────────────────────────────
ENV PYTHONPATH=/app
ENV API_BASE=http://localhost:8000

# ── Expose port required by HuggingFace Spaces ─────────────────────────────
EXPOSE 7860

# ── Default command: run both servers via bash script ───────────────────────
CMD ["./start.sh"]
