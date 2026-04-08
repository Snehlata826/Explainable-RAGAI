FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download spacy model
RUN python -m spacy download en_core_web_sm

COPY . .

RUN mkdir -p data/raw_docs data/faiss_index data/feedback logs

ENV PYTHONPATH=/app
# Change this line
ENV API_BASE_URL=http://0.0.0.0:7860

# 🔐 Do NOT hardcode secrets
# Use environment variables instead
ENV HF_MODEL=mistralai/Mistral-7B-Instruct-v0.2

# HuggingFace requires port 7860
EXPOSE 7860

COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

CMD ["/app/start.sh"]