#!/bin/bash
# Start FastAPI in background
PYTHONPATH=. uvicorn api.main:app --host 0.0.0.0 --port 8005 &

# Wait for backend to be ready
sleep 5

# Start Streamlit on port 7860 (required by HF Spaces)
PYTHONPATH=. streamlit run ui/app.py \
  --server.port 7860 \
  --server.address 0.0.0.0 \
  --server.headless true \
  --server.enableCORS false \
  --server.enableXsrfProtection false