#!/bin/bash

echo "Starting FastAPI backend..."
uvicorn api.main:app --host 0.0.0.0 --port 8000 &

sleep 5

echo "Starting Streamlit UI..."
streamlit run ui/app.py \
  --server.port 7860 \
  --server.address 0.0.0.0 \
  --server.enableCORS false \
  --server.enableXsrfProtection false