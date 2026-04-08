#!/bin/bash

echo "🚀 Starting FastAPI backend..."
PYTHONPATH=. uvicorn api.main:app --host 0.0.0.0 --port 8005 &

echo "⏳ Waiting for backend to be ready..."
sleep 5

echo "🚀 Starting Streamlit UI..."
PYTHONPATH=. streamlit run ui/app.py \
  --server.port 7860 \
  --server.address 0.0.0.0 \
  --server.headless true \
  --browser.gatherUsageStats false