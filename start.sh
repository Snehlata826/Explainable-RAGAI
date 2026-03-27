#!/bin/bash
# start.sh
# Run FastAPI backend in the background on port 8000
uvicorn api.main:app --host 0.0.0.0 --port 8000 &

# Wait briefly for FastAPI to initialize
sleep 5

# Run Streamlit frontend in the foreground on HF's required port 7860
streamlit run ui/app.py --server.port 7860 --server.address 0.0.0.0
