#!/bin/bash
set -e

echo "=== Starting DeliveryETA Application ==="

# Check if model artifacts exist
if [ ! -f "/workspace/models/model.pkl" ] || [ ! -f "/workspace/models/scaler.pkl" ]; then
    echo "Model artifacts not found. Training and saving best Random Forest model..."
    python /workspace/scripts/train_pipeline.py
else
    echo "Model artifacts found in /workspace/models."
fi

PORT=${STREAMLIT_SERVER_PORT:-8501}
ADDRESS=${STREAMLIT_SERVER_ADDRESS:-0.0.0.0}

echo "Launching Streamlit application on ${ADDRESS}:${PORT}..."
exec streamlit run /workspace/app/app.py \
    --server.port="${PORT}" \
    --server.address="${ADDRESS}" \
    --server.headless=true \
    --browser.gatherUsageStats=false
