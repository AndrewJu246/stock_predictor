#!/bin/bash
# start.sh — Runs worker (background) + dashboard (foreground)
# Used as Railway's single entry point

set -e

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$APP_DIR"

# Ensure data directories exist
mkdir -p data/models

# Start worker in background
echo "[start.sh] Starting worker..."
python worker.py &
WORKER_PID=$!

# Start Streamlit dashboard in foreground
echo "[start.sh] Starting dashboard on port ${PORT:-8501}..."
streamlit run app.py \
    --server.port "${PORT:-8501}" \
    --server.headless true \
    --server.address 0.0.0.0 \
    --server.fileWatcherType none &
WEB_PID=$!

# Handle shutdown — kill both when either exits
cleanup() {
    echo "[start.sh] Shutting down..."
    kill $WORKER_PID 2>/dev/null
    kill $WEB_PID 2>/dev/null
    wait
    echo "[start.sh] Stopped."
}

trap cleanup SIGTERM SIGINT EXIT

# Wait for either process to exit
wait -n $WORKER_PID $WEB_PID
cleanup
