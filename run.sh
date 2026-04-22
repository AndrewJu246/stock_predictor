#!/bin/bash
# run.sh — Setup and launch the Stock Predictor
# Usage: bash run.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="venv"
PYTHON="python3.12"

# Check if python3.12 exists, fall back to python3
if ! command -v $PYTHON &> /dev/null; then
    PYTHON="python3"
    echo "python3.12 not found, using $(python3 --version)"
fi

# Create venv if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    $PYTHON -m venv "$VENV_DIR"
fi

# Activate
source "$VENV_DIR/bin/activate"

# Install dependencies
echo "Checking dependencies..."
pip install -q -r requirements.txt 2>/dev/null

# Ensure data directory exists
mkdir -p data/models

# Run the app
echo ""
echo "Starting Stock Predictor..."
echo "Dashboard will open at http://localhost:8501"
echo ""
streamlit run app.py --server.headless true
