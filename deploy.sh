#!/bin/bash
# deploy.sh — Oracle Cloud deployment script
# Sets up the Stock Predictor with background worker + optional web dashboard
#
# Usage: bash deploy.sh
# Run on a fresh Oracle Cloud Always Free Ubuntu instance

set -e

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_USER="$(whoami)"
VENV_DIR="$APP_DIR/venv"

echo "=================================================="
echo "  Stock Predictor — Oracle Cloud Deployment"
echo "=================================================="
echo "App directory: $APP_DIR"
echo "User: $APP_USER"
echo ""

# ── Step 1: System dependencies ──────────────────────────────────────────────
echo "[1/5] Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq python3 python3-pip python3-venv sqlite3 > /dev/null 2>&1
echo "  Done."

# ── Step 2: Python venv ──────────────────────────────────────────────────────
echo "[2/5] Setting up Python environment..."
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
pip install -q --upgrade pip
pip install -q -r "$APP_DIR/requirements.txt"
echo "  Done. Python $(python3 --version)"

# ── Step 3: Create data directories ─────────────────────────────────────────
echo "[3/5] Creating data directories..."
mkdir -p "$APP_DIR/data/models"
echo "  Done."

# ── Step 4: Create systemd service — Worker ─────────────────────────────────
echo "[4/5] Creating systemd services..."

# Worker service (background predictions)
sudo tee /etc/systemd/system/stock-predictor-worker.service > /dev/null << EOF
[Unit]
Description=Stock Predictor Background Worker
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$APP_USER
WorkingDirectory=$APP_DIR
Environment="PATH=$VENV_DIR/bin:/usr/local/bin:/usr/bin"
ExecStart=$VENV_DIR/bin/python3 $APP_DIR/worker.py
Restart=always
RestartSec=30
StandardOutput=journal
StandardError=journal

# Graceful shutdown
KillSignal=SIGTERM
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
EOF

# Web dashboard service (optional, for remote access)
sudo tee /etc/systemd/system/stock-predictor-web.service > /dev/null << EOF
[Unit]
Description=Stock Predictor Web Dashboard
After=network-online.target stock-predictor-worker.service
Wants=network-online.target

[Service]
Type=simple
User=$APP_USER
WorkingDirectory=$APP_DIR
Environment="PATH=$VENV_DIR/bin:/usr/local/bin:/usr/bin"
ExecStart=$VENV_DIR/bin/streamlit run $APP_DIR/app.py --server.port 8501 --server.headless true --server.address 0.0.0.0
Restart=always
RestartSec=30
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

echo "  Services created."

# ── Step 5: Enable and start ─────────────────────────────────────────────────
echo "[5/5] Enabling services..."
sudo systemctl daemon-reload
sudo systemctl enable stock-predictor-worker.service
sudo systemctl enable stock-predictor-web.service
sudo systemctl start stock-predictor-worker.service
sudo systemctl start stock-predictor-web.service

echo ""
echo "=================================================="
echo "  Deployment Complete!"
echo "=================================================="
echo ""
echo "Services running:"
echo "  Worker:    sudo systemctl status stock-predictor-worker"
echo "  Dashboard: sudo systemctl status stock-predictor-web"
echo ""
echo "View logs:"
echo "  Worker:    sudo journalctl -u stock-predictor-worker -f"
echo "  Dashboard: sudo journalctl -u stock-predictor-web -f"
echo ""
echo "Manage:"
echo "  Stop:      sudo systemctl stop stock-predictor-worker"
echo "  Restart:   sudo systemctl restart stock-predictor-worker"
echo "  Disable:   sudo systemctl disable stock-predictor-worker"
echo ""
echo "Dashboard accessible at: http://<your-server-ip>:8501"
echo "  (Make sure port 8501 is open in Oracle Cloud security rules)"
echo ""
