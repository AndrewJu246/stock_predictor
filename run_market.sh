#!/bin/bash
# run_market.sh — Keeps Mac awake during market hours and runs the worker.
#
# How it works:
#   - During market hours (Mon-Fri, 6AM-2PM PT): prevents Mac sleep, runs predictions
#   - After hours: releases Mac to sleep normally
#   - Loops forever — launchd keeps it alive across reboots
#
# Usage (manual):   bash run_market.sh
# Usage (auto):     Installed via launchd (see install instructions below)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Activate venv
PYTHON="$SCRIPT_DIR/venv/bin/python3"
if [ ! -f "$PYTHON" ]; then
    echo "[ERROR] venv not found. Run 'bash run.sh' first to set up."
    exit 1
fi

LOG_FILE="$SCRIPT_DIR/data/worker.log"
mkdir -p "$SCRIPT_DIR/data"

log() {
    echo "[$(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S PT')] $1" | tee -a "$LOG_FILE"
}

is_market_hours() {
    local HOUR=$(TZ=America/Los_Angeles date +%H)
    local DOW=$(TZ=America/Los_Angeles date +%u)  # 1=Mon, 7=Sun

    # Weekdays (1-5) and between 6:00 AM and 2:00 PM PT
    if [ "$DOW" -le 5 ] && [ "$HOUR" -ge 6 ] && [ "$HOUR" -lt 14 ]; then
        return 0  # true
    fi
    return 1  # false
}

CAFF_PID=""

cleanup() {
    log "Shutting down..."
    if [ -n "$CAFF_PID" ] && kill -0 "$CAFF_PID" 2>/dev/null; then
        kill "$CAFF_PID" 2>/dev/null
    fi
    if [ -n "$WORKER_PID" ] && kill -0 "$WORKER_PID" 2>/dev/null; then
        kill "$WORKER_PID" 2>/dev/null
    fi
    exit 0
}

trap cleanup SIGINT SIGTERM

log "========================================="
log "Stock Predictor — Market Hours Runner"
log "========================================="

while true; do
    if is_market_hours; then
        # Prevent idle + display sleep while worker runs
        if [ -z "$CAFF_PID" ] || ! kill -0 "$CAFF_PID" 2>/dev/null; then
            caffeinate -is &
            CAFF_PID=$!
            log "Caffeinate started (PID $CAFF_PID) — Mac will stay awake"
        fi

        # Start worker if not running
        if [ -z "$WORKER_PID" ] || ! kill -0 "$WORKER_PID" 2>/dev/null; then
            log "Starting worker..."
            "$PYTHON" "$SCRIPT_DIR/worker.py" >> "$LOG_FILE" 2>&1 &
            WORKER_PID=$!
            log "Worker started (PID $WORKER_PID)"
        fi

        # Check every 5 minutes
        sleep 300

    else
        # Market closed — stop worker and release caffeinate
        if [ -n "$WORKER_PID" ] && kill -0 "$WORKER_PID" 2>/dev/null; then
            log "Market closed — stopping worker"
            kill "$WORKER_PID" 2>/dev/null
            wait "$WORKER_PID" 2>/dev/null
            WORKER_PID=""
        fi

        if [ -n "$CAFF_PID" ] && kill -0 "$CAFF_PID" 2>/dev/null; then
            kill "$CAFF_PID" 2>/dev/null
            CAFF_PID=""
            log "Caffeinate released — Mac can sleep"
        fi

        # Sleep 15 min then check again
        # (Mac may sleep during this — that's fine, it picks up when it wakes)
        log "Off hours — sleeping 15 min"
        sleep 900
    fi
done
