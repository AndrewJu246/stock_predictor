#!/bin/bash
# install_service.sh — Install or uninstall the Stock Predictor background service.
#
# Usage:
#   bash install_service.sh install    — Start on login, run during market hours
#   bash install_service.sh uninstall  — Stop and remove the service
#   bash install_service.sh status     — Check if it's running
#   bash install_service.sh logs       — View recent logs

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
PLIST_NAME="com.stockpredictor.worker"
PLIST_SRC="$APP_DIR/$PLIST_NAME.plist"
PLIST_DST="$HOME/Library/LaunchAgents/$PLIST_NAME.plist"
LOG_FILE="$APP_DIR/data/worker.log"

case "${1:-help}" in

install)
    echo "Installing Stock Predictor service..."
    echo "  App directory: $APP_DIR"

    # Make sure venv exists
    if [ ! -d "$APP_DIR/venv" ]; then
        echo "  Setting up venv first..."
        bash "$APP_DIR/run.sh" &
        sleep 5
        kill %1 2>/dev/null
    fi

    mkdir -p "$APP_DIR/data"
    mkdir -p "$HOME/Library/LaunchAgents"

    # Update plist with actual path
    sed "s|__APP_DIR__|$APP_DIR|g" "$PLIST_SRC" > "$PLIST_DST"

    # Unload if already loaded
    launchctl unload "$PLIST_DST" 2>/dev/null

    # Load
    launchctl load "$PLIST_DST"

    echo ""
    echo "  ✅ Service installed and running!"
    echo ""
    echo "  What happens now:"
    echo "    • Worker starts automatically when you log in"
    echo "    • During market hours (Mon-Fri 6AM-2PM PT): predictions run every 10 min"
    echo "    • Your Mac stays awake during market hours, sleeps normally after"
    echo "    • Email alerts sent for strong signals"
    echo "    • Daily summary emailed at market close"
    echo ""
    echo "  Commands:"
    echo "    bash install_service.sh status   — Check if running"
    echo "    bash install_service.sh logs      — View logs"
    echo "    bash install_service.sh uninstall — Stop and remove"
    echo ""
    ;;

uninstall)
    echo "Uninstalling Stock Predictor service..."
    launchctl unload "$PLIST_DST" 2>/dev/null
    rm -f "$PLIST_DST"
    echo "  ✅ Service stopped and removed."
    echo "  Your Mac will no longer run predictions automatically."
    ;;

status)
    if launchctl list | grep -q "$PLIST_NAME"; then
        echo "✅ Service is RUNNING"
        echo ""
        # Show PID
        PID=$(launchctl list | grep "$PLIST_NAME" | awk '{print $1}')
        if [ "$PID" != "-" ] && [ -n "$PID" ]; then
            echo "  PID: $PID"
        fi
        # Show last few log lines
        if [ -f "$LOG_FILE" ]; then
            echo ""
            echo "  Last 5 log entries:"
            tail -5 "$LOG_FILE" | sed 's/^/    /'
        fi
    else
        echo "❌ Service is NOT running"
        echo "  Run: bash install_service.sh install"
    fi
    ;;

logs)
    if [ -f "$LOG_FILE" ]; then
        echo "Showing last 50 log entries (Ctrl+C to exit live view):"
        echo "---"
        tail -50 "$LOG_FILE"
        echo "---"
        echo ""
        echo "For live logs: tail -f $LOG_FILE"
    else
        echo "No logs yet. Is the service running?"
        echo "  Run: bash install_service.sh status"
    fi
    ;;

*)
    echo "Stock Predictor Service Manager"
    echo ""
    echo "Usage:"
    echo "  bash install_service.sh install    — Install and start"
    echo "  bash install_service.sh uninstall  — Stop and remove"
    echo "  bash install_service.sh status     — Check if running"
    echo "  bash install_service.sh logs       — View recent logs"
    ;;

esac
