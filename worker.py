"""
worker.py — Background worker that runs the prediction pipeline on a schedule.
Designed to run 24/7 as a systemd service.

Schedule:
  - Every 10 min: Fetch data, run predictions, check for strong signals
  - Every 1 hour: Resolve old predictions against actual outcomes
  - Every 24 hours: Retrain models, send daily summary
"""

import time
import sys
import signal
from datetime import datetime, timedelta
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent))

import db
from data_fetcher import get_watchlist
from predictor import predict, predict_all, resolve_predictions, train_model
from sentiment import get_ticker_sentiment
from notifier import send_signal_alert, send_daily_summary

# ── Schedule timers ──────────────────────────────────────────────────────────

PREDICT_INTERVAL = 10 * 60       # 10 minutes
RESOLVE_INTERVAL = 60 * 60       # 1 hour
RETRAIN_INTERVAL = 24 * 60 * 60  # 24 hours
SUMMARY_HOUR = 18                # Send daily summary at 6 PM local time

running = True


def shutdown_handler(signum, frame):
    global running
    print(f"\n[INFO] Received signal {signum}, shutting down gracefully...")
    running = False


signal.signal(signal.SIGINT, shutdown_handler)
signal.signal(signal.SIGTERM, shutdown_handler)


# ── Worker tasks ─────────────────────────────────────────────────────────────

def run_predictions():
    """Fetch data and run predictions for all watchlist tickers."""
    watchlist = get_watchlist()
    if not watchlist:
        print("[WARN] Watchlist is empty.")
        return []

    print(f"[{_now()}] Running predictions for {len(watchlist)} tickers...")

    all_predictions = []
    for horizon in ["next_day", "weekly"]:
        predictions = predict_all(horizon)
        all_predictions.extend(predictions)

        # Count results
        ok = sum(1 for p in predictions if "error" not in p)
        err = sum(1 for p in predictions if "error" in p)
        print(f"  {horizon}: {ok} predictions, {err} errors")

    # Check for strong signals and notify
    send_signal_alert(all_predictions)

    return all_predictions


def run_resolve():
    """Resolve old predictions against actual market data."""
    print(f"[{_now()}] Resolving past predictions...")
    count = resolve_predictions()
    print(f"  Resolved {count} predictions")
    return count


def run_retrain():
    """Retrain all models with latest data."""
    watchlist = get_watchlist()
    print(f"[{_now()}] Retraining models for {len(watchlist)} tickers...")

    for ticker in watchlist:
        for horizon in ["next_day", "weekly"]:
            try:
                result = train_model(ticker, horizon)
                if "error" in result:
                    print(f"  {ticker}/{horizon}: {result['error']}")
                else:
                    acc = result.get("ensemble_accuracy", "?")
                    n = result.get("num_models", "?")
                    print(f"  {ticker}/{horizon}: {acc}% accuracy, {n} models")
            except Exception as e:
                print(f"  {ticker}/{horizon}: ERROR — {e}")

    print(f"  Retraining complete.")


def run_daily_summary():
    """Send daily summary email."""
    print(f"[{_now()}] Sending daily summary...")
    predictions = predict_all("next_day")
    stats = db.get_accuracy_stats(last_n=100)
    send_daily_summary(predictions, stats)


def refresh_sentiment():
    """Refresh news sentiment for all watchlist tickers."""
    watchlist = get_watchlist()
    for ticker in watchlist:
        try:
            get_ticker_sentiment(ticker, use_cache=False)
        except Exception:
            pass


def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ── Main loop ────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print(f"  Stock Predictor Worker")
    print(f"  Started at: {_now()}")
    print(f"  Predictions: every {PREDICT_INTERVAL // 60} min")
    print(f"  Resolve: every {RESOLVE_INTERVAL // 60} min")
    print(f"  Retrain: every {RETRAIN_INTERVAL // 3600} hours")
    print(f"  Daily summary: at {SUMMARY_HOUR}:00")
    print("=" * 60)

    last_predict = 0
    last_resolve = 0
    last_retrain = 0
    last_summary_date = None

    # Run immediately on start
    try:
        refresh_sentiment()
        run_predictions()
        run_resolve()
    except Exception as e:
        print(f"[ERROR] Initial run failed: {e}")

    while running:
        now = time.time()
        current_dt = datetime.now()

        try:
            # Predictions (every 10 min)
            if now - last_predict >= PREDICT_INTERVAL:
                refresh_sentiment()
                run_predictions()
                last_predict = now

            # Resolve (every 1 hour)
            if now - last_resolve >= RESOLVE_INTERVAL:
                run_resolve()
                last_resolve = now

            # Retrain (every 24 hours)
            if now - last_retrain >= RETRAIN_INTERVAL:
                run_retrain()
                last_retrain = now

            # Daily summary (once per day at SUMMARY_HOUR)
            today = current_dt.date()
            if (current_dt.hour >= SUMMARY_HOUR and
                    last_summary_date != today):
                run_daily_summary()
                last_summary_date = today

        except Exception as e:
            print(f"[ERROR] {_now()}: {e}")

        # Sleep in short intervals so we can catch shutdown signals
        for _ in range(30):  # 30 seconds total
            if not running:
                break
            time.sleep(1)

    print(f"[{_now()}] Worker stopped.")


if __name__ == "__main__":
    main()
