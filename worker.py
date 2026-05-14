"""
worker.py — Background worker with market-hours scheduling.
Designed for Railway deployment (conserves free-tier hours).

Schedule (all times Pacific):
  Market hours (Mon-Fri 6:00AM - 1:30PM PT):
    - Every 10 min: Fetch data, run predictions, check for strong signals
    - Every 1 hour: Resolve old predictions against actual outcomes
  
  Market close (1:30 PM PT):
    - Retrain models
    - Send daily summary email
  
  Off-hours / weekends:
    - Sleep (minimal resource usage)
"""

import time
import sys
import signal
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent))

import db
from data_fetcher import get_watchlist
from predictor import predict, predict_all, resolve_predictions, train_model, _prewarm_shared_cache
from sentiment import get_ticker_sentiment
from notifier import send_daily_summary

# ── Config ───────────────────────────────────────────────────────────────────

MARKET_TZ = ZoneInfo("America/Los_Angeles")
MARKET_OPEN_HOUR = 6      # 6:00 AM PT (30 min before market open)
MARKET_OPEN_MIN = 0
MARKET_CLOSE_HOUR = 13    # 1:00 PM PT (market close)
MARKET_CLOSE_MIN = 30     # 1:30 PM PT (30 min buffer after close)

PREDICT_INTERVAL = 10 * 60       # 10 minutes
RESOLVE_INTERVAL = 60 * 60       # 1 hour
OFF_HOURS_SLEEP = 15 * 60        # 15 min sleep when market closed

running = True


def shutdown_handler(signum, frame):
    global running
    print(f"\n[INFO] Received signal {signum}, shutting down gracefully...")
    running = False


signal.signal(signal.SIGINT, shutdown_handler)
signal.signal(signal.SIGTERM, shutdown_handler)


# ── Market hours logic ───────────────────────────────────────────────────────

def _now_pt():
    """Current time in Pacific."""
    return datetime.now(MARKET_TZ)


def _is_market_hours():
    """Check if we're in active market hours (weekdays, 6:00AM - 1:30PM PT)."""
    now = _now_pt()

    # Weekdays only (0=Mon, 4=Fri)
    if now.weekday() > 4:
        return False

    market_open = now.replace(hour=MARKET_OPEN_HOUR, minute=MARKET_OPEN_MIN,
                               second=0, microsecond=0)
    market_close = now.replace(hour=MARKET_CLOSE_HOUR, minute=MARKET_CLOSE_MIN,
                                second=0, microsecond=0)

    return market_open <= now <= market_close


def _minutes_until_market_open():
    """Calculate minutes until next market open."""
    now = _now_pt()

    # Find next weekday
    next_open = now.replace(hour=MARKET_OPEN_HOUR, minute=MARKET_OPEN_MIN,
                             second=0, microsecond=0)

    if now >= next_open:
        next_open += timedelta(days=1)

    # Skip weekends
    while next_open.weekday() > 4:
        next_open += timedelta(days=1)

    diff = (next_open - now).total_seconds() / 60
    return int(diff)


def _now_str():
    return _now_pt().strftime("%Y-%m-%d %H:%M:%S PT")


# ── Worker tasks ─────────────────────────────────────────────────────────────

def run_predictions():
    """Fetch data and run predictions for all watchlist tickers."""
    watchlist = get_watchlist()
    if not watchlist:
        print("[WARN] Watchlist is empty.")
        return []

    print(f"[{_now_str()}] Running predictions for {len(watchlist)} tickers...")

    all_predictions = []
    for horizon in ["next_day", "weekly"]:
        predictions = predict_all(horizon)
        all_predictions.extend(predictions)

        ok = sum(1 for p in predictions if "error" not in p)
        err = sum(1 for p in predictions if "error" in p)
        print(f"  {horizon}: {ok} predictions, {err} errors")

    return all_predictions


def run_resolve():
    """Resolve old predictions against actual market data."""
    print(f"[{_now_str()}] Resolving past predictions...")
    count = resolve_predictions()
    print(f"  Resolved {count} predictions")
    return count


def run_retrain():
    """Retrain all models with latest data."""
    watchlist = get_watchlist()
    print(f"[{_now_str()}] Retraining models for {len(watchlist)} tickers...")
    _prewarm_shared_cache("2y")

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

    print("  Retraining complete.")


def run_daily_summary():
    """Send daily summary email."""
    print(f"[{_now_str()}] Sending daily summary...")
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


# ── Main loop ────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Stock Predictor Worker (Market Hours Mode)")
    print(f"  Started at: {_now_str()}")
    print(f"  Market hours: {MARKET_OPEN_HOUR}:{MARKET_OPEN_MIN:02d} - "
          f"{MARKET_CLOSE_HOUR}:{MARKET_CLOSE_MIN:02d} PT, Mon-Fri")
    print(f"  Prediction interval: {PREDICT_INTERVAL // 60} min")
    print("=" * 60)

    last_predict = 0
    last_resolve = 0
    last_summary_date = None

    while running:
        now = time.time()
        now_pt = _now_pt()

        if _is_market_hours():
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

            except Exception as e:
                print(f"[ERROR] {_now_str()}: {e}")

            # Sleep 30 seconds between checks
            for _ in range(30):
                if not running:
                    break
                time.sleep(1)

        else:
            # Off-hours: do end-of-day tasks once, then sleep
            today = now_pt.date()
            if (now_pt.hour >= MARKET_CLOSE_HOUR and
                    last_summary_date != today and
                    now_pt.weekday() <= 4):
                # Market just closed — run end-of-day tasks
                try:
                    print(f"[{_now_str()}] Market closed. Running end-of-day tasks...")
                    run_resolve()
                    run_retrain()
                    run_daily_summary()
                    last_summary_date = today
                except Exception as e:
                    print(f"[ERROR] End-of-day: {e}")

            # Sleep longer off-hours to save resources
            mins_until = _minutes_until_market_open()
            print(f"[{_now_str()}] Market closed. Next open in ~{mins_until} min. "
                  f"Sleeping {OFF_HOURS_SLEEP // 60} min...")

            for _ in range(OFF_HOURS_SLEEP):
                if not running:
                    break
                time.sleep(1)

    print(f"[{_now_str()}] Worker stopped.")


if __name__ == "__main__":
    main()
