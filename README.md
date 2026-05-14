# Stock Predictor

A self-improving stock prediction dashboard that combines a multi-model ML ensemble with news sentiment and market context to forecast price movements.

## Quick Start

```bash
cp .env.example .env   # add your HF token + email creds
bash run.sh
```

The script creates a virtual environment, installs dependencies, and opens the dashboard at `http://localhost:8501`.

## What It Does

**Predictions** — 3-model ensemble (Gradient Boosting + Random Forest + XGBoost) trained on 40+ features: technical indicators (RSI, MACD, Bollinger Bands, SMA crossovers, volume ratios, volatility), market context (S&P 500, VIX, Treasury yields), sector relative strength, CNN Fear & Greed Index, and news sentiment. Generates next-day and weekly directional forecasts with confidence scores and consensus voting.

**Self-improvement** — Every prediction is logged to SQLite. The system automatically compares past predictions against actual outcomes, calculates accuracy, and retrains models using this feedback loop.

**News Sentiment** — Analyzes headlines from Yahoo Finance and Google News RSS feeds using FinBERT (via HuggingFace Inference API, free) with VADER as fallback. Finance-specific lexicon weights terms like "bullish," "downgrade," and "surge."

**Risk Management** — ATR-based stop-loss suggestions, position sizing calculator, and portfolio diversification analysis with sector concentration warnings.

**Background Worker** — Runs predictions every 10 minutes during market hours (Mon-Fri 6AM-2PM PT), retrains at market close, and sends email alerts for strong signals plus a daily summary.

## Dashboard Tabs

- **Overview** — All watchlist stocks at a glance: price, change, signal, confidence, consensus, sentiment, and accuracy
- **Stock Detail** — Candlestick chart with SMA overlays, per-stock predictions, news headlines, and risk analysis
- **Accuracy** — Cumulative accuracy chart and full prediction history
- **Backtest** — Walk-forward historical simulation with equity curves
- **Portfolio** — Track positions, P&L, diversification, and portfolio value over time
- **Features** — See which factors drive predictions most, broken down by category

## Managing Your Watchlist

Use the sidebar to add/remove tickers. Default: AAPL, MSFT, GOOGL, AMZN, TSLA.

## Background Worker

```bash
bash run_market.sh              # run manually
bash install_service.sh install # auto-start on login via launchd
bash install_service.sh status  # check if running
bash install_service.sh logs    # view logs
```

## File Structure

```
stock_predictor/
├── app.py              # Streamlit dashboard
├── predictor.py        # Multi-model ensemble, training, self-improvement
├── data_fetcher.py     # Stock data (yfinance), news (RSS), market indicators
├── sentiment.py        # FinBERT (API) + VADER fallback
├── backtester.py       # Walk-forward backtesting engine
├── risk_manager.py     # Stop-loss, position sizing, diversification
├── notifier.py         # Email alerts and daily summaries
├── worker.py           # Background worker with market-hours scheduling
├── db.py               # SQLite storage
├── config.json         # Watchlist + non-secret settings
├── .env.example        # Template for secrets (HF token, email creds)
├── requirements.txt    # Dependencies
├── run.sh              # One-command setup + launch
├── run_market.sh       # Market-hours worker runner
└── install_service.sh  # launchd service installer
```

## Requirements

- Python 3.10+ (3.12 recommended)
- Internet connection (for stock data and news)
- HuggingFace token (free, optional — falls back to VADER without it)

## Notes

- First run takes a minute — it trains models for each stock in your watchlist.
- Predictions improve over time as the system accumulates resolved prediction data.
- This is for personal/educational use only — not financial advice.
