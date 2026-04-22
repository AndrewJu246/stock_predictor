# Stock Predictor

A self-improving stock prediction dashboard that combines technical analysis with news sentiment to forecast price movements.

## Quick Start

```bash
bash run.sh
```

That's it. The script creates a virtual environment, installs dependencies, and opens the dashboard at `http://localhost:8501`.

## What It Does

**Predictions** — Uses a Gradient Boosting model trained on 20+ technical indicators (RSI, MACD, Bollinger Bands, SMA crossovers, volume ratios, volatility) plus news sentiment analysis. Generates next-day and weekly directional forecasts with confidence scores.

**Self-improvement** — Every prediction is logged. When you click "Resolve Past Predictions," the system compares old predictions against what actually happened, calculates accuracy, and uses this feedback to retrain the models. Over time, the model gets more data and should improve.

**News Sentiment** — Pulls headlines from Yahoo Finance and Google News RSS feeds (no API key needed). Analyzes them with VADER sentiment analysis, enhanced with a finance-specific lexicon (terms like "bullish," "downgrade," "surge" get weighted appropriately).

## Dashboard Tabs

- **Overview** — All your watchlist stocks at a glance: price, change, signal (Buy/Sell/Hold), confidence, sentiment, and accuracy stats.
- **Stock Detail** — Candlestick chart with SMA overlays, per-stock predictions, and news sentiment breakdown with clickable headlines.
- **Accuracy** — Track how well the model is doing over time. Shows cumulative accuracy chart and full prediction history.

## Managing Your Watchlist

Use the sidebar to add/remove tickers. Default: AAPL, MSFT, GOOGL, AMZN, TSLA.

## File Structure

```
stock_predictor/
├── app.py              # Streamlit dashboard
├── data_fetcher.py     # Stock prices (yfinance) + news (RSS)
├── sentiment.py        # VADER sentiment with finance lexicon
├── predictor.py        # ML model, training, prediction, self-improvement
├── db.py               # SQLite database for tracking everything
├── config.json         # Watchlist + settings
├── requirements.txt    # Dependencies
├── run.sh              # One-command setup + launch
└── data/               # Auto-created: database + saved models
    ├── predictor.db
    └── models/
```

## Requirements

- Python 3.10+ (3.12 recommended)
- Internet connection (for stock data and news)
- No API keys needed

## Notes

- First run takes a minute — it's training models for each stock in your watchlist.
- Predictions improve over time as the system accumulates resolved prediction data.
- This is for personal/educational use only — not financial advice.
