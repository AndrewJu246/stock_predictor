# Stock Predictor

## Project Summary
ML ensemble (GBM + RF + XGBoost) predicting daily/weekly stock direction from 12 data source categories. Optimized for alpha over buy-and-hold, not just accuracy. Self-improving via a resolve → retrain → recalibrate feedback loop.

## Architecture
- **predictor.py** — Core ML engine. `train_model()`, `predict()`, `select_features()`, ensemble voting, probability calibration (Platt scaling), magnitude + class-balance sample weighting, regime detection.
- **data_fetcher.py** — 12 data sources: OHLCV (yfinance), market indices, sector ETFs, Fear & Greed, earnings proximity, insider transactions, FRED macro, fundamentals. Tiered caching (5min market, 24h fundamentals).
- **backtester.py** — Walk-forward backtesting with confidence-proportional position sizing. Returns alpha, Sharpe, max drawdown, win rate, expectancy.
- **sentiment.py** — FinBERT via HuggingFace Inference API, VADER fallback. Requires `HF_TOKEN` in `.env`.
- **app.py** — Streamlit dashboard. Tabs: Overview, Detail, Accuracy, Backtest, Portfolio, Features.
- **worker.py** — Market-hours background worker (6AM-1:30PM PT, Mon-Fri). Predictions every 10min, resolves hourly, retrains + emails at close.
- **db.py** — SQLite storage. Tables: predictions, model_meta, news_cache, daily_sentiment, portfolio, portfolio_snapshots.
- **risk_manager.py** — Stop-loss (ATR-based), position sizing, sector diversification analysis.
- **notifier.py** — Gmail SMTP alerts for strong signals + daily summaries.

## Key Technical Details
- Features: ~70 raw → 30 selected via variance threshold + mutual information (per-fold, no leakage)
- Sample weights: `|return| * class_balance`, clipped at 95th percentile
- Calibration: `CalibratedClassifierCV(cv=TimeSeriesSplit(3), method='sigmoid')` — sklearn 1.8 compatible
- Position sizing: `min(1.0, max(0.0, (confidence - 0.5) * 3))`
- Noise filter: drops training samples where `|return| < 0.3%` daily / `0.5%` weekly
- Regime: bull/bear/sideways from S&P vs 200d SMA + VIX + yield curve

## Environment
- Python 3.12, sklearn 1.8, xgboost
- Secrets in `.env`: `HF_TOKEN`, `FRED_API_KEY`, `EMAIL_FROM/PASSWORD/TO`, `EMAIL_ENABLED`
- Production runs from `/Users/andrew/stock_predictor/` (separate from this git repo)
- Git branch: `Local-Deploy`

## Conventions
- All changes go to worktree → commit → merge to `Local-Deploy`
- Use `/challenge` for deep review (3 ranked suggestions), auto-triggers light mode after feature completion
- Use `/inspect` for bug diagnosis
- `MODEL.md` has the full shareable technical overview
- Never hardcode API keys — always `python-dotenv`
