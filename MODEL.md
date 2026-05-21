# Stock Predictor — Model Overview

A machine learning system that predicts short-term stock price direction and sizes positions by conviction to generate alpha over buy-and-hold.

---

## What It Predicts

Binary classification: will the stock close **higher** or **lower** than today?

```
y = 1  if  Close(t+1) > Close(t)
y = 0  otherwise
```

Two horizons are supported: **next-day** (1 trading day) and **weekly** (5 trading days).

---

## Data Sources

Each prediction draws from 12 categories of input data:

| Category | Source | Examples |
|---|---|---|
| Price & Volume | Yahoo Finance | OHLCV, returns, log volume |
| Technical Indicators | Computed | RSI, MACD, Bollinger Bands, ATR, OBV, Stochastic |
| Market Indices | Yahoo Finance | S&P 500, VIX, USD Index |
| Treasury & Yield Curve | Yahoo Finance | 3M/5Y/10Y yields, yield curve spread, inversion flag |
| Sector Relative Strength | Yahoo Finance | Sector ETF vs. stock performance |
| Fear & Greed Index | CNN | Score, 5-day MA, extreme fear/greed flags |
| Earnings Proximity | Yahoo Finance | Days to/since earnings, pre/post-earnings flags |
| Insider Transactions | Yahoo Finance | Net buys/sells, 30-day rolling activity, signal |
| Macro (FRED) | Federal Reserve | CPI (YoY/MoM), unemployment, fed funds rate |
| Fundamentals | Yahoo Finance | P/E, PEG, margins, ROE, debt-to-equity, EPS |
| News Sentiment | HuggingFace (FinBERT) | Article sentiment scores, 5-day MA, momentum |
| Market Regime | Computed | Bull/bear/sideways classification (see below) |

---

## Feature Engineering

### Raw Features (~70)

Built from the data sources above. Includes momentum indicators, moving average crossovers, volatility ratios, relative strength vs. market and sector, and seasonality (day of week, month).

### Feature Selection (70 -> 30)

Two-stage reduction to prevent overfitting:

1. **Variance filter** — removes near-constant features (threshold: 1e-6)
2. **Mutual information ranking** — scores each feature by `I(X; Y)` and keeps the top 30

Feature selection runs independently inside each cross-validation fold to prevent data leakage.

---

## Model Architecture

### Ensemble of Three Models

| Model | Type | Key Parameters |
|---|---|---|
| Gradient Boosting | Sequential trees | 100 estimators, max depth 4 |
| Random Forest | Parallel trees | 150 estimators, max depth 6 |
| XGBoost | Gradient-boosted trees | 100 estimators, max depth 4 |

### Ensemble Voting

Each model outputs a calibrated probability of "up." The final prediction is a weighted average:

```
P(up) = sum(w_i * p_i) / sum(w_i)
```

- `p_i` = model i's calibrated probability of "up"
- `w_i` = model i's cross-validation accuracy (floored at 0.5)

If `P(up) > 0.5`, predict **up**; otherwise predict **down**.
Confidence = `P(up)` if predicting up, else `1 - P(up)`.

### Probability Calibration

Raw `predict_proba` outputs from tree models are systematically overconfident. Each model is wrapped in Platt scaling (sigmoid calibration) via `CalibratedClassifierCV` with time-series cross-validation. This maps raw scores to empirically accurate probabilities.

---

## Training

### Sample Weighting

Each training sample receives a weight combining two factors:

```
weight_j = magnitude_j * class_balance_j
```

- **Magnitude**: `|actual return|`, clipped at the 95th percentile and normalized to mean 1. Big moves influence the model more than flat days.
- **Class balance**: `N / (2 * N_class)`. Prevents bull-market bias where the model learns to always predict "up."

### Noise Filtering

Days where the actual return is below a threshold are excluded from training:
- Next-day: `|return| < 0.3%`
- Weekly: `|return| < 0.5%`

These are effectively coin-flip labels that degrade the training signal.

### Cross-Validation

5-fold `TimeSeriesSplit` (respects temporal ordering — no future data leaks into training). Feature selection runs inside each fold independently.

---

## Market Regime Detection

Each trading day is classified as **bull**, **sideways**, or **bear** using three signals:

| Signal | Bull | Bear |
|---|---|---|
| S&P 500 vs. 200-day SMA | Above | Below |
| VIX level | < 25 | > 35 |
| Yield curve | Normal | Inverted |

A composite regime score determines the classification:
- Score >= 2: **Bull**
- Score <= -1: **Bear**
- Otherwise: **Sideways**

Regime affects signal thresholds: bear markets require +5% higher confidence for signals; bull markets relax thresholds by -3%.

---

## Position Sizing

Positions scale linearly with confidence (not binary all-in or all-cash):

```
position_size = min(1.0, max(0.0, (confidence - 0.5) * 3))
```

| Confidence | Position Size |
|---|---|
| 50% | 0% (cash) |
| 60% | 30% |
| 67% | 50% |
| 75% | 75% |
| 83%+ | 100% |

---

## Evaluation Metrics

The system optimizes for **market outperformance**, not just directional accuracy.

| Metric | Formula | What It Measures |
|---|---|---|
| Alpha | Strategy return - buy & hold return | Outperformance vs. market |
| Sharpe Ratio | sqrt(252) * mean(r) / std(r) | Risk-adjusted returns (annualized) |
| Max Drawdown | Largest peak-to-trough decline | Worst-case loss |
| Win Rate | Correct predictions / total | Directional accuracy |
| Expectancy | (win_rate * avg_win) - (loss_rate * avg_loss) | Expected value per trade |

---

## Self-Improvement Loop

The system tracks its own performance and adapts:

1. **Log** — every prediction is saved with ticker, direction, confidence, and timestamp
2. **Resolve** — after the prediction horizon passes, actual market data is compared to the prediction
3. **Retrain** — models retrain with the latest data after enough new outcomes accumulate
4. **Recalibrate** — confidence thresholds update based on empirical accuracy per confidence bin

This cycle runs automatically via a background worker during market hours.

---

## Backtesting

Walk-forward simulation mirrors live trading:

1. Train on the first N days
2. Predict the next day
3. Record whether the prediction was correct
4. Slide the window forward and repeat

The backtester uses identical models, features, and sample weights as production to ensure honest results.

---

## Tech Stack

- **Language**: Python 3.12
- **ML**: scikit-learn, XGBoost
- **Data**: yfinance, FRED API, HuggingFace Inference API
- **Sentiment**: FinBERT (via HuggingFace), VADER fallback
- **Storage**: SQLite
- **Dashboard**: Streamlit
- **Scheduling**: launchd (macOS), market-hours-aware worker
