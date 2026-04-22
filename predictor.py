"""
predictor.py — ML prediction engine using Gradient Boosting.
Features: technical indicators + news sentiment.
Self-improves by retraining with accuracy feedback.
"""

import pickle
import numpy as np
import pandas as pd
import ta
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from datetime import datetime

import db
from data_fetcher import fetch_stock_data, fetch_market_data, fetch_sector_data, fetch_earnings_proximity
from sentiment import get_ticker_sentiment

MODELS_DIR = Path(__file__).parent / "data" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ── Feature engineering ──────────────────────────────────────────────────────

def build_features(df: pd.DataFrame, sentiment_score: float = 0.0) -> pd.DataFrame:
    """Add technical indicator features to OHLCV dataframe.
    If market/sector columns are present in df, those get included too."""
    feat = pd.DataFrame(index=df.index)

    # Price-based
    feat["return_1d"] = df["Close"].pct_change(1)
    feat["return_5d"] = df["Close"].pct_change(5)
    feat["return_10d"] = df["Close"].pct_change(10)
    feat["return_20d"] = df["Close"].pct_change(20)

    # Moving averages
    feat["sma_10"] = ta.trend.sma_indicator(df["Close"], window=10)
    feat["sma_20"] = ta.trend.sma_indicator(df["Close"], window=20)
    feat["sma_50"] = ta.trend.sma_indicator(df["Close"], window=50)
    feat["price_vs_sma10"] = (df["Close"] - feat["sma_10"]) / feat["sma_10"]
    feat["price_vs_sma20"] = (df["Close"] - feat["sma_20"]) / feat["sma_20"]
    feat["price_vs_sma50"] = (df["Close"] - feat["sma_50"]) / feat["sma_50"]
    feat["sma_10_20_cross"] = (feat["sma_10"] > feat["sma_20"]).astype(int)

    # EMA
    feat["ema_12"] = ta.trend.ema_indicator(df["Close"], window=12)
    feat["ema_26"] = ta.trend.ema_indicator(df["Close"], window=26)
    feat["ema_cross"] = (feat["ema_12"] > feat["ema_26"]).astype(int)

    # RSI
    feat["rsi"] = ta.momentum.rsi(df["Close"], window=14)
    feat["rsi_oversold"] = (feat["rsi"] < 30).astype(int)
    feat["rsi_overbought"] = (feat["rsi"] > 70).astype(int)

    # Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(df["High"], df["Low"], df["Close"])
    feat["stoch_k"] = stoch.stoch()
    feat["stoch_d"] = stoch.stoch_signal()

    # MACD
    macd = ta.trend.MACD(df["Close"])
    feat["macd"] = macd.macd()
    feat["macd_signal"] = macd.macd_signal()
    feat["macd_diff"] = macd.macd_diff()
    feat["macd_cross"] = (feat["macd"] > feat["macd_signal"]).astype(int)

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df["Close"], window=20)
    feat["bb_high"] = bb.bollinger_hband_indicator()
    feat["bb_low"] = bb.bollinger_lband_indicator()
    feat["bb_width"] = bb.bollinger_wband()
    feat["bb_pct"] = bb.bollinger_pband()

    # Volume
    feat["volume_sma"] = df["Volume"].rolling(window=10).mean()
    feat["volume_ratio"] = df["Volume"] / feat["volume_sma"]
    feat["volume_trend"] = df["Volume"].pct_change(5)

    # Volatility
    feat["volatility_10d"] = df["Close"].pct_change().rolling(10).std()
    feat["volatility_20d"] = df["Close"].pct_change().rolling(20).std()
    feat["volatility_ratio"] = feat["volatility_10d"] / feat["volatility_20d"]

    # ATR
    feat["atr"] = ta.volatility.average_true_range(df["High"], df["Low"],
                                                     df["Close"], window=14)

    # On-Balance Volume
    feat["obv"] = ta.volume.on_balance_volume(df["Close"], df["Volume"])
    feat["obv_sma"] = feat["obv"].rolling(10).mean()

    # Momentum
    feat["momentum_10"] = ta.momentum.roc(df["Close"], window=10)
    feat["momentum_20"] = ta.momentum.roc(df["Close"], window=20)

    # ── Market-wide features (if present in df) ──────────────────────────
    market_cols = [c for c in df.columns if c.startswith(("sp500_", "vix_", "treasury_", "sector_"))]
    for col in market_cols:
        feat[col] = df[col]

    # Stock vs. market relative strength
    if "sp500_return" in df.columns:
        feat["vs_market"] = feat["return_1d"] - df["sp500_return"]
        feat["vs_market_5d"] = feat["return_5d"] - df["sp500_return_5d"]

    # Sector relative strength
    if "sector_return" in df.columns:
        feat["vs_sector"] = feat["return_1d"] - df["sector_return"]

    # Sentiment
    feat["sentiment"] = sentiment_score

    # Day of week (market patterns)
    feat["day_of_week"] = df.index.dayofweek

    # Month (seasonal patterns)
    feat["month"] = df.index.month

    return feat


def prepare_dataset(ticker: str, horizon: str = "next_day"):
    """
    Build feature matrix X and target y.
    horizon: 'next_day' (1-day forward return) or 'weekly' (5-day forward return)
    """
    df = fetch_stock_data(ticker, period="1y")
    if df.empty or len(df) < 60:
        return None, None, None

    # Merge market-wide data
    try:
        market = fetch_market_data(period="1y")
        if not market.empty:
            df = df.join(market, how="left", rsuffix="_dup")
            # Drop any duplicate columns
            df = df[[c for c in df.columns if not c.endswith("_dup")]]
            df = df.ffill()
    except Exception:
        pass

    # Merge sector data
    try:
        sector = fetch_sector_data(ticker, period="1y")
        if not sector.empty:
            df = df.join(sector, how="left", rsuffix="_dup")
            df = df[[c for c in df.columns if not c.endswith("_dup")]]
            df = df.ffill()
    except Exception:
        pass

    # Get current sentiment
    try:
        sent = get_ticker_sentiment(ticker)
        sentiment_score = sent["avg_sentiment"]
    except Exception:
        sentiment_score = 0.0

    features = build_features(df, sentiment_score)

    # Target: direction of future price movement
    if horizon == "next_day":
        target = (df["Close"].shift(-1) > df["Close"]).astype(int)
        pct_target = df["Close"].pct_change(1).shift(-1)
    else:  # weekly
        target = (df["Close"].shift(-5) > df["Close"]).astype(int)
        pct_target = df["Close"].pct_change(5).shift(-5)

    # Combine and handle NaN
    combined = features.copy()
    combined["target"] = target
    combined["pct_target"] = pct_target

    # Fill NaN features with 0 (early rows where indicators haven't warmed up)
    feature_cols_only = [c for c in combined.columns if c not in ["target", "pct_target"]]
    combined[feature_cols_only] = combined[feature_cols_only].fillna(0)

    # Only drop rows where target is NaN
    combined = combined.dropna(subset=["target", "pct_target"])

    if len(combined) < 30:
        return None, None, None

    X = combined.drop(columns=["target", "pct_target"])
    y = combined["target"]

    return X, y, combined


# ── Model training ───────────────────────────────────────────────────────────

def train_model(ticker: str, horizon: str = "next_day") -> dict:
    """Train a GradientBoosting model for the given ticker and horizon."""
    X, y, combined = prepare_dataset(ticker, horizon)

    if X is None:
        return {"error": f"Not enough data for {ticker}"}

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Time-series cross-validation
    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
    )

    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring="accuracy")

    # Train on full data
    model.fit(X_scaled, y)

    # Save model + scaler
    model_path = MODELS_DIR / f"{ticker}_{horizon}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler, "features": list(X.columns)}, f)

    # Log model metadata
    avg_accuracy = round(cv_scores.mean() * 100, 1)
    db.save_model_meta(ticker, horizon, len(X), avg_accuracy, list(X.columns))

    return {
        "ticker": ticker,
        "horizon": horizon,
        "cv_accuracy": avg_accuracy,
        "cv_std": round(cv_scores.std() * 100, 1),
        "sample_size": len(X),
        "features": list(X.columns),
    }


def load_model(ticker: str, horizon: str = "next_day"):
    """Load a trained model from disk."""
    model_path = MODELS_DIR / f"{ticker}_{horizon}.pkl"
    if not model_path.exists():
        return None
    with open(model_path, "rb") as f:
        return pickle.load(f)


# ── Prediction ───────────────────────────────────────────────────────────────

def predict(ticker: str, horizon: str = "next_day") -> dict:
    """
    Make a prediction for the given ticker.
    Auto-trains if no model exists.
    """
    bundle = load_model(ticker, horizon)

    # Train if needed
    if bundle is None:
        result = train_model(ticker, horizon)
        if "error" in result:
            return result
        bundle = load_model(ticker, horizon)

    model = bundle["model"]
    scaler = bundle["scaler"]
    feature_names = bundle["features"]

    # Build current features
    df = fetch_stock_data(ticker, period="6mo")
    if df.empty:
        return {"error": "No stock data available"}

    # Merge market + sector data
    try:
        market = fetch_market_data(period="6mo")
        if not market.empty:
            df = df.join(market, how="left", rsuffix="_dup")
            df = df[[c for c in df.columns if not c.endswith("_dup")]]
            df = df.ffill()
    except Exception:
        pass
    try:
        sector = fetch_sector_data(ticker, period="6mo")
        if not sector.empty:
            df = df.join(sector, how="left", rsuffix="_dup")
            df = df[[c for c in df.columns if not c.endswith("_dup")]]
            df = df.ffill()
    except Exception:
        pass

    try:
        sent = get_ticker_sentiment(ticker)
        sentiment_score = sent["avg_sentiment"]
    except Exception:
        sentiment_score = 0.0

    features = build_features(df, sentiment_score)
    features = features.fillna(0)

    if features.empty:
        return {"error": "Could not compute features"}

    # Use the latest row
    latest = features.iloc[[-1]][feature_names]
    X_scaled = scaler.transform(latest)

    # Predict
    pred_class = model.predict(X_scaled)[0]
    pred_proba = model.predict_proba(X_scaled)[0]

    direction = "up" if pred_class == 1 else "down"
    confidence = float(max(pred_proba))

    # Get model accuracy info
    meta = db.get_latest_model_meta(ticker, horizon)
    model_accuracy = meta["accuracy"] if meta else None

    # Get historical prediction accuracy from our tracking
    tracked = db.get_accuracy_stats(ticker, horizon, last_n=50)

    # Save prediction for future tracking (skip if one was made recently)
    if not db.has_recent_prediction(ticker, horizon, minutes=10):
        db.save_prediction(ticker, horizon, direction, confidence)

    return {
        "ticker": ticker,
        "horizon": horizon,
        "direction": direction,
        "confidence": round(confidence * 100, 1),
        "sentiment": sentiment_score,
        "sentiment_label": sent.get("label", "N/A") if sentiment_score != 0 else "N/A",
        "model_cv_accuracy": model_accuracy,
        "tracked_accuracy": tracked["accuracy"],
        "tracked_total": tracked["total"],
        "signal": _get_signal(direction, confidence),
    }


def predict_all(horizon: str = "next_day") -> list:
    """Run predictions for all watchlist tickers."""
    from data_fetcher import get_watchlist
    results = []
    for ticker in get_watchlist():
        try:
            results.append(predict(ticker, horizon))
        except Exception as e:
            results.append({"ticker": ticker, "error": str(e)})
    return results


def _get_signal(direction: str, confidence: float) -> str:
    """Convert prediction to a trading signal label."""
    if confidence >= 0.7:
        return f"Strong {'Buy' if direction == 'up' else 'Sell'}"
    elif confidence >= 0.55:
        return f"{'Buy' if direction == 'up' else 'Sell'}"
    else:
        return "Hold"


# ── Self-improvement: resolve old predictions ────────────────────────────────

def resolve_predictions():
    """
    Check unresolved predictions against actual market data.
    This is the self-improvement loop.
    """
    unresolved = db.get_unresolved_predictions()
    resolved_count = 0

    for pred in unresolved:
        ticker = pred["ticker"]
        horizon = pred["horizon"]
        created = datetime.fromisoformat(pred["created_at"])

        # Determine when the prediction should have resolved
        from datetime import timedelta as td
        if horizon == "next_day":
            check_after = created + td(days=1)
        else:  # weekly
            check_after = created + td(days=5)

        if datetime.utcnow() < check_after:
            continue  # Not enough time has passed

        # Fetch actual price data
        try:
            df = fetch_stock_data(ticker, period="1mo")
            if df.empty:
                continue

            # Find the price at creation and after the horizon
            created_date = created.date()
            df_dates = df.index.date

            # Find closest dates
            before_prices = df[df.index.date <= created_date]
            after_prices = df[df.index.date >= check_after.date()]

            if before_prices.empty or after_prices.empty:
                continue

            price_at_pred = before_prices.iloc[-1]["Close"]
            price_at_check = after_prices.iloc[0]["Close"]

            actual_change = ((price_at_check - price_at_pred) / price_at_pred) * 100
            actual_direction = "up" if actual_change > 0 else "down"

            db.resolve_prediction(pred["id"], actual_direction, round(actual_change, 2))
            resolved_count += 1

        except Exception:
            continue

    # Retrain models if we resolved enough predictions
    if resolved_count >= 5:
        from data_fetcher import get_watchlist
        for ticker in get_watchlist():
            for h in ["next_day", "weekly"]:
                try:
                    train_model(ticker, h)
                except Exception:
                    pass

    return resolved_count
