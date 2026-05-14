"""
predictor.py — Multi-model ensemble prediction engine.
Uses Gradient Boosting + Random Forest + XGBoost for robust predictions.
Self-improves by retraining with accuracy feedback.
"""

import pickle
import numpy as np
import pandas as pd
import ta
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
from pathlib import Path
from datetime import datetime, timedelta, timezone

import db
from data_fetcher import (fetch_stock_data, fetch_market_data, fetch_sector_data,
                          fetch_fear_greed_history, fetch_earnings_history,
                          fetch_insider_transactions, fetch_fred_data,
                          fetch_fundamentals)
from sentiment import get_ticker_sentiment, get_sentiment_history_df

MODELS_DIR = Path(__file__).parent / "data" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── Ensemble model definitions ───────────────────────────────────────────────

def _get_models():
    """Return dict of named models for the ensemble.
    Class balance is handled uniformly via sample_weight in train_model()."""
    models = {
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            subsample=0.8, random_state=42,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=150, max_depth=6, min_samples_leaf=5,
            random_state=42, n_jobs=-1,
        ),
    }
    if HAS_XGBOOST:
        models["XGBoost"] = XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, eval_metric="logloss",
            verbosity=0,
        )
    return models


# ── Feature engineering ──────────────────────────────────────────────────────

def build_features(df: pd.DataFrame, sentiment_df: pd.DataFrame = None) -> pd.DataFrame:
    """Add technical indicator features to OHLCV dataframe.
    sentiment_df: date-indexed DataFrame with 'sentiment', 'sentiment_ma5',
                  'sentiment_change' columns (from get_sentiment_history_df)."""
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
    market_cols = [c for c in df.columns if c.startswith((
        "sp500_", "vix_", "treasury_", "sector_", "usd_", "yield_curve_",
    ))]
    for col in market_cols:
        feat[col] = df[col]

    # Stock vs. market relative strength
    if "sp500_return" in df.columns:
        feat["vs_market"] = feat["return_1d"] - df["sp500_return"]
        feat["vs_market_5d"] = feat["return_5d"] - df["sp500_return_5d"]

    # Sector relative strength
    if "sector_return" in df.columns:
        feat["vs_sector"] = feat["return_1d"] - df["sector_return"]

    # Fear & Greed Index (if present in df)
    fg_cols = [c for c in df.columns if c.startswith(("fear_greed", "extreme_fear", "extreme_greed"))]
    for col in fg_cols:
        feat[col] = df[col]

    # Earnings proximity (if present in df)
    earnings_cols = [c for c in df.columns if c.startswith(("days_to_earnings", "earnings_"))]
    for col in earnings_cols:
        feat[col] = df[col]

    # Insider transaction signals (if present in df)
    insider_cols = [c for c in df.columns if c.startswith("insider_")]
    for col in insider_cols:
        feat[col] = df[col]

    # FRED macro data (if present in df)
    fred_cols = [c for c in df.columns if c.startswith((
        "cpi", "unemployment", "fed_funds",
    ))]
    for col in fred_cols:
        feat[col] = df[col]

    # Fundamental valuation (if present in df)
    fund_cols = [c for c in df.columns if c.startswith((
        "pe_ratio", "forward_pe", "peg_ratio", "price_to_book", "eps",
        "revenue_growth", "earnings_growth", "profit_margin", "roe",
        "debt_to_equity", "pe_vs_sector",
    ))]
    for col in fund_cols:
        feat[col] = df[col]

    # ── Sentiment time series (joined by date) ──────────────────────────
    if sentiment_df is not None and not sentiment_df.empty:
        feat = feat.join(sentiment_df, how="left")
        for col in ["sentiment", "sentiment_ma5", "sentiment_change"]:
            if col in feat.columns:
                feat[col] = feat[col].ffill().fillna(0)
    else:
        feat["sentiment"] = 0.0
        feat["sentiment_ma5"] = 0.0
        feat["sentiment_change"] = 0.0

    # Day of week (market patterns)
    feat["day_of_week"] = df.index.dayofweek

    # Month (seasonal patterns)
    feat["month"] = df.index.month

    # Market regime (if enough market data is present)
    if "sp500_close" in df.columns:
        regime_df = _compute_regime_series(df)
        for col in regime_df.columns:
            feat[col] = regime_df[col]

    return feat


def _compute_regime_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify each trading day as bull(2), sideways(1), or bear(0).
    Uses S&P 500 vs 200d SMA, VIX level, and yield curve.
    """
    regime = pd.DataFrame(index=df.index)

    sp500_sma200 = df["sp500_close"].rolling(200, min_periods=50).mean()
    above_sma200 = (df["sp500_close"] > sp500_sma200).astype(int)
    regime["sp500_above_200sma"] = above_sma200

    if "vix_close" in df.columns:
        regime["vix_high"] = (df["vix_close"] > 25).astype(int)
        regime["vix_extreme"] = (df["vix_close"] > 35).astype(int)
    else:
        regime["vix_high"] = 0
        regime["vix_extreme"] = 0

    yield_inverted = df.get("yield_curve_inverted", pd.Series(0, index=df.index))

    # Regime score: higher = more bullish
    regime["regime_score"] = (
        above_sma200 * 2
        - regime["vix_high"]
        - regime["vix_extreme"]
        - yield_inverted
    )

    # Classify: 2=bull, 1=sideways, 0=bear
    regime["regime"] = 1  # default sideways
    regime.loc[regime["regime_score"] >= 2, "regime"] = 2
    regime.loc[regime["regime_score"] <= -1, "regime"] = 0

    return regime


def detect_current_regime() -> dict:
    """Get the current market regime classification for display."""
    from data_fetcher import fetch_market_data
    market = fetch_market_data(period="1y")
    if market.empty or "sp500_close" not in market.columns:
        return {"regime": "unknown", "label": "Unknown", "details": {}}

    regime_df = _compute_regime_series(market)
    latest = regime_df.iloc[-1]

    regime_val = int(latest["regime"])
    labels = {2: "Bull", 1: "Sideways", 0: "Bear"}
    label = labels.get(regime_val, "Unknown")

    vix_val = float(market["vix_close"].iloc[-1]) if "vix_close" in market.columns else None
    sp500_val = float(market["sp500_close"].iloc[-1])
    sp500_sma200 = float(market["sp500_close"].rolling(200, min_periods=50).mean().iloc[-1])
    yield_spread = float(market["yield_curve_spread"].iloc[-1]) if "yield_curve_spread" in market.columns else None

    return {
        "regime": regime_val,
        "label": label,
        "details": {
            "sp500": round(sp500_val, 2),
            "sp500_200sma": round(sp500_sma200, 2),
            "sp500_vs_200sma_pct": round((sp500_val / sp500_sma200 - 1) * 100, 1),
            "vix": round(vix_val, 1) if vix_val else None,
            "yield_spread": round(yield_spread, 2) if yield_spread is not None else None,
        },
    }


def _merge_external(df, fetcher, **kwargs):
    """Join external data into df, handling dupes and gaps."""
    try:
        ext = fetcher(**kwargs)
        if not ext.empty:
            df = df.join(ext, how="left", rsuffix="_dup")
            df = df[[c for c in df.columns if not c.endswith("_dup")]]
            df = df.ffill()
    except Exception:
        pass
    return df


def prepare_dataset(ticker: str, horizon: str = "next_day"):
    """
    Build feature matrix X and target y.
    horizon: 'next_day' (1-day forward return) or 'weekly' (5-day forward return)
    """
    df = fetch_stock_data(ticker, period="2y")
    if df.empty or len(df) < 60:
        return None, None, None

    df = _merge_external(df, fetch_market_data, period="2y")
    df = _merge_external(df, fetch_sector_data, ticker=ticker, period="2y")
    df = _merge_external(df, fetch_fear_greed_history, period="2y")
    df = _merge_external(df, fetch_earnings_history, ticker=ticker, period="2y")
    df = _merge_external(df, fetch_insider_transactions, ticker=ticker, period="2y")
    df = _merge_external(df, fetch_fred_data, period="2y")
    df = _merge_external(df, fetch_fundamentals, ticker=ticker, period="2y")

    sentiment_df = get_sentiment_history_df(ticker)
    features = build_features(df, sentiment_df)

    # Target: direction of future price movement
    if horizon == "next_day":
        target = (df["Close"].shift(-1) > df["Close"]).astype(int)
        pct_target = df["Close"].pct_change(1).shift(-1)
        noise_threshold = 0.003  # 0.3% — below this is noise for daily
    else:  # weekly
        target = (df["Close"].shift(-5) > df["Close"]).astype(int)
        pct_target = df["Close"].pct_change(5).shift(-5)
        noise_threshold = 0.005  # 0.5% — below this is noise for weekly

    # Combine and drop rows with any NaN (indicator warm-up + future target)
    combined = features.copy()
    combined["target"] = target
    combined["pct_target"] = pct_target
    combined = combined.dropna()

    # Filter out "flat" days where the move is within noise range
    # These are coin-flip labels that degrade training signal
    meaningful = combined[combined["pct_target"].abs() >= noise_threshold]
    if len(meaningful) < 30:
        meaningful = combined  # fall back to full set if too few remain

    X = meaningful.drop(columns=["target", "pct_target"])
    y = meaningful["target"]

    return X, y, meaningful


MAX_FEATURES = 30


def select_features(X: pd.DataFrame, y: pd.Series, max_features: int = MAX_FEATURES) -> tuple:
    """
    Two-stage feature selection:
    1. Drop near-zero variance features
    2. Rank by mutual information, keep top max_features
    Returns (X_selected, selected_columns, selection_info).
    """
    original_count = X.shape[1]

    # Stage 1: variance filter — drop constant or near-constant columns
    vt = VarianceThreshold(threshold=1e-6)
    try:
        vt.fit(X.fillna(0))
        kept_mask = vt.get_support()
        X_var = X.loc[:, kept_mask]
    except Exception:
        X_var = X

    dropped_variance = original_count - X_var.shape[1]

    # Stage 2: mutual information ranking
    if X_var.shape[1] <= max_features:
        return X_var, list(X_var.columns), {
            "original": original_count,
            "after_variance_filter": X_var.shape[1],
            "dropped_low_variance": dropped_variance,
            "final": X_var.shape[1],
            "method": "variance_filter_only",
        }

    try:
        mi_scores = mutual_info_classif(X_var.fillna(0), y, random_state=42, n_neighbors=5)
        mi_series = pd.Series(mi_scores, index=X_var.columns).sort_values(ascending=False)
        top_features = mi_series.head(max_features).index.tolist()
        X_selected = X_var[top_features]
    except Exception:
        X_selected = X_var.iloc[:, :max_features]
        top_features = list(X_selected.columns)

    return X_selected, top_features, {
        "original": original_count,
        "after_variance_filter": X_var.shape[1],
        "dropped_low_variance": dropped_variance,
        "final": len(top_features),
        "method": "variance_filter + mutual_information",
    }


# ── Model training ───────────────────────────────────────────────────────────

def train_model(ticker: str, horizon: str = "next_day") -> dict:
    """Train a 3-model ensemble for the given ticker and horizon."""
    X_raw, y, combined = prepare_dataset(ticker, horizon)

    if X_raw is None:
        return {"error": f"Not enough data for {ticker}"}

    # Magnitude + class-balance weighted training
    mag_weights = combined.loc[X_raw.index, "pct_target"].abs()
    mag_weights = mag_weights.clip(upper=mag_weights.quantile(0.95))
    mag_weights = (mag_weights / mag_weights.mean()).values

    # Class balance: compensate for bull-market bias in training data
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    class_balance = np.where(y.values == 1,
                             len(y) / (2 * max(n_pos, 1)),
                             len(y) / (2 * max(n_neg, 1)))
    mag_weights = mag_weights * class_balance
    mag_weights = mag_weights / mag_weights.mean()

    # Feature selection on full training set for the final model
    X, selected_features, selection_info = select_features(X_raw, y)

    # Final scaler and fit on full selected data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    models = _get_models()
    trained = {}
    cv_results = {}
    tscv = TimeSeriesSplit(n_splits=5)

    # Per-fold CV with feature selection inside each fold (no leakage)
    for name, model in models.items():
        fold_scores = []
        try:
            for train_idx, test_idx in tscv.split(X_raw):
                X_train_fold = X_raw.iloc[train_idx]
                y_train_fold = y.iloc[train_idx]
                X_test_fold = X_raw.iloc[test_idx]
                y_test_fold = y.iloc[test_idx]

                X_train_sel, fold_features, _ = select_features(X_train_fold, y_train_fold)
                available_test = [f for f in fold_features if f in X_test_fold.columns]
                X_test_sel = X_test_fold[available_test]
                for f in fold_features:
                    if f not in X_test_sel.columns:
                        X_test_sel[f] = 0
                X_test_sel = X_test_sel[fold_features]

                fold_scaler = StandardScaler()
                X_train_sc = fold_scaler.fit_transform(X_train_sel)
                X_test_sc = fold_scaler.transform(X_test_sel)

                fold_model = clone(model)
                fold_weights = mag_weights[train_idx]
                fold_model.fit(X_train_sc, y_train_fold, sample_weight=fold_weights)
                fold_scores.append(fold_model.score(X_test_sc, y_test_fold))

            # Fit final model on full selected+scaled data
            model.fit(X_scaled, y, sample_weight=mag_weights)

            # Calibrate probabilities via Platt scaling on held-out tail
            cal_n = max(20, len(X_scaled) // 5)
            calibrated = CalibratedClassifierCV(model, cv="prefit", method="sigmoid")
            calibrated.fit(X_scaled[-cal_n:], y.iloc[-cal_n:],
                           sample_weight=mag_weights[-cal_n:])
            trained[name] = calibrated
            cv_results[name] = {
                "accuracy": round(np.mean(fold_scores) * 100, 1),
                "std": round(np.std(fold_scores) * 100, 1),
            }
        except Exception as e:
            cv_results[name] = {"accuracy": 0, "std": 0, "error": str(e)}
            print(f"[WARN] {name} failed for {ticker}/{horizon}: {e}")

    if not trained:
        errors = {k: v.get("error", "unknown") for k, v in cv_results.items() if "error" in v}
        return {"error": f"All models failed to train: {errors}"}

    # Save ensemble bundle
    model_path = MODELS_DIR / f"{ticker}_{horizon}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "models": trained,
            "scaler": scaler,
            "features": list(X.columns),
            "cv_results": cv_results,
        }, f)

    # Overall ensemble accuracy = weighted average of individual CV scores
    valid_accs = [v["accuracy"] for v in cv_results.values() if v["accuracy"] > 0]
    ensemble_accuracy = round(sum(valid_accs) / len(valid_accs), 1) if valid_accs else 0

    db.save_model_meta(ticker, horizon, len(X), ensemble_accuracy, list(X.columns))

    return {
        "ticker": ticker,
        "horizon": horizon,
        "ensemble_accuracy": ensemble_accuracy,
        "model_results": cv_results,
        "sample_size": len(X),
        "num_models": len(trained),
        "features": list(X.columns),
        "feature_selection": selection_info,
    }


def load_model(ticker: str, horizon: str = "next_day"):
    """Load a trained ensemble from disk."""
    model_path = MODELS_DIR / f"{ticker}_{horizon}.pkl"
    if not model_path.exists():
        return None
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    # Handle legacy single-model bundles
    if "model" in bundle and "models" not in bundle:
        bundle["models"] = {"GradientBoosting": bundle.pop("model")}
        bundle["cv_results"] = {"GradientBoosting": {"accuracy": 0, "std": 0}}
    return bundle


# ── Feature importance ───────────────────────────────────────────────────────

def get_feature_importance(ticker: str, horizon: str = "next_day", top_n: int = 20) -> dict:
    """
    Extract and aggregate feature importance across all ensemble models.
    Returns ranked features with importance scores.
    """
    bundle = load_model(ticker, horizon)
    if bundle is None:
        return {"error": "No trained model found. Run a prediction first."}

    models = bundle["models"]
    feature_names = bundle["features"]

    # Collect importances from each model
    all_importances = {}
    model_contributions = {}

    for name, model in models.items():
        try:
            # Unwrap CalibratedClassifierCV to access base model
            base = model
            if hasattr(model, "calibrated_classifiers_"):
                base = model.calibrated_classifiers_[0].estimator

            if hasattr(base, "feature_importances_"):
                importances = base.feature_importances_
                model_contributions[name] = dict(zip(feature_names, importances))

                for feat, imp in zip(feature_names, importances):
                    if feat not in all_importances:
                        all_importances[feat] = []
                    all_importances[feat].append(imp)
        except Exception:
            continue

    if not all_importances:
        return {"error": "Could not extract feature importances"}

    # Average importance across models
    avg_importance = {
        feat: round(sum(scores) / len(scores), 6)
        for feat, scores in all_importances.items()
    }

    # Sort by importance
    ranked = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)

    # Categorize features
    categories = {
        "Technical": ["return_", "sma_", "ema_", "rsi", "macd", "bb_", "stoch_",
                      "atr", "obv", "momentum_", "volume_", "volatility_",
                      "price_vs_"],
        "Market": ["sp500_", "vix_", "treasury_", "vs_market", "usd_",
                   "yield_curve_"],
        "Sector": ["sector_", "vs_sector"],
        "Sentiment": ["sentiment", "sentiment_ma5", "sentiment_change"],
        "Fear & Greed": ["fear_greed", "extreme_fear", "extreme_greed"],
        "Earnings": ["days_to_earnings", "earnings_near", "earnings_week"],
        "Insider": ["insider_buys", "insider_sells", "insider_net", "insider_buy_30d",
                     "insider_sell_30d", "insider_net_30d", "insider_signal"],
        "Macro (FRED)": ["cpi", "cpi_yoy_change", "cpi_mom_change", "unemployment",
                          "unemployment_change", "fed_funds", "fed_funds_change"],
        "Regime": ["regime", "regime_score", "sp500_above_200sma", "vix_high", "vix_extreme"],
        "Fundamentals": ["pe_ratio", "forward_pe", "peg_ratio", "price_to_book", "eps",
                          "revenue_growth", "earnings_growth", "profit_margin", "roe",
                          "debt_to_equity", "pe_vs_sector"],
        "Seasonality": ["day_of_week", "month"],
    }

    category_totals = {}
    for feat, imp in avg_importance.items():
        categorized = False
        for cat, prefixes in categories.items():
            if any(feat.startswith(p) or feat == p for p in prefixes):
                category_totals[cat] = category_totals.get(cat, 0) + imp
                categorized = True
                break
        if not categorized:
            category_totals["Other"] = category_totals.get("Other", 0) + imp

    # Normalize category totals to percentages
    total_imp = sum(category_totals.values())
    category_pct = {
        cat: round((val / total_imp) * 100, 1) if total_imp > 0 else 0
        for cat, val in sorted(category_totals.items(), key=lambda x: x[1], reverse=True)
    }

    return {
        "ticker": ticker,
        "horizon": horizon,
        "top_features": ranked[:top_n],
        "all_features": ranked,
        "category_breakdown": category_pct,
        "num_features": len(feature_names),
        "num_models": len(model_contributions),
    }


# ── Prediction ───────────────────────────────────────────────────────────────

def predict(ticker: str, horizon: str = "next_day") -> dict:
    """
    Make an ensemble prediction for the given ticker.
    Each model votes, and the final prediction is the weighted consensus.
    Auto-trains if no model exists.
    """
    bundle = load_model(ticker, horizon)

    # Train if needed
    if bundle is None:
        result = train_model(ticker, horizon)
        if "error" in result:
            return result
        bundle = load_model(ticker, horizon)

    models = bundle["models"]
    scaler = bundle["scaler"]
    feature_names = bundle["features"]
    cv_results = bundle.get("cv_results", {})

    # Build current features
    df = fetch_stock_data(ticker, period="6mo")
    if df.empty:
        return {"error": "No stock data available"}

    df = _merge_external(df, fetch_market_data, period="6mo")
    df = _merge_external(df, fetch_sector_data, ticker=ticker, period="6mo")
    df = _merge_external(df, fetch_fear_greed_history, period="6mo")
    df = _merge_external(df, fetch_earnings_history, ticker=ticker, period="6mo")
    df = _merge_external(df, fetch_insider_transactions, ticker=ticker, period="6mo")
    df = _merge_external(df, fetch_fred_data, period="6mo")
    df = _merge_external(df, fetch_fundamentals, ticker=ticker, period="6mo")

    # Fetch fresh sentiment (saves to daily_sentiment table) and get history
    try:
        sent = get_ticker_sentiment(ticker)
    except Exception:
        sent = {}
    sentiment_df = get_sentiment_history_df(ticker)

    features = build_features(df, sentiment_df)
    features = features.fillna(0)

    if features.empty:
        return {"error": "Could not compute features"}

    # Use the latest row — only use features the model was trained on
    available = [f for f in feature_names if f in features.columns]
    latest = features.iloc[[-1]][available]

    # Pad missing columns with 0
    for f in feature_names:
        if f not in latest.columns:
            latest[f] = 0
    latest = latest[feature_names]

    X_scaled = scaler.transform(latest)

    # ── Ensemble voting ──────────────────────────────────────────────────
    votes = []
    model_details = []

    for name, model in models.items():
        try:
            pred_class = model.predict(X_scaled)[0]
            pred_proba = model.predict_proba(X_scaled)[0]
            direction = "up" if pred_class == 1 else "down"
            confidence = float(max(pred_proba))

            # Weight by CV accuracy (better models get more say)
            cv_acc = cv_results.get(name, {}).get("accuracy", 50) / 100
            weight = max(cv_acc, 0.5)  # Floor at 0.5

            votes.append({
                "name": name,
                "direction": direction,
                "confidence": confidence,
                "weight": weight,
                "cv_accuracy": cv_results.get(name, {}).get("accuracy", 0),
            })
            model_details.append(f"{name}: {direction} ({confidence*100:.0f}%)")
        except Exception:
            continue

    if not votes:
        return {"error": "All models failed to predict"}

    # Weighted ensemble: sum weighted probabilities for "up"
    total_weight = sum(v["weight"] for v in votes)
    up_score = sum(
        v["weight"] * (v["confidence"] if v["direction"] == "up" else 1 - v["confidence"])
        for v in votes
    ) / total_weight

    direction = "up" if up_score > 0.5 else "down"
    confidence = up_score if direction == "up" else 1 - up_score

    # Consensus info
    up_votes = sum(1 for v in votes if v["direction"] == "up")
    down_votes = len(votes) - up_votes
    consensus = f"{max(up_votes, down_votes)}/{len(votes)} agree"

    # Get model accuracy info
    meta = db.get_latest_model_meta(ticker, horizon)
    model_accuracy = meta["accuracy"] if meta else None

    # Get historical prediction accuracy from our tracking
    tracked = db.get_accuracy_stats(ticker, horizon, last_n=50)

    # Save prediction for future tracking (skip if one was made recently)
    if not db.has_recent_prediction(ticker, horizon, minutes=10):
        db.save_prediction(ticker, horizon, direction, confidence)

    regime = detect_current_regime()

    return {
        "ticker": ticker,
        "horizon": horizon,
        "direction": direction,
        "confidence": round(confidence * 100, 1),
        "consensus": consensus,
        "model_votes": votes,
        "model_details": model_details,
        "sentiment": sent.get("avg_sentiment", 0),
        "sentiment_label": sent.get("label", "N/A"),
        "model_cv_accuracy": model_accuracy,
        "tracked_accuracy": tracked["accuracy"],
        "tracked_total": tracked["total"],
        "signal": _get_signal(direction, confidence, ticker, regime.get("regime")),
        "market_regime": regime,
    }


def _prewarm_shared_cache(period: str = "6mo"):
    """Fetch ticker-independent data once so per-ticker predict() calls hit cache."""
    for fetcher in [fetch_market_data, fetch_fear_greed_history, fetch_fred_data]:
        try:
            fetcher(period=period)
        except Exception:
            pass


def predict_all(horizon: str = "next_day") -> list:
    """Run predictions for all watchlist tickers."""
    from data_fetcher import get_watchlist
    _prewarm_shared_cache("6mo")
    results = []
    for ticker in get_watchlist():
        try:
            results.append(predict(ticker, horizon))
        except Exception as e:
            results.append({"ticker": ticker, "error": str(e)})
    return results


def calibrate_confidence(ticker: str = None, min_samples: int = 20) -> dict:
    """
    Analyze resolved predictions to measure actual accuracy per confidence bin.
    Returns calibration data and recommended signal thresholds.
    """
    stats = db.get_accuracy_stats(ticker=ticker)
    preds = stats.get("predictions", [])

    if len(preds) < min_samples:
        return {
            "calibrated": False,
            "reason": f"Need {min_samples}+ resolved predictions, have {len(preds)}",
            "thresholds": {"strong": 0.7, "normal": 0.55},
        }

    bins = [(50, 55), (55, 60), (60, 65), (65, 70), (70, 75), (75, 80), (80, 100)]
    calibration = []

    for low, high in bins:
        in_bin = [p for p in preds
                  if low <= p["pred_confidence"] * 100 < high]
        if not in_bin:
            continue
        correct = sum(1 for p in in_bin if p["correct"] == 1)
        acc = correct / len(in_bin)
        calibration.append({
            "bin": f"{low}-{high}%",
            "count": len(in_bin),
            "accuracy": round(acc * 100, 1),
            "reported_confidence": round((low + high) / 2, 1),
        })

    strong_threshold = 0.7
    normal_threshold = 0.55

    for entry in calibration:
        if entry["accuracy"] >= 65 and entry["count"] >= 5:
            candidate = entry["bin"].split("-")[0]
            normal_threshold = min(normal_threshold, int(candidate) / 100)
        if entry["accuracy"] >= 75 and entry["count"] >= 5:
            candidate = entry["bin"].split("-")[0]
            strong_threshold = min(strong_threshold, int(candidate) / 100)

    return {
        "calibrated": True,
        "bins": calibration,
        "total_predictions": len(preds),
        "overall_accuracy": stats["accuracy"],
        "thresholds": {
            "strong": round(strong_threshold, 2),
            "normal": round(normal_threshold, 2),
        },
    }


_calibration_cache = {}
_CALIBRATION_TTL = 3600


def _get_signal(direction: str, confidence: float, ticker: str = None,
                regime: int = None) -> str:
    """Convert prediction to a trading signal label using calibrated thresholds.
    Regime-aware: bear markets widen the hold zone, bull markets narrow it."""
    strong = 0.7
    normal = 0.55

    if ticker:
        now = datetime.now()
        if (ticker in _calibration_cache and
                (now - _calibration_cache[ticker][1]).total_seconds() < _CALIBRATION_TTL):
            cal = _calibration_cache[ticker][0]
        else:
            cal = calibrate_confidence(ticker)
            _calibration_cache[ticker] = (cal, now)

        if cal.get("calibrated"):
            strong = cal["thresholds"]["strong"]
            normal = cal["thresholds"]["normal"]

    # Regime adjustment: bear markets require higher confidence for signals
    if regime == 0:  # Bear
        strong = min(strong + 0.05, 0.85)
        normal = min(normal + 0.05, 0.75)
    elif regime == 2:  # Bull
        strong = max(strong - 0.03, 0.60)
        normal = max(normal - 0.03, 0.50)

    if confidence >= strong:
        return f"Strong {'Buy' if direction == 'up' else 'Sell'}"
    elif confidence >= normal:
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
        if horizon == "next_day":
            check_after = created + timedelta(days=1)
        else:  # weekly
            check_after = created + timedelta(days=5)

        if datetime.now(timezone.utc).replace(tzinfo=None) < check_after:
            continue  # Not enough time has passed

        # Fetch actual price data
        try:
            df = fetch_stock_data(ticker, period="1mo")
            if df.empty:
                continue

            # Find the price at creation and after the horizon
            created_date = created.date()

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
