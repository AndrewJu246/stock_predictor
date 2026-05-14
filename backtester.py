"""
backtester.py — Walk-forward backtesting engine.
Simulates how the model would have performed historically.
"""

import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from datetime import datetime

from data_fetcher import (fetch_stock_data, fetch_market_data, fetch_sector_data,
                          fetch_fear_greed_history, fetch_earnings_history,
                          fetch_insider_transactions, fetch_fred_data,
                          fetch_fundamentals)
from predictor import build_features, _merge_external, _prewarm_shared_cache, _get_models
from sentiment import get_sentiment_history_df


def run_backtest(ticker: str, horizon: str = "next_day",
                 train_window: int = 90, test_step: int = 3,
                 progress_callback=None) -> dict:
    """
    Walk-forward backtest with ensemble voting.
    - Train on `train_window` days, predict the next day/week, slide forward.
    - Returns accuracy stats + equity curve.
    
    Args:
        ticker: Stock ticker
        horizon: 'next_day' or 'weekly'
        train_window: Days of data to train on before each prediction
        test_step: How many days to step forward (1 for daily)
    """
    # Fetch extended history for backtesting
    df = fetch_stock_data(ticker, period="2y")
    if df.empty or len(df) < train_window + 30:
        return {"error": f"Not enough historical data for {ticker} (need {train_window + 30}+ days, got {len(df)})"}

    df = _merge_external(df, fetch_market_data, period="2y")
    df = _merge_external(df, fetch_sector_data, ticker=ticker, period="2y")
    df = _merge_external(df, fetch_fear_greed_history, period="2y")
    df = _merge_external(df, fetch_earnings_history, ticker=ticker, period="2y")
    df = _merge_external(df, fetch_insider_transactions, ticker=ticker, period="2y")
    df = _merge_external(df, fetch_fred_data, period="2y")
    df = _merge_external(df, fetch_fundamentals, ticker=ticker, period="2y")

    # Use historical sentiment if available
    sentiment_df = get_sentiment_history_df(ticker)
    features = build_features(df, sentiment_df)

    # Build target
    if horizon == "next_day":
        shift = -1
        noise_threshold = 0.3  # pct_change is already in %, so 0.3 = 0.3%
    else:
        shift = -5
        noise_threshold = 0.5

    target = (df["Close"].shift(shift) > df["Close"]).astype(int)
    pct_change = ((df["Close"].shift(shift) - df["Close"]) / df["Close"] * 100)

    # Combine and drop all NaN rows (indicator warm-up + future target)
    combined = features.copy()
    combined["target"] = target
    combined["pct_change"] = pct_change
    combined["close"] = df["Close"]
    combined = combined.dropna()

    # Filter out flat/noise days from training data
    combined = combined[combined["pct_change"].abs() >= noise_threshold]

    # Magnitude-weighted training: big moves matter more
    mag_weights = combined["pct_change"].abs()
    mag_weights = mag_weights.clip(upper=mag_weights.quantile(0.95))
    mag_weights = (mag_weights / mag_weights.mean()).values

    if len(combined) < train_window + 10:
        return {"error": "Not enough clean data after feature engineering"}

    # Walk-forward simulation
    results = []
    feature_cols = [c for c in combined.columns if c not in ["target", "pct_change", "close"]]

    start_idx = train_window
    end_idx = len(combined) - abs(shift)
    total_steps = len(range(start_idx, end_idx, test_step))
    step_count = 0

    # Use same models as production for honest backtest results
    bt_models = _get_models()

    for i in range(start_idx, end_idx, test_step):
        step_count += 1
        if progress_callback:
            progress_callback(step_count / total_steps)

        train_data = combined.iloc[:i]
        test_row = combined.iloc[i:i+1]

        if len(train_data) < 30 or test_row.empty:
            continue

        X_train = train_data[feature_cols]
        y_train = train_data["target"]
        X_test = test_row[feature_cols]

        try:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train all ensemble models and vote
            votes_up = 0
            votes_total = 0
            weighted_up_score = 0
            total_weight = 0

            for name, mdl in bt_models.items():
                try:
                    mdl = clone(mdl)
                    train_weights = mag_weights[:len(train_data)]
                    mdl.fit(X_train_scaled, y_train, sample_weight=train_weights)
                    p_class = mdl.predict(X_test_scaled)[0]
                    p_proba = mdl.predict_proba(X_test_scaled)[0]
                    conf = float(max(p_proba))
                    direction_up = p_class == 1

                    up_prob = conf if direction_up else 1 - conf
                    weighted_up_score += up_prob
                    total_weight += 1
                    votes_total += 1
                    if direction_up:
                        votes_up += 1
                except Exception:
                    continue

            if votes_total == 0:
                continue

            ensemble_up_score = weighted_up_score / total_weight
            pred = 1 if ensemble_up_score > 0.5 else 0
            pred_proba = ensemble_up_score if pred == 1 else 1 - ensemble_up_score

            actual = int(test_row["target"].iloc[0])
            actual_pct = float(test_row["pct_change"].iloc[0])
            close_price = float(test_row["close"].iloc[0])

            results.append({
                "date": test_row.index[0].strftime("%Y-%m-%d"),
                "predicted": "up" if pred == 1 else "down",
                "actual": "up" if actual == 1 else "down",
                "confidence": round(pred_proba * 100, 1),
                "actual_pct": round(actual_pct, 2),
                "close": round(close_price, 2),
                "correct": pred == actual,
                "consensus": f"{max(votes_up, votes_total - votes_up)}/{votes_total}",
            })
        except Exception:
            continue

    if not results:
        return {"error": "Backtest produced no results"}

    # Calculate stats
    df_results = pd.DataFrame(results)
    total = len(df_results)
    correct = df_results["correct"].sum()
    accuracy = round(correct / total * 100, 1) if total else 0

    # Simulate confidence-proportional strategy returns:
    # Position size scales with conviction — higher confidence = larger allocation
    strategy_returns = []
    buy_hold_returns = []
    for _, row in df_results.iterrows():
        daily_ret = row["actual_pct"] / 100
        conf = row["confidence"] / 100
        if row["predicted"] == "up":
            position_size = min(1.0, max(0.0, (conf - 0.5) * 3))
            strategy_returns.append(daily_ret * position_size)
        else:
            strategy_returns.append(0)
        buy_hold_returns.append(daily_ret)

    # Cumulative returns
    strategy_cumulative = [(1 + r) for r in strategy_returns]
    buyhold_cumulative = [(1 + r) for r in buy_hold_returns]

    strategy_total = 1
    buyhold_total = 1
    strategy_curve = []
    buyhold_curve = []

    for s, b in zip(strategy_cumulative, buyhold_cumulative):
        strategy_total *= s
        buyhold_total *= b
        strategy_curve.append(strategy_total)
        buyhold_curve.append(buyhold_total)

    df_results["strategy_equity"] = strategy_curve
    df_results["buyhold_equity"] = buyhold_curve

    # Rolling accuracy (last 20 predictions)
    df_results["rolling_accuracy"] = df_results["correct"].rolling(
        window=min(20, max(5, total // 5)), min_periods=5
    ).mean() * 100

    # High-confidence accuracy
    high_conf = df_results[df_results["confidence"] >= 60]
    high_conf_acc = round(high_conf["correct"].mean() * 100, 1) if len(high_conf) > 0 else 0

    # ── Alpha metrics ───────────────────────────────────────────────────
    strategy_ret_pct = round((strategy_total - 1) * 100, 2)
    buyhold_ret_pct = round((buyhold_total - 1) * 100, 2)
    alpha = round(strategy_ret_pct - buyhold_ret_pct, 2)

    # Sharpe ratio (annualized)
    ret_array = np.array(strategy_returns)
    if len(ret_array) > 1 and ret_array.std() > 0:
        sharpe = round(np.sqrt(252) * ret_array.mean() / ret_array.std(), 2)
    else:
        sharpe = 0.0

    # Max drawdown
    running_max = 1.0
    max_dd = 0.0
    for eq in strategy_curve:
        running_max = max(running_max, eq)
        dd = (running_max - eq) / running_max
        max_dd = max(max_dd, dd)
    max_drawdown = round(max_dd * 100, 2)

    # Win rate and expectancy
    winning = df_results[df_results["correct"] == True]
    losing = df_results[df_results["correct"] == False]
    avg_win = abs(winning["actual_pct"].mean()) if len(winning) > 0 else 0
    avg_loss = abs(losing["actual_pct"].mean()) if len(losing) > 0 else 0
    win_rate = round(len(winning) / total * 100, 1) if total else 0
    expectancy = round(
        (win_rate / 100) * avg_win - (1 - win_rate / 100) * avg_loss, 3
    ) if total else 0

    return {
        "ticker": ticker,
        "horizon": horizon,
        "total_predictions": total,
        "correct": int(correct),
        "accuracy": accuracy,
        "high_confidence_accuracy": high_conf_acc,
        "high_confidence_count": len(high_conf),
        "strategy_return": strategy_ret_pct,
        "buyhold_return": buyhold_ret_pct,
        "alpha": alpha,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "expectancy": expectancy,
        "results": df_results.to_dict("records"),
        "date_range": f"{results[0]['date']} to {results[-1]['date']}",
    }


def run_backtest_all(horizon: str = "next_day") -> list:
    """Run backtest for all watchlist tickers."""
    from data_fetcher import get_watchlist
    _prewarm_shared_cache("2y")
    results = []
    for ticker in get_watchlist():
        try:
            results.append(run_backtest(ticker, horizon))
        except Exception as e:
            results.append({"ticker": ticker, "error": str(e)})
    return results
