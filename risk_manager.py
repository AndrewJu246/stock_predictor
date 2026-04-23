"""
risk_manager.py — Risk management tools.
Stop-loss suggestions, position sizing, and portfolio diversification analysis.
"""

import pandas as pd
import numpy as np
import ta
from data_fetcher import fetch_stock_data, fetch_current_price, get_ticker_sector


# ── Stop-Loss Suggestions ────────────────────────────────────────────────────

def calculate_stop_loss(ticker: str, method: str = "atr") -> dict:
    """
    Calculate suggested stop-loss levels for a ticker.

    Methods:
        atr: Based on Average True Range (volatility-adaptive)
        pct: Fixed percentage from current price
        support: Based on recent support levels
    """
    df = fetch_stock_data(ticker, period="3mo")
    if df.empty or len(df) < 20:
        return {"error": "Not enough data"}

    current = df["Close"].iloc[-1]
    results = {"ticker": ticker, "current_price": round(current, 2)}

    # ATR-based stop-loss (most common professional method)
    atr_val = ta.volatility.average_true_range(
        df["High"], df["Low"], df["Close"], window=14
    ).iloc[-1]

    if not np.isnan(atr_val):
        # Conservative: 2x ATR, Moderate: 1.5x ATR, Aggressive: 1x ATR
        results["atr"] = round(atr_val, 2)
        results["stop_conservative"] = round(current - (2.0 * atr_val), 2)
        results["stop_moderate"] = round(current - (1.5 * atr_val), 2)
        results["stop_aggressive"] = round(current - (1.0 * atr_val), 2)
        results["stop_conservative_pct"] = round((2.0 * atr_val / current) * 100, 1)
        results["stop_moderate_pct"] = round((1.5 * atr_val / current) * 100, 1)
        results["stop_aggressive_pct"] = round((1.0 * atr_val / current) * 100, 1)

    # Percentage-based
    results["stop_5pct"] = round(current * 0.95, 2)
    results["stop_8pct"] = round(current * 0.92, 2)
    results["stop_10pct"] = round(current * 0.90, 2)

    # Support level (recent 20-day low)
    recent_low = df["Low"].tail(20).min()
    results["recent_support"] = round(recent_low, 2)
    results["support_distance_pct"] = round(
        ((current - recent_low) / current) * 100, 1
    )

    # Volatility context
    daily_vol = df["Close"].pct_change().std()
    results["daily_volatility_pct"] = round(daily_vol * 100, 2)
    results["annualized_volatility_pct"] = round(daily_vol * np.sqrt(252) * 100, 1)

    return results


# ── Position Sizing ──────────────────────────────────────────────────────────

def calculate_position_size(ticker: str, portfolio_value: float,
                             risk_per_trade_pct: float = 2.0,
                             stop_loss_pct: float = None) -> dict:
    """
    Calculate recommended position size based on risk tolerance.

    Args:
        ticker: Stock ticker
        portfolio_value: Total portfolio value in $
        risk_per_trade_pct: Max % of portfolio to risk per trade (default 2%)
        stop_loss_pct: Custom stop-loss %. If None, uses ATR-based moderate stop.
    """
    if portfolio_value <= 0:
        return {"error": "Portfolio value must be positive"}

    price_data = fetch_current_price(ticker)
    if "error" in price_data:
        return price_data

    current_price = price_data["price"]

    # Get stop-loss distance
    if stop_loss_pct is None:
        sl = calculate_stop_loss(ticker)
        if "error" in sl:
            stop_loss_pct = 5.0  # Default fallback
        else:
            stop_loss_pct = sl.get("stop_moderate_pct", 5.0)

    # Max dollar risk per trade
    max_risk_dollars = portfolio_value * (risk_per_trade_pct / 100)

    # Risk per share = current price * stop_loss_pct
    risk_per_share = current_price * (stop_loss_pct / 100)

    if risk_per_share <= 0:
        return {"error": "Invalid stop-loss calculation"}

    # Recommended shares
    recommended_shares = int(max_risk_dollars / risk_per_share)
    position_value = recommended_shares * current_price
    position_pct = (position_value / portfolio_value) * 100

    return {
        "ticker": ticker,
        "current_price": current_price,
        "portfolio_value": round(portfolio_value, 2),
        "risk_per_trade_pct": risk_per_trade_pct,
        "max_risk_dollars": round(max_risk_dollars, 2),
        "stop_loss_pct": round(stop_loss_pct, 1),
        "risk_per_share": round(risk_per_share, 2),
        "recommended_shares": recommended_shares,
        "position_value": round(position_value, 2),
        "position_pct_of_portfolio": round(position_pct, 1),
    }


# ── Portfolio Diversification ────────────────────────────────────────────────

def analyze_diversification(positions: list) -> dict:
    """
    Analyze portfolio diversification by sector.

    Args:
        positions: List of dicts with 'ticker', 'shares', 'buy_price' keys
                   (from db.get_open_positions)
    """
    if not positions:
        return {"error": "No positions to analyze"}

    # Get sector for each position
    sector_exposure = {}
    position_details = []
    total_value = 0

    for p in positions:
        ticker = p["ticker"]
        cost = p["shares"] * p["buy_price"]
        total_value += cost

        sector = get_ticker_sector(ticker)
        if sector not in sector_exposure:
            sector_exposure[sector] = 0
        sector_exposure[sector] += cost

        position_details.append({
            "ticker": ticker,
            "sector": sector,
            "value": round(cost, 2),
        })

    if total_value == 0:
        return {"error": "Portfolio has no value"}

    # Calculate sector percentages
    sector_breakdown = []
    warnings = []

    for sector, value in sorted(sector_exposure.items(),
                                 key=lambda x: x[1], reverse=True):
        pct = (value / total_value) * 100
        sector_breakdown.append({
            "sector": sector,
            "value": round(value, 2),
            "pct": round(pct, 1),
        })

        # Warn if any sector > 40%
        if pct > 40:
            warnings.append(
                f"⚠️ Heavy concentration in {sector} ({pct:.0f}%). "
                f"Consider diversifying — a sector downturn could hit hard."
            )

    # Warn if too few sectors
    if len(sector_exposure) < 3 and len(positions) >= 3:
        warnings.append(
            f"⚠️ Only {len(sector_exposure)} sector(s) represented. "
            f"Spreading across more sectors reduces risk."
        )

    # Warn if single position is too large
    for p in position_details:
        pct = (p["value"] / total_value) * 100
        if pct > 30:
            warnings.append(
                f"⚠️ {p['ticker']} is {pct:.0f}% of your portfolio. "
                f"Consider trimming to reduce single-stock risk."
            )

    # Concentration score (Herfindahl index, lower = more diversified)
    hhi = sum((s["pct"] / 100) ** 2 for s in sector_breakdown)
    if hhi > 0.5:
        diversity_rating = "Poor"
    elif hhi > 0.3:
        diversity_rating = "Fair"
    elif hhi > 0.15:
        diversity_rating = "Good"
    else:
        diversity_rating = "Excellent"

    return {
        "total_value": round(total_value, 2),
        "num_positions": len(positions),
        "num_sectors": len(sector_exposure),
        "sector_breakdown": sector_breakdown,
        "position_details": position_details,
        "warnings": warnings,
        "diversity_rating": diversity_rating,
        "hhi": round(hhi, 3),
    }


# ── Risk Summary for a Single Stock ─────────────────────────────────────────

def get_risk_summary(ticker: str, portfolio_value: float = 10000) -> dict:
    """
    Get a complete risk summary for a ticker:
    stop-loss levels, position sizing, and volatility context.
    """
    stop_loss = calculate_stop_loss(ticker)
    sizing = calculate_position_size(ticker, portfolio_value)

    return {
        "ticker": ticker,
        "stop_loss": stop_loss,
        "position_sizing": sizing,
    }
