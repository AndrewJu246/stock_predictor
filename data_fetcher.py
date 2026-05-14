"""
data_fetcher.py — Pulls stock price data via yfinance and news via RSS feeds.
"""

import yfinance as yf
import feedparser
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "config.json"

# Simple time-based cache for expensive API calls
_cache = {}
_CACHE_TTL = 300  # 5 minutes


def _get_cached(key):
    """Return cached value if it exists and is fresh, else None."""
    if key in _cache:
        val, ts = _cache[key]
        if (datetime.now() - ts).total_seconds() < _CACHE_TTL:
            return val
    return None


def _set_cached(key, val):
    _cache[key] = (val, datetime.now())


def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


def save_config(cfg):
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)


# ── Watchlist management ─────────────────────────────────────────────────────

def get_watchlist():
    return load_config()["watchlist"]


def add_ticker(ticker: str):
    cfg = load_config()
    t = ticker.upper().strip()
    if t not in cfg["watchlist"]:
        # Validate ticker exists by fetching recent history
        hist = yf.Ticker(t).history(period="5d")
        if hist.empty:
            raise ValueError(f"Ticker '{t}' not found or has no data")
        cfg["watchlist"].append(t)
        save_config(cfg)
    return cfg["watchlist"]


def remove_ticker(ticker: str):
    cfg = load_config()
    t = ticker.upper().strip()
    cfg["watchlist"] = [x for x in cfg["watchlist"] if x != t]
    save_config(cfg)
    return cfg["watchlist"]


# ── Stock data ───────────────────────────────────────────────────────────────

def fetch_stock_data(ticker: str, period: str = "6mo") -> pd.DataFrame:
    """Fetch OHLCV data for a ticker (cached 5 min)."""
    cache_key = f"stock_{ticker}_{period}"
    cached = _get_cached(cache_key)
    if cached is not None:
        return cached

    tk = yf.Ticker(ticker)
    df = tk.history(period=period, auto_adjust=True)

    if df.empty:
        return df

    # Normalize timezone — strip tz so joins work cleanly
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Basic derived columns
    df["Return"] = df["Close"].pct_change()
    df["Return_5d"] = df["Close"].pct_change(5)
    df["Log_Volume"] = df["Volume"].apply(lambda v: np.log1p(v) if v > 0 else 0)

    _set_cached(cache_key, df)
    return df


def fetch_current_price(ticker: str) -> dict:
    """Get the current / latest price info for a ticker (cached 5 min)."""
    cache_key = f"price_{ticker}"
    cached = _get_cached(cache_key)
    if cached is not None:
        return cached

    tk = yf.Ticker(ticker)
    info = tk.info
    hist = tk.history(period="5d")

    if hist.empty:
        return {"ticker": ticker, "error": "No data"}

    latest = hist.iloc[-1]
    prev = hist.iloc[-2] if len(hist) > 1 else latest

    change = latest["Close"] - prev["Close"]
    change_pct = (change / prev["Close"]) * 100 if prev["Close"] else 0

    result = {
        "ticker": ticker,
        "price": round(latest["Close"], 2),
        "change": round(change, 2),
        "change_pct": round(change_pct, 2),
        "volume": int(latest["Volume"]),
        "high": round(latest["High"], 2),
        "low": round(latest["Low"], 2),
        "name": info.get("shortName", ticker),
        "market_cap": info.get("marketCap"),
        "updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    _set_cached(cache_key, result)
    return result


def fetch_batch_prices(tickers: list) -> list:
    """Fetch current prices for multiple tickers."""
    results = []
    for t in tickers:
        try:
            results.append(fetch_current_price(t))
        except Exception as e:
            results.append({"ticker": t, "error": str(e)})
    return results


# ── News fetching ────────────────────────────────────────────────────────────

def fetch_news(ticker: str, max_articles: int = 20) -> list:
    """Fetch news articles for a ticker from RSS feeds."""
    cfg = load_config()
    sources = cfg["settings"]["news_sources"]
    articles = []

    for url_template in sources:
        url = url_template.replace("{ticker}", ticker)
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:max_articles]:
                articles.append({
                    "title": entry.get("title", ""),
                    "source": feed.feed.get("title", "Unknown"),
                    "url": entry.get("link", ""),
                    "published": entry.get("published", ""),
                })
        except Exception:
            continue

    # Deduplicate by URL
    seen = set()
    unique = []
    for a in articles:
        if a["url"] not in seen:
            seen.add(a["url"])
            unique.append(a)

    return unique[:max_articles]


# ── Market-wide data sources ────────────────────────────────────────────────

# Sector ETF mapping
SECTOR_ETFS = {
    "Technology": "XLK",
    "Healthcare": "XLV",
    "Financial Services": "XLF",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
    "Communication Services": "XLC",
}


def fetch_market_data(period: str = "6mo") -> pd.DataFrame:
    """
    Fetch market-wide indicators: S&P 500, VIX, Treasury yields.
    Cached for 5 minutes to avoid redundant API calls.
    """
    cache_key = f"market_{period}"
    cached = _get_cached(cache_key)
    if cached is not None:
        return cached

    market = pd.DataFrame()

    indicators = {
        "^GSPC": "sp500",       # S&P 500
        "^VIX": "vix",          # Volatility Index
        "^TNX": "treasury_10y", # 10-Year Treasury Yield
        "^IRX": "treasury_3m",  # 3-Month Treasury Bill (short-term rates)
        "^FVX": "treasury_5y",  # 5-Year Treasury Yield (mid curve)
        "DX-Y.NYB": "usd",     # US Dollar Index
    }

    for symbol, name in indicators.items():
        try:
            tk = yf.Ticker(symbol)
            hist = tk.history(period=period, auto_adjust=True)
            if not hist.empty:
                if hist.index.tz is not None:
                    hist.index = hist.index.tz_localize(None)
                market[f"{name}_close"] = hist["Close"]
                market[f"{name}_return"] = hist["Close"].pct_change()
                market[f"{name}_return_5d"] = hist["Close"].pct_change(5)
        except Exception:
            continue

    # Derived macro features
    if "treasury_10y_close" in market.columns and "treasury_3m_close" in market.columns:
        market["yield_curve_spread"] = market["treasury_10y_close"] - market["treasury_3m_close"]
        market["yield_curve_inverted"] = (market["yield_curve_spread"] < 0).astype(int)

    _set_cached(cache_key, market)
    return market


def fetch_sector_data(ticker: str, period: str = "6mo") -> pd.DataFrame:
    """Fetch the sector ETF performance for a given ticker's sector (cached 5 min)."""
    cache_key = f"sector_data_{ticker}_{period}"
    cached = _get_cached(cache_key)
    if cached is not None:
        return cached

    try:
        sector = get_ticker_sector(ticker)
        etf_ticker = SECTOR_ETFS.get(sector)

        if not etf_ticker:
            return pd.DataFrame()

        tk = yf.Ticker(etf_ticker)
        hist = tk.history(period=period, auto_adjust=True)
        if hist.empty:
            return pd.DataFrame()

        if hist.index.tz is not None:
            hist.index = hist.index.tz_localize(None)

        result = pd.DataFrame(index=hist.index)
        result["sector_close"] = hist["Close"]
        result["sector_return"] = hist["Close"].pct_change()
        result["sector_return_5d"] = hist["Close"].pct_change(5)
        result["sector_momentum"] = hist["Close"].pct_change(20)

        _set_cached(cache_key, result)
        return result
    except Exception:
        return pd.DataFrame()


def get_ticker_sector(ticker: str) -> str:
    """Get the sector for a ticker (cached 5 min)."""
    cache_key = f"sector_{ticker}"
    cached = _get_cached(cache_key)
    if cached is not None:
        return cached
    try:
        sector = yf.Ticker(ticker).info.get("sector", "Unknown")
        _set_cached(cache_key, sector)
        return sector
    except Exception:
        return "Unknown"


def fetch_earnings_proximity(ticker: str) -> dict:
    """
    Check how close the next earnings date is.
    Earnings proximity affects volatility and price action.
    """
    try:
        tk = yf.Ticker(ticker)
        cal = tk.calendar
        if cal is not None and not cal.empty:
            if "Earnings Date" in cal.index:
                dates = cal.loc["Earnings Date"]
                if hasattr(dates, '__len__') and len(dates) > 0:
                    next_earnings = pd.Timestamp(dates.iloc[0])
                else:
                    next_earnings = pd.Timestamp(dates)

                days_until = (next_earnings - pd.Timestamp.now()).days
                return {
                    "next_earnings": next_earnings.strftime("%Y-%m-%d"),
                    "days_until": max(0, days_until),
                    "is_near": days_until <= 14,
                }
    except Exception:
        pass

    return {"next_earnings": None, "days_until": None, "is_near": False}


def fetch_earnings_history(ticker: str, period: str = "2y") -> pd.DataFrame:
    """Build a date-indexed DataFrame with days-to-nearest-earnings for each trading day."""
    cache_key = f"earnings_hist_{ticker}_{period}"
    cached = _get_cached(cache_key)
    if cached is not None:
        return cached

    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period=period, auto_adjust=True)
        if hist.empty:
            return pd.DataFrame()

        # Get historical earnings dates from the earnings_dates attribute
        try:
            edates = tk.get_earnings_dates(limit=20)
            if edates is not None and not edates.empty:
                earnings_dates = sorted(edates.index.tz_localize(None) if edates.index.tz else edates.index)
            else:
                return pd.DataFrame()
        except Exception:
            return pd.DataFrame()

        if hist.index.tz is not None:
            hist.index = hist.index.tz_localize(None)

        result = pd.DataFrame(index=hist.index)
        earnings_dates = sorted(earnings_dates)

        days_to_earnings = []
        for day in hist.index:
            future = [abs((ed - day).days) for ed in earnings_dates]
            days_to_earnings.append(min(future) if future else 90)

        result["days_to_earnings"] = days_to_earnings
        result["earnings_near"] = (result["days_to_earnings"] <= 14).astype(int)
        result["earnings_week"] = (result["days_to_earnings"] <= 7).astype(int)

        _set_cached(cache_key, result)
        return result
    except Exception:
        return pd.DataFrame()


# ── Fear & Greed Index ──────────────────────────────────────────────────────

def fetch_fear_greed() -> dict:
    """
    Fetch current CNN Fear & Greed Index (cached 5 min).
    Returns dict with score (0-100), rating, and historical snapshots.
    """
    cache_key = "fear_greed_current"
    cached = _get_cached(cache_key)
    if cached is not None:
        return cached

    try:
        import fear_greed
        data = fear_greed.get()
        result = {
            "score": data["score"],
            "rating": data["rating"],
            "history": data.get("history", {}),
            "indicators": data.get("indicators", {}),
        }
        _set_cached(cache_key, result)
        return result
    except Exception as e:
        return {"score": None, "rating": "unavailable", "error": str(e)}


def fetch_fear_greed_history(period: str = "6mo") -> pd.DataFrame:
    """
    Fetch historical Fear & Greed data as a DataFrame.
    Aligns with stock data for feature engineering.
    """
    cache_key = f"fear_greed_hist_{period}"
    cached = _get_cached(cache_key)
    if cached is not None:
        return cached

    try:
        import fear_greed
        # Map our period to fear_greed's format
        last_map = {"3mo": "3m", "6mo": "6m", "1y": "1y", "2y": "1y"}
        last = last_map.get(period, "6m")

        history = fear_greed.get_history(last=last)
        if not history:
            return pd.DataFrame()

        df = pd.DataFrame(history)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        df = df.rename(columns={"score": "fear_greed"})
        df = df[["fear_greed"]]

        # Add derived features
        df["fear_greed_ma5"] = df["fear_greed"].rolling(5).mean()
        df["fear_greed_change"] = df["fear_greed"].diff()
        df["extreme_fear"] = (df["fear_greed"] < 25).astype(int)
        df["extreme_greed"] = (df["fear_greed"] > 75).astype(int)

        _set_cached(cache_key, df)
        return df
    except Exception:
        return pd.DataFrame()

