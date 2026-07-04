"""
data_fetcher.py — Pulls stock price data via yfinance and news via RSS feeds.
"""

import os
import yfinance as yf
import feedparser
import requests
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

CONFIG_PATH = Path(__file__).parent / "config.json"

# Simple time-based cache for expensive API calls
_cache = {}
_CACHE_TTL = 300  # 5 minutes


def _get_cached(key, ttl=None):
    """Return cached value if it exists and is fresh, else None."""
    if key in _cache:
        val, ts, custom_ttl = _cache[key]
        effective_ttl = ttl or custom_ttl or _CACHE_TTL
        if (datetime.now() - ts).total_seconds() < effective_ttl:
            return val
    return None


def _set_cached(key, val, ttl=None):
    _cache[key] = (val, datetime.now(), ttl)


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

        days_to_list = []
        days_since_list = []
        for day in hist.index:
            future = [(ed - day).days for ed in earnings_dates if ed >= day]
            past = [(day - ed).days for ed in earnings_dates if ed < day]
            days_to_list.append(min(future) if future else 90)
            days_since_list.append(min(past) if past else 90)

        result["days_to_earnings"] = days_to_list
        result["days_since_earnings"] = days_since_list
        result["earnings_near"] = (result["days_to_earnings"] <= 14).astype(int)
        result["earnings_week"] = (result["days_to_earnings"] <= 7).astype(int)
        result["post_earnings"] = (result["days_since_earnings"] <= 5).astype(int)

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
        # Map our period to fear_greed's format. The source only goes back
        # ~2y, so longer periods get the max — callers neutral-fill the rest
        last_map = {"3mo": "3m", "6mo": "6m", "1y": "1y", "2y": "2y", "5y": "2y"}
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


# ── Insider Transactions (via yfinance) ────────────────────────────────────

def fetch_insider_transactions(ticker: str, period: str = "2y") -> pd.DataFrame:
    """
    Fetch insider buy/sell transactions via yfinance.
    Returns a date-indexed DataFrame with net insider activity signals.
    """
    cache_key = f"insider_{ticker}_{period}"
    cached = _get_cached(cache_key)
    if cached is not None:
        return cached

    try:
        tk = yf.Ticker(ticker)
        transactions = tk.insider_transactions
        if transactions is None or transactions.empty:
            return pd.DataFrame()

        df = transactions.copy()

        if "Start Date" in df.columns:
            df["date"] = pd.to_datetime(df["Start Date"], errors="coerce")
        elif "Date" in df.columns:
            df["date"] = pd.to_datetime(df["Date"], errors="coerce")
        else:
            return pd.DataFrame()

        df = df.dropna(subset=["date"])
        if df.empty:
            return pd.DataFrame()

        if "Shares" not in df.columns and "Value" not in df.columns:
            return pd.DataFrame()

        shares_col = "Shares" if "Shares" in df.columns else None
        text_col = "Text" if "Text" in df.columns else ("Transaction" if "Transaction" in df.columns else None)

        buys = []
        sells = []
        for _, row in df.iterrows():
            txt = str(row.get(text_col, "")).lower() if text_col else ""
            shares = abs(float(row.get(shares_col, 0))) if shares_col else 0

            if any(w in txt for w in ["purchase", "bought", "buy", "acquisition"]):
                buys.append({"date": row["date"], "shares": shares, "type": "buy"})
            elif any(w in txt for w in ["sale", "sold", "sell", "disposition"]):
                sells.append({"date": row["date"], "shares": shares, "type": "sell"})

        all_txns = buys + sells
        if not all_txns:
            return pd.DataFrame()

        txn_df = pd.DataFrame(all_txns)
        txn_df = txn_df.set_index("date").sort_index()

        hist = tk.history(period=period, auto_adjust=True)
        if hist.empty:
            return pd.DataFrame()
        if hist.index.tz is not None:
            hist.index = hist.index.tz_localize(None)

        result = pd.DataFrame(index=hist.index)

        daily_buys = txn_df[txn_df["type"] == "buy"].resample("D")["shares"].sum()
        daily_sells = txn_df[txn_df["type"] == "sell"].resample("D")["shares"].sum()

        result["insider_buys"] = daily_buys.reindex(result.index).fillna(0)
        result["insider_sells"] = daily_sells.reindex(result.index).fillna(0)
        result["insider_net"] = result["insider_buys"] - result["insider_sells"]

        result["insider_buy_30d"] = result["insider_buys"].rolling(30, min_periods=1).sum()
        result["insider_sell_30d"] = result["insider_sells"].rolling(30, min_periods=1).sum()
        result["insider_net_30d"] = result["insider_buy_30d"] - result["insider_sell_30d"]
        result["insider_signal"] = np.sign(result["insider_net_30d"])

        _set_cached(cache_key, result)
        return result
    except Exception:
        return pd.DataFrame()


# ── FRED Macro Data ────────────────────────────────────────────────────────

FRED_SERIES = {
    "CPIAUCSL": "cpi",
    "UNRATE": "unemployment",
    "FEDFUNDS": "fed_funds",
}

FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"


def _fetch_fred_series(series_id: str, api_key: str, period: str = "2y") -> pd.DataFrame:
    """Fetch a single FRED series as a date-indexed DataFrame."""
    # 5y gets extra headroom so cpi_yoy_change (365d pct_change) has warm-up
    period_days = {"3mo": 120, "6mo": 200, "1y": 400, "2y": 800, "5y": 2200}
    lookback = period_days.get(period, 800)
    start = (datetime.now() - timedelta(days=lookback)).strftime("%Y-%m-%d")

    resp = requests.get(FRED_BASE_URL, params={
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start,
    }, timeout=15)

    if resp.status_code != 200:
        return pd.DataFrame()

    data = resp.json().get("observations", [])
    if not data:
        return pd.DataFrame()

    rows = []
    for obs in data:
        if obs["value"] != ".":
            rows.append({"date": obs["date"], "value": float(obs["value"])})

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    return df


def fetch_fred_data(period: str = "2y") -> pd.DataFrame:
    """
    Fetch CPI, unemployment rate, and fed funds rate from FRED.
    Monthly data forward-filled into daily frequency.
    Requires FRED_API_KEY in .env (free at https://fred.stlouisfed.org/docs/api/api_key.html).
    """
    cache_key = f"fred_{period}"
    cached = _get_cached(cache_key)
    if cached is not None:
        return cached

    api_key = os.environ.get("FRED_API_KEY", "")
    if not api_key:
        return pd.DataFrame()

    result = pd.DataFrame()

    for series_id, col_name in FRED_SERIES.items():
        try:
            df = _fetch_fred_series(series_id, api_key, period)
            if not df.empty:
                df = df.rename(columns={"value": col_name})
                if result.empty:
                    result = df
                else:
                    result = result.join(df, how="outer")
        except Exception:
            continue

    if result.empty:
        return result

    # Resample monthly -> daily and forward-fill
    result = result.resample("D").ffill()

    # Derived features
    if "cpi" in result.columns:
        result["cpi_yoy_change"] = result["cpi"].pct_change(365)
        result["cpi_mom_change"] = result["cpi"].pct_change(30)
    if "fed_funds" in result.columns:
        result["fed_funds_change"] = result["fed_funds"].diff()
    if "unemployment" in result.columns:
        result["unemployment_change"] = result["unemployment"].diff()

    _set_cached(cache_key, result)
    return result


# ── Fundamental Valuation Data ─────────────────────────────────────────────

_FUNDAMENTALS_TTL = 86400  # 24 hours — fundamentals change quarterly

FUNDAMENTAL_KEYS = {
    "trailingPE": "pe_ratio",
    "forwardPE": "forward_pe",
    "pegRatio": "peg_ratio",
    "priceToBook": "price_to_book",
    "trailingEps": "eps",
    "revenueGrowth": "revenue_growth",
    "earningsGrowth": "earnings_growth",
    "profitMargins": "profit_margin",
    "returnOnEquity": "roe",
    "debtToEquity": "debt_to_equity",
}


def fetch_fundamentals(ticker: str, period: str = "2y") -> pd.DataFrame:
    """
    Fetch fundamental valuation metrics for a ticker.
    Returns a DataFrame with constant values across all trading days
    (fundamentals change quarterly, forward-filled into daily data).
    Cached for 24 hours.
    """
    cache_key = f"fundamentals_{ticker}"
    cached = _get_cached(cache_key, ttl=_FUNDAMENTALS_TTL)
    if cached is not None:
        return cached

    try:
        tk = yf.Ticker(ticker)
        info = tk.info
        if not info:
            return pd.DataFrame()

        hist = tk.history(period=period, auto_adjust=True)
        if hist.empty:
            return pd.DataFrame()
        if hist.index.tz is not None:
            hist.index = hist.index.tz_localize(None)

        result = pd.DataFrame(index=hist.index)

        for yf_key, col_name in FUNDAMENTAL_KEYS.items():
            val = info.get(yf_key)
            if val is not None and val != 0:
                result[col_name] = float(val)
            else:
                result[col_name] = np.nan

        # Relative valuation vs sector median
        sector = info.get("sector", "")
        sector_pe = _get_sector_median_pe(sector)
        if sector_pe and "pe_ratio" in result.columns:
            pe_val = result["pe_ratio"].iloc[0]
            if not np.isnan(pe_val) and sector_pe > 0:
                result["pe_vs_sector"] = pe_val / sector_pe

        _set_cached(cache_key, result, ttl=_FUNDAMENTALS_TTL)
        return result
    except Exception:
        return pd.DataFrame()


_SECTOR_PE_MEDIANS = {
    "Technology": 30,
    "Healthcare": 22,
    "Financial Services": 14,
    "Consumer Cyclical": 20,
    "Consumer Defensive": 22,
    "Energy": 12,
    "Industrials": 20,
    "Materials": 16,
    "Real Estate": 35,
    "Utilities": 18,
    "Communication Services": 18,
}


def _get_sector_median_pe(sector: str) -> float:
    return _SECTOR_PE_MEDIANS.get(sector, 0)
