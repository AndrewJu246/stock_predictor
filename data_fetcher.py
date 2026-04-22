"""
data_fetcher.py — Pulls stock price data via yfinance and news via RSS feeds.
"""

import yfinance as yf
import feedparser
import pandas as pd
import json
from datetime import datetime, timedelta
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "config.json"


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
        # Validate ticker exists
        info = yf.Ticker(t).info
        if not info or info.get("regularMarketPrice") is None:
            raise ValueError(f"Ticker '{t}' not found")
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
    """Fetch OHLCV data for a ticker. Returns a DataFrame with technical columns."""
    tk = yf.Ticker(ticker)
    df = tk.history(period=period, auto_adjust=True)

    if df.empty:
        return df

    # Basic derived columns
    df["Return"] = df["Close"].pct_change()
    df["Return_5d"] = df["Close"].pct_change(5)
    import numpy as np
    df["Log_Volume"] = df["Volume"].apply(lambda v: np.log1p(v) if v > 0 else 0)

    return df


def fetch_current_price(ticker: str) -> dict:
    """Get the current / latest price info for a ticker."""
    tk = yf.Ticker(ticker)
    info = tk.info
    hist = tk.history(period="5d")

    if hist.empty:
        return {"ticker": ticker, "error": "No data"}

    latest = hist.iloc[-1]
    prev = hist.iloc[-2] if len(hist) > 1 else latest

    change = latest["Close"] - prev["Close"]
    change_pct = (change / prev["Close"]) * 100 if prev["Close"] else 0

    return {
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
