"""
sentiment.py — Financial sentiment analysis.
Primary: FinBERT via HuggingFace Inference API (free, no local torch needed).
Fallback: VADER with finance lexicon.
"""

import requests
import time
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from data_fetcher import fetch_news
import db

# ── HuggingFace Inference API (2026 router) ──────────────────────────────────

# Models to try in order (first working one wins)
HF_MODELS = [
    "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
    "ahmedrachid/FinancialBERT-Sentiment-Analysis",
    "ProsusAI/finbert",
]
HF_ROUTER_BASE = "https://router.huggingface.co/hf-inference/models/"
HF_HEADERS = {}

HAS_FINBERT = False
HF_API_URL = None


def _load_hf_token():
    """Load HF token from env var (Railway) or config.json (local)."""
    global HF_HEADERS
    import os

    # Check env var first (Railway deployment)
    token = os.environ.get("HF_TOKEN", "")

    # Fall back to config.json
    if not token:
        try:
            import json
            from pathlib import Path
            cfg_path = Path(__file__).parent / "config.json"
            with open(cfg_path) as f:
                cfg = json.load(f)
            token = cfg.get("settings", {}).get("hf_token", "")
        except Exception:
            pass

    if token:
        HF_HEADERS["Authorization"] = f"Bearer {token}"
        return True
    return False


def _check_finbert_api():
    """Test which HF financial sentiment model is available."""
    global HAS_FINBERT, HF_API_URL

    has_token = _load_hf_token()
    if not has_token:
        print("[WARN] No HuggingFace token found. Add 'hf_token' to config.json settings.")
        print("       Get a free token at: https://huggingface.co/settings/tokens")
        print("       Using VADER fallback.")
        return

    for model_id in HF_MODELS:
        url = HF_ROUTER_BASE + model_id + "/pipeline/text-classification"
        try:
            resp = requests.post(
                url,
                headers=HF_HEADERS,
                json={"inputs": "Stock market rises on strong earnings."},
                timeout=15,
            )
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list):
                    HAS_FINBERT = True
                    HF_API_URL = url
                    print(f"[INFO] Financial sentiment API connected: {model_id}")
                    return
            elif resp.status_code == 503:
                print(f"[INFO] {model_id} loading, retrying in 15s...")
                time.sleep(15)
                resp = requests.post(url, headers=HF_HEADERS,
                                     json={"inputs": "Test."}, timeout=15)
                if resp.status_code == 200:
                    HAS_FINBERT = True
                    HF_API_URL = url
                    print(f"[INFO] Financial sentiment API connected: {model_id}")
                    return
            print(f"[INFO] {model_id}: status {resp.status_code}, trying next...")
        except Exception as e:
            print(f"[INFO] {model_id}: {e}, trying next...")
            continue

    print("[WARN] No financial sentiment API available. Using VADER fallback.")


# Test on import
_check_finbert_api()


def _finbert_analyze(texts: list) -> list:
    """
    Call FinBERT API for a list of texts.
    Returns list of sentiment scores (-1 to 1).
    """
    if not texts:
        return []

    try:
        resp = requests.post(
            HF_API_URL,
            headers=HF_HEADERS,
            json={"inputs": texts},
            timeout=30,
        )
        if resp.status_code != 200:
            return None

        results = resp.json()

        scores = []
        for result in results:
            if isinstance(result, list):
                # Find the label with highest score
                best = max(result, key=lambda x: x["score"])
                label = best["label"].lower()
                score = best["score"]

                if label == "positive":
                    scores.append(round(score, 3))
                elif label == "negative":
                    scores.append(round(-score, 3))
                else:
                    scores.append(0.0)
            else:
                scores.append(0.0)

        return scores
    except Exception:
        return None


# ── VADER fallback ───────────────────────────────────────────────────────────

vader_analyzer = SentimentIntensityAnalyzer()

finance_lexicon = {
    "bull": 1.5, "bullish": 2.0, "bear": -1.5, "bearish": -2.0,
    "surge": 2.0, "surged": 2.0, "plunge": -2.5, "plunged": -2.5,
    "rally": 2.0, "crash": -3.0, "soar": 2.5, "soared": 2.5,
    "tumble": -2.0, "tumbled": -2.0, "boom": 2.0, "bust": -2.0,
    "upgrade": 1.8, "downgrade": -1.8, "outperform": 1.5,
    "underperform": -1.5, "beat": 1.2, "miss": -1.2, "missed": -1.2,
    "bankruptcy": -3.5, "layoff": -1.5, "layoffs": -1.5,
    "growth": 1.0, "decline": -1.0, "profit": 1.5, "loss": -1.5,
    "revenue": 0.5, "debt": -0.8, "dividend": 1.0,
    "acquisition": 0.8, "merger": 0.5, "ipo": 1.0,
    "lawsuit": -1.2, "investigation": -1.0, "recall": -1.5,
    "innovation": 1.2, "breakthrough": 2.0, "partnership": 0.8,
}
vader_analyzer.lexicon.update(finance_lexicon)


# ── Public API ───────────────────────────────────────────────────────────────

def analyze_headline(headline: str) -> float:
    """Return sentiment score for a headline (-1 to 1)."""
    if not headline or not headline.strip():
        return 0.0

    if HAS_FINBERT:
        scores = _finbert_analyze([headline])
        if scores and scores[0] is not None:
            return scores[0]

    # VADER fallback
    return vader_analyzer.polarity_scores(headline)["compound"]


def analyze_news_batch(articles: list) -> list:
    """Analyze sentiment for a list of article dicts. Adds 'sentiment' key."""
    headlines = [a.get("title", "") for a in articles]

    if HAS_FINBERT and headlines:
        # Batch call — HF API handles batches efficiently
        scores = _finbert_analyze(headlines)
        if scores and len(scores) == len(articles):
            for article, score in zip(articles, scores):
                article["sentiment"] = score
            return articles

    # VADER fallback
    for article in articles:
        article["sentiment"] = vader_analyzer.polarity_scores(
            article.get("title", "")
        )["compound"]
    return articles


def get_ticker_sentiment(ticker: str, use_cache: bool = True) -> dict:
    """
    Fetch and analyze news for a ticker.
    Returns aggregate sentiment + individual article scores.
    """
    if use_cache:
        cached = db.get_recent_news(ticker, days=1)
        if cached:
            sentiments = [a["sentiment"] for a in cached if a["sentiment"] is not None]
            if sentiments:
                avg = sum(sentiments) / len(sentiments)
                return {
                    "ticker": ticker,
                    "avg_sentiment": round(avg, 3),
                    "num_articles": len(cached),
                    "positive": sum(1 for s in sentiments if s > 0.05),
                    "negative": sum(1 for s in sentiments if s < -0.05),
                    "neutral": sum(1 for s in sentiments if -0.05 <= s <= 0.05),
                    "articles": cached,
                    "label": _sentiment_label(avg),
                    "engine": "cached",
                }

    articles = fetch_news(ticker)
    articles = analyze_news_batch(articles)
    db.cache_news(ticker, articles)

    sentiments = [a["sentiment"] for a in articles]
    avg = sum(sentiments) / len(sentiments) if sentiments else 0.0

    return {
        "ticker": ticker,
        "avg_sentiment": round(avg, 3),
        "num_articles": len(articles),
        "positive": sum(1 for s in sentiments if s > 0.05),
        "negative": sum(1 for s in sentiments if s < -0.05),
        "neutral": sum(1 for s in sentiments if -0.05 <= s <= 0.05),
        "articles": articles,
        "label": _sentiment_label(avg),
        "engine": "finbert" if HAS_FINBERT else "vader",
    }


def _sentiment_label(score: float) -> str:
    if score >= 0.15:
        return "Positive"
    elif score <= -0.15:
        return "Negative"
    else:
        return "Neutral"


def get_engine_name() -> str:
    """Return which sentiment engine is active."""
    return "FinBERT (API)" if HAS_FINBERT else "VADER"
