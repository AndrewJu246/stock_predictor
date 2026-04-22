"""
sentiment.py — Analyzes news sentiment using VADER (no API keys needed).
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from data_fetcher import fetch_news
import db

analyzer = SentimentIntensityAnalyzer()

# Add finance-specific lexicon boosts
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
analyzer.lexicon.update(finance_lexicon)


def analyze_headline(headline: str) -> float:
    """Return compound sentiment score for a headline (-1 to 1)."""
    scores = analyzer.polarity_scores(headline)
    return scores["compound"]


def analyze_news_batch(articles: list) -> list:
    """Analyze sentiment for a list of article dicts. Adds 'sentiment' key."""
    for article in articles:
        article["sentiment"] = analyze_headline(article.get("title", ""))
    return articles


def get_ticker_sentiment(ticker: str, use_cache: bool = True) -> dict:
    """
    Fetch and analyze news for a ticker.
    Returns aggregate sentiment + individual article scores.
    """
    # Check cache first
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
                }

    # Fetch fresh news
    articles = fetch_news(ticker)
    articles = analyze_news_batch(articles)

    # Cache results
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
    }


def _sentiment_label(score: float) -> str:
    if score >= 0.15:
        return "Positive"
    elif score <= -0.15:
        return "Negative"
    else:
        return "Neutral"
