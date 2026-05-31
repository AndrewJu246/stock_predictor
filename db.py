"""
db.py — SQLite storage for predictions, accuracy tracking, and model metadata.
"""

import sqlite3
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path


def _utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)

DB_PATH = Path(__file__).parent / "data" / "predictor.db"


def get_conn():
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    conn = get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS predictions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker      TEXT NOT NULL,
            created_at  TEXT NOT NULL,
            horizon     TEXT NOT NULL,          -- 'next_day' or 'weekly'
            pred_direction TEXT NOT NULL,       -- 'up' or 'down'
            pred_confidence REAL NOT NULL,      -- 0-1
            pred_pct_change REAL,               -- predicted % move
            actual_direction TEXT,              -- filled in later
            actual_pct_change REAL,             -- filled in later
            resolved_at TEXT,                   -- when we checked the outcome
            correct     INTEGER                 -- 1/0/NULL
        );

        CREATE TABLE IF NOT EXISTS model_meta (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker      TEXT NOT NULL,
            horizon     TEXT NOT NULL,
            trained_at  TEXT NOT NULL,
            sample_size INTEGER,
            accuracy    REAL,
            features    TEXT                    -- JSON list of feature names
        );

        CREATE TABLE IF NOT EXISTS news_cache (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker      TEXT NOT NULL,
            fetched_at  TEXT NOT NULL,
            title       TEXT,
            source      TEXT,
            sentiment   REAL,
            url         TEXT UNIQUE
        );

        CREATE INDEX IF NOT EXISTS idx_pred_ticker ON predictions(ticker);
        CREATE INDEX IF NOT EXISTS idx_pred_unresolved ON predictions(ticker, resolved_at);
        CREATE INDEX IF NOT EXISTS idx_news_ticker ON news_cache(ticker, fetched_at);

        CREATE TABLE IF NOT EXISTS portfolio (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker      TEXT NOT NULL,
            shares      REAL NOT NULL,
            buy_price   REAL NOT NULL,
            buy_date    TEXT NOT NULL,
            sell_price  REAL,
            sell_date   TEXT,
            status      TEXT NOT NULL DEFAULT 'open',  -- 'open' or 'closed'
            notes       TEXT
        );

        CREATE TABLE IF NOT EXISTS portfolio_snapshots (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_at TEXT NOT NULL,
            total_value REAL,
            total_cost  REAL,
            total_pnl   REAL,
            pnl_pct     REAL
        );

        CREATE INDEX IF NOT EXISTS idx_portfolio_status ON portfolio(status);

        CREATE TABLE IF NOT EXISTS daily_sentiment (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker      TEXT NOT NULL,
            date        TEXT NOT NULL,
            avg_score   REAL NOT NULL,
            num_articles INTEGER,
            positive    INTEGER,
            negative    INTEGER,
            neutral     INTEGER,
            engine      TEXT,
            UNIQUE(ticker, date)
        );

        CREATE INDEX IF NOT EXISTS idx_daily_sentiment ON daily_sentiment(ticker, date);
    """)
    conn.commit()
    conn.close()


# ── Predictions ──────────────────────────────────────────────────────────────

def save_prediction(ticker, horizon, direction, confidence, pct_change=None):
    conn = get_conn()
    conn.execute(
        """INSERT INTO predictions (ticker, created_at, horizon, pred_direction,
           pred_confidence, pred_pct_change)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (ticker, _utcnow().isoformat(), horizon, direction,
         confidence, pct_change)
    )
    conn.commit()
    conn.close()


def get_unresolved_predictions(ticker=None):
    conn = get_conn()
    q = "SELECT * FROM predictions WHERE resolved_at IS NULL"
    params = []
    if ticker:
        q += " AND ticker = ?"
        params.append(ticker)
    rows = conn.execute(q, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def resolve_prediction(pred_id, actual_direction, actual_pct_change):
    conn = get_conn()
    row = conn.execute("SELECT pred_direction FROM predictions WHERE id=?",
                       (pred_id,)).fetchone()
    if not row:
        conn.close()
        return
    correct = 1 if row["pred_direction"] == actual_direction else 0
    conn.execute(
        """UPDATE predictions
           SET actual_direction=?, actual_pct_change=?, resolved_at=?, correct=?
           WHERE id=?""",
        (actual_direction, actual_pct_change, _utcnow().isoformat(),
         correct, pred_id)
    )
    conn.commit()
    conn.close()
    return correct


def get_accuracy_stats(ticker=None, horizon=None, last_n=None, include_pre_fix=False):
    conn = get_conn()
    q = "SELECT * FROM predictions WHERE resolved_at IS NOT NULL"
    params = []
    if not include_pre_fix:
        q += " AND COALESCE(pre_fix, 0) = 0"
    if ticker:
        q += " AND ticker = ?"
        params.append(ticker)
    if horizon:
        q += " AND horizon = ?"
        params.append(horizon)
    q += " ORDER BY resolved_at DESC"
    if last_n:
        q += " LIMIT ?"
        params.append(last_n)

    rows = conn.execute(q, params).fetchall()
    conn.close()

    if not rows:
        return {"total": 0, "correct": 0, "accuracy": 0.0, "predictions": []}

    total = len(rows)
    correct = sum(1 for r in rows if r["correct"] == 1)
    return {
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total * 100, 1) if total else 0.0,
        "predictions": [dict(r) for r in rows],
    }


# ── Model metadata ──────────────────────────────────────────────────────────

def save_model_meta(ticker, horizon, sample_size, accuracy, features):
    conn = get_conn()
    conn.execute(
        """INSERT INTO model_meta (ticker, horizon, trained_at, sample_size,
           accuracy, features) VALUES (?, ?, ?, ?, ?, ?)""",
        (ticker, horizon, _utcnow().isoformat(), sample_size,
         accuracy, json.dumps(features))
    )
    conn.commit()
    conn.close()


def get_latest_model_meta(ticker, horizon):
    conn = get_conn()
    row = conn.execute(
        """SELECT * FROM model_meta
           WHERE ticker=? AND horizon=?
           ORDER BY trained_at DESC LIMIT 1""",
        (ticker, horizon)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


# ── News cache ───────────────────────────────────────────────────────────────

def cache_news(ticker, articles):
    """articles: list of dicts with title, source, sentiment, url"""
    conn = get_conn()
    now = _utcnow().isoformat()
    for a in articles:
        try:
            conn.execute(
                """INSERT OR IGNORE INTO news_cache
                   (ticker, fetched_at, title, source, sentiment, url)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (ticker, now, a.get("title"), a.get("source"),
                 a.get("sentiment"), a.get("url"))
            )
        except sqlite3.IntegrityError:
            pass
    conn.commit()
    conn.close()


def get_recent_news(ticker, days=7):
    conn = get_conn()
    cutoff = (_utcnow() - timedelta(days=days)).isoformat()
    rows = conn.execute(
        """SELECT * FROM news_cache
           WHERE ticker=? AND fetched_at >= ?
           ORDER BY fetched_at DESC""",
        (ticker, cutoff)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Daily Sentiment ─────────────────────────────────────────────────────────

def save_daily_sentiment(ticker, date_str, avg_score, num_articles=0,
                         positive=0, negative=0, neutral=0, engine=""):
    conn = get_conn()
    try:
        conn.execute(
            """INSERT OR REPLACE INTO daily_sentiment
               (ticker, date, avg_score, num_articles, positive, negative, neutral, engine)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (ticker.upper(), date_str, avg_score, num_articles,
             positive, negative, neutral, engine)
        )
        conn.commit()
    except Exception:
        pass
    conn.close()


def get_sentiment_history(ticker, days=500):
    conn = get_conn()
    cutoff = (_utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    rows = conn.execute(
        """SELECT date, avg_score, num_articles FROM daily_sentiment
           WHERE ticker=? AND date >= ?
           ORDER BY date ASC""",
        (ticker.upper(), cutoff)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# Auto-init on import
init_db()


def has_recent_prediction(ticker, horizon, minutes=10):
    """Check if a prediction was already made within the last N minutes."""
    conn = get_conn()
    cutoff = (_utcnow() - timedelta(minutes=minutes)).isoformat()
    row = conn.execute(
        """SELECT COUNT(*) as cnt FROM predictions
           WHERE ticker=? AND horizon=? AND created_at >= ?""",
        (ticker, horizon, cutoff)
    ).fetchone()
    conn.close()
    return row["cnt"] > 0


# ── Portfolio ────────────────────────────────────────────────────────────────

def add_position(ticker, shares, buy_price, buy_date=None, notes=""):
    conn = get_conn()
    if buy_date is None:
        buy_date = _utcnow().strftime("%Y-%m-%d")
    conn.execute(
        """INSERT INTO portfolio (ticker, shares, buy_price, buy_date, status, notes)
           VALUES (?, ?, ?, ?, 'open', ?)""",
        (ticker.upper(), shares, buy_price, buy_date, notes)
    )
    conn.commit()
    conn.close()


def close_position(position_id, sell_price, sell_date=None):
    conn = get_conn()
    if sell_date is None:
        sell_date = _utcnow().strftime("%Y-%m-%d")
    conn.execute(
        """UPDATE portfolio SET sell_price=?, sell_date=?, status='closed'
           WHERE id=?""",
        (sell_price, sell_date, position_id)
    )
    conn.commit()
    conn.close()


def get_open_positions():
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM portfolio WHERE status='open' ORDER BY buy_date DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_closed_positions():
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM portfolio WHERE status='closed' ORDER BY sell_date DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_all_positions():
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM portfolio ORDER BY status, buy_date DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_position(position_id):
    conn = get_conn()
    conn.execute("DELETE FROM portfolio WHERE id=?", (position_id,))
    conn.commit()
    conn.close()


def save_portfolio_snapshot(total_value, total_cost, total_pnl, pnl_pct):
    """Save snapshot, but only if the last one is >1 hour old."""
    conn = get_conn()
    # Check if we already have a recent snapshot
    cutoff = (_utcnow() - timedelta(hours=1)).isoformat()
    recent = conn.execute(
        "SELECT COUNT(*) as cnt FROM portfolio_snapshots WHERE snapshot_at >= ?",
        (cutoff,)
    ).fetchone()
    if recent and recent["cnt"] > 0:
        conn.close()
        return  # Skip — too recent

    conn.execute(
        """INSERT INTO portfolio_snapshots (snapshot_at, total_value, total_cost,
           total_pnl, pnl_pct) VALUES (?, ?, ?, ?, ?)""",
        (_utcnow().isoformat(), total_value, total_cost, total_pnl, pnl_pct)
    )
    conn.commit()
    conn.close()


def get_portfolio_snapshots(last_n=100):
    conn = get_conn()
    rows = conn.execute(
        """SELECT * FROM portfolio_snapshots
           ORDER BY snapshot_at DESC LIMIT ?""",
        (last_n,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
