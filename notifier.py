"""
notifier.py — Email notifications for strong signals and daily summaries.
Uses Gmail SMTP (free, no third-party service needed).
"""

import smtplib
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "config.json"


def _get_email_config():
    """Load email settings from env vars (Railway) or config.json (local)."""
    import os

    # Check env vars first (Railway deployment)
    if os.environ.get("EMAIL_ENABLED", "").lower() == "true":
        return {
            "enabled": True,
            "smtp_server": os.environ.get("EMAIL_SMTP_SERVER", "smtp.gmail.com"),
            "smtp_port": int(os.environ.get("EMAIL_SMTP_PORT", "587")),
            "from_address": os.environ.get("EMAIL_FROM", ""),
            "password": os.environ.get("EMAIL_PASSWORD", ""),
            "to_address": os.environ.get("EMAIL_TO", ""),
        }

    # Fall back to config.json
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    return cfg.get("settings", {}).get("email", {})


def _send_email(subject: str, body_html: str):
    """Send an email via SMTP."""
    email_cfg = _get_email_config()

    if not email_cfg.get("enabled"):
        return False

    required = ["smtp_server", "smtp_port", "from_address", "password", "to_address"]
    for field in required:
        if not email_cfg.get(field):
            print(f"[WARN] Email not configured: missing '{field}'")
            return False

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = email_cfg["from_address"]
        msg["To"] = email_cfg["to_address"]
        msg.attach(MIMEText(body_html, "html"))

        with smtplib.SMTP(email_cfg["smtp_server"], email_cfg["smtp_port"]) as server:
            server.starttls()
            server.login(email_cfg["from_address"], email_cfg["password"])
            server.send_message(msg)

        print(f"[INFO] Email sent: {subject}")
        return True
    except Exception as e:
        print(f"[WARN] Email failed: {e}")
        return False


def send_signal_alert(predictions: list):
    """
    Send an alert if any predictions have strong signals.
    Only sends for Strong Buy or Strong Sell.
    """
    strong_signals = [p for p in predictions
                      if "error" not in p and "Strong" in p.get("signal", "")]

    if not strong_signals:
        return

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    rows = ""
    for p in strong_signals:
        color = "#4CAF50" if "Buy" in p["signal"] else "#f44336"
        rows += f"""
        <tr>
            <td style="padding:8px;border-bottom:1px solid #333">{p['ticker']}</td>
            <td style="padding:8px;border-bottom:1px solid #333;color:{color};font-weight:bold">{p['signal']}</td>
            <td style="padding:8px;border-bottom:1px solid #333">{p['confidence']}%</td>
            <td style="padding:8px;border-bottom:1px solid #333">{p.get('consensus', '—')}</td>
            <td style="padding:8px;border-bottom:1px solid #333">{p.get('sentiment_label', 'N/A')}</td>
        </tr>
        """

    tickers = ", ".join(p["ticker"] for p in strong_signals)
    subject = f"📈 Strong Signal: {tickers} — {now}"

    body = f"""
    <html><body style="font-family:Arial,sans-serif;background:#1a1a1a;color:#e0e0e0;padding:20px">
    <h2 style="color:#4CAF50">Stock Predictor Alert</h2>
    <p>{len(strong_signals)} strong signal(s) detected at {now}:</p>
    <table style="border-collapse:collapse;width:100%;background:#2a2a2a;border-radius:8px">
        <tr style="background:#333">
            <th style="padding:10px;text-align:left">Ticker</th>
            <th style="padding:10px;text-align:left">Signal</th>
            <th style="padding:10px;text-align:left">Confidence</th>
            <th style="padding:10px;text-align:left">Consensus</th>
            <th style="padding:10px;text-align:left">Sentiment</th>
        </tr>
        {rows}
    </table>
    <p style="color:#888;font-size:12px;margin-top:20px">
        This is an automated alert from your Stock Predictor. Not financial advice.
    </p>
    </body></html>
    """

    _send_email(subject, body)


def send_daily_summary(predictions: list, accuracy_stats: dict = None):
    """Send a daily summary of all predictions and portfolio performance."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    rows = ""
    for p in predictions:
        if "error" in p:
            continue
        color = "#4CAF50" if p["direction"] == "up" else "#f44336"
        rows += f"""
        <tr>
            <td style="padding:6px;border-bottom:1px solid #333">{p['ticker']}</td>
            <td style="padding:6px;border-bottom:1px solid #333;color:{color}">{p['signal']}</td>
            <td style="padding:6px;border-bottom:1px solid #333">{p['confidence']}%</td>
            <td style="padding:6px;border-bottom:1px solid #333">{p.get('consensus', '—')}</td>
            <td style="padding:6px;border-bottom:1px solid #333">{p.get('sentiment_label', 'N/A')}</td>
        </tr>
        """

    acc_section = ""
    if accuracy_stats and accuracy_stats.get("total", 0) > 0:
        acc_section = f"""
        <h3 style="color:#2196F3">Accuracy Tracker</h3>
        <p>Total predictions resolved: {accuracy_stats['total']} |
           Correct: {accuracy_stats['correct']} |
           Accuracy: {accuracy_stats['accuracy']}%</p>
        """

    subject = f"📊 Daily Summary — {now}"

    body = f"""
    <html><body style="font-family:Arial,sans-serif;background:#1a1a1a;color:#e0e0e0;padding:20px">
    <h2 style="color:#2196F3">Stock Predictor — Daily Summary</h2>
    <p>Generated at {now}</p>

    <h3 style="color:#4CAF50">Predictions</h3>
    <table style="border-collapse:collapse;width:100%;background:#2a2a2a;border-radius:8px">
        <tr style="background:#333">
            <th style="padding:8px;text-align:left">Ticker</th>
            <th style="padding:8px;text-align:left">Signal</th>
            <th style="padding:8px;text-align:left">Confidence</th>
            <th style="padding:8px;text-align:left">Consensus</th>
            <th style="padding:8px;text-align:left">Sentiment</th>
        </tr>
        {rows}
    </table>

    {acc_section}

    <p style="color:#888;font-size:12px;margin-top:20px">
        Automated daily summary. Not financial advice.
    </p>
    </body></html>
    """

    _send_email(subject, body)
