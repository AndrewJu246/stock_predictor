"""
app.py — Streamlit dashboard for the Stock Predictor.
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

# Must be first Streamlit call
st.set_page_config(
    page_title="Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Auto-refresh every 5 minutes (300000 ms)
refresh_count = st_autorefresh(interval=5 * 60 * 1000, key="auto_refresh")

import db
from data_fetcher import get_watchlist, add_ticker, remove_ticker, fetch_current_price, fetch_stock_data
from sentiment import get_ticker_sentiment
from predictor import predict, predict_all, resolve_predictions, train_model

# Auto-resolve old predictions on every refresh cycle
try:
    resolve_predictions()
except Exception:
    pass


# ── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.title("📈 Stock Predictor")
st.sidebar.markdown("---")

# Watchlist management
st.sidebar.subheader("Watchlist")
watchlist = get_watchlist()
st.sidebar.write(", ".join(watchlist) if watchlist else "No stocks added")

col1, col2 = st.sidebar.columns(2)
with col1:
    new_ticker = st.text_input("Add ticker", placeholder="e.g. NVDA")
    if st.button("Add", use_container_width=True):
        if new_ticker:
            try:
                updated = add_ticker(new_ticker)
                st.success(f"Added {new_ticker.upper()}")
                st.rerun()
            except ValueError as e:
                st.error(str(e))

with col2:
    remove_t = st.selectbox("Remove", [""] + watchlist)
    if st.button("Remove", use_container_width=True):
        if remove_t:
            remove_ticker(remove_t)
            st.success(f"Removed {remove_t}")
            st.rerun()

st.sidebar.markdown("---")

# Actions
st.sidebar.subheader("Actions")
if st.sidebar.button("🔄 Resolve Past Predictions", use_container_width=True):
    with st.spinner("Checking predictions against actual data..."):
        count = resolve_predictions()
    st.sidebar.success(f"Resolved {count} predictions")

if st.sidebar.button("🧠 Retrain All Models", use_container_width=True):
    with st.spinner("Retraining models..."):
        for t in watchlist:
            for h in ["next_day", "weekly"]:
                try:
                    train_model(t, h)
                except Exception:
                    pass
    st.sidebar.success("Models retrained!")


# ── Main content ─────────────────────────────────────────────────────────────

st.title("Stock Predictor Dashboard")
st.caption(f"Auto-refreshes every 5 min · Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · Cycle #{refresh_count}")

if not watchlist:
    st.info("Add some tickers in the sidebar to get started.")
    st.stop()

# Tabs
tab_overview, tab_detail, tab_accuracy = st.tabs(["Overview", "Stock Detail", "Accuracy"])


# ── TAB: Overview ────────────────────────────────────────────────────────────

with tab_overview:
    st.subheader("Market Overview")

    # Predictions
    horizon = st.radio("Prediction horizon", ["next_day", "weekly"],
                       horizontal=True, format_func=lambda x: "Next Day" if x == "next_day" else "Weekly")

    with st.spinner("Running predictions..."):
        predictions = predict_all(horizon)

    # Build overview table
    rows = []
    for p in predictions:
        if "error" in p:
            rows.append({
                "Ticker": p.get("ticker", "?"),
                "Price": "—",
                "Signal": "Error",
                "Confidence": "—",
                "Sentiment": "—",
                "Model Acc.": "—",
                "Track Acc.": "—",
            })
            continue

        # Fetch current price
        try:
            price_data = fetch_current_price(p["ticker"])
            price_str = f"${price_data['price']}"
            change_str = f"{price_data['change_pct']:+.2f}%"
        except Exception:
            price_str = "—"
            change_str = ""

        rows.append({
            "Ticker": p["ticker"],
            "Price": price_str,
            "Change": change_str,
            "Signal": p["signal"],
            "Direction": p["direction"].upper(),
            "Confidence": f"{p['confidence']}%",
            "Sentiment": p.get("sentiment_label", "N/A"),
            "Model Acc.": f"{p['model_cv_accuracy']}%" if p.get("model_cv_accuracy") else "—",
            "Track Acc.": f"{p['tracked_accuracy']}% ({p['tracked_total']})" if p.get("tracked_total", 0) > 0 else "No data yet",
        })

    df_overview = pd.DataFrame(rows)

    # Color-code the signal column
    def color_signal(val):
        if "Strong Buy" in str(val):
            return "background-color: #1b5e20; color: white"
        elif "Buy" in str(val):
            return "background-color: #388e3c; color: white"
        elif "Strong Sell" in str(val):
            return "background-color: #b71c1c; color: white"
        elif "Sell" in str(val):
            return "background-color: #c62828; color: white"
        elif "Hold" in str(val):
            return "background-color: #f57f17; color: white"
        return ""

    styled = df_overview.style.map(color_signal, subset=["Signal"])
    st.dataframe(styled, use_container_width=True, hide_index=True)


# ── TAB: Stock Detail ───────────────────────────────────────────────────────

with tab_detail:
    selected = st.selectbox("Select stock", watchlist)

    if selected:
        col_price, col_pred = st.columns([2, 1])

        with col_price:
            st.subheader(f"{selected} Price Chart")
            df_hist = fetch_stock_data(selected, period="6mo")

            if not df_hist.empty:
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=df_hist.index,
                    open=df_hist["Open"],
                    high=df_hist["High"],
                    low=df_hist["Low"],
                    close=df_hist["Close"],
                    name="Price",
                ))

                # Add SMA lines
                sma20 = df_hist["Close"].rolling(20).mean()
                sma50 = df_hist["Close"].rolling(50).mean()
                fig.add_trace(go.Scatter(x=df_hist.index, y=sma20,
                                          name="SMA 20", line=dict(width=1)))
                fig.add_trace(go.Scatter(x=df_hist.index, y=sma50,
                                          name="SMA 50", line=dict(width=1)))

                fig.update_layout(
                    height=450,
                    xaxis_rangeslider_visible=False,
                    template="plotly_dark",
                    margin=dict(t=20, b=20),
                )
                st.plotly_chart(fig, use_container_width=True)

        with col_pred:
            st.subheader("Predictions")

            for h in ["next_day", "weekly"]:
                label = "Next Day" if h == "next_day" else "Weekly"
                try:
                    p = predict(selected, h)
                    if "error" not in p:
                        direction_icon = "🟢" if p["direction"] == "up" else "🔴"
                        st.markdown(f"**{label}:** {direction_icon} {p['signal']}")
                        st.markdown(f"Confidence: {p['confidence']}%")
                        if p.get("tracked_total", 0) > 0:
                            st.markdown(f"Track record: {p['tracked_accuracy']}% "
                                        f"({p['tracked_total']} predictions)")
                        st.markdown("---")
                    else:
                        st.warning(f"{label}: {p['error']}")
                except Exception as e:
                    st.error(f"{label}: {e}")

            # Sentiment section
            st.subheader("News Sentiment")
            try:
                sent = get_ticker_sentiment(selected, use_cache=False)
                sentiment_color = {"Positive": "🟢", "Negative": "🔴", "Neutral": "🟡"}
                icon = sentiment_color.get(sent["label"], "⚪")
                st.markdown(f"{icon} **{sent['label']}** (score: {sent['avg_sentiment']})")
                st.markdown(f"Articles: {sent['num_articles']} "
                            f"(+{sent['positive']} / ={sent['neutral']} / -{sent['negative']})")

                # Show top headlines
                if sent["articles"]:
                    with st.expander("Recent headlines"):
                        for a in sent["articles"][:8]:
                            s = a.get("sentiment", 0)
                            icon = "🟢" if s > 0.05 else ("🔴" if s < -0.05 else "⚪")
                            title = a.get("title", "No title")
                            url = a.get("url", "")
                            st.markdown(f"{icon} [{title}]({url})")

            except Exception as e:
                st.warning(f"Could not fetch sentiment: {e}")


# ── TAB: Accuracy ────────────────────────────────────────────────────────────

with tab_accuracy:
    st.subheader("Prediction Accuracy Tracker")
    st.caption("This is how the model self-improves: predictions are compared against "
               "actual outcomes, and models retrain based on this feedback.")

    acc_ticker = st.selectbox("Filter by ticker", ["All"] + watchlist, key="acc_ticker")
    acc_horizon = st.selectbox("Filter by horizon", ["All", "next_day", "weekly"],
                               key="acc_horizon",
                               format_func=lambda x: {"All": "All", "next_day": "Next Day",
                                                       "weekly": "Weekly"}[x])

    t = acc_ticker if acc_ticker != "All" else None
    h = acc_horizon if acc_horizon != "All" else None
    stats = db.get_accuracy_stats(t, h, last_n=100)

    if stats["total"] == 0:
        st.info("No resolved predictions yet. Predictions are checked against actual "
                "outcomes automatically. Click **Resolve Past Predictions** in the sidebar "
                "after some time has passed.")
    else:
        # Summary metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Predictions", stats["total"])
        c2.metric("Correct", stats["correct"])
        c3.metric("Accuracy", f"{stats['accuracy']}%")

        # Accuracy over time
        preds_df = pd.DataFrame(stats["predictions"])
        preds_df["resolved_at"] = pd.to_datetime(preds_df["resolved_at"])
        preds_df = preds_df.sort_values("resolved_at")
        preds_df["rolling_acc"] = preds_df["correct"].expanding().mean() * 100

        fig_acc = go.Figure()
        fig_acc.add_trace(go.Scatter(
            x=preds_df["resolved_at"],
            y=preds_df["rolling_acc"],
            mode="lines+markers",
            name="Cumulative Accuracy",
        ))
        fig_acc.add_hline(y=50, line_dash="dash", line_color="gray",
                          annotation_text="Random (50%)")
        fig_acc.update_layout(
            height=350,
            yaxis_title="Accuracy %",
            template="plotly_dark",
            margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig_acc, use_container_width=True)

        # Recent predictions table
        with st.expander("Recent prediction history"):
            display_df = preds_df[["ticker", "horizon", "pred_direction",
                                    "pred_confidence", "actual_direction",
                                    "actual_pct_change", "correct",
                                    "resolved_at"]].copy()
            display_df.columns = ["Ticker", "Horizon", "Predicted", "Confidence",
                                  "Actual", "Actual %", "Correct", "Resolved"]
            display_df["Correct"] = display_df["Correct"].map({1: "✅", 0: "❌"})
            st.dataframe(display_df, use_container_width=True, hide_index=True)


# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("⚠️ This tool is for educational and personal analysis only. "
           "Not financial advice. Past predictions do not guarantee future results.")
