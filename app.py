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
from data_fetcher import get_watchlist, add_ticker, remove_ticker, fetch_current_price, fetch_stock_data, fetch_fear_greed
from sentiment import get_ticker_sentiment, get_engine_name
from predictor import predict, predict_all, resolve_predictions, train_model, get_feature_importance
from backtester import run_backtest
from risk_manager import calculate_stop_loss, calculate_position_size, analyze_diversification

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
    if st.button("Add", width="stretch"):
        if new_ticker:
            try:
                updated = add_ticker(new_ticker)
                st.success(f"Added {new_ticker.upper()}")
                st.rerun()
            except ValueError as e:
                st.error(str(e))

with col2:
    remove_t = st.selectbox("Remove", [""] + watchlist)
    if st.button("Remove", width="stretch"):
        if remove_t:
            remove_ticker(remove_t)
            st.success(f"Removed {remove_t}")
            st.rerun()

st.sidebar.markdown("---")

# Actions
st.sidebar.subheader("Actions")
if st.sidebar.button("🔄 Resolve Past Predictions", width="stretch"):
    with st.spinner("Checking predictions against actual data..."):
        count = resolve_predictions()
    st.sidebar.success(f"Resolved {count} predictions")

if st.sidebar.button("🧠 Retrain All Models", width="stretch"):
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
st.caption(f"Auto-refreshes every 5 min · Sentiment: {get_engine_name()} · Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · Cycle #{refresh_count}")

if not watchlist:
    st.info("Add some tickers in the sidebar to get started.")
    st.stop()

# Tabs
tab_overview, tab_detail, tab_accuracy, tab_backtest, tab_portfolio, tab_features = st.tabs(
    ["Overview", "Stock Detail", "Accuracy", "Backtest", "Portfolio", "Features"]
)


# ── TAB: Overview ────────────────────────────────────────────────────────────

with tab_overview:
    st.subheader("Market Overview")

    # Fear & Greed Index
    try:
        fg = fetch_fear_greed()
        if fg.get("score") is not None:
            fg_score = fg["score"]
            fg_rating = fg["rating"].title()
            fg_hist = fg.get("history", {})

            # Color based on score
            if fg_score <= 25:
                fg_color = "🔴"
            elif fg_score <= 45:
                fg_color = "🟠"
            elif fg_score <= 55:
                fg_color = "🟡"
            elif fg_score <= 75:
                fg_color = "🟢"
            else:
                fg_color = "🔵"

            fg_cols = st.columns([1, 1, 1, 1, 1])
            fg_cols[0].metric("Fear & Greed", f"{fg_color} {fg_score:.0f}", fg_rating)
            if fg_hist.get("1w"):
                fg_cols[1].metric("1 Week Ago", f"{fg_hist['1w']:.0f}")
            if fg_hist.get("1m"):
                fg_cols[2].metric("1 Month Ago", f"{fg_hist['1m']:.0f}")
            if fg_hist.get("3m"):
                fg_cols[3].metric("3 Months Ago", f"{fg_hist['3m']:.0f}")
            if fg_hist.get("1y"):
                fg_cols[4].metric("1 Year Ago", f"{fg_hist['1y']:.0f}")

            st.markdown("---")
    except Exception:
        pass

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
                "Change": "—",
                "Signal": "Error",
                "Direction": "—",
                "Confidence": "—",
                "Consensus": "—",
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
            "Consensus": p.get("consensus", "—"),
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
    st.dataframe(styled, width="stretch", hide_index=True)


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
                st.plotly_chart(fig, width="stretch")

        with col_pred:
            st.subheader("Predictions")

            for h in ["next_day", "weekly"]:
                label = "Next Day" if h == "next_day" else "Weekly"
                try:
                    p = predict(selected, h)
                    if "error" not in p:
                        direction_icon = "🟢" if p["direction"] == "up" else "🔴"
                        st.markdown(f"**{label}:** {direction_icon} {p['signal']}")
                        st.markdown(f"Confidence: {p['confidence']}% · {p.get('consensus', '')}")

                        # Show individual model votes
                        if p.get("model_votes"):
                            vote_strs = []
                            for v in p["model_votes"]:
                                icon = "↑" if v["direction"] == "up" else "↓"
                                vote_strs.append(f"{v['name']}: {icon} {v['confidence']*100:.0f}%")
                            st.caption(" · ".join(vote_strs))

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
                sent = get_ticker_sentiment(selected, use_cache=True)
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

        # ── Risk Management section (full width) ────────────────────────
        st.markdown("---")
        st.subheader(f"Risk Analysis — {selected}")

        try:
            sl = calculate_stop_loss(selected)
            if "error" not in sl:
                rc1, rc2, rc3 = st.columns(3)

                with rc1:
                    st.markdown("**ATR Stop-Loss Levels**")
                    st.markdown(f"Conservative: **${sl['stop_conservative']}** "
                                f"(-{sl['stop_conservative_pct']}%)")
                    st.markdown(f"Moderate: **${sl['stop_moderate']}** "
                                f"(-{sl['stop_moderate_pct']}%)")
                    st.markdown(f"Aggressive: **${sl['stop_aggressive']}** "
                                f"(-{sl['stop_aggressive_pct']}%)")

                with rc2:
                    st.markdown("**Fixed % Stop-Loss**")
                    st.markdown(f"5%: ${sl['stop_5pct']}")
                    st.markdown(f"8%: ${sl['stop_8pct']}")
                    st.markdown(f"10%: ${sl['stop_10pct']}")
                    st.markdown(f"Recent support: ${sl['recent_support']} "
                                f"(-{sl['support_distance_pct']}%)")

                with rc3:
                    st.markdown("**Volatility**")
                    st.markdown(f"Daily: {sl['daily_volatility_pct']}%")
                    st.markdown(f"Annualized: {sl['annualized_volatility_pct']}%")
                    st.markdown(f"ATR (14-day): ${sl['atr']}")

                # Position sizing calculator
                with st.expander("📐 Position Size Calculator"):
                    ps_port = st.number_input("Your portfolio value ($)",
                                               min_value=100.0, value=10000.0,
                                               step=1000.0, key="ps_port")
                    ps_risk = st.slider("Risk per trade (%)", 0.5, 5.0, 2.0,
                                         0.5, key="ps_risk")

                    sizing = calculate_position_size(selected, ps_port, ps_risk)
                    if "error" not in sizing:
                        st.markdown(f"**Recommended:** {sizing['recommended_shares']} shares "
                                    f"(${sizing['position_value']:,.2f} = "
                                    f"{sizing['position_pct_of_portfolio']:.1f}% of portfolio)")
                        st.markdown(f"Max risk: ${sizing['max_risk_dollars']:.2f} | "
                                    f"Stop-loss: {sizing['stop_loss_pct']}% | "
                                    f"Risk per share: ${sizing['risk_per_share']:.2f}")
                    else:
                        st.warning(sizing["error"])
            else:
                st.warning(sl["error"])
        except Exception as e:
            st.warning(f"Could not calculate risk: {e}")


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
        st.plotly_chart(fig_acc, width="stretch")

        # Recent predictions table
        with st.expander("Recent prediction history"):
            display_df = preds_df[["ticker", "horizon", "pred_direction",
                                    "pred_confidence", "actual_direction",
                                    "actual_pct_change", "correct",
                                    "resolved_at"]].copy()
            display_df.columns = ["Ticker", "Horizon", "Predicted", "Confidence",
                                  "Actual", "Actual %", "Correct", "Resolved"]
            display_df["Correct"] = display_df["Correct"].map({1: "✅", 0: "❌"})
            st.dataframe(display_df, width="stretch", hide_index=True)


# ── TAB: Backtest ────────────────────────────────────────────────────────────

with tab_backtest:
    st.subheader("Historical Backtesting")
    st.caption("Walk-forward simulation: train on past data, predict the next day/week, "
               "slide forward. Shows how the model would have performed historically.")

    bt_col1, bt_col2 = st.columns(2)
    with bt_col1:
        bt_ticker = st.selectbox("Ticker", watchlist, key="bt_ticker")
    with bt_col2:
        bt_horizon = st.selectbox("Horizon", ["next_day", "weekly"], key="bt_horizon",
                                   format_func=lambda x: "Next Day" if x == "next_day" else "Weekly")

    if st.button("Run Backtest", width="content"):
        progress_bar = st.progress(0, text=f"Backtesting {bt_ticker} ({bt_horizon})...")

        def update_progress(pct):
            progress_bar.progress(min(pct, 1.0),
                                  text=f"Backtesting {bt_ticker}... {int(pct*100)}%")

        bt = run_backtest(bt_ticker, bt_horizon, progress_callback=update_progress)
        progress_bar.empty()

        if "error" in bt:
            st.error(bt["error"])
        else:
            # Summary metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{bt['accuracy']}%")
            m2.metric("Predictions", bt["total_predictions"])
            m3.metric("Strategy Return", f"{bt['strategy_return']:+.1f}%")
            m4.metric("Buy & Hold Return", f"{bt['buyhold_return']:+.1f}%")

            st.caption(f"Period: {bt['date_range']}")

            if bt.get("high_confidence_count", 0) > 0:
                st.info(f"High-confidence predictions (≥60%): "
                        f"{bt['high_confidence_accuracy']}% accurate "
                        f"({bt['high_confidence_count']} predictions)")

            # Equity curve chart
            bt_df = pd.DataFrame(bt["results"])
            bt_df["date"] = pd.to_datetime(bt_df["date"])

            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(
                x=bt_df["date"], y=bt_df["strategy_equity"],
                name="Model Strategy",
                line=dict(color="#4CAF50", width=2),
            ))
            fig_eq.add_trace(go.Scatter(
                x=bt_df["date"], y=bt_df["buyhold_equity"],
                name="Buy & Hold",
                line=dict(color="#FF9800", width=2, dash="dash"),
            ))
            fig_eq.add_hline(y=1.0, line_dash="dot", line_color="gray",
                              annotation_text="Break Even")
            fig_eq.update_layout(
                title="Strategy vs Buy & Hold",
                height=400, yaxis_title="Growth of $1",
                template="plotly_dark", margin=dict(t=40, b=20),
            )
            st.plotly_chart(fig_eq, width="stretch")

            # Rolling accuracy chart
            if "rolling_accuracy" in bt_df.columns:
                fig_ra = go.Figure()
                fig_ra.add_trace(go.Scatter(
                    x=bt_df["date"], y=bt_df["rolling_accuracy"],
                    name="Rolling Accuracy",
                    line=dict(color="#2196F3", width=2),
                ))
                fig_ra.add_hline(y=50, line_dash="dash", line_color="gray",
                                  annotation_text="Random (50%)")
                fig_ra.update_layout(
                    title="Rolling Prediction Accuracy",
                    height=300, yaxis_title="Accuracy %",
                    template="plotly_dark", margin=dict(t=40, b=20),
                )
                st.plotly_chart(fig_ra, width="stretch")

            # Prediction detail table
            with st.expander("Full backtest results"):
                show_df = bt_df[["date", "close", "predicted", "actual",
                                  "confidence", "actual_pct", "correct"]].copy()
                show_df.columns = ["Date", "Price", "Predicted", "Actual",
                                    "Confidence %", "Actual Move %", "Correct"]
                show_df["Correct"] = show_df["Correct"].map({True: "✅", False: "❌"})
                st.dataframe(show_df, width="stretch", hide_index=True)


# ── TAB: Portfolio ───────────────────────────────────────────────────────────

with tab_portfolio:
    st.subheader("Portfolio Tracker")
    st.caption("Log your positions and track P&L against model signals.")

    # ── Add new position ─────────────────────────────────────────────────
    with st.expander("➕ Add Position", expanded=False):
        pc1, pc2, pc3, pc4 = st.columns(4)
        with pc1:
            pos_ticker = st.text_input("Ticker", placeholder="AAPL", key="pos_ticker")
        with pc2:
            pos_shares = st.number_input("Shares", min_value=0.01, value=1.0,
                                          step=1.0, key="pos_shares")
        with pc3:
            pos_price = st.number_input("Buy Price ($)", min_value=0.01,
                                         value=100.0, step=0.01, key="pos_price")
        with pc4:
            pos_date = st.date_input("Buy Date", key="pos_date")

        pos_notes = st.text_input("Notes (optional)", key="pos_notes")

        if st.button("Add Position"):
            if pos_ticker:
                try:
                    db.add_position(pos_ticker, pos_shares, pos_price,
                                     pos_date.strftime("%Y-%m-%d"), pos_notes)
                    st.success(f"Added {pos_shares} shares of {pos_ticker.upper()} @ ${pos_price}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.warning("Enter a ticker symbol")

    # ── Open positions ───────────────────────────────────────────────────
    open_positions = db.get_open_positions()

    if open_positions:
        st.subheader("Open Positions")

        # Fetch current prices for all open tickers
        open_tickers = list(set(p["ticker"] for p in open_positions))
        price_map = {}
        for t in open_tickers:
            try:
                pd_info = fetch_current_price(t)
                price_map[t] = pd_info["price"]
            except Exception:
                price_map[t] = None

        # Build positions table
        pos_rows = []
        total_cost = 0
        total_value = 0
        total_pnl = 0

        for p in open_positions:
            current = price_map.get(p["ticker"])
            cost = p["shares"] * p["buy_price"]
            total_cost += cost

            if current:
                value = p["shares"] * current
                pnl = value - cost
                pnl_pct = (pnl / cost) * 100
                total_value += value
                total_pnl += pnl
            else:
                value = None
                pnl = None
                pnl_pct = None

            pos_rows.append({
                "ID": p["id"],
                "Ticker": p["ticker"],
                "Shares": p["shares"],
                "Buy Price": f"${p['buy_price']:.2f}",
                "Current": f"${current:.2f}" if current else "—",
                "Cost": f"${cost:.2f}",
                "Value": f"${value:.2f}" if value else "—",
                "P&L": f"${pnl:+.2f}" if pnl is not None else "—",
                "P&L %": f"{pnl_pct:+.1f}%" if pnl_pct is not None else "—",
                "Date": p["buy_date"],
            })

        # Summary metrics
        sm1, sm2, sm3, sm4 = st.columns(4)
        sm1.metric("Total Cost", f"${total_cost:,.2f}")
        sm2.metric("Current Value", f"${total_value:,.2f}" if total_value else "—")
        sm3.metric("Total P&L", f"${total_pnl:+,.2f}" if total_pnl else "—")
        pnl_pct_total = (total_pnl / total_cost * 100) if total_cost > 0 else 0
        sm4.metric("Return", f"{pnl_pct_total:+.1f}%")

        # Save snapshot for history tracking (once per refresh)
        if total_value > 0:
            try:
                db.save_portfolio_snapshot(total_value, total_cost, total_pnl,
                                           round(pnl_pct_total, 2))
            except Exception:
                pass

        # Positions table
        pos_df = pd.DataFrame(pos_rows)
        st.dataframe(pos_df, width="stretch", hide_index=True)

        # Close position form
        with st.expander("📤 Close a Position"):
            close_id = st.selectbox(
                "Select position to close",
                [f"{p['ID']} — {p['Ticker']} ({p['Shares']} shares @ {p['Buy Price']})"
                 for p in pos_rows],
                key="close_select"
            )
            close_price = st.number_input("Sell Price ($)", min_value=0.01,
                                           value=100.0, step=0.01, key="close_price")
            if st.button("Close Position"):
                pid = int(close_id.split(" — ")[0])
                db.close_position(pid, close_price)
                st.success("Position closed!")
                st.rerun()

        # Delete position
        with st.expander("🗑️ Delete a Position"):
            del_id = st.selectbox(
                "Select position to delete",
                [f"{p['ID']} — {p['Ticker']} ({p['Shares']} shares @ {p['Buy Price']})"
                 for p in pos_rows],
                key="del_select"
            )
            if st.button("Delete Position"):
                pid = int(del_id.split(" — ")[0])
                db.delete_position(pid)
                st.success("Position deleted!")
                st.rerun()

        # ── Diversification analysis ─────────────────────────────────────
        st.markdown("---")
        st.subheader("Portfolio Risk & Diversification")

        try:
            div = analyze_diversification(open_positions)
            if "error" not in div:
                dc1, dc2, dc3 = st.columns(3)
                dc1.metric("Sectors", div["num_sectors"])
                dc2.metric("Positions", div["num_positions"])
                dc3.metric("Diversity Rating", div["diversity_rating"])

                # Sector breakdown
                if div["sector_breakdown"]:
                    sect_df = pd.DataFrame(div["sector_breakdown"])
                    sect_df.columns = ["Sector", "Value ($)", "Weight (%)"]
                    st.dataframe(sect_df, width="stretch", hide_index=True)

                # Warnings
                for w in div.get("warnings", []):
                    st.warning(w)

                if not div.get("warnings"):
                    st.success("Portfolio diversification looks healthy.")
        except Exception as e:
            st.warning(f"Could not analyze diversification: {e}")

    else:
        st.info("No open positions. Add one above to start tracking your portfolio.")

    # ── Closed positions ─────────────────────────────────────────────────
    closed = db.get_closed_positions()
    if closed:
        with st.expander(f"📋 Closed Positions ({len(closed)})"):
            closed_rows = []
            total_realized = 0
            for p in closed:
                cost = p["shares"] * p["buy_price"]
                revenue = p["shares"] * (p["sell_price"] or 0)
                pnl = revenue - cost
                total_realized += pnl
                closed_rows.append({
                    "Ticker": p["ticker"],
                    "Shares": p["shares"],
                    "Buy": f"${p['buy_price']:.2f}",
                    "Sell": f"${p['sell_price']:.2f}" if p["sell_price"] else "—",
                    "P&L": f"${pnl:+.2f}",
                    "P&L %": f"{(pnl/cost*100):+.1f}%" if cost > 0 else "—",
                    "Bought": p["buy_date"],
                    "Sold": p["sell_date"] or "—",
                })
            st.metric("Total Realized P&L", f"${total_realized:+,.2f}")
            st.dataframe(pd.DataFrame(closed_rows), width="stretch", hide_index=True)

    # ── Portfolio value over time ────────────────────────────────────────
    snapshots = db.get_portfolio_snapshots(last_n=500)
    if len(snapshots) > 2:
        with st.expander("📈 Portfolio Value History"):
            snap_df = pd.DataFrame(snapshots)
            snap_df["snapshot_at"] = pd.to_datetime(snap_df["snapshot_at"])
            snap_df = snap_df.sort_values("snapshot_at")

            fig_port = go.Figure()
            fig_port.add_trace(go.Scatter(
                x=snap_df["snapshot_at"], y=snap_df["total_value"],
                name="Portfolio Value", line=dict(color="#4CAF50", width=2),
            ))
            fig_port.add_trace(go.Scatter(
                x=snap_df["snapshot_at"], y=snap_df["total_cost"],
                name="Total Cost", line=dict(color="#FF9800", width=1, dash="dash"),
            ))
            fig_port.update_layout(
                height=350, yaxis_title="$",
                template="plotly_dark", margin=dict(t=20, b=20),
            )
            st.plotly_chart(fig_port, width="stretch")


# ── TAB: Features ────────────────────────────────────────────────────────────

with tab_features:
    st.subheader("Feature Importance Analysis")
    st.caption("See which factors drive the model's predictions most. "
               "This helps you understand what the model is actually learning.")

    fi_col1, fi_col2 = st.columns(2)
    with fi_col1:
        fi_ticker = st.selectbox("Ticker", watchlist, key="fi_ticker")
    with fi_col2:
        fi_horizon = st.selectbox("Horizon", ["next_day", "weekly"], key="fi_horizon",
                                   format_func=lambda x: "Next Day" if x == "next_day" else "Weekly")

    fi = get_feature_importance(fi_ticker, fi_horizon)

    if "error" in fi:
        st.warning(fi["error"])
    else:
        # Category breakdown
        st.subheader("What Drives Predictions")
        cat = fi["category_breakdown"]
        if cat:
            cat_cols = st.columns(len(cat))
            for i, (category, pct) in enumerate(cat.items()):
                cat_cols[i].metric(category, f"{pct}%")

        st.markdown("---")

        # Top features bar chart
        st.subheader(f"Top 20 Features ({fi['num_features']} total)")
        top = fi["top_features"]
        if top:
            feat_names = [f[0] for f in top]
            feat_scores = [f[1] for f in top]

            fig_fi = go.Figure()
            fig_fi.add_trace(go.Bar(
                x=feat_scores,
                y=feat_names,
                orientation="h",
                marker_color="#4CAF50",
            ))
            fig_fi.update_layout(
                height=max(400, len(top) * 25),
                yaxis=dict(autorange="reversed"),
                xaxis_title="Importance Score",
                template="plotly_dark",
                margin=dict(t=10, b=20, l=200),
            )
            st.plotly_chart(fig_fi, width="stretch")

        # Full feature list
        with st.expander(f"All {fi['num_features']} features"):
            all_feats = fi["all_features"]
            feat_df = pd.DataFrame(all_feats, columns=["Feature", "Importance"])
            feat_df["Importance"] = feat_df["Importance"].apply(lambda x: f"{x:.6f}")
            feat_df.index = range(1, len(feat_df) + 1)
            feat_df.index.name = "Rank"
            st.dataframe(feat_df, width="stretch")


# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("⚠️ This tool is for educational and personal analysis only. "
           "Not financial advice. Past predictions do not guarantee future results.")
