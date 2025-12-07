import sys
from pathlib import Path
import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import yfinance as yf
from sklearn.exceptions import NotFittedError
import requests
import feedparser

# -------------------------------------------------------------------
# Make backend code importable
# -------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = BASE_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(SCRIPTS_DIR))

from train_meta_daily import (   # type: ignore
    MODEL_DIR,
    fetch_yahoo_news_sentiment,
    prepare_lstm_feature_dataframe,
    prepare_prophet_dataframe,
    load_main_models,
    load_meta_models,
    make_lstm_sequence,
)

TICKER = "BTC-USD"

# -------------------------------------------------------------------
# Streamlit config
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Bitcoin Model Dashboard",
    page_icon="📈",
    layout="wide",
)

# -------------------------------------------------------------------
# Modern / glassmorphism CSS
# -------------------------------------------------------------------
st.markdown(
    """
<style>
body {
  background-color: #020617;
}

.block-container {
  padding-top: 2.2rem;
}

/* Glass cards */
.glass-card {
  background: rgba(15,23,42,0.9);
  border-radius: 1rem;
  border: 1px solid rgba(148,163,184,0.25);
  box-shadow: 0 18px 45px rgba(15,23,42,0.9);
  padding: 1rem 1.3rem;
}

/* News small text */
.news-meta {
  color:#9ca3af;
  font-size:0.8rem;
}

/* Sidebar tweaks */
[data-testid="stSidebar"] {
  background: radial-gradient(circle at top, #020617 0, #020617 40%, #020617 100%);
}
</style>
""",
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------
# Price data helpers
# -------------------------------------------------------------------

def fetch_btc_history_app() -> pd.DataFrame:
    df = yf.download(
        TICKER,
        period="max",
        interval="1d",
        auto_adjust=False,
        progress=False,
        group_by="ticker",
    )
    if df.empty:
        raise RuntimeError("yfinance returned empty BTC-USD history in app.")

    df.index = pd.to_datetime(df.index).tz_localize(None)

    if isinstance(df.columns, pd.MultiIndex):
        if TICKER in df.columns.levels[0]:
            df = df[TICKER]
        else:
            flat_cols = []
            for col in df.columns:
                if isinstance(col, tuple):
                    name = next((x for x in col[::-1] if x not in (None, "")), col[-1])
                    flat_cols.append(str(name))
                else:
                    flat_cols.append(str(col))
            df.columns = flat_cols
    else:
        df.columns = [str(c) for c in df.columns]

    return df


@st.cache_data(ttl=600, show_spinner=False)
def load_price_history():
    return fetch_btc_history_app()


@st.cache_resource(show_spinner=False)
def load_all_models():
    (
        arima_model,
        prophet_model,
        lstm_model,
        lstm_scaler,
        lstm_feature_cols,
    ) = load_main_models()
    meta_arima, meta_prophet, meta_lstm = load_meta_models()
    return (
        arima_model,
        prophet_model,
        lstm_model,
        lstm_scaler,
        lstm_feature_cols,
        meta_arima,
        meta_prophet,
        meta_lstm,
    )

# -------------------------------------------------------------------
# News helpers (Yahoo for sentiment, CoinGecko + RSS for UI)
# -------------------------------------------------------------------

def format_time_ago(epoch_seconds: int) -> str:
    publish_dt = dt.datetime.utcfromtimestamp(epoch_seconds)
    now = dt.datetime.utcnow()
    diff = now - publish_dt

    seconds = int(diff.total_seconds())
    if seconds < 60:
        return "just now"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes} min ago"
    hours = minutes // 60
    if hours < 24:
        return f"{hours} h ago"
    days = hours // 24
    if days < 7:
        return f"{days} day{'s' if days != 1 else ''} ago"
    weeks = days // 7
    if weeks < 4:
        return f"{weeks} week{'s' if weeks != 1 else ''} ago"
    months = days // 30
    if months < 12:
        return f"{months} month{'s' if months != 1 else ''} ago"
    years = days // 365
    return f"{years} year{'s' if years != 1 else ''} ago"


@st.cache_data(ttl=600, show_spinner=False)
def fetch_top_news(max_items: int = 3):
    try:
        ticker = yf.Ticker(TICKER)
        raw_news = getattr(ticker, "news", None) or []
    except Exception:
        return []

    if not isinstance(raw_news, (list, tuple)) or not raw_news:
        return []

    def _get_ts(item):
        ts = item.get("providerPublishTime") or item.get("pubDate") or 0
        if isinstance(ts, (int, float)) and ts > 1e12:
            ts = int(ts / 1000)
        return int(ts) if isinstance(ts, (int, float)) else 0

    sorted_news = sorted(raw_news, key=_get_ts, reverse=True)
    selected = sorted_news[:max_items]

    out = []
    for item in selected:
        ts = _get_ts(item)
        time_ago = format_time_ago(ts) if ts > 0 else ""

        title = (item.get("title") or "").strip()
        summary = (item.get("summary") or "").strip()
        link = (item.get("link") or "").strip()

        if not title:
            if summary and summary != ".":
                title = summary
            elif link:
                title = link.split("/")[-1] or "News article"
            else:
                title = "News article"

        publisher = (
            item.get("publisher")
            or item.get("provider")
            or "Unknown source"
        )

        out.append(
            {
                "title": title,
                "publisher": publisher,
                "link": link or "#",
                "time_ago": time_ago or "just now",
            }
        )
    return out


@st.cache_data(ttl=60, show_spinner=False)
def fetch_coingecko_market():
    url = "https://api.coingecko.com/api/v3/coins/bitcoin"
    params = {
        "localization": "false",
        "tickers": "false",
        "market_data": "true",
        "community_data": "false",
        "developer_data": "false",
        "sparkline": "false",
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return None

    m = data.get("market_data", {})
    try:
        return {
            "price": float(m["current_price"]["usd"]),
            "high_24h": float(m["high_24h"]["usd"]),
            "low_24h": float(m["low_24h"]["usd"]),
            "volume_24h": float(m["total_volume"]["usd"]),
            "market_cap": float(m["market_cap"]["usd"]),
            "change_24h_pct": float(m.get("price_change_percentage_24h", 0.0)),
        }
    except Exception:
        return None


@st.cache_data(ttl=600, show_spinner=False)
def fetch_bitcoin_news(max_items: int = 3):
    feeds = [
        "https://feeds.feedburner.com/CoinDesk",
        "https://cointelegraph.com/rss",
    ]

    entries = []
    for url in feeds:
        try:
            feed = feedparser.parse(url)
        except Exception:
            continue

        for e in feed.entries:
            title = e.get("title", "")
            summary = e.get("summary", "")
            text = (title + " " + summary).lower()

            if "bitcoin" not in text and "btc" not in text:
                continue

            published = e.get("published", "") or e.get("updated", "")
            link = e.get("link", "")

            entries.append(
                {
                    "title": title,
                    "summary": summary,
                    "link": link,
                    "published": published,
                }
            )

    entries.sort(key=lambda x: x.get("published", ""), reverse=True)
    return entries[:max_items]

# -------------------------------------------------------------------
# Forecasts (ARIMA / LSTM / Prophet + meta for 1-day)
# -------------------------------------------------------------------

def multi_step_forecasts(
    horizon_days: int,
    use_arima: bool,
    use_lstm: bool,
    use_prophet: bool,
):
    price_df = load_price_history()
    latest_ts = price_df.index.max()
    target_date = latest_ts.date()
    close_D = float(price_df["Close"].iloc[-1])

    sentiment_D = fetch_yahoo_news_sentiment(target_date)

    (
        arima_model,
        prophet_model,
        lstm_model,
        lstm_scaler,
        lstm_feature_cols,
        meta_arima,
        meta_prophet,
        meta_lstm,
    ) = load_all_models()

    prophet_df_full = prepare_prophet_dataframe(price_df)

    # ARIMA
    arima_future = None
    meta_arima_day1 = None
    if use_arima:
        raw_arima = arima_model.predict(n_periods=horizon_days)
        arima_future = np.asarray(raw_arima, dtype=float)

        if horizon_days == 1:
            X_meta = np.array([[arima_future[0], close_D, sentiment_D]], dtype=float)
            try:
                meta_arima_day1 = float(meta_arima.predict(X_meta)[0])
            except NotFittedError:
                meta_arima_day1 = None

    # Prophet
    prophet_future = None
    meta_prophet_day1 = None
    if use_prophet:
        df_hist = prophet_df_full[prophet_df_full["ds"].dt.date <= target_date].copy()
        last_row = df_hist.iloc[-1]

        future_ds = [last_row["ds"] + dt.timedelta(days=i) for i in range(1, horizon_days + 1)]
        future = pd.DataFrame({"ds": future_ds})
        for col in ["returns", "log_returns", "volatility", "volume_norm", "ma7"]:
            future[col] = last_row[col]
        future = future.fillna(0.0)

        forecast = prophet_model.predict(future)
        prophet_future = forecast["yhat"].values
        if horizon_days == 1:
            X_meta = np.array([[prophet_future[0], close_D, sentiment_D]], dtype=float)
            try:
                meta_prophet_day1 = float(meta_prophet.predict(X_meta)[0])
            except NotFittedError:
                meta_prophet_day1 = None

    # LSTM
    lstm_future = None
    meta_lstm_day1 = None
    if use_lstm:
        price_future = price_df.copy()
        lstm_future_vals = []
        current_date = latest_ts
        current_close = close_D

        for _ in range(1, horizon_days + 1):
            df_feat = prepare_lstm_feature_dataframe(price_future, lstm_feature_cols)
            df_feat = df_feat.sort_index()
            df_feat_cut = df_feat.loc[df_feat.index <= current_date]

            X_last = make_lstm_sequence(df_feat_cut, lstm_feature_cols, lstm_scaler, window_size=60)
            pred_log_return = float(lstm_model.predict(X_last, verbose=0)[0][0])
            next_close = current_close * np.exp(pred_log_return)

            lstm_future_vals.append(next_close)

            next_date = current_date + dt.timedelta(days=1)
            last_row_pf = price_future.iloc[-1].copy()
            last_row_pf["Close"] = next_close
            last_row_pf["Open"] = next_close
            last_row_pf["High"] = next_close
            last_row_pf["Low"] = next_close
            price_future.loc[next_date] = last_row_pf
            price_future = price_future.sort_index()

            current_date = next_date
            current_close = next_close

        lstm_future = np.array(lstm_future_vals, dtype=float)
        if horizon_days == 1:
            X_meta = np.array([[lstm_future[0], close_D, sentiment_D]], dtype=float)
            try:
                meta_lstm_day1 = float(meta_lstm.predict(X_meta)[0])
            except NotFittedError:
                meta_lstm_day1 = None

    return {
        "price_df": price_df,
        "target_date": target_date,
        "close_D": close_D,
        "sentiment_D": sentiment_D,
        "arima_future": arima_future,
        "prophet_future": prophet_future,
        "lstm_future": lstm_future,
        "meta_arima_day1": meta_arima_day1,
        "meta_prophet_day1": meta_prophet_day1,
        "meta_lstm_day1": meta_lstm_day1,
    }

# -------------------------------------------------------------------
# SIDEBAR: controls (hideable, like before)
# -------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Controls")

    st.markdown("**Models**")
    use_arima = st.checkbox("ARIMA", value=True)
    use_lstm = st.checkbox("LSTM", value=True)
    use_prophet = st.checkbox("Prophet", value=True)

    price_df_sidebar = load_price_history()
    latest_ts_sidebar = price_df_sidebar.index.max()
    latest_date = latest_ts_sidebar.date()

    st.markdown("**Forecast until**")
    end_date = st.date_input(
        "End date",
        value=latest_date + dt.timedelta(days=7),
        min_value=latest_date + dt.timedelta(days=1),
        max_value=latest_date + dt.timedelta(days=60),
    )

    horizon_days = (end_date - latest_date).days
    if horizon_days < 1:
        horizon_days = 1

    st.caption(
        f"Latest close: {latest_date.isoformat()}  \n"
        f"Horizon: {horizon_days} day(s)"
    )

# -------------------------------------------------------------------
# Run forecasts
# -------------------------------------------------------------------
with st.spinner("Fetching BTC data and running models..."):
    preds = multi_step_forecasts(
        horizon_days=horizon_days,
        use_arima=use_arima,
        use_lstm=use_lstm,
        use_prophet=use_prophet,
    )

price_df = preds["price_df"]
target_date = preds["target_date"]
close_D = preds["close_D"]
sentiment_D = preds["sentiment_D"]

last_row = price_df.iloc[-1]
today_high = float(last_row["High"])
today_low = float(last_row["Low"])
today_open = float(last_row["Open"])
today_vol = float(last_row["Volume"])

# -------------------------------------------------------------------
# Main layout: MIDDLE (wide) + RIGHT
# -------------------------------------------------------------------
mid_col, right_col = st.columns([3.6, 1.8], gap="large")

# -------------------------------------------------------------------
# MIDDLE: model price predictions + chart
# -------------------------------------------------------------------
with mid_col:
    # ---- Model predictions summary (top) ----
    st.markdown("#### Model Price Predictions")
    st.markdown('<div class="glass-card" style="margin-bottom:0.9rem;">', unsafe_allow_html=True)

    rows = []
    col_name = f"Price on {end_date.isoformat()}"

    if use_arima and preds["arima_future"] is not None:
        if horizon_days == 1 and preds["meta_arima_day1"] is not None:
            final_val = preds["meta_arima_day1"]
        else:
            final_val = preds["arima_future"][-1]
        rows.append({"Model": "ARIMA", col_name: final_val})

    if use_lstm and preds["lstm_future"] is not None:
        if horizon_days == 1 and preds["meta_lstm_day1"] is not None:
            final_val = preds["meta_lstm_day1"]
        else:
            final_val = preds["lstm_future"][-1]
        rows.append({"Model": "LSTM", col_name: final_val})

    if use_prophet and preds["prophet_future"] is not None:
        if horizon_days == 1 and preds["meta_prophet_day1"] is not None:
            final_val = preds["meta_prophet_day1"]
        else:
            final_val = preds["prophet_future"][-1]
        rows.append({"Model": "Prophet", col_name: final_val})

    if rows:
        df_preds = pd.DataFrame(rows)
        st.write(df_preds.style.format({col_name: "${:,.2f}"}))
    else:
        st.write("Select at least one model.")

    st.markdown("</div>", unsafe_allow_html=True)

    # ---- Price & Future Paths chart (bottom) ----
    st.markdown("#### Price & Future Paths")

    recent = price_df.tail(180).copy()
    recent_dates = recent.index

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=recent_dates,
            open=recent["Open"],
            high=recent["High"],
            low=recent["Low"],
            close=recent["Close"],
            name="Price",
            increasing_line=dict(width=1.2),
            decreasing_line=dict(width=1.2),
            showlegend=True,
        )
    )

    last_ts = price_df.index.max()
    last_close = float(price_df["Close"].iloc[-1])
    future_dates = [
        last_ts + dt.timedelta(days=i)
        for i in range(1, horizon_days + 1)
    ]

    all_future_vals = []

    # ARIMA
    if use_arima and preds["arima_future"] is not None:
        if horizon_days == 1 and preds["meta_arima_day1"] is not None:
            final_val = float(preds["meta_arima_day1"])
            name = "ARIMA (meta-corrected 1d)"
        else:
            final_val = float(preds["arima_future"][-1])
            name = "ARIMA (base)"

        if np.isfinite(final_val):
            x_plot = [last_ts, future_dates[-1]]
            y_plot = [last_close, final_val]
            all_future_vals.append(final_val)

            fig.add_trace(
                go.Scatter(
                    x=x_plot,
                    y=y_plot,
                    mode="lines",
                    name=name,
                    line=dict(dash="dash", width=2),
                )
            )

    # LSTM
    if use_lstm and preds["lstm_future"] is not None:
        y_vals = np.asarray(preds["lstm_future"], dtype=float)
        name = "LSTM (base)"
        if horizon_days == 1 and preds["meta_lstm_day1"] is not None:
            y_vals = np.array([preds["meta_lstm_day1"]], dtype=float)
            name = "LSTM (meta-corrected 1d)"

        x_plot = [last_ts] + future_dates[: len(y_vals)]
        y_plot = [last_close] + y_vals.tolist()
        all_future_vals.extend(y_vals.tolist())

        fig.add_trace(
            go.Scatter(
                x=x_plot,
                y=y_plot,
                mode="lines",
                name=name,
                line=dict(dash="dot", width=2),
            )
        )

    # Prophet
    if use_prophet and preds["prophet_future"] is not None:
        y_vals = np.asarray(preds["prophet_future"], dtype=float)
        name = "Prophet (base)"
        if horizon_days == 1 and preds["meta_prophet_day1"] is not None:
            y_vals = np.array([preds["meta_prophet_day1"]], dtype=float)
            name = "Prophet (meta-corrected 1d)"

        x_plot = [last_ts] + future_dates[: len(y_vals)]
        y_plot = [last_close] + y_vals.tolist()
        all_future_vals.extend(y_vals.tolist())

        fig.add_trace(
            go.Scatter(
                x=x_plot,
                y=y_plot,
                mode="lines",
                name=name,
                line=dict(dash="longdash", width=2),
            )
        )

    recent_min = recent["Low"].min()
    recent_max = recent["High"].max()
    if all_future_vals:
        y_min = min(recent_min, min(all_future_vals))
        y_max = max(recent_max, max(all_future_vals))
    else:
        y_min, y_max = recent_min, recent_max
    pad = (y_max - y_min) * 0.1 if y_max > y_min else 1.0

    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        height=430,
        showlegend=True,
        legend=dict(bgcolor="rgba(15,23,42,0.7)", bordercolor="rgba(148,163,184,0.3)"),
        plot_bgcolor="rgba(15,23,42,0.95)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            rangeslider=dict(visible=False),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(148,163,184,0.18)",
            range=[y_min - pad, y_max + pad],
        ),
        font=dict(color="#e5e7eb"),
    )

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------------------------
# RIGHT: snapshot (top) + news (bottom)
# -------------------------------------------------------------------
with right_col:
    st.markdown("#### Bitcoin Market Snapshot")
    market = fetch_coingecko_market()
    st.markdown('<div class="glass-card" style="margin-bottom:0.9rem;">', unsafe_allow_html=True)

    if market is None:
        st.write("Could not load data from CoinGecko.")
    else:
        c1, c2 = st.columns(2)

        with c1:
            st.metric(
                "Price (USD)",
                f"${market['price']:,.2f}",
                f"{market['change_24h_pct']:.2f}% (24h)",
            )
            st.metric(
                "24h High (USD)",
                f"${market['high_24h']:,.2f}",
            )
            st.metric(
                "Market Cap (USD)",
                f"${market['market_cap']:,.0f}",
            )

        with c2:
            st.metric(
                "24h Low (USD)",
                f"${market['low_24h']:,.2f}",
            )
            st.metric(
                "24h Volume (USD)",
                f"${market['volume_24h']:,.0f}",
            )

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("#### Top 3 Bitcoin News")
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    news_items = fetch_bitcoin_news(max_items=3)
    if not news_items:
        st.write("No Bitcoin news available right now.")
    else:
        for item in news_items:
            st.markdown(
                f"- [{item['title']}]({item['link']})  \n"
                f"  <span class='news-meta'>{item['published']}</span>",
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)
