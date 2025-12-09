"""
Daily partial training of meta-models using:
- main model predictions (ARIMA, Prophet, LSTM)
- Yahoo Finance BTC-USD close price
- sentiment score from top-3 Yahoo Finance news

Run this once per day AFTER daily close is finalized
(~05:45–06:00 IST, i.e. ~00:15–00:30 UTC).

Feature design (per day D, UTC):
    close_D     = real close price of BTC at D (from Yahoo Finance)
    sentiment_D = mean VADER compound of top-3 Yahoo BTC-USD news for D

    arima_pred_D   = ARIMA model's one-step-ahead prediction for D+1
    prophet_pred_D = Prophet model's one-step-ahead prediction for D+1
    lstm_pred_D    = LSTM model's one-step-ahead price prediction for D+1
                     (using log_return target and reconstructing price)

Meta-features:
    X_meta_arima   = [arima_pred_D,   close_D, sentiment_D]
    X_meta_prophet = [prophet_pred_D, close_D, sentiment_D]
    X_meta_lstm    = [lstm_pred_D,    close_D, sentiment_D]
    y              = close_D

On the very first run:
    - these 3-feature vectors define the fixed input dimension of your
      (fresh) SGDRegressor meta models.
"""

import os
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import joblib
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from tensorflow.keras.models import load_model
from sklearn.linear_model import SGDRegressor

# --------------------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------------------

# Default: use ../models relative to this file (portable: local + GitHub + Streamlit)
DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_DIR = Path(os.getenv("MODEL_DIR", str(DEFAULT_MODEL_DIR))).expanduser()

# Base + meta model paths
ARIMA_PATH         = MODEL_DIR / "arima_model.pkl"
META_ARIMA_PATH    = MODEL_DIR / "meta_arima_1d.pkl"

PROPHET_PATH       = MODEL_DIR / "prophet_model.pkl"
META_PROPHET_PATH  = MODEL_DIR / "meta_prophet_1d.pkl"

LSTM_MODEL_PATH    = MODEL_DIR / "btc_lstm_logreturn_multifeature.h5"
LSTM_SCALER_PATH   = MODEL_DIR / "btc_feature_scaler.pkl"
LSTM_FEATURES_PATH = MODEL_DIR / "btc_feature_columns.pkl"
META_LSTM_PATH     = MODEL_DIR / "meta_lstm_1d.pkl"

# Optional: log file to track each day’s training row
META_LOG_PATH      = MODEL_DIR / "meta_training_log.csv"

TICKER = "BTC-USD"

# --------------------------------------------------------------------------------------
# DATE HELPERS
# --------------------------------------------------------------------------------------

def utc_yesterday() -> dt.date:
    """Return yesterday's date in UTC."""
    return dt.datetime.utcnow().date() - dt.timedelta(days=1)

# --------------------------------------------------------------------------------------
# YAHOO FINANCE: PRICE
# --------------------------------------------------------------------------------------

def fetch_btc_history() -> pd.DataFrame:
    df = yf.download(
        TICKER,
        period="max",
        interval="1d",
        auto_adjust=False,
        progress=False,
    )
    if df.empty:
        raise RuntimeError("yfinance returned empty BTC-USD history.")

    # Ensure datetime index (UTC-naive)
    df.index = pd.to_datetime(df.index).tz_localize(None)

    # ✅ FIXED: flatten MultiIndex correctly
    if isinstance(df.columns, pd.MultiIndex):
        flat_cols = []
        for col in df.columns:
            if isinstance(col, tuple):
                # take the FIRST non-empty element = 'Adj Close', 'Close', 'High', ...
                name = next((x for x in col if x not in (None, "")), col[0])
                flat_cols.append(str(name))
            else:
                flat_cols.append(str(col))
        df.columns = flat_cols
    else:
        df.columns = [str(c) for c in df.columns]

    return df


def get_close_for_date(df: pd.DataFrame, target_date: dt.date) -> float:
    """Get the close price for a specific UTC date (robust to column naming)."""
    day_rows = df.loc[df.index.date == target_date]
    if day_rows.empty:
        raise ValueError(f"No BTC data for date {target_date}")

    cols = list(day_rows.columns)

    # 1) Prefer exact 'Close'
    if "Close" in day_rows.columns:
        close_col = "Close"
    else:
        # 2) Fallback: any column that looks like a close price
        close_candidates = [c for c in cols if "close" in str(c).lower()]
        if not close_candidates:
            raise RuntimeError(
                f"No 'Close'-like column found for date {target_date}. "
                f"Columns: {cols}"
            )
        close_col = close_candidates[0]
        print(
            f"[WARN] 'Close' not found, using '{close_col}' as close column "
            f"for date {target_date}"
        )

    return float(day_rows[close_col].iloc[0])


# --------------------------------------------------------------------------------------
# YAHOO FINANCE: NEWS → SENTIMENT
# --------------------------------------------------------------------------------------

def fetch_yahoo_news_sentiment(target_date: dt.date) -> float:
    """
    Use yfinance's news for BTC-USD, filter to target_date UTC,
    take last 3 articles, compute mean VADER compound score.

    If no news → sentiment = 0.0 (neutral).
    """
    try:
        ticker = yf.Ticker(TICKER)
        raw_news = ticker.news or []
    except Exception as e:
        print(f"[WARN] Failed to fetch Yahoo news: {e}. Using sentiment=0.0")
        return 0.0

    start_dt = dt.datetime.combine(target_date, dt.time.min)
    end_dt   = dt.datetime.combine(target_date + dt.timedelta(days=1), dt.time.min)

    selected = []
    for item in raw_news:
        ts = item.get("providerPublishTime")
        if ts is None:
            continue
        publish_dt = dt.datetime.utcfromtimestamp(ts)
        if start_dt <= publish_dt < end_dt:
            selected.append(item)

    selected = sorted(selected, key=lambda x: x.get("providerPublishTime", 0))[-3:]

    if not selected:
        print(f"[WARN] No Yahoo BTC news for {target_date}, using sentiment=0.0")
        return 0.0

    analyzer = SentimentIntensityAnalyzer()
    scores = []
    for item in selected:
        title = item.get("title", "")
        summary = item.get("summary", "")
        text = f"{title}. {summary}"
        compound = analyzer.polarity_scores(text)["compound"]
        scores.append(compound)

    sentiment = float(np.mean(scores))
    print(f"[INFO] Sentiment for {target_date}: {sentiment:.4f} (from {len(scores)} articles)")
    return sentiment

# --------------------------------------------------------------------------------------
# LOAD MAIN MODELS
# --------------------------------------------------------------------------------------

def load_main_models():
    """Load ARIMA, Prophet, LSTM and LSTM aux files."""
    if not ARIMA_PATH.exists():
        raise FileNotFoundError(f"Missing ARIMA model at {ARIMA_PATH}")
    if not PROPHET_PATH.exists():
        raise FileNotFoundError(f"Missing Prophet model at {PROPHET_PATH}")
    if not LSTM_MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing LSTM model at {LSTM_MODEL_PATH}")
    if not LSTM_SCALER_PATH.exists():
        raise FileNotFoundError(f"Missing LSTM scaler at {LSTM_SCALER_PATH}")
    if not LSTM_FEATURES_PATH.exists():
        raise FileNotFoundError(f"Missing LSTM feature columns at {LSTM_FEATURES_PATH}")

    arima_model = joblib.load(ARIMA_PATH)
    prophet_model = joblib.load(PROPHET_PATH)

    lstm_model = load_model(LSTM_MODEL_PATH, compile=False)
    lstm_scaler = joblib.load(LSTM_SCALER_PATH)
    lstm_feature_cols = joblib.load(LSTM_FEATURES_PATH)
    return arima_model, prophet_model, lstm_model, lstm_scaler, lstm_feature_cols

# --------------------------------------------------------------------------------------
# LSTM FEATURE ENGINEERING
# --------------------------------------------------------------------------------------

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Standard RSI(14) implementation."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.fillna(50.0)
    return rsi


def prepare_lstm_feature_dataframe(price_df: pd.DataFrame, feature_cols) -> pd.DataFrame:
    """
    Match LSTM notebook:
        raw: Open, High, Low, Close, Volume
        engineered:
            log_return  = log(Close_t / Close_{t-1})
            ma7         = 7-day moving average of Close
            ma21        = 21-day moving average of Close
            volatility_7= 7-day rolling std of log_return
            rsi14       = 14-day RSI

        Total predictors = 10:
            [Open, High, Low, Close, Volume,
             log_return, ma7, ma21, volatility_7, rsi14]
    """
    df = price_df.copy().sort_index()

    required_base = ["Open", "High", "Low", "Close", "Volume"]
    for col in required_base:
        if col not in df.columns:
            raise ValueError(f"Price dataframe missing required column '{col}'")

    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["log_return"] = df["log_return"].fillna(0.0)

    df["ma7"] = df["Close"].rolling(window=7, min_periods=1).mean()
    df["ma21"] = df["Close"].rolling(window=21, min_periods=1).mean()

    df["volatility_7"] = df["log_return"].rolling(window=7, min_periods=1).std()
    df["volatility_7"] = df["volatility_7"].fillna(0.0)

    df["rsi14"] = compute_rsi(df["Close"], period=14)

    for col in feature_cols:
        if col not in df.columns:
            raise ValueError(
                f"Required LSTM feature column '{col}' missing in prepared df. "
                f"Update prepare_lstm_feature_dataframe() or check btc_feature_columns.pkl."
            )

    return df


def make_lstm_sequence(
    df_features: pd.DataFrame,
    feature_cols,
    scaler,
    window_size: int = 60,
) -> np.ndarray:
    """
    Build the last 'window_size' sequence for LSTM (shape: (1, window_size, n_features)).
    """
    if len(df_features) < window_size:
        raise ValueError(
            f"Not enough rows ({len(df_features)}) to build LSTM window of size {window_size}"
        )

    window = df_features[feature_cols].iloc[-window_size:].values
    window_scaled = scaler.transform(window)
    X = window_scaled.reshape(1, window_size, len(feature_cols))
    return X

# --------------------------------------------------------------------------------------
# LSTM PARTIAL TRAINING
# --------------------------------------------------------------------------------------

def partial_train_lstm_on_latest_day(
    lstm_model,
    price_df: pd.DataFrame,
    lstm_feature_cols,
    lstm_scaler,
    target_date: dt.date,
    window_size: int = 60,
    epochs: int = 1,
):
    """
    Incrementally train the LSTM on the NEWEST labeled data point (target_date).

    - Uses last `window_size` days BEFORE target_date as input sequence.
    - Uses log_return at target_date as the target (y).
    """
    df_features = prepare_lstm_feature_dataframe(price_df, lstm_feature_cols)
    df_features = df_features.loc[df_features.index.date <= target_date]

    if len(df_features) < window_size + 1:
        print(
            f"[WARN] Not enough data ({len(df_features)}) to partial-train LSTM "
            f"for date {target_date} with window_size={window_size}. Skipping."
        )
        return lstm_model

    df_hist = df_features.copy()
    tail = df_hist.iloc[-(window_size + 1):]

    X_window = tail[lstm_feature_cols].iloc[:-1].values
    y_log_return = tail["log_return"].iloc[-1]

    X_window_scaled = lstm_scaler.transform(X_window)
    X_train = X_window_scaled.reshape(1, window_size, len(lstm_feature_cols))
    y_train = np.array([y_log_return], dtype=float)

    print(
        f"[INFO] Partial training LSTM for target_date={target_date} "
        f"(one sample, epochs={epochs})"
    )

    lstm_model.fit(X_train, y_train, epochs=epochs, batch_size=1, verbose=0)
    lstm_model.save(LSTM_MODEL_PATH)
    print(f"[INFO] Saved updated LSTM model to {LSTM_MODEL_PATH}")

    return lstm_model

# --------------------------------------------------------------------------------------
# PROPHET FEATURE ENGINEERING
# --------------------------------------------------------------------------------------

def prepare_prophet_dataframe(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Recreate the exact Prophet training dataframe:

        ds, y, returns, log_returns, volatility, volume_norm, ma7
    """
    df_raw = price_df.copy()
    df_raw.index = pd.to_datetime(df_raw.index).tz_localize(None)

    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df_raw.columns)
    if missing:
        raise ValueError(
            f"Price dataframe missing columns: {missing}. Got: {df_raw.columns.tolist()}"
        )

    close = df_raw["Close"].astype(float)
    high = df_raw["High"].astype(float)
    low = df_raw["Low"].astype(float)
    vol = df_raw["Volume"].astype(float)

    # Base Prophet-style dataframe
    df = pd.DataFrame(
        {
            "ds": df_raw.index,      # datetime
            "y": close.values,       # close price
        }
    )

    # Engineered features (matching the notebook)
    df["returns"] = df["y"].pct_change()
    df["log_returns"] = np.log(df["y"]).diff()
    df["volatility"] = (high - low).values

    vol_mean = float(vol.mean())
    vol_std = float(vol.std(ddof=0)) or 1.0
    df["volume_norm"] = (vol - vol_mean) / vol_std
    df["volume_norm"] = df["volume_norm"].fillna(0.0)

    df["ma7"] = df["y"].rolling(window=7).mean()

    # Handle NaNs
    df["returns"] = df["returns"].fillna(0.0)
    df["log_returns"] = df["log_returns"].fillna(0.0)

    # Drop early rows where ma7 is NaN and reset index
    df = df.dropna(subset=["ma7"]).reset_index(drop=True)

    # Final sanity check
    if "ds" not in df.columns:
        raise RuntimeError(
            f"prepare_prophet_dataframe(): 'ds' column missing. Columns: {df.columns.tolist()}"
        )

    return df

# --------------------------------------------------------------------------------------
# MAIN MODEL PREDICTIONS
# --------------------------------------------------------------------------------------
def predict_with_arima(arima_model) -> float:
    """Next-day prediction from ARIMA (auto_arima-style)."""
    forecast = arima_model.predict(n_periods=1)
    return float(forecast[0])



def predict_with_prophet(
    prophet_model,
    prophet_df: pd.DataFrame,
    target_date: dt.date,
) -> float:
    """
    Prophet predicts next day's 'yhat' with extra regressors.

    prophet_df: output of prepare_prophet_dataframe(price_df)
    filtered up to target_date.
    """
    df_hist = prophet_df[prophet_df["ds"].dt.date <= target_date].copy()
    if df_hist.empty:
        raise ValueError("No Prophet data up to target_date")

    last_row = df_hist.iloc[-1]

    future_ds = last_row["ds"] + dt.timedelta(days=1)
    future = pd.DataFrame(
        {
            "ds": [future_ds],
            "returns": [last_row["returns"]],
            "log_returns": [last_row["log_returns"]],
            "volatility": [last_row["volatility"]],
            "volume_norm": [last_row["volume_norm"]],
            "ma7": [last_row["ma7"]],
        }
    )

    forecast = prophet_model.predict(future)
    return float(forecast["yhat"].iloc[0])


def predict_with_lstm(
    lstm_model,
    price_df: pd.DataFrame,
    lstm_feature_cols,
    lstm_scaler,
    target_date: dt.date,
    window_size: int = 60,
) -> float:
    """
    LSTM one-step-ahead price prediction:

    - Model predicts next-day log_return.
    - Reconstruct price_next = last_price * exp(predicted_log_return)
    """
    df_features = prepare_lstm_feature_dataframe(price_df, lstm_feature_cols)
    df_features = df_features.loc[df_features.index.date <= target_date]

    X_last = make_lstm_sequence(
        df_features,
        lstm_feature_cols,
        lstm_scaler,
        window_size=window_size,
    )
    pred_log_return = float(lstm_model.predict(X_last, verbose=0)[0][0])

    last_price = float(df_features["Close"].iloc[-1])
    next_price = last_price * np.exp(pred_log_return)
    return next_price

# --------------------------------------------------------------------------------------
# META MODELS: LOAD / INIT / SAVE
# --------------------------------------------------------------------------------------

def init_fresh_meta_model() -> SGDRegressor:
    """Fresh meta SGDRegressor (matches your meta-notebook setup)."""
    return SGDRegressor(
        loss="squared_error",
        learning_rate="invscaling",
        eta0=0.001,
        max_iter=1,
        warm_start=True,
        random_state=42,
    )


def load_or_init_meta_model(path: Path) -> SGDRegressor:
    """Load existing meta model if path exists, otherwise create + save a fresh one."""
    if path.exists():
        return joblib.load(path)
    model = init_fresh_meta_model()
    joblib.dump(model, path)
    return model


def load_meta_models():
    meta_arima   = load_or_init_meta_model(META_ARIMA_PATH)
    meta_prophet = load_or_init_meta_model(META_PROPHET_PATH)
    meta_lstm    = load_or_init_meta_model(META_LSTM_PATH)
    return meta_arima, meta_prophet, meta_lstm


def save_meta_models(meta_arima, meta_prophet, meta_lstm):
    joblib.dump(meta_arima, META_ARIMA_PATH)
    joblib.dump(meta_prophet, META_PROPHET_PATH)
    joblib.dump(meta_lstm, META_LSTM_PATH)

# --------------------------------------------------------------------------------------
# MAIN UPDATE LOGIC (ONE DAY)
# --------------------------------------------------------------------------------------

def update_meta_models_for_date(target_date: dt.date):
    """
    Full pipeline for a single UTC date D:

    1. Fetch full BTC history
    2. Compute close_D
    3. Compute sentiment_D from Yahoo news
    4. Prepare Prophet dataframe
    5. Load base models
    6. Partial-train LSTM on D
    7. Get base predictions for D+1
    8. Update meta models with one row
    9. Log the row
    """
    print(f"[INFO] Updating meta models for date {target_date}")

    price_df = fetch_btc_history()
    close_D = get_close_for_date(price_df, target_date)
    sentiment_D = fetch_yahoo_news_sentiment(target_date)
    prophet_df = prepare_prophet_dataframe(price_df)

    arima_model, prophet_model, lstm_model, lstm_scaler, lstm_feature_cols = load_main_models()

    lstm_model = partial_train_lstm_on_latest_day(
        lstm_model=lstm_model,
        price_df=price_df,
        lstm_feature_cols=lstm_feature_cols,
        lstm_scaler=lstm_scaler,
        target_date=target_date,
        window_size=60,
        epochs=1,
    )

    arima_pred = predict_with_arima(arima_model)
    prophet_pred = predict_with_prophet(prophet_model, prophet_df, target_date)
    lstm_pred = predict_with_lstm(
        lstm_model,
        price_df,
        lstm_feature_cols,
        lstm_scaler,
        target_date,
        window_size=60,
    )

    print(f"[INFO] Base model predictions for D+1 (from D={target_date}):")
    print(f"       ARIMA   pred: {arima_pred}")
    print(f"       Prophet pred: {prophet_pred}")
    print(f"       LSTM    pred: {lstm_pred}")
    print(f"[INFO] close_D = {close_D}, sentiment_D = {sentiment_D}")

    X_meta_arima   = np.array([[arima_pred,   close_D, sentiment_D]], dtype=float)
    X_meta_prophet = np.array([[prophet_pred, close_D, sentiment_D]], dtype=float)
    X_meta_lstm    = np.array([[lstm_pred,    close_D, sentiment_D]], dtype=float)
    y = np.array([close_D], dtype=float)

    meta_arima, meta_prophet, meta_lstm = load_meta_models()

    meta_arima.partial_fit(X_meta_arima, y)
    meta_prophet.partial_fit(X_meta_prophet, y)
    meta_lstm.partial_fit(X_meta_lstm, y)

    save_meta_models(meta_arima, meta_prophet, meta_lstm)

    log_row = {
        "ds": target_date.isoformat(),
        "close": close_D,
        "sentiment": sentiment_D,
        "arima_pred": arima_pred,
        "prophet_pred": prophet_pred,
        "lstm_pred": lstm_pred,
    }

    if META_LOG_PATH.exists():
        log_df = pd.read_csv(META_LOG_PATH)
        log_df = pd.concat([log_df, pd.DataFrame([log_row])], ignore_index=True)
    else:
        log_df = pd.DataFrame([log_row])

    log_df.to_csv(META_LOG_PATH, index=False)
    print(f"[INFO] Appended meta training row to {META_LOG_PATH}")

# --------------------------------------------------------------------------------------
# ENTRY POINT
# --------------------------------------------------------------------------------------

def main():
    target = utc_yesterday()
    print(f"[INFO] UTC now: {dt.datetime.utcnow()}")
    print(f"[INFO] Training meta models for UTC date: {target}")

    price_df = fetch_btc_history()
    if target not in set(price_df.index.date):
        print(f"[WARN] No BTC close available for {target} yet. Exiting without training.")
        return

    update_meta_models_for_date(target)


if __name__ == "__main__":
    main()



