from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from loadModels import load_models
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timezone
import requests

# --- Create FastAPI app ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Tuned seq_length map
seq_length_map = {
    "ethereum": 30,
    "bitcoin": 20,
    "xrp": 12
}

COIN_SYMBOL_MAP = {
    "bitcoin": "BTCUSDT",
    "ethereum": "ETHUSDT",
    "xrp": "XRPUSDT"
}

# fetch historical data from binance API

def fetch_binance_data(symbol: str, interval: str = "1d", limit: int = 1000):
    url = "https://api.binance.com/api/v3/klines"

    start_ts = int(datetime(2018, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)

    all_rows = []

    while True:
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
            "startTime": start_ts
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if not data:
            break

        all_rows.extend(data)

        if len(data) < limit:
            break

        start_ts = data[-1][6] + 1
    return all_rows

# --- Utility: build fresh dataframe ---

def load_df(coin: str) -> pd.DataFrame:
    coin_lower = coin.lower()

    if coin_lower not in COIN_SYMBOL_MAP:
        raise HTTPException(
            status_code = 400,
            detail=f"Unsupported coin '{coin}'. Supported: {list(COIN_SYMBOL_MAP.keys())}"
        )
    
    symbol = COIN_SYMBOL_MAP[coin_lower]

    try:
        raw = fetch_binance_data(symbol=symbol)
    except requests.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Binance API error: {str(e)}")
    
    records = []
    for k in raw:
        records.append({
            "Start": pd.to_datetime(k[0], unit="ms", utc=True).normalize(),
            "End": pd.to_datetime(k[6], unit="ms", utc=True).normalize(),
            "Open": float(k[1]),
            "High": float(k[2]),
            "Low": float(k[3]),
            "Close": float(k[4]),
            "Volume": float(k[5]),
            "Market Cap": float(k[7])
        })

    df = pd.DataFrame(records)

    df['timeClose'] = pd.to_datetime(df['Start']).dt.tz_localize(None)
    df['priceClose'] = df['Close']
    df = df.sort_values(by='timeClose')
    df = df[['timeClose', 'priceClose', 'Volume', 'Market Cap']].drop_duplicates().dropna()
    df.set_index('timeClose', inplace=True)
    df['SMA_10'] = df['priceClose'].rolling(10).mean()
    df['EMA_10'] = df['priceClose'].ewm(span=10).mean()
    df['priceChange'] = df['priceClose'].pct_change()
    df = df[['priceClose', 'SMA_10', 'EMA_10', 'priceChange']].dropna()
    return df

# --- Prediction helpers ---
def prepare_residuals(df, sarimax_fit, scaler):
    exog = df[['SMA_10', 'EMA_10', 'priceChange']]
    endog = df['priceClose']
    fitted = sarimax_fit.get_prediction(exog=exog).predicted_mean
    residuals = endog - fitted
    residuals = residuals.dropna()
    residuals_scaled = scaler.transform(residuals.values.reshape(-1, 1))
    return residuals_scaled

@app.get("/predict/next-day/{coin}")
def predict_next_day(coin: str):
    sarimax_fit, lstm_model, xgb_model, scaler = load_models(coin)
    seq_length = seq_length_map[coin.lower()]
    df = load_df(coin)
    residuals_scaled = prepare_residuals(df, sarimax_fit, scaler)

    last_row = df.iloc[-1]
    next_exog = pd.DataFrame([{
        'SMA_10': last_row['SMA_10'],
        'EMA_10': last_row['EMA_10'],
        'priceChange': last_row['priceChange']
    }])

    arima_pred = sarimax_fit.get_forecast(steps=1, exog=next_exog).predicted_mean.iloc[0]

    lstm_seq = residuals_scaled[-seq_length:].reshape(1, seq_length, 1)
    lstm_resid = lstm_model.predict(lstm_seq, verbose=0)
    lstm_resid_inv = scaler.inverse_transform(lstm_resid)[0, 0]

    xgb_input = residuals_scaled[-30:].reshape(1, -1)
    xgb_resid = xgb_model.predict(xgb_input)
    xgb_resid_inv = scaler.inverse_transform(xgb_resid.reshape(-1, 1))[0, 0]

    final_price = arima_pred + (lstm_resid_inv + xgb_resid_inv) / 2
    return {"coin": coin, "next_day_price": round(final_price, 2)}

@app.get("/predict/next-7-days/{coin}")
def predict_next_7_days(coin: str):
    sarimax_fit, lstm_model, xgb_model, scaler = load_models(coin)
    seq_length = seq_length_map[coin.lower()]
    df = load_df(coin)
    residuals_scaled = prepare_residuals(df, sarimax_fit, scaler)

    future_preds = []
    temp_residuals = residuals_scaled.copy()
    last_price = df['priceClose'].iloc[-1]

    for _ in range(7):
        last_row = df.iloc[-1]
        next_exog = pd.DataFrame([{
            'SMA_10': last_row['SMA_10'],
            'EMA_10': last_row['EMA_10'],
            'priceChange': last_row['priceChange']
        }])

        arima_pred = sarimax_fit.get_forecast(steps=1, exog=next_exog).predicted_mean.iloc[0]

        lstm_seq = temp_residuals[-seq_length:].reshape(1, seq_length, 1)
        lstm_resid = lstm_model.predict(lstm_seq, verbose=0)
        lstm_resid_inv = scaler.inverse_transform(lstm_resid)[0, 0]

        xgb_input = temp_residuals[-30:].reshape(1, -1)
        xgb_resid = xgb_model.predict(xgb_input)
        xgb_resid_inv = scaler.inverse_transform(xgb_resid.reshape(-1, 1))[0, 0]

        hybrid_price = arima_pred + (lstm_resid_inv + xgb_resid_inv) / 2
        future_preds.append(round(hybrid_price, 2))

        simulated_resid = hybrid_price - last_price
        last_price = hybrid_price
        new_scaled = scaler.transform([[simulated_resid]])
        temp_residuals = np.append(temp_residuals, new_scaled)[-len(temp_residuals):]

    return {"coin": coin, "next_7_days": future_preds}

@app.get("/summary/{coin}")
def get_summary(coin: str, year: int = None, month: int = None):
    df = load_df(coin)
    df['Year'] = df.index.year.astype(int)
    df['Month'] = df.index.month.astype(int)
    df['Quarter'] = df.index.to_period('Q').astype(str)

    if year: df = df[df['Year'] == year]
    if month: df = df[df['Month'] == month]

    if df.empty:
        raise HTTPException(status_code=404, detail="No data found.")

    yearly = df.groupby('Year')['priceClose'].agg(['min', 'max', 'mean', 'last']).reset_index()
    quarterly = df.groupby('Quarter')['priceClose'].agg(['min', 'max', 'mean', 'last']).reset_index()
    monthly = df.groupby(df.index.to_period('M'))['priceClose'].agg(['min', 'max', 'mean', 'last']).reset_index()

    return {
        "coin": coin,
        "yearly": yearly.round(2).to_dict(orient='records'),
        "quarterly": quarterly.round(2).to_dict(orient='records'),
        "monthly": monthly.round(2).to_dict(orient='records')
    }

@app.get("/daily-changes/{coin}/{year}/{month}")
def daily_changes(coin: str, year: int, month: int):
    df = load_df(coin)
    filtered = df[(df.index.year == year) & (df.index.month == month)]
    if filtered.empty:
        raise HTTPException(status_code=404, detail="No data for that month.")

    grouped = filtered.resample('D').agg({'priceClose': ['first', 'last']})
    grouped.columns = ['open', 'close']
    grouped.dropna(inplace=True)

    results = []
    for idx, row in grouped.iterrows():
        results.append({
            "date": str(idx.date()),
            "open": round(row['open'], 2),
            "close": round(row['close'], 2)
        })
    return {"coin": coin, "daily_changes": results}

@app.get("/start-date/{coin}")
def start_date(coin: str):
    df = load_df(coin)
    return {"coin": coin, "start_date": str(df.index.min().date())}

@app.get("/end-date/{coin}")
def end_date(coin: str):
    df = load_df(coin)
    return {"coin": coin, "end_date": str(df.index.max().date())}

@app.get("/healthcheck")
def healthcheck():
    return {"status": "✅ FastAPI multi-coin server running fine!"}

@app.get("/version")
def version():
    return {
        "api": "Multi-Coin Crypto Hybrid Forecast",
        "version": "2.0.0",
        "models": {
            "SARIMAX": "SARIMAX",
            "LSTM": "BiLSTM + Attention",
            "XGBoost": "XGBRegressor"
        }
    }