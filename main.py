from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
from loadModels import load_models
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow requests from all origins (you can restrict this later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with your domain(s) in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models once at startup
arima_fit, lstm_model, xgb_model, scaler = load_models()

# Load and preprocess latest data
df = pd.read_excel("ethereum-20241022000009342.xlsx")
df['timeClose'] = pd.to_datetime(df['timeClose'], unit='ms')
df = df.sort_values(by='timeClose')[['timeClose', 'priceClose']].drop_duplicates().dropna()

# Feature engineering
df['SMA_10'] = df['priceClose'].rolling(window=10).mean()
df['EMA_10'] = df['priceClose'].ewm(span=10).mean()
df['priceChange'] = df['priceClose'].pct_change()

# Keep only valid rows (all required inputs present)
df = df[['timeClose', 'priceClose', 'SMA_10', 'EMA_10', 'priceChange']].dropna()
df.set_index('timeClose', inplace=True)

seq_length = 30

# Prepare exog and endog
endog_all = df['priceClose']
exog_all = df[['SMA_10', 'EMA_10', 'priceChange']]

# Generate residuals using prediction on full data
fitted_all = arima_fit.get_prediction(exog=exog_all).predicted_mean
residuals = endog_all - fitted_all
residuals = residuals.dropna()

if len(residuals) < seq_length + 1:
    raise ValueError(f"❌ Not enough residuals: found {len(residuals)}, need at least {seq_length + 1}")

# Scale residuals
residuals_scaled = scaler.fit_transform(residuals.values.reshape(-1, 1))

# --- Prediction Logic ---

def predict_next_day():
    last_row = df.iloc[-1]
    next_exog = pd.DataFrame([{
        'SMA_10': last_row['SMA_10'],
        'EMA_10': last_row['EMA_10'],
        'priceChange': last_row['priceChange']
    }])
    arima_pred = arima_fit.get_forecast(steps=1, exog=next_exog).predicted_mean.iloc[0]

    lstm_seq = residuals_scaled[-seq_length:].reshape(1, seq_length, 1)
    lstm_resid = lstm_model.predict(lstm_seq, verbose=0)
    lstm_resid_inv = scaler.inverse_transform(lstm_resid)[0, 0]

    xgb_input = residuals_scaled[-30:].reshape(1, -1)
    xgb_resid = xgb_model.predict(xgb_input)
    xgb_resid_inv = scaler.inverse_transform(xgb_resid.reshape(-1, 1))[0, 0]

    return arima_pred + (lstm_resid_inv + xgb_resid_inv) / 2

def predict_next_n_days(n=7):
    future_preds = []
    temp_residuals = residuals_scaled.copy()
    last_price = df['priceClose'].iloc[-1]

    for _ in range(n):
        last_row = df.iloc[-1]
        next_exog = pd.DataFrame([{
            'SMA_10': last_row['SMA_10'],
            'EMA_10': last_row['EMA_10'],
            'priceChange': last_row['priceChange']
        }])
        arima_pred = arima_fit.get_forecast(steps=1, exog=next_exog).predicted_mean.iloc[0]

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

    return future_preds

# --- Get price data summary ---

def get_price_summary_by_period():
    """
    Returns min, max, mean, and last price grouped by Month, Quarter, and Year.
    """
    summary_df = df.reset_index()
    summary_df['Year'] = summary_df['timeClose'].dt.year
    summary_df['Month'] = summary_df['timeClose'].dt.to_period('M').astype(str)
    summary_df['Quarter'] = summary_df['timeClose'].dt.to_period('Q').astype(str)

    yearly = summary_df.groupby('Year')['priceClose'].agg(['min', 'max', 'mean', 'last']).reset_index()
    quarterly = summary_df.groupby('Quarter')['priceClose'].agg(['min', 'max', 'mean', 'last']).reset_index()
    monthly = summary_df.groupby('Month')['priceClose'].agg(['min', 'max', 'mean', 'last']).reset_index()

    return {
        "yearly": yearly.round(2).to_dict(orient='records'),
        "quarterly": quarterly.round(2).to_dict(orient='records'),
        "monthly": monthly.round(2).to_dict(orient='records'),
    }

def get_daily_changes_for_month(year: int, month: int):
    """
    Returns daily price changes for a given month and year.
    """
    # Filter the data by year and month
    filtered = df[(df.index.year == year) & (df.index.month == month)]

    if filtered.empty:
        return {"message": "No data found for the selected month."}

    # Prepare daily data: date, open, close, change
    result = []
    grouped = filtered.resample('D').agg({'priceClose': ['first', 'last']})
    grouped.columns = ['open', 'close']
    grouped.dropna(inplace=True)

    for index, row in grouped.iterrows():
        result.append({
            "date": str(index.date()),
            "open": round(row['open'], 2),
            "close": round(row['close'], 2),
        })

    return result

def get_start_day():
    start_date = df.index.min().date()
    return {
        "date": str(start_date)
    }

def get_end_day():
    end_date = df.index.max().date()
    return {
        "date": str(end_date)
    }
# --- FastAPI Endpoints ---

@app.get("/predict/next-day")
def get_next_day():
    return {"next_day_price": round(predict_next_day(), 2)}

@app.get("/predict/next-7-days")
def get_next_7_days():
    return {"next_7_days": predict_next_n_days(7)}

@app.get("/summary")
def get_price_summary():
    return get_price_summary_by_period()

@app.get("/daily-changes/{year}/{month}")
def get_daily_changes(year: int, month: int):
    return get_daily_changes_for_month(year, month)

@app.get("/start-date")
def get_start_date():
    return get_start_day()

@app.get("/end-date")
def get_end_date():
    return get_end_day()

@app.get("/healthcheck")
def healthcheck():
    return {"status": "✅ Server is up and running!"}

@app.get("/version")
def version():
    return {
        "app": "Ethereum Hybrid Forecast API",
        "version": "1.0.0",
        "models": {
            "ARIMA": "SARIMAX (refit on startup)",
            "LSTM": "Bidirectional LSTM (saved .keras)",
            "XGBoost": "XGBRegressor (v3.0.0)"
        }
    }
