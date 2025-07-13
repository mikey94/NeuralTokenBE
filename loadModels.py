import pickle
from tensorflow.keras.models import load_model
from attention import Attention
import xgboost as xgb

def load_models(coin_name: str):
    base_path = f"models/{coin_name}"
    # Load SARIMAX
    with open(f"{base_path}/sarimax_model.pkl", "rb") as f:
        arima_fit = pickle.load(f)

    # Load LSTM model
    lstm_model = load_model(f"{base_path}/lstm_model.keras", custom_objects={"Attention": Attention})

    # Load XGBoost model
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(f"{base_path}/xgb_model.json")

    # Load Scaler
    with open(f"{base_path}/residual_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    return arima_fit, lstm_model, xgb_model, scaler
