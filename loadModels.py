import pickle
from tensorflow.keras.models import load_model
import xgboost as xgb

def load_models():
    # Load SARIMAX
    with open("models/sarimax_model.pkl", "rb") as f:
        arima_fit = pickle.load(f)

    # Load LSTM model
    lstm_model = load_model("models/lstm_model.keras")

    # Load XGBoost model
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model("models/xgb_model.json")

    # Load Scaler
    with open("models/residual_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    return arima_fit, lstm_model, xgb_model, scaler
