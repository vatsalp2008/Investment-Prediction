import os
import numpy as np
import pandas as pd
import yfinance as yf
import json
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense
from datetime import datetime, timedelta
import joblib

STOCKS = {
    "AAPL": "Apple",
    "AMZN": "Amazon",
    "GC=F": "Gold",
    "GOOGL": "Google",
    "MSFT": "Microsoft",
    "NVDA": "Nvidia",
    "TSLA": "Tesla"
}

INPUT_DAYS = 90
PREDICT_DAYS = 7
MODELS_DIR = "models"
PREDICTIONS_DIR = "predictions"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)

def fetch_stock_data(symbol):
    end = datetime.now()
    start = end - timedelta(days=730)  # 2 years
    df = yf.download(symbol, start=start, end=end)
    return df['Close'].dropna()

def create_sequences(data, input_days):
    X, y = [], []
    for i in range(len(data) - input_days):
        X.append(data[i:i + input_days])
        y.append(data[i + input_days])
    return np.array(X), np.array(y)

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=False, input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def predict_future(model, last_sequence, days, scaler):
    predictions = []
    current_seq = last_sequence.copy()

    for _ in range(days):
        pred = model.predict(current_seq.reshape(1, INPUT_DAYS, 1), verbose=0)
        predictions.append(pred[0][0])
        current_seq = np.append(current_seq[1:], pred)

    predictions = np.array(predictions).reshape(-1, 1)
    return scaler.inverse_transform(predictions).flatten()

def train_and_predict():
    all_predictions = {}

    for symbol in STOCKS:
        print(f"Training model for {symbol}...")
        data = fetch_stock_data(symbol)
        if data.empty:
            print(f"[ERROR] No data fetched for {symbol}")
            continue

        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))

        X, y = create_sequences(scaled_data, INPUT_DAYS)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        model = build_lstm_model((INPUT_DAYS, 1))
        model.fit(X, y, epochs=20, batch_size=32, verbose=0)

        # Save model and scaler
        model.save(os.path.join(MODELS_DIR, f"{symbol}.h5"))
        joblib.dump(scaler, os.path.join(MODELS_DIR, f"{symbol}_scaler.pkl"))

        # Predict future 7 days
        last_seq = scaled_data[-INPUT_DAYS:].reshape(INPUT_DAYS)
        predicted_prices = predict_future(model, last_seq, PREDICT_DAYS, scaler)

        # Save individual forecast to .npy file
        np.save(os.path.join(MODELS_DIR, f"{symbol}_forecast.npy"), predicted_prices)

        # Also save in predictions.json
        all_predictions[symbol] = predicted_prices.tolist()
        print(f"{symbol} prediction complete.")

    # Save all predictions as JSON
    with open(os.path.join(PREDICTIONS_DIR, "predictions.json"), "w") as f:
        json.dump(all_predictions, f, indent=4)

    print("All predictions saved successfully.")

if __name__ == "__main__":
    train_and_predict()
