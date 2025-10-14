import numpy as np
import yfinance as yf

def make_prediction(stock):
    # Load the forecasted 7-day prices
    try:
        forecast = np.load(f"models/{stock}_forecast.npy").tolist()
    except FileNotFoundError:
        forecast = [0.0 for _ in range(7)]  # fallback if forecast file is missing

    # Fetch latest stock price
    stock_info = yf.Ticker(stock)
    hist = stock_info.history(period="1d")
    if not hist.empty:
        latest_price = round(hist["Close"].iloc[-1], 2)
    else:
        latest_price = 0.0  # fallback if no recent data

    return latest_price, forecast
