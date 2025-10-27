from flask import Flask, render_template, jsonify
from utils.data_fetcher import get_stock_data
from utils.predictor import make_prediction
import yfinance as yf
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Home Page
@app.route("/")
def home():
    return render_template("index.html")

# Market Page (Auto-Updating)
@app.route("/market")
def market():
    stocks = ["AAPL", "AMZN", "GC=F", "GOOGL", "MSFT", "NVDA", "TSLA"]
    stock_data = {}

    for stock in stocks:
        stock_info = yf.Ticker(stock)
        hist = stock_info.history(period="6mo")

        if not hist.empty:
            stock_data[stock] = {
                "latest_price": round(hist["Close"].iloc[-1], 2),
                "history": hist["Close"].tail(30).to_dict()
            }

    return render_template("market.html", stock_data=stock_data)

# API for auto-updating stock prices
@app.route("/market-data")
def market_data():
    stocks = ["AAPL", "AMZN", "GC=F", "GOOGL", "MSFT", "NVDA", "TSLA"]
    stock_data = {}

    for stock in stocks:
        stock_info = yf.Ticker(stock)
        hist = stock_info.history(period="6mo")

        if not hist.empty:
            stock_data[stock] = {
                "latest_price": round(hist["Close"].iloc[-1], 2)
            }

    return jsonify(stock_data)

# Prediction Stock Selection Page
@app.route("/predict")
def predict():
    return render_template("predict.html")

# Prediction Result Page
@app.route("/predict/<stock>")
def predict_stock(stock):
    latest_price, forecast = make_prediction(stock)

    # Plot graph
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, 8), forecast, marker='o', linestyle='-', color='#66b3a1', label="Predicted Price")
    plt.xlabel("Days Ahead")
    plt.ylabel("Price (USD)")
    plt.title(f"{stock} Stock Price Prediction")
    plt.legend()
    plt.grid(True)

    # Convert plot to base64
    img = io.BytesIO()
    plt.savefig(img, format="png", bbox_inches="tight")
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return render_template("predict_stock.html", stock_name=stock, latest_price=latest_price, plot_url=plot_url)

if __name__ == "__main__":
    app.run(debug=True)
