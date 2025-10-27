import json
import yfinance as yf
from datetime import datetime, timedelta

STOCKS = {
    "AAPL": "Apple",
    "AMZN": "Amazon",
    "GC=F": "Gold",
    "GOOGL": "Google",
    "MSFT": "Microsoft",
    "NVDA": "Nvidia",
    "TSLA": "Tesla"
}

# Load existing metrics
with open("accuracy_report.json", "r") as f:
    metrics = json.load(f)

accuracy_with_percentage = {}

for symbol in STOCKS:
    try:
        # Fetch last 7 actual closing prices
        end = datetime.now()
        start = end - timedelta(days=14)  # buffer for 7 trading days
        df = yf.download(symbol, start=start, end=end)
        actuals = df["Close"].dropna()[-7:]
        avg_actual = float(actuals.mean())

        rmse = metrics[symbol]["RMSE"]
        accuracy_percent = 100 - ((rmse / avg_actual) * 100)

        accuracy_with_percentage[symbol] = {
            "MSE": round(metrics[symbol]["MSE"], 4),
            "MAE": round(metrics[symbol]["MAE"], 4),
            "RMSE": round(rmse, 4),
            "Avg_Actual": round(avg_actual, 2),
            "Accuracy (%)": round(accuracy_percent, 2)
        }

    except Exception as e:
        accuracy_with_percentage[symbol] = {
            "error": str(e)
        }

# Save updated report
with open("accuracy_report_with_percentage.json", "w") as f:
    json.dump(accuracy_with_percentage, f, indent=4)

print("Updated accuracy with percentage saved to accuracy_report_with_percentage.json")
