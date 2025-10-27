import yfinance as yf

def get_stock_data(symbol, period="6mo"):
    stock = yf.Ticker(symbol)
    hist = stock.history(period=period)

    if hist.empty:
        return None
    
    return hist[['Close']]
