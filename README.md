# Investment Predictor 📈💰

A sophisticated Flask-based investment prediction platform that provides real-time market data and AI-powered 7-day price forecasts for both **stocks** and **commodities** (Gold) using LSTM neural networks.

## Overview

Investment Predictor is a comprehensive financial analysis tool that helps investors make informed decisions by combining real-time market monitoring with machine learning predictions. The platform covers major tech stocks and gold futures, making it suitable for diverse investment portfolios.

## Investment Assets Covered

### 📊 **Stocks**
- **AAPL** - Apple Inc.
- **AMZN** - Amazon.com Inc.
- **GOOGL** - Alphabet Inc. (Google)
- **MSFT** - Microsoft Corporation
- **NVDA** - NVIDIA Corporation
- **TSLA** - Tesla Inc.

### 🏆 **Commodities**
- **GC=F** - Gold Futures

## Features

- **🔄 Real-Time Market Dashboard**: Live tracking with auto-refresh every second
- **🤖 AI-Powered Predictions**: 7-day price forecasts using LSTM deep learning
- **📊 Interactive Visualizations**: Dynamic charts showing predicted price trends
- **🎯 High Accuracy Models**: 93-98% accuracy across different assets
- **🖼️ Visual Asset Selection**: Intuitive UI with asset logos
- **📱 Responsive Design**: Works on desktop and mobile devices
- **🔧 Automated Model Training**: Built-in training pipeline for model updates

## Model Performance

| Asset | Type | Accuracy | MAE | RMSE | Volatility |
|-------|------|----------|-----|------|------------|
| **GC=F** | Commodity | 98.41% | $44.31 | $57.64 | Moderate |
| **MSFT** | Stock | 97.80% | $8.62 | $11.02 | Low |
| **GOOGL** | Stock | 96.97% | $6.13 | $7.19 | Low |
| **AMZN** | Stock | 96.23% | $7.17 | $8.79 | Low |
| **AAPL** | Stock | 95.17% | $8.11 | $11.35 | Moderate |
| **NVDA** | Stock | 94.64% | $7.60 | $9.26 | Moderate |
| **TSLA** | Stock | 93.12% | $18.82 | $24.53 | High |

*Note: Gold shows higher absolute error due to its price scale (~$3,600/oz vs stocks ~$200-500)*

## Tech Stack

- **Backend Framework**: Flask (Python)
- **Machine Learning**: TensorFlow/Keras LSTM Networks
- **Data Source**: Yahoo Finance API (yfinance)
- **Data Processing**: NumPy, Pandas, Scikit-learn
- **Visualization**: Matplotlib
- **Model Storage**: HDF5 (.h5) format
- **Frontend**: HTML5, CSS3, JavaScript (with jQuery)

## Installation

### Prerequisites
- Python 3.8+
- pip package manager
- 2GB+ free disk space (for models)

### Quick Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd investment-predictor
```

2. **Create virtual environment**
```bash
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Train models (first time setup)**
```bash
python model_trainer.py
```
This will:
- Download 2 years of historical data
- Train LSTM models for each asset
- Save models and scalers
- Generate initial predictions

5. **Run the application**
```bash
python app.py
```

6. **Access the platform**
Open browser and navigate to: `http://localhost:5000`

## Project Structure

```
investment-predictor/
│
├── app.py                           # Main Flask application
├── model_trainer.py                 # LSTM model training pipeline
├── accuracy_checker.py              # Model performance evaluation
├── requirements.txt                 # Python dependencies
│
├── models/                          # Trained models and predictions
│   ├── AAPL.h5                     # Apple LSTM model
│   ├── AAPL_scaler.pkl             # Apple data scaler
│   ├── AAPL_forecast.npy           # Apple 7-day forecast
│   ├── GC=F.h5                     # Gold LSTM model
│   └── ...                         # Other asset models
│
├── predictions/                     # Prediction outputs
│   └── predictions.json            # All asset predictions
│
├── utils/                          # Utility modules
│   ├── data_fetcher.py            # Yahoo Finance data retrieval
│   └── predictor.py               # Prediction logic
│
├── templates/                      # HTML templates
│   ├── index.html                 # Landing page
│   ├── market.html                # Live market dashboard
│   ├── predict.html               # Asset selection grid
│   └── predict_stock.html         # Prediction results
│
├── static/                         # Static assets
│   ├── images/                    # Asset logos & backgrounds
│   │   ├── apple.png
│   │   ├── gold.png
│   │   └── ...
│   ├── css/                       # Stylesheets
│   └── js/                        # JavaScript files
│
└── accuracy_report_with_percentage.json  # Model metrics
```

## Usage Guide

### 1. **Home Page**
- Landing page with Investment Predictor branding
- Access navigation menu (☰) for all features

### 2. **Market Dashboard** (`/market`)
- Real-time prices for all tracked assets
- Auto-updates every second
- Visual cards showing current values
- Perfect for monitoring your portfolio

### 3. **Prediction Center** (`/predict`)
- Grid layout with asset logos
- Click any asset to generate 7-day forecast
- Supports both stocks and gold

### 4. **Prediction Results** (`/predict/<symbol>`)
- Current price display
- 7-day price prediction chart
- Visual trend analysis

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Home page with navigation |
| `/market` | GET | Live market dashboard |
| `/market-data` | GET | JSON API for real-time prices |
| `/predict` | GET | Asset selection interface |
| `/predict/<symbol>` | GET | 7-day prediction for specific asset |

## Model Architecture

### LSTM Configuration
- **Input Sequence**: 90 days of historical closing prices
- **Architecture**: LSTM(50 units) → Dense(1)
- **Training**: 20 epochs, batch size 32
- **Optimizer**: Adam
- **Loss Function**: Mean Squared Error
- **Data Normalization**: MinMaxScaler

### Training Process
```python
# Run training pipeline
python model_trainer.py

# This will:
# 1. Download 2 years of historical data
# 2. Create 90-day sequences for training
# 3. Train individual LSTM models
# 4. Generate 7-day forecasts
# 5. Save models and predictions
```

## Dependencies

Create `requirements.txt`:
```
Flask==2.3.3
yfinance==0.2.28
numpy==1.24.3
pandas==2.0.3
matplotlib==3.7.2
tensorflow==2.13.0
keras==2.13.1
scikit-learn==1.3.0
joblib==1.3.2
```

## Performance Monitoring

Run accuracy evaluation:
```bash
python accuracy_checker.py
```

This generates:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Accuracy percentage
- Average actual prices for comparison

## Investment Insights

### Best Performers
- **Gold (GC=F)**: Highest accuracy (98.41%), good for portfolio stability
- **GOOGL & AMZN**: Most predictable stocks (>96% accuracy)

### Moderate Risk
- **AAPL, MSFT, NVDA**: Solid predictions with moderate volatility

### High Volatility
- **TSLA**: Highest prediction error due to market volatility

## Risk Disclaimer

⚠️ **IMPORTANT INVESTMENT WARNING**

This platform is for **educational and informational purposes only**. 

- Past performance does not guarantee future results
- AI predictions are based on historical patterns which may not repeat
- Always conduct your own research before making investment decisions
- Consider consulting with licensed financial advisors
- Never invest more than you can afford to lose
- Commodity and stock markets carry inherent risks

## Future Enhancements

- [ ] Add cryptocurrency predictions (BTC, ETH)
- [ ] Include more commodities (Silver, Oil, Natural Gas)
- [ ] Implement portfolio optimization algorithms
- [ ] Add technical indicators (RSI, MACD, Bollinger Bands)
- [ ] Include sentiment analysis from news/social media
- [ ] Create risk assessment metrics
- [ ] Add backtesting capabilities
- [ ] Implement stop-loss recommendations
- [ ] Include dividend yield analysis
- [ ] Add options pricing predictions

## Troubleshooting

### Common Issues

1. **Models not found error**
   - Run `python model_trainer.py` first

2. **Real-time data not updating**
   - Check internet connection
   - Verify market hours (NYSE: 9:30 AM - 4:00 PM EST)

3. **High prediction errors**
   - Retrain models with recent data
   - Check for market anomalies or major events

## Contributing

We welcome contributions! Areas of interest:
- Additional asset classes
- Improved ML models
- UI/UX enhancements
- Performance optimizations

## License

MIT License - See LICENSE file for details

## Acknowledgments

- Yahoo Finance for market data API
- TensorFlow team for deep learning framework
- Flask community for web framework
- Financial data providers and market makers

## Support

For issues or questions, please open an issue in the repository.

---

**Remember**: Invest responsibly. This tool is meant to assist, not replace, prudent investment analysis.

*Last updated: October 2025*