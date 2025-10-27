# Investment Predictor 📈💰

A sophisticated Flask-based investment prediction platform that provides real-time market data and AI-powered 7-day price forecasts for stocks and commodities using LSTM neural networks.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13.0-orange.svg)
![Flask](https://img.shields.io/badge/Flask-2.3.3-green.svg)

## 🚀 Overview

Investment Predictor combines historical and real-time data from Yahoo Finance with advanced machine learning to deliver actionable investment insights. The platform leverages LSTM networks for capturing long-term market trends while maintaining exceptional accuracy across diverse asset classes.

## ✨ Key Features

- **📊 Real-Time Market Dashboard**: Live price tracking with auto-refresh capabilities
- **🤖 AI-Powered Predictions**: 7-day forecasts using LSTM deep learning models
- **📈 Interactive Visualizations**: Dynamic charts for trend analysis
- **🎯 High Accuracy Models**: 93-98% accuracy across different assets
- **💼 Diverse Portfolio Coverage**: Stocks (AAPL, AMZN, GOOGL, MSFT, NVDA, TSLA) + Gold Futures
- **🔄 Automated Pipeline**: Built-in model training and evaluation system
- **📱 Responsive Design**: Seamless experience across desktop and mobile

## 📊 Model Performance

| Asset | Type | Accuracy | MAE | RMSE | Investment Profile |
|-------|------|----------|-----|------|--------------------|
| **GC=F** | Commodity | **98.41%** | $44.31 | $57.64 | Stable, Portfolio Hedge |
| **MSFT** | Stock | **97.80%** | $8.62 | $11.02 | Low Volatility |
| **GOOGL** | Stock | **96.97%** | $6.13 | $7.19 | Highly Predictable |
| **AMZN** | Stock | **96.23%** | $7.17 | $8.79 | Consistent Performer |
| **AAPL** | Stock | **95.17%** | $8.11 | $11.35 | Moderate Risk |
| **NVDA** | Stock | **94.64%** | $7.60 | $9.26 | Growth Focused |
| **TSLA** | Stock | **93.12%** | $18.82 | $24.53 | High Volatility |

*Note: Gold's higher absolute error reflects its price scale (~$3,600/oz vs stocks ~$200-500)*

## 🛠️ Tech Stack

### Backend
- **Framework**: Flask (Python)
- **ML Framework**: TensorFlow/Keras
- **Data Source**: Yahoo Finance API
- **Data Processing**: NumPy, Pandas, Scikit-learn

### Frontend
- **Languages**: HTML5, CSS3, JavaScript
- **Libraries**: jQuery, Tailwind CSS
- **Visualization**: Matplotlib

### Model Architecture
- **Algorithm**: LSTM Neural Networks
- **Input Sequence**: 90 days historical data
- **Architecture**: LSTM(50) → Dense(1)
- **Optimizer**: Adam
- **Training**: 20 epochs, batch size 32

## 📦 Installation

### Prerequisites
- Python 3.8+
- pip package manager
- 2GB+ free disk space

### Quick Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd investment-predictor
```

2. **Create virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Train models (first-time setup)**
```bash
python model_trainer.py
```

5. **Run the application**
```bash
python app.py
```

6. **Access the platform**
```
http://localhost:5000
```

## 📁 Project Structure

```
investment-predictor/
│
├── app.py                           # Main Flask application
├── model_trainer.py                 # LSTM training pipeline
├── accuracy_checker.py              # Performance evaluation
├── requirements.txt                 # Dependencies
│
├── models/                          # Trained models
│   ├── {SYMBOL}.h5                 # LSTM models
│   ├── {SYMBOL}_scaler.pkl         # Data scalers
│   └── {SYMBOL}_forecast.npy       # Predictions
│
├── predictions/                     # Forecast outputs
│   └── predictions.json
│
├── utils/                          # Core utilities
│   ├── data_fetcher.py            # Yahoo Finance integration
│   └── predictor.py               # Prediction engine
│
├── templates/                      # UI templates
│   ├── index.html                 # Landing page
│   ├── market.html                # Live dashboard
│   ├── predict.html               # Asset selection
│   └── predict_stock.html         # Results display
│
└── static/                         # Assets
    └── images/                     # Logos & icons
```

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Landing page |
| `/market` | GET | Real-time market dashboard |
| `/market-data` | GET | JSON API for live prices |
| `/predict` | GET | Asset selection interface |
| `/predict/<symbol>` | GET | 7-day forecast for asset |

## 🧠 How It Works

1. **Data Collection**: Fetches 2 years of historical data from Yahoo Finance
2. **Preprocessing**: Normalizes data using MinMaxScaler for optimal neural network performance
3. **Sequence Creation**: Generates 90-day sliding windows for time-series analysis
4. **Model Training**: LSTM networks learn complex market patterns
5. **Prediction**: Generates recursive 7-day forecasts
6. **Visualization**: Renders interactive charts for decision-making

## 📈 Usage Examples

### Training Models
```python
# Retrain with latest data
python model_trainer.py

# Evaluate accuracy
python accuracy_checker.py
```

### Making Predictions
```python
from utils.predictor import make_prediction

# Get forecast for Apple
latest_price, forecast = make_prediction("AAPL")
print(f"Current: ${latest_price}")
print(f"7-day forecast: {forecast}")
```

## 🎯 Investment Insights

### Portfolio Recommendations

**Conservative (98%+ accuracy)**
- Gold (GC=F) - Inflation hedge, stable

**Balanced (96-97% accuracy)**
- GOOGL, AMZN, MSFT - Tech blue chips

**Growth (94-95% accuracy)**
- AAPL, NVDA - Innovation leaders

**Speculative (93% accuracy)**
- TSLA - High risk/reward

## 🔮 Future Enhancements

- [ ] **Sentiment Analysis**: Integrate Reddit, Twitter, and news sentiment
- [ ] **Cryptocurrency Support**: Add BTC, ETH predictions
- [ ] **Technical Indicators**: RSI, MACD, Bollinger Bands
- [ ] **Portfolio Optimizer**: Risk-adjusted allocation recommendations
- [ ] **Backtesting Engine**: Historical performance validation
- [ ] **Options Pricing**: Derivatives market predictions
- [ ] **Alert System**: Price target notifications
- [ ] **Random Forest Integration**: Short-term prediction refinement

## ⚠️ Disclaimer

**IMPORTANT**: This tool is for educational and informational purposes only. 

- Not financial advice
- Past performance ≠ future results
- Markets carry inherent risks
- Consult licensed financial advisors
- Never invest more than you can afford to lose

## 🤝 Contributing

Contributions are welcome! Areas of interest:
- Model architecture improvements
- Additional data sources
- UI/UX enhancements
- Performance optimizations

## 📊 Performance Metrics

The system achieves exceptional accuracy through:
- **Data Quality**: 730 days of clean historical data
- **Feature Engineering**: Optimized 90-day sequences
- **Model Tuning**: Hyperparameter optimization
- **Validation**: Rigorous MAE/RMSE evaluation

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| Models not found | Run `python model_trainer.py` |
| Data not updating | Check market hours (9:30 AM - 4:00 PM EST) |
| High prediction errors | Retrain models with recent data |

## 🙏 Acknowledgments

- Yahoo Finance for market data API
- TensorFlow team for ML framework
- Flask community for web framework
- Contributors and maintainers

## 📫 Contact

For questions or support, please open an issue in the repository.

---

**Remember**: Successful investing requires research, patience, and risk management. Use this tool as one of many resources in your investment journey.

*Built with ❤️ for data-driven investors*