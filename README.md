# 🛡️ GigShield Risk Monitor

A real-time machine learning powered dashboard for detecting fraudulent gig economy transactions.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 🌟 Features

- **Real-time Risk Detection**: ML-powered fraud detection with instant classification
- **Interactive Dashboard**: Live monitoring with auto-refresh every 2 seconds
- **Risk Explanations**: Clear, actionable insights for every prediction
- **Transaction Analysis**: Detailed view of payment verification, account age, response time, and more
- **Smart Filtering**: Filter by risk level, payment amount, and probability scores
- **Visual Analytics**: KPI cards, distribution charts, and probability bars

## 🎯 Risk Categories

- **Safe** ✅ - Low risk transactions from verified, established accounts
- **Caution** ⚠️ - Medium risk requiring additional review
- **High Risk** 🚨 - Potentially fraudulent transactions flagged for immediate attention

## 🚀 Quick Start

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/gigshield-risk-monitor.git
   cd gigshield-risk-monitor
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the dashboard**
   ```bash
   # Terminal 1: Start web server
   python -m uvicorn src.app_server:app --host 0.0.0.0 --port 8000 --reload

   # Terminal 2: Start data pipeline
   python run_pipeline.py
   ```

4. **Open your browser**
   ```
   http://localhost:8000
   ```

### Using Batch Files (Windows)

```bash
# Start dashboard
start_dashboard.bat

# Start data pipeline
start_pipeline.bat

# Stop everything
stop_all.bat
```

## 📊 How It Works

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Feeder    │────▶│   Ingest &   │────▶│  Dashboard  │
│ (Generate)  │     │   Predict    │     │   (View)    │
└─────────────┘     └──────────────┘     └─────────────┘
     ↓                      ↓                    ↓
data/incoming/      data/predictions/    http://localhost:8000
```

1. **Feeder** generates synthetic job posting transactions
2. **Ingest & Predict** processes them through ML model
3. **Dashboard** displays real-time predictions with explanations

## 🧠 Machine Learning Model

- **Algorithm**: Logistic Regression (trained on labeled transaction data)
- **Features**: Payment verification, account age, response time, complaints, profile completeness
- **Output**: Risk probability scores + class predictions (safe/caution/high_risk)

## 📁 Project Structure

```
gigshield-risk-monitor/
├── src/
│   ├── app_server.py          # FastAPI web server
│   ├── feeder.py              # Transaction generator
│   ├── ingest_predict.py      # ML prediction engine
│   ├── train_model.py         # Model training
│   └── utils/                 # Helper utilities
├── web/
│   └── index.html             # Dashboard UI
├── config/
│   └── config.yaml            # Configuration
├── data/
│   ├── incoming/              # New transactions
│   ├── predictions/           # Prediction results
│   └── processed/             # Processed files
├── artifacts/
│   ├── preprocess/            # Scaler & encoder
│   └── models/                # Trained ML models
├── requirements.txt           # Python dependencies
├── Procfile                   # Deployment config
└── README.md
```

## 🌐 Deployment

Deploy to cloud platforms in minutes! See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for detailed instructions.

### Supported Platforms:
- ☁️ **Render** (Recommended - Free tier)
- ☁️ **Railway** (Easy deployment)
- ☁️ **Heroku** (Classic option)

Quick deploy to Render:
1. Push to GitHub
2. Connect to Render
3. Auto-deploy! 🎉

## 🎨 Dashboard Features

### KPI Cards
- Total transactions processed
- Average payment amount
- Average account age
- High-risk transaction count (prob ≥ 0.70)

### Risk Factor Analysis
Each prediction includes explanations:
- ⚠️ Payment verification status
- 🆕 Account age indicators
- 🐌 Response time analysis
- 💰 Payment amount flags
- ✅ Trust indicators

### Interactive Filters
- Filter by risk label (safe/caution/high_risk)
- Minimum high-risk probability
- Payment amount range (min/max)

## 🛠️ Configuration

Edit `config/config.yaml` to customize:
- Model selection (logreg, xgboost, etc.)
- Polling intervals
- File paths
- MLflow tracking

## 📈 API Endpoints

- `GET /` - Dashboard UI
- `GET /api/health` - Health check
- `GET /api/summary` - Summary statistics
- `GET /api/predictions` - Transaction predictions (with filters)

## 🔧 Technologies Used

- **Backend**: FastAPI, Python 3.10+
- **ML**: scikit-learn, XGBoost, pandas, numpy
- **Frontend**: Vanilla JavaScript, Tailwind CSS
- **Deployment**: Uvicorn, Gunicorn

## 📝 License

MIT License - feel free to use for your projects!

## 🤝 Contributing

Contributions welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests

## 👨‍💻 Author

Created with ❤️ for detecting fraud in the gig economy

## 🙏 Acknowledgments

- Built with Claude Code AI Assistant
- Inspired by real-world fraud detection challenges
- Designed for transparency and explainability

---

**⭐ Star this repo if you find it useful!**
