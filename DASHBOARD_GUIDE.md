# GigShield Risk Monitor Dashboard

## Overview
This dashboard displays **transaction details** from the feeder and **predictions** from the ingest_predict pipeline in real-time.

## Quick Start

### Option 1: Use Batch Files (Windows)
1. **Start the data pipeline**:
   - Double-click `start_pipeline.bat`
   - This opens 2 windows:
     - **Feeder**: Creates synthetic job transactions
     - **Predictions**: Analyzes transactions and generates risk predictions

2. **Start the dashboard**:
   - Double-click `start_dashboard.bat`
   - Open browser to: http://localhost:8000

### Option 2: Manual Start
```bash
# Terminal 1: Start feeder (creates transactions)
python src/feeder.py

# Terminal 2: Start prediction engine
python src/ingest_predict.py

# Terminal 3: Start web dashboard
python -m uvicorn src.app_server:app --host 0.0.0.0 --port 8000 --reload
```

Then open: http://localhost:8000

## Dashboard Features

### 1. **KPI Cards**
- Total rows processed
- Average pay amount
- Average account age
- High-risk transactions (prob ≥ 0.70)

### 2. **Pie Chart**
- Distribution of predictions: Safe, Caution, High Risk

### 3. **Filters**
- Filter by prediction label
- Filter by minimum high-risk probability
- Filter by pay amount range

### 4. **Transaction Table**
Shows all transactions with:
- **Prediction**: Risk label (safe/caution/high_risk)
- **Risk Score**: Probability of high risk
- **Transaction Details**: Pay amount, account age, response time
- **Source**: Original file name
- **Actions**: "View Details" button

### 5. **Detail Modal**
Click "View Details" on any row to see:
- **Transaction Details**:
  - Pay amount & verification status
  - Account age
  - Response time
  - Source file
- **Prediction Results**:
  - Predicted label
  - High risk probability
  - All class probabilities with visual bars

## Data Flow

```
feeder.py → data/incoming/
           ↓
ingest_predict.py → data/predictions/predictions_log.csv
                   ↓
app_server.py (FastAPI) → web/index.html (React Dashboard)
```

## Files
- **src/feeder.py**: Generates synthetic job transactions
- **src/ingest_predict.py**: Processes transactions and makes predictions
- **src/app_server.py**: FastAPI backend serving predictions data
- **web/index.html**: React-based dashboard UI

## Auto-Refresh
The dashboard automatically refreshes every 2 seconds to show the latest predictions.
