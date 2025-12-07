Bitcoin Model Dashboard — Multi-Model Price Forecasting System

An end-to-end Bitcoin price forecasting dashboard that combines multiple time-series models, live market data, and real-time news — designed as an industry-grade ML prototype.

This project integrates forecasting, meta-learning, live APIs, and an interactive UI to simulate how real-world ML systems are built and monitored.

📌 Key Features
🔹 Multi-Model Price Forecasting

ARIMA — classical statistical time-series model

LSTM — deep learning model trained on engineered features

Prophet — trend + seasonality forecasting

Supports multi-day forecasts and rolling inference

🔹 Meta-Model Correction (1-Day Horizon)

Separate meta-models refine base predictions

Uses:

Base model prediction

Latest BTC close

News sentiment score

Improves short-term accuracy

🔹 Live Market Snapshot (CoinGecko)

Current BTC price (USD)

24h high / low

24h trading volume

Market capitalization

Auto-refresh with caching

🔹 Bitcoin News Integration

Curated live news from CoinDesk and CoinTelegraph

Filtered for Bitcoin-relevant articles

Displayed directly in the dashboard

🔹 Interactive Streamlit Dashboard

Model selection & forecast horizon controls

Candlestick price chart with forecast overlays

Clean glassmorphism UI

Optimized layout:

Left: controls (collapsible sidebar)

Middle: predictions & chart

Right: snapshot & news

🧠 System Architecture (High Level)
Yahoo Finance      CoinGecko        Crypto News RSS
     |                 |                  |
     v                 v                  v
  Price Data     Market Snapshot        News Text
     |                                    |
     |                              Sentiment
     v                                    |
 ARIMA / Prophet / LSTM        -------------- 
     |                           Meta Models
     v                                |
 Forecasts  -------------------------+
     |
     v
 Streamlit Dashboard (UI)

📊 Models Used
Model	Purpose
ARIMA	Baseline statistical forecasting
LSTM	Non-linear temporal patterns
Prophet	Trend & seasonality handling
Meta-Models	Short-term prediction correction
🛠️ Tech Stack

Languages & Frameworks

Python

Streamlit

Plotly

Data & APIs

Yahoo Finance (yfinance)

CoinGecko API

CoinDesk & CoinTelegraph RSS

Machine Learning

scikit-learn

TensorFlow

pmdarima

Prophet

NLP

VADER Sentiment Analysis

📁 Project Structure
bitcoin-meta-pipeline/
│
├── app/
│   └── streamlit_app.py   # Main dashboard
│
├── scripts/
│   ├── train_meta_daily.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   └── model_utils.py
│
├── models/
│   ├── arima/
│   ├── lstm/
│   ├── prophet/
│   └── meta_models/
│
├── requirements.txt
├── README.md
└── .gitignore

⚙️ Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/your-username/bitcoin-model-dashboard.git
cd bitcoin-model-dashboard

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run the App
streamlit run app/streamlit_app.py

✅ Requirements

Make sure your requirements.txt contains:

streamlit
plotly
numpy
pandas
yfinance
scikit-learn
tensorflow
pmdarima
prophet
vaderSentiment
requests
feedparser
python-dateutil
pytz
joblib

📌 Design Philosophy

Modular architecture (training ≠ inference)

Safe handling of NaN / infinite predictions

API-first thinking

Production-style caching & timeouts

UI tailored for decision-making, not demos

🚀 Future Improvements

Automated daily retraining (cron / scheduler)

Model versioning & experiment tracking

Confidence intervals for forecasts

FastAPI backend for inference

Cloud deployment (AWS / GCP / Azure)

MLOps integration (MLflow)

🎯 Project Level

Industry-level ML prototype
Suitable for:

Data Scientist / ML Engineer portfolios

Final-year or major academic projects

Interview case studies

👤 Author

Om
B.Tech Computer Science
Focus: Data Science, Machine Learning, and Applied ML Systems
