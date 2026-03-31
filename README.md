# 🏦  — 📈 EquityLens

🚀 A full-stack machine learning system for real-time stock analysis and 5-day return prediction.
Built as a full-stack ML + fintech dashboard project.

## ⚡ Overview

EquityLens is a quantitative finance project that predicts short-term stock returns using machine learning and technical indicators.

📊 15 engineered financial features (RSI, MACD, volatility, trend)
🤖 XGBoost model for non-linear prediction
🔁 Walk-forward validation (no data leakage)
🌐 Flask API for real-time inference
💻 Interactive dashboard for visualization

## 🏗️ Tech Stack
ML: XGBoost, Scikit-learn
Backend: Flask (REST API)
Frontend: HTML, CSS, JavaScript
Data: Yahoo Finance (yfinance)

**Live Demo:** Coming soon · **Author:** [Your Name] · **GitHub:** [Your GitHub URL]

---

## 🎯 The Problem

Predicting short-term equity returns is the core problem in quantitative finance. Naive approaches fail because:

1. **Data Leakage** — Random train/test splits assume future price data exists in training set (it doesn't)
2. **Regime Changes** — Market regimes shift; a model trained on bull markets fails in corrections
3. **Non-Linear Relationships** — Return drivers interact in complex, non-obvious ways (momentum × volatility × mean reversion)

This project solves these problems by:
- Using **walk-forward validation** — trains on past data, tests strictly on future data
- Engineering **15 features** across 5 financial domains (momentum, volatility, trend, mean reversion, return)
- Deploying **XGBoost regression** — captures non-linear patterns institutional investors use

The result: a model that respects financial time-series logic and can be confidently explained in a quant interview.

---


### 🤖 Why XGBoost?

| Model | MAE | R² | Why? |
|-------|-----|----|----|
| **XGBoost** | **2.795%** | **-0.0496** | ✅ Captures non-linear momentum/volatility interaction |
| Random Forest | 3.121% | -0.0891 | Overfits to random patterns in training set |
| Linear Regression | 3.445% | -0.1523 | Assumes linearity; returns are regime-dependent |

XGBoost was selected because equity returns are **regime-dependent** — momentum works in strong trends but fails in mean-reversion regimes. Gradient boosting captures these transitions; linear models cannot.

---


## 🏗️ Architecture

```
index.html  (Premium fintech dashboard UI)
     │
     │ POST /predict {ticker, lookback}
     │ GET  /health
     ▼
api.py  (Flask REST API)
     │
     │ Loads trained artifacts
     │ Engineers features on-the-fly
     │ Serves predictions in <500ms
     ▼
model_artifacts/
  ├── model.pkl         (XGBoost regressor, serialized)
  ├── scaler.pkl        (StandardScaler for feature normalization)
  ├── features.pkl      (List of feature names)
  └── train_model.py    (Reproducible training script)
```

**Why this architecture?**
- **Stateless API** — No database; predictions computed on-demand from yfinance data
- **Separation of concerns** — Model training (offline) vs. serving (online)
- **Reproducible** — train_model.py can retrain from scratch; same approach as prod systems

---

## ⚡ Quick Start

### Step 1: Install Dependencies
```bash
pip install flask flask-cors xgboost scikit-learn pandas numpy yfinance
```

### Step 2: Train the Model (if needed)
```bash
python train_model.py
```
Creates: `model.pkl`, `scaler.pkl`, `features.pkl`

### Step 3: Start the API & Open Dashboard
```bash
python api.py
```
Opens browser at `http://localhost:5000` → dashboard loads automatically

✅ **Done.** Your app is live. Select JPM/GS/BAC or type any ticker.

---

## 🛠️ Tech Stack

| Layer | Technology | Why? |
|-------|-----------|------|
| **ML Framework** | XGBoost, Scikit-learn | Gradient boosting for non-linear patterns; industry standard |
| **Backend** | Python, Flask, REST API | Lightweight, fast, production-proven |
| **Frontend** | HTML5, CSS3, Vanilla JS | Zero dependencies; works in any browser |
| **Data Source** | Yahoo Finance (yfinance) | Free, reliable, real-time pricing data |
| **Deployment** | Localhost (dev) / Cloud-ready (prod) | REST API scales to cloud with 1 click |

---

## 📁 Project Structure

```
stock-market-analyzer/
├── train_model.py           # Fetch data, engineer features, train XGBoost, save artifacts
├── api.py                   # Flask REST API — load model, serve predictions
├── index.html               # Premium fintech dashboard (no build step needed)
├── model.pkl                # Trained XGBoost model
├── scaler.pkl               # StandardScaler (fitted on training data)
├── features.pkl             # List of 15 feature names (for consistency)
└── README.md                # This file
```

---

## ⚠️ Disclaimer

This project is for **educational purposes** and **portfolio demonstration**. Predictions are not financial advice. Backtested performance ≠ future results. Markets are inherently unpredictable; use this tool as one input among many, never as the sole basis for trading decisions.

---

## 👤 Author

**[Your Name]**

- **GitHub:** [@your-github-handle](https://github.com/your-github-handle)
- **LinkedIn:** [your-linkedin-url](https://linkedin.com/in/your-profile)
- **Email:** your-email@domain.com

Portfolio project demonstrating applied machine learning in quantitative finance — built to show understanding of time-series validation, feature engineering, and production ML systems.

---

## 📜 License

This project is open source. Feel free to fork, modify, and use for learning.
