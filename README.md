## 🏦📈 EquityLens

🚀 A full-stack machine learning system for real-time stock analysis and 5-day return prediction.
Built as a full-stack ML + fintech dashboard project.

---

## ⚡ Overview

**EquityLens** predicts short-term stock returns using machine learning and technical indicators.

* 📊 15 engineered financial features (RSI, MACD, volatility, trend)
* 🤖 XGBoost model for non-linear prediction
* 🔁 Walk-forward validation (no data leakage)
* 🌐 Flask API for real-time inference
* 💻 Interactive dashboard

---

## 🏗️ Tech Stack

* **ML:** XGBoost, Scikit-learn
* **Backend:** Flask (REST API)
* **Frontend:** HTML, CSS, JavaScript
* **Data:** Yahoo Finance (yfinance)

---

## ⚡ How It Works

1. Fetch live stock data
2. Compute 15 technical indicators
3. Scale features
4. Predict 5-day return
5. Generate Buy/Sell/Hold signal

---

## 🚀 Run Locally

```bash
pip install -r requirements.txt
python api.py
```

Open: http://localhost:5000

---

## 📁 Structure

```
equitylens/
├── api.py
├── train_model.py
├── index.html
├── model.pkl
├── scaler.pkl
├── features.pkl
└── README.md
```

---

## ⚠️ Disclaimer

This project is for educational purposes only. Not financial advice.

---

## 👤 Author

**Sashank Peddada**

GitHub: https://github.com/Grastivastin
