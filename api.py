"""
api.py — Stock Market Analyzer · Flask REST API
=================================================
Loads the trained XGBoost model and serves real-time
5-day return predictions via a REST endpoint.

Usage:
    python api.py

Endpoints:
    POST /predict   { "ticker": "JPM" }  → prediction + features
    GET  /health                          → API status
    GET  /                                → serves index.html
"""

import warnings
warnings.filterwarnings("ignore")

import os
import pickle
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ── App Setup ─────────────────────────────────────────────────────────────────

app = Flask(__name__, static_folder=".")
CORS(app)

# ── Load Model Artifacts ──────────────────────────────────────────────────────

print("=" * 60)
print("  🚀 Stock Market Analyzer API")
print("=" * 60)

try:
    with open("model.pkl",    "rb") as f: model    = pickle.load(f)
    with open("scaler.pkl",   "rb") as f: scaler   = pickle.load(f)
    with open("features.pkl", "rb") as f: features = pickle.load(f)
    MODEL_LOADED = True
    print(f"  Model Status : ✅ Loaded")
    print(f"  Features     : {len(features)}")
except Exception as e:
    MODEL_LOADED = False
    print(f"  Model Status : ❌ Failed to load — {e}")
    print("  Make sure model.pkl, scaler.pkl, features.pkl are in the same folder.")

print("=" * 60)

# ── Feature Engineering ───────────────────────────────────────────────────────

def compute_rsi(series, window=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(window).mean()
    loss  = (-delta.clip(upper=0)).rolling(window).mean()
    rs    = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast    = series.ewm(span=fast, adjust=False).mean()
    ema_slow    = series.ewm(span=slow, adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

def compute_bollinger(series, window=20):
    sma   = series.rolling(window).mean()
    std   = series.rolling(window).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    return (series - lower) / (upper - lower + 1e-9)

def compute_atr(high, low, close, window=14):
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window).mean()

def engineer_features(ticker: str):
    """Download live data and compute all 15 features."""
    raw = yf.download(ticker, period="120d", auto_adjust=True, progress=False)

    if raw.empty or len(raw) < 60:
        raise ValueError(f"Not enough data for {ticker}. Check the ticker symbol.")

    # Handle multi-level columns from yfinance
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(1)

    close  = raw["Close"].squeeze().astype(float)
    high   = raw["High"].squeeze().astype(float)
    low    = raw["Low"].squeeze().astype(float)
    volume = raw["Volume"].squeeze().astype(float)

    df = pd.DataFrame(index=close.index)

    # Momentum
    df["rsi_14"]      = compute_rsi(close, 14)
    df["rsi_7"]       = compute_rsi(close, 7)
    macd_line, sig    = compute_macd(close)
    df["macd"]        = macd_line
    df["macd_signal"] = sig
    df["macd_hist"]   = macd_line - sig

    # Mean reversion
    df["bb_pct_b"]    = compute_bollinger(close, 20)

    # Trend / regime
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    df["price_vs_ma50"] = (close - ma50) / (ma50 + 1e-9) * 100
    df["price_vs_ma20"] = (close - ma20) / (ma20 + 1e-9) * 100
    df["ma20_vs_ma50"]  = (ma20 - ma50) / (ma50 + 1e-9) * 100

    # Returns
    df["return_1d"]  = close.pct_change(1)  * 100
    df["return_5d"]  = close.pct_change(5)  * 100
    df["return_20d"] = close.pct_change(20) * 100

    # Volatility
    atr_vals           = compute_atr(high, low, close)
    df["atr_pct"]      = atr_vals / (close + 1e-9) * 100
    df["volatility_20d"] = close.pct_change().rolling(20).std() * 100

    # Volume
    vol_ma  = volume.rolling(20).mean()
    vol_std = volume.rolling(20).std()
    df["volume_zscore"] = (volume - vol_ma) / (vol_std + 1e-9)

    df = df.dropna()
    if df.empty:
        raise ValueError(f"Could not compute features for {ticker}.")

    latest = df.iloc[-1]
    return latest, close, raw

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the frontend dashboard."""
    return send_from_directory(".", "index.html")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":       "online",
        "model_loaded": MODEL_LOADED,
        "features":     len(features) if MODEL_LOADED else 0,
        "timestamp":    datetime.datetime.now().isoformat()
    })

@app.route("/predict", methods=["POST"])
def predict():
    if not MODEL_LOADED:
        return jsonify({"error": "Model not loaded. Run train_model.py first."}), 503

    body   = request.get_json(force=True, silent=True) or {}
    ticker = body.get("ticker", "").strip().upper()

    if not ticker:
        return jsonify({"error": "Provide a ticker symbol, e.g. { 'ticker': 'JPM' }"}), 400

    try:
        feat_row, close_series, raw = engineer_features(ticker)
    except Exception as e:
        return jsonify({"error": str(e)}), 422

    try:
        # Build feature vector in the exact order the model was trained on
        X = np.array([[feat_row[f] for f in features]])
        X_scaled = scaler.transform(X)
        pred_return = float(model.predict(X_scaled)[0])
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    # Prices
    current_price     = float(close_series.iloc[-1])
    past_price        = float(close_series.iloc[-6]) if len(close_series) >= 6 else current_price
    five_day_change   = ((current_price - past_price) / (past_price + 1e-9)) * 100

    # Signal
    if pred_return > 1.5:
        signal = "Buy"
    elif pred_return < -1.5:
        signal = "Sell"
    else:
        signal = "Hold"

    signal_color = {
        "Buy":  "#4db896",
        "Sell": "#c4576a",
        "Hold": "#c9a86c"
    }[signal]

    # Latest OHLCV
    last = raw.iloc[-1]
    def safe(val):
        try:
            v = val.item() if hasattr(val, 'item') else float(val)
            return round(v, 4)
        except:
            return 0.0

    ohlcv = {
        "open":   safe(last["Open"]),
        "high":   safe(last["High"]),
        "low":    safe(last["Low"]),
        "close":  safe(last["Close"]),
        "volume": safe(last["Volume"])
    }

    return jsonify({
        "ticker":               ticker,
        "current_price":        round(current_price, 2),
        "five_day_change_pct":  round(five_day_change, 2),
        "prediction_5d_return": round(pred_return, 4),
        "signal":               signal,
        "color":                signal_color,
        "timestamp":            datetime.datetime.now().isoformat(),
        "features":             {k: round(float(feat_row[k]), 6) for k in features},
        "latest_ohlcv":         ohlcv
    })

# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n  Starting Flask server on http://localhost:5000")
    print(f"  Open your browser at: http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
