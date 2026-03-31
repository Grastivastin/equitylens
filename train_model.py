"""
train_model.py — Multi-Factor Equity Risk & Return Analyzer
============================================================
Trains an XGBoost model to predict 5-day forward returns
using technical momentum, volatility, and trend features.

Usage:
    python train_model.py

Output:
    model.pkl       — Trained XGBoost model
    scaler.pkl      — Feature scaler
    features.pkl    — Feature name list (used by api.py)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import pickle
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

# ── 1. Configuration ──────────────────────────────────────────────────────────

TICKERS   = ["JPM", "GS", "BAC", "AAPL", "MSFT", "NVDA", "SPY", "QQQ"]
START     = "2018-01-01"
END       = "2024-12-31"
HORIZON   = 5        # predict 5-day forward return
N_SPLITS  = 5        # walk-forward validation splits

print("=" * 60)
print("  Multi-Factor Equity Risk & Return Analyzer")
print("  Model Training Script")
print("=" * 60)

# ── 2. Download Price Data ────────────────────────────────────────────────────

print(f"\n[1/5] Downloading price data for: {TICKERS}")
raw = yf.download(TICKERS, start=START, end=END, auto_adjust=True, progress=False)

# Use 'Close' prices
prices = raw["Close"].dropna(how="all")
print(f"      Downloaded {len(prices)} trading days across {len(TICKERS)} tickers.")

# ── 3. Feature Engineering ────────────────────────────────────────────────────

print("\n[2/5] Engineering features...")

def compute_rsi(series, window=14):
    """Relative Strength Index — momentum oscillator (0-100)."""
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(window).mean()
    loss  = (-delta.clip(upper=0)).rolling(window).mean()
    rs    = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def compute_macd(series, fast=12, slow=26, signal=9):
    """MACD line and signal line — trend + momentum."""
    ema_fast   = series.ewm(span=fast, adjust=False).mean()
    ema_slow   = series.ewm(span=slow, adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

def compute_bollinger(series, window=20):
    """Bollinger Band %B — position within bands (mean reversion signal)."""
    sma   = series.rolling(window).mean()
    std   = series.rolling(window).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    pct_b = (series - lower) / (upper - lower + 1e-9)
    return pct_b

def compute_atr(high, low, close, window=14):
    """Average True Range — volatility measure."""
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window).mean()

all_frames = []

for ticker in TICKERS:
    if ticker not in prices.columns:
        continue

    close  = prices[ticker].dropna()

    # Download full OHLCV for ATR
    ohlcv  = yf.download(ticker, start=START, end=END,
                          auto_adjust=True, progress=False)
    high   = ohlcv["High"].squeeze()
    low    = ohlcv["Low"].squeeze()
    volume = ohlcv["Volume"].squeeze()

    df = pd.DataFrame(index=close.index)
    df["close"]   = close

    # ── Target: 5-day forward return ──────────────────────────────────────────
    df["target"]  = close.pct_change(HORIZON).shift(-HORIZON) * 100

    # ── Momentum Features ─────────────────────────────────────────────────────
    df["rsi_14"]          = compute_rsi(close, 14)
    df["rsi_7"]           = compute_rsi(close, 7)

    macd_line, sig_line   = compute_macd(close)
    df["macd"]            = macd_line
    df["macd_signal"]     = sig_line
    df["macd_hist"]       = macd_line - sig_line

    # ── Mean Reversion Features ───────────────────────────────────────────────
    df["bb_pct_b"]        = compute_bollinger(close, 20)

    # ── Trend / Regime Features ───────────────────────────────────────────────
    df["ma_50"]           = close.rolling(50).mean()
    df["ma_20"]           = close.rolling(20).mean()
    df["price_vs_ma50"]   = (close - df["ma_50"]) / df["ma_50"] * 100
    df["price_vs_ma20"]   = (close - df["ma_20"]) / df["ma_20"] * 100
    df["ma20_vs_ma50"]    = (df["ma_20"] - df["ma_50"]) / df["ma_50"] * 100

    # ── Return Features ───────────────────────────────────────────────────────
    df["return_1d"]       = close.pct_change(1) * 100
    df["return_5d"]       = close.pct_change(5) * 100
    df["return_20d"]      = close.pct_change(20) * 100

    # ── Volatility Features ───────────────────────────────────────────────────
    atr_vals              = compute_atr(high.reindex(close.index),
                                        low.reindex(close.index), close)
    df["atr_pct"]         = atr_vals / close * 100
    df["volatility_20d"]  = close.pct_change().rolling(20).std() * 100

    # ── Volume Features ───────────────────────────────────────────────────────
    vol_aligned           = volume.reindex(close.index)
    vol_ma                = vol_aligned.rolling(20).mean()
    vol_std               = vol_aligned.rolling(20).std()
    df["volume_zscore"]   = (vol_aligned - vol_ma) / (vol_std + 1e-9)

    df["ticker"]          = ticker
    all_frames.append(df)

data = pd.concat(all_frames).dropna()
print(f"      Final dataset: {len(data):,} samples × {data.shape[1]} columns")

# ── 4. Walk-Forward Validation ────────────────────────────────────────────────

print("\n[3/5] Running walk-forward cross-validation...")

FEATURE_COLS = [
    "rsi_14", "rsi_7",
    "macd", "macd_signal", "macd_hist",
    "bb_pct_b",
    "price_vs_ma50", "price_vs_ma20", "ma20_vs_ma50",
    "return_1d", "return_5d", "return_20d",
    "atr_pct", "volatility_20d",
    "volume_zscore"
]

X = data[FEATURE_COLS].values
y = data["target"].values

# Walk-forward: train on past, test on future — no data leakage
n         = len(X)
fold_size = n // (N_SPLITS + 1)
mae_scores, r2_scores = [], []

for i in range(1, N_SPLITS + 1):
    train_end   = i * fold_size
    test_end    = train_end + fold_size
    if test_end > n:
        break

    X_train, y_train = X[:train_end], y[:train_end]
    X_test,  y_test  = X[train_end:test_end], y[train_end:test_end]

    scaler_fold   = StandardScaler()
    X_train_sc    = scaler_fold.fit_transform(X_train)
    X_test_sc     = scaler_fold.transform(X_test)

    model_fold    = XGBRegressor(
        n_estimators=300, learning_rate=0.05,
        max_depth=5, subsample=0.8,
        colsample_bytree=0.8, random_state=42,
        verbosity=0
    )
    model_fold.fit(X_train_sc, y_train,
                   eval_set=[(X_test_sc, y_test)],
                   verbose=False)

    preds = model_fold.predict(X_test_sc)
    mae   = mean_absolute_error(y_test, preds)
    r2    = r2_score(y_test, preds)
    mae_scores.append(mae)
    r2_scores.append(r2)
    print(f"      Fold {i}: MAE = {mae:.4f}%  |  R² = {r2:.4f}")

print(f"\n      Walk-Forward MAE : {np.mean(mae_scores):.4f}% ± {np.std(mae_scores):.4f}%")
print(f"      Walk-Forward R²  : {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")

# ── 5. Train Final Model on All Data ─────────────────────────────────────────

print("\n[4/5] Training final model on full dataset...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

final_model = XGBRegressor(
    n_estimators=400, learning_rate=0.04,
    max_depth=5, subsample=0.8,
    colsample_bytree=0.8, random_state=42,
    verbosity=0
)
final_model.fit(X_scaled, y)

# Feature importance summary
importances = pd.Series(
    final_model.feature_importances_,
    index=FEATURE_COLS
).sort_values(ascending=False)

print("\n      Top 5 most important features:")
for feat, imp in importances.head(5).items():
    bar = "█" * int(imp * 200)
    print(f"      {feat:<22} {bar} {imp:.4f}")

# ── 6. Save Artifacts ────────────────────────────────────────────────────────

print("\n[5/5] Saving model artifacts...")

with open("model.pkl",    "wb") as f: pickle.dump(final_model,  f)
with open("scaler.pkl",   "wb") as f: pickle.dump(scaler,       f)
with open("features.pkl", "wb") as f: pickle.dump(FEATURE_COLS, f)

print("      ✓  model.pkl    saved")
print("      ✓  scaler.pkl   saved")
print("      ✓  features.pkl saved")

print("\n" + "=" * 60)
print("  Training complete! Run  python api.py  to start the server.")
print("=" * 60 + "\n")
