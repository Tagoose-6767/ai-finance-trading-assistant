import yfinance as yf
import pandas as pd
import numpy as np
import requests
import time
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# -----------------------------
# Setup
# -----------------------------
start = time.time()
print("RUNNING IMPROVED MODEL 🚀")

tickers = ["AAPL", "META", "NVDA", "XOM", "CVX"]

# -----------------------------
# Load Data
# -----------------------------
all_data = []

for ticker in tickers:
    temp = yf.download(ticker, start="2020-01-01", end="2024-01-01")

    temp.columns = temp.columns.droplevel(1)
    temp = temp[["Close"]]
    temp["Ticker"] = ticker
    temp = temp.reset_index()

    all_data.append(temp)

data = pd.concat(all_data, ignore_index=True)

# -----------------------------
# Feature Engineering
# -----------------------------
data["Return"] = data.groupby("Ticker")["Close"].pct_change()
data["MA5"] = data.groupby("Ticker")["Close"].transform(lambda x: x.rolling(5).mean())
data["MA10"] = data.groupby("Ticker")["Close"].transform(lambda x: x.rolling(10).mean())

for i in range(1, 4):
    data[f"Lag_{i}"] = data.groupby("Ticker")["Return"].shift(i)

data["Price_vs_MA"] = data["Close"] / data["MA10"]

# NEW FEATURES
data["Volatility"] = data.groupby("Ticker")["Return"].transform(lambda x: x.rolling(10).std())
data["RSI"] = 100 - (100 / (1 + data.groupby("Ticker")["Return"].transform(lambda x: x.rolling(14).mean())))

# Target FIXED
data["Target"] = (data.groupby("Ticker")["Return"].shift(-1) > 0).astype(int)

data = data.dropna().copy()

# -----------------------------
# Features
# -----------------------------
features = ["Return", "MA5", "MA10", "Price_vs_MA", "Volatility", "RSI"] + [f"Lag_{i}" for i in range(1,4)]

X = data[features]
y = data["Target"]

# -----------------------------
# Split
# -----------------------------
split_index = int(len(X) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# Models
# -----------------------------
xgb = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

lr = LogisticRegression(max_iter=1000)
rf = RandomForestClassifier(n_estimators=100)

xgb.fit(X_train, y_train)
lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

# -----------------------------
# Predictions
# -----------------------------
xgb_preds = xgb.predict(X_test)
lr_preds = lr.predict(X_test)
rf_preds = rf.predict(X_test)

# Ensemble (FIXED)
xgb_probs = xgb.predict_proba(X_test)[:, 1]
lr_probs = lr.predict_proba(X_test)[:, 1]

probs = (xgb_probs + lr_probs) / 2
ensemble_preds = (probs > 0.5).astype(int)

# -----------------------------
# Evaluation
# -----------------------------
print("XGBoost Accuracy:", accuracy_score(y_test, xgb_preds))
print("Logistic Accuracy:", accuracy_score(y_test, lr_preds))
print("Random Forest Accuracy:", accuracy_score(y_test, rf_preds))
print("Ensemble Accuracy:", accuracy_score(y_test, ensemble_preds))

print("\nClassification Report (Ensemble):")
print(classification_report(y_test, ensemble_preds))

# -----------------------------
# Graph
# -----------------------------
plt.figure(figsize=(10,5))
plt.plot(y_test.values, label="Actual")
plt.plot(ensemble_preds, label="Predicted", alpha=0.7)

plt.legend()
plt.title("Prediction vs Actual")
plt.savefig("prediction_graph.png")

print("Graph saved 📈")
print("Runtime:", time.time() - start)
