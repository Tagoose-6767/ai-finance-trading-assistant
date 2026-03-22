import yfinance as yf
import pandas as pd
import numpy as np
import requests
import time
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# -----------------------------
# Setup
# -----------------------------
start = time.time()
print("STEP 3 RUNNING 😤")

analyzer = SentimentIntensityAnalyzer()
API_KEY = "01dcba7a5df045d38611a5878b764060"

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
# News Sentiment (FAST VERSION)
# -----------------------------
def get_news_sentiment(ticker):
    url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey={API_KEY}"
    response = requests.get(url).json()

    sentiments = []

    for article in response.get("articles", [])[:5]:
        text = article["title"]
        score = analyzer.polarity_scores(text)["compound"]
        sentiments.append(score)

    if len(sentiments) == 0:
        return 0

    return sum(sentiments) / len(sentiments)

print("Fetching sentiment...")
sentiment_map = {}

for ticker in tickers:
    print(f"Getting sentiment for {ticker}")
    sentiment_map[ticker] = get_news_sentiment(ticker)

data["Sentiment"] = data["Ticker"].map(sentiment_map)
print("Sentiment done ✅")

# -----------------------------
# Feature Engineering (CLEAN)
# -----------------------------
data["Return"] = data.groupby("Ticker")["Close"].pct_change()
data["MA5"] = data.groupby("Ticker")["Close"].transform(lambda x: x.rolling(5).mean())
data["MA10"] = data.groupby("Ticker")["Close"].transform(lambda x: x.rolling(10).mean())

for i in range(1, 4):
    data[f"Lag_{i}"] = data.groupby("Ticker")["Return"].shift(i)

data["Price_vs_MA"] = data["Close"] / data["MA10"]

# -----------------------------
# Target (CORRECT)
# -----------------------------
data["Target"] = data.groupby("Ticker")["Return"].shift(-1)
data["Target"] = (data["Target"] > 0.02).astype(int)

data = data.dropna().copy()

# -----------------------------
# Features
# -----------------------------
features = ["Return", "MA5", "MA10", "Price_vs_MA"] + [f"Lag_{i}" for i in range(1,4)] + ["Sentiment"]

X = data[features]
y = data["Target"]

# -----------------------------
# Time-Based Split
# -----------------------------
split_index = int(len(X) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# -----------------------------
# Models (FAST + OPTIMIZED)
# -----------------------------
model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

model2 = LogisticRegression(max_iter=1000)
model2.fit(X_train, y_train)

# -----------------------------
# Ensemble Prediction
# -----------------------------
rf_probs = model.predict_proba(X_test)[:, 1]
lr_probs = model2.predict_proba(X_test)[:, 1]

probs = (rf_probs + lr_probs) / 2

# Rank-based prediction
sorted_indices = np.argsort(probs)[::-1]
n_pos = int(sum(y_test))

preds = np.zeros_like(probs)
preds[sorted_indices[:n_pos]] = 1

# -----------------------------
# Evaluation
# -----------------------------
accuracy = accuracy_score(y_test, preds)
print("Baseline Accuracy:", accuracy)

preds2 = model2.predict(X_test)
accuracy2 = accuracy_score(y_test, preds2)
print("Logistic Regression Accuracy:", accuracy2)

print("Actual positives:", sum(y_test))
print("Predicted positives:", sum(preds))

print(classification_report(y_test, preds))

# -----------------------------
# Graph
# -----------------------------
plt.figure(figsize=(10,5))
plt.plot(y_test.values, label="Actual")
plt.plot(preds, label="Predicted", alpha=0.7)

plt.legend()
plt.title("Prediction vs Actual")
plt.savefig("prediction_graph.png")

print("Graph saved! 📈")
print("Total runtime:", time.time() - start)