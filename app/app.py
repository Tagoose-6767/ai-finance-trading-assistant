import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

st.title("📈 AI Trading Assistant")

API_KEY = "YOUR_API_KEY"
analyzer = SentimentIntensityAnalyzer()

# -----------------------------
# Input
# -----------------------------
ticker = st.selectbox("Choose stock:", ["AAPL", "TSLA", "NVDA", "META"])

# -----------------------------
# Sentiment (ONLY for prediction)
# -----------------------------
def get_news_sentiment(ticker):
    url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey={API_KEY}"
    response = requests.get(url).json()

    sentiments = []
    for article in response.get("articles", [])[:5]:
        text = article["title"]
        score = analyzer.polarity_scores(text)["compound"]
        sentiments.append(score)

    return sum(sentiments)/len(sentiments) if sentiments else 0

# -----------------------------
# Load Data
# -----------------------------
data = yf.download(ticker, start="2020-01-01")

if data.empty:
    st.error("Invalid ticker")
else:
    data["Return"] = data["Close"].pct_change()
    data["MA5"] = data["Close"].rolling(5).mean()
    data["MA10"] = data["Close"].rolling(10).mean()

    for i in range(1, 4):
        data[f"Lag_{i}"] = data["Return"].shift(i)

    data["Price_vs_MA"] = data["Close"] / data["MA10"]

    # NEW FEATURES
    data["Volatility"] = data["Return"].rolling(10).std()
    data["RSI"] = 100 - (100 / (1 + data["Return"].rolling(14).mean()))

    # Target FIXED
    data["Target"] = (data["Return"].shift(-1) > 0).astype(int)

    data = data.dropna()

    features = ["Return", "MA5", "MA10", "Price_vs_MA", "Volatility", "RSI"] + [f"Lag_{i}" for i in range(1,4)]

    X = data[features]
    y = data["Target"]

    # Split
    split = int(len(X)*0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Models
    model = XGBClassifier(n_estimators=100, max_depth=4)
    model.fit(X_train, y_train)

    model2 = LogisticRegression(max_iter=1000)
    model2.fit(X_train, y_train)

    # Latest point
    latest = scaler.transform(X.iloc[-1:])

    xgb_prob = model.predict_proba(latest)[0][1]
    lr_prob = model2.predict_proba(latest)[0][1]

    prob = (xgb_prob + lr_prob) / 2

    # Sentiment ONLY for display
    sentiment = get_news_sentiment(ticker)

    # -----------------------------
    # UI
    # -----------------------------
    st.subheader(f"📊 {ticker} Analysis")

    st.write(f"🧠 Sentiment Score: {round(sentiment, 3)}")
    st.write(f"🤖 Model Confidence: {round(prob, 3)}")

    st.progress(float(prob))

    st.write("Model Breakdown:")
    st.write(f"XGBoost: {round(xgb_prob, 3)}")
    st.write(f"Logistic: {round(lr_prob, 3)}")

    if prob > 0.65:
        st.success("🔥 BUY SIGNAL")
    elif prob > 0.5:
        st.warning("⚠️ HOLD")
    else:
        st.error("❌ SELL")

    st.line_chart(data["Close"])
