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

API_KEY = st.secrets["API_KEY"]
analyzer = SentimentIntensityAnalyzer()

# -----------------------------
# Input
# -----------------------------
ticker = st.selectbox("Choose stock:", ["AAPL", "TSLA", "NVDA", "META"])

# -----------------------------
# Cached Data Loader
# -----------------------------
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, start="2020-01-01")

    # Fix MultiIndex issue
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    return data

# -----------------------------
# Cached Sentiment
# -----------------------------
@st.cache_data(ttl=3600)
def get_news_sentiment(ticker):
    url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey={API_KEY}"
    response = requests.get(url).json()

    sentiments = []
    for article in response.get("articles", [])[:5]:
        score = analyzer.polarity_scores(article["title"])["compound"]
        sentiments.append(score)

    return sum(sentiments)/len(sentiments) if sentiments else 0

# -----------------------------
# Cached Model Training
# -----------------------------
@st.cache_resource
def train_models(X_train, y_train):
    model = XGBClassifier(n_estimators=100, max_depth=4)
    model.fit(X_train, y_train)

    model2 = LogisticRegression(max_iter=1000)
    model2.fit(X_train, y_train)

    return model, model2

# -----------------------------
# Load Data
# -----------------------------
data = load_data(ticker)

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

    # Target
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
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Cached model training
    model, model2 = train_models(X_train_scaled, y_train)

    # Latest point
    latest = scaler.transform(X.iloc[-1:])

    xgb_prob = model.predict_proba(latest)[0][1]
    lr_prob = model2.predict_proba(latest)[0][1]

    prob = (xgb_prob + lr_prob) / 2

    # Cached sentiment
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
