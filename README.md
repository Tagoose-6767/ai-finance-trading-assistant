# 📈 AI Finance Trading Assistant

An AI-powered stock prediction system that combines machine learning, technical indicators, and sentiment analysis to generate trading signals.

## 🚀 Features
- Ensemble ML models (XGBoost + Logistic Regression)
- Technical indicators (MA, RSI, volatility)
- Real-time stock data using yfinance
- News sentiment analysis (VADER)
- Streamlit interactive dashboard

## 📊 Models Used
- XGBoost
- Logistic Regression
- Random Forest (evaluation)

## 🧠 Methodology
- Time-series feature engineering
- Lag features + moving averages
- Binary classification (predict next-day direction)
- Ensemble probability averaging

## 📈 Results
- Achieves strong predictive performance across multiple stocks
- Visualization of predictions vs actual

## 🖥️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app/app.py

![Python](https://img.shields.io/badge/Python-3.10-blue)
![ML](https://img.shields.io/badge/Machine%20Learning-XGBoost-green)
