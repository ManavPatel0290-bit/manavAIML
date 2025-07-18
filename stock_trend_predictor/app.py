# app.py

import streamlit as st
from joblib import load
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="Live Stock Trend Predictor", page_icon="📈")

st.title("📈 Live Stock Trend Predictor")

# Load Model
model = load('stock_trend_model.joblib')

# Stock Input
symbol = st.text_input("Enter NSE stock symbol (e.g., RELIANCE.NS):", 'RELIANCE.NS')

if st.button("Predict Trend"):
    with st.spinner("Fetching latest data..."):
        data = yf.download(symbol, period='7d')

        if data.empty:
            st.warning("No data found. Check symbol.")
        else:
            data['Open-Close'] = data['Open'] - data['Close']
            data['High-Low'] = data['High'] - data['Low']
            data['MA5'] = data['Close'].rolling(5).mean()
            data = data.dropna()

            if len(data) > 0:
                latest = data.iloc[-1]
                features = [[
                    latest['Open-Close'],
                    latest['High-Low'],
                    latest['MA5']
                ]]
                prediction = model.predict(features)[0]

                trend = "📈 UP" if prediction == 1 else "📉 DOWN"
                st.success(f"Predicted Trend for **{symbol}**: {trend}")
            else:
                st.warning("Not enough data to calculate features.")
