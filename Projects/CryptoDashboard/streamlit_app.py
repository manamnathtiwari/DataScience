# streamlit_app.py

import streamlit as st
from app.data_ingestion import fetch_and_store_crypto_data
from app.spark_analysis import analyze_coin
from app.recommendation import get_recommendation
from config import REDIS_HOST, REDIS_PORT
import redis
import json
import matplotlib.pyplot as plt

# Redis client
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)

# --------------------- UI ------------------------
st.set_page_config(page_title="Crypto AI Advisor", layout="wide")
st.title("📊 Crypto Investment Advisor (Big Data + AI)")

# Sidebar inputs
coin = st.sidebar.selectbox("Select Coin", ["bitcoin", "ethereum", "dogecoin"])
days = st.sidebar.slider("Select Time Frame (Days)", 1, 90, 7)

if st.sidebar.button("🔍 Analyze"):

    with st.spinner("Fetching and processing data..."):
        # Step 1: Fetch and store
        fetch_and_store_crypto_data(coin, days)

        # Step 2: Analysis
        df = analyze_coin(coin, days)

        # Step 3: Recommendation
        recommendation = get_recommendation(df, coin)

        # Step 4: Redis (latest price)
        redis_data = redis_client.get(f"crypto:{coin}:latest")
        latest = json.loads(redis_data) if redis_data else {"price": "N/A", "timestamp": "N/A"}

    # -------------------- Output Display -----------------------

    st.subheader(f"📈 Latest {coin.capitalize()} Price")
    st.metric(label="Current Price (USD)", value=f"${latest['price']:.2f}", delta="Live from Redis")

    st.subheader("📊 Technical Indicator Trends (last 10 points)")
    st.dataframe(df, use_container_width=True)

    st.subheader("🧠 Gemini Recommendation")
    st.success(recommendation)

    # Plot
    st.subheader("📉 Price Chart")
    fig, ax = plt.subplots()
    ax.plot(df["timestamp"], df["price"], label="Price", color="blue")
    if "SMA_5" in df.columns:
        ax.plot(df["timestamp"], df["SMA_5"], label="SMA 5", color="orange")
    ax.legend()
    st.pyplot(fig)
