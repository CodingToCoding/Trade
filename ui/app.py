import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import os
from PIL import Image

# Backend ile bağlantı (REST API veya doğrudan import)
BACKEND_MODE = os.getenv("BACKEND_MODE", "import")  # "import" veya "api"

if BACKEND_MODE == "import":
    from brain.brain_pro import BrainPro
    brain = BrainPro([
        "rsi_ema_macd", "fibonacci", "whale_tracker", "spike_detector", "wallet_analyzer", "influencer_ring", "twitter_sentiment", "rug_checker"
    ])
    # Dummy fiyatlar (gerçekte API'den alınmalı)
    live_prices = {"PEPE": 0.0005, "DOGE": 0.07, "BONK": 0.00009}
else:
    API_URL = os.getenv("API_URL", "http://localhost:8000")

def get_brain_data(user_profile=None):
    if BACKEND_MODE == "import":
        if user_profile is None:
            user_profile = {"risk_level": "orta"}
        return brain.live_advice_report(live_prices, user_profile)
    else:
        resp = requests.post(f"{API_URL}/live_advice_report", json=user_profile or {"risk_level": "orta"})
        return resp.json()

# --- Streamlit UI Ayarları ---
st.set_page_config(page_title="Meme Coin AI Brain Dashboard", layout="wide", page_icon="🧠")
st.markdown("""
    <style>
    .main {background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); color: #fff;}
    .stApp {background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);}
    .block-container {padding-top: 2rem;}
    .css-1d391kg {background: rgba(30,41,59,0.8)!important; border-radius: 1.5rem;}
    .metric-label, .metric-value {color: #38bdf8!important;}
    .stDataFrame {background: #0f172a!important;}
    </style>
""", unsafe_allow_html=True)

st.title("🧠 Meme Coin AI Brain Dashboard")
st.caption("Solana Meme Coin Trading Bot — Railway & Streamlit Edition")

# --- Sidebar: Kullanıcı Profili ve Ayarlar ---
st.sidebar.header("Kullanıcı Profili ve Ayarlar")
risk_level = st.sidebar.selectbox("Risk Tercihi", ["düşük", "orta", "yüksek"], index=1)
user_profile = {"risk_level": risk_level}
refresh_rate = st.sidebar.slider("Yenileme Sıklığı (sn)", 5, 60, 15)

# --- Ana Panel ---
placeholder = st.empty()

while True:
    with placeholder.container():
        data = get_brain_data(user_profile)
        dashboard = data.get("dashboard", {})
        advice = data.get("advice", "")
        alerts = data.get("alerts", [])
        heatmap = pd.DataFrame(data.get("success_heatmap", []))
        portfolio = dashboard.get("portfolio", {})
        weights = dashboard.get("weights", {})
        decision_history = dashboard.get("decision_history", [])
        # --- Üstte canlı alarm/uyarı kutuları ---
        if alerts:
            for alert in alerts:
                st.error(f"⚠️ {alert}")
        # --- 3 Ana Kolon: Algoritma Ağırlıkları, Karar Geçmişi, Portföy ---
        col1, col2, col3 = st.columns([2,2,3])
        with col1:
            st.subheader("🔬 Algoritma Ağırlıkları")
            st.bar_chart(pd.Series(weights))
            st.metric("Ağırlık Ortalaması", f"{np.mean(list(weights.values())):.2f}")
            st.metric("Ağırlık Std", f"{np.std(list(weights.values())):.2f}")
        with col2:
            st.subheader("📈 Karar Geçmişi ve Başarı")
            if decision_history:
                st.line_chart([d["score"] for d in decision_history])
                success_rate = dashboard.get("success_rate", 0.0)
                st.metric("Başarı Oranı", f"{100*success_rate:.1f}%")
            if not heatmap.empty:
                st.subheader("Risk/Success Heatmap")
                st.dataframe(heatmap.style.background_gradient(cmap="coolwarm", subset=[c for c in heatmap.columns if c!="decision"]))
        with col3:
            st.subheader("💼 Portföy ve Pozisyonlar")
            st.metric("Toplam Değer", f"{portfolio.get('total_value', 0):,.2f}")
            st.metric("Bakiye", f"{portfolio.get('balance', 0):,.2f}")
            st.metric("Risk (Max Drawdown)", f"{100*portfolio.get('max_drawdown', 0):.1f}%")
            if portfolio.get("positions"):
                df_pos = pd.DataFrame(portfolio["positions"])
                st.dataframe(df_pos.style.background_gradient(cmap="Blues", subset=["pnl", "drawdown"]))
            else:
                st.info("Açık pozisyon yok.")
        # --- Coin Analizi ve Sentiment ---
        st.markdown("---")
        st.subheader("🔥 Canlı Coin Analizi ve Sentiment")
        # Dummy coinler (gerçekte backend'den alınmalı)
        coins = [
            {"symbol": "PEPE", "price": 0.0005, "liquidity": 50000, "score": 0.7, "sentiment": "Bullish"},
            {"symbol": "DOGE", "price": 0.07, "liquidity": 80000, "score": 0.2, "sentiment": "Neutral"},
            {"symbol": "BONK", "price": 0.00009, "liquidity": 30000, "score": -0.4, "sentiment": "Bearish"},
        ]
        df_coins = pd.DataFrame(coins)
        st.dataframe(df_coins.style.background_gradient(cmap="coolwarm", subset=["score"]))
        # --- Kişiselleştirilmiş Öneri ve Rapor ---
        st.markdown("---")
        st.subheader("🤖 AI Beyin Önerisi ve Raporu")
        st.success(f"{advice}")
        # --- Footer ---
        st.markdown("<center><h3 style='color:#38bdf8;'>AI Brain Railway Edition • Modern & Responsive • Powered by Streamlit</h3></center>", unsafe_allow_html=True)
    time.sleep(refresh_rate) 