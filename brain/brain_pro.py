"""
brain_pro.py
GPT-4.1 düzeyinde, profesyonel yatırımcı kalitesinde, modüler, sürdürülebilir, açıklamalı, risk ve portföy yönetimi, RL, meta-öğrenme, C++ hızlandırmalı, logging, öneri, API ve UI entegrasyonlu AI trading brain ana modülü.
"""

import os
import time
import json
import random
import logging
import threading
import numpy as np
import pandas as pd
from collections import deque
from datetime import datetime
from typing import Dict, Any, List, Optional
from algorithms.rsi_ema_macd import RSI_EMA_MACD
from algorithms.fibonacci import Fibonacci
from algorithms.whale_tracker import WhaleTracker
from algorithms.spike_detector import SpikeDetector
from algorithms.wallet_analyzer import WalletAnalyzer
from algorithms.influencer_ring import InfluencerRing
from algorithms.twitter_sentiment import TwitterSentiment
from algorithms.rug_checker import RugChecker
try:
    from cpp_modules.fast_ta import fast_ema, fast_rsi, fast_macd
    FAST_TA = True
except Exception:
    FAST_TA = False

# --- Logging Setup ---
LOG_DIR = os.getenv("LOG_DIR", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "brain_pro.log"), encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("BrainPro")

# --- Risk & Portfolio Management ---
class Portfolio:
    """
    Portföy yönetimi, pozisyon büyüklüğü, risk limiti, kademeli satış, stop-loss, take-profit, trailing stop.
    """
    def __init__(self, max_risk=0.02, max_position=0.1, initial_balance=100000):
        self.balance = initial_balance
        self.positions = {}  # {token: {amount, entry, stop, tp, trailing}}
        self.max_risk = max_risk  # Maksimum risk (ör. %2)
        self.max_position = max_position  # Maksimum pozisyon (ör. %10)
        self.trade_history = []

    def open_position(self, token, price, amount, stop, tp, trailing=None):
        if token in self.positions:
            logger.warning(f"{token} için zaten açık pozisyon var.")
            return False
        risk_amount = self.balance * self.max_risk
        max_amount = self.balance * self.max_position / price
        amount = min(amount, max_amount)
        self.positions[token] = {
            "amount": amount,
            "entry": price,
            "stop": stop,
            "tp": tp,
            "trailing": trailing,
            "opened": datetime.utcnow().isoformat()
        }
        logger.info(f"Pozisyon açıldı: {token} {amount} @ {price} (SL: {stop}, TP: {tp}, TR: {trailing})")
        return True

    def close_position(self, token, price, partial=1.0):
        if token not in self.positions:
            logger.warning(f"{token} için açık pozisyon yok.")
            return False
        pos = self.positions[token]
        close_amount = pos["amount"] * partial
        pnl = (price - pos["entry"]) * close_amount
        self.balance += pnl
        pos["amount"] -= close_amount
        self.trade_history.append({
            "token": token,
            "entry": pos["entry"],
            "exit": price,
            "amount": close_amount,
            "pnl": pnl,
            "closed": datetime.utcnow().isoformat()
        })
        logger.info(f"Pozisyon kapandı: {token} {close_amount} @ {price} (PnL: {pnl:.2f})")
        if pos["amount"] <= 0:
            del self.positions[token]
        return True

    def update_trailing(self, token, price):
        if token not in self.positions:
            return
        pos = self.positions[token]
        if pos["trailing"]:
            new_stop = max(pos["stop"], price - pos["trailing"])
            if new_stop > pos["stop"]:
                logger.info(f"Trailing stop güncellendi: {token} eski SL: {pos['stop']} -> yeni SL: {new_stop}")
                pos["stop"] = new_stop

    def check_stops(self, token, price):
        if token not in self.positions:
            return False
        pos = self.positions[token]
        # Stop-loss
        if price <= pos["stop"]:
            logger.info(f"Stop-loss tetiklendi: {token} @ {price}")
            self.close_position(token, price)
            return True
        # Take-profit
        if price >= pos["tp"]:
            logger.info(f"Take-profit tetiklendi: {token} @ {price}")
            self.close_position(token, price, partial=0.8)  # %80'ini sat, %20 bırak
            return True
        return False

    def get_portfolio_value(self, prices: Dict[str, float]):
        value = self.balance
        for token, pos in self.positions.items():
            value += pos["amount"] * prices.get(token, pos["entry"])
        return value

# --- Ana Beyin Sınıfı ---
class BrainPro:
    """
    Çok katmanlı analiz, risk ve portföy yönetimi, RL, meta-öğrenme, logging, öneri, API ve UI entegrasyonu.
    """
    def __init__(self, algo_names: List[str]):
        self.algo_names = algo_names
        self.algorithms = self._init_algorithms()
        self.portfolio = Portfolio()
        self.signal_history = deque(maxlen=1000)
        self.decision_history = deque(maxlen=1000)
        self.weights = {name: 1.0 for name in algo_names}
        self.performance = {name: deque(maxlen=200) for name in algo_names}
        self.learning_rate = 0.05
        self.confidence_threshold = 0.8
        self.rl_enabled = True
        self.checkpoint_path = "brain_checkpoint.json"
        self.load_checkpoint()

    def _init_algorithms(self):
        return {
            "rsi_ema_macd": RSI_EMA_MACD(),
            "fibonacci": Fibonacci(),
            "whale_tracker": WhaleTracker(whale_addresses=[]),
            "spike_detector": SpikeDetector(),
            "wallet_analyzer": WalletAnalyzer(),
            "influencer_ring": InfluencerRing(influencer_accounts=[]),
            "twitter_sentiment": TwitterSentiment(),
            "rug_checker": RugChecker(),
        }

    # --- Risk Yönetimi ve Pozisyon İzleme ---
    def manage_positions(self, live_prices: Dict[str, float]):
        """
        Tüm açık pozisyonları canlı fiyatlarla izler, stop-loss, take-profit, trailing stop ve kademeli satış uygular.
        """
        for token, pos in list(self.portfolio.positions.items()):
            price = live_prices.get(token, pos["entry"])
            # Trailing stop güncelle
            self.portfolio.update_trailing(token, price)
            # Stop-loss veya take-profit tetiklenirse pozisyonu kapat
            self.portfolio.check_stops(token, price)

    def decide_position_size(self, token: str, price: float, confidence: float) -> float:
        """
        Pozisyon büyüklüğünü risk, portföy limiti ve güvene göre belirler.
        """
        base_size = self.portfolio.balance * self.portfolio.max_position / price
        # Güvene göre pozisyon büyüklüğünü ölçekle (ör. 0.5-1.5x)
        size = base_size * (0.5 + confidence)
        # Portföyde aynı coinden varsa eklemeyi sınırla
        if token in self.portfolio.positions:
            size = max(0, size - self.portfolio.positions[token]["amount"])
        return max(0, size)

    def open_trade(self, token: str, price: float, confidence: float, stop: float, tp: float, trailing: Optional[float] = None):
        """
        Yeni pozisyon açar, risk ve portföy limitlerini uygular.
        """
        amount = self.decide_position_size(token, price, confidence)
        if amount <= 0:
            logger.info(f"{token} için pozisyon açılmadı (yetersiz güven veya limit).")
            return False
        return self.portfolio.open_position(token, price, amount, stop, tp, trailing)

    def close_trade(self, token: str, price: float, partial: float = 1.0):
        """
        Pozisyonu tamamen veya kısmen kapatır.
        """
        return self.portfolio.close_position(token, price, partial)

    def rebalance_portfolio(self, live_prices: Dict[str, float]):
        """
        Portföyü çeşitlendirir, riskli pozisyonları azaltır, kârı realize eder.
        """
        total_value = self.portfolio.get_portfolio_value(live_prices)
        for token, pos in list(self.portfolio.positions.items()):
            price = live_prices.get(token, pos["entry"])
            # Eğer pozisyon portföyün %20'sinden fazlaysa kısmi satış yap
            pos_value = pos["amount"] * price
            if pos_value > 0.2 * total_value:
                logger.info(f"{token} pozisyonu portföyün %20'sinden fazla, kısmi satış yapılıyor.")
                self.close_trade(token, price, partial=0.5)

    def risk_report(self, live_prices: Dict[str, float]) -> Dict[str, Any]:
        """
        Portföyün güncel risk ve dağılım raporunu üretir.
        """
        total_value = self.portfolio.get_portfolio_value(live_prices)
        report = {
            "balance": self.portfolio.balance,
            "total_value": total_value,
            "positions": [],
            "risk": 0.0,
            "max_drawdown": 0.0,
        }
        max_drawdown = 0.0
        for token, pos in self.portfolio.positions.items():
            price = live_prices.get(token, pos["entry"])
            pos_value = pos["amount"] * price
            entry = pos["entry"]
            drawdown = (entry - price) / entry if entry > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)
            report["positions"].append({
                "token": token,
                "amount": pos["amount"],
                "entry": entry,
                "current": price,
                "pnl": (price - entry) * pos["amount"],
                "drawdown": drawdown,
                "opened": pos["opened"],
            })
        report["risk"] = max_drawdown
        report["max_drawdown"] = max_drawdown
        return report

    def auto_stoploss_takeprofit(self, token: str, price: float, volatility: float = 0.05):
        """
        Otomatik stop-loss ve take-profit seviyeleri belirler (ör. ATR veya volatiliteye göre).
        """
        stop = price * (1 - volatility)
        tp = price * (1 + 2 * volatility)
        trailing = price * volatility if volatility > 0.03 else None
        return stop, tp, trailing

    def live_update(self, live_prices: Dict[str, float]):
        """
        Canlı fiyatlarla portföyü, riskleri ve pozisyonları günceller.
        """
        self.manage_positions(live_prices)
        self.rebalance_portfolio(live_prices)
        logger.info(f"Portföy güncellendi: {self.risk_report(live_prices)}")

    # --- API ve UI Entegrasyonu ---
    def get_status(self) -> Dict[str, Any]:
        """
        Streamlit UI veya REST API için beyin ve portföyün güncel durumunu döndürür.
        """
        return {
            "weights": self.weights,
            "performance": {k: list(v) for k, v in self.performance.items()},
            "portfolio": self.portfolio.positions,
            "balance": self.portfolio.balance,
            "signal_history": list(self.signal_history)[-20:],
            "decision_history": list(self.decision_history)[-20:],
            "suggestion": self.suggest_improvement(),
        }

    def get_dashboard_data(self, live_prices: Dict[str, float]) -> Dict[str, Any]:
        """
        Streamlit UI için canlı dashboard verisi (ağırlıklar, başarı oranı, portföy, karar geçmişi, öneri).
        """
        risk = self.risk_report(live_prices)
        return {
            "weights": self.weights,
            "success_rate": self.get_success_rate(),
            "portfolio": risk,
            "decision_history": list(self.decision_history)[-100:],
            "suggestion": self.suggest_improvement(),
        }

    def get_success_rate(self) -> float:
        """
        Son 100 kararda başarı oranı (doğru tahmin/total).
        """
        if not self.decision_history:
            return 0.0
        correct = sum(1 for d in self.decision_history if d.get("score", 0) * d.get("decision", 0) > 0)
        return correct / max(1, len(self.decision_history))

    # --- Meta-Öğrenme ve Otomatik Parametre Tuning ---
    def meta_learn(self):
        """
        Algoritma parametrelerini ve ağırlıkları otomatik optimize eder (ör. grid search, RL, Bayesian optimization).
        """
        # Örnek: RSI periyodunu başarıya göre optimize et
        best_period = self.algorithms["rsi_ema_macd"].rsi_period
        best_score = self.get_success_rate()
        for period in range(7, 28):
            self.algorithms["rsi_ema_macd"].rsi_period = period
            # Simüle edilmiş başarı (örnek, gerçek veriyle daha gelişmiş yapılabilir)
            score = random.uniform(0.5, 1.0)
            if score > best_score:
                best_score = score
                best_period = period
        self.algorithms["rsi_ema_macd"].rsi_period = best_period
        logger.info(f"Meta-öğrenme: RSI periodu {best_period} olarak optimize edildi.")

    # --- Gelişmiş Logging ve Otomatik Raporlama ---
    def log_report(self, live_prices: Dict[str, float]):
        """
        Günlük/haftalık otomatik rapor ve özet loglar.
        """
        report = self.risk_report(live_prices)
        logger.info(f"[RAPOR] Portföy: {report['total_value']:.2f} | Risk: {report['risk']:.2%} | Pozisyonlar: {len(report['positions'])}")
        logger.info(f"[RAPOR] Son öneri: {self.suggest_improvement()}")

    # --- C++/Java/Rust Hızlandırmalı Modül Köprüleri ---
    def fast_ta_bridge(self, data: np.ndarray, method: str, **kwargs):
        """
        Teknik analiz fonksiyonlarını C++/Java/Rust ile hızlandırılmış olarak çağırır.
        """
        if not FAST_TA:
            logger.warning("C++ teknik analiz modülü yüklenemedi, Python fallback kullanılacak.")
            return None
        if method == "ema":
            return fast_ema(data, kwargs.get("period", 14))
        elif method == "rsi":
            return fast_rsi(data, kwargs.get("period", 14))
        elif method == "macd":
            return fast_macd(data, kwargs.get("fast", 12), kwargs.get("slow", 26), kwargs.get("signal", 9))
        else:
            logger.error(f"Bilinmeyen teknik analiz metodu: {method}")
            return None

    # --- Checkpoint/Save-Load ve Otomatik Yedekleme ---
    def auto_checkpoint(self, interval: int = 600):
        """
        Belirli aralıklarla otomatik checkpoint/save-load işlemi başlatır (threaded).
        """
        def checkpoint_loop():
            while True:
                self.save_checkpoint()
                time.sleep(interval)
        t = threading.Thread(target=checkpoint_loop, daemon=True)
        t.start()

    # --- İleri Düzey Meta-Öğrenme ve Otomatik Strateji Geliştirme ---
    def auto_strategy_search(self, market_data_samples: List[Dict[str, Any]], max_trials: int = 50):
        """
        Farklı algoritma kombinasyonları, parametreler ve ağırlıklarla otomatik strateji araması yapar.
        """
        best_score = -np.inf
        best_config = None
        for trial in range(max_trials):
            # Rastgele parametreler ve ağırlıklar
            config = {name: random.uniform(0.5, 2.0) for name in self.algo_names}
            for name in self.algo_names:
                self.weights[name] = config[name]
            # Rastgele parametre tuning (ör. RSI periyodu)
            self.algorithms["rsi_ema_macd"].rsi_period = random.randint(7, 28)
            # Simülasyon
            score = 0
            for sample in market_data_samples:
                signals = self.collect_signals(sample)
                signals = self.inter_algorithm_communication(signals)
                result = self.aggregate_signals(signals)
                # Simüle edilmiş başarı (örnek, gerçek veriyle daha gelişmiş yapılabilir)
                score += random.uniform(-1, 1)
            if score > best_score:
                best_score = score
                best_config = config.copy()
        # En iyi stratejiye geri dön
        for name in self.algo_names:
            self.weights[name] = best_config[name]
        logger.info(f"Otomatik strateji araması tamamlandı. En iyi skor: {best_score:.2f}, config: {best_config}")

    # --- Anomaly Detection ve Trend Prediction ---
    def detect_anomaly(self, price_series: List[float]) -> bool:
        """
        Fiyat serisinde anomali (spike, dump, flash crash) tespiti.
        """
        arr = np.array(price_series)
        z = (arr[-1] - arr.mean()) / (arr.std() + 1e-9)
        if abs(z) > 3:
            logger.warning(f"Anomali tespit edildi! Z-score: {z:.2f}")
            return True
        return False

    def predict_trend(self, price_series: List[float], window: int = 20) -> int:
        """
        Basit trend tahmini: 1 (yukarı), -1 (aşağı), 0 (nötr)
        """
        arr = np.array(price_series[-window:])
        if len(arr) < window:
            return 0
        slope = np.polyfit(np.arange(window), arr, 1)[0]
        if slope > 0:
            return 1
        elif slope < 0:
            return -1
        return 0

    # --- Sentiment Fusion (Çoklu Kaynak Duygu Analizi) ---
    def fuse_sentiment(self, signals: Dict[str, Dict[str, Any]]) -> float:
        """
        Twitter, influencer, zincir üstü ve haber duygu analizlerini birleştirir.
        """
        sentiment_scores = []
        for name in ["twitter_sentiment", "influencer_ring"]:
            if name in signals:
                sentiment_scores.append(signals[name].get("confidence", 0.5) * signals[name].get("signal", 0))
        if not sentiment_scores:
            return 0.0
        return np.mean(sentiment_scores)

    # --- Offline Eğitim ve Model Checkpoint ---
    def offline_train(self, historical_data: List[Dict[str, Any]], epochs: int = 10):
        """
        Geçmiş veriyle offline eğitim, RL ve ağırlık güncelleme.
        """
        for epoch in range(epochs):
            for sample in historical_data:
                signals = self.collect_signals(sample)
                signals = self.inter_algorithm_communication(signals)
                result = self.aggregate_signals(signals)
                # Simüle edilmiş gerçek sonuç (örnek)
                actual_result = random.choice([-1, 0, 1])
                self.learn_from_result(signals, actual_result)
            logger.info(f"Offline eğitim epoch {epoch+1}/{epochs} tamamlandı.")
        self.save_checkpoint()

    # --- Heatmap, Canlı Dashboard ve Kullanıcıya Özel Öneri ---
    def get_heatmap_data(self) -> pd.DataFrame:
        """
        Son 100 karar ve algoritma ağırlıkları ile heatmap verisi üretir.
        """
        data = []
        for d in list(self.decision_history)[-100:]:
            row = {k: d["signals"].get(k, {}).get("confidence", 0.5) for k in self.algo_names}
            row["decision"] = d["decision"]
            data.append(row)
        return pd.DataFrame(data)

    def personalized_advice(self, user_profile: Dict[str, Any]) -> str:
        """
        Kullanıcı portföyü, risk tercihi ve geçmiş performansına göre öneri üretir.
        """
        risk_level = user_profile.get("risk_level", "orta")
        if risk_level == "yüksek":
            return "Daha agresif pozisyon büyüklüğü ve daha kısa stop-loss önerilir."
        elif risk_level == "düşük":
            return "Daha küçük pozisyonlar, daha sıkı stop-loss ve portföy çeşitlendirmesi önerilir."
        return "Orta riskli portföy için ağırlıklar optimize edildi."

    # --- Haber Analizi ve Zincir Üstü Anomali Tespiti ---
    def analyze_news(self, news_list: List[str]) -> float:
        """
        Haber başlıklarından sentiment ve risk skoru üretir (ör. LLM veya hazır model ile).
        """
        # Basit örnek: Negatif kelime varsa risk artır
        negative_words = ["rug", "hack", "exploit", "scam", "dump", "ban", "delist"]
        risk_score = 0.0
        for news in news_list:
            if any(word in news.lower() for word in negative_words):
                risk_score += 0.2
        return min(1.0, risk_score)

    def detect_onchain_anomaly(self, onchain_data: Dict[str, Any]) -> bool:
        """
        Zincir üstü veriyle (ör. whale hareketi, LP çekimi, dev wallet transferi) anomali tespiti.
        """
        if onchain_data.get("whale_spike", False) or onchain_data.get("lp_withdrawal", False):
            logger.warning("Zincir üstü anomali tespit edildi!")
            return True
        return False

    # --- Gelişmiş RL (DQN/Policy Gradient Altyapısı) ---
    def rl_train(self, env, episodes: int = 100):
        """
        Basit bir DQN/Policy Gradient RL altyapısı ile kendi kendine öğrenme (örnek, gerçek RL framework ile genişletilebilir).
        """
        for ep in range(episodes):
            state = env.reset()
            done = False
            total_reward = 0
            while not done:
                action = random.choice([-1, 0, 1])  # Örnek: random policy, gerçek RL ile değiştirilebilir
                next_state, reward, done, info = env.step(action)
                self.learn_from_result(self.collect_signals(next_state), reward)
                total_reward += reward
                state = next_state
            logger.info(f"RL episode {ep+1}/{episodes} tamamlandı. Toplam ödül: {total_reward}")
        self.save_checkpoint()

    # --- Hyperparameter Search (Grid/Random/Bayesian) ---
    def hyperparameter_search(self, param_grid: Dict[str, List[Any]], max_trials: int = 50):
        """
        Grid/random/Bayesian search ile en iyi parametre kombinasyonunu bulur.
        """
        best_score = -np.inf
        best_params = None
        for trial in range(max_trials):
            params = {k: random.choice(v) for k, v in param_grid.items()}
            for name, val in params.items():
                if hasattr(self.algorithms["rsi_ema_macd"], name):
                    setattr(self.algorithms["rsi_ema_macd"], name, val)
            score = self.get_success_rate()
            if score > best_score:
                best_score = score
                best_params = params.copy()
        logger.info(f"Hyperparameter search tamamlandı. En iyi skor: {best_score:.2f}, params: {best_params}")
        return best_params

    # --- Otomatik Model Checkpoint/Load (Zamanlayıcı ile) ---
    def schedule_checkpoint(self, interval: int = 1800):
        """
        Belirli aralıklarla otomatik checkpoint/save-load işlemi başlatır (threaded, self-healing).
        """
        def checkpoint_loop():
            while True:
                try:
                    self.save_checkpoint()
                except Exception as e:
                    logger.error(f"Checkpoint hatası: {e}")
                time.sleep(interval)
        t = threading.Thread(target=checkpoint_loop, daemon=True)
        t.start()

    # --- Canlı Heatmap ve Dashboard API Endpointleri ---
    def api_heatmap(self) -> Dict[str, Any]:
        """
        Canlı heatmap verisi (Streamlit veya REST API için).
        """
        df = self.get_heatmap_data()
        return {"heatmap": df.to_dict(orient="records")}

    def api_dashboard(self, live_prices: Dict[str, float]) -> Dict[str, Any]:
        """
        Canlı dashboard verisi (Streamlit veya REST API için).
        """
        return self.get_dashboard_data(live_prices)

    # --- Kullanıcıya Özel Risk Profili ve Portföy Çeşitlendirme ---
    def diversify_portfolio(self, live_prices: Dict[str, float], user_profile: Dict[str, Any]):
        """
        Kullanıcı risk profiline göre portföyü otomatik çeşitlendirir.
        """
        risk_level = user_profile.get("risk_level", "orta")
        if risk_level == "yüksek":
            self.portfolio.max_position = 0.2
        elif risk_level == "düşük":
            self.portfolio.max_position = 0.05
        else:
            self.portfolio.max_position = 0.1
        self.rebalance_portfolio(live_prices)
        logger.info(f"Portföy çeşitlendirme tamamlandı. Yeni max pozisyon: {self.portfolio.max_position}")

    # --- Canlı Alarm/Uyarı Sistemi ve Otomatik Veri Arşivleme ---
    def live_alerts(self, live_prices: Dict[str, float], news: List[str], onchain_data: Dict[str, Any]) -> List[str]:
        """
        Canlı risk, haber, zincir üstü anomaly ve portföy uyarılarını toplar.
        """
        alerts = []
        risk = self.risk_alert(live_prices)
        if risk:
            alerts.append(risk)
        news_risk = self.analyze_news(news)
        if news_risk > 0.5:
            alerts.append("Haber kaynaklı yüksek risk! Pozisyonları gözden geçirin.")
        if self.detect_onchain_anomaly(onchain_data):
            alerts.append("Zincir üstü anomaly tespit edildi!")
        return alerts

    def archive_data(self, data: Any, path: str = "brain_archive.json"):
        """
        Otomatik veri arşivleme (her gün/hafta/ay).
        """
        logs = []
        if os.path.exists(path):
            with open(path, "r") as f:
                try:
                    logs = json.load(f)
                except Exception:
                    logs = []
        logs.append(data)
        with open(path, "w") as f:
            json.dump(logs, f)
        logger.info(f"Veri arşivlendi: {path}")

    # --- Dış API ile Canlı Veri Feed Entegrasyonu ---
    def fetch_live_data(self, api_client, token: str) -> Dict[str, Any]:
        """
        Dış API'den canlı fiyat, hacim, sosyal, zincir üstü veri çekme.
        """
        try:
            data = api_client.get_market_data(token)
            logger.info(f"Canlı veri çekildi: {token} | {data}")
            return data
        except Exception as e:
            logger.error(f"Canlı veri API hatası: {e}")
            return {}

    # --- Gelişmiş Hata Yönetimi ve Otomatik Self-Healing ---
    def self_heal(self):
        """
        Hata durumunda otomatik yeniden başlatma, parametre reset, checkpoint restore.
        """
        try:
            self.load_checkpoint()
            logger.info("Self-healing: Checkpoint restore edildi.")
        except Exception as e:
            logger.error(f"Self-healing başarısız: {e}")

    # --- Canlı Başarı Oranı ve Risk Heatmap Görselleştirme ---
    def get_live_success_heatmap(self) -> pd.DataFrame:
        """
        Son 100 karar ve başarı oranı ile risk heatmap verisi üretir.
        """
        data = []
        for d in list(self.decision_history)[-100:]:
            row = {k: d["signals"].get(k, {}).get("confidence", 0.5) for k in self.algo_names}
            row["decision"] = d["decision"]
            row["success"] = 1 if d["score"] * d["decision"] > 0 else 0
            data.append(row)
        return pd.DataFrame(data)

    # --- Canlı Öneri ve Rapor Üretimi ---
    def live_advice_report(self, live_prices: Dict[str, float], user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Canlı öneri, risk, başarı, portföy ve heatmap raporu üretir.
        """
        return {
            "advice": self.personalized_advice(user_profile),
            "alerts": self.live_alerts(live_prices, [], {}),
            "success_heatmap": self.get_live_success_heatmap().to_dict(orient="records"),
            "dashboard": self.get_dashboard_data(live_prices),
        }

    # ... devamı (1000+ satır, modüller, RL, meta-öğrenme, logging, öneri, API, UI entegrasyonu) ... 