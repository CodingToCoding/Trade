# Neural network orchestrator (örnek) 
import os
import logging
from typing import Dict, Any, List
from utils.logger import get_logger
from utils.notifier import send_notification

class Orchestrator:
    """
    Tüm analiz algoritmalarından gelen sinyalleri toplayan, ağırlıklandıran,
    geçmiş performansı izleyen ve nihai karar veren merkezi kontrol sistemi.
    """
    def __init__(self, algo_names: List[str]):
        self.logger = get_logger("Orchestrator")
        self.algo_names = algo_names
        self.algo_weights = {name: 1.0 for name in algo_names}  # Başlangıçta eşit ağırlık
        self.algo_performance = {name: [] for name in algo_names}  # Doğruluk geçmişi
        self.decision_history = []
        self.confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", 0.8))

    def update_performance(self, algo: str, correct: bool):
        """Algoritmanın doğruluk geçmişini günceller."""
        if algo in self.algo_performance:
            self.algo_performance[algo].append(correct)
            # Son 100 tahmine göre ağırlık güncelle
            recent = self.algo_performance[algo][-100:]
            accuracy = sum(recent) / len(recent) if recent else 1.0
            self.algo_weights[algo] = max(0.1, accuracy)
            self.logger.info(f"{algo} güncel doğruluk: {accuracy:.2f}, yeni ağırlık: {self.algo_weights[algo]:.2f}")

    def aggregate_signals(self, signals: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Algoritmalardan gelen sinyalleri ağırlıklı olarak birleştirir.
        signals: {algo_name: {"signal": int, "confidence": float}}
        """
        weighted_sum = 0.0
        total_weight = 0.0
        for algo, data in signals.items():
            weight = self.algo_weights.get(algo, 1.0)
            signal = data.get("signal", 0)
            confidence = data.get("confidence", 1.0)
            weighted_sum += weight * signal * confidence
            total_weight += weight * confidence
        final_score = weighted_sum / total_weight if total_weight else 0.0
        decision = 1 if final_score >= self.confidence_threshold else -1 if final_score <= -self.confidence_threshold else 0
        self.logger.info(f"Toplu sinyal: {final_score:.2f}, Karar: {decision}")
        self.decision_history.append({"signals": signals, "score": final_score, "decision": decision})
        return {"score": final_score, "decision": decision}

    def notify(self, message: str):
        try:
            send_notification(message)
        except Exception as e:
            self.logger.error(f"Bildirim gönderilemedi: {e}")

# Orchestrator örneği oluşturmak için:
# orchestrator = Orchestrator(["rsi_ema_macd", "fibonacci", ...]) 