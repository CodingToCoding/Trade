# Sinyal üretim ağı (örnek) 
import os
import logging
from typing import Dict, Any
from core.orchestrator import Orchestrator
from utils.logger import get_logger
from utils.notifier import send_notification

class DecisionEngine:
    """
    Algoritmalardan sinyal toplar, orchestrator ile iletişim kurar,
    karar üretir ve risk yönetimi uygular.
    """
    def __init__(self, algorithms: Dict[str, Any]):
        self.logger = get_logger("DecisionEngine")
        self.algorithms = algorithms
        self.orchestrator = Orchestrator(list(algorithms.keys()))
        self.max_position_size = float(os.getenv("MAX_POSITION_SIZE", 1000))
        self.min_liquidity = float(os.getenv("MIN_LIQUIDITY", 10000))
        self.shadow_mode = os.getenv("SHADOW_MODE", "false").lower() == "true"

    def collect_signals(self, market_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Tüm algoritmalardan sinyal toplar."""
        signals = {}
        for name, algo in self.algorithms.items():
            try:
                result = algo.generate_signal(market_data)
                signals[name] = result
                self.logger.info(f"{name} sinyali: {result}")
            except Exception as e:
                self.logger.error(f"{name} algoritmasında hata: {e}")
        return signals

    def evaluate(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sinyalleri toplar, orchestrator ile nihai kararı üretir ve risk yönetimi uygular.
        """
        signals = self.collect_signals(market_data)
        # Risk ve güvenlik kontrolleri
        if not self._risk_checks(market_data):
            self.logger.warning("Risk kontrolleri geçilemedi. İşlem yapılmayacak.")
            return {"decision": 0, "reason": "Risk check failed"}
        result = self.orchestrator.aggregate_signals(signals)
        if self.shadow_mode:
            self.logger.info("Shadow mode aktif, işlem simüle edilecek.")
            send_notification(f"[SHADOW] Karar: {result}")
        else:
            send_notification(f"Karar: {result}")
        return result

    def _risk_checks(self, market_data: Dict[str, Any]) -> bool:
        """
        Likidite, rug pull, pozisyon büyüklüğü gibi risk kontrollerini uygular.
        """
        liquidity = market_data.get("liquidity", 0)
        rug_risk = market_data.get("rug_risk", False)
        position_size = market_data.get("position_size", 0)
        if liquidity < self.min_liquidity:
            self.logger.warning(f"Likidite çok düşük: {liquidity}")
            return False
        if rug_risk:
            self.logger.warning("Rug pull riski tespit edildi!")
            return False
        if position_size > self.max_position_size:
            self.logger.warning(f"Pozisyon büyüklüğü limiti aşıldı: {position_size}")
            return False
        return True 