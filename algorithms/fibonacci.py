# Fibonacci algoritması (örnek) 
import numpy as np
import pandas as pd
from utils.logger import get_logger

class Fibonacci:
    """
    Fibonacci seviyelerine göre dönüş ya da breakout tespiti yapar.
    """
    def __init__(self, lookback=100):
        self.lookback = lookback
        self.logger = get_logger("Fibonacci")

    def generate_signal(self, market_data):
        """
        market_data: {'close': [float], ...}
        return: {'signal': int, 'confidence': float}
        """
        try:
            closes = np.array(market_data['close'], dtype=float)
            if len(closes) < self.lookback:
                self.logger.warning("Yetersiz veri: Fibonacci için daha fazla kapanış fiyatı gerekli.")
                return {'signal': 0, 'confidence': 0.0}
            recent = closes[-self.lookback:]
            high = np.max(recent)
            low = np.min(recent)
            diff = high - low
            levels = [high - diff * r for r in [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]]
            price = closes[-1]
            signal = 0
            confidence = 0.0
            # Dönüş veya breakout tespiti
            if price > levels[1] and price < levels[2]:
                signal = 1  # Destekten yukarı dönüş
                confidence = 0.7
            elif price < levels[5] and price > levels[4]:
                signal = -1  # Dirençten aşağı dönüş
                confidence = 0.7
            elif price > levels[0]:
                signal = 1  # ATH breakout
                confidence = 0.9
            elif price < levels[-1]:
                signal = -1  # ATL breakdown
                confidence = 0.9
            else:
                signal = 0
                confidence = 0.5
            self.logger.info(f"Fibonacci seviyeleri: {levels}, Fiyat: {price}, Sinyal: {signal}, Güven: {confidence}")
            return {'signal': signal, 'confidence': confidence}
        except Exception as e:
            self.logger.error(f"Fibonacci sinyal üretiminde hata: {e}")
            return {'signal': 0, 'confidence': 0.0} 