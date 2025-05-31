# Spike detector algoritması (örnek) 
import numpy as np
from utils.logger import get_logger

class SpikeDetector:
    """
    Anormal hacim ve fiyat hareketlerini analiz eder.
    """
    def __init__(self, window=20, threshold=3.0):
        self.window = window
        self.threshold = threshold
        self.logger = get_logger("SpikeDetector")

    def generate_signal(self, market_data):
        """
        market_data: {'volume': [float], 'close': [float], ...}
        return: {'signal': int, 'confidence': float}
        """
        try:
            volumes = np.array(market_data['volume'], dtype=float)
            closes = np.array(market_data['close'], dtype=float)
            if len(volumes) < self.window or len(closes) < self.window:
                self.logger.warning("Yetersiz veri: Spike analizi için daha fazla veri gerekli.")
                return {'signal': 0, 'confidence': 0.0}
            vol_mean = np.mean(volumes[-self.window-1:-1])
            vol_std = np.std(volumes[-self.window-1:-1])
            price_mean = np.mean(closes[-self.window-1:-1])
            price_std = np.std(closes[-self.window-1:-1])
            vol_z = (volumes[-1] - vol_mean) / (vol_std + 1e-9)
            price_z = (closes[-1] - price_mean) / (price_std + 1e-9)
            signal = 0
            confidence = 0.0
            if vol_z > self.threshold and price_z > self.threshold:
                signal = 1  # Pozitif spike
                confidence = min(1.0, (vol_z + price_z) / (2 * self.threshold))
            elif vol_z < -self.threshold and price_z < -self.threshold:
                signal = -1  # Negatif spike
                confidence = min(1.0, (-vol_z - price_z) / (2 * self.threshold))
            else:
                signal = 0
                confidence = 0.5
            self.logger.info(f"Spike analizi: vol_z={vol_z:.2f}, price_z={price_z:.2f}, Sinyal: {signal}, Güven: {confidence}")
            return {'signal': signal, 'confidence': confidence}
        except Exception as e:
            self.logger.error(f"SpikeDetector sinyal üretiminde hata: {e}")
            return {'signal': 0, 'confidence': 0.0} 