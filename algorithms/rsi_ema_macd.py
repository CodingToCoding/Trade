# RSI, EMA ve MACD algoritması (örnek) 
import numpy as np
import pandas as pd
import logging
from utils.logger import get_logger
try:
    from brain.fast_ta_bridge import fast_ema, fast_rsi, fast_macd
    FAST_TA = True
except Exception:
    FAST_TA = False

class RSI_EMA_MACD:
    """
    RSI, EMA ve MACD kombinasyonuna göre pozisyon sinyali üretir.
    """
    def __init__(self, rsi_period=14, ema_period=21, macd_fast=12, macd_slow=26, macd_signal=9):
        self.rsi_period = rsi_period
        self.ema_period = ema_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.logger = get_logger("RSI_EMA_MACD")

    def generate_signal(self, market_data):
        """
        market_data: {'close': [float], ...}
        return: {'signal': int, 'confidence': float}
        """
        try:
            closes = np.array(market_data['close'], dtype=float)
            if len(closes) < max(self.rsi_period, self.ema_period, self.macd_slow, self.macd_signal) + 1:
                self.logger.warning("Yetersiz veri: Teknik analiz için daha fazla kapanış fiyatı gerekli.")
                return {'signal': 0, 'confidence': 0.0}
            if FAST_TA:
                rsi = fast_rsi(closes, self.rsi_period)
                ema = fast_ema(closes, self.ema_period)
                macd = fast_macd(closes, self.macd_fast, self.macd_slow, self.macd_signal)
                macd_signal_val = fast_ema(closes, self.macd_signal)
            else:
                rsi = self._rsi(closes)[-1]
                ema = self._ema(closes)[-1]
                macd, macd_signal_val = self._macd(closes)
                macd = macd[-1]
                macd_signal_val = macd_signal_val[-1]
            signal = 0
            confidence = 0.0
            if rsi < 30 and macd > macd_signal_val and closes[-1] > ema:
                signal = 1
                confidence = min(1.0, (30 - rsi) / 30 + (macd - macd_signal_val))
            elif rsi > 70 and macd < macd_signal_val and closes[-1] < ema:
                signal = -1
                confidence = min(1.0, (rsi - 70) / 30 + (macd_signal_val - macd))
            else:
                signal = 0
                confidence = 0.5
            self.logger.info(f"RSI: {rsi:.2f}, EMA: {ema:.2f}, MACD: {macd:.2f}, MACD_Signal: {macd_signal_val:.2f}, Sinyal: {signal}, Güven: {confidence:.2f}")
            return {'signal': signal, 'confidence': confidence}
        except Exception as e:
            self.logger.error(f"RSI_EMA_MACD sinyal üretiminde hata: {e}")
            return {'signal': 0, 'confidence': 0.0}

    def _rsi(self, closes):
        delta = np.diff(closes)
        up = np.where(delta > 0, delta, 0)
        down = np.where(delta < 0, -delta, 0)
        roll_up = pd.Series(up).rolling(self.rsi_period).mean()
        roll_down = pd.Series(down).rolling(self.rsi_period).mean()
        rs = roll_up / (roll_down + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        rsi = np.concatenate([np.full(self.rsi_period, np.nan), rsi[self.rsi_period-1:]])
        return rsi

    def _ema(self, closes):
        return pd.Series(closes).ewm(span=self.ema_period, adjust=False).mean().values

    def _macd(self, closes):
        ema_fast = pd.Series(closes).ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = pd.Series(closes).ewm(span=self.macd_slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=self.macd_signal, adjust=False).mean()
        return macd.values, macd_signal.values 