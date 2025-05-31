# Whale tracker algoritması (örnek) 
from utils.logger import get_logger
from utils.solana_api import get_wallet_transfers

class WhaleTracker:
    """
    Büyük cüzdan (whale) hareketlerini tespit eder ve hacim spike sinyali üretir.
    """
    def __init__(self, whale_addresses, min_spike=100000):
        self.whale_addresses = whale_addresses
        self.min_spike = min_spike
        self.logger = get_logger("WhaleTracker")

    def generate_signal(self, market_data):
        """
        market_data: {'token_address': str, ...}
        return: {'signal': int, 'confidence': float}
        """
        try:
            token_address = market_data.get('token_address')
            if not token_address:
                self.logger.warning("Token adresi eksik.")
                return {'signal': 0, 'confidence': 0.0}
            spikes = 0
            for whale in self.whale_addresses:
                transfers = get_wallet_transfers(whale, token_address)
                for t in transfers:
                    if t['amount'] >= self.min_spike:
                        spikes += 1
            if spikes > 0:
                self.logger.info(f"Whale hareketi tespit edildi! Spike sayısı: {spikes}")
                return {'signal': 1, 'confidence': min(1.0, spikes * 0.2)}
            else:
                return {'signal': 0, 'confidence': 0.5}
        except Exception as e:
            self.logger.error(f"WhaleTracker sinyal üretiminde hata: {e}")
            return {'signal': 0, 'confidence': 0.0} 