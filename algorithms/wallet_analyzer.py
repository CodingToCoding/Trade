# Wallet analyzer algoritması (örnek) 
from utils.logger import get_logger
from utils.solana_api import get_token_holders, get_lp_info

class WalletAnalyzer:
    """
    Token owner wallet'ları ve LP davranışını inceler (rug pull risk kontrolü).
    """
    def __init__(self, min_lp_lock_time=86400):
        self.min_lp_lock_time = min_lp_lock_time  # En az 1 gün kilit
        self.logger = get_logger("WalletAnalyzer")

    def generate_signal(self, market_data):
        """
        market_data: {'token_address': str, ...}
        return: {'signal': int, 'confidence': float, 'rug_risk': bool}
        """
        try:
            token_address = market_data.get('token_address')
            if not token_address:
                self.logger.warning("Token adresi eksik.")
                return {'signal': 0, 'confidence': 0.0, 'rug_risk': True}
            holders = get_token_holders(token_address)
            lp_info = get_lp_info(token_address)
            # Rug risk analizi
            rug_risk = False
            if not lp_info['locked'] or lp_info['lock_time'] < self.min_lp_lock_time:
                rug_risk = True
            if any(h['amount'] > 0.5 * lp_info['total_supply'] for h in holders):
                rug_risk = True
            signal = 0 if rug_risk else 1
            confidence = 0.9 if not rug_risk else 0.1
            self.logger.info(f"WalletAnalyzer: Rug risk: {rug_risk}, Sinyal: {signal}, Güven: {confidence}")
            return {'signal': signal, 'confidence': confidence, 'rug_risk': rug_risk}
        except Exception as e:
            self.logger.error(f"WalletAnalyzer sinyal üretiminde hata: {e}")
            return {'signal': 0, 'confidence': 0.0, 'rug_risk': True} 