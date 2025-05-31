# Rug checker algoritması (örnek) 
from utils.logger import get_logger
from utils.solana_api import get_token_contract_info, check_honeypot

class RugChecker:
    """
    Token kontrat yapısını, LP kilit durumunu, honeypot olup olmadığını kontrol eder.
    """
    def __init__(self):
        self.logger = get_logger("RugChecker")

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
            contract_info = get_token_contract_info(token_address)
            honeypot = check_honeypot(token_address)
            rug_risk = False
            if not contract_info['verified'] or not contract_info['lp_locked'] or honeypot:
                rug_risk = True
            signal = 0 if rug_risk else 1
            confidence = 0.9 if not rug_risk else 0.1
            self.logger.info(f"RugChecker: Rug risk: {rug_risk}, Sinyal: {signal}, Güven: {confidence}")
            return {'signal': signal, 'confidence': confidence, 'rug_risk': rug_risk}
        except Exception as e:
            self.logger.error(f"RugChecker sinyal üretiminde hata: {e}")
            return {'signal': 0, 'confidence': 0.0, 'rug_risk': True} 