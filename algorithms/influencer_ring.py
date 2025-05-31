from utils.logger import get_logger
from utils.twitter_api import get_influencer_tweets

class InfluencerRing:
    """
    Twitter üzerindeki influencer tweet zincirleriyle pump hareketlerini analiz eder.
    """
    def __init__(self, influencer_accounts, min_mentions=3):
        self.influencer_accounts = influencer_accounts
        self.min_mentions = min_mentions
        self.logger = get_logger("InfluencerRing")

    def generate_signal(self, market_data):
        """
        market_data: {'token_symbol': str, ...}
        return: {'signal': int, 'confidence': float}
        """
        try:
            token_symbol = market_data.get('token_symbol')
            if not token_symbol:
                self.logger.warning("Token sembolü eksik.")
                return {'signal': 0, 'confidence': 0.0}
            total_mentions = 0
            for account in self.influencer_accounts:
                tweets = get_influencer_tweets(account)
                mentions = sum(1 for t in tweets if token_symbol.lower() in t['text'].lower())
                total_mentions += mentions
            if total_mentions >= self.min_mentions:
                self.logger.info(f"Influencer pump tespit edildi! Mention sayısı: {total_mentions}")
                return {'signal': 1, 'confidence': min(1.0, total_mentions / 10)}
            else:
                return {'signal': 0, 'confidence': 0.5}
        except Exception as e:
            self.logger.error(f"InfluencerRing sinyal üretiminde hata: {e}")
            return {'signal': 0, 'confidence': 0.0} 