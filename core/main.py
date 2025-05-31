# Sistemi başlatan ana dosya (örnek) 

import time
import traceback
import os
from core.decision_engine import DecisionEngine
from algorithms.rsi_ema_macd import RSI_EMA_MACD
from algorithms.fibonacci import Fibonacci
from algorithms.whale_tracker import WhaleTracker
from algorithms.spike_detector import SpikeDetector
from algorithms.wallet_analyzer import WalletAnalyzer
from algorithms.influencer_ring import InfluencerRing
from algorithms.twitter_sentiment import TwitterSentiment
from algorithms.rug_checker import RugChecker
from utils.logger import get_logger
from utils.solana_api import get_token_holders, get_lp_info
from utils.twitter_api import get_recent_tweets
from utils.notifier import send_notification

logger = get_logger("Main")

# Kullanıcıdan veya .env'den alınacak parametreler
TOKEN_ADDRESS = os.getenv("TOKEN_ADDRESS", "")
TOKEN_SYMBOL = os.getenv("TOKEN_SYMBOL", "")
WHALE_ADDRESSES = os.getenv("WHALE_ADDRESSES", "").split(",") if os.getenv("WHALE_ADDRESSES") else []
INFLUENCER_ACCOUNTS = os.getenv("INFLUENCER_ACCOUNTS", "").split(",") if os.getenv("INFLUENCER_ACCOUNTS") else []

# Algoritma nesnelerini oluştur
algorithms = {
    "rsi_ema_macd": RSI_EMA_MACD(),
    "fibonacci": Fibonacci(),
    "whale_tracker": WhaleTracker(whale_addresses=WHALE_ADDRESSES),
    "spike_detector": SpikeDetector(),
    "wallet_analyzer": WalletAnalyzer(),
    "influencer_ring": InfluencerRing(influencer_accounts=INFLUENCER_ACCOUNTS),
    "twitter_sentiment": TwitterSentiment(),
    "rug_checker": RugChecker(),
}
engine = DecisionEngine(algorithms)

# Performans takibi için geçmiş kararlar
algo_performance = {name: [] for name in algorithms.keys()}


def fetch_market_data():
    """Gerçek zamanlı market ve sosyal veri toplar."""
    # Burada gerçek API çağrıları ile veri çekilmeli
    # Örnek veri (gerçekle değiştirilmeli)
    close = [1.0]*120  # Fiyat verisi
    volume = [1000.0]*120  # Hacim verisi
    liquidity = 20000
    position_size = 500
    tweets = get_recent_tweets(f"${TOKEN_SYMBOL}") if TOKEN_SYMBOL else []
    holders = get_token_holders(TOKEN_ADDRESS) if TOKEN_ADDRESS else []
    lp_info = get_lp_info(TOKEN_ADDRESS) if TOKEN_ADDRESS else {"locked": True, "lock_time": 90000, "total_supply": 1000000}
    return {
        "close": close,
        "volume": volume,
        "token_address": TOKEN_ADDRESS,
        "token_symbol": TOKEN_SYMBOL,
        "tweets": [t["text"] for t in tweets],
        "liquidity": liquidity,
        "position_size": position_size,
        "holders": holders,
        "lp_info": lp_info,
    }

def main():
    logger.info("Meme Coin Trading Bot başlatıldı.")
    while True:
        try:
            market_data = fetch_market_data()
            # Her algoritmanın çıktısı alınır
            signals = {}
            for name, algo in algorithms.items():
                result = algo.generate_signal(market_data)
                signals[name] = result
            # Orchestrator ile nihai karar
            result = engine.orchestrator.aggregate_signals(signals)
            # Performans güncellemesi (örnek: rastgele doğru/yanlış)
            for name in signals:
                # Burada gerçek trade sonucu ile doğru/yanlış güncellenmeli
                engine.orchestrator.update_performance(name, correct=True)
            # Bildirim ve log
            send_notification(f"Karar: {result}")
            logger.info(f"Karar sonucu: {result}")
            time.sleep(30)
        except Exception as e:
            logger.error(f"Ana döngüde hata: {e}\n{traceback.format_exc()}")
            time.sleep(60) 