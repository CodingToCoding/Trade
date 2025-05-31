# Twitter API işlemleri için yardımcı fonksiyonlar 
import os
import tweepy
from typing import List, Dict, Any
from utils.logger import get_logger

TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN", "")
logger = get_logger("TwitterAPI")

# Tweepy ile Twitter API bağlantısı
client = None
if TWITTER_BEARER_TOKEN:
    client = tweepy.Client(bearer_token=TWITTER_BEARER_TOKEN)

def get_recent_tweets(query: str, max_results: int = 50) -> List[Dict[str, Any]]:
    """Belirli bir arama sorgusuna göre son tweetleri döndürür."""
    try:
        if not client:
            logger.error("Twitter API bağlantısı yok.")
            return []
        tweets = client.search_recent_tweets(query=query, max_results=max_results, tweet_fields=["text", "created_at", "author_id"])
        return [
            {"text": t.text, "created_at": t.created_at, "author_id": t.author_id}
            for t in tweets.data
        ] if tweets.data else []
    except Exception as e:
        logger.error(f"get_recent_tweets hata: {e}")
        return []

def get_influencer_tweets(username: str, max_results: int = 20) -> List[Dict[str, Any]]:
    """Belirli bir influencer hesabından son tweetleri döndürür."""
    try:
        if not client:
            logger.error("Twitter API bağlantısı yok.")
            return []
        user = client.get_user(username=username)
        if not user.data:
            return []
        user_id = user.data.id
        tweets = client.get_users_tweets(id=user_id, max_results=max_results, tweet_fields=["text", "created_at"])
        return [
            {"text": t.text, "created_at": t.created_at}
            for t in tweets.data
        ] if tweets.data else []
    except Exception as e:
        logger.error(f"get_influencer_tweets hata: {e}")
        return [] 