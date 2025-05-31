# Bildirim göndermek için yardımcı fonksiyonlar 
import os
import requests
from utils.logger import get_logger

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")
logger = get_logger("Notifier")

def send_notification(message: str):
    """Telegram ve Discord'a bildirim gönderir."""
    telegram_sent = False
    discord_sent = False
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
            data = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
            resp = requests.post(url, data=data)
            resp.raise_for_status()
            telegram_sent = True
        except Exception as e:
            logger.error(f"Telegram bildirimi gönderilemedi: {e}")
    if DISCORD_WEBHOOK_URL:
        try:
            resp = requests.post(DISCORD_WEBHOOK_URL, json={"content": message})
            resp.raise_for_status()
            discord_sent = True
        except Exception as e:
            logger.error(f"Discord bildirimi gönderilemedi: {e}")
    if not telegram_sent and not discord_sent:
        logger.warning("Hiçbir bildirim gönderilemedi.") 