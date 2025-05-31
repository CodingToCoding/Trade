# Twitter sentiment algoritması (örnek) 
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from utils.logger import get_logger

class TwitterSentiment:
    """
    NLP model ile tweetlerin duygu analizini yapar (FOMO/FUD puanı).
    """
    def __init__(self, model_path='models/nöral_ag_model', tokenizer_path='models/tokenizer'):
        self.logger = get_logger("TwitterSentiment")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model.eval()
        except Exception as e:
            self.logger.error(f"Model yüklenemedi: {e}")
            self.tokenizer = None
            self.model = None

    def generate_signal(self, market_data):
        """
        market_data: {'tweets': [str], ...}
        return: {'signal': int, 'confidence': float}
        """
        try:
            tweets = market_data.get('tweets', [])
            if not tweets or self.model is None or self.tokenizer is None:
                self.logger.warning("Tweet verisi veya model eksik.")
                return {'signal': 0, 'confidence': 0.0}
            sentiments = []
            for tweet in tweets:
                inputs = self.tokenizer(tweet, return_tensors="pt", truncation=True, padding=True)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    score = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
                    sentiments.append(score)
            sentiments = np.array(sentiments)
            avg_sentiment = sentiments.mean(axis=0)
            # 0: negative, 1: neutral, 2: positive
            if avg_sentiment[2] > 0.6:
                signal = 1
                confidence = avg_sentiment[2]
            elif avg_sentiment[0] > 0.6:
                signal = -1
                confidence = avg_sentiment[0]
            else:
                signal = 0
                confidence = avg_sentiment[1]
            self.logger.info(f"Tweet sentiment: {avg_sentiment}, Sinyal: {signal}, Güven: {confidence}")
            return {'signal': signal, 'confidence': float(confidence)}
        except Exception as e:
            self.logger.error(f"TwitterSentiment sinyal üretiminde hata: {e}")
            return {'signal': 0, 'confidence': 0.0} 