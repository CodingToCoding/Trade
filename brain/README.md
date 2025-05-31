# Brain Modülü

Bu klasör, projenin merkezi beyin ve gelişmiş karar verme mekanizmasını içerir.

## Dosyalar
- `neural_brain.py`: Çok algoritmalı, reinforcement learning destekli merkezi karar ağı
- `brain_engine.py`: NeuralBrain'i kullanan, tüm sinyalleri işleyen ve öğrenen ana motor
- `brain_utils.py`: Sinyal normalizasyonu, ağırlık istatistikleri ve karar analizi yardımcıları

## Özellikler
- Algo-IQ: Algoritma ağırlıklarını geçmiş performansa göre dinamik günceller
- Inter-Algorithm Communication: Algoritmalar arası çapraz doğrulama ve güven artırma
- Reinforcement Learning: Gerçek trade sonuçlarına göre ağırlık güncelleme
- Karar geçmişi ve istatistiksel analiz

## Kullanım
BrainEngine, algoritma isimleriyle başlatılır ve sinyal işleme ile öğrenme fonksiyonları çağrılır:

```python
from brain.brain_engine import BrainEngine
engine = BrainEngine(["rsi_ema_macd", "fibonacci", ...])
result = engine.process_signals(signals)
engine.feedback(signals, actual_result)
``` 