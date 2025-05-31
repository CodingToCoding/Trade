# Meme Coin Trading Bot

Solana ağında yeni çıkan meme coinlerde otomatik al-sat kararları verebilen, gelişmiş yapay zekâ destekli trading botu.

## Klasör Yapısı
- config/: Ayar ve sabitler
- data/: Veri depolama
- core/: Ana sistem ve orchestrator
- algorithms/: Analiz ve sinyal algoritmaları
- utils/: Yardımcı araçlar
- models/: Nöral ağ ve tokenizer dosyaları

## Kurulum ve Kullanım
Detaylar ileride eklenecektir.

# Proje Dosya Yapısı

```
project-root/
│
├── config/
│   ├── settings.py
│   └── constants.py
│
├── data/
│   ├── historical/
│   └── live/
│
├── core/
│   ├── main.py
│   ├── orchestrator.py
│   └── decision_engine.py
│
├── algorithms/
│   ├── rsi_ema_macd.py
│   ├── fibonacci.py
│   ├── whale_tracker.py
│   ├── twitter_sentiment.py
│   ├── wallet_analyzer.py
│   ├── spike_detector.py
│   ├── rug_checker.py
│   └── influencer_ring.py
│
├── utils/
│   ├── solana_api.py
│   ├── twitter_api.py
│   ├── notifier.py
│   └── logger.py
│
├── models/
│   ├── nöral_ag_model.pt
│   └── tokenizer.pkl
│
├── requirements.txt
├── Procfile
├── start.py
└── README.md
```

Eksik klasör ve dosyalar oluşturulmalıdır:
- data/ (historical/, live/ alt klasörleriyle)
- utils/ (solana_api.py, twitter_api.py, notifier.py, logger.py)
- models/ (nöral_ag_model.pt, tokenizer.pkl)
- requirements.txt
- Procfile
- start.py
