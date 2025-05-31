# Model Dosyaları

Bu klasörde, TwitterSentiment ve diğer AI modülleri için gerekli olan model ve tokenizer dosyaları bulunur.

- nöral_ag_model.pt : HuggingFace/transformers uyumlu PyTorch duygu analizi modeli
- tokenizer.pkl     : İlgili tokenizer (HuggingFace veya pickle formatında)

Gerçek dosyalar büyük olduğu için burada yer almıyor. Kendi modelinizi eğitip veya HuggingFace'den indirip bu klasöre koymalısınız.

Örnek indirme kodu (transformers ile):

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')
model.save_pretrained('./models/nöral_ag_model.pt')
tokenizer = AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')
tokenizer.save_pretrained('./models/tokenizer.pkl')
``` 