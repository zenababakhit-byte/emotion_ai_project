 8. Gün – Veri Ön İşleme, Veri Artırma ve PyTorch DataLoader

 Amaç
FER2013 veri setini PyTorch ile eğitime hazır hâle getirmek; ön işleme ve veri artırma adımlarını yazmak, eğitim/doğrulama/test bölmelerini oluşturmak ve sınıf sırasıyla ağırlıklarını kalıcı olarak kaydetmek.

 Yapılan Çalışmalar
- `scikit-learn` ve `pytest` paketlerini kurup `pip freeze > requirements.txt` ile bağımlılıkları güncelledim.
- `src/dataset.py` dosyasını yazdım: `ImageFolder` ile veriyi yükledim, yalnızca eğitim setine yatay çevirme, döndürme ve kaydırma (augmentation) uyguladım; doğrulama ve test setlerini artırmasız bıraktım.
- Eğitim verisini `seed=42` ve sınıf oranlarını koruyarak %80 eğitim / %20 doğrulama olarak böldüm, sınıf ağırlıklarını hesapladım ve `models/label_map.json` ile `models/class_weights.json` dosyalarını kaydettim.
- Windows'ta takılma olmaması için `batch_size=64` ve `num_workers=0` ile DataLoader'ları oluşturdum.
- `tests/test_dataset.py` dosyasını yazıp `python -m pytest -v tests/test_dataset.py` ile batch şekli ve bölme çakışmazlığını doğruladım.

 Kullanılan Teknolojiler
- Python, PyTorch, Torchvision, Scikit-learn, NumPy, Matplotlib, pytest, Git

Elde Edilen Sonuç
- Sınıf sırası (`label_map.json`): `{'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'neutral': 4, 'sad': 5, 'surprise': 6}`
- Sınıf ağırlıkları (`class_weights.json`): `{0: 1.0266, 1: 9.4011, 2: 1.0012, 3: 0.5684, 4: 0.826, 5: 0.8491, 6: 1.2933}`
- Veri sayıları: Train `22.967` | Val `5.742` | Test `7.178`
- Batch şekli: `[64, 1, 48, 48]` (etiket `[64]`), piksel aralığı `[-1.00, 1.00]`

 Karşılaşılan Sorunlar ve Çözüm
- İlk çalıştırmada `scikit-learn` yavaş yüklendiği için `Ctrl+C` ile durdurup `KeyboardInterrupt` aldım; tekrar çalıştırıp bekleyince sorunsuz bitti.
- `pytest` kurulu değildi; `pip install pytest` ile kurup `python -m pytest -v tests/test_dataset.py` ile testi geçirdim.

 Ekran Görüntüsü
[Gün 08 DataLoader Çıktısı](../../screenshots/gun08_dataloader.png)
Terminalde sınıf sırası, ağırlıklar, bölme sayıları ve `[64, 1, 48, 48]` batch şekli çıktısı.*

[Gün 08 Veri Artırma Örnekleri](../../screenshots/gun08_augmentation.png)
Eğitim setinden veri artırma uygulanmış 16 adet 48x48 gri tonlamalı örnek yüz.

 Ertesi Gün Planı
9. günde CPU'da hızlı çalışacak `EmotionCNN` mimarisini `src/model.py` içinde yazıp parametre sayısını ve tek görüntü çıkarım süresini (ms) ölçeceğim.