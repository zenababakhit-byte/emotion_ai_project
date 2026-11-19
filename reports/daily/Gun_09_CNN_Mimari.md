# 9. Gün – CNN Model Mimarisinin Tasarlanması

**Tarih:** 19.11.2026  
**Proje:** emotion_ai_project

---

## Amaç
7 duygu sınıfını ayırt edebilecek ve CPU üzerinde gerçek zamanlı çalışabilecek kadar hafif bir CNN mimarisi (`EmotionCNN`) tasarlayıp parametre sayısı, boyut akışı ve çıkarım hızını doğrulamak.

## Yapılan Çalışmalar
- `src/model.py` dosyasında 3 evrişim bloğundan (`32 -> 64 -> 128`) ve tam bağlantılı sınıflandırıcıdan oluşan `EmotionCNN` sınıfını yazdım.
- Eğitimde kullanılacak bileşenleri (`CrossEntropyLoss` + sınıf ağırlıkları, `Adam` optimizörü `lr=1e-3, weight_decay=1e-4`, `ReduceLROnPlateau` zamanlayıcısı) aynı dosyada tanımladım.
- `torch.randn(1, 1, 48, 48)` sahte girdisiyle ileri geçiş testi yaptım, eğitilebilir parametre sayısını ve CPU'da 100 tekrar ortalamasıyla tek görüntü çıkarım süresini ölçtüm.
- `tests/test_model.py` ile tekil (`[1, 7]`) ve batch (`[16, 7]`) çıkış şekli testlerini yazıp `python -m pytest -v tests/test_model.py` ile doğruladım.

### Model Mimari Özeti
| Katman / Blok | İşlem | Çıkış Boyutu |
| :--- | :--- | :--- |
| Giriş | 48x48 gri tonlamalı yüz | `[N, 1, 48, 48]` |
| Blok 1 | 2x (Conv2d 3x3, 32 + BN + ReLU) + MaxPool(2) + Dropout(0.25) | `[N, 32, 24, 24]` |
| Blok 2 | 2x (Conv2d 3x3, 64 + BN + ReLU) + MaxPool(2) + Dropout(0.25) | `[N, 64, 12, 12]` |
| Blok 3 | 2x (Conv2d 3x3, 128 + BN + ReLU) + MaxPool(2) + Dropout(0.25) | `[N, 128, 6, 6]` |
| Sınıflandırıcı | Flatten -> Linear(4608, 256) -> ReLU -> Dropout(0.5) -> Linear(256, 7) | `[N, 7]` |

## Kullanılan Teknolojiler
- Python, PyTorch, pytest, Git

## Elde Edilen Sonuç
- **Giriş / Çıkış şekli:** `[1, 1, 48, 48]` -> `[1, 7]`
- **Toplam eğitilebilir parametre:** `1.469.031`
- **Ortalama CPU çıkarım süresi:** `1.30 ms` (100 tekrar ortalaması, `< 10 ms` hedefinin oldukça altında)
- **Oluşturulan dosyalar:** `src/model.py`, `tests/test_model.py`

## Karşılaşılan Sorunlar ve Çözüm
- `test_model.py` ilk başta yanlışlıkla `tests/__pycache__/` içine açıldığı için `pytest` dosyayı bulamadı; `tests/` klasörüne taşınınca test `1 passed` ile geçti.

## Ekran Görüntüsü
![Gün 09 Model Özeti](../../screenshots/gun09_model.png)
*Terminalde `EmotionCNN` mimarisi, `1.469.031` parametre sayısı, `[1, 7]` çıkış şekli ve `1.30 ms` CPU çıkarım süresi çıktısı.*

![Gün 09 Pytest Çıktısı](../../screenshots/gun09_test.png)
*`tests/test_model.py` dosyasının `pytest` ile hatasız (`PASSED`) geçtiğini gösteren terminal çıktısı.*

## Ertesi Gün Planı
10. günde `src/train.py` betiği yazılarak önce 2 epoch'luk hızlı doğrulama (smoke test), ardından FER2013 üzerinde tam eğitim yapılacak; en iyi ağırlıklar `models/best_model.pth` olarak kaydedilip öğrenme eğrisi grafiği üretilecektir.
