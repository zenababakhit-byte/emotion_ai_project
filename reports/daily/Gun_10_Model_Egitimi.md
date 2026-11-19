10. Gün – Modelin Eğitilmesi

Amaç
EmotionCNN modelini FER2013 eğitim verisi üzerinde eğitmek, doğrulama başarımını izleyerek en iyi ağırlıkları models/best_model.pth olarak kaydetmek ve eğitim/doğrulama öğrenme eğrilerini üretmektir.

Yapılan Çalışmalar
- src/train.py eğitim betiğini yazdım: epoch döngüsü, model.train() / model.eval() geçişleri, her epoch sonunda reports/training_log.csv kaydı, models/last_checkpoint.pth yedeklemesi, erken durdurma (patience=7) ve eğitim sonunda öğrenme eğrisi çizimi eklendi.
- Tam eğitime geçmeden önce python -m src.train --smoke-test komutuyla 512 eğitim ve 128 doğrulama örneği üzerinde 2 epoch hızlı doğrulama yaptım; eğitim kaybının 3.2601 değerinden 2.0203 değerine, doğrulama kaybının 1.9294 değerinden 1.8555 değerine düştüğünü doğruladım.
- Ardından python -m src.train --epochs 20 komutuyla CPU üzerinde 20 epoch tam eğitimi çalıştırdım.
- Her iyileşmede en iyi model ağırlıkları models/best_model.pth olarak kaydedildi ve eğitim sonunda reports/ogrenme_egrisi.png grafiği üretildi.

Kullanılan Teknolojiler
- Python, PyTorch, Pandas, Matplotlib, Git

Elde Edilen Sonuç
- Cihaz: CPU (epoch başına ortalama yaklaşık 82 saniye)
- Epoch 01: train_loss=1.9787, train_acc=18.04%, val_loss=1.9282, val_acc=25.64%
- Epoch 08: train_loss=1.7732, train_acc=31.64%, val_loss=1.7068, val_acc=39.81%
- Epoch 16: train_loss=1.5970, train_acc=39.02%, val_loss=1.4403, val_acc=46.31%
- Epoch 20 (En İyi): train_loss=1.4867, train_acc=42.85%, val_loss=1.3449, val_acc=48.17%
- Overfitting gözlemlenmedi; doğrulama kaybı son epoch değerine kadar düşmeye devam etti.
- Oluşturulan dosyalar: src/train.py, models/best_model.pth (Git dışı), reports/training_log.csv, reports/ogrenme_egrisi.png

Karşılaşılan Sorunlar ve Çözüm
- Eğitim CPU üzerinde çalıştığı için süreyi makul tutmak adına 30 epoch yerine 20 epoch ile ilk eğitim tamamlandı. Model 20. epoch sonunda hâlâ gelişme gösterdiği için 11. gündeki iyileştirme turunda ek eğitim veya hiperparametre iyileştirmesi planlandı.

Ekran Görüntüsü
- screenshots/gun10_egitim.png: Terminalde 20 epoch eğitim logları (loss ve accuracy değerleri) ile en iyi model kayıt çıktısı.
- screenshots/gun10_egri.png: Eğitim ve doğrulama kayıp/doğruluk değerlerini gösteren öğrenme eğrisi grafiği.

Ertesi Gün Planı
11. günde best_model.pth test seti üzerinde değerlendirilecek; karışıklık matrisi ve sınıf bazlı metrikler (precision, recall, F1) çıkarılacak, hata analizi yapılıp iyileştirme turu ile nihai model (models/final_model.pth) seçilecektir.