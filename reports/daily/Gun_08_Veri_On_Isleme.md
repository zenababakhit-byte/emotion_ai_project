8. Gün – Veri Ön İşleme, Veri Artırma ve PyTorch DataLoader

Amaç
FER2013 verisini PyTorch eğitimine hazır hâle getirmek, yalnızca eğitim setine veri artırma uygulamak ve sınıf sırasıyla ağırlıklarını kaydetmek.

Yapılan Çalışmalar
- scikit-learn ve pytest kurup requirements.txt dosyasını güncelledim.
- src/dataset.py içinde veriyi %80 eğitim / %20 doğrulama olarak (seed=42, stratified) böldüm; yalnızca eğitim setine yatay çevirme, döndürme ve kaydırma ekledim.
- models/label_map.json ile models/class_weights.json dosyalarını kaydedip batch_size=64, num_workers=0 ile DataLoader kurdum.
- tests/test_dataset.py ile batch şeklini ve bölme çakışmazlığını doğruladım.

Kullanılan Teknolojiler
- Python, PyTorch, Torchvision, Scikit-learn, NumPy, Matplotlib, pytest, Git

Elde Edilen Sonuç
- Train: 22.967 | Val: 5.742 | Test: 7.178 | Batch şekli: [64, 1, 48, 48] | Aralık: [-1.00, 1.00]
- label_map.json (7 sınıf) ve class_weights.json (disgust: 9.4011, happy: 0.5684) kaydedildi.

Karşılaşılan Sorunlar ve Çözüm
- İlk çalıştırmada scikit-learn yavaş yüklenince Ctrl+C ile durdurmuştum; tekrar çalıştırıp bekleyince sorunsuz bitti.
- Eksik olan pytest kurularak test 1 passed ile geçirildi.

Ekran Görüntüsü
- screenshots/gun08_dataloader.png: Sınıf sırası, bölme sayıları ve batch şekli çıktısı.
- screenshots/gun08_augmentation.png: Veri artırma uygulanmış 16 örnek yüz.

Ertesi Gün Planı
9. günde EmotionCNN mimarisini src/model.py içine yazıp parametre sayısını ve CPU çıkarım süresini ölçeceğim.