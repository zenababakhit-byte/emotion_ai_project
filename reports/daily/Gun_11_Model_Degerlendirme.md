11. Gün – Model Değerlendirme ve İyileştirme

Amaç
Eğitilen EmotionCNN modelini test seti üzerinde değerlendirmek, karışıklık matrisi ile hata analizi yapmak, iyileştirme turu uygulayıp nihai modeli (models/final_model.pth) seçmek.

Yapılan Çalışmalar
- src/evaluate.py ile accuracy, macro-F1, classification_report, karışıklık matrisi ve yanlış tahmin edilen 16 örneği ürettim.
- Deney 1 sonucunda test doğruluğu %48.20, macro-F1 0.4057 çıktı; disgust sınıf ağırlığının (9.40) fazla yüksek olduğu görüldü.
- Deney 2 için src/improve_train.py ile sınıf ağırlıklarını karekök (sqrt) ile yumuşatıp best_model.pth üzerinden lr=5e-4 ile 10 epoch fine-tuning yaptım.
- Deney 2 ile doğruluk %60.69 ve macro-F1 0.5562 oldu; bu modeli models/final_model.pth olarak kaydedip reports/model_sonuclari.md özetini yazdım.

Kullanılan Teknolojiler
- Python, PyTorch, Scikit-learn, Matplotlib, NumPy, Git

Elde Edilen Sonuç
- Deney 1: Accuracy %48.20 | Macro-F1 0.4057
- Deney 2 (Nihai): Accuracy %60.69 | Macro-F1 0.5562 (+12.49 puan artış)
- En yüksek F1: happy (0.8337) ve surprise (0.7525)
- En çok karışan çiftler: sad -> neutral (311), fear -> sad (263), neutral -> sad (200)

Karşılaşılan Sorunlar ve Çözüm
- İlk deneyde agresif sınıf ağırlığı nedeniyle disgust yanlış pozitifleri yüksekti; ağırlıklar karekök (sqrt) ile yumuşatılıp fine-tuning yapılarak tüm sınıflarda dengeli artış sağlandı.

Ekran Görüntüsü
- screenshots/gun11_rapor.png: Deney 2 classification_report çıktısı (%60.69 doğruluk).
- screenshots/gun11_karisiklik.png: 7 sınıflı test karışıklık matrisi ısı haritası.

Ertesi Gün Planı
12. günde final_model.pth ile yüz tespitini birleştirerek canlı kamerada duygu etiketi ve güven yüzdesi gösteren src/emotion_predictor.py ve src/main.py uygulamalarını yazacağım.