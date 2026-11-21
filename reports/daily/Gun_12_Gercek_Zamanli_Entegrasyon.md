12. Gün – Modelin Gerçek Zamanlı Sisteme Entegrasyonu

Amaç
Web kamerası, yüz tespiti ve 11. günde seçilen models/final_model.pth duygu modelini tek bir akışta birleştirerek canlı kamerada her yüz için duygu etiketi ve güven yüzdesi gösteren uygulamayı geliştirmek.

Yapılan Çalışmalar
- src/emotion_predictor.py dosyasını yazdım: final_model.pth ağırlıklarını eval() modunda yükledim, sınıf isimlerini models/label_map.json üzerinden okudum ve eğitimdekiyle birebir aynı preprocess(face_bgr) ve predict(face_bgr) fonksiyonlarını ekledim.
- tests/test_preprocess.py ile canlı ön işleme çıktısının eğitimdeki eval_tf dönüşümüyle uyumlu olduğunu pytest üzerinden doğruladım.
- src/main.py dosyasını oluşturdum: canlı kamera akışında yüz tespiti, %15 dolgu (padding) ile yüz kırpma, duyguya göre renkli kutu ve etiket çizimi, yüz yokken Yuz bulunamadi uyarısı ve s tuşuyla anlık ekran görüntüsü kaydetme özelliklerini ekledim.
- Canlı kamerada farklı yüz ifadelerini (mutlu, şaşkın, nötr, kızgın/üzgün) deneyip s tuşuyla kanıt ekran görüntülerini kaydettim.

Kullanılan Teknolojiler
- Python, OpenCV, MediaPipe, PyTorch, NumPy, pytest, Git

Elde Edilen Sonuç
- tests/test_preprocess.py testi geçti; eğitim ve çıkarım ön işlemesinin birebir uyumlu olduğu doğrulandı.
- python -m src.main komutu ile canlı kamerada yüz tespiti, renkli kutu, duygu etiketi, güven yüzdesi ve FPS akıcı şekilde çalıştı.
- Yüz kameradan çıktığında veya kırpma alanı boş olduğunda program çökmeden Yuz bulunamadi uyarısı verdi.
- En az 3 farklı duygu ifadesinin canlı ekran görüntüleri screenshots/gun12_canli_.png olarak kaydedildi.

Karşılaşılan Sorunlar ve Çözüm
- Şaşkın, mutlu ve nötr ifadeler çok net algılanırken korku ve tiksinme ifadelerinin canlı kamerada doğal taklit edilmesi daha zor olduğu için güven skorları daha düşük kaldı; bu durum FER2013 veri setinin bilinen sınırlılığı olarak not edildi.

Ekran Görüntüsü
- screenshots/gun12_canli_*.png: Canlı kamera akışında yüz kutusu, duygu etiketi, güven yüzdesi ve FPS değerini gösteren en az 3 farklı duyguya ait ekran görüntüleri.

Ertesi Gün Planı
13. günde son 10 karenin ortalamasıyla tahmin yumuşatma (smoothing), %40 altı için Belirsiz eşiği, sağ üstte 7 duygu için olasılık çubukları ve 1000 karelik detaylı FPS/gecikme performans ölçümü (reports/performans.csv) eklenecektir.