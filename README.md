Gerçek Zamanlı Yüz İfadeleri ve Duygu Analizi

1. Proje Tanımı

Bu projenin amacı, bilgisayar kamerasından alınan gerçek zamanlı görüntüler üzerinde insan yüzlerini tespit etmek ve tespit edilen yüzlerin ifadelerini yapay zekâ yöntemleri kullanarak analiz etmektir. Sistem, yüz görüntüsünü işleyerek temel duygu sınıflarından birini tahmin eder ve sonucu gerçek zamanlı olarak kullanıcıya gösterir.

2. Projenin Amacı

- Kamera görüntüsünü gerçek zamanlı olarak almak.
- Görüntü içerisindeki insan yüzlerini otomatik olarak tespit etmek.
- Tespit edilen yüz bölgelerini duygu analizi için hazırlamak.
- Yapay zekâ modeli kullanarak yüz ifadesini sınıflandırmak.
- Tahmin edilen duyguyu ve güven skorunu ekranda göstermek.
- Sistemin gerçek zamanlı performansını FPS ve gecikme değerleriyle ölçmek.

3. Duygu Sınıfları

1. Angry - Kızgın
2. Disgust - Tiksinme
3. Fear - Korku
4. Happy - Mutlu
5. Sad - Üzgün
6. Surprise - Şaşkın
7. Neutral - Nötr

4. Sistem Girdisi

Sistemin temel girdisi bilgisayarın web kamerasından alınan gerçek zamanlı video görüntüsüdür.

5. Sistem Çıktısı

Sistem, tespit edilen her yüz için şu bilgileri gösterir:
- Yüzü çevreleyen renkli tespit kutusu
- Tahmin edilen duygu etiketi ve güven yüzdesi
- Sağ üst köşede 7 duygu için canlı olasılık çubukları
- Sol üst köşede gerçek zamanlı FPS, yüz sayısı ve gecikme (ms) bilgisi

6. Temel İşlem Akışı

Kamera -> Video karesi -> Yüz tespiti -> Yüz bölgesinin kırpılması (%15 dolgu) -> Ön işleme (48x48 gri tonlama) -> EmotionCNN modeli -> 10 kare yumuşatma -> Sonucun ekranda gösterilmesi

7. Fonksiyonel Gereksinimler

- FR-01: Sistem kameradan gerçek zamanlı görüntü alabilmelidir.
- FR-02: Sistem görüntü içerisindeki yüzleri otomatik olarak tespit edebilmelidir.
- FR-03: Sistem aynı görüntü içerisinde birden fazla yüzü tespit edebilmelidir.
- FR-04: Sistem tespit edilen yüz bölgesini duygu analizi için ön işleyebilmelidir.
- FR-05: Sistem yapay zekâ modeli kullanarak duygu tahmini yapabilmelidir.
- FR-06: Sistem tahmin edilen duygu etiketini ekranda gösterebilmelidir.
- FR-07: Sistem tahmine ait güven skorunu gösterebilmelidir.
- FR-08: Sistem gerçek zamanlı FPS değerini gösterebilmelidir.
- FR-09: Sistem yüz bulunmadığı durumlarda hata vermeden çalışmaya devam edebilmelidir.

8. Fonksiyonel Olmayan Gereksinimler

- NFR-01: Sistem gerçek zamanlı ve akıcı (CPU üzerinde en az 15 FPS) çalışmalıdır.
- NFR-02: Kod modüler, okunabilir ve geliştirilebilir olmalıdır.
- NFR-03: Sistem beklenmeyen kamera veya model hatalarında kontrollü şekilde davranmalıdır.
- NFR-04: Kullanıcı arayüzündeki sonuçlar anlaşılır olmalıdır.
- NFR-05: Kullanıcı görüntüleri kalıcı olarak saklanmamalıdır.
- NFR-06: Duygu tahminlerinin kesin bir psikolojik değerlendirme olmadığı dikkate alınmalıdır.

9. Başarı Kriterleri

- Kamera görüntüsü başarıyla alınmalıdır.
- Yüz tespiti gerçek zamanlı çalışmalıdır.
- Duygu modeli sisteme entegre edilmelidir.
- Duygu ve güven skoru görüntülenmelidir.
- FPS ve gecikme ölçülmelidir.
- Model performansı uygun metriklerle değerlendirilmelidir.
- Sistem farklı test senaryolarında ve otomatik testlerde denenmelidir.

10. Kullanılan Teknolojiler

- Python
- OpenCV
- MediaPipe
- PyTorch ve Torchvision
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- pytest ve Git

11. Proje Durumu

Proje 15 günlük plan doğrultusunda tamamlandı (v1.0). Gerçek zamanlı kamera akışı, yüz tespiti, PyTorch EmotionCNN modeli, tahmin yumuşatma, olasılık çubukları ve otomatik testler çalışır durumdadır.

12. Proje Yapısı

emotion_ai_project/
├── data/fer2013/          (FER2013 veri seti, Git dışı)
├── models/                (label_map.json, class_weights.json, final_model.pth)
├── notebooks/             (Veri analizi not defterleri)
├── reports/
│   ├── daily/             (Günlük çalışma raporları, Gün 1–15)
│   └── final/             (Final raporu)
├── screenshots/           (Günlük kanıt ekran görüntüleri)
├── src/                   (config.py, dataset.py, model.py, train.py, improve_train.py,
│                           evaluate.py, emotion_predictor.py, utils.py, main.py)
├── tests/                 (test_dataset.py, test_model.py, test_preprocess.py, test_detector.py)
├── requirements.txt
├── README.md
└── .gitignore

13. Kurulum ve Çalıştırma

Sanal ortam oluşturma ve paket kurulumu:
- python -m venv venv
- .\venv\Scripts\Activate.ps1
- pip install -r requirements.txt

Uygulamayı ve testleri çalıştırma:
- Canlı duygu analizi (s: ekran görüntüsü, q: çıkış): python -m src.main
- 1000 karelik performans ölçümü: python -m src.main --benchmark 1000
- Otomatik testler: python -m pytest -v tests/

14. Model ve Gerçek Zamanlı Performans Sonuçları

┌──────────────────────────┬────────────────────────────────────┐
│ Metrik                   │ Değer                              │
├──────────────────────────┼────────────────────────────────────┤
│ Mimari                   │ EmotionCNN (1.47M param, PyTorch)  │
│ Test Doğruluğu (7.178)   │ %60.69 (Deney 1: %48.20)           │
│ Macro-F1 Skoru           │ 0.5562 (Deney 1: 0.4057)           │
│ En Güçlü Sınıflar        │ Happy (0.83 F1), Surprise (0.75)   │
│ Ortalama Hız (1000 kare) │ 23.91 FPS (Hedef >= 15 FPS)        │
│ Saf AI Gecikmesi (CPU)   │ 4.69 ms (Toplam 41.8, p95 63 ms)   │
│ Otomatik Testler         │ 5 / 5 PASSED (pytest)              │
└──────────────────────────┴────────────────────────────────────┘

15. Bilinen Sınırlılıklar, Etik ve Gizlilik

- FER2013 veri setindeki düşük çözünürlük (48x48) ve etiket gürültüsü nedeniyle fear ve disgust sınıflarındaki F1 skorları diğer sınıflara göre daha düşüktür.
- Kamera görüntüleri kalıcı olarak kaydedilmez; yalnızca kullanıcı s tuşuna bastığında test amaçlı ekran görüntüsü alınır.
- Üretilen duygu tahminleri yalnızca görsel yüz ifadesi sınıflandırmasıdır ve kesin bir psikolojik ruh hali ölçümü değildir.