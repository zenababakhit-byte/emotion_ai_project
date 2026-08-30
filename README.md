# Gerçek Zamanlı Yüz İfadeleri ve Duygu Analizi

## 1. Proje Tanımı

Bu projenin amacı, bilgisayar kamerasından alınan gerçek zamanlı
görüntüler üzerinde insan yüzlerini tespit etmek ve tespit edilen
yüzlerin ifadelerini yapay zekâ yöntemleri kullanarak analiz etmektir.

Sistem, yüz görüntüsünü işleyerek temel duygu sınıflarından birini
tahmin edecek ve sonucu gerçek zamanlı olarak kullanıcıya gösterecektir.

## 2. Projenin Amacı

Projenin temel amaçları şunlardır:

- Kamera görüntüsünü gerçek zamanlı olarak almak.
- Görüntü içerisindeki insan yüzlerini otomatik olarak tespit etmek.
- Tespit edilen yüz bölgelerini duygu analizi için hazırlamak.
- Yapay zekâ modeli kullanarak yüz ifadesini sınıflandırmak.
- Tahmin edilen duyguyu ve güven skorunu ekranda göstermek.
- Sistemin gerçek zamanlı performansını FPS ve gecikme değerleriyle ölçmek.

## 3. Duygu Sınıfları

İlk aşamada sistem aşağıdaki yedi duygu sınıfını kullanacaktır:

1. Angry - Kızgın
2. Disgust - Tiksinme
3. Fear - Korku
4. Happy - Mutlu
5. Sad - Üzgün
6. Surprise - Şaşkın
7. Neutral - Nötr

## 4. Sistem Girdisi

Sistemin temel girdisi bilgisayarın web kamerasından alınan
gerçek zamanlı video görüntüsüdür.

İlerleyen aşamalarda kayıtlı video dosyalarının analiz edilmesi de
desteklenecektir.

## 5. Sistem Çıktısı

Sistem, tespit edilen her yüz için aşağıdaki bilgileri gösterecektir:

- Yüzü çevreleyen bir tespit kutusu
- Tahmin edilen duygu etiketi
- Tahmin güven skoru
- Gerçek zamanlı FPS bilgisi

## 6. Temel İşlem Akışı

Kamera
→ Video karesi
→ Yüz tespiti
→ Yüz bölgesinin kırpılması
→ Ön işleme
→ Yapay zekâ modeli
→ Duygu tahmini
→ Sonucun ekranda gösterilmesi

## 7. Fonksiyonel Gereksinimler

FR-01: Sistem kameradan gerçek zamanlı görüntü alabilmelidir.

FR-02: Sistem görüntü içerisindeki yüzleri otomatik olarak
tespit edebilmelidir.

FR-03: Sistem aynı görüntü içerisinde birden fazla yüzü
tespit edebilmelidir.

FR-04: Sistem tespit edilen yüz bölgesini duygu analizi için
ön işleyebilmelidir.

FR-05: Sistem yapay zekâ modeli kullanarak duygu tahmini yapabilmelidir.

FR-06: Sistem tahmin edilen duygu etiketini ekranda gösterebilmelidir.

FR-07: Sistem tahmine ait güven skorunu gösterebilmelidir.

FR-08: Sistem gerçek zamanlı FPS değerini gösterebilmelidir.

FR-09: Sistem yüz bulunmadığı durumlarda hata vermeden çalışmaya
devam edebilmelidir.

FR-10: Sistem kayıtlı video dosyalarını da analiz edebilmelidir.

## 8. Fonksiyonel Olmayan Gereksinimler

NFR-01: Sistem mümkün olduğunca gerçek zamanlı ve akıcı çalışmalıdır.

NFR-02: Kod modüler, okunabilir ve geliştirilebilir olmalıdır.

NFR-03: Sistem beklenmeyen kamera veya model hatalarında
kontrollü şekilde davranmalıdır.

NFR-04: Kullanıcı arayüzündeki sonuçlar anlaşılır olmalıdır.

NFR-05: Kullanıcı görüntüleri gereksiz yere saklanmamalıdır.

NFR-06: Duygu tahminlerinin kesin bir psikolojik değerlendirme
olmadığı dikkate alınmalıdır.

## 9. Başarı Kriterleri

Proje başarılı kabul edilebilmesi için:

- Kamera görüntüsü başarıyla alınmalıdır.
- Yüz tespiti gerçek zamanlı çalışmalıdır.
- Duygu modeli sisteme entegre edilmelidir.
- Duygu ve güven skoru görüntülenmelidir.
- FPS veya gecikme ölçülmelidir.
- Model performansı uygun metriklerle değerlendirilmelidir.
- Sistem farklı test senaryolarında denenmelidir.

## 10. Kullanılabilecek Teknolojiler

- Python
- OpenCV
- NumPy
- TensorFlow/Keras veya PyTorch
- MediaPipe veya OpenCV tabanlı yüz tespiti
- Pandas
- Matplotlib
- Scikit-learn
- Git

## 11. Proje Durumu

Proje aktif olarak geliştirilmektedir.

## 12. Proje Yapısı

emotion_ai_project/
│
├── data/
├── models/
├── notebooks/
├── reports/
│   ├── daily/
│   └── final/
├── screenshots/
├── src/
├── tests/
├── requirements.txt
├── README.md
└── .gitignore