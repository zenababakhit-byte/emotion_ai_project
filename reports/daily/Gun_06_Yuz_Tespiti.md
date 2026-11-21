6. Gün – MediaPipe ile Gerçek Zamanlı Yüz Tespiti

Amaç
MediaPipe kurulumunu yapıp kameradaki yüzleri canlı tespit etmek, kutuyla işaretlemek ve bunu yeniden kullanılabilir bir sınıf hâline getirmek.

Yapılan Çalışmalar
- MediaPipe (sürüm 1.0.1, Tasks API) çalışmasını doğruladım, blaze_face_short_range.tflite modelini models/ klasörüne indirdim ve requirements.txt dosyasını güncelledim.
- src/face_detector.py içinde FaceDetector sınıfını, src/face_detection_demo.py içinde ise kutu, güven yüzdesi, yüz sayısı ve FPS gösteren canlı demoyu yazıp beş farklı senaryoda test ettim.

Kullanılan Teknolojiler
- Python, OpenCV, MediaPipe (Tasks API), Git, Visual Studio Code

Elde Edilen Sonuç

┌────────────────────────────────────┬──────────────────────────┐
│ Senaryo                            │ Sonuç                    │
├────────────────────────────────────┼──────────────────────────┤
│ Tek yüz, karşıdan                  │ Bulundu, %95             │
│ İki yüz (ben + telefon fotoğrafı)  │ 2 yüz: %90 ve %52        │
│ Yarısı elle kapalı                 │ Bulundu, %66             │
│ Uzak mesafe                        │ Bulundu, %53             │
│ Düşük ışık                         │ Bulundu, %86             │
└────────────────────────────────────┴──────────────────────────┘

Uzak mesafe (%53) ve telefondaki yüz (%52) nedeniyle güven eşiğini 0.5 seviyesinde tuttum.

Karşılaşılan Sorunlar ve Çözüm
- Yüz modeli kütüphaneyle gelmediği için models/ klasörüne ayrıca indirdim.
- Terminaldeki TensorFlow Lite uyarılarının hata değil bilgi mesajı olduğu doğrulandı.

Ekran Görüntüsü
- screenshots/Gun_06_Ekran_01.png – Gun_06_Ekran_05.png: Beş test senaryosu için yüz kutusu, güven yüzdesi, yüz sayısı ve FPS çıktıları.

Ertesi Gün Planı
7. günde FER2013 veri seti indirilecek, sınıf dağılımı ve örnek görüntüler incelenerek keşifsel veri analizi yapılacaktır.