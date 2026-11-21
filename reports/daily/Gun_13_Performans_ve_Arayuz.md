13. Gün – Performans Ölçümü, Tahmin Kararlılığı ve Arayüz

Amaç
Gerçek zamanlı sistemin performansını 1000 kare boyunca sayısal olarak ölçmek (FPS ve gecikme), tahminlerdeki titreşimi yumuşatma ile azaltmak ve arayüze olasılık çubukları eklemek.

Yapılan Çalışmalar
- src/utils.py dosyasını oluşturdum: her yüz için son 10 karenin olasılıklarını tutan MultiFaceSmoother (deque), %40 altı güven için Belirsiz eşiği, kare bazlı süre kaydeden PerformanceLogger ve sağ üstte 7 duygu için yarı saydam olasılık çubukları çizen draw_probability_bars fonksiyonunu yazdım.
- src/main.py dosyasını güncelledim: yumuşatılmış tahminler, sol üstte FPS/yüz sayısı/gecikme göstergesi, yüz yokken Yuz bulunamadi uyarısı ve --benchmark 1000 ölçüm modu eklendi.
- python -m src.main --benchmark 1000 çalıştırarak 1000 karelik performans verisini reports/performans.csv dosyasına kaydettim ve arayüz ekran görüntüsünü (screenshots/gun13_arayuz.png) aldım.

Kullanılan Teknolojiler
- Python, OpenCV, PyTorch, collections.deque, NumPy, Pandas, Git

Elde Edilen Sonuç
1000 Kare Performans Ölçümü (CPU):

| Aşama                  | Ortalama (ms) | p95 Gecikme (ms) |
| ---------------------- | ------------- | ---------------- |
| Yüz Tespiti            | 2.22          | 3.06             |
| Ön İşleme (48x48)      | 0.20          | 0.28             |
| CNN Model Çıkarımı     | 2.27          | 3.31             |
| Toplam Kare (Kamera)   | 41.82         | 63.00            |

- Ortalama FPS: 23.91 FPS (Hedef >= 15 FPS başarıyla sağlandı)
- 10 karelik hareketli ortalama ile etiket titreşimi belirgin biçimde azaldı.
- Oluşturulan / güncellenen dosyalar: src/utils.py, src/main.py, reports/performans.csv

Karşılaşılan Sorunlar ve Çözüm
- Toplam kare süresi ağırlıklı olarak kamera kare okuma hızına bağlı olarak 41.82 ms ölçüldü; yüz tespiti ve model çıkarımı toplamda 4.69 ms sürdüğü için ek optimizasyona gerek kalmadan 23.91 FPS hedeflenen 15 FPS sınırının oldukça üzerinde çalıştı.

Ekran Görüntüsü
- screenshots/gun13_arayuz.png: Sağ üstte 7 duygu olasılık çubuğu, sol üstte FPS/yüz sayısı/gecikme ve yüz üzerinde yumuşatılmış etiket görünen canlı arayüz.
- screenshots/gun13_performans.png: Terminalde 1000 karelik performans özeti (23.91 FPS, ortalama ve p95 gecikme değerleri).

Ertesi Gün Planı
14. günde tüm otomatik pytest testleri (test_dataset, test_model, test_preprocess, test_detector) tamamlanacak, sabitler src/config.py içinde toplanacak, senaryo test tablosu doldurulacak ve README.md güncellenecektir.