FINAL PROJE RAPORU
Yapay Zekâ ile Gerçek Zamanlı Yüz İfadeleri ve Duygu Analizi

1. Giriş ve Amaç
Bu projede bilgisayar kamerasından gerçek zamanlı yüz tespiti yapan ve PyTorch tabanlı bir CNN modeliyle 7 temel duyguyu (angry, disgust, fear, happy, neutral, sad, surprise) güven yüzdesiyle ekranda gösteren bir sistem geliştirdim. Temel hedef, uygulamanın CPU üzerinde en az 15 FPS hızında kararlı çalışmasıydı.

2. Yöntem ve Sistem Akışı
Kamera (640x480) -> Yüz Tespiti -> %15 Dolgulu Kırpma -> 48x48 Gri Tonlama ve Normalizasyon -> EmotionCNN -> 10 Kare Yumuşatma (deque) -> %40 Altı Belirsiz Eşiği -> Renkli Kutu ve Olasılık Çubukları.

3. Veri Seti (FER2013) ve Analiz
Eğitimde 48x48 gri tonlamalı FER2013 veri setini kullandım. Eğitim verisini (28.709 görüntü) sınıf oranlarını koruyarak %80 eğitim (22.967) ve %20 doğrulama (5.742) olarak böldüm; 7.178 görüntülük test setini yalnızca son değerlendirme için ayırdım. İncelemede özellikle disgust sınıfında belirgin veri azlığı ve etiket gürültüsü gördüm.

4. Model Mimarisi ve Eğitim
CPU'da hızlı çalışması için 3 evrişim bloğundan (32 -> 64 -> 128) ve 1.469.031 parametreden oluşan EmotionCNN mimarisini yazdım. Veri artırmayı yalnızca eğitim setine uyguladım. İlk olarak 20 epoch tam dengeli ağırlıklarla eğittim (Deney 1), ardından ağırlıkları karekök (sqrt) ile yumuşatıp 10 epoch fine-tuning yaparak nihai modeli (models/final_model.pth) elde ettim.

5. Başarım Sonuçları (FER2013 Test, 7.178 Görüntü)

┌─────────┬──────────────────────────────┬──────────┬──────────┐
│ Deney   │ Yöntem                       │ Accuracy │ Macro-F1 │
├─────────┼──────────────────────────────┼──────────┼──────────┤
│ Deney 1 │ 20 epoch, tam ağırlık (1e-3) │ %48.20   │ 0.4057   │
│ Deney 2 │ +10 epoch, sqrt ağır. (5e-4) │ %60.69   │ 0.5562   │
└─────────┴──────────────────────────────┴──────────┴──────────┘

En yüksek F1 skorları happy (0.83) ve surprise (0.75) sınıflarında çıktı. En çok karışan çiftler sad -> neutral (311) ve fear -> sad (263) oldu.

6. Gerçek Zamanlı Performans (1000 Kare, CPU)

┌───────────────────────┬───────────────┬──────────────┐
│ Aşama                 │ Ortalama (ms) │ p95 (ms)     │
├───────────────────────┼───────────────┼──────────────┤
│ Yüz Tespiti           │ 2.22          │ 3.06         │
│ Ön İşleme (48x48)     │ 0.20          │ 0.28         │
│ CNN Model Çıkarımı    │ 2.27          │ 3.31         │
│ Toplam Kare (Kamera)  │ 41.82         │ 63.00        │
└───────────────────────┴───────────────┴──────────────┘

Ortalama 23.91 FPS ölçüldü (15 FPS hedefi rahatça aşıldı). Saf yapay zekâ süresi kare başına yalnızca 4.69 ms sürdü.

7. Karşılaşılan Sorunlar ve Çözümler
- İlk eğitimde yüksek disgust ağırlığı yanlış pozitifleri artırdı; karekök ağırlık ve fine-tuning ile doğruluk %48.20 seviyesinden %60.69 seviyesine çıktı.
- Windows ortamında DataLoader takılmasını önlemek için num_workers=0 kullandım.
- pytest tüm klasörleri tarayıp kamera betiğini açınca testleri python -m pytest -v tests/ ile çalıştırarak 5/5 PASSED sonucunu aldım.

8. Sınırlılıklar ve Etik Notlar
- Düşük çözünürlük (48x48) ve veri gürültüsü nedeniyle fear ve disgust sınıflarında başarım daha sınırlı kaldı; aşırı loş ışıkta güven düşüyor.
- Kamera görüntüleri kalıcı olarak kaydedilmez. Sistem yalnızca görsel yüz ifadesi tahmini yapar, kesin bir psikolojik ölçüm değildir.

9. Gelecek Çalışmalar
Daha dengeli veri setleri (AffectNet, RAF-DB), MobileNetV3/ResNet ile transfer öğrenme ve kayıtlı video analizi eklenebilir.

10. Sonuç
15 günlük plan sonunda %60.69 test doğruluğu, 23.91 FPS canlı hız ve 5/5 geçen otomatik testlerle çalışan, modüler ve belgelenmiş bir duygu analizi sistemi (v1.0) teslim edildi.

Ek: 15 Günlük Özet
- Gün 01–04: Kapsam, sanal ortam, Git ve literatür özeti.
- Gün 05–06: Canlı kamera prototipi ve yüz tespiti.
- Gün 07–08: FER2013 analizi, ön işleme, artırma ve DataLoader.
- Gün 09–11: EmotionCNN tasarımı, 20 epoch eğitim, değerlendirme ve iyileştirme (%60.69).
- Gün 12–13: Canlı entegrasyon, 10 kare yumuşatma, olasılık çubukları ve 23.91 FPS ölçümü.
- Gün 14–15: pytest (5/5), senaryo testleri, README, final raporu ve v1.0 teslimi.