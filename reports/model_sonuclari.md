Model Değerlendirme ve Karşılaştırma Sonuçları

Test Veri Seti: FER2013 Test (7.178 görüntü, 7 sınıf)
Nihai Seçilen Model: models/final_model.pth (Deney 2)

Deney Karşılaştırma Tablosu

| Deney   | Açıklama                                         | Test Accuracy | Macro-F1 | Weighted-F1 |
| ------- | ------------------------------------------------ | ------------- | -------- | ----------- |
| Deney 1 | 20 epoch, tam dengeli sınıf ağırlıkları (lr=1e-3)| %48.20        | 0.4057   | 0.4669      |
| Deney 2 | +10 epoch fine-tuning, sqrt ağırlıklar (lr=5e-4) | %60.69        | 0.5562   | 0.6002      |

Nihai Model (Deney 2) Sınıf Bazlı Metrikler

| Sınıf    | Precision | Recall | F1-Score | Destek (Adet) |
| -------- | --------- | ------ | -------- | ------------- |
| angry    | 0.5722    | 0.4718 | 0.5172   | 958           |
| disgust  | 0.2895    | 0.5946 | 0.3894   | 111           |
| fear     | 0.4717    | 0.2520 | 0.3285   | 1024          |
| happy    | 0.8470    | 0.8207 | 0.8337   | 1774          |
| neutral  | 0.5107    | 0.6772 | 0.5823   | 1233          |
| sad      | 0.4689    | 0.5132 | 0.4900   | 1247          |
| surprise | 0.7260    | 0.7810 | 0.7525   | 831           |

Hata Analizi
- En yüksek başarım belirgin yüz özelliklerine sahip happy (%83.4 F1) ve surprise (%75.3 F1) sınıflarında elde edildi.
- En çok karışan sınıf çiftleri: sad -> neutral (311), fear -> sad (263), neutral -> sad (200), fear -> neutral (168) ve angry -> neutral (152).
- Üzgün ve nötr ifadelerin yüz hatları düşük çözünürlükte (48x48) birbirine çok benzediği için en sık karışma bu çiftte görüldü.
- Disgust (111 örnek) ve fear sınıfları veri setindeki az örnek ve etiket gürültüsü nedeniyle diğer sınıflara göre daha düşük F1 skorunda kaldı.