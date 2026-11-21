7. Gün – FER2013 Veri Setinin İndirilmesi ve Keşifsel Veri Analizi

Amaç
FER2013 veri setini projeye ekleyip klasör yapısını doğrulamak, sınıf dağılımını incelemek ve olası veri dengesizliği ile etiket sorunlarını önceden tespit etmek.

Yapılan Çalışmalar
- FER2013 veri setini indirip data/fer2013/ (train ve test) klasörüne çıkardım ve .gitignore dosyasına data/fer2013/ satırını ekledim.
- src/data_explore.py dosyasını yazarak Pandas ile sınıf sayılarını çıkardım, Matplotlib ile dağılım grafiği ve her sınıftan 8 rastgele örnek yüz ızgarası ürettim.
- Çıktıları reports/sinif_dagilimi.csv, reports/sinif_dagilimi.png ve reports/ornek_yuzler.png olarak kaydettim.

Kullanılan Teknolojiler
- Python, Pandas, Matplotlib, Pillow, Git, Visual Studio Code

Elde Edilen Sonuç
Toplam 35.887 adet 48x48 gri tonlamalı görüntü (28.709 eğitim, 7.178 test) doğrulandı:

┌──────────┬────────┬───────┬─────────────────┐
│ Sınıf    │ Eğitim │ Test  │ Eğitimdeki Oran │
├──────────┼────────┼───────┼─────────────────┤
│ Happy    │ 7.215  │ 1.774 │ %25.1           │
│ Neutral  │ 4.965  │ 1.233 │ %17.3           │
│ Sad      │ 4.830  │ 1.247 │ %16.8           │
│ Fear     │ 4.097  │ 1.024 │ %14.3           │
│ Angry    │ 3.995  │ 958   │ %13.9           │
│ Surprise │ 3.171  │ 831   │ %11.0           │
│ Disgust  │ 436    │ 111   │ %1.5            │
└──────────┴────────┴───────┴─────────────────┘

Happy sınıfı Disgust sınıfından yaklaşık 16 kat fazla örnek içeriyor; bu dengesizliği 8. günde sınıf ağırlıklarıyla ele alacağım.

Karşılaşılan Sorunlar ve Çözüm
- git status çıktısında data/fer2013/ takip dışı göründü; .gitignore dosyasına data/fer2013/ eklenerek yaklaşık 36.000 görüntü Git dışında bırakıldı.
- Örnek görsellerde filigranlı, kısmen kapalı veya gürültülü etiketli görüntüler fark edildi ve sınırlılık olarak not edildi.

Ekran Görüntüsü
- screenshots/Gun_07_Ekran_01.png ve screenshots/Gun_07_Ekran_02.png: Sınıf dağılımı grafiği, sayı tablosu ve 7 sınıftan örnek yüz ızgarası.

Ertesi Gün Planı
8. günde veriyi PyTorch için hazırlayacağım: dönüşümler, %80/%20 eğitim/doğrulama bölmesi, veri artırma, sınıf ağırlıkları ve DataLoader kurulacaktır.