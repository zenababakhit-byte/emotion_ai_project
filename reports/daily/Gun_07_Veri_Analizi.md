7. Gün – FER2013 Veri Setinin İndirilmesi ve Keşifsel Veri Analizi

Amaç: Bugün, model eğitiminde kullanacağım FER2013 veri setini projeye eklemeyi, yapısını tanımayı ve sınıf dağılımına bakarak olası sorunları önceden görmeyi hedefledim.

Yapılan Çalışmalar: FER2013 veri setini Kaggle’dan indirip data/fer2013/ klasörüne çıkardım. Veri seti train ve test klasörlerinden oluşuyor ve her birinin içinde yedi duygu sınıfı için ayrı bir alt klasör var. Veri seti çok sayıda dosya içerdiği için Git’in bunları takip etmesini istemedim ve .gitignore dosyasına data/fer2013/ satırını ekledim.

Ardından src/data_explore.py dosyasını yazdım. Bu dosya Pandas ile her sınıftaki görüntü sayısını tabloya döküyor, Matplotlib ile sınıf dağılımı grafiğini çiziyor ve her sınıftan rastgele 8 görüntüyü bir ızgarada gösteriyor. Çıktıları reports/sinif_dagilimi.csv, reports/sinif_dagilimi.png ve reports/ornek_yuzler.png olarak kaydettim.

Kullanılan Teknolojiler: Python, Pandas, Matplotlib, Pillow, Git, Visual Studio Code

Elde Edilen Sonuç: Veri setinde 28.709 eğitim ve 7.178 test olmak üzere toplam 35.887 görüntü var. Sınıflara göre dağılım şöyle:

| Sınıf | Eğitim | Test | Eğitimdeki oran |
|---|---|---|---|
| Happy | 7.215 | 1.774 | %25,1 |
| Neutral | 4.965 | 1.233 | %17,3 |
| Sad | 4.830 | 1.247 | %16,8 |
| Fear | 4.097 | 1.024 | %14,3 |
| Angry | 3.995 | 958 | %13,9 |
| Surprise | 3.171 | 831 | %11,0 |
| Disgust | 436 | 111 | %1,5 |

Görüntülerin hepsi 48x48 piksel boyutunda ve gri tonlamalı. Dağılım dengeli değil: Happy sınıfında Disgust sınıfından yaklaşık 16 kat fazla örnek var. Bu yüzden modelin Disgust sınıfını öğrenmekte zorlanmasını bekliyorum. 8. günde sınıf ağırlıklarını kullanarak bu sorunu azaltmaya çalışacağım. Disgust için test setinde yalnızca 111 görüntü olduğundan, bu sınıfın test sonuçlarına da biraz temkinli yaklaşmam gerekecek.

Karşılaşılan Sorunlar ve Çözüm: git status çıktısında data/ klasörünün takip edilmeyen dosyalar arasında göründüğünü fark ettim. .gitignore dosyasında yalnızca data/raw/ ve data/processed/ yazıyordu. data/fer2013/ satırını ekleyince yaklaşık 36.000 görüntü Git’in dışında kaldı.

Örnek görüntülere bakarken FER2013'ün etiketlerinde gürültü olduğunu da gördüm. Bazı görüntülerde filigran var, bazıları heykel ya da çizgi film karakteri, bazılarında yüz el veya telefonla kısmen kapalı ve bazı etiketler (özellikle Fear) tartışmaya açık. Bu yüzden modelin başarısının sınırlı kalabileceğini şimdiden hesaba katıyorum.

Ekran Görüntüsü: Görüntülerde sınıf dağılımı grafiği, terminaldeki sayı tablosu ve her sınıftan örnek yüzlerin bulunduğu ızgara yer alıyor.

Ertesi Gün Planı: Yarın veriyi PyTorch ile eğitime hazır hale getireceğim. Dönüşümleri tanımlayacağım, eğitim verisini eğitim ve doğrulama olarak ayıracağım, veri artırma ekleyeceğim, sınıf ağırlıklarını hesaplayacağım ve DataLoader yapısını kuracağım.7. Gün – FER2013 Veri Setinin İndirilmesi ve Keşifsel Veri Analizi

Amaç: Bugün, model eğitiminde kullanacağım FER2013 veri setini projeye eklemeyi, yapısını tanımayı ve sınıf dağılımına bakarak olası sorunları önceden görmeyi hedefledim.

Yapılan Çalışmalar: FER2013 veri setini Kaggle’dan indirip data/fer2013/ klasörüne çıkardım. Veri seti train ve test klasörlerinden oluşuyor ve her birinin içinde yedi duygu sınıfı için ayrı bir alt klasör var. Veri seti çok sayıda dosya içerdiği için Git’in bunları takip etmesini istemedim ve .gitignore dosyasına data/fer2013/ satırını ekledim.

Ardından src/data_explore.py dosyasını yazdım. Bu dosya Pandas ile her sınıftaki görüntü sayısını tabloya döküyor, Matplotlib ile sınıf dağılımı grafiğini çiziyor ve her sınıftan rastgele 8 görüntüyü bir ızgarada gösteriyor. Çıktıları reports/sinif_dagilimi.csv, reports/sinif_dagilimi.png ve reports/ornek_yuzler.png olarak kaydettim.

Kullanılan Teknolojiler: Python, Pandas, Matplotlib, Pillow, Git, Visual Studio Code

Elde Edilen Sonuç: Veri setinde 28.709 eğitim ve 7.178 test olmak üzere toplam 35.887 görüntü var. Sınıflara göre dağılım şöyle:

Sınıf	Eğitim	Test	Eğitimdeki oran
Happy	7.215	1.774	%25,1
Neutral	4.965	1.233	%17,3
Sad	4.830	1.247	%16,8
Fear	4.097	1.024	%14,3
Angry	3.995	958	%13,9
Surprise	3.171	831	%11,0
Disgust	436	111	%1,5

Görüntülerin hepsi 48x48 piksel boyutunda ve gri tonlamalı. Dağılım dengeli değil: Happy sınıfında Disgust sınıfından yaklaşık 16 kat fazla örnek var. Bu yüzden modelin Disgust sınıfını öğrenmekte zorlanmasını bekliyorum. 8. günde sınıf ağırlıklarını kullanarak bu sorunu azaltmaya çalışacağım. Disgust için test setinde yalnızca 111 görüntü olduğundan, bu sınıfın test sonuçlarına da biraz temkinli yaklaşmam gerekecek.

Karşılaşılan Sorunlar ve Çözüm: git status çıktısında data/ klasörünün takip edilmeyen dosyalar arasında göründüğünü fark ettim. .gitignore dosyasında yalnızca data/raw/ ve data/processed/ yazıyordu. data/fer2013/ satırını ekleyince yaklaşık 36.000 görüntü Git’in dışında kaldı.

Örnek görüntülere bakarken FER2013'ün etiketlerinde gürültü olduğunu da gördüm. Bazı görüntülerde filigran var, bazıları heykel ya da çizgi film karakteri, bazılarında yüz el veya telefonla kısmen kapalı ve bazı etiketler (özellikle Fear) tartışmaya açık. Bu yüzden modelin başarısının sınırlı kalabileceğini şimdiden hesaba katıyorum.

Ekran Görüntüsü: Görüntülerde sınıf dağılımı grafiği, terminaldeki sayı tablosu ve her sınıftan örnek yüzlerin bulunduğu ızgara yer alıyor.

Ertesi Gün Planı: Yarın veriyi PyTorch ile eğitime hazır hale getireceğim. Dönüşümleri tanımlayacağım, eğitim verisini eğitim ve doğrulama olarak ayıracağım, veri artırma ekleyeceğim, sınıf ağırlıklarını hesaplayacağım ve DataLoader yapısını kuracağım.