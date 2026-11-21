4. Gün – Teknik Literatür Özeti ve Yaklaşım

1. Bilgisayarlı Görü Nedir?

Bilgisayarlı görü, bilgisayarların görüntü ve videolardan anlamlı bilgi elde etmesini sağlayan yapay zekâ alanıdır.

Bizim projemizde bilgisayarlı görü şu işlemleri gerçekleştirecek:

Kamera
   ↓
Video Karesi
   ↓
Yüz Tespiti
   ↓
Yüzün Kırpılması
   ↓
Duygu Analizi
   ↓
Sonucun Gösterilmesi

Örneğin kamera bir görüntü üretir:

[ Kamera Görüntüsü ]

       😀

Sistem önce görüntüde yüzün nerede olduğunu bulur:

┌─────────────┐
│             │
│     😀      │
│             │
└─────────────┘

Daha sonra bu yüz görüntüsü yapay zekâ modeline gönderilir.

2. Yüz Tespiti Nedir?

Yüz tespiti, bir görüntü veya video içerisindeki insan yüzlerinin konumunu belirleme işlemidir.

Sistem genellikle yüzün konumunu şu dört değer ile temsil eder:

(x, y, width, height)

Örneğin:

x = 100
y = 80
width = 200
height = 200

Bu bilgiler kullanılarak yüzün etrafına bir bounding box (sınırlayıcı kutu) çizilir.

Projemizde yüz tespiti için şu seçenekler değerlendirilebilir:

- MediaPipe
- OpenCV Haar Cascade
- OpenCV DNN

Plana göre proje için MediaPipe veya OpenCV DNN tabanlı yüz tespiti önerilmektedir.

Bizim seçimimiz: MediaPipe

Neden?
- Gerçek zamanlı çalışmaya uygundur.
- Yüz tespiti hızlıdır.
- Birden fazla yüzü algılayabilir.
- Kullanımı görece kolaydır.
- OpenCV ile birlikte kullanılabilir.

3. Duygu Tanıma Nedir?

Yüz tespiti yapıldıktan sonra yüz bölgesi kırpılır.

Kamera görüntüsü
       ↓
Yüz tespiti
       ↓
Yüz bölgesi
       ↓
CNN modeli
       ↓
Duygu tahmini

Model aşağıdaki sınıflardan birini tahmin eder:

┌──────────┬──────────────────┐
│ Duygu    │ İngilizce Etiket │
├──────────┼──────────────────┤
│ Kızgın   │ Angry            │
│ Tiksinme │ Disgust          │
│ Korku    │ Fear             │
│ Mutlu    │ Happy            │
│ Üzgün    │ Sad              │
│ Şaşkın   │ Surprise         │
│ Nötr     │ Neutral          │
└──────────┴──────────────────┘

Bu yedi sınıf proje planında belirlenmiştir.

4. CNN Nedir?

CNN (Convolutional Neural Network), görüntüleri analiz etmek için kullanılan bir yapay sinir ağı türüdür.

CNN bir yüz görüntüsündeki özellikleri öğrenmeye çalışır:

Giriş Yüzü
    ↓
Convolution
    ↓
ReLU
    ↓
Pooling
    ↓
Convolution
    ↓
Pooling
    ↓
Fully Connected Layer
    ↓
Softmax
    ↓
Duygu Tahmini

Örneğin:

Happy     : 0.78
Neutral   : 0.12
Surprise  : 0.05
Sad       : 0.03
Angry     : 0.02

Model en yüksek olasılığı seçer:

Tahmin: Happy
Güven: %78

5. Gerçek Zamanlı Çıkarım Nedir?

Real-Time Inference, kameradan gelen görüntülerin sürekli olarak analiz edilmesidir.

Örnek akış:

while kamera_acik:
    görüntü = kameradan_kare_al()
    yüzleri_bul()
    her_yüz_için:
        yüzü_kırp()
        ön_işleme_yap()
        duygu_tahmini_yap()
    sonucu_ekranda_göster()

Bu işlem saniyede birçok kez gerçekleşir.

Performans FPS (Frames Per Second) ile ölçülebilir:

FPS = 25

Bu, sistemin saniyede yaklaşık 25 görüntü karesi işlediğini gösterir.

Gün 04 planında özellikle gerçek zamanlı çıkarım kavramının incelenmesi istenmektedir.

6. Projemiz İçin Teknik Yaklaşım

Bugün sonunda şu teknolojileri seçiyoruz:

┌────────────────────┬──────────────┐
│ Bileşen            │ Seçim        │
├────────────────────┼──────────────┤
│ Programlama dili   │ Python       │
│ Kamera ve görüntü  │ OpenCV       │
│ Yüz tespiti        │ MediaPipe    │
│ Duygu modeli       │ PyTorch      │
│ Model türü         │ CNN          │
│ Veri seti          │ FER2013      │
│ Veri analizi       │ Pandas       │
│ Görselleştirme     │ Matplotlib   │
│ Değerlendirme      │ Scikit-learn │
│ Sürüm kontrolü     │ Git          │
└────────────────────┴──────────────┘