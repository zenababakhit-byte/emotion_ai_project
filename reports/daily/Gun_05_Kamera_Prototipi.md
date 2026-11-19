5. Gün – Web Kamerası Erişimi ve İlk Canlı Kamera Prototipi

Amaç: Bugün OpenCV ile bilgisayarın web kamerasına erişmeyi, kameradan canlı görüntü almayı ve projenin ilk kamera prototipini çalıştırmayı hedefledim.

Yapılan Çalışmalar: Sanal ortamı açtıktan sonra src/camera_test.py dosyasını yazdım. Kamerayı cv2.VideoCapture ile başlattım ve açılıp açılmadığını isOpened() ile kontrol ettim. Ardından her kareyi camera.read() ile okuyup pencerede gösterdim. Ekranın köşesine, son 30 karenin ortalamasıyla hesaplanan FPS değerini ve “Emotion AI - Camera Test” başlığını ekledim. Programı q tuşuyla kapatabiliyorum. Kapanırken release() ve destroyAllWindows() çağırarak kamerayı serbest bırakıyorum.

Kullanılan Teknolojiler: Python, OpenCV, Python Virtual Environment (venv), Visual Studio Code, Git

Elde Edilen Sonuç: İlk canlı kamera prototipi sorunsuz çalıştı: kameradan görüntü alıp ekranda anlık olarak gösteriyor. Ekranda yaklaşık 86 FPS görünüyor, ancak bu değer döngünün hızını gösteriyor ve kameranın gerçek kare hızını tam olarak yansıtmayabilir. Gerçek ölçümü (FPS ve gecikme) 13. günde yapacağım.

Karşılaşılan Sorunlar ve Çözüm: İlk denemede FPS değerini her kare için ayrı hesapladığım için sonuç 176 civarında çıktı ve sürekli dalgalandı. Hesabı son 30 karenin ortalamasına çevirince değer daha kararlı hale geldi. Kamera erişiminde ise herhangi bir sorun yaşamadım.

Ekran Görüntüsü: Görüntüde kamera penceresi, canlı görüntü, FPS değeri ve proje başlığı görülüyor.

Ertesi Gün Planı: Sıradaki adımda MediaPipe’ı kuracağım. Kameradaki yüzleri canlı olarak tespit edip kutu içine alacağım ve bu işlemi daha sonra tekrar kullanabileceğim bir sınıfa dönüştüreceğim.