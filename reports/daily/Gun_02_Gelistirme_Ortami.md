2. Gün – Geliştirme Ortamı Kurulumu

Amaç
Python 3.14.7 sanal ortamını (venv) kurmak, gerekli kütüphaneleri yükleyip çalıştığını doğrulamak ve requirements.txt dosyasını oluşturmak.

Yapılan Çalışmalar
- python -m venv venv ile sanal ortamı kurup aktif ettim.
- OpenCV, NumPy, Pandas, Matplotlib ve PyTorch (CPU) paketlerini yükledim.
- src/environment_test.py dosyasını yazıp çalıştırarak paket sürümlerini ve CUDA durumunu test ettim.
- pip freeze > requirements.txt ile bağımlılık listesini kaydettim.

Kullanılan Teknolojiler
- Python 3.14.7, venv, OpenCV 5.0.0, NumPy 2.5.2, Pandas 3.0.5, Matplotlib 3.11.1, PyTorch 2.13.0+cpu

Elde Edilen Sonuç
- Tüm kütüphaneler hatasız çalıştı, CUDA: False (CPU modu) doğrulandı ve requirements.txt kaydedildi.

Karşılaşılan Sorunlar ve Çözüm
- CUDA destekli GPU bulunmadığı görüldüğü için ilerleyen günlerde CPU üzerinde hızlı çalışacak hafif bir CNN modeli seçilmesi kararlaştırıldı.

Ekran Görüntüsü
- screenshots/Gun_02_Ekran_01.png: src/environment_test.py kodu ve tüm sürüm bilgilerini gösteren terminal çıktısı.

Ertesi Gün Planı
3. günde Git deposu başlatılacak, .gitignore ayarlanacak ve ilk commit atılacaktır.