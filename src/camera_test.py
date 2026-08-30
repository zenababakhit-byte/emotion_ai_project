import cv2
import time


# Kamerayi baslat
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("HATA: Kamera acilamadi!")
    exit()

print("=" * 50)
print("GERCEK ZAMANLI YUZ IFADELERI VE DUYGU ANALIZI")
print("=" * 50)
print("Kamera basariyla acildi.")
print("Programdan cikmak icin Q tusuna basin.")


previous_time = time.time()

while True:

    # Kameradan bir kare al
    success, frame = camera.read()

    if not success:
        print("HATA: Kameradan goruntu alinamadi!")
        break

    # FPS hesaplama
    current_time = time.time()

    elapsed_time = current_time - previous_time

    if elapsed_time > 0:
        fps = 1 / elapsed_time
    else:
        fps = 0

    previous_time = current_time

    # FPS bilgisini ekrana yaz
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Baslik
    cv2.putText(
        frame,
        "Emotion AI - Camera Test",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # Goruntuyu goster
    cv2.imshow(
        "Gercek Zamanli Duygu Analizi",
        frame
    )

    # Q tusuna basilirsa cik
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# Kaynaklari serbest birak
camera.release()

cv2.destroyAllWindows()

print("Kamera kapatildi.")
print("Program basariyla sonlandirildi.")