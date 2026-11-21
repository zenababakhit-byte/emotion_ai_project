import os
import time
import cv2

from src.emotion_predictor import EmotionPredictor

EMOTION_COLORS = {
    "happy": (0, 255, 0),
    "angry": (0, 0, 255),
    "sad": (255, 120, 0),
    "surprise": (0, 255, 255),
    "fear": (255, 0, 255),
    "disgust": (0, 160, 100),
    "neutral": (220, 220, 220),
}


class SimpleFaceDetector:
    """Once src/face_detector.py sinifini dener; uyumsuzluk varsa Haar Cascade kullanir."""

    def __init__(self):
        self.custom_detector = None
        self.haar = None
        try:
            from src.face_detector import FaceDetector
            det = FaceDetector()
            if hasattr(det, "detect"):
                self.custom_detector = det
                print("Yuz dedektoru: src.face_detector.FaceDetector")
                return
        except Exception:
            pass

        cascade_path = "models/haarcascade_frontalface_default.xml"
        if not os.path.exists(cascade_path):
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.haar = cv2.CascadeClassifier(cascade_path)
        print("Yuz dedektoru: OpenCV Haar Cascade")

    def detect(self, frame):
        if self.custom_detector is not None:
            raw = self.custom_detector.detect(frame) or []
            boxes = []
            for item in raw:
                if len(item) >= 4:
                    boxes.append((int(item[0]), int(item[1]), int(item[2]), int(item[3])))
            return boxes

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]


def crop_face_with_padding(frame, box, pad_ratio=0.15):
    h_img, w_img = frame.shape[:2]
    x, y, w, h = box
    px = int(w * pad_ratio)
    py = int(h * pad_ratio)
    x1 = max(0, x - px)
    y1 = max(0, y - py)
    x2 = min(w_img, x + w + px)
    y2 = min(h_img, y + h + py)
    if x2 <= x1 or y2 <= y1:
        return None, (x1, y1, x2, y2)
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


def main():
    detector = SimpleFaceDetector()
    predictor = EmotionPredictor(
        model_path="models/final_model.pth",
        label_map_path="models/label_map.json",
    )

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Kamera acilamadi! Kamera baglantisini veya izinleri kontrol edin.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    os.makedirs("screenshots", exist_ok=True)

    prev_time = time.perf_counter()
    print("Canli duygu analizi basladi. [s]: ekran goruntusu kaydet | [q]: cikis")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Kamera karesi okunamadi, cikiliyor...")
                break

            frame = cv2.flip(frame, 1)
            boxes = detector.detect(frame)
            last_label = "none"

            if not boxes:
                cv2.putText(
                    frame, "Yuz bulunamadi", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 165, 255), 2,
                )
            else:
                for box in boxes:
                    face_roi, (x1, y1, x2, y2) = crop_face_with_padding(frame, box, pad_ratio=0.15)
                    if face_roi is None or face_roi.size == 0:
                        continue

                    label, conf, _ = predictor.predict(face_roi)
                    last_label = label
                    color = EMOTION_COLORS.get(label, (255, 255, 255))
                    text = f"{label.capitalize()} {conf * 100:.0f}%"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    text_y = max(25, y1 - 10)
                    cv2.putText(
                        frame, text, (x1, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2,
                    )

            now = time.perf_counter()
            fps = 1.0 / max(1e-6, now - prev_time)
            prev_time = now
            cv2.putText(
                frame, f"FPS: {fps:.1f} | Yuz: {len(boxes)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2,
            )

            cv2.imshow("Canli Duygu Analizi - Gun 12", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # q veya ESC
                break
            elif key == ord("s"):
                ts = int(time.time())
                out_file = f"screenshots/gun12_canli_{last_label}_{ts}.png"
                cv2.imwrite(out_file, frame)
                print(f"Ekran goruntusu kaydedildi: {out_file}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()