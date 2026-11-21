import argparse
import os
import time
import cv2
import torch
import torch.nn.functional as F

from src.emotion_predictor import EmotionPredictor, preprocess
from src.utils import MultiFaceSmoother, PerformanceLogger, draw_probability_bars

EMOTION_COLORS = {
    "happy": (0, 255, 0),
    "angry": (0, 0, 255),
    "sad": (255, 120, 0),
    "surprise": (0, 255, 255),
    "fear": (255, 0, 255),
    "disgust": (0, 160, 100),
    "neutral": (220, 220, 220),
    "Belirsiz": (160, 160, 160),
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
            return [(int(i[0]), int(i[1]), int(i[2]), int(i[3])) for i in raw if len(i) >= 4]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]


def crop_face_with_padding(frame, box, pad_ratio=0.15):
    h_img, w_img = frame.shape[:2]
    x, y, w, h = box
    px, py = int(w * pad_ratio), int(h * pad_ratio)
    x1, y1 = max(0, x - px), max(0, y - py)
    x2, y2 = min(w_img, x + w + px), min(h_img, y + h + py)
    if x2 <= x1 or y2 <= y1:
        return None, (x1, y1, x2, y2)
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


def main(max_frames: int = 0, smooth_window: int = 10, conf_threshold: float = 0.40):
    detector = SimpleFaceDetector()
    predictor = EmotionPredictor(
        model_path="models/final_model.pth",
        label_map_path="models/label_map.json",
    )
    smoother = MultiFaceSmoother(predictor.class_names, window=smooth_window)
    perf = PerformanceLogger()

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Kamera acilamadi! Kamera baglantisini veya izinleri kontrol edin.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    os.makedirs("screenshots", exist_ok=True)

    frame_count = 0
    print("Uygulama basladi. [s]: ekran goruntusu (gun13_arayuz.png) | [q]: cikis")

    try:
        while True:
            t_frame_start = time.perf_counter()
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Kamera karesi okunamadi, cikiliyor...")
                break

            frame = cv2.flip(frame, 1)
            frame_count += 1
            smoother.step_frame()

            t0 = time.perf_counter()
            boxes = detector.detect(frame)
            detect_ms = (time.perf_counter() - t0) * 1000.0

            preprocess_ms = 0.0
            infer_ms = 0.0
            primary_probs = None

            if not boxes:
                cv2.putText(
                    frame, "Yuz bulunamadi", (10, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 165, 255), 2,
                )
            else:
                for idx, box in enumerate(boxes):
                    face_roi, (x1, y1, x2, y2) = crop_face_with_padding(frame, box, pad_ratio=0.15)
                    if face_roi is None or face_roi.size == 0:
                        continue

                    tp0 = time.perf_counter()
                    tensor = preprocess(face_roi).to(predictor.device)
                    preprocess_ms += (time.perf_counter() - tp0) * 1000.0

                    ti0 = time.perf_counter()
                    with torch.no_grad():
                        logits = predictor.model(tensor)
                        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
                    infer_ms += (time.perf_counter() - ti0) * 1000.0

                    raw_probs = {predictor.idx_to_class[i]: float(probs[i]) for i in range(len(probs))}
                    label, conf, avg_probs = smoother.update(box, raw_probs, threshold=conf_threshold)

                    if idx == 0:
                        primary_probs = avg_probs

                    color = EMOTION_COLORS.get(label, (255, 255, 255))
                    text = f"{label.capitalize()} {conf * 100:.0f}%" if label != "Belirsiz" else "Belirsiz"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame, text, (x1, max(25, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2,
                    )

            draw_probability_bars(frame, primary_probs, EMOTION_COLORS)

            total_ms = (time.perf_counter() - t_frame_start) * 1000.0
            fps = 1000.0 / max(total_ms, 1e-3)
            perf.log(frame_count, len(boxes), detect_ms, preprocess_ms, infer_ms, total_ms)

            cv2.putText(
                frame, f"FPS: {fps:.1f} | Yuz: {len(boxes)} | {total_ms:.1f}ms", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2,
            )

            if max_frames > 0:
                cv2.putText(
                    frame, f"Benchmark: {frame_count}/{max_frames}", (10, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1,
                )

            cv2.imshow("Canli Duygu Analizi - Gun 13", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("s"):
                out_file = "screenshots/gun13_arayuz.png"
                cv2.imwrite(out_file, frame)
                print(f"Arayuz ekran goruntusu kaydedildi: {out_file}")

            if max_frames > 0 and frame_count >= max_frames:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        perf.save_and_summarize("reports/performans.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gercek zamanli duygu analizi uygulamasi")
    parser.add_argument("--benchmark", type=int, default=0, help="Ornegin 1000 kare olcum yapip cik")
    parser.add_argument("--window", type=int, default=10, help="Yumusatma pencere uzunlugu")
    parser.add_argument("--threshold", type=float, default=0.40, help="Belirsiz esigi")
    args = parser.parse_args()
    main(max_frames=args.benchmark, smooth_window=args.window, conf_threshold=args.threshold)