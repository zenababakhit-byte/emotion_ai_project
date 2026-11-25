import argparse
import os
import cv2

from src.analytics import SessionAnalytics
from src.config import EMOTION_COLORS
from src.emotion_predictor import EmotionPredictor
from src.face_tracker import FaceAligner, MultiPersonTracker
from src.main import SimpleFaceDetector, crop_face_with_padding


def analyze_image(image_path: str, out_path: str = None, use_align: bool = True):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Goruntu okunamadi: {image_path}")

    detector = SimpleFaceDetector()
    predictor = EmotionPredictor()
    aligner = FaceAligner() if use_align else None

    boxes = detector.detect(img)
    results = []
    annotated = img.copy()

    for idx, box in enumerate(boxes, start=1):
        roi, (x1, y1, x2, y2) = crop_face_with_padding(img, box, pad_ratio=0.15)
        if roi is None or roi.size == 0:
            continue
        if aligner is not None:
            roi = aligner.align(roi)
        label, conf, probs = predictor.predict(roi)
        results.append({"id": idx, "label": label, "confidence": conf, "probs": probs})

        color = EMOTION_COLORS.get(label, (255, 255, 255))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated, f"ID#{idx} {label.capitalize()} {conf*100:.0f}%",
            (x1, max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
        )

    if out_path is None:
        os.makedirs("reports/media", exist_ok=True)
        base = os.path.splitext(os.path.basename(image_path))[0]
        out_path = os.path.join("reports/media", f"{base}_analiz.png")

    cv2.imwrite(out_path, annotated)
    print(f"Fotograf analizi tamamlandi ({len(results)} yuz) -> {out_path}")
    for r in results:
        print(f"  ID#{r['id']}: {r['label']} ({r['confidence']*100:.1f}%)")
    return out_path, results


def analyze_video(video_path: str, out_path: str = None, use_align: bool = True):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Video acilamadi: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if out_path is None:
        os.makedirs("reports/media", exist_ok=True)
        base = os.path.splitext(os.path.basename(video_path))[0]
        out_path = os.path.join("reports/media", f"{base}_analiz.mp4")

    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    detector = SimpleFaceDetector()
    predictor = EmotionPredictor()
    aligner = FaceAligner() if use_align else None
    tracker = MultiPersonTracker(predictor.class_names, window=10)
    analytics = SessionAnalytics(session_name="video")

    frames = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frames += 1
            tracker.step_frame()
            boxes = detector.detect(frame)

            for box in boxes:
                roi, (x1, y1, x2, y2) = crop_face_with_padding(frame, box, pad_ratio=0.15)
                if roi is None or roi.size == 0:
                    continue
                if aligner is not None:
                    roi = aligner.align(roi)
                _, _, probs = predictor.predict(roi)
                pid, label, conf, _ = tracker.update(box, probs, threshold=0.40)
                analytics.record(pid, label, conf, probs)

                color = EMOTION_COLORS.get(label, (255, 255, 255))
                text = f"ID#{pid} {label.capitalize()} {conf*100:.0f}%" if label != "Belirsiz" else f"ID#{pid} Belirsiz"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, text, (x1, max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            writer.write(frame)
    finally:
        cap.release()
        writer.release()

    csv_path, png_path = analytics.export()
    print(f"Video analizi tamamlandi ({frames} kare) -> {out_path}")
    print(f"Oturum raporu: {csv_path} | Grafik: {png_path}")
    print("Kisi ozetleri:", tracker.get_person_summaries())
    return out_path, csv_path, png_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fotograf veya video duygu analizi")
    parser.add_argument("--image", type=str, help="Analiz edilecek fotograf yolu")
    parser.add_argument("--video", type=str, help="Analiz edilecek video yolu (.mp4/.avi)")
    parser.add_argument("--no-align", action="store_true", help="Yuz hizalamayi kapat")
    args = parser.parse_args()

    if args.image:
        analyze_image(args.image, use_align=not args.no_align)
    elif args.video:
        analyze_video(args.video, use_align=not args.no_align)
    else:
        print("Kullanim: python -m src.media_analyzer --image yol.jpg  VEYA  --video yol.mp4")