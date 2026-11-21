import os
import cv2
import numpy as np

from src.main import SimpleFaceDetector, crop_face_with_padding


def test_detector_handles_blank_image_without_crash():
    detector = SimpleFaceDetector()
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    boxes = detector.detect(blank)
    assert isinstance(boxes, list)
    assert len(boxes) == 0


def test_detector_on_saved_sample_if_available():
    sample_path = "screenshots/gun13_arayuz.png"
    if not os.path.exists(sample_path):
        return

    frame = cv2.imread(sample_path)
    assert frame is not None
    detector = SimpleFaceDetector()
    boxes = detector.detect(frame)
    assert isinstance(boxes, list)

    for box in boxes:
        roi, (x1, y1, x2, y2) = crop_face_with_padding(frame, box, pad_ratio=0.15)
        assert roi is not None and roi.size > 0
        assert 0 <= x1 < x2 <= frame.shape[1]
        assert 0 <= y1 < y2 <= frame.shape[0]