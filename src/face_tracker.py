import math
from collections import Counter, deque
import cv2
import numpy as np


class FaceAligner:
    """Goz merkezlerini bularak yuz acisini yatay hizalar; goz bulunamazsa orijinali dondurur."""

    def __init__(self, max_angle: float = 25.0):
        self.max_angle = max_angle
        eye_xml = cv2.data.haarcascades + "haarcascade_eye.xml"
        self.eye_cascade = cv2.CascadeClassifier(eye_xml)

    def align(self, face_bgr):
        if self.eye_cascade.empty() or face_bgr is None or face_bgr.size == 0:
            return face_bgr
        h, w = face_bgr.shape[:2]
        if h < 40 or w < 40:
            return face_bgr

        upper_gray = cv2.cvtColor(face_bgr[: int(h * 0.6), :], cv2.COLOR_BGR2GRAY)
        eyes = self.eye_cascade.detectMultiScale(
            upper_gray, scaleFactor=1.1, minNeighbors=5, minSize=(12, 12)
        )
        if len(eyes) < 2:
            return face_bgr

        # En buyuk iki gozu sec ve soldan saga sirala
        eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]
        eyes = sorted(eyes, key=lambda e: e[0])
        (x1, y1, w1, h1), (x2, y2, w2, h2) = eyes
        left_center = (x1 + w1 / 2.0, y1 + h1 / 2.0)
        right_center = (x2 + w2 / 2.0, y2 + h2 / 2.0)

        dx = right_center[0] - left_center[0]
        dy = right_center[1] - left_center[1]
        if dx < 5:
            return face_bgr

        angle = math.degrees(math.atan2(dy, dx))
        if abs(angle) > self.max_angle or abs(angle) < 1.5:
            return face_bgr

        center = (w / 2.0, h / 2.0)
        mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(face_bgr, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


class MultiPersonTracker:
    """Her kisiye kalici ID (#1, #2, ...) atar, olasiliklari yumusatir ve kisi bazli ozet tutar."""

    def __init__(self, class_names, window: int = 10, max_dist: float = 110.0, max_missing: int = 25):
        self.class_names = class_names
        self.window = window
        self.max_dist = max_dist
        self.max_missing = max_missing
        self.tracks = {}
        self.history_summary = {}
        self.next_id = 1
        self.frame_idx = 0

    def step_frame(self):
        self.frame_idx += 1
        stale = [
            tid for tid, info in self.tracks.items()
            if self.frame_idx - info["last_seen"] > self.max_missing
        ]
        for tid in stale:
            del self.tracks[tid]

    def update(self, box, probs_dict, threshold: float = 0.40):
        x, y, w, h = box
        cx, cy = x + w / 2.0, y + h / 2.0

        best_id = None
        best_dist = self.max_dist
        for tid, info in self.tracks.items():
            px, py = info["center"]
            dist = math.hypot(cx - px, cy - py)
            if dist < best_dist:
                best_dist = dist
                best_id = tid

        if best_id is None:
            best_id = self.next_id
            self.next_id += 1
            self.tracks[best_id] = {
                "center": (cx, cy),
                "probs": deque(maxlen=self.window),
                "last_seen": self.frame_idx,
            }
            self.history_summary[best_id] = Counter()

        vec = np.array([probs_dict[c] for c in self.class_names], dtype=np.float32)
        self.tracks[best_id]["center"] = (cx, cy)
        self.tracks[best_id]["last_seen"] = self.frame_idx
        self.tracks[best_id]["probs"].append(vec)

        avg_vec = np.mean(self.tracks[best_id]["probs"], axis=0)
        best_idx = int(np.argmax(avg_vec))
        conf = float(avg_vec[best_idx])
        raw_label = self.class_names[best_idx]
        label = raw_label if conf >= threshold else "Belirsiz"

        self.history_summary[best_id][label] += 1
        avg_dict = {self.class_names[i]: float(avg_vec[i]) for i in range(len(self.class_names))}
        return best_id, label, conf, avg_dict

    def get_person_summaries(self):
        summaries = {}
        for tid, counts in self.history_summary.items():
            total = sum(counts.values())
            if total == 0:
                continue
            top_label, top_cnt = counts.most_common(1)[0]
            summaries[tid] = {
                "frames": total,
                "dominant": top_label,
                "ratio_pct": round(100.0 * top_cnt / total, 1),
            }
        return summaries
