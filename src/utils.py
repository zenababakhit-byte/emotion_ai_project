import csv
import math
import os
import time
from collections import deque
import cv2
import numpy as np


class MultiFaceSmoother:
    """Her yuz icin son `window` karenin olasilik vektorunu deque ile yumusatir."""

    def __init__(self, class_names, window: int = 10, max_dist: float = 100.0):
        self.class_names = class_names
        self.window = window
        self.max_dist = max_dist
        self.tracks = {}  # track_id -> {"center": (cx, cy), "probs": deque, "last_seen": int}
        self.next_id = 0
        self.frame_idx = 0

    def step_frame(self):
        self.frame_idx += 1
        # 20 karedir gorulmeyen eski yuzleri temizle
        stale = [tid for tid, t in self.tracks.items() if self.frame_idx - t["last_seen"] > 20]
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

        vec = np.array([probs_dict[c] for c in self.class_names], dtype=np.float32)
        self.tracks[best_id]["center"] = (cx, cy)
        self.tracks[best_id]["last_seen"] = self.frame_idx
        self.tracks[best_id]["probs"].append(vec)

        avg_vec = np.mean(self.tracks[best_id]["probs"], axis=0)
        best_cls_idx = int(np.argmax(avg_vec))
        best_conf = float(avg_vec[best_cls_idx])
        raw_label = self.class_names[best_cls_idx]
        display_label = raw_label if best_conf >= threshold else "Belirsiz"
        avg_dict = {self.class_names[i]: float(avg_vec[i]) for i in range(len(self.class_names))}
        return display_label, best_conf, avg_dict


class PerformanceLogger:
    """Kare bazli sure olcumlerini tutar, reports/performans.csv yazar ve ozet basar."""

    def __init__(self):
        self.records = []

    def log(self, frame_no, num_faces, detect_ms, preprocess_ms, infer_ms, total_ms):
        fps = 1000.0 / max(total_ms, 1e-3)
        self.records.append({
            "frame": frame_no,
            "faces": num_faces,
            "detect_ms": round(detect_ms, 3),
            "preprocess_ms": round(preprocess_ms, 3),
            "infer_ms": round(infer_ms, 3),
            "total_ms": round(total_ms, 3),
            "fps": round(fps, 2),
        })

    def save_and_summarize(self, out_csv="reports/performans.csv"):
        if not self.records:
            print("Kaydedilecek performans verisi yok.")
            return

        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        fieldnames = ["frame", "faces", "detect_ms", "preprocess_ms", "infer_ms", "total_ms", "fps"]
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.records)

        det = np.array([r["detect_ms"] for r in self.records])
        pre = np.array([r["preprocess_ms"] for r in self.records])
        inf = np.array([r["infer_ms"] for r in self.records])
        tot = np.array([r["total_ms"] for r in self.records])
        avg_fps = 1000.0 / max(float(np.mean(tot)), 1e-3)

        print("\n" + "=" * 60)
        print(f"PERFORMANS OZETI ({len(self.records)} kare) -> {out_csv}")
        print("=" * 60)
        print(f"Ortalama FPS             : {avg_fps:.2f} FPS (Hedef >= 15 FPS)")
        print(f"Yuz tespiti (ort / p95)  : {np.mean(det):.2f} ms / {np.percentile(det, 95):.2f} ms")
        print(f"On isleme   (ort / p95)  : {np.mean(pre):.2f} ms / {np.percentile(pre, 95):.2f} ms")
        print(f"Model cik.  (ort / p95)  : {np.mean(inf):.2f} ms / {np.percentile(inf, 95):.2f} ms")
        print(f"Toplam kare (ort / p95)  : {np.mean(tot):.2f} ms / {np.percentile(tot, 95):.2f} ms")
        print("=" * 60)


def draw_probability_bars(frame, probs_dict, colors):
    """Sag ust kosede 7 duygu icin yari saydam olasilik cubuklari cizer."""
    if not probs_dict:
        return

    h_img, w_img = frame.shape[:2]
    panel_w = 215
    row_h = 24
    panel_h = 28 + len(probs_dict) * row_h
    x0 = w_img - panel_w - 10
    y0 = 10

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    cv2.putText(
        frame, "Duygu Olasiliklari", (x0 + 8, y0 + 18),
        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1,
    )

    bar_x = x0 + 82
    bar_max_w = 88
    for idx, (cls_name, prob) in enumerate(probs_dict.items()):
        ry = y0 + 36 + idx * row_h
        color = colors.get(cls_name, (200, 200, 200))

        cv2.putText(
            frame, cls_name.capitalize(), (x0 + 8, ry),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1,
        )
        cv2.rectangle(frame, (bar_x, ry - 11), (bar_x + bar_max_w, ry + 2), (70, 70, 70), -1)
        fill_w = int(bar_max_w * max(0.0, min(1.0, prob)))
        cv2.rectangle(frame, (bar_x, ry - 11), (bar_x + fill_w, ry + 2), color, -1)
        cv2.putText(
            frame, f"{prob * 100:4.0f}%", (bar_x + bar_max_w + 5, ry),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1,
        )