import csv
import os
import time
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


class SessionAnalytics:
    """Oturum boyunca tum duygu tahminlerini kaydeder; CSV + 2 panelli PNG raporu uretir."""

    def __init__(self, session_name: str = "session"):
        self.session_name = session_name
        self.start_time = time.perf_counter()
        self.rows = []

    def record(self, person_id: int, emotion: str, confidence: float, probs_dict=None):
        elapsed = round(time.perf_counter() - self.start_time, 2)
        row = {
            "time_sec": elapsed,
            "person_id": person_id,
            "emotion": emotion,
            "confidence": round(confidence, 4),
        }
        if probs_dict:
            for k, v in probs_dict.items():
                row[f"prob_{k}"] = round(float(v), 4)
        self.rows.append(row)

    def get_summary(self):
        if not self.rows:
            return {"total_records": 0, "duration_sec": 0.0, "dominant": "-", "unique_people": 0}
        counts = Counter(r["emotion"] for r in self.rows)
        dominant, dom_cnt = counts.most_common(1)[0]
        return {
            "total_records": len(self.rows),
            "duration_sec": self.rows[-1]["time_sec"],
            "dominant": dominant,
            "dominant_pct": round(100.0 * dom_cnt / len(self.rows), 1),
            "unique_people": len({r["person_id"] for r in self.rows}),
            "distribution": dict(counts),
        }

    def export(self, out_dir: str = "reports/sessions"):
        if not self.rows:
            return None, None
        os.makedirs(out_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(out_dir, f"{self.session_name}_{ts}.csv")
        png_path = os.path.join(out_dir, f"{self.session_name}_{ts}.png")

        df = pd.DataFrame(self.rows)
        df.to_csv(csv_path, index=False, encoding="utf-8")

        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

        # 1. Duygu Dagilimi Bar Grafigi
        counts = df["emotion"].value_counts()
        pcts = (counts / len(df)) * 100.0
        bars = axes[0].bar(pcts.index, pcts.values, color="#2E7D9B")
        axes[0].set_title("Oturum Duygu Dagilimi (%)")
        axes[0].set_ylabel("Oran (%)")
        axes[0].tick_params(axis="x", rotation=30)
        axes[0].grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, pcts.values):
            axes[0].text(bar.get_x() + bar.get_width() / 2, val + 1, f"%{val:.1f}", ha="center", fontsize=9)

        # 2. Zaman Cizelgesi (Guven Skoru + Duygu Degisimi)
        emotions = list(counts.index)
        emo_to_y = {e: i for i, e in enumerate(emotions)}
        y_vals = [emo_to_y[e] for e in df["emotion"]]
        axes[1].scatter(df["time_sec"], y_vals, c=df["confidence"], cmap="viridis", s=16, alpha=0.8)
        axes[1].set_yticks(range(len(emotions)))
        axes[1].set_yticklabels(emotions)
        axes[1].set_title("Zaman Icinde Duygu Degisimi (Renk: Guven)")
        axes[1].set_xlabel("Sure (saniye)")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(png_path, dpi=150)
        plt.close()
        return csv_path, png_path