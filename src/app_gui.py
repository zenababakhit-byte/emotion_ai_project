import os
import time
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from PIL import Image, ImageTk

from src.analytics import SessionAnalytics
from src.config import EMOTION_COLORS
from src.emotion_predictor import EmotionPredictor
from src.face_tracker import FaceAligner, MultiPersonTracker
from src.main import SimpleFaceDetector, crop_face_with_padding
from src.media_analyzer import analyze_image, analyze_video

HEX_COLORS = {
    "happy": "#2ECC71",
    "angry": "#E74C3C",
    "sad": "#3498DB",
    "surprise": "#F1C40F",
    "fear": "#9B59B6",
    "disgust": "#1ABC9C",
    "neutral": "#BDC3C7",
    "Belirsiz": "#7F8C8D",
}


class EmotionAppGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Emotion AI v2.0 - Gercek Zamanli Duygu Analizi")
        self.root.geometry("1120x680")
        self.root.configure(bg="#12161F")

        self.detector = SimpleFaceDetector()
        self.predictor = EmotionPredictor()
        self.aligner = FaceAligner()
        self.tracker = MultiPersonTracker(self.predictor.class_names, window=10)
        self.analytics = SessionAnalytics(session_name="gui_live")

        self.cap = None
        self.running = False
        self.use_align = tk.BooleanVar(value=True)
        self.last_frame = None

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_ui(self):
        # Ust baslik cubugu
        header = tk.Frame(self.root, bg="#1B2432", height=54)
        header.pack(fill="x")
        tk.Label(
            header,
            text="Emotion AI v2.0 - Gercek Zamanli Yuz Ifadesi ve Duygu Analizi",
            bg="#1B2432", fg="#ECF0F1", font=("Segoe UI", 14, "bold"),
        ).pack(side="left", padx=16, pady=12)

        self.status_lbl = tk.Label(
            header, text="Hazir", bg="#1B2432", fg="#2ECC71", font=("Segoe UI", 11, "bold")
        )
        self.status_lbl.pack(side="right", padx=16)

        # Ana govde
        body = tk.Frame(self.root, bg="#12161F")
        body.pack(fill="both", expand=True, padx=12, pady=10)

        # Sol video/goruntu alani
        left = tk.Frame(body, bg="#1B2432", bd=1, relief="solid")
        left.pack(side="left", fill="both", expand=True, padx=(0, 10))

        self.video_lbl = tk.Label(
            left,
            text="Kamerayi baslatin veya bir fotograf/video secin",
            bg="#0D1117", fg="#95A5A6", font=("Segoe UI", 13),
        )
        self.video_lbl.pack(fill="both", expand=True, padx=8, pady=8)

        # Alt kontrol butonlari
        controls = tk.Frame(left, bg="#1B2432")
        controls.pack(fill="x", padx=8, pady=(0, 8))

        btn_cfg = {"font": ("Segoe UI", 10, "bold"), "fg": "white", "bd": 0, "padx": 12, "pady": 7, "cursor": "hand2"}
        tk.Button(controls, text="Kamera Baslat", bg="#27AE60", command=self.start_camera, **btn_cfg).pack(side="left", padx=4)
        tk.Button(controls, text="Durdur", bg="#C0392B", command=self.stop_camera, **btn_cfg).pack(side="left", padx=4)
        tk.Button(controls, text="Fotograf Analiz", bg="#2980B9", command=self.open_image, **btn_cfg).pack(side="left", padx=4)
        tk.Button(controls, text="Video Analiz", bg="#8E44AD", command=self.open_video, **btn_cfg).pack(side="left", padx=4)
        tk.Button(controls, text="Rapor Kaydet (CSV+PNG)", bg="#D35400", command=self.export_report, **btn_cfg).pack(side="right", padx=4)

        # Sag analitik paneli
        right = tk.Frame(body, bg="#1B2432", width=330)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        tk.Label(right, text="Canli Istatistikler", bg="#1B2432", fg="#A8D0E6", font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=14, pady=(12, 4))
        self.metrics_lbl = tk.Label(
            right, text="FPS: 0.0  |  Yuz: 0  |  Gecikme: 0 ms",
            bg="#1B2432", fg="#ECF0F1", font=("Consolas", 10),
        )
        self.metrics_lbl.pack(anchor="w", padx=14, pady=(0, 8))

        tk.Checkbutton(
            right, text="Yuz Hizalama (Face Alignment)", variable=self.use_align,
            bg="#1B2432", fg="#ECF0F1", selectcolor="#12161F",
            activebackground="#1B2432", activeforeground="#ECF0F1", font=("Segoe UI", 10),
        ).pack(anchor="w", padx=12, pady=(0, 10))

        tk.Label(right, text="7 Duygu Olasiliklari", bg="#1B2432", fg="#A8D0E6", font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=14, pady=(4, 6))
        self.bar_canvas = tk.Canvas(right, bg="#12161F", height=230, highlightthickness=0)
        self.bar_canvas.pack(fill="x", padx=14, pady=(0, 12))

        tk.Label(right, text="Aktif Kisi ID Ozetleri", bg="#1B2432", fg="#A8D0E6", font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=14, pady=(4, 4))
        self.people_lbl = tk.Label(
            right, text="Henuz kisi tespit edilmedi.", justify="left",
            bg="#12161F", fg="#D5DBDB", font=("Consolas", 10), anchor="nw", padx=10, pady=8,
        )
        self.people_lbl.pack(fill="both", expand=True, padx=14, pady=(0, 14))

    def _draw_bars(self, probs_dict):
        self.bar_canvas.delete("all")
        if not probs_dict:
            return
        y = 14
        for cls_name, prob in probs_dict.items():
            color = HEX_COLORS.get(cls_name, "#BDC3C7")
            self.bar_canvas.create_text(8, y + 7, text=cls_name.capitalize(), anchor="w", fill="#ECF0F1", font=("Segoe UI", 10))
            self.bar_canvas.create_rectangle(88, y, 245, y + 15, fill="#2C3E50", outline="")
            fill_w = int(157 * max(0.0, min(1.0, prob)))
            self.bar_canvas.create_rectangle(88, y, 88 + fill_w, y + 15, fill=color, outline="")
            self.bar_canvas.create_text(252, y + 7, text=f"{prob*100:4.1f}%", anchor="w", fill="#ECF0F1", font=("Consolas", 9))
            y += 30

    def _show_bgr(self, bgr_img):
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        pil.thumbnail((740, 520))
        imgtk = ImageTk.PhotoImage(image=pil)
        self.video_lbl.imgtk = imgtk
        self.video_lbl.configure(image=imgtk, text="")

    def start_camera(self):
        if self.running:
            return
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Kamera Hatasi", "Kamera acilamadi!")
            return
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.running = True
        self.status_lbl.configure(text="Canli Kamera Aktif", fg="#2ECC71")
        self._update_loop()

    def stop_camera(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.status_lbl.configure(text="Durduruldu", fg="#F39C12")

    def _update_loop(self):
        if not self.running or self.cap is None:
            return
        t0 = time.perf_counter()
        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.stop_camera()
            return

        frame = cv2.flip(frame, 1)
        self.tracker.step_frame()
        boxes = self.detector.detect(frame)
        primary_probs = None

        for idx, box in enumerate(boxes):
            roi, (x1, y1, x2, y2) = crop_face_with_padding(frame, box, pad_ratio=0.15)
            if roi is None or roi.size == 0:
                continue
            if self.use_align.get():
                roi = self.aligner.align(roi)
            _, _, raw_probs = self.predictor.predict(roi)
            pid, label, conf, avg_probs = self.tracker.update(box, raw_probs, threshold=0.40)
            self.analytics.record(pid, label, conf, avg_probs)
            if idx == 0:
                primary_probs = avg_probs

            color = EMOTION_COLORS.get(label, (255, 255, 255))
            text = f"ID#{pid} {label.capitalize()} {conf*100:.0f}%" if label != "Belirsiz" else f"ID#{pid} Belirsiz"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, text, (x1, max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        dt_ms = (time.perf_counter() - t0) * 1000.0
        fps = 1000.0 / max(dt_ms, 1e-3)
        self.metrics_lbl.configure(text=f"FPS: {fps:.1f}  |  Yuz: {len(boxes)}  |  {dt_ms:.1f} ms")
        self._draw_bars(primary_probs)

        summaries = self.tracker.get_person_summaries()
        if summaries:
            lines = [
                f"ID#{pid}: {info['dominant'].capitalize()} (%{info['ratio_pct']}, {info['frames']} kare)"
                for pid, info in list(summaries.items())[-6:]
            ]
            self.people_lbl.configure(text="\n".join(lines))

        self.last_frame = frame
        self._show_bgr(frame)
        self.root.after(15, self._update_loop)

    def open_image(self):
        self.stop_camera()
        path = filedialog.askopenfilename(
            title="Fotograf Sec",
            filetypes=[("Goruntu Dosyalari", "*.jpg *.jpeg *.png *.bmp")],
        )
        if not path:
            return
        out_path, results = analyze_image(path, use_align=self.use_align.get())
        annotated = cv2.imread(out_path)
        if annotated is not None:
            self._show_bgr(annotated)
        if results:
            self._draw_bars(results[0]["probs"])
            lines = [f"ID#{r['id']}: {r['label'].capitalize()} ({r['confidence']*100:.1f}%)" for r in results]
            self.people_lbl.configure(text="\n".join(lines))
        self.status_lbl.configure(text=f"Fotograf analiz edildi ({len(results)} yuz)", fg="#3498DB")

    def open_video(self):
        self.stop_camera()
        path = filedialog.askopenfilename(
            title="Video Sec",
            filetypes=[("Video Dosyalari", "*.mp4 *.avi *.mov *.mkv")],
        )
        if not path:
            return
        self.status_lbl.configure(text="Video isleniyor...", fg="#F1C40F")
        self.root.update()
        out_vid, csv_p, png_p = analyze_video(path, use_align=self.use_align.get())
        self.status_lbl.configure(text="Video analizi tamamlandi", fg="#2ECC71")
        messagebox.showinfo("Video Analizi Tamamlandi", f"Cikis videosu:\n{out_vid}\n\nRapor:\n{csv_p}\n{png_p}")

    def export_report(self):
        csv_p, png_p = self.analytics.export()
        if csv_p is None:
            messagebox.showwarning("Veri Yok", "Once kamerayi baslatip biraz duygu verisi toplayin.")
            return
        summary = self.analytics.get_summary()
        messagebox.showinfo(
            "Oturum Raporu Kaydedildi",
            f"Baskin Duygu: {summary['dominant']} (%{summary['dominant_pct']})\n"
            f"Toplam Kayit: {summary['total_records']} | Kisi: {summary['unique_people']}\n\n"
            f"CSV: {csv_p}\nGrafik: {png_p}",
        )

    def on_close(self):
        self.stop_camera()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    EmotionAppGUI(root)
    root.mainloop()

