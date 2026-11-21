"""Proje genelindeki sabit ayarlar ve dosya yollari."""

DATA_DIR = "data/fer2013"
MODEL_PATH = "models/final_model.pth"
LABEL_MAP_PATH = "models/label_map.json"
CLASS_WEIGHTS_PATH = "models/class_weights.json"
PERF_CSV_PATH = "reports/performans.csv"

IMAGE_SIZE = (48, 48)
NUM_CLASSES = 7
BATCH_SIZE = 64
RANDOM_SEED = 42

CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

FACE_PADDING_RATIO = 0.15
SMOOTHING_WINDOW = 10
CONFIDENCE_THRESHOLD = 0.40

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