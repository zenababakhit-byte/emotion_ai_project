import os
import random

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

ROOT = "data/fer2013"
REPORT_DIR = "reports"
os.makedirs(REPORT_DIR, exist_ok=True)
random.seed(42)


def count_classes(split):
    path = os.path.join(ROOT, split)
    return {c: len(os.listdir(os.path.join(path, c))) for c in sorted(os.listdir(path))}


# 1) Sinif dagilimi tablosu
df = pd.DataFrame({"train": count_classes("train"), "test": count_classes("test")})
df["toplam"] = df["train"] + df["test"]
df["train_yuzde"] = (df["train"] / df["train"].sum() * 100).round(1)
print(df)
print("Toplam train:", df["train"].sum(), "| Toplam test:", df["test"].sum())
df.to_csv(os.path.join(REPORT_DIR, "sinif_dagilimi.csv"))

# 2) Dagilim grafigi
ax = df[["train", "test"]].plot(kind="bar", figsize=(9, 5), title="FER2013 sinif dagilimi")
ax.set_xlabel("Duygu sinifi")
ax.set_ylabel("Goruntu sayisi")
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "sinif_dagilimi.png"), dpi=150)
plt.show()

# 3) Her siniftan ornek goruntuler (satir = sinif, sutun = 8 ornek)
classes = list(df.index)
fig, axes = plt.subplots(len(classes), 8, figsize=(12, 10))
for i, c in enumerate(classes):
    folder = os.path.join(ROOT, "train", c)
    files = random.sample(os.listdir(folder), 8)
    for j, f in enumerate(files):
        axes[i, j].imshow(Image.open(os.path.join(folder, f)), cmap="gray")
        axes[i, j].axis("off")
    axes[i, 0].set_title(c, fontsize=9, loc="left")
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "ornek_yuzler.png"), dpi=150)
plt.show()

# 4) Goruntu ozellikleri kontrolu
first = os.listdir(os.path.join(ROOT, "train", classes[0]))[0]
sample = Image.open(os.path.join(ROOT, "train", classes[0], first))
print("Ornek goruntu boyutu:", sample.size, "| mod:", sample.mode)