import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src.dataset import get_dataloaders
from src.model import EmotionCNN


def plot_confusion_matrix(cm, class_names, out_path="reports/karisiklik_matrisi.png"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="Gercek Sinif",
        xlabel="Tahmin Edilen Sinif",
        title="FER2013 Test Seti Karisiklik Matrisi (Confusion Matrix)",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=9,
            )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Karisiklik matrisi kaydedildi: {out_path}")


def plot_misclassified(mistakes, class_names, out_path="reports/hatali_ornekler.png"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    count = min(16, len(mistakes))
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))

    for i, ax in enumerate(axes.flat):
        if i < count:
            img, true_idx, pred_idx = mistakes[i]
            img = img * 0.5 + 0.5  # denormalize
            ax.imshow(img, cmap="gray")
            ax.set_title(
                f"G: {class_names[true_idx]}\nT: {class_names[pred_idx]}",
                fontsize=9,
                color="darkred",
            )
        ax.axis("off")

    plt.suptitle("Yanlis Siniflandirilan 16 Test Ornegi (G: Gercek, T: Tahmin)", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Hatali ornekler kaydedildi: {out_path}")


def evaluate(model_path="models/best_model.pth", label_map_path="models/label_map.json"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(label_map_path, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)
    idx_to_class = {int(v): k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    _, _, test_loader, _, _ = get_dataloaders(batch_size=64, num_workers=0, save_metadata=False)

    model = EmotionCNN(num_classes=len(class_names)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    y_true = []
    y_pred = []
    mistakes = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            labels_np = labels.numpy()

            y_true.extend(labels_np.tolist())
            y_pred.extend(preds.tolist())

            if len(mistakes) < 16:
                imgs_np = images.cpu().squeeze(1).numpy()
                for img, t, p in zip(imgs_np, labels_np, preds):
                    if t != p and len(mistakes) < 16:
                        mistakes.append((img, int(t), int(p)))

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred)

    print(f"Model: {model_path}")
    print(f"Test Dogrulugu (Accuracy): {acc * 100:.2f}%")
    print(f"Macro-F1 Skoru           : {macro_f1:.4f}")
    print("\nSiniflandirma Raporu (Classification Report):")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    # En cok karisan sinif ciftleri
    pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j:
                pairs.append((int(cm[i, j]), class_names[i], class_names[j]))
    pairs.sort(reverse=True)
    print("En cok karisan 5 sinif cifti (Gercek -> Tahmin):")
    for cnt, src_cls, dst_cls in pairs[:5]:
        print(f"  {src_cls} -> {dst_cls}: {cnt} adet")

    plot_confusion_matrix(cm, class_names, "reports/karisiklik_matrisi.png")
    plot_misclassified(mistakes, class_names, "reports/hatali_ornekler.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EmotionCNN test degerlendirme betigi")
    parser.add_argument("--model", type=str, default="models/best_model.pth")
    args = parser.parse_args()
    evaluate(model_path=args.model)