import argparse
import csv
import os
import time
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from src.dataset import get_dataloaders
from src.model import EmotionCNN, get_training_components


def run_epoch(model, loader, criterion, optimizer=None, device="cpu"):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            if is_train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += batch_size

    return total_loss / total, correct / total


def plot_learning_curves(log_csv="reports/training_log.csv", out_path="reports/ogrenme_egrisi.png"):
    df = pd.read_csv(log_csv)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(df["epoch"], df["train_loss"], label="Train Loss", marker="o")
    axes[0].plot(df["epoch"], df["val_loss"], label="Val Loss", marker="o")
    axes[0].set_title("Egitim ve Dogrulama Kaybi (Loss)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df["epoch"], df["train_acc"] * 100, label="Train Acc (%)", marker="o")
    axes[1].plot(df["epoch"], df["val_acc"] * 100, label="Val Acc (%)", marker="o")
    axes[1].set_title("Egitim ve Dogrulama Dogrulugu (Accuracy)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Ogrenme egrisi kaydedildi: {out_path}")


def train(epochs=30, batch_size=64, patience=7, smoke_test=False, resume=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kullanilan cihaz: {device}")

    train_loader, val_loader, _, _, _ = get_dataloaders(batch_size=batch_size, num_workers=0)

    if smoke_test:
        print("SMOKE TEST MODU: Kucuk alt kume (512 train / 128 val), 2 epoch")
        train_sub = Subset(train_loader.dataset, list(range(min(512, len(train_loader.dataset)))))
        val_sub = Subset(val_loader.dataset, list(range(min(128, len(val_loader.dataset)))))
        train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_sub, batch_size=batch_size, shuffle=False, num_workers=0)
        epochs = 2

    model = EmotionCNN(num_classes=7).to(device)
    criterion, optimizer, scheduler = get_training_components(model, device=device)

    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    log_csv = "reports/training_log.csv"
    best_path = "models/best_model.pth"
    ckpt_path = "models/last_checkpoint.pth"

    start_epoch = 1
    best_val_acc = 0.0
    no_improve = 0

    if resume and not smoke_test and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        print(f"Checkpoint yuklendi: Epoch {start_epoch} ile devam ediliyor (best_val_acc={best_val_acc:.4f})")
    elif not smoke_test:
        with open(log_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "duration_sec"])

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.perf_counter()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device)
        duration = time.perf_counter() - t0

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc*100:.2f}% | "
            f"val_loss={val_loss:.4f} val_acc={val_acc*100:.2f}% | "
            f"lr={current_lr:.6f} | {duration:.1f}s"
        )

        if smoke_test:
            continue

        with open(log_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                round(train_loss, 4),
                round(train_acc, 4),
                round(val_loss, 4),
                round(val_acc, 4),
                current_lr,
                round(duration, 2),
            ])

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_acc": max(best_val_acc, val_acc),
        }, ckpt_path)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve = 0
            torch.save(model.state_dict(), best_path)
            print(f"  -> Yeni en iyi model kaydedildi: {best_path} (val_acc={best_val_acc*100:.2f}%)")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Erken durdurma: {patience} epoch boyunca iyilesme olmadi.")
                break

    if not smoke_test:
        plot_learning_curves(log_csv, "reports/ogrenme_egrisi.png")
        print(f"Egitim tamamlandi. En iyi val_acc: {best_val_acc*100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EmotionCNN egitim betigi")
    parser.add_argument("--epochs", type=int, default=20, help="Toplam epoch sayisi (CPU icin 20 onerilir)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--smoke-test", action="store_true", help="2 epoch hizli dogrulama")
    parser.add_argument("--resume", action="store_true", help="last_checkpoint.pth ile devam et")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        smoke_test=args.smoke_test,
        resume=args.resume,
    )