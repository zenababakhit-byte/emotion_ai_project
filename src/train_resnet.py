import argparse
import json
import math
import os
import time
import torch
import torch.nn as nn
from torchvision import models

from src.dataset import get_dataloaders
from src.train import run_epoch


def build_resnet18_emotion(num_classes: int = 7, pretrained: bool = True) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    # 48x48 gri tonlamali yuzler icin ilk katmani uyarla (detay kaybini onler)
    old_w = model.conv1.weight.data
    model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    if pretrained:
        model.conv1.weight.data = old_w.mean(dim=1, keepdim=True)[:, :, 2:5, 2:5]
    model.maxpool = nn.Identity()
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.fc.in_features, num_classes),
    )
    return model


def train_resnet(epochs: int = 15, batch_size: int = 64, lr: float = 1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"ResNet18 Transfer Ogrenme | Cihaz: {device} | Epoch: {epochs}")

    train_loader, val_loader, _, _, _ = get_dataloaders(
        batch_size=batch_size, num_workers=0, save_metadata=False
    )

    with open("models/class_weights.json", "r", encoding="utf-8") as f:
        cw = json.load(f)
    soft_weights = [math.sqrt(cw[str(i)]) for i in range(len(cw))]

    model = build_resnet18_emotion(num_classes=7, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(soft_weights, dtype=torch.float32, device=device),
        label_smoothing=0.05,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    os.makedirs("models", exist_ok=True)
    out_path = "models/resnet18_emotion.pth"
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, None, device)
        scheduler.step()
        dur = time.perf_counter() - t0
        cur_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc*100:.2f}% | "
            f"val_loss={va_loss:.4f} val_acc={va_acc*100:.2f}% | "
            f"lr={cur_lr:.6f} | {dur:.1f}s"
        )
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), out_path)
            print(f"  -> Yeni en iyi ResNet18 kaydedildi: {out_path} (val_acc={best_val_acc*100:.2f}%)")

    print(f"ResNet18 egitimi tamamlandi. En iyi val_acc: {best_val_acc*100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ResNet18 Transfer Ogrenme")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    train_resnet(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)