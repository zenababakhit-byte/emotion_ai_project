import json
import math
import time
import torch
import torch.nn as nn

from src.dataset import get_dataloaders
from src.model import EmotionCNN
from src.train import run_epoch


def improve(
    init_weights="models/best_model.pth",
    out_path="models/best_model_v2.pth",
    epochs=10,
    lr=5e-4,
    batch_size=64,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Deney 2 (Yumusatilmis Sinif Agirliklari + Fine-Tuning) | Cihaz: {device}")

    train_loader, val_loader, _, _, _ = get_dataloaders(
        batch_size=batch_size, num_workers=0, save_metadata=False
    )

    with open("models/class_weights.json", "r", encoding="utf-8") as f:
        cw = json.load(f)
    soft_weights = [math.sqrt(cw[str(i)]) for i in range(len(cw))]
    print("Yumusatilmis agirliklar (sqrt):", [round(w, 3) for w in soft_weights])

    model = EmotionCNN(num_classes=7).to(device)
    model.load_state_dict(torch.load(init_weights, map_location=device))

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(soft_weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device)
        duration = time.perf_counter() - t0

        scheduler.step(val_loss)
        cur_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc*100:.2f}% | "
            f"val_loss={val_loss:.4f} val_acc={val_acc*100:.2f}% | "
            f"lr={cur_lr:.6f} | {duration:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), out_path)
            print(f"  -> Yeni en iyi Deney 2 modeli: {out_path} (val_acc={best_val_acc*100:.2f}%)")

    print(f"Deney 2 tamamlandi. En iyi val_acc: {best_val_acc*100:.2f}%")


if __name__ == "__main__":
    improve()