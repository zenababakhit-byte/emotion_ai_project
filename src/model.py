import json
import os
import time
import torch
import torch.nn as nn


def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2),
        nn.Dropout(0.25),
    )


class EmotionCNN(nn.Module):
    """48x48 gri tonlamali yuz goruntuleri icin 7 sinifli hafif CNN modeli.
    Boyut akisi: [N, 1, 48, 48] -> [N, 32, 24, 24] -> [N, 64, 12, 12] -> [N, 128, 6, 6] -> [N, 7]
    """

    def __init__(self, num_classes: int = 7):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(1, 32),
            conv_block(32, 64),
            conv_block(64, 128),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_training_components(
    model: nn.Module,
    class_weights_path: str = "models/class_weights.json",
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: torch.device = torch.device("cpu"),
):
    weight_tensor = None
    if os.path.exists(class_weights_path):
        with open(class_weights_path, "r", encoding="utf-8") as f:
            cw = json.load(f)
        weights = [cw[str(i)] for i in range(len(cw))]
        weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    return criterion, optimizer, scheduler


def benchmark_inference(model: nn.Module, runs: int = 100) -> float:
    model.eval()
    dummy = torch.randn(1, 1, 48, 48)
    with torch.no_grad():
        # Isinma (warm-up)
        for _ in range(10):
            _ = model(dummy)
        start = time.perf_counter()
        for _ in range(runs):
            _ = model(dummy)
        elapsed_ms = (time.perf_counter() - start) * 1000.0 / runs
    return elapsed_ms


if __name__ == "__main__":
    model = EmotionCNN(num_classes=7)
    print(model)
    print("-" * 60)

    dummy_input = torch.randn(1, 1, 48, 48)
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)

    total_params = count_parameters(model)
    avg_ms = benchmark_inference(model, runs=100)
    criterion, optimizer, scheduler = get_training_components(model)

    print(f"Giris sekli                  : {list(dummy_input.shape)}")
    print(f"Cikis sekli                  : {list(output.shape)}")
    print(f"Egitilebilir parametre sayisi: {total_params:,}")
    print(f"Ortalama cikarim suresi (CPU): {avg_ms:.2f} ms (100 tekrar)")
    print(f"Kayip fonksiyonu             : {criterion.__class__.__name__} (sinif agirlikli)")
    print(f"Optimizor                    : {optimizer.__class__.__name__} (lr=1e-3, wd=1e-4)")
    print(f"Zamanlayici                  : {scheduler.__class__.__name__}")