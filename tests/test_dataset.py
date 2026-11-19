import os
import json
from src.dataset import get_dataloaders


def test_dataloader_shapes_and_splits():
    train_loader, val_loader, test_loader, class_to_idx, _ = get_dataloaders(
        batch_size=64, save_metadata=True
    )

    assert len(class_to_idx) == 7
    assert os.path.exists("models/label_map.json")
    assert os.path.exists("models/class_weights.json")

    # Train ve val indekslerinde cakismanin olmadigini dogrula
    train_indices = set(train_loader.dataset.indices)
    val_indices = set(val_loader.dataset.indices)
    assert len(train_indices.intersection(val_indices)) == 0

    images, labels = next(iter(train_loader))
    assert list(images.shape) == [64, 1, 48, 48]
    assert list(labels.shape) == [64]

    with open("models/label_map.json", "r", encoding="utf-8") as f:
        saved_map = json.load(f)
    assert saved_map == class_to_idx