import json
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt


def get_transforms():
    train_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    eval_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    return train_tf, eval_tf


def get_dataloaders(
    data_dir="data/fer2013",
    batch_size=64,
    val_ratio=0.2,
    seed=42,
    num_workers=0,
    save_metadata=True,
):
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")

    train_tf, eval_tf = get_transforms()

    # Artirma sadece train setine uygulansin diye iki ayri ImageFolder ornegi
    full_train_aug = datasets.ImageFolder(train_dir, transform=train_tf)
    full_train_clean = datasets.ImageFolder(train_dir, transform=eval_tf)
    test_dataset = datasets.ImageFolder(test_dir, transform=eval_tf)

    targets = np.array(full_train_aug.targets)
    indices = np.arange(len(targets))

    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_ratio,
        random_state=seed,
        stratify=targets,
    )

    train_dataset = Subset(full_train_aug, train_idx)
    val_dataset = Subset(full_train_clean, val_idx)

    # Sinif agirliklarini yalnizca egitim bolmesinden hesapla
    train_targets = targets[train_idx]
    classes = np.unique(train_targets)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=train_targets,
    )
    class_weights_dict = {int(c): round(float(w), 4) for c, w in zip(classes, weights)}

    if save_metadata:
        os.makedirs("models", exist_ok=True)
        with open("models/label_map.json", "w", encoding="utf-8") as f:
            json.dump(full_train_aug.class_to_idx, f, indent=2)
        with open("models/class_weights.json", "w", encoding="utf-8") as f:
            json.dump(class_weights_dict, f, indent=2)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader, full_train_aug.class_to_idx, class_weights_dict


def save_augmentation_preview(train_loader, class_to_idx, out_path="screenshots/gun08_augmentation.png"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    images, labels = next(iter(train_loader))

    fig, axes = plt.subplots(2, 8, figsize=(14, 4))
    for i, ax in enumerate(axes.flat):
        img = images[i].squeeze(0).numpy() * 0.5 + 0.5  # denormalize
        ax.imshow(img, cmap="gray")
        ax.set_title(idx_to_class[int(labels[i])], fontsize=9)
        ax.axis("off")

    plt.suptitle("Gun 08 - Artirilmis (Augmented) Egitim Ornekleri", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Augmentation onizlemesi kaydedildi: {out_path}")


if __name__ == "__main__":
    train_loader, val_loader, test_loader, class_to_idx, class_weights = get_dataloaders()

    print("class_to_idx:", class_to_idx)
    print("class_weights:", class_weights)
    print(f"Train adet: {len(train_loader.dataset)}")
    print(f"Val adet  : {len(val_loader.dataset)}")
    print(f"Test adet : {len(test_loader.dataset)}")

    images, labels = next(iter(train_loader))
    print("Batch goruntu sekli:", list(images.shape))
    print("Batch etiket sekli :", list(labels.shape))
    print(f"Piksel araligi     : min={images.min().item():.2f}, max={images.max().item():.2f}")

    save_augmentation_preview(train_loader, class_to_idx)