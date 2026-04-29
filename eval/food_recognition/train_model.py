"""
Fine-tune CNN backbones for food nutrition regression.

Supported backbones:
- mobilenet_v3_small (default)
- mobilenet_v3_large
- efficientnet_b0
- mobilenetv4_conv_small
- tf_efficientnet_lite4

Targets: total_mass, total_calories, total_fat, total_carb, total_protein
Device: MPS (Apple Silicon)

Usage:
    python eval/train_model.py --model mobilenet_v3_small
"""

import sys
import os
import pickle
import json
import argparse
import random
import io
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import timm
from torchvision import models, transforms
from PIL import Image

EVAL_DIR = Path(__file__).resolve().parent
DATASET_DIR = EVAL_DIR / "food_dataset"


def get_model_dir(model_name: str) -> Path:
    """Get model directory for specific model."""
    return EVAL_DIR / f"model_{model_name}"


MODEL_DIR = None  # Will be set in train()

METRICS = ["total_mass", "total_calories", "total_fat", "total_carb", "total_protein"]


# ──────────────────────────────────────────────
# Model Factory
# ──────────────────────────────────────────────
def create_model(model_name: str, pretrained=True) -> nn.Module:
    """Create model with specified backbone."""
    if model_name in ("mobilenet_v3_small", "mobilenet_v3_large", "efficientnet_b0"):
        if model_name == "mobilenet_v3_small":
            backbone = models.mobilenet_v3_small(
                weights="DEFAULT" if pretrained else None
            )
            num_features = 576
        elif model_name == "mobilenet_v3_large":
            backbone = models.mobilenet_v3_large(
                weights="DEFAULT" if pretrained else None
            )
            num_features = 960
        else:  # efficientnet_b0
            backbone = models.efficientnet_b0(weights="DEFAULT" if pretrained else None)
            num_features = 1280

        backbone.classifier = nn.Sequential(
            nn.Linear(num_features, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 5)
        )
        return backbone
    else:
        # Use timm for other models
        backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=5)
        return backbone


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────
class FoodNutritionDataset(Dataset):
    def __init__(
        self, df: pd.DataFrame, transform=None, target_mean=None, target_std=None
    ):
        self.df = df.reset_index(drop=True)
        self.transform = transform

        # Compute or use provided normalization stats
        targets = self.df[METRICS].values.astype(np.float32)
        if target_mean is None:
            self.target_mean = targets.mean(axis=0)
            self.target_std = targets.std(axis=0)
            self.target_std[self.target_std == 0] = 1.0  # avoid div by zero
        else:
            self.target_mean = target_mean
            self.target_std = target_std

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Decode image
        img = Image.open(io.BytesIO(row["rgb_image"])).convert("RGB")
        if self.transform:
            img = self.transform(img)

        # Normalized targets
        raw = np.array([row[m] for m in METRICS], dtype=np.float32)
        normalized = (raw - self.target_mean) / self.target_std

        return img, torch.tensor(normalized), torch.tensor(raw)


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────
def train(
    model_name: str = "mobilenet_v3_small",
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 32,
    seed: int = 42,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using MPS (Apple Silicon) for {model_name}")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("WARNING: No GPU, using CPU")

    # Set model directory
    global MODEL_DIR
    MODEL_DIR = get_model_dir(model_name)
    MODEL_DIR.mkdir(exist_ok=True)
    print(f"  Model output: {MODEL_DIR}")

    # LOCK: Check if model is already trained
    if (MODEL_DIR / "best_model.pt").exists():
        print(f"Model '{model_name}' already trained. Skipping.")
        return

    # Load data
    print("Loading dataset...")
    gt = pd.read_excel(DATASET_DIR / "dishes.xlsx")
    with open(DATASET_DIR / "dish_images.pkl", "rb") as f:
        images_df = pickle.load(f)
    dataset = images_df.merge(gt, left_on="dish", right_on="dish_id", how="inner")
    print(f"  Total: {len(dataset)} samples")

    # Split: use same seed=42 shuffle as eval to get the same test set
    all_ids = dataset["dish"].tolist()
    random.seed(42)
    shuffled_ids = all_ids.copy()
    random.shuffle(shuffled_ids)
    test_ids = set(shuffled_ids[:30])

    train_df = dataset[~dataset["dish"].isin(test_ids)]
    val_split = int(len(train_df) * 0.9)

    # Shuffle train_df before splitting
    random.seed(seed)
    train_df = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    train_split = train_df.iloc[:val_split]
    val_split_df = train_df.iloc[val_split:]

    print(
        f"  Train: {len(train_split)} | Val: {len(val_split_df)} | Test (held out): {len(test_ids)}"
    )

    # Transforms
    train_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Datasets
    train_ds = FoodNutritionDataset(train_split, transform=train_transform)
    val_ds = FoodNutritionDataset(
        val_split_df,
        transform=val_transform,
        target_mean=train_ds.target_mean,
        target_std=train_ds.target_std,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Save normalization stats
    norm_stats = {
        "target_mean": train_ds.target_mean.tolist(),
        "target_std": train_ds.target_std.tolist(),
        "metrics": METRICS,
    }
    with open(MODEL_DIR / "norm_stats.json", "w") as f:
        json.dump(norm_stats, f, indent=2)
    print(f"  Target mean: {dict(zip(METRICS, train_ds.target_mean.tolist()))}")
    print(f"  Target std:  {dict(zip(METRICS, train_ds.target_std.tolist()))}")

    # Model
    model = create_model(model_name).to(device)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.SmoothL1Loss()

    best_val_loss = float("inf")
    print(f"\nTraining for {epochs} epochs (lr={lr}, batch={batch_size})...\n")

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        for imgs, targets_norm, _ in train_loader:
            imgs = imgs.to(device)
            targets_norm = targets_norm.to(device)

            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, targets_norm)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)

        train_loss /= len(train_ds)

        # Validate
        model.eval()
        val_loss = 0.0
        val_mae_raw = np.zeros(5)
        val_count = 0
        with torch.no_grad():
            for imgs, targets_norm, targets_raw in val_loader:
                imgs = imgs.to(device)
                targets_norm = targets_norm.to(device)

                preds_norm = model(imgs)
                loss = criterion(preds_norm, targets_norm)
                val_loss += loss.item() * imgs.size(0)

                # Denormalize for MAE
                preds_raw = (
                    preds_norm.cpu().numpy() * train_ds.target_std
                    + train_ds.target_mean
                )
                targets_raw_np = targets_raw.numpy()
                val_mae_raw += np.abs(preds_raw - targets_raw_np).sum(axis=0)
                val_count += imgs.size(0)

        val_loss /= len(val_ds)
        val_mae_raw /= val_count

        scheduler.step()

        # Log
        lr_now = scheduler.get_last_lr()[0]
        mae_str = " | ".join(f"{METRICS[i]}={val_mae_raw[i]:.1f}" for i in range(5))
        print(
            f"  Epoch {epoch + 1:>3}/{epochs}  train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  lr={lr_now:.6f}"
        )
        print(f"    Val MAE: {mae_str}")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_DIR / "best_model.pt")
            print(f"    ✓ Saved best model (val_loss={val_loss:.4f})")

    # Save final model too
    torch.save(model.state_dict(), MODEL_DIR / "final_model.pt")

    print(f"\nTraining complete. Best val_loss={best_val_loss:.4f}")
    print(f"Model saved to: {MODEL_DIR}")


def main():
    parser = argparse.ArgumentParser(
        description="Train food nutrition regression model"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mobilenet_v3_small",
        choices=[
            "mobilenet_v3_small",
            "mobilenet_v3_large",
            "efficientnet_b0",
            "mobilenetv4_conv_small",
            "tf_efficientnet_lite4",
            "mobilenetv4_conv_large",
        ],
        help="Backbone model to use",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(
        model_name=args.model,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
