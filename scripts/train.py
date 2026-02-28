"""
Train ResInceptionNet on EMNIST alphabetic data (52 classes, 64x64 input).

Usage:
    python scripts/train.py --epochs 60 --batch-size 128 --output-dir resources/
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

# Allow imports from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from character_classification import ResInceptionNet


class EMNISTDataset(Dataset):
    def __init__(self, pt_path, transform=None):
        data = torch.load(pt_path, weights_only=True)
        self.images = data["images"]   # uint8 [N,1,64,64]
        self.labels = data["labels"]   # long [N]
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]  # [1,64,64] uint8
        # Convert to HxW uint8 numpy for PIL-compatible transforms
        img_np = img.squeeze(0).numpy()  # [64,64]
        label = self.labels[idx]
        if self.transform:
            img_np = self.transform(img_np)
        return img_np, label


def compute_mean_std(pt_path):
    """Compute per-channel mean and std of the training images (float in [0,1])."""
    data = torch.load(pt_path, weights_only=True)
    images = data["images"].float() / 255.0  # [N,1,64,64]
    mean = images.mean().item()
    std = images.std().item()
    return mean, std


def main():
    parser = argparse.ArgumentParser(description="Train ResInceptionNet on EMNIST 64x64 alphabetic data")
    parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs (default: 60)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size (default: 128)")
    parser.add_argument("--output-dir", type=str, default="resources/", help="Directory to save model and logs")
    parser.add_argument("--data-dir", type=str, default="data/", help="Directory containing .pt data files")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_pt = os.path.join(args.data_dir, "emnist_train.pt")
    val_pt = os.path.join(args.data_dir, "emnist_val.pt")

    print("\n=== Computing normalization statistics from training set ===")
    mean, std = compute_mean_std(train_pt)
    print(f"  Mean: {mean:.4f}, Std: {std:.4f}")

    norm_stats_path = os.path.join(args.data_dir, "norm_stats.json")
    with open(norm_stats_path, "w") as f:
        json.dump({"mean": mean, "std": std}, f)
    print(f"  Saved normalization stats to {norm_stats_path}")

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, shear=5, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,)),
    ])
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,)),
    ])

    train_dataset = EMNISTDataset(train_pt, transform=train_transform)
    val_dataset = EMNISTDataset(val_pt, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    print(f"\nTrain samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    model = ResInceptionNet(num_classes=52).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    start_epoch = 0
    best_val_acc = 0.0
    last_ckpt_path = os.path.join(args.output_dir, "last_checkpoint.pth")
    best_model_path = os.path.join(args.output_dir, "best_model.pth")
    log_path = os.path.join(args.output_dir, "training_log.csv")

    if args.resume and os.path.exists(last_ckpt_path):
        print(f"\nResuming from {last_ckpt_path}...")
        ckpt = torch.load(last_ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        print(f"  Resumed from epoch {start_epoch}, best val acc so far: {best_val_acc:.4f}")

    log_exists = os.path.exists(log_path) and args.resume
    log_file = open(log_path, "a" if log_exists else "w", newline="")
    log_writer = csv.writer(log_file)
    if not log_exists:
        log_writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

    print("\n=== Starting training ===")
    for epoch in range(start_epoch, args.epochs):
        # --- Train ---
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [train]", leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += images.size(0)

        train_loss = train_loss_sum / train_total
        train_acc = train_correct / train_total

        # --- Validate ---
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0
        with torch.inference_mode():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [val]", leave=False):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss_sum += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += images.size(0)

        val_loss = val_loss_sum / val_total
        val_acc = val_correct / val_total

        scheduler.step()

        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        log_writer.writerow([epoch + 1, f"{train_loss:.6f}", f"{train_acc:.6f}",
                              f"{val_loss:.6f}", f"{val_acc:.6f}"])
        log_file.flush()

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"  ✓ New best model saved (val_acc={val_acc:.4f})")

        # Save last checkpoint
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_val_acc": best_val_acc,
        }, last_ckpt_path)

    log_file.close()
    print(f"\n=== Training complete! Best val acc: {best_val_acc:.4f} ===")
    print(f"Best model saved to: {best_model_path}")
    print(f"Training log saved to: {log_path}")


if __name__ == "__main__":
    main()
