"""
Evaluate a trained ResInceptionNet model on the EMNIST test set.

Prints overall top-1 accuracy, per-class accuracy, and saves a confusion matrix.

Usage:
    python scripts/evaluate.py --model-path resources/best_model.pth
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

# Allow imports from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from character_classification import ResInceptionNet

CLASSES = list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')
NUM_CLASSES = 52


class EMNISTTestDataset(Dataset):
    def __init__(self, pt_path, transform=None):
        data = torch.load(pt_path, weights_only=True)
        self.images = data["images"]   # uint8 [N,1,64,64]
        self.labels = data["labels"]   # long [N]
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx].squeeze(0).numpy()  # [64,64] uint8
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


def main():
    parser = argparse.ArgumentParser(description="Evaluate ResInceptionNet on EMNIST test set")
    parser.add_argument("--model-path", type=str, default="resources/best_model.pth",
                        help="Path to the trained model .pth file")
    parser.add_argument("--data-dir", type=str, default="data/",
                        help="Directory containing emnist_test.pt")
    parser.add_argument("--output-dir", type=str, default="resources/",
                        help="Directory to save confusion matrix image")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for evaluation")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load normalization stats
    norm_stats_path = os.path.join(args.data_dir, "norm_stats.json")
    if os.path.exists(norm_stats_path):
        with open(norm_stats_path, "r") as f:
            stats = json.load(f)
        mean = stats["mean"]
        std = stats["std"]
        print(f"Loaded normalization stats from {norm_stats_path}: mean={mean:.4f}, std={std:.4f}")
    else:
        mean, std = 0.1751, 0.3332
        print(f"norm_stats.json not found, using defaults: mean={mean}, std={std}")

    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,)),
    ])

    test_pt = os.path.join(args.data_dir, "emnist_test.pt")
    test_dataset = EMNISTTestDataset(test_pt, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    print(f"Test samples: {len(test_dataset)}")

    model = ResInceptionNet(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded model from {args.model_path}")

    all_preds = []
    all_labels = []

    print("\nRunning inference...")
    with torch.inference_mode():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu()
            all_preds.append(preds)
            all_labels.append(labels)

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    overall_acc = (all_preds == all_labels).mean()
    print(f"\nOverall Top-1 Accuracy: {overall_acc * 100:.2f}%")

    # Per-class accuracy
    print(f"\n{'Char':<6} {'Correct':<10} {'Total':<10} {'Accuracy%':<10}")
    print("-" * 36)
    for i, char in enumerate(CLASSES):
        mask = all_labels == i
        total = mask.sum()
        correct = (all_preds[mask] == i).sum() if total > 0 else 0
        acc_pct = (correct / total * 100) if total > 0 else 0.0
        print(f"  {char:<4} {correct:<10} {total:<10} {acc_pct:<10.2f}")

    # Confusion matrix
    print("\nGenerating confusion matrix...")
    conf_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for true, pred in zip(all_labels, all_preds):
        conf_matrix[true][pred] += 1

    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(conf_matrix, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASSES, fontsize=7)
    ax.set_yticklabels(CLASSES, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()

    cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=120)
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")


if __name__ == "__main__":
    main()
