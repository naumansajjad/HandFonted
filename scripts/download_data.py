"""
Download and preprocess EMNIST ByClass dataset for HandFonted training.

Filters to 52 alphabetic classes (A-Z, a-z), resizes to 64x64, and saves
train/val/test splits as .pt files.

Usage:
    python scripts/download_data.py --output-dir data/
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch
from torchvision import datasets
from tqdm import tqdm

# EMNIST ByClass: indices 0-9 are digits, 10-35 are A-Z uppercase, 36-61 are a-z lowercase
ALPHA_START_IDX = 10
ALPHA_END_IDX = 61  # inclusive
NUM_ALPHA_CLASSES = 52  # 26 upper + 26 lower


def emnist_label_to_alpha_idx(emnist_label):
    """Remap EMNIST ByClass label (10-61) to 0-51 alphabetic index."""
    return emnist_label - ALPHA_START_IDX


def fix_emnist_image(img_array):
    """EMNIST images are stored transposed — apply transpose and horizontal flip."""
    img = np.transpose(img_array)
    img = np.fliplr(img)
    return img


def resize_to_64(img_array):
    """Resize a 28x28 grayscale image to 64x64 using Lanczos interpolation."""
    return cv2.resize(img_array, (64, 64), interpolation=cv2.INTER_LANCZOS4)


def process_split(dataset, split_name):
    """Process a split of the EMNIST dataset, filtering to alphabetic classes only."""
    images_list = []
    labels_list = []

    print(f"\nProcessing {split_name} split ({len(dataset)} total samples)...")
    for img_pil, label in tqdm(dataset, desc=split_name):
        if label < ALPHA_START_IDX or label > ALPHA_END_IDX:
            continue

        alpha_label = emnist_label_to_alpha_idx(label)

        img_array = np.array(img_pil, dtype=np.uint8)
        img_array = fix_emnist_image(img_array)
        img_array = resize_to_64(img_array)

        images_list.append(img_array)
        labels_list.append(alpha_label)

    images_tensor = torch.tensor(np.array(images_list), dtype=torch.uint8).unsqueeze(1)  # [N,1,64,64]
    labels_tensor = torch.tensor(labels_list, dtype=torch.long)

    return images_tensor, labels_tensor


def print_class_distribution(labels_tensor, split_name):
    """Print a summary of class distribution."""
    classes = list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')
    print(f"\nClass distribution — {split_name}:")
    print(f"{'Char':<6} {'Count':<8}")
    print("-" * 14)
    for i, char in enumerate(classes):
        count = (labels_tensor == i).sum().item()
        print(f"  {char:<4} {count:<8}")


def main():
    parser = argparse.ArgumentParser(description="Download and preprocess EMNIST ByClass dataset")
    parser.add_argument("--output-dir", type=str, default="data/",
                        help="Directory to save processed .pt files (default: data/)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=== Step 1: Downloading EMNIST ByClass dataset ===")
    print("(This may take a few minutes on first run...)")

    train_dataset = datasets.EMNIST(
        root=os.path.join(args.output_dir, "raw"),
        split="byclass",
        train=True,
        download=True
    )
    test_dataset = datasets.EMNIST(
        root=os.path.join(args.output_dir, "raw"),
        split="byclass",
        train=False,
        download=True
    )

    print(f"\nFull train set size: {len(train_dataset)}")
    print(f"Full test set size:  {len(test_dataset)}")

    print("\n=== Step 2: Processing and filtering to alphabetic classes (A-Z, a-z) ===")
    train_images, train_labels = process_split(train_dataset, "train")
    test_images, test_labels = process_split(test_dataset, "test")

    # 90/10 split of train into train/val
    n_train = len(train_images)
    n_val = n_train // 10
    n_train_final = n_train - n_val

    # Shuffle before splitting
    perm = torch.randperm(n_train)
    train_images = train_images[perm]
    train_labels = train_labels[perm]

    val_images = train_images[n_train_final:]
    val_labels = train_labels[n_train_final:]
    train_images = train_images[:n_train_final]
    train_labels = train_labels[:n_train_final]

    print(f"\nFinal split sizes:")
    print(f"  Train: {len(train_images)}")
    print(f"  Val:   {len(val_images)}")
    print(f"  Test:  {len(test_images)}")

    print("\n=== Step 3: Saving processed datasets ===")
    train_path = os.path.join(args.output_dir, "emnist_train.pt")
    val_path = os.path.join(args.output_dir, "emnist_val.pt")
    test_path = os.path.join(args.output_dir, "emnist_test.pt")

    torch.save({"images": train_images, "labels": train_labels}, train_path)
    print(f"  Saved {train_path}")
    torch.save({"images": val_images, "labels": val_labels}, val_path)
    print(f"  Saved {val_path}")
    torch.save({"images": test_images, "labels": test_labels}, test_path)
    print(f"  Saved {test_path}")

    print_class_distribution(train_labels, "Train")

    print("\n=== Done! ===")


if __name__ == "__main__":
    main()
