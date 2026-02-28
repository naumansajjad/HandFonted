#!/usr/bin/env bash
set -e

echo "=== Step 1: Download & preprocess EMNIST data ==="
python scripts/download_data.py --output-dir data/

echo "=== Step 2: Train model ==="
python scripts/train.py --epochs 60 --batch-size 128 --output-dir resources/

echo "=== Step 3: Evaluate model ==="
python scripts/evaluate.py --model-path resources/best_model.pth

echo "=== All done! Best model saved to resources/best_model.pth ==="
