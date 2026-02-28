# ✍️ HandFonted

**Turn your handwriting into a functional .ttf font file. Try it now on the live web application!**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-handfonted.xyz-brightgreen?style=for-the-badge&logo=rocket)](https://handfonted.xyz)

This repository contains the source code for the command-line tool that powers the HandFonted web app. It provides a complete pipeline to take an image of handwritten characters, segment them, classify them using a PyTorch model, and build a working TrueType font.

---

### Table of Contents
* [How It Works](#how-it-works)
* [Features](#features)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Usage](#usage)
* [Training Your Own Model](#training-your-own-model)
* [License](#license)

---

### How It Works

The HandFonted pipeline consists of three main stages:
1.  **Segmentation (`character_segmentation.py`):** Uses OpenCV to perform image processing (adaptive thresholding, morphological operations) to find and extract individual character images from the source. It includes a smart heuristic to merge dots with 'i' and 'j' characters.
2.  **Classification (`character_classification.py`):** A custom `ResInceptionNet` model (built with PyTorch) classifies each character image. It uses the Hungarian algorithm (`linear_sum_assignment`) to ensure each character from the input sheet is uniquely assigned to a letter.
3.  **Font Creation (`font_creation.py`):** The classified images are vectorized using `scikit-image`. These vector outlines are then used to replace the corresponding glyphs in a base template font (`.ttf`), generating a new, custom font file.

---

### Features
- **End-to-End Pipeline:** From a single image to a usable `.ttf` font.
- **Custom Deep Learning Model:** A hybrid ResNet-Inception model for accurate character classification.
- **Intelligent Dot Merging:** Correctly handles dotted characters like 'i' and 'j'.
- **Vectorization:** Converts pixel images into smooth, scalable font glyphs.
- **Customizable:** Control font name, style, and stroke thickness.

---

### Getting Started

#### Prerequisites
- Python 3.8+
- An image of your handwriting.

#### Installation
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/reshamgaire/handfonted.git
    cd handfonted
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    # .\venv\Scripts\activate
    # On macOS/Linux:
    # source venv/bin/activate
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```
---

### Usage

Run the main script from the command line.

**Basic Usage:**
```bash
python main.py --input-image "path/to/your/handwriting.jpg" --output-path "output/my_font.ttf"
```

**Customized Usage:**
```bash
python main.py \
    --input-image "examples/good_example.jpg" \
    --output-path "output/ReshamHand.ttf" \
    --font-name "Resham Hand" \
    --font-style "Regular" \
    --thickness 110 \
    --model-path "resources/best_ResInceptionNet_model0.8811.pth" \
    --base-font "resources/arial.ttf"
```

---

### Training Your Own Model

You can retrain the character classification model on EMNIST data using the provided scripts. The model is trained at **64×64** input resolution for best results.

#### 1. Install requirements

```bash
pip install -r requirements.txt
```

#### 2. One-command full pipeline

**Linux / macOS:**
```bash
bash scripts/run_all.sh
```

**Windows:**
```bat
scripts\run_all.bat
```

#### 3. Step-by-step

| Step | Script | Description |
|------|--------|-------------|
| 1 | `scripts/download_data.py` | Downloads EMNIST ByClass, filters to A-Z/a-z (52 classes), resizes to 64×64, saves `.pt` files |
| 2 | `scripts/train.py` | Trains `ResInceptionNet` with augmentation, saves best model and training log |
| 3 | `scripts/evaluate.py` | Reports top-1 accuracy, per-class breakdown, and saves a confusion matrix |

Run each step individually:
```bash
python scripts/download_data.py --output-dir data/
python scripts/train.py --epochs 60 --batch-size 128 --output-dir resources/
python scripts/evaluate.py --model-path resources/best_model.pth
```

To resume an interrupted training run:
```bash
python scripts/train.py --resume
```

#### 4. Tips for best input image quality

- Scan at **300 DPI** or higher for clearest results
- Use **white paper** and **dark ink** (black or dark blue)
- Avoid shadows, folds, or reflections on the paper
- Write each character clearly within its designated box
- Ensure good contrast between ink and background

#### 5. Using the new trained model

Pass `--model-path` to `main.py` to use your retrained model:

```bash
python main.py \
    --input-image "examples/good_example.jpg" \
    --output-path "output/MyFont.ttf" \
    --model-path "resources/best_model.pth"
```

> **Note:** The model must be retrained at 64×64 input for best results. The existing pre-trained `.pth` file was trained at 28×28 and will need to be replaced after running `scripts/train.py`.

---

### License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---