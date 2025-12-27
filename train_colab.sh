#!/bin/bash
# One-click training script for Google Colab
# Usage: !bash train_colab.sh yolov8  OR  !bash train_colab.sh rfdetr

set -e

MODEL=${1:-yolov8}
DATASET_URL=${2:-""}

echo "=============================================="
echo "CHANA CLASSIFIER - $MODEL TRAINING"
echo "=============================================="

# Install dependencies
echo "[1/5] Installing dependencies..."
pip install -q ultralytics rfdetr pycocotools gdown

# Check if dataset exists or download
if [ ! -d "/content/dataset" ]; then
    echo "[2/5] Setting up dataset..."

    if [ -f "/content/drive/MyDrive/chana_dataset.zip" ]; then
        echo "Extracting from Google Drive..."
        unzip -q /content/drive/MyDrive/chana_dataset.zip -d /content/
    elif [ -n "$DATASET_URL" ]; then
        echo "Downloading from URL..."
        gdown "$DATASET_URL" -O /content/dataset.zip
        unzip -q /content/dataset.zip -d /content/
    else
        echo "ERROR: No dataset found!"
        echo "Please either:"
        echo "  1. Upload chana_dataset.zip to Google Drive root"
        echo "  2. Provide dataset URL: bash train_colab.sh yolov8 'your_gdrive_url'"
        exit 1
    fi
else
    echo "[2/5] Dataset already exists."
fi

# Verify dataset
echo "[3/5] Verifying dataset..."
TRAIN_COUNT=$(find /content/dataset/train/images -name "*.jpg" | wc -l)
VALID_COUNT=$(find /content/dataset/valid/images -name "*.jpg" | wc -l)
TEST_COUNT=$(find /content/dataset/test/images -name "*.jpg" | wc -l)
echo "Train: $TRAIN_COUNT, Valid: $VALID_COUNT, Test: $TEST_COUNT"

# Create data.yaml
echo "[4/5] Creating config..."
cat > /content/data.yaml << EOF
path: /content/dataset
train: train/images
val: valid/images
test: test/images
nc: 9
names:
  0: Bad Black Channa
  1: Bad Kabuli Chana
  2: Good Black Channa
  3: Good Kabuli Chana
  4: foreign material
  5: Bad Soya
  6: Good Soya
  7: Bad Matar
  8: Good Matar
EOF

# Train
echo "[5/5] Starting training..."
if [ "$MODEL" == "yolov8" ]; then
    python3 << 'PYTHON_SCRIPT'
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
results = model.train(
    data='/content/data.yaml',
    epochs=100,
    batch=32,
    imgsz=640,
    optimizer='AdamW',
    degrees=180,
    flipud=0.5,
    fliplr=0.5,
    hsv_s=0.7,
    hsv_v=0.4,
    mosaic=1.0,
    mixup=0.1,
    close_mosaic=10,
    device=0,
    workers=4,
    amp=True,
    project='/content/drive/MyDrive/chana_models',
    name='yolov8n_chana',
    patience=20,
    plots=True,
)

# Evaluate
print("\n" + "="*50)
print("EVALUATING ON TEST SET")
print("="*50)
best = YOLO('/content/drive/MyDrive/chana_models/yolov8n_chana/weights/best.pt')
results = best.val(data='/content/data.yaml', split='test')
print(f"\nmAP50: {results.box.map50:.4f}")
print(f"mAP50-95: {results.box.map:.4f}")
PYTHON_SCRIPT

elif [ "$MODEL" == "rfdetr" ]; then
    # Convert to COCO format first
    echo "Converting to COCO format..."
    python3 scripts/convert_to_coco.py --input /content/dataset --output /content/dataset_coco

    python3 << 'PYTHON_SCRIPT'
from rfdetr import RFDETRBase

model = RFDETRBase()
model.train(
    dataset_dir="/content/dataset_coco",
    epochs=50,
    batch_size=8,
    grad_accum_steps=4,
    lr=1e-4,
    output_dir="/content/drive/MyDrive/chana_models/rfdetr_chana",
    checkpoint_freq=10,
    use_ema=True,
)

print("\n" + "="*50)
print("TRAINING COMPLETE!")
print("="*50)
PYTHON_SCRIPT

else
    echo "Unknown model: $MODEL"
    echo "Usage: bash train_colab.sh [yolov8|rfdetr]"
    exit 1
fi

echo ""
echo "=============================================="
echo "TRAINING COMPLETE!"
echo "Model saved to: /content/drive/MyDrive/chana_models/"
echo "=============================================="
