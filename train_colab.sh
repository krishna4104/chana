#!/bin/bash
# One-click training script for Google Colab
# Usage: !bash train_colab.sh [gdrive_url_or_id] [yolov8|rfdetr]
# Example: !bash train_colab.sh "https://drive.google.com/file/d/1MFjRPb70EpCg5YgGTqBvS_2b_sapOGPY/view?usp=sharing" yolov8
# Or:      !bash train_colab.sh 1MFjRPb70EpCg5YgGTqBvS_2b_sapOGPY yolov8
# Or:      !bash train_colab.sh  (uses default dataset)

set -e

INPUT=${1:-"1MFjRPb70EpCg5YgGTqBvS_2b_sapOGPY"}
MODEL=${2:-yolov8}

# Extract file ID from URL if full URL provided
if [[ "$INPUT" == *"drive.google.com"* ]]; then
    GDRIVE_ID=$(echo "$INPUT" | sed -n 's/.*\/d\/\([^\/]*\).*/\1/p')
else
    GDRIVE_ID="$INPUT"
fi

echo "=============================================="
echo "CHANA CLASSIFIER - $MODEL TRAINING"
echo "Dataset ID: $GDRIVE_ID"
echo "=============================================="

# Install dependencies
echo "[1/6] Installing dependencies..."
pip install -q ultralytics rfdetr pycocotools gdown

# Download dataset from Google Drive
echo "[2/6] Downloading dataset from Google Drive..."
if [ ! -f "/content/dataset.zip" ]; then
    gdown "https://drive.google.com/uc?id=$GDRIVE_ID" -O /content/dataset.zip
else
    echo "Dataset zip already exists, skipping download."
fi

# Extract dataset
echo "[3/6] Extracting dataset..."
if [ ! -d "/content/dataset" ]; then
    unzip -q -o /content/dataset.zip -d /content/

    # Handle different extracted folder names
    if [ -d "/content/filtered_dataset" ]; then
        mv /content/filtered_dataset /content/dataset
    elif [ ! -d "/content/dataset" ]; then
        # Find any extracted folder that's not sample_data
        EXTRACTED=$(ls -d /content/*/ 2>/dev/null | grep -vE 'sample_data|chana|drive' | head -1)
        if [ -n "$EXTRACTED" ] && [ "$EXTRACTED" != "/content/dataset/" ]; then
            mv "$EXTRACTED" /content/dataset 2>/dev/null || true
        fi
    fi
else
    echo "Dataset already extracted."
fi

# Check dataset structure and fix if needed
echo "[4/6] Verifying dataset structure..."
if [ -d "/content/dataset/train/images" ]; then
    echo "Standard structure detected."
elif [ -d "/content/dataset/images/train" ]; then
    echo "Alternative structure detected, reorganizing..."
    mkdir -p /content/dataset_fixed/{train,valid,test}/{images,labels}
    mv /content/dataset/images/train/* /content/dataset_fixed/train/images/ 2>/dev/null || true
    mv /content/dataset/images/valid/* /content/dataset_fixed/valid/images/ 2>/dev/null || true
    mv /content/dataset/images/test/* /content/dataset_fixed/test/images/ 2>/dev/null || true
    mv /content/dataset/labels/train/* /content/dataset_fixed/train/labels/ 2>/dev/null || true
    mv /content/dataset/labels/valid/* /content/dataset_fixed/valid/labels/ 2>/dev/null || true
    mv /content/dataset/labels/test/* /content/dataset_fixed/test/labels/ 2>/dev/null || true
    rm -rf /content/dataset
    mv /content/dataset_fixed /content/dataset
else
    echo "Checking for flat structure..."
    ls -la /content/dataset/
fi

# Count images
TRAIN_COUNT=$(find /content/dataset/train/images -type f \( -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" \) 2>/dev/null | wc -l)
VALID_COUNT=$(find /content/dataset/valid/images -type f \( -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" \) 2>/dev/null | wc -l)
TEST_COUNT=$(find /content/dataset/test/images -type f \( -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" \) 2>/dev/null | wc -l)
echo "Train: $TRAIN_COUNT, Valid: $VALID_COUNT, Test: $TEST_COUNT"

if [ "$TRAIN_COUNT" -eq 0 ]; then
    echo "ERROR: No training images found!"
    echo "Dataset structure:"
    find /content/dataset -type d | head -20
    exit 1
fi

# Create data.yaml
echo "[5/6] Creating config..."
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

# Mount Google Drive for saving models
echo "Mounting Google Drive..."
python3 -c "from google.colab import drive; drive.mount('/content/drive')" 2>/dev/null || echo "Drive already mounted or not in Colab"
mkdir -p /content/drive/MyDrive/chana_models

# Train
echo "[6/6] Starting training..."
if [ "$MODEL" == "yolov8" ]; then
    python3 << 'PYTHON_SCRIPT'
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
results = model.train(
    data='/content/data.yaml',
    epochs=50,           # Reduced - early stopping will kick in anyway
    batch=64,            # Increased for faster training on T4
    imgsz=512,           # Slightly smaller for speed, still good quality
    optimizer='AdamW',
    lr0=0.002,           # Slightly higher LR for faster convergence
    degrees=180,
    flipud=0.5,
    fliplr=0.5,
    hsv_s=0.7,
    hsv_v=0.4,
    mosaic=1.0,
    mixup=0.1,
    close_mosaic=5,      # Disable mosaic earlier
    device=0,
    workers=8,           # More workers for faster data loading
    amp=True,
    cache='ram',         # Cache images in RAM for faster loading
    project='/content/drive/MyDrive/chana_models',
    name='yolov8n_chana',
    patience=10,         # Stop earlier if no improvement
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
    echo "Usage: bash train_colab.sh <gdrive_id> [yolov8|rfdetr]"
    exit 1
fi

echo ""
echo "=============================================="
echo "TRAINING COMPLETE!"
echo "Model saved to: /content/drive/MyDrive/chana_models/"
echo "=============================================="
