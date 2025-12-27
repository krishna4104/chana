# Chana Quality Classifier

Real-time grain quality detection using YOLOv8 and RF-DETR for automated chana (chickpea) grading.

## Dataset

- **Total Images**: 35,019
- **Classes**: 9

| ID | Class |
|----|-------|
| 0 | Bad Black Channa |
| 1 | Bad Kabuli Chana |
| 2 | Good Black Channa |
| 3 | Good Kabuli Chana |
| 4 | Foreign Material |
| 5 | Bad Soya |
| 6 | Good Soya |
| 7 | Bad Matar |
| 8 | Good Matar |

## Quick Start (Google Colab)

### Option 1: YOLOv8 Training
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/chana-classifier/blob/main/notebooks/train_yolov8.ipynb)

### Option 2: RF-DETR Training (SOTA)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/chana-classifier/blob/main/notebooks/train_rfdetr.ipynb)

## Project Structure

```
chana-classifier/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── train_yolov8.ipynb      # YOLOv8 training notebook
│   └── train_rfdetr.ipynb      # RF-DETR training notebook
├── scripts/
│   ├── convert_to_coco.py      # YOLO to COCO format converter
│   ├── evaluate.py             # Model evaluation
│   └── inference.py            # Real-time inference
├── configs/
│   └── data.yaml               # Dataset config
└── weights/                    # Trained model weights (after training)
```

## Setup

### 1. Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/chana-classifier.git
cd chana-classifier
```

### 2. Upload Dataset to Google Drive
Upload `dataset.zip` to your Google Drive root folder.

### 3. Open Colab Notebook
- For YOLOv8: Open `notebooks/train_yolov8.ipynb`
- For RF-DETR: Open `notebooks/train_rfdetr.ipynb`

## Local Training

```bash
# Install dependencies
pip install -r requirements.txt

# Train YOLOv8
python scripts/train_yolov8.py

# Train RF-DETR
python scripts/train_rfdetr.py
```

## Results

| Model | mAP50 | mAP50-95 | FPS (T4) | Size |
|-------|-------|----------|----------|------|
| YOLOv8n | 76.6% | 41.8% | 172 | 6MB |
| RF-DETR-S | TBD | TBD | TBD | TBD |

## License

MIT License
