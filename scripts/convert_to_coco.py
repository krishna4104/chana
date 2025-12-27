#!/usr/bin/env python3
"""
Convert YOLO format dataset to COCO JSON format for RF-DETR training.
Usage: python convert_to_coco.py --input /path/to/yolo --output /path/to/coco
"""

import json
import os
import shutil
import argparse
from PIL import Image
from tqdm import tqdm

CLASS_NAMES = [
    "Bad Black Channa",
    "Bad Kabuli Chana",
    "Good Black Channa",
    "Good Kabuli Chana",
    "foreign material",
    "Bad Soya",
    "Good Soya",
    "Bad Matar",
    "Good Matar"
]

def yolo_to_coco(yolo_dir, output_dir, split):
    """Convert YOLO format to COCO JSON format."""
    images_dir = os.path.join(yolo_dir, split, 'images')
    labels_dir = os.path.join(yolo_dir, split, 'labels')

    out_split_dir = os.path.join(output_dir, split)
    os.makedirs(out_split_dir, exist_ok=True)

    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": name} for i, name in enumerate(CLASS_NAMES)]
    }

    ann_id = 1
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])

    for img_id, img_file in enumerate(tqdm(image_files, desc=f"Converting {split}"), 1):
        img_path = os.path.join(images_dir, img_file)
        label_path = os.path.join(labels_dir, os.path.splitext(img_file)[0] + '.txt')

        # Get image dimensions
        with Image.open(img_path) as img:
            width, height = img.size

        # Copy image to output
        shutil.copy(img_path, os.path.join(out_split_dir, img_file))

        coco["images"].append({
            "id": img_id,
            "file_name": img_file,
            "width": width,
            "height": height
        })

        # Parse YOLO labels
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center, y_center, w, h = map(float, parts[1:5])

                        # Convert to COCO format (absolute coordinates)
                        x = (x_center - w/2) * width
                        y = (y_center - h/2) * height
                        w = w * width
                        h = h * height

                        coco["annotations"].append({
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": class_id,
                            "bbox": [x, y, w, h],
                            "area": w * h,
                            "iscrowd": 0
                        })
                        ann_id += 1

    # Save COCO JSON
    json_path = os.path.join(out_split_dir, '_annotations.coco.json')
    with open(json_path, 'w') as f:
        json.dump(coco, f)

    print(f"{split}: {len(coco['images'])} images, {len(coco['annotations'])} annotations")
    return coco

def main():
    parser = argparse.ArgumentParser(description='Convert YOLO to COCO format')
    parser.add_argument('--input', required=True, help='Input YOLO dataset directory')
    parser.add_argument('--output', required=True, help='Output COCO dataset directory')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    for split in ['train', 'valid', 'test']:
        if os.path.exists(os.path.join(args.input, split)):
            yolo_to_coco(args.input, args.output, split)

    print("\nConversion complete!")

if __name__ == "__main__":
    main()
