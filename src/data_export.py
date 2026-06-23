"""
Export a small subset of `pathikg/drone-detection-dataset` to YOLO directory structure.

Outputs:
  output/
    images/train/*.jpg
    images/val/*.jpg
    labels/train/*.txt
    labels/val/*.txt

Each label file contains lines: `class x_center y_center width height` (normalized floats).
"""
import argparse
import os
import random
from pathlib import Path
from datasets import load_dataset
import shutil
from tqdm import tqdm
from PIL import Image
from .utils import coco_to_yolo


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def export_subset(dataset_name: str, output_dir: str, train_size: int, val_size: int, seed: int = 42):
    out = Path(output_dir)
    images_train = out / "images" / "train"
    images_val = out / "images" / "val"
    labels_train = out / "labels" / "train"
    labels_val = out / "labels" / "val"
    for d in [images_train, images_val, labels_train, labels_val]:
        ensure_dir(d)

    # Load dataset (robust to returned mapping)
    ds_all = load_dataset(dataset_name)
    # Try to collect into a single list of examples
    examples = []
    if isinstance(ds_all, dict):
        # concat splits
        for split in ds_all:
            examples.extend(list(ds_all[split]))
    else:
        examples = list(ds_all)

    random.seed(seed)
    random.shuffle(examples)
    if train_size + val_size > len(examples):
        raise ValueError("Not enough examples in dataset for requested sizes")

    train_ex = examples[:train_size]
    val_ex = examples[train_size:train_size+val_size]

    def save_examples(ex_list, img_dir: Path, label_dir: Path):
        for i, ex in enumerate(tqdm(ex_list, desc=f"Exporting to {img_dir.parent.name}")):
            # image may be PIL image or dict
            image = ex.get("image") if isinstance(ex, dict) else None
            if image is None:
                # fallback: try keys
                for k in ex:
                    if k.lower() == "image":
                        image = ex[k]
                        break
            if image is None:
                raise RuntimeError("Could not find image in example keys: " + ",".join(ex.keys()))

            # Save image to disk
            img_name = f"{i:06d}.jpg"
            img_path = img_dir / img_name
            if isinstance(image, str):
                # local path
                shutil.copy(image, img_path)
            else:
                image.save(img_path)

            # Extract COCO-style bboxes: try common keys
            bboxes = None
            categories = None
            if "objects" in ex and isinstance(ex["objects"], dict):
                bboxes = ex["objects"].get("bbox")
                categories = ex["objects"].get("category")
            if "annotations" in ex and isinstance(ex["annotations"], dict):
                # sometimes annotations is dict with lists
                a = ex["annotations"]
                bboxes = a.get("bbox")
                categories = a.get("category_id") or a.get("category")
            if bboxes is None and "annotations" in ex and isinstance(ex["annotations"], list):
                # list of dicts
                boxes = []
                cats = []
                for ann in ex["annotations"]:
                    if "bbox" in ann:
                        boxes.append(ann["bbox"])
                        cats.append(ann.get("category_id", ann.get("category", 0)))
                bboxes = boxes
                categories = cats
            if bboxes is None and "bbox" in ex:
                bboxes = ex["bbox"]

            # When bboxes found as list of lists
            if bboxes is None:
                # No object in image; write empty label file
                (label_dir / (img_name.replace('.jpg', '.txt'))).write_text("")
            else:
                # Ensure we have width/height
                with Image.open(img_path) as im:
                    w, h = im.size
                # bboxes may be nested
                lines = []
                if len(bboxes) == 0:
                    pass
                else:
                    # handle single bbox dict/list
                    if isinstance(bboxes[0], (int, float)):
                        # single bbox
                        yolo = coco_to_yolo(bboxes, w, h)
                        yolo = [max(0.0, min(1.0, v)) for v in yolo]
                        cls = int(categories[0]) if categories else 0
                        lines.append(f"{cls} {' '.join(f'{v:.6f}' for v in yolo)}")
                    else:
                        for j, bb in enumerate(bboxes):
                            # bb may be dict
                            if isinstance(bb, dict):
                                raw = bb.get("bbox") or bb.get("box")
                                if raw is None:
                                    continue
                                bb = raw
                            yolo = coco_to_yolo(bb, w, h)
                            yolo = [max(0.0, min(1.0, v)) for v in yolo]
                            cls = int(categories[j]) if categories and j < len(categories) else 0
                            lines.append(f"{cls} {' '.join(f'{v:.6f}' for v in yolo)}")

                (label_dir / (img_name.replace('.jpg', '.txt'))).write_text("\n".join(lines))

    save_examples(train_ex, images_train, labels_train)
    save_examples(val_ex, images_val, labels_val)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default="pathikg/drone-detection-dataset")
    parser.add_argument("--output", default="data/yolo_subset")
    parser.add_argument("--train-size", type=int, default=500)
    parser.add_argument("--val-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    export_subset(args.dataset_name, args.output, args.train_size, args.val_size, args.seed)


if __name__ == "__main__":
    main()
