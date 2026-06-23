"""
Download and export a small subset from a Hugging Face dataset to YOLO format.

Outputs into `--output`:
  images/train, images/val, labels/train, labels/val, dataset.yaml

Each label file follows YOLO format: `class_id x_center y_center width height` (normalized floats).
"""
import argparse
from pathlib import Path
from datasets import load_dataset
from PIL import Image
import random
import shutil
import yaml
from .coco_to_yolo import coco_to_yolo




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

    # To minimize downloads, request only the slices we need (datasets supports split slicing)
    try:
        ds_train = load_dataset(dataset_name, split=f"train[:{train_size}]")
    except Exception:
        # fallback to full train split (may be large)
        ds_train = load_dataset(dataset_name, split="train")

    # Determine validation split
    try:
        ds_val = load_dataset(dataset_name, split=f"test[:{val_size}]")
    except Exception:
        try:
            ds_val = load_dataset(dataset_name, split=f"validation[:{val_size}]")
        except Exception:
            # fallback: take slice after train in train split
            try:
                ds_val = load_dataset(dataset_name, split=f"train[{train_size}:{train_size+val_size}]")
            except Exception:
                ds_val = []

    def save_dataset(ds_obj, img_dir: Path, lbl_dir: Path):
        for i, ex in enumerate(ds_obj):
            image_obj = ex.get("image") if isinstance(ex, dict) else None
            if image_obj is None:
                for k in ex:
                    if k.lower() == "image":
                        image_obj = ex[k]
                        break
            if image_obj is None:
                continue
            img_name = f"{i:06d}.jpg"
            img_path = img_dir / img_name
            if isinstance(image_obj, str):
                shutil.copy(image_obj, img_path)
            else:
                if hasattr(image_obj, "save"):
                    image_obj.save(img_path)
                else:
                    Image.fromarray(image_obj).save(img_path)

            bboxes = []
            categories = []
            if "objects" in ex and isinstance(ex["objects"], dict):
                bboxes = ex["objects"].get("bbox") or []
                categories = ex["objects"].get("category") or []
            if "annotations" in ex:
                ann = ex["annotations"]
                if isinstance(ann, dict):
                    bboxes = ann.get("bbox") or []
                    categories = ann.get("category_id") or ann.get("category") or []
                if isinstance(ann, list):
                    for a in ann:
                        if "bbox" in a:
                            bboxes.append(a["bbox"])
                            categories.append(a.get("category_id", a.get("category", 0)))
            if not bboxes and "bbox" in ex:
                b = ex["bbox"]
                if isinstance(b, list) and len(b) > 0 and isinstance(b[0], (list, tuple)):
                    bboxes = b

            lbl_path = lbl_dir / img_name.replace('.jpg', '.txt')
            if not bboxes:
                lbl_path.write_text("")
            else:
                with Image.open(img_path) as im:
                    w, h = im.size
                lines = []
                for j, bb in enumerate(bboxes):
                    x_c, y_c, bw, bh = coco_to_yolo(bb, w, h)
                    x_c = max(0.0, min(1.0, x_c))
                    y_c = max(0.0, min(1.0, y_c))
                    bw = max(0.0, min(1.0, bw))
                    bh = max(0.0, min(1.0, bh))
                    cls = int(categories[j]) if j < len(categories) else 0
                    lines.append(f"{cls} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}")
                lbl_path.write_text("\n".join(lines))

    save_dataset(ds_train, images_train, labels_train)
    save_dataset(ds_val, images_val, labels_val)

    data_yaml = {
        "path": str(out),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": ["drone"],
    }
    with open(out / "dataset.yaml", "w") as f:
        yaml.safe_dump(data_yaml, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="pathikg/drone-detection-dataset")
    parser.add_argument("--output", default="data/yolo")
    parser.add_argument("--train-size", type=int, default=500)
    parser.add_argument("--val-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    export_subset(args.dataset, args.output, args.train_size, args.val_size, args.seed)


if __name__ == "__main__":
    main()
