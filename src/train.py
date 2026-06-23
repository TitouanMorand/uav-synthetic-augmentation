"""
Minimal Ultralytics YOLO training wrapper for the exported dataset.

This script generates a small dataset YAML pointing to the images/ and labels/ folders
and runs a short training run for smoke-testing.
"""
import argparse
from pathlib import Path
import yaml


def make_data_yaml(data_dir: Path, output: Path):
    # With `path` set, Ultralytics resolves train/val relative to the dataset root.
    d = {
        "path": str(data_dir),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": ["drone"],
    }
    with open(output, "w") as f:
        yaml.safe_dump(d, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="YOLO dataset directory")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--model", default="yolov8n.pt", help="Ultralytics model checkpoint or name")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    data_yaml = data_dir / "data.yaml"
    make_data_yaml(data_dir, data_yaml)

    # Import here to avoid requiring ultralytics for other scripts
    from ultralytics import YOLO

    model = YOLO(args.model)
    # train - minimal config for smoke test
    model.train(data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz)


if __name__ == "__main__":
    main()
