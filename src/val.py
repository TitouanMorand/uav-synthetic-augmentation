"""
Minimal validation wrapper using Ultralytics YOLO.
"""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data-dir", required=True)
    args = parser.parse_args()
    from ultralytics import YOLO

    model = YOLO(args.weights)
    data_yaml = Path(args.data_dir) / "data.yaml"
    model.val(data=str(data_yaml))


if __name__ == "__main__":
    main()
