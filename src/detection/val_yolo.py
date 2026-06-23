"""
Validate a trained Ultralytics YOLO baseline.
"""
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Validate a trained YOLO baseline.")
    parser.add_argument(
        "--weights",
        default="runs/baseline/yolov8n_smoke/weights/best.pt",
        help="Path to trained YOLO weights",
    )
    parser.add_argument("--data", default="data/yolo/dataset.yaml", help="YOLO dataset YAML")
    parser.add_argument("--imgsz", type=int, default=640, help="Validation image size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--project", default="runs/baseline", help="Output directory")
    parser.add_argument("--name", default="val_yolov8n_smoke", help="Validation run name")
    parser.add_argument("--device", default=None, help="Device passed to Ultralytics, e.g. cpu, 0, mps")
    parser.add_argument("--exist-ok", action="store_true", help="Allow reusing an existing run directory")
    return parser.parse_args()


def main():
    args = parse_args()
    weights_path = Path(args.weights)
    data_path = Path(args.data)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")

    from ultralytics import YOLO

    model = YOLO(str(weights_path))
    val_kwargs = {
        "data": str(data_path),
        "imgsz": args.imgsz,
        "seed": args.seed,
        "project": str(Path(args.project).resolve()),
        "name": args.name,
        "exist_ok": args.exist_ok,
    }
    if args.device:
        val_kwargs["device"] = args.device

    model.val(**val_kwargs)


if __name__ == "__main__":
    main()
