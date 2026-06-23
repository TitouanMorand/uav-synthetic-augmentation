"""
Train a lightweight Ultralytics YOLO baseline on the exported drone dataset.
"""
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Train a YOLO smoke-test baseline.")
    parser.add_argument("--data", default="data/yolo/dataset.yaml", help="YOLO dataset YAML")
    parser.add_argument("--model", default="yolov8n.pt", help="Ultralytics model checkpoint")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs for the smoke test")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--project", default="runs/baseline", help="Output directory")
    parser.add_argument("--name", default="yolov8n_smoke", help="Run name under project")
    parser.add_argument("--batch", type=int, default=-1, help="Batch size, -1 lets Ultralytics choose")
    parser.add_argument("--device", default=None, help="Device passed to Ultralytics, e.g. cpu, 0, mps")
    parser.add_argument("--workers", type=int, default=0, help="Dataloader workers")
    parser.add_argument("--exist-ok", action="store_true", help="Allow reusing an existing run directory")
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")

    from ultralytics import YOLO

    model = YOLO(args.model)
    train_kwargs = {
        "data": str(data_path),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "seed": args.seed,
        "project": args.project,
        "name": args.name,
        "batch": args.batch,
        "workers": args.workers,
        "exist_ok": args.exist_ok,
    }
    if args.device:
        train_kwargs["device"] = args.device

    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
