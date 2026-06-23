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
    parser.add_argument("--fraction", type=float, default=1.0, help="Fraction of training data to use")
    parser.add_argument("--cache", action="store_true", help="Cache images for faster repeated runs")
    parser.add_argument("--plots", action="store_true", help="Save training plots")
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
        "project": str(Path(args.project).resolve()),
        "name": args.name,
        "batch": args.batch,
        "workers": args.workers,
        "fraction": args.fraction,
        "cache": args.cache,
        "plots": args.plots,
        "exist_ok": args.exist_ok,
    }
    if args.device:
        train_kwargs["device"] = args.device

    model.train(**train_kwargs)

    save_dir = Path(model.trainer.save_dir)
    results_csv = save_dir / "results.csv"
    if results_csv.exists():
        from src.detection.summarize_results import summarize_results

        summary_path = summarize_results(results_csv)
        print(f"Readable metrics summary: {summary_path}")


if __name__ == "__main__":
    main()
