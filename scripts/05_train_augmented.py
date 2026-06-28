from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.train_eval import evaluate_yolo, train_yolo
from src.utils import save_json, set_seed


VALID_EXPERIMENTS = ["classic", "object_preserving"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO on an augmented dataset.")

    parser.add_argument(
        "--experiment",
        type=str,
        default="classic",
        choices=VALID_EXPERIMENTS,
        help="Augmented experiment to train.",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()

    set_seed(int(config["yolo"]["seed"]))

    experiment_cfg = config["experiments"][args.experiment]

    dataset_root = Path(experiment_cfg["dataset_root"])
    dataset_yaml = dataset_root / "dataset.yaml"

    run_name = args.run_name or experiment_cfg["name"]

    epochs = args.epochs
    if epochs is None:
        epochs = int(config["yolo"]["epochs_augmented"])

    train_summary = train_yolo(
        config=config,
        dataset_yaml=dataset_yaml,
        run_name=run_name,
        epochs=epochs,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
    )

    eval_report = evaluate_yolo(
        config=config,
        dataset_yaml=dataset_yaml,
        weights_path=Path(train_summary["best_weights"]),
        run_name=run_name,
        splits=["val", "test"],
        device=args.device,
    )

    result = {
        "experiment": args.experiment,
        "training": train_summary,
        "evaluation": eval_report,
    }

    save_json(
        result,
        Path(config["paths"]["reports"]) / f"{run_name}_full_result.json",
    )

    print("Step 05 completed successfully.")


if __name__ == "__main__":
    main()
