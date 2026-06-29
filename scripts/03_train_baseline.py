from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.train_eval import train_and_evaluate_baseline
from src.utils import save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate YOLO baseline on HF drone dataset.")

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

    run_name = args.run_name or config["experiments"]["baseline"]["name"]

    result = train_and_evaluate_baseline(
        config=config,
        run_name=run_name,
        epochs=args.epochs,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
    )

    save_json(
        result,
        Path(config["paths"]["reports"]) / f"{run_name}_full_result.json",
    )

    print("Step 03 completed successfully.")


if __name__ == "__main__":
    main()
