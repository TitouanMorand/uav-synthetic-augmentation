from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.dataset import prepare_hf_yolo_dataset
from src.utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()
    set_seed(int(config["yolo"]["seed"]))

    report = prepare_hf_yolo_dataset(
        config=config,
        overwrite=args.overwrite,
    )

    if not report["validation"]["is_valid"]:
        print("Dataset validation failed.")
        raise SystemExit(1)

    if report["leakage"]["has_leakage"]:
        print("Potential train/val/test leakage detected.")
        raise SystemExit(1)

    print("Step 01 completed successfully.")


if __name__ == "__main__":
    main()
