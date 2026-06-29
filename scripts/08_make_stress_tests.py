from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.stress import make_stress_datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create synthetic stress-test datasets.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()

    reports = make_stress_datasets(
        config=config,
        overwrite=args.overwrite,
    )

    for mode, report in reports.items():
        if not report["validation"]["is_valid"]:
            print(f"Stress dataset validation failed: {mode}")
            raise SystemExit(1)

    print("Step 08 completed successfully.")


if __name__ == "__main__":
    main()
