from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.diffusion import make_diffusion_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a 3-mode diffusion comparison grid.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()

    report = make_diffusion_grid(
        config=config,
        overwrite=args.overwrite,
        max_images=args.max_images,
        start_index=args.start_index,
    )

    if not report["validation"]["is_valid"]:
        print("Diffusion grid dataset validation failed.")
        raise SystemExit(1)

    print("Step 07 completed successfully.")


if __name__ == "__main__":
    main()
