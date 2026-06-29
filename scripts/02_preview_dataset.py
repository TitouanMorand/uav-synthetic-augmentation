from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.visualize import make_box_distribution_plot, make_contact_sheet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-images", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()

    dataset_root = Path(config["dataset"]["output_root"])
    previews_dir = Path(config["paths"]["previews"])

    for split in ["train", "val", "test"]:
        output_path = previews_dir / f"{split}_contact_sheet.jpg"
        make_contact_sheet(
            dataset_root=dataset_root,
            split=split,
            output_path=output_path,
            max_images=args.max_images,
        )
        print(f"Saved {split} preview: {output_path}")

    plot_path = previews_dir / "box_distribution.png"
    make_box_distribution_plot(
        dataset_root=dataset_root,
        output_path=plot_path,
    )
    print(f"Saved box distribution plot: {plot_path}")

    print("Step 02 completed successfully.")


if __name__ == "__main__":
    main()
