from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.augment import make_classic_augmentations
from src.config import load_config
from src.visualize import make_box_distribution_plot, make_contact_sheet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create classic augmented YOLO dataset.")

    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--per-image", type=int, default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()

    report = make_classic_augmentations(
        config=config,
        overwrite=args.overwrite,
        per_image=args.per_image,
    )

    if not report["validation"]["is_valid"]:
        print("Classic augmented dataset validation failed.")
        raise SystemExit(1)

    output_root = Path(report["output_root"])
    previews_dir = Path(config["paths"]["previews"])

    preview_path = previews_dir / "classic_train_contact_sheet.jpg"
    make_contact_sheet(
        dataset_root=output_root,
        split="train",
        output_path=preview_path,
        max_images=24,
    )

    box_plot_path = previews_dir / "classic_box_distribution.png"
    make_box_distribution_plot(
        dataset_root=output_root,
        output_path=box_plot_path,
    )

    print(f"Saved classic preview: {preview_path}")
    print(f"Saved classic box distribution: {box_plot_path}")
    print("Step 04 completed successfully.")


if __name__ == "__main__":
    main()
