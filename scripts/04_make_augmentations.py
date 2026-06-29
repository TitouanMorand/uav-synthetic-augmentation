from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.augment import (
    make_classic_augmentations,
    make_object_preserving_augmentations,
)
from src.config import load_config
from src.visualize import make_box_distribution_plot, make_contact_sheet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create augmented YOLO datasets.")

    parser.add_argument(
        "--mode",
        type=str,
        default="classic",
        choices=["classic", "object_preserving", "all"],
        help="Which augmentation dataset to create.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--per-image", type=int, default=None)

    return parser.parse_args()


def preview_dataset(
    dataset_root: Path,
    previews_dir: Path,
    prefix: str,
) -> None:
    preview_path = previews_dir / f"{prefix}_train_contact_sheet.jpg"
    make_contact_sheet(
        dataset_root=dataset_root,
        split="train",
        output_path=preview_path,
        max_images=24,
    )

    box_plot_path = previews_dir / f"{prefix}_box_distribution.png"
    make_box_distribution_plot(
        dataset_root=dataset_root,
        output_path=box_plot_path,
    )

    print(f"Saved preview: {preview_path}")
    print(f"Saved box distribution: {box_plot_path}")


def main() -> None:
    args = parse_args()
    config = load_config()

    previews_dir = Path(config["paths"]["previews"])

    if args.mode in ["classic", "all"]:
        classic_report = make_classic_augmentations(
            config=config,
            overwrite=args.overwrite,
            per_image=args.per_image,
        )

        if not classic_report["validation"]["is_valid"]:
            print("Classic augmented dataset validation failed.")
            raise SystemExit(1)

        preview_dataset(
            dataset_root=Path(classic_report["output_root"]),
            previews_dir=previews_dir,
            prefix="classic",
        )

    if args.mode in ["object_preserving", "all"]:
        object_report = make_object_preserving_augmentations(
            config=config,
            overwrite=args.overwrite,
            per_image=args.per_image,
        )

        if not object_report["validation"]["is_valid"]:
            print("Object-preserving augmented dataset validation failed.")
            raise SystemExit(1)

        preview_dataset(
            dataset_root=Path(object_report["output_root"]),
            previews_dir=previews_dir,
            prefix="object_preserving",
        )

    print("Step 04 completed successfully.")


if __name__ == "__main__":
    main()
