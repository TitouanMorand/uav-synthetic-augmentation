from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.diffusion import make_diffusion_preview_dataset
from src.visualize import make_box_distribution_plot, make_contact_sheet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create object-preserving diffusion preview images.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-images", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()

    report = make_diffusion_preview_dataset(
        config=config,
        overwrite=args.overwrite,
        max_images=args.max_images,
    )

    if not report["validation"]["is_valid"]:
        print("Diffusion preview dataset validation failed.")
        raise SystemExit(1)

    output_root = Path(report["output_root"])
    previews_dir = Path(config["paths"]["previews"])

    contact_sheet = previews_dir / "diffusion_preview_contact_sheet.jpg"
    make_contact_sheet(
        dataset_root=output_root,
        split="train",
        output_path=contact_sheet,
        max_images=24,
    )

    box_plot = previews_dir / "diffusion_preview_box_distribution.png"
    make_box_distribution_plot(
        dataset_root=output_root,
        output_path=box_plot,
    )

    print(f"Saved diffusion preview: {contact_sheet}")
    print(f"Saved box distribution: {box_plot}")
    print("Step 07 completed successfully.")


if __name__ == "__main__":
    main()
