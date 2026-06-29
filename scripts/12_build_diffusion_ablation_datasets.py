from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.augment import copy_split
from src.config import load_config
from src.dataset import validate_yolo_dataset
from src.utils import ensure_dir, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build YOLO datasets with different numbers of diffusion images.")

    parser.add_argument("--sizes", nargs="+", type=int, default=[75, 150, 300])
    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


def write_dataset_yaml(output_root: Path) -> Path:
    dataset_yaml = {
        "path": str(output_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 1,
        "names": ["drone"],
    }

    path = output_root / "dataset.yaml"

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(dataset_yaml, f, sort_keys=False)

    return path


def build_dataset(config: dict, accepted_df: pd.DataFrame, size: int, overwrite: bool) -> dict:
    baseline_root = Path(config["experiments"]["baseline"]["dataset_root"])
    output_root = Path(f"data/augmented/yolo_drone_diffusion_reinsert_night_n{size:03d}")

    if len(accepted_df) < size:
        raise ValueError(f"Requested size={size}, but only {len(accepted_df)} accepted diffusion images available.")

    if output_root.exists():
        if overwrite:
            shutil.rmtree(output_root)
        else:
            raise FileExistsError(f"Output dataset exists: {output_root}. Use --overwrite.")

    for split in ["train", "val", "test"]:
        ensure_dir(output_root / "images" / split)
        ensure_dir(output_root / "labels" / split)

    copied_train = copy_split(baseline_root, output_root, "train")
    copied_val = copy_split(baseline_root, output_root, "val")
    copied_test = copy_split(baseline_root, output_root, "test")

    subset = accepted_df.head(size).copy()

    for _, row in subset.iterrows():
        src_image = Path(row["generated_image"])
        src_label = Path(row["generated_label"])

        dst_image = output_root / "images" / "train" / src_image.name
        dst_label = output_root / "labels" / "train" / src_label.name

        shutil.copy2(src_image, dst_image)
        shutil.copy2(src_label, dst_label)

    dataset_yaml = write_dataset_yaml(output_root)
    validation = validate_yolo_dataset(output_root)

    report = {
        "size": size,
        "output_root": str(output_root),
        "dataset_yaml": str(dataset_yaml),
        "copied_real_train": copied_train,
        "added_diffusion_train": size,
        "total_train": copied_train + size,
        "copied_val": copied_val,
        "copied_test": copied_test,
        "validation": validation,
    }

    return report


def main() -> None:
    args = parse_args()
    config = load_config()

    tables_dir = ensure_dir(config["paths"]["tables"])
    reports_dir = ensure_dir(config["paths"]["reports"])

    accepted_csv = tables_dir / "diffusion_reinsert_night_pool_accepted.csv"

    if not accepted_csv.exists():
        raise FileNotFoundError(
            f"Missing accepted diffusion pool CSV: {accepted_csv}. "
            f"Run scripts/11_generate_diffusion_reinsert_pool.py first."
        )

    accepted_df = pd.read_csv(accepted_csv)

    reports = {}

    for size in args.sizes:
        report = build_dataset(
            config=config,
            accepted_df=accepted_df,
            size=size,
            overwrite=args.overwrite,
        )

        if not report["validation"]["is_valid"]:
            print(f"Validation failed for diffusion dataset N={size}")
            raise SystemExit(1)

        reports[f"n{size:03d}"] = report

        print(
            f"Built diffusion ablation dataset N={size}: "
            f"{report['output_root']} | total_train={report['total_train']}"
        )

    save_json(reports, reports_dir / "diffusion_ablation_datasets_report.json")

    print("Step 12 completed successfully.")


if __name__ == "__main__":
    main()
