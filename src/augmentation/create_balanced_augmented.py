"""Create a balanced real/night YOLO training set for fair ablation.

The full `data/yolo_aug_night` dataset contains every real training image plus
one augmented copy, so it has twice as many training samples as the baseline.
This script creates a controlled dataset with the same total train size as the
baseline by sampling N real images and pairing each with its night copy.
"""

import argparse
import os
import random
import shutil
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def list_real_images(image_dir: Path) -> list[Path]:
    """Return non-augmented images from a generated real+night YOLO train folder."""
    return sorted(
        p
        for p in image_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTS and not p.stem.endswith("_night")
    )


def write_dataset_yaml(output_root: Path, val_images_dir: Path) -> None:
    val_path = os.path.relpath(val_images_dir, output_root)
    yaml_text = "\n".join(
        [
            "names:",
            "- drone",
            "nc: 1",
            f"path: {output_root.as_posix()}",
            "train: images/train",
            f"val: {Path(val_path).as_posix()}",
            "",
        ]
    )
    (output_root / "dataset.yaml").write_text(yaml_text)


def create_balanced_dataset(
    source_aug_root: Path,
    source_real_root: Path,
    output_root: Path,
    pairs: int,
    seed: int,
    overwrite: bool,
) -> None:
    """Create a dataset containing `pairs` real images and `pairs` night images."""
    src_img_dir = source_aug_root / "images" / "train"
    src_lbl_dir = source_aug_root / "labels" / "train"
    val_img_dir = source_real_root / "images" / "val"
    for required in [src_img_dir, src_lbl_dir, val_img_dir]:
        if not required.exists():
            raise FileNotFoundError(f"Required directory not found: {required}")

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists: {output_root}. Use --overwrite.")
        shutil.rmtree(output_root)

    out_img_dir = output_root / "images" / "train"
    out_lbl_dir = output_root / "labels" / "train"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    real_images = list_real_images(src_img_dir)
    if pairs > len(real_images):
        raise ValueError(f"Requested {pairs} pairs, but only {len(real_images)} real images are available.")

    rng = random.Random(seed)
    selected = rng.sample(real_images, pairs)
    for image_path in selected:
        night_path = src_img_dir / f"{image_path.stem}_night{image_path.suffix.lower()}"
        label_path = src_lbl_dir / f"{image_path.stem}.txt"
        night_label_path = src_lbl_dir / f"{image_path.stem}_night.txt"
        for required in [night_path, label_path, night_label_path]:
            if not required.exists():
                raise FileNotFoundError(f"Missing paired file: {required}")

        shutil.copy2(image_path, out_img_dir / image_path.name)
        shutil.copy2(night_path, out_img_dir / night_path.name)
        shutil.copy2(label_path, out_lbl_dir / label_path.name)
        shutil.copy2(night_label_path, out_lbl_dir / night_label_path.name)

    write_dataset_yaml(output_root, val_img_dir)
    print(f"Wrote balanced augmented dataset to {output_root}")
    print(f"Real train images: {pairs}")
    print(f"Night train images: {pairs}")
    print(f"Total train images: {pairs * 2}")
    print(f"Validation images: {val_img_dir} (real only)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a balanced real/night YOLO dataset.")
    parser.add_argument("--source-aug", default="data/yolo_aug_night", help="Full real+night YOLO dataset")
    parser.add_argument("--source-real", default="data/yolo", help="Original real YOLO dataset")
    parser.add_argument("--output", default="data/yolo_aug_night_balanced", help="Output balanced dataset")
    parser.add_argument("--pairs", type=int, default=250, help="Number of real/night pairs")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic sampling seed")
    parser.add_argument("--overwrite", action="store_true", help="Replace output if it exists")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    create_balanced_dataset(
        source_aug_root=Path(args.source_aug),
        source_real_root=Path(args.source_real),
        output_root=Path(args.output),
        pairs=args.pairs,
        seed=args.seed,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
