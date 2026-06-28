from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from src.dataset import validate_yolo_dataset
from src.utils import ensure_dir, save_json


YoloLabel = tuple[int, float, float, float, float]


def read_yolo_labels(label_path: str | Path) -> list[YoloLabel]:
    label_path = Path(label_path)

    if not label_path.exists():
        return []

    labels: list[YoloLabel] = []

    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue

        cls, x, y, w, h = line.split()
        labels.append((int(cls), float(x), float(y), float(w), float(h)))

    return labels


def write_yolo_labels(labels: list[YoloLabel], label_path: str | Path) -> None:
    label_path = Path(label_path)
    ensure_dir(label_path.parent)

    lines = [
        f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
        for cls, x, y, w, h in labels
    ]

    label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def copy_split(
    input_root: Path,
    output_root: Path,
    split: str,
) -> int:
    input_image_dir = input_root / "images" / split
    input_label_dir = input_root / "labels" / split

    output_image_dir = ensure_dir(output_root / "images" / split)
    output_label_dir = ensure_dir(output_root / "labels" / split)

    count = 0

    for image_path in sorted(input_image_dir.glob("*.jpg")):
        label_path = input_label_dir / f"{image_path.stem}.txt"

        if not label_path.exists():
            continue

        shutil.copy2(image_path, output_image_dir / image_path.name)
        shutil.copy2(label_path, output_label_dir / label_path.name)
        count += 1

    return count


def horizontal_flip(
    image: np.ndarray,
    labels: list[YoloLabel],
) -> tuple[np.ndarray, list[YoloLabel]]:
    flipped_image = cv2.flip(image, 1)

    flipped_labels = []

    for cls, x, y, w, h in labels:
        flipped_labels.append((cls, 1.0 - x, y, w, h))

    return flipped_image, flipped_labels


def random_brightness_contrast(
    image: np.ndarray,
    rng: random.Random,
) -> np.ndarray:
    alpha = rng.uniform(0.75, 1.30)
    beta = rng.uniform(-35, 35)

    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def random_hsv_shift(
    image: np.ndarray,
    rng: random.Random,
) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.int16)

    hue_shift = rng.randint(-10, 10)
    sat_scale = rng.uniform(0.75, 1.30)
    val_scale = rng.uniform(0.75, 1.30)

    hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_scale, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * val_scale, 0, 255)

    hsv = hsv.astype(np.uint8)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def random_blur_or_noise(
    image: np.ndarray,
    rng: random.Random,
) -> np.ndarray:
    if rng.random() < 0.5:
        kernel_size = rng.choice([3, 5])
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    noise_sigma = rng.uniform(4, 12)
    noise = rng.normalvariate(0, noise_sigma)

    noise_array = np.random.normal(
        loc=noise,
        scale=noise_sigma,
        size=image.shape,
    )

    noisy = image.astype(np.float32) + noise_array
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)

    return noisy


def make_classic_variant(
    image: np.ndarray,
    labels: list[YoloLabel],
    rng: random.Random,
    variant_index: int,
) -> tuple[np.ndarray, list[YoloLabel], list[str]]:
    """
    Apply simple, explainable augmentations.

    Photometric transforms do not change labels.
    Horizontal flip changes x_center -> 1 - x_center.
    """
    output_image = image.copy()
    output_labels = labels[:]
    transforms = []

    if variant_index % 2 == 0:
        output_image = random_brightness_contrast(output_image, rng)
        transforms.append("brightness_contrast")

        output_image = random_hsv_shift(output_image, rng)
        transforms.append("hsv_shift")

    else:
        output_image = random_blur_or_noise(output_image, rng)
        transforms.append("blur_or_noise")

        output_image = random_brightness_contrast(output_image, rng)
        transforms.append("brightness_contrast")

    if rng.random() < 0.5:
        output_image, output_labels = horizontal_flip(output_image, output_labels)
        transforms.append("horizontal_flip")

    return output_image, output_labels, transforms


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


def make_classic_augmentations(
    config: dict[str, Any],
    overwrite: bool = False,
    per_image: int | None = None,
) -> dict[str, Any]:
    seed = int(config["yolo"]["seed"])
    rng = random.Random(seed)
    np.random.seed(seed)

    input_root = Path(config["experiments"]["baseline"]["dataset_root"])
    output_root = Path(config["experiments"]["classic"]["dataset_root"])

    reports_dir = ensure_dir(config["paths"]["reports"])
    tables_dir = ensure_dir(config["paths"]["tables"])

    per_image = int(per_image or config["augmentation"]["classic_per_image"])

    if not input_root.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_root}")

    if output_root.exists():
        if overwrite:
            shutil.rmtree(output_root)
        else:
            raise FileExistsError(
                f"Output dataset already exists: {output_root}. "
                f"Use --overwrite to recreate it."
            )

    for split in ["train", "val", "test"]:
        ensure_dir(output_root / "images" / split)
        ensure_dir(output_root / "labels" / split)

    copied_train = copy_split(input_root, output_root, "train")
    copied_val = copy_split(input_root, output_root, "val")
    copied_test = copy_split(input_root, output_root, "test")

    input_train_images = sorted((input_root / "images" / "train").glob("*.jpg"))
    input_train_labels = input_root / "labels" / "train"

    augmentation_rows = []
    augmented_count = 0

    for image_path in input_train_images:
        label_path = input_train_labels / f"{image_path.stem}.txt"

        image = cv2.imread(str(image_path))
        labels = read_yolo_labels(label_path)

        if image is None or not labels:
            continue

        for variant_index in range(per_image):
            augmented_image, augmented_labels, transforms = make_classic_variant(
                image=image,
                labels=labels,
                rng=rng,
                variant_index=variant_index,
            )

            augmented_stem = f"aug{variant_index:02d}_{image_path.stem}"
            output_image_path = output_root / "images" / "train" / f"{augmented_stem}.jpg"
            output_label_path = output_root / "labels" / "train" / f"{augmented_stem}.txt"

            cv2.imwrite(str(output_image_path), augmented_image)
            write_yolo_labels(augmented_labels, output_label_path)

            augmentation_rows.append(
                {
                    "source_image": str(image_path),
                    "augmented_image": str(output_image_path),
                    "transforms": "+".join(transforms),
                    "num_boxes": len(augmented_labels),
                }
            )

            augmented_count += 1

    dataset_yaml = write_dataset_yaml(output_root)

    validation_report = validate_yolo_dataset(output_root)

    import pandas as pd

    table_path = tables_dir / "classic_augmentation_index.csv"
    pd.DataFrame(augmentation_rows).to_csv(table_path, index=False)

    report = {
        "type": "classic_augmentation",
        "input_root": str(input_root),
        "output_root": str(output_root),
        "dataset_yaml": str(dataset_yaml),
        "per_image": per_image,
        "copied_real_train": copied_train,
        "copied_val": copied_val,
        "copied_test": copied_test,
        "augmented_train": augmented_count,
        "total_train_images": copied_train + augmented_count,
        "validation": validation_report,
        "table": str(table_path),
    }

    save_json(report, reports_dir / "classic_augmentation_report.json")

    print("Classic augmentation completed.")
    print(f"Input dataset: {input_root}")
    print(f"Output dataset: {output_root}")
    print(f"Copied real train images: {copied_train}")
    print(f"Generated augmented train images: {augmented_count}")
    print(f"Total train images: {copied_train + augmented_count}")
    print(f"Validation valid: {validation_report['is_valid']}")

    return report
