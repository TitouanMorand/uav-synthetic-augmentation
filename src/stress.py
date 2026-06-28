from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from src.dataset import validate_yolo_dataset
from src.utils import ensure_dir, save_json


STRESS_MODES = ["night", "haze", "low_contrast"]


def apply_night(image: np.ndarray) -> np.ndarray:
    img = image.astype(np.float32)

    # BGR channels in OpenCV.
    img[:, :, 0] = img[:, :, 0] * 0.55 + 20   # blue
    img[:, :, 1] = img[:, :, 1] * 0.22        # green
    img[:, :, 2] = img[:, :, 2] * 0.14        # red

    height, width = img.shape[:2]

    yy, xx = np.mgrid[0:height, 0:width]
    cx, cy = width / 2, height / 2
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    dist = dist / max(dist.max(), 1.0)
    vignette = 1.0 - 0.40 * dist

    img = img * vignette[:, :, None]

    noise = np.random.normal(0, 4, size=img.shape)
    img = img + noise

    return np.clip(img, 0, 255).astype(np.uint8)


def apply_haze(image: np.ndarray) -> np.ndarray:
    img = image.astype(np.float32)
    haze = np.full_like(img, 220, dtype=np.float32)

    img = img * 0.58 + haze * 0.42
    img = cv2.GaussianBlur(img.astype(np.uint8), (5, 5), 0)

    return np.clip(img, 0, 255).astype(np.uint8)


def apply_low_contrast(image: np.ndarray) -> np.ndarray:
    img = image.astype(np.float32)

    mean = img.mean(axis=(0, 1), keepdims=True)
    img = mean + 0.45 * (img - mean)
    img = img * 0.85

    return np.clip(img, 0, 255).astype(np.uint8)


def apply_stress(image: np.ndarray, mode: str) -> np.ndarray:
    if mode == "night":
        return apply_night(image)

    if mode == "haze":
        return apply_haze(image)

    if mode == "low_contrast":
        return apply_low_contrast(image)

    raise ValueError(f"Unknown stress mode: {mode}")


def write_dataset_yaml(output_root: Path) -> Path:
    dataset_yaml = {
        "path": str(output_root.resolve()),
        "train": "images/val",
        "val": "images/val",
        "test": "images/test",
        "nc": 1,
        "names": ["drone"],
    }

    path = output_root / "dataset.yaml"

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(dataset_yaml, f, sort_keys=False)

    return path


def make_one_stress_dataset(
    input_root: Path,
    output_root: Path,
    mode: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    if output_root.exists():
        if overwrite:
            shutil.rmtree(output_root)
        else:
            raise FileExistsError(f"Output dataset exists: {output_root}. Use overwrite.")

    for split in ["val", "test"]:
        ensure_dir(output_root / "images" / split)
        ensure_dir(output_root / "labels" / split)

    rows = []

    for split in ["val", "test"]:
        input_image_dir = input_root / "images" / split
        input_label_dir = input_root / "labels" / split

        output_image_dir = output_root / "images" / split
        output_label_dir = output_root / "labels" / split

        for image_path in sorted(input_image_dir.glob("*.jpg")):
            label_path = input_label_dir / f"{image_path.stem}.txt"

            if not label_path.exists():
                continue

            image = cv2.imread(str(image_path))

            if image is None:
                continue

            stressed = apply_stress(image, mode=mode)

            output_image_path = output_image_dir / image_path.name
            output_label_path = output_label_dir / label_path.name

            cv2.imwrite(str(output_image_path), stressed)
            shutil.copy2(label_path, output_label_path)

            rows.append(
                {
                    "mode": mode,
                    "split": split,
                    "source_image": str(image_path),
                    "stress_image": str(output_image_path),
                    "label": str(output_label_path),
                }
            )

    dataset_yaml = write_dataset_yaml(output_root)
    validation_report = validate_yolo_dataset(output_root)

    report = {
        "mode": mode,
        "input_root": str(input_root),
        "output_root": str(output_root),
        "dataset_yaml": str(dataset_yaml),
        "num_images": len(rows),
        "validation": validation_report,
    }

    return report


def make_stress_datasets(
    config: dict[str, Any],
    overwrite: bool = False,
) -> dict[str, Any]:
    input_root = Path(config["experiments"]["baseline"]["dataset_root"])
    reports_dir = ensure_dir(config["paths"]["reports"])

    stress_root = Path("data/stress")
    ensure_dir(stress_root)

    reports = {}

    for mode in STRESS_MODES:
        output_root = stress_root / f"yolo_drone_hf_300_{mode}"

        report = make_one_stress_dataset(
            input_root=input_root,
            output_root=output_root,
            mode=mode,
            overwrite=overwrite,
        )

        reports[mode] = report

    save_json(reports, reports_dir / "stress_datasets_report.json")

    return reports
