from __future__ import annotations

import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import ensure_dir


def read_yolo_labels(label_path: str | Path) -> list[tuple[int, float, float, float, float]]:
    label_path = Path(label_path)

    if not label_path.exists():
        return []

    labels = []

    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue

        cls, x, y, w, h = line.split()
        labels.append((int(cls), float(x), float(y), float(w), float(h)))

    return labels


def draw_yolo_boxes(image, labels):
    h_img, w_img = image.shape[:2]

    for cls, x, y, w, h in labels:
        x1 = int((x - w / 2) * w_img)
        y1 = int((y - h / 2) * h_img)
        x2 = int((x + w / 2) * w_img)
        y2 = int((y + h / 2) * h_img)

        x1 = max(0, min(w_img - 1, x1))
        y1 = max(0, min(h_img - 1, y1))
        x2 = max(0, min(w_img - 1, x2))
        y2 = max(0, min(h_img - 1, y2))

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            "drone",
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    return image


def make_contact_sheet(
    dataset_root: str | Path,
    split: str,
    output_path: str | Path,
    max_images: int = 24,
    thumb_size: int = 320,
) -> Path:
    dataset_root = Path(dataset_root)
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    image_paths = sorted((dataset_root / "images" / split).glob("*.jpg"))[:max_images]

    if not image_paths:
        raise FileNotFoundError(f"No images found for split={split}")

    thumbs = []

    for image_path in image_paths:
        label_path = dataset_root / "labels" / split / f"{image_path.stem}.txt"

        image = cv2.imread(str(image_path))
        if image is None:
            continue

        labels = read_yolo_labels(label_path)
        image = draw_yolo_boxes(image, labels)
        image = cv2.resize(image, (thumb_size, thumb_size))
        thumbs.append(image)

    cols = 4
    rows = math.ceil(len(thumbs) / cols)

    canvas = 255 * np.ones((rows * thumb_size, cols * thumb_size, 3), dtype="uint8")

    for i, thumb in enumerate(thumbs):
        row = i // cols
        col = i % cols
        y1 = row * thumb_size
        x1 = col * thumb_size
        canvas[y1:y1 + thumb_size, x1:x1 + thumb_size] = thumb

    cv2.imwrite(str(output_path), canvas)

    return output_path


def make_box_distribution_plot(
    dataset_root: str | Path,
    output_path: str | Path,
) -> Path:
    dataset_root = Path(dataset_root)
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    rows = []

    for split in ["train", "val", "test"]:
        label_dir = dataset_root / "labels" / split

        for label_path in sorted(label_dir.glob("*.txt")):
            labels = read_yolo_labels(label_path)

            for _, _, _, w, h in labels:
                rows.append(
                    {
                        "split": split,
                        "box_width_relative": w,
                        "box_height_relative": h,
                        "box_area_relative": w * h,
                    }
                )

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError("No boxes found for distribution plot.")

    plt.figure(figsize=(8, 5))
    plt.scatter(
        df["box_width_relative"],
        df["box_height_relative"],
        s=10,
        alpha=0.5,
    )
    plt.xlabel("Relative box width")
    plt.ylabel("Relative box height")
    plt.title("YOLO box size distribution")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()

    return output_path
