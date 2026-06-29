from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from src.utils import ensure_dir, save_json, sha256_file


@dataclass
class PreparedSample:
    split: str
    image_path: Path
    label_path: Path
    image_id: str
    width: int
    height: int
    num_boxes: int
    image_sha256: str


def coco_box_to_yolo(
    box: list[float],
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float] | None:
    x, y, w, h = box

    if w <= 0 or h <= 0:
        return None

    x1 = max(0.0, float(x))
    y1 = max(0.0, float(y))
    x2 = min(float(image_width), float(x + w))
    y2 = min(float(image_height), float(y + h))

    clipped_w = x2 - x1
    clipped_h = y2 - y1

    if clipped_w <= 1 or clipped_h <= 1:
        return None

    x_center = ((x1 + x2) / 2.0) / image_width
    y_center = ((y1 + y2) / 2.0) / image_height
    width = clipped_w / image_width
    height = clipped_h / image_height

    values = (x_center, y_center, width, height)

    if not all(0.0 <= v <= 1.0 for v in values):
        return None

    return values


def extract_yolo_labels(
    sample: dict[str, Any],
    target_class_id: int,
) -> list[str]:
    image = sample["image"]
    image_width, image_height = image.size

    objects = sample.get("objects", {})
    boxes = objects.get("bbox", [])
    categories = objects.get("category", [])

    yolo_lines = []

    for box, category in zip(boxes, categories):
        if int(category) != int(target_class_id):
            continue

        yolo_box = coco_box_to_yolo(box, image_width, image_height)

        if yolo_box is None:
            continue

        x_center, y_center, width, height = yolo_box

        yolo_lines.append(
            f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        )

    return yolo_lines


def load_streaming_hf_dataset(hf_name: str, seed: int):
    """
    Load the HF dataset in streaming mode to avoid RAM spikes on local machines.

    We try the common 'train' split first. If that fails, we fall back to loading
    the dataset dict in streaming mode and use the first available split.
    """
    try:
        dataset = load_dataset(hf_name, split="train", streaming=True)
        return dataset.shuffle(seed=seed, buffer_size=1000)
    except Exception:
        dataset_dict = load_dataset(hf_name, streaming=True)
        first_split = list(dataset_dict.keys())[0]
        dataset = dataset_dict[first_split]
        return dataset.shuffle(seed=seed, buffer_size=1000)


def target_split_for_index(index: int, train_count: int, val_count: int) -> str:
    if index < train_count:
        return "train"

    if index < train_count + val_count:
        return "val"

    return "test"


def prepare_hf_yolo_dataset(config: dict[str, Any], overwrite: bool = False) -> dict[str, Any]:
    dataset_cfg = config["dataset"]
    paths_cfg = config["paths"]

    hf_name = dataset_cfg["hf_name"]
    class_id = int(dataset_cfg["class_id"])
    class_name = dataset_cfg["class_name"]

    output_root = Path(dataset_cfg["output_root"])
    reports_dir = ensure_dir(paths_cfg["reports"])
    tables_dir = ensure_dir(paths_cfg["tables"])

    train_count = int(dataset_cfg["split"]["train"])
    val_count = int(dataset_cfg["split"]["val"])
    test_count = int(dataset_cfg["split"]["test"])
    total_needed = train_count + val_count + test_count

    seed = int(config["yolo"]["seed"])

    if output_root.exists() and overwrite:
        shutil.rmtree(output_root)

    for split in ["train", "val", "test"]:
        ensure_dir(output_root / "images" / split)
        ensure_dir(output_root / "labels" / split)

    print(f"Loading Hugging Face dataset in streaming mode: {hf_name}")
    print(f"Target split: train={train_count}, val={val_count}, test={test_count}")

    dataset_stream = load_streaming_hf_dataset(hf_name=hf_name, seed=seed)

    prepared_rows: list[PreparedSample] = []
    rejected = {
        "no_valid_box": 0,
        "image_error": 0,
    }

    seen = 0
    accepted = 0

    progress = tqdm(total=total_needed, desc="Writing valid YOLO samples")

    for sample in dataset_stream:
        seen += 1

        if accepted >= total_needed:
            break

        try:
            image: Image.Image = sample["image"].convert("RGB")
            labels = extract_yolo_labels(sample, class_id)
        except Exception:
            rejected["image_error"] += 1
            continue

        if not labels:
            rejected["no_valid_box"] += 1
            continue

        split = target_split_for_index(
            index=accepted,
            train_count=train_count,
            val_count=val_count,
        )

        width, height = image.size
        image_id = str(sample.get("image_id", f"sample_{seen:08d}"))
        safe_stem = f"{split}_{accepted:06d}_{image_id}".replace("/", "_").replace(" ", "_")

        image_path = output_root / "images" / split / f"{safe_stem}.jpg"
        label_path = output_root / "labels" / split / f"{safe_stem}.txt"

        image.save(image_path, quality=95)
        label_path.write_text("\n".join(labels) + "\n", encoding="utf-8")

        prepared_rows.append(
            PreparedSample(
                split=split,
                image_path=image_path,
                label_path=label_path,
                image_id=image_id,
                width=width,
                height=height,
                num_boxes=len(labels),
                image_sha256=sha256_file(image_path),
            )
        )

        accepted += 1
        progress.update(1)

    progress.close()

    if accepted < total_needed:
        raise RuntimeError(
            f"Could only prepare {accepted} valid samples, but {total_needed} were requested."
        )

    dataset_yaml = {
        "path": str(output_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 1,
        "names": ["drone"],
    }

    with (output_root / "dataset.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(dataset_yaml, f, sort_keys=False)

    rows = [
        {
            "split": row.split,
            "image_path": str(row.image_path),
            "label_path": str(row.label_path),
            "image_id": row.image_id,
            "width": row.width,
            "height": row.height,
            "num_boxes": row.num_boxes,
            "image_sha256": row.image_sha256,
        }
        for row in prepared_rows
    ]

    df = pd.DataFrame(rows)
    df.to_csv(tables_dir / "dataset_index.csv", index=False)

    validation_report = validate_yolo_dataset(output_root)
    leakage_report = check_split_leakage(df)

    report = {
        "hf_dataset": hf_name,
        "loading_mode": "streaming",
        "class_name": class_name,
        "class_id": class_id,
        "output_root": str(output_root),
        "requested_split": {
            "train": train_count,
            "val": val_count,
            "test": test_count,
        },
        "written_split": df.groupby("split").size().to_dict(),
        "samples_seen_before_accepting_target": seen,
        "total_accepted_samples": accepted,
        "rejected": rejected,
        "total_boxes": int(df["num_boxes"].sum()),
        "validation": validation_report,
        "leakage": leakage_report,
        "dataset_yaml": str(output_root / "dataset.yaml"),
    }

    save_json(report, reports_dir / "dataset_preparation_report.json")
    save_json(leakage_report, reports_dir / "split_leakage_report.json")
    save_json(validation_report, reports_dir / "annotation_validation_report.json")

    print("\nDataset preparation completed.")
    print(f"Dataset root: {output_root}")
    print(f"Dataset YAML: {output_root / 'dataset.yaml'}")
    print(f"Report: {reports_dir / 'dataset_preparation_report.json'}")

    return report


def validate_yolo_dataset(dataset_root: str | Path) -> dict[str, Any]:
    dataset_root = Path(dataset_root)

    report: dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "splits": {},
        "errors": [],
    }

    all_box_widths = []
    all_box_heights = []

    for split in ["train", "val", "test"]:
        image_dir = dataset_root / "images" / split
        label_dir = dataset_root / "labels" / split

        image_paths = sorted(image_dir.glob("*.jpg"))
        label_paths = sorted(label_dir.glob("*.txt"))

        split_errors = []
        num_boxes = 0
        small_boxes = 0

        for image_path in image_paths:
            label_path = label_dir / f"{image_path.stem}.txt"

            if not label_path.exists():
                split_errors.append(f"Missing label for {image_path.name}")
                continue

            lines = [
                line.strip()
                for line in label_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

            for line in lines:
                parts = line.split()

                if len(parts) != 5:
                    split_errors.append(f"Invalid label line in {label_path.name}: {line}")
                    continue

                cls, x, y, w, h = parts
                values = [float(x), float(y), float(w), float(h)]

                if int(cls) != 0:
                    split_errors.append(f"Invalid class in {label_path.name}: {cls}")

                if not all(0.0 <= v <= 1.0 for v in values):
                    split_errors.append(f"Out-of-range box in {label_path.name}: {line}")

                box_w = float(w)
                box_h = float(h)

                all_box_widths.append(box_w)
                all_box_heights.append(box_h)

                num_boxes += 1

                if box_w < 0.05 or box_h < 0.05:
                    small_boxes += 1

        report["splits"][split] = {
            "num_images": len(image_paths),
            "num_labels": len(label_paths),
            "num_boxes": num_boxes,
            "num_small_boxes_relative_lt_0_05": small_boxes,
            "errors": split_errors,
        }

        report["errors"].extend([f"{split}: {error}" for error in split_errors])

    width_series = pd.Series(all_box_widths)
    height_series = pd.Series(all_box_heights)

    report["box_stats"] = {
        "mean_width": float(width_series.mean()) if not width_series.empty else None,
        "mean_height": float(height_series.mean()) if not height_series.empty else None,
        "median_width": float(width_series.median()) if not width_series.empty else None,
        "median_height": float(height_series.median()) if not height_series.empty else None,
    }

    report["is_valid"] = len(report["errors"]) == 0

    return report


def check_split_leakage(df: pd.DataFrame) -> dict[str, Any]:
    leakage = {
        "image_sha256_overlap": {},
        "image_id_overlap": {},
        "has_leakage": False,
    }

    split_names = ["train", "val", "test"]

    for i, split_a in enumerate(split_names):
        for split_b in split_names[i + 1:]:
            hashes_a = set(df[df["split"] == split_a]["image_sha256"])
            hashes_b = set(df[df["split"] == split_b]["image_sha256"])
            ids_a = set(df[df["split"] == split_a]["image_id"])
            ids_b = set(df[df["split"] == split_b]["image_id"])

            hash_overlap = sorted(hashes_a.intersection(hashes_b))
            id_overlap = sorted(ids_a.intersection(ids_b))

            pair = f"{split_a}_vs_{split_b}"

            leakage["image_sha256_overlap"][pair] = len(hash_overlap)
            leakage["image_id_overlap"][pair] = len(id_overlap)

            if hash_overlap or id_overlap:
                leakage["has_leakage"] = True

    return leakage
