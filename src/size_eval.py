from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

from src.utils import ensure_dir, resolve_device, save_json


SIZE_BUCKETS = {
    "very_tiny": (0, 16),
    "tiny": (16, 32),
    "small": (32, 64),
    "medium_plus": (64, float("inf")),
}


def read_yolo_labels(label_path: Path, image_width: int, image_height: int) -> list[dict[str, Any]]:
    labels = []

    if not label_path.exists():
        return labels

    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue

        cls, x, y, w, h = line.split()

        x = float(x)
        y = float(y)
        w = float(w)
        h = float(h)

        x1 = (x - w / 2) * image_width
        y1 = (y - h / 2) * image_height
        x2 = (x + w / 2) * image_width
        y2 = (y + h / 2) * image_height

        max_side = max(w * image_width, h * image_height)
        bucket = size_bucket(max_side)

        labels.append(
            {
                "cls": int(cls),
                "xyxy": np.array([x1, y1, x2, y2], dtype=np.float32),
                "max_side_px": float(max_side),
                "bucket": bucket,
                "matched": False,
            }
        )

    return labels


def size_bucket(max_side_px: float) -> str:
    for name, (low, high) in SIZE_BUCKETS.items():
        if low <= max_side_px < high:
            return name

    return "medium_plus"


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h

    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])

    union = area_a + area_b - inter

    if union <= 0:
        return 0.0

    return float(inter / union)


def average_precision(tp: np.ndarray, fp: np.ndarray, num_gt: int) -> float:
    if num_gt == 0:
        return float("nan")

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)

    recall = tp_cum / max(num_gt, 1)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)

    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    indices = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1])

    return float(ap)


def evaluate_size_stratified(
    config: dict[str, Any],
    run_name: str,
    weights_path: str | Path,
    dataset_root: str | Path,
    dataset_name: str,
    split: str,
    conf_for_fp: float = 0.25,
    pred_conf: float = 0.001,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    dataset_root = Path(dataset_root)
    weights_path = Path(weights_path)

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    image_dir = dataset_root / "images" / split
    label_dir = dataset_root / "labels" / split

    image_paths = sorted(image_dir.glob("*.jpg"))

    device = resolve_device(config["yolo"]["device"])
    imgsz = int(config["yolo"]["image_size"])

    model = YOLO(str(weights_path))

    gt_by_bucket = {bucket: 0 for bucket in SIZE_BUCKETS}
    pred_rows_by_bucket = {bucket: [] for bucket in SIZE_BUCKETS}

    total_fp_at_conf = 0
    total_images = len(image_paths)

    for image_path in tqdm(image_paths, desc=f"Size eval {run_name} {dataset_name}/{split}"):
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        label_path = label_dir / f"{image_path.stem}.txt"
        gts = read_yolo_labels(label_path, width, height)

        for gt in gts:
            gt_by_bucket[gt["bucket"]] += 1

        results = model.predict(
            source=str(image_path),
            imgsz=imgsz,
            conf=pred_conf,
            iou=0.7,
            device=device,
            verbose=False,
        )[0]

        predictions = []

        if results.boxes is not None and len(results.boxes) > 0:
            xyxy = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()

            for box, conf in zip(xyxy, confs):
                predictions.append(
                    {
                        "xyxy": box.astype(np.float32),
                        "conf": float(conf),
                    }
                )

        predictions = sorted(predictions, key=lambda x: x["conf"], reverse=True)

        matched_gt_for_fp = set()

        for pred_idx, pred in enumerate(predictions):
            best_iou = 0.0
            best_gt_idx = None

            for gt_idx, gt in enumerate(gts):
                iou = box_iou(pred["xyxy"], gt["xyxy"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if pred["conf"] >= conf_for_fp:
                if best_iou < iou_threshold:
                    total_fp_at_conf += 1
                elif best_gt_idx is not None:
                    matched_gt_for_fp.add(best_gt_idx)

            for bucket in SIZE_BUCKETS:
                bucket_gts = [
                    (idx, gt)
                    for idx, gt in enumerate(gts)
                    if gt["bucket"] == bucket
                ]

                best_bucket_iou = 0.0
                best_bucket_gt_idx = None

                for gt_idx, gt in bucket_gts:
                    if gt.get("matched_for_ap", {}).get(bucket, False):
                        continue

                    iou = box_iou(pred["xyxy"], gt["xyxy"])
                    if iou > best_bucket_iou:
                        best_bucket_iou = iou
                        best_bucket_gt_idx = gt_idx

                is_tp = False

                if best_bucket_iou >= iou_threshold and best_bucket_gt_idx is not None:
                    gts[best_bucket_gt_idx].setdefault("matched_for_ap", {})[bucket] = True
                    is_tp = True

                pred_rows_by_bucket[bucket].append(
                    {
                        "conf": pred["conf"],
                        "tp": 1 if is_tp else 0,
                        "fp": 0 if is_tp else 1,
                    }
                )

    bucket_rows = []

    for bucket in SIZE_BUCKETS:
        rows = sorted(
            pred_rows_by_bucket[bucket],
            key=lambda x: x["conf"],
            reverse=True,
        )

        num_gt = gt_by_bucket[bucket]

        if rows:
            tp = np.array([row["tp"] for row in rows], dtype=np.float32)
            fp = np.array([row["fp"] for row in rows], dtype=np.float32)
            ap50 = average_precision(tp, fp, num_gt)

            recall50 = float(tp.sum() / max(num_gt, 1)) if num_gt > 0 else float("nan")
        else:
            ap50 = float("nan") if num_gt == 0 else 0.0
            recall50 = float("nan") if num_gt == 0 else 0.0

        bucket_rows.append(
            {
                "run_name": run_name,
                "dataset": dataset_name,
                "split": split,
                "bucket": bucket,
                "num_gt": int(num_gt),
                "ap50": ap50,
                "recall50": recall50,
            }
        )

    report = {
        "run_name": run_name,
        "dataset": dataset_name,
        "split": split,
        "weights_path": str(weights_path),
        "dataset_root": str(dataset_root),
        "num_images": total_images,
        "fp_per_image_conf_0_25": float(total_fp_at_conf / max(total_images, 1)),
        "buckets": bucket_rows,
    }

    return report


def save_size_eval_outputs(
    config: dict[str, Any],
    reports: list[dict[str, Any]],
) -> None:
    reports_dir = ensure_dir(config["paths"]["reports"])
    tables_dir = ensure_dir(config["paths"]["tables"])

    save_json(
        {"reports": reports},
        reports_dir / "size_stratified_eval_report.json",
    )

    bucket_rows = []
    fp_rows = []

    for report in reports:
        bucket_rows.extend(report["buckets"])
        fp_rows.append(
            {
                "run_name": report["run_name"],
                "dataset": report["dataset"],
                "split": report["split"],
                "num_images": report["num_images"],
                "fp_per_image_conf_0_25": report["fp_per_image_conf_0_25"],
            }
        )

    pd.DataFrame(bucket_rows).to_csv(
        tables_dir / "size_stratified_metrics.csv",
        index=False,
    )

    pd.DataFrame(fp_rows).to_csv(
        tables_dir / "fp_per_image_metrics.csv",
        index=False,
    )
