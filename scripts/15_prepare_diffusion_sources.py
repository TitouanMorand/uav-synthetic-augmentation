from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.augment import read_yolo_labels
from src.config import load_config
from src.diffusion_v2 import (
    compute_inset_window_score,
    compute_internal_rectangle_score,
    compute_mask_coverage,
    compute_ui_rectangle_score,
    compute_vertical_seam_score,
    labels_box_stats,
)
from src.utils import ensure_dir


def draw_preview(
    image_path: Path,
    labels,
    output_path: Path,
    accepted: bool,
    reasons: list[str],
    source_index: int,
) -> None:
    ensure_dir(output_path.parent)

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    w_img, h_img = image.size

    for _, x, y, w, h in labels:
        x1 = int((x - w / 2) * w_img)
        y1 = int((y - h / 2) * h_img)
        x2 = int((x + w / 2) * w_img)
        y2 = int((y + h / 2) * h_img)
        draw.rectangle([x1, y1, x2, y2], outline="lime", width=3)

    banner_h = 54
    canvas = Image.new("RGB", (image.width, image.height + banner_h), (245, 245, 245))
    canvas.paste(image, (0, banner_h))

    status = "ACCEPTED" if accepted else "REJECTED"
    reason_text = "" if accepted else " | " + ", ".join(reasons[:2])
    text = f"#{source_index:04d} — {status}{reason_text}"

    draw_canvas = ImageDraw.Draw(canvas)
    draw_canvas.text((8, 10), text, fill=(0, 0, 0))

    canvas.thumbnail((460, 460))
    canvas.save(output_path, quality=95)


def make_contact_sheet(image_paths: list[Path], output_path: Path, cols: int = 5) -> None:
    if not image_paths:
        return

    ensure_dir(output_path.parent)

    thumbs = []

    for path in image_paths:
        img = Image.open(path).convert("RGB")
        img = img.resize((240, 240))
        thumbs.append(img)

    rows = int(np.ceil(len(thumbs) / cols))
    canvas = Image.new("RGB", (cols * 240, rows * 240), (20, 20, 20))

    for i, img in enumerate(thumbs):
        x = (i % cols) * 240
        y = (i // cols) * 240
        canvas.paste(img, (x, y))

    canvas.save(output_path, quality=95)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare clean source candidates for diffusion.")
    parser.add_argument("--max-preview", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = load_config()
    diffusion_cfg = yaml.safe_load(Path("configs/diffusion_v2.yaml").read_text())["diffusion_v2"]
    source_cfg = diffusion_cfg["source_filter"]

    blacklist_path = Path("configs/diffusion_source_blacklist.txt")
    blacklisted_indices = set()

    if blacklist_path.exists():
        for line in blacklist_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            blacklisted_indices.add(int(line))

    dataset_root = Path(config["experiments"]["baseline"]["dataset_root"])
    image_dir = dataset_root / "images" / "train"
    label_dir = dataset_root / "labels" / "train"

    tables_dir = ensure_dir(config["paths"]["tables"])
    previews_dir = ensure_dir(Path(config["paths"]["previews"]) / "diffusion_v2_sources")

    rows = []
    accepted_preview_paths = []
    rejected_preview_paths = []

    for source_index, image_path in enumerate(sorted(image_dir.glob("*.jpg"))):
        label_path = label_dir / f"{image_path.stem}.txt"
        labels = read_yolo_labels(label_path)

        if not labels:
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            continue

        h_img, w_img = image.shape[:2]

        mask_coverage = compute_mask_coverage(labels=labels, image_size=(w_img, h_img), margin_px=0)
        rect_score = compute_internal_rectangle_score(image)
        seam_score = compute_vertical_seam_score(image)
        ui_score = compute_ui_rectangle_score(image)
        inset_score = compute_inset_window_score(image)
        box_stats = labels_box_stats(labels, image_size=(w_img, h_img))

        reasons = []

        if mask_coverage > float(source_cfg["max_mask_coverage"]):
            reasons.append("mask_coverage_too_high")

        if rect_score > float(source_cfg["max_internal_rectangle_score"]):
            reasons.append("internal_rectangle_score_high")

        if seam_score > float(source_cfg["max_vertical_seam_score"]):
            reasons.append("vertical_seam_score_high")

        if ui_score > float(source_cfg["max_ui_rectangle_score"]):
            reasons.append("ui_rectangle_score_high")

        if inset_score > float(source_cfg["max_inset_window_score"]):
            reasons.append("inset_window_score_high")

        if box_stats["max_box_side_px"] < float(source_cfg["min_box_max_side_px"]):
            reasons.append("box_too_tiny_for_diffusion_source")

        if box_stats["max_box_side_px"] > float(source_cfg["max_box_max_side_px"]):
            reasons.append("box_too_large_for_diffusion_source")

        if box_stats["max_box_area_ratio"] > float(source_cfg["max_box_area_ratio"]):
            reasons.append("box_area_ratio_too_large")

        if source_index in blacklisted_indices:
            reasons.append("manual_blacklist")

        accepted = len(reasons) == 0

        preview_path = previews_dir / ("accepted" if accepted else "rejected") / f"{source_index:04d}_{image_path.name}"

        draw_preview(
            image_path=image_path,
            labels=labels,
            output_path=preview_path,
            accepted=accepted,
            reasons=reasons,
            source_index=source_index,
        )

        if accepted and len(accepted_preview_paths) < args.max_preview:
            accepted_preview_paths.append(preview_path)

        if not accepted and len(rejected_preview_paths) < args.max_preview:
            rejected_preview_paths.append(preview_path)

        rows.append(
            {
                "source_index": source_index,
                "accepted": accepted,
                "rejection_reasons": ";".join(reasons),
                "manual_blacklisted": source_index in blacklisted_indices,
                "image_path": str(image_path),
                "label_path": str(label_path),
                "width": w_img,
                "height": h_img,
                "num_boxes": len(labels),
                "mask_coverage": mask_coverage,
                "internal_rectangle_score": rect_score,
                "vertical_seam_score": seam_score,
                "ui_rectangle_score": ui_score,
                "inset_window_score": inset_score,
                **box_stats,
            }
        )

    df = pd.DataFrame(rows)

    candidates_path = tables_dir / "diffusion_v2_source_candidates.csv"
    accepted_path = tables_dir / "diffusion_v2_source_accepted.csv"
    rejected_path = tables_dir / "diffusion_v2_source_rejected.csv"

    df.to_csv(candidates_path, index=False)
    df[df["accepted"]].to_csv(accepted_path, index=False)
    df[~df["accepted"]].to_csv(rejected_path, index=False)

    make_contact_sheet(accepted_preview_paths, previews_dir / "accepted_contact_sheet.jpg")
    make_contact_sheet(rejected_preview_paths, previews_dir / "rejected_contact_sheet.jpg")

    print("Diffusion source filtering completed.")
    print(f"Total candidates: {len(df)}")
    print(f"Accepted: {int(df['accepted'].sum())}")
    print(f"Rejected: {int((~df['accepted']).sum())}")
    print(f"Candidates CSV: {candidates_path}")
    print(f"Accepted CSV: {accepted_path}")
    print(f"Rejected CSV: {rejected_path}")
    print(f"Preview dir: {previews_dir}")

    suspicious = df.sort_values("inset_window_score", ascending=False).head(20)
    print("\nTop 20 inset_window_score:")
    print(
        suspicious[
            [
                "source_index",
                "accepted",
                "inset_window_score",
                "ui_rectangle_score",
                "internal_rectangle_score",
                "vertical_seam_score",
                "image_path",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
