"""Create contact sheets for diffusion augmentation results."""

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.augmentation.masks_from_boxes import (
    DEFAULT_BOX_MARGIN_PX,
    DEFAULT_BOX_MARGIN_RATIO,
    background_inpaint_mask_from_labels,
    object_mask_from_labels,
    overlay_mask_on_image,
)
from src.utils import yolo_to_pixel_coords


def read_manifest(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_rgb(path: Path) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(f"Preview image not found: {path}")
    return Image.open(path).convert("RGB")


def draw_yolo_boxes(image: Image.Image, label_path: Path) -> Image.Image:
    image = image.copy()
    arr = np.array(image.convert("RGB"))[:, :, ::-1].copy()
    h, w = arr.shape[:2]
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = parts[0]
            xc, yc, bw, bh = map(float, parts[1:5])
            x1, y1, x2, y2 = yolo_to_pixel_coords(xc, yc, bw, bh, w, h)
            cv2.rectangle(arr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(arr, cls, (max(0, x1), max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return Image.fromarray(arr[:, :, ::-1])


def mask_to_rgb(mask: Image.Image) -> Image.Image:
    return mask.convert("L").convert("RGB")


def resize_tile(image: Image.Image, width: int) -> Image.Image:
    image = image.convert("RGB")
    scale = width / max(1, image.width)
    height = max(1, int(image.height * scale))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def labeled_tile(image: Image.Image, label: str, width: int) -> np.ndarray:
    tile = resize_tile(image, width)
    arr = np.array(tile)
    bar = np.full((28, width, 3), 24, dtype=np.uint8)
    cv2.putText(bar, label, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (235, 235, 235), 1, cv2.LINE_AA)
    return np.vstack([bar, arr])


def make_contact_sheet(rows: list[dict], out_path: Path, tile_width: int, mask_margin: int, relative_margin: float) -> None:
    rendered_rows = []
    for row in rows:
        src_image_path = Path(row["source_image_path"])
        src_label_path = Path(row["source_label_path"])
        out_image_path = Path(row["output_image_path"])
        out_label_path = Path(row["output_label_path"])

        original = load_rgb(src_image_path)
        generated = load_rgb(out_image_path)
        mask = background_inpaint_mask_from_labels(src_label_path, original.size, mask_margin, relative_margin)
        mask_overlay = overlay_mask_on_image(original, mask)
        generated_boxes = draw_yolo_boxes(generated, out_label_path)

        tiles = [
            labeled_tile(original, "original", tile_width),
            labeled_tile(mask_overlay, "mask overlay", tile_width),
            labeled_tile(generated, row["generation_mode"], tile_width),
            labeled_tile(generated_boxes, "generated + boxes", tile_width),
        ]
        row_h = max(tile.shape[0] for tile in tiles)
        padded = []
        for tile in tiles:
            canvas = np.full((row_h, tile_width, 3), 24, dtype=np.uint8)
            canvas[: tile.shape[0], : tile.shape[1]] = tile
            padded.append(canvas)
        rendered_rows.append(np.hstack(padded))

    if not rendered_rows:
        raise RuntimeError("No rows selected for preview.")

    sheet = np.vstack(rendered_rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(sheet).save(out_path, quality=95)


def make_object_preserving_sheet(
    rows: list[dict],
    out_path: Path,
    tile_width: int,
    mask_margin: int,
    relative_margin: float,
) -> None:
    rendered_rows = []
    for row in rows:
        src_image_path = Path(row["source_image_path"])
        src_label_path = Path(row["source_label_path"])
        out_image_path = Path(row["output_image_path"])
        out_label_path = Path(row["output_label_path"])
        if not out_label_path.exists():
            out_label_path = src_label_path

        original = load_rgb(src_image_path)
        original_boxes = draw_yolo_boxes(original, src_label_path)
        generated_after = load_rgb(out_image_path)
        generated_before_path = Path(row.get("generated_before_reinsertion_path", row["output_image_path"]))
        generated_before = load_rgb(generated_before_path)

        protection_path = row.get("protection_mask_path")
        inpaint_path = row.get("inpaint_mask_path")
        if protection_path and Path(protection_path).exists():
            protection_mask = Image.open(protection_path).convert("L")
        else:
            protection_mask = object_mask_from_labels(src_label_path, original.size, mask_margin, relative_margin)
        if inpaint_path and Path(inpaint_path).exists():
            inpaint_mask = Image.open(inpaint_path).convert("L")
        else:
            inpaint_mask = background_inpaint_mask_from_labels(src_label_path, original.size, mask_margin, relative_margin)

        generated_boxes = draw_yolo_boxes(generated_after, out_label_path)
        status = "accepted" if row.get("accepted_for_training", False) else "rejected"
        reason_text = ",".join(row.get("rejection_reasons", []))[:28]

        tiles = [
            labeled_tile(original, "original", tile_width),
            labeled_tile(original_boxes, "original + box", tile_width),
            labeled_tile(mask_to_rgb(protection_mask), "protection mask", tile_width),
            labeled_tile(mask_to_rgb(inpaint_mask), "inpaint mask", tile_width),
            labeled_tile(generated_before, "before reinsertion", tile_width),
            labeled_tile(generated_after, f"after reinsertion {status}", tile_width),
            labeled_tile(generated_boxes, f"after + box {reason_text}", tile_width),
        ]
        row_h = max(tile.shape[0] for tile in tiles)
        padded = []
        for tile in tiles:
            canvas = np.full((row_h, tile_width, 3), 24, dtype=np.uint8)
            canvas[: tile.shape[0], : tile.shape[1]] = tile
            padded.append(canvas)
        rendered_rows.append(np.hstack(padded))

    if not rendered_rows:
        raise RuntimeError("No rows selected for object-preserving preview.")

    sheet = np.vstack(rendered_rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(sheet).save(out_path, quality=95)


def parse_args():
    parser = argparse.ArgumentParser(description="Preview diffusion grid outputs from a manifest JSONL.")
    parser.add_argument("--manifest", default="data/synthetic/diffusion_grid/manifest.jsonl")
    parser.add_argument("--out-dir", default="data/previews/diffusion")
    parser.add_argument("--num", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tile-width", type=int, default=260)
    parser.add_argument("--mask-margin", type=int, default=DEFAULT_BOX_MARGIN_PX)
    parser.add_argument("--relative-margin", type=float, default=DEFAULT_BOX_MARGIN_RATIO)
    parser.add_argument("--strict-object-preserving", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = read_manifest(Path(args.manifest))
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    selected = rows[: args.num]
    if args.strict_object_preserving:
        out_path = Path(args.out_dir) / "object_preserving_contact_sheet.jpg"
        make_object_preserving_sheet(selected, out_path, args.tile_width, args.mask_margin, args.relative_margin)
    else:
        out_path = Path(args.out_dir) / "diffusion_contact_sheet.jpg"
        make_contact_sheet(selected, out_path, args.tile_width, args.mask_margin, args.relative_margin)
    print(f"Preview written: {out_path}")


if __name__ == "__main__":
    main()
