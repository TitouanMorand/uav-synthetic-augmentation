"""
Generate a classical night/low-light YOLO training set.

The augmentation is photometric only: it changes brightness, contrast, color
temperature, and adds a vignette/noise effect without moving pixels. Because the
image geometry is unchanged, YOLO labels can be copied as-is.
"""
import argparse
import math
import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

from src.utils import yolo_to_pixel_coords


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def list_images(image_dir: Path) -> list[Path]:
    return sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def copy_split(src_root: Path, dst_root: Path, split: str) -> None:
    src_img_dir = src_root / "images" / split
    src_lbl_dir = src_root / "labels" / split
    dst_img_dir = dst_root / "images" / split
    dst_lbl_dir = dst_root / "labels" / split
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    for image_path in list_images(src_img_dir):
        shutil.copy2(image_path, dst_img_dir / image_path.name)
        label_path = src_lbl_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            shutil.copy2(label_path, dst_lbl_dir / label_path.name)


def apply_night_effect(
    image: np.ndarray,
    darkness: float = 0.38,
    contrast: float = 0.9,
    blue_gain: float = 1.18,
    noise_std: float = 4.0,
    vignette_strength: float = 0.45,
) -> np.ndarray:
    """Apply a simple low-light effect while preserving image geometry."""
    if not 0.0 < darkness <= 1.0:
        raise ValueError("darkness must be in (0, 1]")

    img = image.astype(np.float32)

    # Lower scene illumination in HSV space, which approximates reducing the
    # value channel while keeping object edges and box geometry unchanged.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 2] *= darkness
    low_light = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    img = low_light.astype(np.float32)

    # Shift the color balance toward blue/cyan, a common night-camera artifact.
    img[..., 0] *= blue_gain
    img[..., 1] *= 1.02
    img[..., 2] *= 0.82

    # Mild contrast compression simulates low-light loss of detail.
    img = (img - 127.5) * contrast + 127.5

    if vignette_strength > 0:
        h, w = img.shape[:2]
        y = np.linspace(-1.0, 1.0, h, dtype=np.float32)
        x = np.linspace(-1.0, 1.0, w, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)
        radius = np.sqrt(xx**2 + yy**2)
        vignette = 1.0 - vignette_strength * np.clip(radius, 0.0, 1.0) ** 2
        img *= vignette[..., None]

    if noise_std > 0:
        noise = np.random.normal(0.0, noise_std, img.shape).astype(np.float32)
        img += noise

    return np.clip(img, 0, 255).astype(np.uint8)


def draw_boxes(image: np.ndarray, label_path: Path, color: tuple[int, int, int]) -> np.ndarray:
    out = image.copy()
    h, w = out.shape[:2]
    if not label_path.exists():
        return out

    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls = parts[0]
        xc, yc, bw, bh = map(float, parts[1:5])
        x1, y1, x2, y2 = yolo_to_pixel_coords(xc, yc, bw, bh, w, h)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            out,
            cls,
            (max(0, x1), max(14, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return out


def make_contact_sheets(
    pairs: list[tuple[Path, Path, Path]],
    preview_dir: Path,
    max_images: int,
    seed: int,
    thumb_width: int = 360,
    sheet_cols: int = 2,
) -> None:
    preview_dir.mkdir(parents=True, exist_ok=True)
    if not pairs:
        return

    rng = random.Random(seed)
    selected = pairs.copy()
    rng.shuffle(selected)
    selected = selected[:max_images]

    tiles = []
    for original_path, augmented_path, label_path in selected:
        original = cv2.imread(str(original_path))
        augmented = cv2.imread(str(augmented_path))
        if original is None or augmented is None:
            continue

        original = draw_boxes(original, label_path, (0, 255, 0))
        augmented = draw_boxes(augmented, label_path, (0, 255, 255))

        h, w = original.shape[:2]
        scale = thumb_width / max(w, 1)
        thumb_size = (thumb_width, max(1, int(h * scale)))
        original = cv2.resize(original, thumb_size, interpolation=cv2.INTER_AREA)
        augmented = cv2.resize(augmented, thumb_size, interpolation=cv2.INTER_AREA)

        label_bar = np.zeros((30, thumb_width * 2, 3), dtype=np.uint8)
        cv2.putText(label_bar, "original", (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
        cv2.putText(
            label_bar,
            "night augmented",
            (thumb_width + 10, 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            1,
        )
        tiles.append(np.vstack([label_bar, np.hstack([original, augmented])]))

    if not tiles:
        return

    rows_per_sheet = max(1, math.ceil(len(tiles) / sheet_cols))
    for sheet_idx in range(math.ceil(len(tiles) / (sheet_cols * rows_per_sheet))):
        chunk = tiles[sheet_idx * sheet_cols * rows_per_sheet : (sheet_idx + 1) * sheet_cols * rows_per_sheet]
        tile_h = max(tile.shape[0] for tile in chunk)
        tile_w = max(tile.shape[1] for tile in chunk)
        padded = []
        for tile in chunk:
            canvas = np.full((tile_h, tile_w, 3), 24, dtype=np.uint8)
            canvas[: tile.shape[0], : tile.shape[1]] = tile
            padded.append(canvas)

        rows = []
        for i in range(0, len(padded), sheet_cols):
            row_tiles = padded[i : i + sheet_cols]
            while len(row_tiles) < sheet_cols:
                row_tiles.append(np.full((tile_h, tile_w, 3), 24, dtype=np.uint8))
            rows.append(np.hstack(row_tiles))
        sheet = np.vstack(rows)
        cv2.imwrite(str(preview_dir / f"night_contact_sheet_{sheet_idx:02d}.jpg"), sheet)


def write_dataset_yaml(out_root: Path, val_images_dir: Path) -> None:
    val_path = os.path.relpath(val_images_dir, out_root)
    yaml_text = "\n".join(
        [
            "names:",
            "- drone",
            "nc: 1",
            f"path: {out_root.as_posix()}",
            "train: images/train",
            f"val: {Path(val_path).as_posix()}",
            "",
        ]
    )
    (out_root / "dataset.yaml").write_text(yaml_text)


def generate_night_dataset(
    input_root: Path,
    output_root: Path,
    darkness: float,
    seed: int,
    preview_count: int,
    overwrite: bool,
) -> None:
    train_img_dir = input_root / "images" / "train"
    train_lbl_dir = input_root / "labels" / "train"
    val_img_dir = input_root / "images" / "val"
    val_lbl_dir = input_root / "labels" / "val"
    for required in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        if not required.exists():
            raise FileNotFoundError(f"Required YOLO directory not found: {required}")

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_root}. Use --overwrite to replace it.")
        shutil.rmtree(output_root)

    np.random.seed(seed)
    copy_split(input_root, output_root, "train")
    copy_split(input_root, output_root, "val")

    out_train_img_dir = output_root / "images" / "train"
    out_train_lbl_dir = output_root / "labels" / "train"
    preview_pairs = []

    for image_path in list_images(train_img_dir):
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        augmented = apply_night_effect(image, darkness=darkness)
        aug_name = f"{image_path.stem}_night{image_path.suffix.lower()}"
        aug_path = out_train_img_dir / aug_name
        cv2.imwrite(str(aug_path), augmented)

        label_path = train_lbl_dir / f"{image_path.stem}.txt"
        aug_label_path = out_train_lbl_dir / f"{Path(aug_name).stem}.txt"
        if label_path.exists():
            shutil.copy2(label_path, aug_label_path)
            preview_pairs.append((image_path, aug_path, label_path))

    write_dataset_yaml(output_root, input_root / "images" / "val")
    make_contact_sheets(preview_pairs, output_root / "previews", preview_count, seed)

    print(f"Wrote night dataset to {output_root}")
    print(f"Train images: {len(list_images(out_train_img_dir))} (real + night augmented)")
    print(f"Val images: {len(list_images(output_root / 'images' / 'val'))} (real only)")
    print(f"Dataset YAML: {output_root / 'dataset.yaml'}")
    print(f"Preview contact sheets: {output_root / 'previews'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a real + classical-night YOLO training set.")
    parser.add_argument("--input", default="data/yolo", help="Input YOLO dataset root")
    parser.add_argument("--output", default="data/yolo_aug_night", help="Output YOLO dataset root")
    parser.add_argument("--darkness", type=float, default=0.38, help="Brightness multiplier for night images")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for noise and preview sampling")
    parser.add_argument("--preview-count", type=int, default=20, help="Number of original/night pairs in previews")
    parser.add_argument("--overwrite", action="store_true", help="Replace output directory if it already exists")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_night_dataset(
        input_root=Path(args.input),
        output_root=Path(args.output),
        darkness=args.darkness,
        seed=args.seed,
        preview_count=args.preview_count,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
