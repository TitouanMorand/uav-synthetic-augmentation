from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

from src.diffusion import (
    DiffusionBackend,
    compute_diff_stats,
    image_pixel_stats,
    is_suspicious_black_image,
)
from src.utils import ensure_dir


def labels_box_stats(labels, image_size: tuple[int, int]) -> dict[str, float]:
    width, height = image_size
    max_sides = []
    areas = []

    for _, x, y, w, h in labels:
        box_w = w * width
        box_h = h * height
        max_sides.append(max(box_w, box_h))
        areas.append(box_w * box_h)

    return {
        "max_box_side_px": float(max(max_sides)) if max_sides else 0.0,
        "mean_box_side_px": float(np.mean(max_sides)) if max_sides else 0.0,
        "max_box_area_px": float(max(areas)) if areas else 0.0,
    }


def make_box_mask(
    labels,
    image_size: tuple[int, int],
    margin_px: int,
    relative_margin: float = 1.0,
    blur_px: int = 0,
) -> Image.Image:
    width, height = image_size
    mask = np.zeros((height, width), dtype=np.uint8)

    for _, x, y, w, h in labels:
        box_w = w * width
        box_h = h * height

        margin_x = int(round(max(margin_px, box_w * relative_margin)))
        margin_y = int(round(max(margin_px, box_h * relative_margin)))

        x1 = int((x - w / 2) * width) - margin_x
        y1 = int((y - h / 2) * height) - margin_y
        x2 = int((x + w / 2) * width) + margin_x
        y2 = int((y + h / 2) * height) + margin_y

        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width, x2))
        y2 = max(0, min(height, y2))

        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 255

    pil = Image.fromarray(mask, mode="L")

    if blur_px > 0:
        pil = pil.filter(ImageFilter.GaussianBlur(radius=blur_px))

    return pil


def compute_mask_coverage(labels, image_size: tuple[int, int], margin_px: int = 0) -> float:
    mask = make_box_mask(labels, image_size=image_size, margin_px=int(margin_px), relative_margin=0.0)
    arr = np.array(mask.convert("L")) > 0
    return float(arr.mean())


def compute_internal_rectangle_score(image_bgr: np.ndarray) -> float:
    """
    Heuristic for picture-in-picture / screen / UI rectangles.

    Returns roughly the largest internal rectangular contour area ratio.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = gray.shape[:2]
    image_area = h * w
    best_score = 0.0

    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        area_ratio = (bw * bh) / max(image_area, 1)

        if area_ratio < 0.04 or area_ratio > 0.75:
            continue

        # Ignore rectangles touching image border.
        if x < 5 or y < 5 or x + bw > w - 5 or y + bh > h - 5:
            continue

        rect_area = bw * bh
        contour_area = cv2.contourArea(contour)
        rectangularity = contour_area / max(rect_area, 1)

        score = area_ratio * min(1.0, rectangularity * 2.0)
        best_score = max(best_score, float(score))

    return best_score


def compute_vertical_seam_score(image_bgr: np.ndarray) -> float:
    """
    Detect abrupt vertical seams that often indicate split-screen or inset artifacts.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    col_diff = np.abs(gray[:, 1:] - gray[:, :-1]).mean(axis=0)

    if len(col_diff) == 0:
        return 0.0

    h, w = gray.shape[:2]

    # Focus on internal seams, ignore image borders.
    start = int(w * 0.15)
    end = int(w * 0.85)

    internal = col_diff[start:end]

    if len(internal) == 0:
        return 0.0

    return float(np.percentile(internal, 99))


def make_inpaint_mask_from_protection(protection_mask: Image.Image) -> Image.Image:
    """
    Diffusers convention:
    white = repaint
    black = preserve
    """
    return ImageOps.invert(protection_mask.convert("L"))


def make_night_condition(
    original: Image.Image,
    protection_mask: Image.Image,
    condition_strength: float,
    variant_index: int,
) -> Image.Image:
    """
    Create a night-like conditioning image.

    The protected region is preserved. The editable background becomes dark blue,
    low-light, noisy and slightly vignetted. This pushes inpainting toward night
    without asking the model to invent an entirely unrelated scene.
    """
    original = original.convert("RGB")
    arr = np.array(original).astype(np.float32)

    alpha = np.array(protection_mask.convert("L")).astype(np.float32) / 255.0
    alpha = alpha[:, :, None]

    night = arr.copy()

    # RGB night transform.
    night[:, :, 0] = night[:, :, 0] * 0.13
    night[:, :, 1] = night[:, :, 1] * 0.20
    night[:, :, 2] = night[:, :, 2] * 0.42 + 16

    height, width = night.shape[:2]

    y = np.linspace(1.10, 0.70, height).reshape(height, 1, 1)
    night = night * y

    yy, xx = np.mgrid[0:height, 0:width]
    cx, cy = width / 2, height / 2
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max(np.sqrt(cx ** 2 + cy ** 2), 1)
    vignette = 1.0 - 0.35 * dist
    night = night * vignette[:, :, None]

    rng = np.random.default_rng(variant_index + 42)
    night = night + rng.normal(0, 4.0, night.shape)

    night = np.clip(night, 0, 255)

    conditioned_background = (
        arr * (1.0 - condition_strength)
        + night * condition_strength
    )

    conditioned = conditioned_background * (1.0 - alpha) + arr * alpha
    conditioned = np.clip(conditioned, 0, 255).astype(np.uint8)

    return Image.fromarray(conditioned)


def local_luminance_adapt(
    original: Image.Image,
    generated: Image.Image,
    strict_mask: Image.Image,
) -> Image.Image:
    """
    Adjust the reinserted object brightness to reduce pasted-object artifacts.
    The object geometry remains the original one, but its luminance is slightly
    adapted to the generated context.
    """
    orig = np.array(original.convert("RGB")).astype(np.float32)
    gen = np.array(generated.convert("RGB")).astype(np.float32)

    mask = np.array(strict_mask.convert("L")) > 0

    if not mask.any():
        return original.convert("RGB")

    kernel = np.ones((35, 35), np.uint8)
    dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    ring = dilated & (~mask)

    if not ring.any():
        return original.convert("RGB")

    orig_lum = orig.mean(axis=2)
    gen_lum = gen.mean(axis=2)

    orig_ring = float(orig_lum[ring].mean())
    gen_ring = float(gen_lum[ring].mean())

    if orig_ring <= 1:
        factor = 1.0
    else:
        factor = gen_ring / orig_ring

    factor = float(np.clip(factor, 0.45, 1.15))

    adapted = orig.copy()
    adapted[mask] = np.clip(adapted[mask] * factor, 0, 255)

    return Image.fromarray(adapted.astype(np.uint8))


def soft_reinsert_object(
    original: Image.Image,
    generated: Image.Image,
    strict_mask: Image.Image,
    blend_mask: Image.Image,
    adapt_luminance: bool = True,
) -> Image.Image:
    original = original.convert("RGB")
    generated = generated.convert("RGB")

    if generated.size != original.size:
        generated = generated.resize(original.size, Image.Resampling.LANCZOS)

    source = local_luminance_adapt(original, generated, strict_mask) if adapt_luminance else original

    src = np.array(source).astype(np.float32)
    gen = np.array(generated).astype(np.float32)

    alpha = np.array(blend_mask.convert("L")).astype(np.float32) / 255.0
    alpha = alpha[:, :, None]

    out = src * alpha + gen * (1.0 - alpha)
    out = np.clip(out, 0, 255).astype(np.uint8)

    return Image.fromarray(out)


def draw_boxes(image: Image.Image, labels) -> Image.Image:
    arr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    height, width = arr.shape[:2]

    for _, x, y, w, h in labels:
        x1 = int((x - w / 2) * width)
        y1 = int((y - h / 2) * height)
        x2 = int((x + w / 2) * width)
        y2 = int((y + h / 2) * height)

        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width - 1, x2))
        y2 = max(0, min(height - 1, y2))

        cv2.rectangle(arr, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


def overlay_mask(image: Image.Image, mask: Image.Image, color: tuple[int, int, int]) -> Image.Image:
    image_arr = np.array(image.convert("RGB"))
    mask_arr = np.array(mask.convert("L"))

    overlay = image_arr.copy()
    overlay[mask_arr > 0] = color

    blended = cv2.addWeighted(image_arr, 0.60, overlay, 0.40, 0)
    return Image.fromarray(blended)


def save_grid(
    tiles: list[tuple[str, Image.Image]],
    output_path: Path,
    tile_size: int = 220,
) -> None:
    ensure_dir(output_path.parent)

    caption_h = 38
    cols = len(tiles)

    canvas = Image.new("RGB", (cols * tile_size, tile_size + caption_h), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)

    for i, (caption, tile) in enumerate(tiles):
        x = i * tile_size
        tile = tile.convert("RGB").resize((tile_size, tile_size), Image.Resampling.LANCZOS)
        canvas.paste(tile, (x, caption_h))
        draw.text((x + 6, 10), caption, fill=(0, 0, 0))

    canvas.save(output_path, quality=95)


def quality_metadata(
    original: Image.Image,
    generated: Image.Image,
    strict_mask: Image.Image,
    quality_cfg: dict[str, Any],
) -> dict[str, Any]:
    stats = image_pixel_stats(generated)
    diff_stats = compute_diff_stats(original, generated, strict_mask)

    suspicious_black = is_suspicious_black_image(stats)

    reasons = []

    if quality_cfg.get("reject_black_images", True) and suspicious_black:
        reasons.append("suspicious_black_image")

    if diff_stats["background_region_mean_abs_diff"] < float(quality_cfg["min_background_diff"]):
        reasons.append("background_diff_too_low")

    if diff_stats["object_region_mean_abs_diff"] > float(quality_cfg["max_object_diff"]):
        reasons.append("object_diff_too_high")

    if diff_stats["mask_coverage"] > float(quality_cfg["max_mask_coverage"]):
        reasons.append("mask_coverage_too_high")

    return {
        "suspicious_black_image": suspicious_black,
        "accepted_by_auto_gate": len(reasons) == 0,
        "rejection_reasons": reasons,
        "output_pixel_stats": stats,
        **diff_stats,
    }
