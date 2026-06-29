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
    image_area = max(width * height, 1)

    max_sides = []
    areas = []
    area_ratios = []

    for _, x, y, w, h in labels:
        box_w = w * width
        box_h = h * height
        box_area = box_w * box_h

        max_sides.append(max(box_w, box_h))
        areas.append(box_area)
        area_ratios.append(box_area / image_area)

    return {
        "max_box_side_px": float(max(max_sides)) if max_sides else 0.0,
        "mean_box_side_px": float(np.mean(max_sides)) if max_sides else 0.0,
        "max_box_area_px": float(max(areas)) if areas else 0.0,
        "max_box_area_ratio": float(max(area_ratios)) if area_ratios else 0.0,
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


def compute_ui_rectangle_score(image_bgr: np.ndarray) -> float:
    """
    Heuristic for HUD, inset images, picture-in-picture windows and screen overlays.

    This catches rectangular regions even when they touch borders, which is common
    for drone controller views or embedded camera feeds.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    height, width = gray.shape[:2]
    image_area = max(height * width, 1)

    best_score = 0.0

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        area_ratio = (w * h) / image_area

        if area_ratio < 0.008 or area_ratio > 0.55:
            continue

        aspect = max(w / max(h, 1), h / max(w, 1))

        if aspect > 10:
            continue

        rect_area = max(w * h, 1)
        contour_area = cv2.contourArea(contour)
        rectangularity = contour_area / rect_area

        if rectangularity < 0.15:
            continue

        touches_border = (
            x < 8
            or y < 8
            or x + w > width - 8
            or y + h > height - 8
        )

        border_bonus = 1.8 if touches_border else 1.0

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)

        vertex_bonus = 1.3 if len(approx) <= 8 else 1.0

        score = area_ratio * rectangularity * border_bonus * vertex_bonus
        best_score = max(best_score, float(score))

    return best_score


def _local_maxima_indices(values: np.ndarray, threshold: float, min_distance: int, max_peaks: int) -> list[int]:
    candidates = np.where(values >= threshold)[0].tolist()
    candidates = sorted(candidates, key=lambda i: values[i], reverse=True)

    selected = []

    for idx in candidates:
        if all(abs(idx - kept) >= min_distance for kept in selected):
            selected.append(idx)

        if len(selected) >= max_peaks:
            break

    return sorted(selected)


def compute_inset_window_score(image_bgr: np.ndarray) -> float:
    """
    Detect picture-in-picture / embedded drone-camera views.

    Unlike compute_internal_rectangle_score(), this does not require a closed contour.
    It searches for strong horizontal/vertical border lines and measures whether
    they form a plausible inset window.
    """
    height, width = image_bgr.shape[:2]

    if height < 64 or width < 64:
        return 0.0

    target_width = 512

    if width > target_width:
        scale = target_width / width
        resized = cv2.resize(
            image_bgr,
            (target_width, int(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    else:
        resized = image_bgr.copy()
        scale = 1.0

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    edges = cv2.Canny(gray, 60, 150)
    edges01 = (edges > 0).astype(np.float32)

    h, w = gray.shape[:2]

    row_strength = edges01.mean(axis=1)
    col_strength = edges01.mean(axis=0)

    row_strength = cv2.blur(row_strength.reshape(-1, 1), (1, 9)).ravel()
    col_strength = cv2.blur(col_strength.reshape(1, -1), (9, 1)).ravel()

    row_threshold = max(float(np.percentile(row_strength, 97.5)), float(row_strength.mean() + 2.0 * row_strength.std()))
    col_threshold = max(float(np.percentile(col_strength, 97.5)), float(col_strength.mean() + 2.0 * col_strength.std()))

    row_peaks = _local_maxima_indices(
        row_strength,
        threshold=row_threshold,
        min_distance=max(12, h // 25),
        max_peaks=18,
    )

    col_peaks = _local_maxima_indices(
        col_strength,
        threshold=col_threshold,
        min_distance=max(12, w // 25),
        max_peaks=18,
    )

    if len(row_peaks) < 2 or len(col_peaks) < 2:
        return 0.0

    best_score = 0.0

    image_area = h * w

    for y1_i in range(len(row_peaks)):
        for y2_i in range(y1_i + 1, len(row_peaks)):
            y1 = row_peaks[y1_i]
            y2 = row_peaks[y2_i]

            rect_h = y2 - y1

            if rect_h < h * 0.08 or rect_h > h * 0.75:
                continue

            for x1_i in range(len(col_peaks)):
                for x2_i in range(x1_i + 1, len(col_peaks)):
                    x1 = col_peaks[x1_i]
                    x2 = col_peaks[x2_i]

                    rect_w = x2 - x1

                    if rect_w < w * 0.08 or rect_w > w * 0.75:
                        continue

                    area_ratio = (rect_w * rect_h) / max(image_area, 1)

                    if area_ratio < 0.015 or area_ratio > 0.55:
                        continue

                    band = max(2, int(min(h, w) * 0.006))

                    top = edges01[max(0, y1 - band):min(h, y1 + band + 1), x1:x2]
                    bottom = edges01[max(0, y2 - band):min(h, y2 + band + 1), x1:x2]
                    left = edges01[y1:y2, max(0, x1 - band):min(w, x1 + band + 1)]
                    right = edges01[y1:y2, max(0, x2 - band):min(w, x2 + band + 1)]

                    if top.size == 0 or bottom.size == 0 or left.size == 0 or right.size == 0:
                        continue

                    border_density = float(np.mean([top.mean(), bottom.mean(), left.mean(), right.mean()]))

                    inside = gray[y1 + band:y2 - band, x1 + band:x2 - band]

                    if inside.size == 0:
                        continue

                    outer_parts = []

                    if y1 - 3 * band >= 0:
                        outer_parts.append(gray[y1 - 3 * band:y1 - band, x1:x2])
                    if y2 + 3 * band < h:
                        outer_parts.append(gray[y2 + band:y2 + 3 * band, x1:x2])
                    if x1 - 3 * band >= 0:
                        outer_parts.append(gray[y1:y2, x1 - 3 * band:x1 - band])
                    if x2 + 3 * band < w:
                        outer_parts.append(gray[y1:y2, x2 + band:x2 + 3 * band])

                    if outer_parts:
                        outer_mean = float(np.mean([part.mean() for part in outer_parts if part.size > 0]))
                        color_jump = abs(float(inside.mean()) - outer_mean) / 255.0
                    else:
                        color_jump = 0.0

                    near_corner_or_bottom = (
                        x1 < w * 0.12
                        or x2 > w * 0.88
                        or y2 > h * 0.78
                        or y1 < h * 0.12
                    )

                    location_bonus = 1.35 if near_corner_or_bottom else 1.0

                    score = location_bonus * (
                        0.70 * border_density
                        + 0.20 * color_jump
                        + 0.10 * min(1.0, area_ratio * 6.0)
                    )

                    best_score = max(best_score, float(score))

    return best_score
