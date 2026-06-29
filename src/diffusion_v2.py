from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from src.utils import ensure_dir


# =========================
# Basic geometry / drawing
# =========================

def _to_rgb_pil(image: Image.Image) -> Image.Image:
    return image.convert("RGB")


def _labels_to_xyxy(labels, image_size: tuple[int, int]) -> list[tuple[int, int, int, int]]:
    width, height = image_size
    boxes = []

    for _, x, y, w, h in labels:
        x1 = int(round((x - w / 2) * width))
        y1 = int(round((y - h / 2) * height))
        x2 = int(round((x + w / 2) * width))
        y2 = int(round((y + h / 2) * height))

        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(x1 + 1, min(width, x2))
        y2 = max(y1 + 1, min(height, y2))

        boxes.append((x1, y1, x2, y2))

    return boxes


def labels_box_stats(labels, image_size: tuple[int, int]) -> dict[str, float]:
    width, height = image_size
    image_area = max(width * height, 1)

    max_sides = []
    areas = []
    area_ratios = []

    for x1, y1, x2, y2 in _labels_to_xyxy(labels, image_size):
        box_w = x2 - x1
        box_h = y2 - y1
        area = box_w * box_h

        max_sides.append(max(box_w, box_h))
        areas.append(area)
        area_ratios.append(area / image_area)

    return {
        "max_box_side_px": float(max(max_sides)) if max_sides else 0.0,
        "mean_box_side_px": float(np.mean(max_sides)) if max_sides else 0.0,
        "max_box_area_px": float(max(areas)) if areas else 0.0,
        "max_box_area_ratio": float(max(area_ratios)) if area_ratios else 0.0,
    }


def draw_boxes(image: Image.Image, labels) -> Image.Image:
    image = _to_rgb_pil(image)
    draw = ImageDraw.Draw(image)

    for x1, y1, x2, y2 in _labels_to_xyxy(labels, image.size):
        draw.rectangle([x1, y1, x2, y2], outline="lime", width=3)

    return image


def overlay_mask(image: Image.Image, mask: Image.Image, color: tuple[int, int, int]) -> Image.Image:
    image_arr = np.array(_to_rgb_pil(image)).astype(np.uint8)
    mask_arr = np.array(mask.convert("L")) > 0

    overlay = image_arr.copy()
    overlay[mask_arr] = color

    blended = cv2.addWeighted(image_arr, 0.65, overlay, 0.35, 0)
    return Image.fromarray(blended)


def preview_alpha(alpha_mask: Image.Image) -> Image.Image:
    arr = np.array(alpha_mask.convert("L"))
    rgb = np.stack([arr, arr, arr], axis=-1)
    return Image.fromarray(rgb.astype(np.uint8))


# =========================
# Source filtering heuristics
# =========================

def compute_mask_coverage(labels, image_size: tuple[int, int], margin_px: int = 0) -> float:
    width, height = image_size
    mask = np.zeros((height, width), dtype=np.uint8)

    for x1, y1, x2, y2 in _labels_to_xyxy(labels, image_size):
        x1 = max(0, x1 - margin_px)
        y1 = max(0, y1 - margin_px)
        x2 = min(width, x2 + margin_px)
        y2 = min(height, y2 + margin_px)
        mask[y1:y2, x1:x2] = 1

    return float(mask.mean())


def compute_internal_rectangle_score(image_bgr: np.ndarray) -> float:
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

        if x < 8 or y < 8 or x + bw > w - 8 or y + bh > h - 8:
            continue

        rect_area = max(bw * bh, 1)
        contour_area = cv2.contourArea(contour)
        rectangularity = contour_area / rect_area

        if rectangularity < 0.12:
            continue

        score = area_ratio * rectangularity
        best_score = max(best_score, float(score))

    return best_score


def compute_ui_rectangle_score(image_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = gray.shape[:2]
    image_area = h * w
    best_score = 0.0

    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        area_ratio = (bw * bh) / max(image_area, 1)

        if area_ratio < 0.008 or area_ratio > 0.55:
            continue

        touches_border = (
            x < 8 or y < 8 or x + bw > w - 8 or y + bh > h - 8
        )

        if not touches_border:
            continue

        rect_area = max(bw * bh, 1)
        contour_area = cv2.contourArea(contour)
        rectangularity = contour_area / rect_area

        if rectangularity < 0.10:
            continue

        score = area_ratio * rectangularity * 1.5
        best_score = max(best_score, float(score))

    return best_score


def compute_vertical_seam_score(image_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if gray.shape[1] < 2:
        return 0.0

    col_diff = np.abs(gray[:, 1:] - gray[:, :-1]).mean(axis=0)

    start = int(gray.shape[1] * 0.15)
    end = int(gray.shape[1] * 0.85)
    internal = col_diff[start:end]

    if len(internal) == 0:
        return 0.0

    return float(np.percentile(internal, 99))


def compute_inset_window_score(image_bgr: np.ndarray) -> float:
    """
    Heuristic for picture-in-picture / embedded camera window.

    The previous version was too restrictive and often returned 0.
    This version explicitly tests common window locations and scores
    rectangular borders + color discontinuity.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 60, 150)
    edges01 = (edges > 0).astype(np.float32)

    h, w = gray.shape[:2]
    if h < 64 or w < 64:
        return 0.0

    best_score = 0.0

    width_fracs = [0.18, 0.24, 0.30, 0.36]
    height_fracs = [0.14, 0.18, 0.22, 0.28]

    anchors = [
        ("bottom_left", 0.03, 0.97, 0.97, 0.03),
        ("bottom_right", 0.97, 0.97, 0.03, 0.03),
        ("bottom_center", 0.50, 0.97, 0.50, 0.03),
        ("top_left", 0.03, 0.03, 0.97, 0.97),
        ("top_right", 0.97, 0.03, 0.03, 0.97),
    ]

    for wf in width_fracs:
        for hf in height_fracs:
            ww = int(round(w * wf))
            hh = int(round(h * hf))
            if ww < 24 or hh < 24:
                continue

            for name, ax, ay, _, _ in anchors:
                if "left" in name:
                    x1 = int(round(w * 0.03))
                elif "right" in name:
                    x1 = w - ww - int(round(w * 0.03))
                else:
                    x1 = (w - ww) // 2

                if "top" in name:
                    y1 = int(round(h * 0.03))
                else:
                    y1 = h - hh - int(round(h * 0.03))

                x2 = x1 + ww
                y2 = y1 + hh

                if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
                    continue

                band = max(2, int(min(h, w) * 0.008))
                if x2 - x1 <= 2 * band or y2 - y1 <= 2 * band:
                    continue

                # Border edge density
                border_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.rectangle(border_mask, (x1, y1), (x2 - 1, y2 - 1), 1, thickness=band)

                border_edge_density = float(edges01[border_mask > 0].mean()) if np.any(border_mask > 0) else 0.0

                # Inside/outside intensity discontinuity
                inside = gray[y1 + band:y2 - band, x1 + band:x2 - band]
                if inside.size == 0:
                    continue

                outer_ring_mask = np.zeros((h, w), dtype=np.uint8)
                ox1 = max(0, x1 - 2 * band)
                oy1 = max(0, y1 - 2 * band)
                ox2 = min(w, x2 + 2 * band)
                oy2 = min(h, y2 + 2 * band)

                outer_ring_mask[oy1:oy2, ox1:ox2] = 1
                outer_ring_mask[y1:y2, x1:x2] = 0

                if np.any(outer_ring_mask > 0):
                    outer_mean = float(gray[outer_ring_mask > 0].mean())
                else:
                    outer_mean = float(inside.mean())

                color_jump = abs(float(inside.mean()) - outer_mean) / 255.0

                area_ratio = (ww * hh) / max(h * w, 1)
                location_bonus = 1.20 if ("bottom" in name or "top" in name) else 1.0

                score = location_bonus * (
                    0.70 * border_edge_density +
                    0.25 * color_jump +
                    0.05 * min(1.0, area_ratio / 0.08)
                )

                best_score = max(best_score, float(score))

    return best_score


# =========================
# Object matte / alpha / relighting
# =========================

def _ellipse_mask_for_box(box: tuple[int, int, int, int], image_size: tuple[int, int]) -> np.ndarray:
    width, height = image_size
    mask = np.zeros((height, width), dtype=np.uint8)
    x1, y1, x2, y2 = box

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    ax = max(2, int((x2 - x1) * 0.60))
    ay = max(2, int((y2 - y1) * 0.60))

    cv2.ellipse(mask, (cx, cy), (ax // 2, ay // 2), 0, 0, 360, 255, -1)
    return mask


def make_object_matte(image: Image.Image, labels) -> Image.Image:
    """
    Build an approximate object silhouette from the YOLO box.
    Uses GrabCut when possible, otherwise falls back to an ellipse.
    """
    image = _to_rgb_pil(image)
    image_arr = np.array(image)
    height, width = image_arr.shape[:2]

    final_mask = np.zeros((height, width), dtype=np.uint8)

    boxes = _labels_to_xyxy(labels, image.size)

    for box in boxes:
        x1, y1, x2, y2 = box
        bw = x2 - x1
        bh = y2 - y1

        pad_x = max(10, int(bw * 1.2))
        pad_y = max(10, int(bh * 1.2))

        rx1 = max(0, x1 - pad_x)
        ry1 = max(0, y1 - pad_y)
        rx2 = min(width, x2 + pad_x)
        ry2 = min(height, y2 + pad_y)

        roi = image_arr[ry1:ry2, rx1:rx2].copy()
        roi_h, roi_w = roi.shape[:2]

        # Tiny object fallback
        if roi_h < 12 or roi_w < 12 or bw < 6 or bh < 6:
            final_mask = np.maximum(final_mask, _ellipse_mask_for_box(box, image.size))
            continue

        local_x1 = x1 - rx1
        local_y1 = y1 - ry1
        local_x2 = x2 - rx1
        local_y2 = y2 - ry1

        gc_mask = np.full((roi_h, roi_w), cv2.GC_PR_BGD, dtype=np.uint8)

        # Probable foreground in the box
        gc_mask[local_y1:local_y2, local_x1:local_x2] = cv2.GC_PR_FGD

        # Strong foreground in the inner box
        ix1 = local_x1 + max(1, int(0.20 * bw))
        iy1 = local_y1 + max(1, int(0.20 * bh))
        ix2 = local_x2 - max(1, int(0.20 * bw))
        iy2 = local_y2 - max(1, int(0.20 * bh))

        if ix2 > ix1 and iy2 > iy1:
            gc_mask[iy1:iy2, ix1:ix2] = cv2.GC_FGD

        # Border as probable background
        border = 3
        gc_mask[:border, :] = cv2.GC_BGD
        gc_mask[-border:, :] = cv2.GC_BGD
        gc_mask[:, :border] = cv2.GC_BGD
        gc_mask[:, -border:] = cv2.GC_BGD

        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        try:
            cv2.grabCut(roi, gc_mask, None, bgd_model, fgd_model, 2, cv2.GC_INIT_WITH_MASK)
            roi_fg = np.where(
                (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
                255,
                0,
            ).astype(np.uint8)

            # Morph cleanup
            kernel = np.ones((3, 3), np.uint8)
            roi_fg = cv2.morphologyEx(roi_fg, cv2.MORPH_OPEN, kernel, iterations=1)
            roi_fg = cv2.morphologyEx(roi_fg, cv2.MORPH_CLOSE, kernel, iterations=1)

            area = int((roi_fg > 0).sum())
            box_area = max(1, bw * bh)

            # If GrabCut clearly failed, fallback to ellipse
            if area < 0.05 * box_area or area > 3.0 * box_area:
                ellipse_mask = _ellipse_mask_for_box(box, image.size)
                final_mask = np.maximum(final_mask, ellipse_mask)
            else:
                final_mask[ry1:ry2, rx1:rx2] = np.maximum(final_mask[ry1:ry2, rx1:rx2], roi_fg)

        except Exception:
            ellipse_mask = _ellipse_mask_for_box(box, image.size)
            final_mask = np.maximum(final_mask, ellipse_mask)

    if (final_mask > 0).sum() == 0 and boxes:
        for box in boxes:
            final_mask = np.maximum(final_mask, _ellipse_mask_for_box(box, image.size))

    return Image.fromarray(final_mask, mode="L")


def make_feather_alpha(object_mask: Image.Image, feather_px: int) -> Image.Image:
    """
    alpha = 1 on the object, then gradually decays to 0 outside the object.
    This is the true soft alpha transition we need to avoid halo / hard collage.
    """
    mask = np.array(object_mask.convert("L")) > 0

    if feather_px <= 0:
        return Image.fromarray((mask.astype(np.uint8) * 255), mode="L")

    outside = (~mask).astype(np.uint8)
    dist_out = cv2.distanceTransform(outside, cv2.DIST_L2, 5)

    alpha = np.ones(mask.shape, dtype=np.float32)
    alpha[~mask] = np.clip(1.0 - (dist_out[~mask] / float(feather_px)), 0.0, 1.0)
    alpha[mask] = 1.0

    return Image.fromarray(np.clip(alpha * 255.0, 0, 255).astype(np.uint8), mode="L")


def composite_with_alpha(background: Image.Image, foreground: Image.Image, alpha_mask: Image.Image) -> Image.Image:
    bg = np.array(_to_rgb_pil(background)).astype(np.float32)
    fg = np.array(_to_rgb_pil(foreground)).astype(np.float32)
    alpha = np.array(alpha_mask.convert("L")).astype(np.float32) / 255.0
    alpha = alpha[:, :, None]

    out = fg * alpha + bg * (1.0 - alpha)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


def make_night_condition(image: Image.Image, darkness: float = 0.85) -> Image.Image:
    """
    Build a global low-light version of the image.
    Important: no preserved rectangular patch around the drone.
    """
    arr = np.array(_to_rgb_pil(image)).astype(np.float32)
    h, w = arr.shape[:2]

    # Darken + cool down
    arr[..., 0] *= (0.30 - 0.08 * darkness)   # R
    arr[..., 1] *= (0.38 - 0.08 * darkness)   # G
    arr[..., 2] *= (0.55 - 0.08 * darkness)   # B
    arr[..., 2] += 22.0 * darkness

    # Mild vertical illumination gradient
    grad = np.linspace(1.05, 0.75, h).reshape(h, 1, 1)
    arr *= grad

    # Mild vignette
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w / 2.0, h / 2.0
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    dist /= max(np.sqrt(cx ** 2 + cy ** 2), 1.0)
    vignette = 1.0 - 0.18 * dist
    arr *= vignette[:, :, None]

    # Mild sensor noise
    rng = np.random.default_rng(123)
    arr += rng.normal(0, 3.0, arr.shape)

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")



def relight_object_to_context(
    original: Image.Image,
    generated_context: Image.Image,
    object_mask: Image.Image,
    ring_inner_px: int = 4,
    ring_outer_px: int = 18,
    relight_strength: float = 0.25,
) -> Image.Image:
    """
    Conservative relighting:
    - mainly adapts luminance (L channel),
    - only very small color correction,
    - blends back with the original object to avoid over-darkening.
    """
    original_rgb = np.array(_to_rgb_pil(original)).astype(np.uint8)
    context_rgb = np.array(_to_rgb_pil(generated_context)).astype(np.uint8)
    mask = (np.array(object_mask.convert("L")) > 0).astype(np.uint8)

    if mask.sum() == 0:
        return _to_rgb_pil(original)

    kernel_inner = np.ones((max(1, ring_inner_px), max(1, ring_inner_px)), np.uint8)
    kernel_outer = np.ones((max(1, ring_outer_px), max(1, ring_outer_px)), np.uint8)

    dilated_inner = cv2.dilate(mask, kernel_inner, iterations=1)
    dilated_outer = cv2.dilate(mask, kernel_outer, iterations=1)

    ring = (dilated_outer > 0) & (dilated_inner == 0)
    obj_mask = mask > 0

    if not ring.any():
        return _to_rgb_pil(original)

    orig_lab = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    ctx_lab = cv2.cvtColor(context_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    relit_lab = orig_lab.copy()

    orig_ring_mean = orig_lab[ring].mean(axis=0)
    orig_ring_std = orig_lab[ring].std(axis=0) + 1e-6
    ctx_ring_mean = ctx_lab[ring].mean(axis=0)
    ctx_ring_std = ctx_lab[ring].std(axis=0) + 1e-6

    # --- L channel: conservative partial adaptation
    L = relit_lab[..., 0]
    L_obj = L[obj_mask].copy()

    scale_L = np.clip(ctx_ring_std[0] / orig_ring_std[0], 0.90, 1.08)
    target_L = (L_obj - orig_ring_mean[0]) * scale_L + ctx_ring_mean[0]

    # Partial move only, to avoid over-darkening.
    L_obj = (1.0 - relight_strength) * L_obj + relight_strength * target_L
    L[obj_mask] = np.clip(L_obj, 0, 255)

    # --- A/B channels: only tiny mean shifts
    for ch, max_shift in [(1, 4.0), (2, 6.0)]:
        C = relit_lab[..., ch]
        C_obj = C[obj_mask].copy()

        shift = float(np.clip(ctx_ring_mean[ch] - orig_ring_mean[ch], -max_shift, max_shift))
        C_obj = C_obj + relight_strength * shift
        C[obj_mask] = np.clip(C_obj, 0, 255)

    relit_rgb_full = cv2.cvtColor(relit_lab.astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)

    # Final safety blend with the original object.
    blended = original_rgb.astype(np.float32).copy()
    for c in range(3):
        blended[..., c][obj_mask] = (
            (1.0 - relight_strength) * original_rgb[..., c][obj_mask].astype(np.float32)
            + relight_strength * relit_rgb_full[..., c][obj_mask]
        )

    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return Image.fromarray(blended, mode="RGB")


# =========================
# Metrics / quality gate
# =========================

def _mean_abs_diff(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    diff = np.abs(a.astype(np.float32) - b.astype(np.float32)).mean(axis=-1)
    if mask is None:
        return float(diff.mean())
    if mask.sum() == 0:
        return 0.0
    return float(diff[mask].mean())


def _pixel_stats(image: Image.Image) -> dict[str, float]:
    arr = np.array(_to_rgb_pil(image)).astype(np.float32)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def is_suspicious_black_image(image: Image.Image) -> bool:
    stats = _pixel_stats(image)
    return stats["mean"] < 6.0 or stats["max"] < 20.0


def quality_metadata(
    original: Image.Image,
    generated_context: Image.Image,
    composite: Image.Image,
    object_mask: Image.Image,
    alpha_mask: Image.Image,
    quality_cfg: dict[str, Any],
) -> dict[str, Any]:
    orig = np.array(_to_rgb_pil(original))
    ctx = np.array(_to_rgb_pil(generated_context))
    comp = np.array(_to_rgb_pil(composite))

    obj = np.array(object_mask.convert("L")) > 0
    alpha = np.array(alpha_mask.convert("L")).astype(np.float32) / 255.0

    background_mask = alpha < 0.01
    transition_mask = (alpha > 0.01) & (alpha < 0.99)

    background_diff = _mean_abs_diff(orig, comp, background_mask)
    object_diff = _mean_abs_diff(orig, comp, obj)
    halo_score = _mean_abs_diff(ctx, comp, transition_mask)
    context_diff = _mean_abs_diff(orig, ctx, background_mask)

    suspicious_black = is_suspicious_black_image(composite)

    reasons = []

    if quality_cfg.get("reject_black_images", True) and suspicious_black:
        reasons.append("suspicious_black_image")

    if background_diff < float(quality_cfg["min_background_diff"]):
        reasons.append("background_diff_too_low")

    if halo_score > float(quality_cfg["max_halo_score"]):
        reasons.append("halo_score_too_high")

    if object_diff > float(quality_cfg["max_object_diff"]):
        reasons.append("object_diff_too_high")

    return {
        "accepted_by_auto_gate": len(reasons) == 0,
        "rejection_reasons": reasons,
        "background_region_mean_abs_diff": background_diff,
        "object_region_mean_abs_diff": object_diff,
        "context_region_mean_abs_diff": context_diff,
        "halo_score": halo_score,
        "mask_coverage": float(obj.mean()),
        "output_pixel_mean": float(comp.mean()),
        "output_pixel_std": float(comp.std()),
        "suspicious_black_image": suspicious_black,
    }


# =========================
# Preview grid
# =========================

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
        tile = _to_rgb_pil(tile).resize((tile_size, tile_size), Image.Resampling.LANCZOS)
        canvas.paste(tile, (x, caption_h))
        draw.text((x + 6, 10), caption, fill=(0, 0, 0))

    canvas.save(output_path, quality=95)
