"""Object-preservation metrics for diffusion augmentation candidates."""

from dataclasses import dataclass

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

from src.augmentation.masks_from_boxes import BoxPixels


@dataclass(frozen=True)
class RejectionThresholds:
    min_object_ssim: float = 0.75
    max_object_mad: float = 20.0
    min_background_mad: float = 2.0


def _rgb_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32)


def _union_box(boxes: list[BoxPixels], image_size: tuple[int, int], pad: int = 8) -> tuple[int, int, int, int]:
    width, height = image_size
    x1 = max(0, min(box.x1 for box in boxes) - pad)
    y1 = max(0, min(box.y1 for box in boxes) - pad)
    x2 = min(width, max(box.x2 for box in boxes) + pad)
    y2 = min(height, max(box.y2 for box in boxes) + pad)
    return x1, y1, x2, y2


def _mean_absolute_difference(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))


def _crop_ssim(original_crop: np.ndarray, generated_crop: np.ndarray) -> float:
    h, w = original_crop.shape[:2]
    if h < 7 or w < 7:
        return 1.0 if np.array_equal(original_crop, generated_crop) else 0.0
    return float(
        structural_similarity(
            original_crop,
            generated_crop,
            channel_axis=2,
            data_range=255,
            win_size=min(7, h if h % 2 else h - 1, w if w % 2 else w - 1),
        )
    )


def compute_preservation_metrics(
    original: Image.Image,
    generated_before_reinsertion: Image.Image,
    generated_after_reinsertion: Image.Image,
    boxes: list[BoxPixels],
    protection_mask: Image.Image,
    mode: str,
) -> dict:
    """Compute object and background preservation metrics.

    Object metrics are measured on a crop around the labeled drone boxes.
    Background change is measured outside the protected mask, where inpainting
    is expected to modify the image.
    """
    if not boxes:
        raise ValueError("Preservation metrics require at least one object box.")
    if original.size != generated_after_reinsertion.size:
        raise ValueError(f"Size mismatch: original={original.size}, generated={generated_after_reinsertion.size}")

    orig_arr = _rgb_array(original)
    before_arr = _rgb_array(generated_before_reinsertion)
    after_arr = _rgb_array(generated_after_reinsertion)
    mask_arr = np.asarray(protection_mask.convert("L")) > 0
    x1, y1, x2, y2 = _union_box(boxes, original.size)

    orig_crop = orig_arr[y1:y2, x1:x2]
    before_crop = before_arr[y1:y2, x1:x2]
    after_crop = after_arr[y1:y2, x1:x2]
    background = ~mask_arr

    object_ssim_before = _crop_ssim(orig_crop.astype(np.uint8), before_crop.astype(np.uint8))
    object_ssim_after = _crop_ssim(orig_crop.astype(np.uint8), after_crop.astype(np.uint8))
    object_mad_before = _mean_absolute_difference(orig_crop, before_crop)
    object_mad_after = _mean_absolute_difference(orig_crop, after_crop)
    background_mad_after = _mean_absolute_difference(orig_arr[background], after_arr[background])
    protected_region_mad_after = _mean_absolute_difference(orig_arr[mask_arr], after_arr[mask_arr])

    object_exact_preserved = protected_region_mad_after == 0.0
    protected_box_preserved = mode == "background_inpaint_protected_box" and object_exact_preserved
    reinsertion_restored_crop = mode == "background_inpaint_reinsert_object" and object_exact_preserved

    return {
        "object_crop_xyxy": [x1, y1, x2, y2],
        "object_ssim_before_reinsertion": object_ssim_before,
        "object_ssim_after_reinsertion": object_ssim_after,
        "object_mad_before_reinsertion": object_mad_before,
        "object_mad_after_reinsertion": object_mad_after,
        "background_mad_outside_protection": background_mad_after,
        "protected_region_mad_after_reinsertion": protected_region_mad_after,
        "object_exact_preserved": object_exact_preserved,
        "protected_box_mode_preserved": protected_box_preserved,
        "reinsertion_restored_crop": reinsertion_restored_crop,
    }


def rejection_reasons(
    metrics: dict,
    output_stats: dict,
    size_matches: bool,
    thresholds: RejectionThresholds = RejectionThresholds(),
) -> list[str]:
    reasons = []
    if not size_matches:
        reasons.append("output_size_mismatch")
    if output_stats["mean"] < 5 or output_stats["max"] < 30 or output_stats["std"] < 2:
        reasons.append("almost_black")
    if metrics["object_ssim_after_reinsertion"] < thresholds.min_object_ssim:
        reasons.append("object_crop_ssim_too_low")
    if metrics["object_mad_after_reinsertion"] > thresholds.max_object_mad:
        reasons.append("object_crop_mad_too_high")
    if metrics["background_mad_outside_protection"] < thresholds.min_background_mad:
        reasons.append("background_almost_unchanged")
    return reasons
