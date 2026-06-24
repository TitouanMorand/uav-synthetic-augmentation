"""Reinsert original drone pixels after diffusion background editing."""

from PIL import Image

from src.augmentation.masks_from_boxes import BoxPixels


def extract_object_crop(image: Image.Image, box: BoxPixels) -> Image.Image:
    """Extract the rectangular object crop described by a pixel box."""
    return image.crop((box.x1, box.y1, box.x2, box.y2))


def reinsert_rectangular(generated: Image.Image, original: Image.Image, boxes: list[BoxPixels]) -> Image.Image:
    """Paste original rectangular box regions back into the generated image."""
    out = generated.copy()
    for box in boxes:
        crop = extract_object_crop(original, box)
        out.paste(crop, (box.x1, box.y1))
    return out


def reinsert_with_mask(
    generated: Image.Image,
    original: Image.Image,
    boxes: list[BoxPixels],
    object_mask: Image.Image | None = None,
) -> Image.Image:
    """Paste object crops with an optional binary mask over each crop."""
    out = generated.copy()
    full_mask = object_mask.convert("L") if object_mask is not None else None
    for box in boxes:
        crop = extract_object_crop(original, box)
        mask_crop = full_mask.crop((box.x1, box.y1, box.x2, box.y2)) if full_mask is not None else None
        out.paste(crop, (box.x1, box.y1), mask_crop)
    return out
