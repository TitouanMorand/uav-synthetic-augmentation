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
    """Paste original pixels back into generated image.

    With a protection mask, paste the full original image through that mask so
    the expanded safety region around tiny drones is restored exactly. Without
    a mask, fall back to rectangular box crops.
    """
    out = generated.copy()
    if object_mask is not None:
        out.paste(original.convert("RGB"), (0, 0), object_mask.convert("L"))
        return out

    for box in boxes:
        crop = extract_object_crop(original, box)
        out.paste(crop, (box.x1, box.y1))
    return out
