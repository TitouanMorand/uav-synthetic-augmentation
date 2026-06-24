"""Create diffusion masks from YOLO boxes.

Mask convention:
- Object/protection masks returned by `object_mask_from_labels` use white (255)
  for protected object pixels and black (0) elsewhere.
- Inpainting masks returned by `background_inpaint_mask_from_labels` follow the
  Diffusers convention: white (255) pixels are repainted, black (0) pixels are
  preserved. Therefore the drone box is black and background is white.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.utils import yolo_to_pixel_coords

DEFAULT_BOX_MARGIN_PX = 32
DEFAULT_BOX_MARGIN_RATIO = 2.0


@dataclass(frozen=True)
class BoxPixels:
    class_id: int
    x1: int
    y1: int
    x2: int
    y2: int


def _clip_box(x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> tuple[int, int, int, int]:
    return max(0, x1), max(0, y1), min(width, x2), min(height, y2)


def read_yolo_boxes(label_path: Path, image_size: tuple[int, int]) -> list[BoxPixels]:
    """Read a YOLO label file and convert normalized boxes to pixel boxes."""
    width, height = image_size
    boxes: list[BoxPixels] = []
    if not label_path.exists():
        return boxes

    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        xc, yc, bw, bh = map(float, parts[1:5])
        x1, y1, x2, y2 = yolo_to_pixel_coords(xc, yc, bw, bh, width, height)
        x1, y1, x2, y2 = _clip_box(x1, y1, x2, y2, width, height)
        if x2 > x1 and y2 > y1:
            boxes.append(BoxPixels(cls, x1, y1, x2, y2))
    return boxes


def expand_box(
    box: BoxPixels,
    image_size: tuple[int, int],
    pixel_margin: int = DEFAULT_BOX_MARGIN_PX,
    relative_margin: float = DEFAULT_BOX_MARGIN_RATIO,
) -> BoxPixels:
    """Expand a pixel box by fixed and relative margins, clipped to image size."""
    width, height = image_size
    box_w = box.x2 - box.x1
    box_h = box.y2 - box.y1
    margin_x = int(round(max(pixel_margin, box_w * relative_margin)))
    margin_y = int(round(max(pixel_margin, box_h * relative_margin)))
    x1, y1, x2, y2 = _clip_box(
        box.x1 - margin_x,
        box.y1 - margin_y,
        box.x2 + margin_x,
        box.y2 + margin_y,
        width,
        height,
    )
    return BoxPixels(box.class_id, x1, y1, x2, y2)


def select_protected_boxes(boxes: list[BoxPixels], protect_all_boxes: bool = True) -> list[BoxPixels]:
    """Return boxes to protect. Default protects every labeled drone."""
    if protect_all_boxes:
        return boxes
    if not boxes:
        return []
    return [max(boxes, key=lambda box: (box.x2 - box.x1) * (box.y2 - box.y1))]


def object_mask_from_boxes(
    boxes: list[BoxPixels],
    image_size: tuple[int, int],
    pixel_margin: int = DEFAULT_BOX_MARGIN_PX,
    relative_margin: float = DEFAULT_BOX_MARGIN_RATIO,
    protect_all_boxes: bool = True,
) -> Image.Image:
    """Return a white-on-black mask where white marks protected object regions.

    Small drones need a region larger than the exact YOLO box because diffusion
    can alter object boundaries and nearby pixels even when the nominal object
    box is masked. The default margin therefore uses at least 32 px or 2x box
    size on each side, clipped to the image bounds.
    """
    width, height = image_size
    mask = np.zeros((height, width), dtype=np.uint8)
    for box in select_protected_boxes(boxes, protect_all_boxes):
        expanded = expand_box(box, image_size, pixel_margin, relative_margin)
        mask[expanded.y1 : expanded.y2, expanded.x1 : expanded.x2] = 255
    return Image.fromarray(mask, mode="L")


def object_mask_from_labels(
    label_path: Path,
    image_size: tuple[int, int],
    pixel_margin: int = DEFAULT_BOX_MARGIN_PX,
    relative_margin: float = DEFAULT_BOX_MARGIN_RATIO,
    protect_all_boxes: bool = True,
) -> Image.Image:
    """Read labels and return a white object/protection mask."""
    boxes = read_yolo_boxes(label_path, image_size)
    return object_mask_from_boxes(boxes, image_size, pixel_margin, relative_margin, protect_all_boxes)


def background_inpaint_mask_from_labels(
    label_path: Path,
    image_size: tuple[int, int],
    pixel_margin: int = DEFAULT_BOX_MARGIN_PX,
    relative_margin: float = DEFAULT_BOX_MARGIN_RATIO,
    protect_all_boxes: bool = True,
) -> Image.Image:
    """Return a Diffusers inpaint mask: white background, black protected drones.

    Diffusers inpainting convention is explicit here:
    - white 255 pixels are repainted/generated
    - black 0 pixels are preserved from the source image
    """
    object_mask = np.array(object_mask_from_labels(label_path, image_size, pixel_margin, relative_margin, protect_all_boxes))
    inpaint_mask = np.full_like(object_mask, 255)
    inpaint_mask[object_mask > 0] = 0
    assert np.all(inpaint_mask[object_mask > 0] == 0), "Protected object region must be black in inpaint mask."
    assert np.all(inpaint_mask[object_mask == 0] == 255), "Editable background must be white in inpaint mask."
    return Image.fromarray(inpaint_mask, mode="L")


def save_mask_debug_previews(
    image: Image.Image,
    protection_mask: Image.Image,
    inpaint_mask: Image.Image,
    out_dir: Path,
    stem: str,
) -> dict[str, str]:
    """Save mask images and overlays for debugging mask conventions."""
    out_dir.mkdir(parents=True, exist_ok=True)
    protection_path = out_dir / f"{stem}__protection_mask.png"
    inpaint_path = out_dir / f"{stem}__inpaint_mask.png"
    protection_overlay_path = out_dir / f"{stem}__protection_overlay.jpg"
    inpaint_overlay_path = out_dir / f"{stem}__inpaint_overlay.jpg"
    protection_mask.convert("L").save(protection_path)
    inpaint_mask.convert("L").save(inpaint_path)
    overlay_mask_on_image(image, protection_mask).save(protection_overlay_path, quality=95)
    overlay_mask_on_image(image, inpaint_mask).save(inpaint_overlay_path, quality=95)
    return {
        "protection_mask_path": str(protection_path),
        "inpaint_mask_path": str(inpaint_path),
        "protection_overlay_path": str(protection_overlay_path),
        "inpaint_overlay_path": str(inpaint_overlay_path),
    }


def overlay_mask_on_image(image: Image.Image, mask: Image.Image, alpha: float = 0.45) -> Image.Image:
    """Overlay mask pixels in red for visual debugging."""
    image_rgb = np.array(image.convert("RGB"))
    mask_arr = np.array(mask.convert("L"))
    overlay = image_rgb.copy()
    overlay[mask_arr > 0] = (255, 40, 40)
    blended = cv2.addWeighted(image_rgb, 1.0 - alpha, overlay, alpha, 0.0)
    return Image.fromarray(blended)
