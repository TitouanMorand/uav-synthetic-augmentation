from typing import Tuple, List

def coco_to_yolo(bbox: List[float], img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    """
    Convert COCO bbox [x, y, width, height] (absolute pixels) to YOLO normalized
    [x_center, y_center, width, height] where values are relative to image size.
    """
    x, y, w, h = bbox
    x_c = x + w / 2.0
    y_c = y + h / 2.0
    return x_c / img_w, y_c / img_h, w / img_w, h / img_h

def yolo_to_pixel_coords(xc: float, yc: float, w: float, h: float, img_w: int, img_h: int):
    """Convert normalized YOLO box to pixel coords: (x1,y1,x2,y2)"""
    x_c = xc * img_w
    y_c = yc * img_h
    bw = w * img_w
    bh = h * img_h
    x1 = int(round(x_c - bw / 2.0))
    y1 = int(round(y_c - bh / 2.0))
    x2 = int(round(x_c + bw / 2.0))
    y2 = int(round(y_c + bh / 2.0))
    return x1, y1, x2, y2
