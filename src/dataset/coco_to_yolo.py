from typing import List, Tuple

def coco_to_yolo(bbox: List[float], img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    """
    Convert COCO bbox [x, y, width, height] in absolute pixels to YOLO normalized
    format (x_center, y_center, width, height) with values in [0,1].

    bbox: [x, y, w, h]
    """
    x, y, w, h = bbox
    x_c = x + w / 2.0
    y_c = y + h / 2.0
    return x_c / img_w, y_c / img_h, w / img_w, h / img_h
