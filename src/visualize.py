"""
Draw YOLO boxes on images for visual sanity checks.
"""
import argparse
from pathlib import Path
import cv2
from .utils import yolo_to_pixel_coords
import random


def draw_boxes(image_path: Path, label_path: Path, out_path: Path = None):
    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"Failed to read image {image_path}")
    h, w = img.shape[:2]
    if label_path.exists():
        with open(label_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                cls = parts[0]
                xc, yc, bw, bh = map(float, parts[1:5])
                x1, y1, x2, y2 = yolo_to_pixel_coords(xc, yc, bw, bh, w, h)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, cls, (x1, max(0, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    if out_path:
        cv2.imwrite(str(out_path), img)
    else:
        cv2.imshow("vis", img)
        cv2.waitKey(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo-dir", required=True, help="YOLO output dir containing images/ and labels/")
    parser.add_argument("--num", type=int, default=5)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    base = Path(args.yolo_dir)
    img_dir = base / "images" / "val"
    label_dir = base / "labels" / "val"
    files = list(img_dir.glob("*.jpg"))
    random.shuffle(files)
    files = files[: args.num]
    out_base = Path(args.out_dir) if args.out_dir else None
    if out_base:
        out_base.mkdir(parents=True, exist_ok=True)
    for f in files:
        lbl = label_dir / (f.name.replace('.jpg', '.txt'))
        out_p = out_base / f.name if out_base else None
        draw_boxes(f, lbl, out_p)


if __name__ == "__main__":
    main()
