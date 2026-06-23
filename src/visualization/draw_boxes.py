"""
Draw YOLO-format boxes on images and save previews.
"""
import argparse
from pathlib import Path
import cv2
import random
from src.utils import yolo_to_pixel_coords


def draw_single(image_path: Path, label_path: Path, out_path: Path):
    img = cv2.imread(str(image_path))
    if img is None:
        return False
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
                cv2.putText(img, cls, (max(0,x1), max(12,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    return True


def make_previews(yolo_dir: str, out_dir: str, num: int = 20):
    base = Path(yolo_dir)
    img_dir = base / "images" / "val"
    lbl_dir = base / "labels" / "val"
    previews = Path(out_dir)
    previews.mkdir(parents=True, exist_ok=True)
    files = list(img_dir.glob("*.jpg"))
    if not files:
        raise RuntimeError("No images found in " + str(img_dir))
    random.shuffle(files)
    files = files[:num]
    for f in files:
        lbl = lbl_dir / f.name.replace('.jpg', '.txt')
        out_p = previews / f.name
        draw_single(f, lbl, out_p)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo-dir", default="data/yolo")
    parser.add_argument("--out", default="data/previews")
    parser.add_argument("--num", type=int, default=20)
    args = parser.parse_args()
    make_previews(args.yolo_dir, args.out, args.num)


if __name__ == "__main__":
    main()
