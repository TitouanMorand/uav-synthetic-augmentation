"""
Simple classical night/low-light augmentation that preserves labels.

This file implements a conservative augmentation: reduce brightness, shift color temperature,
and optionally add a slight vignette. Labels are kept unchanged because transforms are global.

TODOs: placeholders for diffusion img2img, object-protected masks, and reinsertion.
"""
import argparse
from pathlib import Path
import cv2
import numpy as np
import shutil


def apply_night(img: np.ndarray, darkness: float = 0.4, blue_shift: float = 15) -> np.ndarray:
    """
    Apply a simple night effect:
    - scale V channel in HSV by `darkness` (0..1)
    - add blue tint by shifting B channel
    """
    img = img.copy()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    # reduce brightness
    hsv[..., 2] = hsv[..., 2] * darkness
    hsv[..., 2] = np.clip(hsv[..., 2], 0, 255)
    img2 = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    # add slight blue tint
    b, g, r = cv2.split(img2)
    blue_shift = int(np.clip(blue_shift, 0, 255))
    b = cv2.add(b, blue_shift)
    out = cv2.merge([b, g, r])
    # optional mild Gaussian blur to simulate low-light softness
    out = cv2.GaussianBlur(out, (3, 3), 0)
    return out


def augment_yolo_dir(yolo_dir: str, out_dir: str, method: str = "night", prefix: str = "aug"):
    src = Path(yolo_dir)
    dst = Path(out_dir)
    # copy tree structure
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    # For each image in images/train and images/val, create augmented image with same label
    for split in ["train", "val"]:
        img_dir = src / "images" / split
        lbl_dir = src / "labels" / split
        out_img_dir = dst / "images" / split
        out_img_dir.mkdir(parents=True, exist_ok=True)
        for img_path in img_dir.glob("*.jpg"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            if method == "night":
                out_img = apply_night(img)
            else:
                out_img = img
            out_name = f"{prefix}_{img_path.name}"
            cv2.imwrite(str(out_img_dir / out_name), out_img)
            # copy corresponding label to new name
            src_lbl = lbl_dir / (img_path.name.replace('.jpg', '.txt'))
            dst_lbl = dst / "labels" / split / (out_name.replace('.jpg', '.txt'))
            if src_lbl.exists():
                shutil.copy(src_lbl, dst_lbl)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--method", default="night", choices=["night"])
    args = parser.parse_args()
    augment_yolo_dir(args.yolo_dir, args.out, args.method)


if __name__ == "__main__":
    main()
