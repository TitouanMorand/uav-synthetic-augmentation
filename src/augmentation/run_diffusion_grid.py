"""Generate a diffusion augmentation grid from a YOLO training directory."""

import argparse
import itertools
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import yaml
from PIL import Image

from src.augmentation.diffusion_backend import DiffusionBackend, DiffusionBackendConfig
from src.augmentation.masks_from_boxes import (
    background_inpaint_mask_from_labels,
    object_mask_from_labels,
    read_yolo_boxes,
)
from src.augmentation.object_reinsert import reinsert_with_mask
from src.augmentation.prompts import NEGATIVE_PROMPT, build_positive_prompt

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
DEFAULT_MODES = [
    "global_img2img",
    "background_inpaint_protected_box",
    "background_inpaint_reinsert_object",
]


def parse_csv_values(value: str, cast=str) -> list:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Diffusion config not found: {path}")
    return yaml.safe_load(path.read_text()) or {}


def list_train_images(image_dir: Path) -> list[Path]:
    return sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def output_stem(source: Path, preset: str, mode: str, strength: float, guidance: float, seed: int) -> str:
    strength_s = str(strength).replace(".", "p")
    guidance_s = str(guidance).replace(".", "p")
    return f"{source.stem}__{preset}__{mode}__s{strength_s}__g{guidance_s}__seed{seed}"


def copy_label(label_path: Path, output_label_path: Path) -> None:
    output_label_path.parent.mkdir(parents=True, exist_ok=True)
    if not label_path.exists():
        raise FileNotFoundError(f"Missing source label: {label_path}")
    shutil.copy2(label_path, output_label_path)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def generate_one(
    backend: DiffusionBackend,
    image_path: Path,
    label_path: Path,
    out_root: Path,
    preset: str,
    mode: str,
    strength: float,
    guidance: float,
    seed: int,
    positive_prompt: str,
    negative_prompt: str,
    pixel_margin: int,
    relative_margin: float,
) -> dict:
    image = Image.open(image_path).convert("RGB")
    size = image.size
    boxes = read_yolo_boxes(label_path, size)
    if not boxes:
        raise ValueError(f"No YOLO boxes found for {image_path}")

    stem = output_stem(image_path, preset, mode, strength, guidance, seed)
    mode_dir = out_root / mode
    output_image_path = mode_dir / "images" / "train" / f"{stem}.jpg"
    output_label_path = mode_dir / "labels" / "train" / f"{stem}.txt"
    output_meta_path = mode_dir / "metadata" / f"{stem}.json"
    output_image_path.parent.mkdir(parents=True, exist_ok=True)

    inpaint_mask = None
    object_mask = object_mask_from_labels(label_path, size, pixel_margin, relative_margin)
    reinsertion_used = False
    model_id = backend.config.img2img_model_id

    if mode == "global_img2img":
        generated = backend.generate_img2img(
            image=image,
            prompt=positive_prompt,
            negative_prompt=negative_prompt,
            strength=strength,
            guidance_scale=guidance,
            seed=seed,
        )
    elif mode in {"background_inpaint_protected_box", "background_inpaint_reinsert_object"}:
        model_id = backend.config.inpaint_model_id
        inpaint_mask = background_inpaint_mask_from_labels(label_path, size, pixel_margin, relative_margin)
        generated = backend.generate_inpaint(
            image=image,
            mask=inpaint_mask,
            prompt=positive_prompt,
            negative_prompt=negative_prompt,
            strength=strength,
            guidance_scale=guidance,
            seed=seed,
        )
        if mode == "background_inpaint_reinsert_object":
            generated = reinsert_with_mask(generated, image, boxes, object_mask)
            reinsertion_used = True
    else:
        raise ValueError(f"Unsupported generation mode: {mode}")

    if generated.size != size:
        raise RuntimeError(f"Generated size {generated.size} does not match source size {size}")

    generated.save(output_image_path, quality=95)
    copy_label(label_path, output_label_path)

    metadata = {
        "source_image_path": str(image_path),
        "source_label_path": str(label_path),
        "output_image_path": str(output_image_path),
        "output_label_path": str(output_label_path),
        "prompt_preset": preset,
        "positive_prompt": positive_prompt,
        "negative_prompt": negative_prompt,
        "generation_mode": mode,
        "model_id": model_id,
        "strength": strength,
        "guidance_scale": guidance,
        "seed": seed,
        "original_image_size": {"width": size[0], "height": size[1]},
        "reinsertion_used": reinsertion_used,
        "box_coordinates_xyxy": [box.__dict__ for box in boxes],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output_meta_path, metadata)
    return metadata


def parse_args():
    parser = argparse.ArgumentParser(description="Generate diffusion augmentations over a YOLO train split.")
    parser.add_argument("--config", default="configs/diffusion.yaml")
    parser.add_argument("--yolo-dir", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--prompt-presets", default="night_lowlight")
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES))
    parser.add_argument("--strength-values", default=None)
    parser.add_argument("--guidance-values", default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--extra-prompt", default=None)
    parser.add_argument("--pixel-margin", type=int, default=None)
    parser.add_argument("--relative-margin", type=float, default=None)
    parser.add_argument("--manifest-name", default="manifest.jsonl")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(Path(args.config))
    paths_cfg = cfg.get("paths", {})
    gen_cfg = cfg.get("generation", {})
    models_cfg = cfg.get("models", {})
    masks_cfg = cfg.get("masks", {})

    yolo_dir = Path(args.yolo_dir or paths_cfg.get("source_yolo_dir", "data/yolo"))
    out_root = Path(args.output_root or paths_cfg.get("output_root", "data/synthetic/diffusion_grid"))
    limit = args.limit if args.limit is not None else int(gen_cfg.get("subset_size", 20))
    presets = parse_csv_values(args.prompt_presets, str)
    modes = parse_csv_values(args.modes, str)
    strengths = parse_csv_values(args.strength_values, float) if args.strength_values else gen_cfg.get("strength_values", [0.35])
    guidance_values = (
        parse_csv_values(args.guidance_values, float) if args.guidance_values else gen_cfg.get("guidance_values", [7.5])
    )
    seeds = parse_csv_values(args.seeds, int) if args.seeds else [int(gen_cfg.get("default_seed", 42))]
    pixel_margin = args.pixel_margin if args.pixel_margin is not None else int(masks_cfg.get("pixel_margin", 16))
    relative_margin = (
        args.relative_margin if args.relative_margin is not None else float(masks_cfg.get("relative_margin", 0.15))
    )

    image_dir = yolo_dir / "images" / "train"
    label_dir = yolo_dir / "labels" / "train"
    images = list_train_images(image_dir)[:limit]
    if not images:
        raise RuntimeError(f"No train images found in {image_dir}")

    backend = DiffusionBackend(
        DiffusionBackendConfig(
            img2img_model_id=models_cfg.get("img2img", "runwayml/stable-diffusion-v1-5"),
            inpaint_model_id=models_cfg.get("inpaint", "runwayml/stable-diffusion-inpainting"),
            device_preference=args.device or cfg.get("device", {}).get("preference", "auto"),
        )
    )
    print(f"Diffusion device: {backend.device}")
    print(f"Generating {len(images)} source images into {out_root}")

    manifest_path = out_root / args.manifest_name
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a") as manifest:
        total = len(images) * len(presets) * len(modes) * len(strengths) * len(guidance_values) * len(seeds)
        index = 0
        for image_path, preset, mode, strength, guidance, seed in itertools.product(
            images, presets, modes, strengths, guidance_values, seeds
        ):
            index += 1
            label_path = label_dir / f"{image_path.stem}.txt"
            prompt = build_positive_prompt(preset, args.extra_prompt)
            print(f"[{index}/{total}] {image_path.name} {preset} {mode} s={strength} g={guidance} seed={seed}")
            metadata = generate_one(
                backend,
                image_path,
                label_path,
                out_root,
                preset,
                mode,
                strength,
                guidance,
                seed,
                prompt,
                NEGATIVE_PROMPT,
                pixel_margin,
                relative_margin,
            )
            manifest.write(json.dumps(metadata) + "\n")
            manifest.flush()

    print(f"Manifest written: {manifest_path}")


if __name__ == "__main__":
    main()
