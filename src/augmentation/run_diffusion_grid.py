"""Generate a diffusion augmentation grid from a YOLO training directory."""

import argparse
import itertools
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import yaml
from PIL import Image
import numpy as np

from src.augmentation.diffusion_backend import DiffusionBackend, DiffusionBackendConfig
from src.augmentation.masks_from_boxes import (
    DEFAULT_BOX_MARGIN_PX,
    DEFAULT_BOX_MARGIN_RATIO,
    background_inpaint_mask_from_labels,
    object_mask_from_labels,
    read_yolo_boxes,
    save_mask_debug_previews,
)
from src.augmentation.object_reinsert import reinsert_with_mask
from src.augmentation.prompts import NEGATIVE_PROMPT, build_positive_prompt
from src.evaluation.preservation_metrics import compute_preservation_metrics, rejection_reasons

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


def image_pixel_stats(image: Image.Image) -> dict:
    arr = np.array(image.convert("RGB"))
    return {
        "dtype": str(arr.dtype),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "nearly_black_pct": float(np.all(arr < 5, axis=2).mean() * 100.0),
    }


def is_suspicious_black_image(stats: dict) -> bool:
    return stats["mean"] < 5 or stats["max"] < 30 or stats["std"] < 2


def dtype_name(dtype) -> str:
    return str(dtype).replace("torch.", "")


def enforce_requested_device(requested: str | None, actual: str, require_device: bool) -> None:
    if require_device and requested and requested != "auto" and actual != requested:
        raise RuntimeError(f"Requested device '{requested}' but diffusion backend selected '{actual}'.")


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
    protect_all_boxes: bool,
) -> dict:
    image = Image.open(image_path).convert("RGB")
    size = image.size
    boxes = read_yolo_boxes(label_path, size)
    if not boxes:
        raise ValueError(f"No YOLO boxes found for {image_path}")

    stem = output_stem(image_path, preset, mode, strength, guidance, seed)
    mode_dir = out_root / mode
    output_image_path = mode_dir / "images" / "train" / f"{stem}.jpg"
    before_reinsert_path = mode_dir / "debug" / "generated_before_reinsertion" / f"{stem}.jpg"
    output_label_path = mode_dir / "labels" / "train" / f"{stem}.txt"
    output_meta_path = mode_dir / "metadata" / f"{stem}.json"
    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    before_reinsert_path.parent.mkdir(parents=True, exist_ok=True)

    object_mask = object_mask_from_labels(label_path, size, pixel_margin, relative_margin, protect_all_boxes)
    inpaint_mask = background_inpaint_mask_from_labels(label_path, size, pixel_margin, relative_margin, protect_all_boxes)
    mask_paths = save_mask_debug_previews(image, object_mask, inpaint_mask, mode_dir / "masks", stem)
    reinsertion_used = False
    model_id = backend.config.img2img_model_id
    pipeline_used = "StableDiffusionImg2ImgPipeline"

    print(f"  source image: {image_path}")
    print(f"  source mode/size: {image.mode}/{image.size}")
    print(f"  device/dtype: {backend.device}/{dtype_name(backend.dtype)}")
    print(f"  prompt: {positive_prompt}")
    print(f"  strength/guidance: {strength}/{guidance}")

    if mode == "global_img2img":
        print(f"  pipeline: {pipeline_used}")
        print("  WARNING: global_img2img is a naive qualitative baseline and may alter labeled objects.")
        generated = backend.generate_img2img(
            image=image,
            prompt=positive_prompt,
            negative_prompt=negative_prompt,
            strength=strength,
            guidance_scale=guidance,
            seed=seed,
        )
        generated_before_reinsertion = generated.convert("RGB")
    elif mode in {"background_inpaint_protected_box", "background_inpaint_reinsert_object"}:
        model_id = backend.config.inpaint_model_id
        pipeline_used = "StableDiffusionInpaintPipeline"
        print(f"  pipeline: {pipeline_used}")
        generated = backend.generate_inpaint(
            image=image,
            mask=inpaint_mask,
            prompt=positive_prompt,
            negative_prompt=negative_prompt,
            strength=strength,
            guidance_scale=guidance,
            seed=seed,
        )
        generated_before_reinsertion = generated.convert("RGB")
        if mode == "background_inpaint_reinsert_object":
            generated = reinsert_with_mask(generated, image, boxes, object_mask)
            reinsertion_used = True
    else:
        raise ValueError(f"Unsupported generation mode: {mode}")

    if generated.size != size:
        raise RuntimeError(f"Generated size {generated.size} does not match source size {size}")

    generated = generated.convert("RGB")
    generated_before_reinsertion.save(before_reinsert_path, quality=95)
    output_stats = image_pixel_stats(generated)
    preservation_metrics = compute_preservation_metrics(
        original=image,
        generated_before_reinsertion=generated_before_reinsertion,
        generated_after_reinsertion=generated,
        boxes=boxes,
        protection_mask=object_mask,
        mode=mode,
    )
    reasons = rejection_reasons(
        metrics=preservation_metrics,
        output_stats=output_stats,
        size_matches=generated.size == size,
    )
    suspicious = is_suspicious_black_image(output_stats)
    accepted_for_training = not reasons
    print(f"  output mode/size: {generated.mode}/{generated.size}")
    print(
        "  output pixels before saving: "
        f"min={output_stats['min']:.1f} max={output_stats['max']:.1f} "
        f"mean={output_stats['mean']:.2f} std={output_stats['std']:.2f} "
        f"nearly_black={output_stats['nearly_black_pct']:.2f}%"
    )
    if suspicious:
        print(f"WARNING: suspicious near-black generated image: {output_image_path}")
    if reasons:
        print(f"WARNING: rejected generated image: {', '.join(reasons)}")

    generated.save(output_image_path, quality=95)
    if accepted_for_training:
        copy_label(label_path, output_label_path)

    metadata = {
        "source_image_path": str(image_path),
        "source_label_path": str(label_path),
        "output_image_path": str(output_image_path),
        "output_label_path": str(output_label_path),
        "generated_before_reinsertion_path": str(before_reinsert_path),
        **mask_paths,
        "prompt_preset": preset,
        "positive_prompt": positive_prompt,
        "negative_prompt": negative_prompt,
        "generation_mode": mode,
        "model_id": model_id,
        "pipeline_used": pipeline_used,
        "device": backend.device,
        "dtype": dtype_name(backend.dtype),
        "strength": strength,
        "guidance_scale": guidance,
        "seed": seed,
        "original_image_size": {"width": size[0], "height": size[1]},
        "output_image_mode": generated.mode,
        "output_image_size": {"width": generated.size[0], "height": generated.size[1]},
        "output_pixel_stats": output_stats,
        "is_suspicious_black_image": suspicious,
        "preservation_metrics": preservation_metrics,
        "accepted_for_training": accepted_for_training,
        "rejection_reasons": reasons,
        "reinsertion_used": reinsertion_used,
        "box_coordinates_xyxy": [box.__dict__ for box in boxes],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output_meta_path, metadata)
    return metadata


def save_debug_comparison(original: Image.Image, generated: Image.Image, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    original_rgb = original.convert("RGB")
    generated_rgb = generated.convert("RGB")
    original_rgb.save(out_dir / "original.jpg", quality=95)
    generated_rgb.save(out_dir / "generated.jpg", quality=95)

    width = original_rgb.width + generated_rgb.width
    height = max(original_rgb.height, generated_rgb.height)
    comparison = Image.new("RGB", (width, height), (24, 24, 24))
    comparison.paste(original_rgb, (0, 0))
    comparison.paste(generated_rgb, (original_rgb.width, 0))
    comparison.save(out_dir / "side_by_side.jpg", quality=95)
    print(f"Debug images written to {out_dir}")


def run_debug_single(args, cfg: dict) -> None:
    paths_cfg = cfg.get("paths", {})
    models_cfg = cfg.get("models", {})
    yolo_dir = Path(args.yolo_dir or paths_cfg.get("source_yolo_dir", "data/yolo"))
    image_dir = yolo_dir / "images" / "train"
    images = list_train_images(image_dir)
    if not images:
        raise RuntimeError(f"No train images found in {image_dir}")

    backend = DiffusionBackend(
        DiffusionBackendConfig(
            img2img_model_id=models_cfg.get("img2img", "runwayml/stable-diffusion-v1-5"),
            inpaint_model_id=models_cfg.get("inpaint", "runwayml/stable-diffusion-inpainting"),
            device_preference=args.device or cfg.get("device", {}).get("preference", "auto"),
        )
    )
    enforce_requested_device(args.device, backend.device, args.require_device)
    image_path = images[0]
    image = Image.open(image_path).convert("RGB")
    prompt = build_positive_prompt("night_lowlight", args.extra_prompt)
    print("Running debug-single generation")
    print(f"  source image: {image_path}")
    print(f"  source mode/size: {image.mode}/{image.size}")
    print(f"  pipeline: StableDiffusionImg2ImgPipeline")
    print(f"  device/dtype: {backend.device}/{dtype_name(backend.dtype)}")
    print(f"  prompt: {prompt}")
    print("  strength/guidance: 0.35/7.5")
    generated = backend.generate_img2img(
        image=image,
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        strength=0.35,
        guidance_scale=7.5,
        seed=42,
    ).convert("RGB")
    if generated.size != image.size:
        raise RuntimeError(f"Generated size {generated.size} does not match source size {image.size}")
    stats = image_pixel_stats(generated)
    print(f"  output mode/size: {generated.mode}/{generated.size}")
    print(
        "  output pixels before saving: "
        f"min={stats['min']:.1f} max={stats['max']:.1f} "
        f"mean={stats['mean']:.2f} std={stats['std']:.2f} "
        f"nearly_black={stats['nearly_black_pct']:.2f}%"
    )
    if is_suspicious_black_image(stats):
        print("WARNING: debug-single generated image is suspiciously near-black.")
    save_debug_comparison(image, generated, Path("data/previews/debug_single"))


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
    parser.add_argument("--require-device", action="store_true")
    parser.add_argument("--extra-prompt", default=None)
    parser.add_argument("--pixel-margin", type=int, default=None)
    parser.add_argument("--relative-margin", type=float, default=None)
    parser.add_argument("--box-margin-px", type=int, default=None)
    parser.add_argument("--box-margin-ratio", type=float, default=None)
    parser.add_argument("--protect-all-boxes", dest="protect_all_boxes", action="store_true", default=None)
    parser.add_argument("--no-protect-all-boxes", dest="protect_all_boxes", action="store_false")
    parser.add_argument("--manifest-name", default="manifest.jsonl")
    parser.add_argument("--debug-single", action="store_true", help="Generate one fixed img2img debug sample")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(Path(args.config))
    if args.debug_single:
        run_debug_single(args, cfg)
        return
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
    pixel_margin = args.box_margin_px
    if pixel_margin is None:
        pixel_margin = args.pixel_margin
    if pixel_margin is None:
        pixel_margin = int(masks_cfg.get("box_margin_px", masks_cfg.get("pixel_margin", DEFAULT_BOX_MARGIN_PX)))
    relative_margin = args.box_margin_ratio
    if relative_margin is None:
        relative_margin = args.relative_margin
    if relative_margin is None:
        relative_margin = float(
            masks_cfg.get("box_margin_ratio", masks_cfg.get("relative_margin", DEFAULT_BOX_MARGIN_RATIO))
        )
    protect_all_boxes = bool(
        args.protect_all_boxes if args.protect_all_boxes is not None else masks_cfg.get("protect_all_boxes", True)
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
    enforce_requested_device(args.device, backend.device, args.require_device)
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
                protect_all_boxes,
            )
            manifest.write(json.dumps(metadata) + "\n")
            manifest.flush()

    print(f"Manifest written: {manifest_path}")


if __name__ == "__main__":
    main()
