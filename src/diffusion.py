from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline
from PIL import Image, ImageDraw, ImageOps
from tqdm import tqdm

from src.augment import copy_split, read_yolo_labels, write_dataset_yaml
from src.dataset import validate_yolo_dataset
from src.utils import ensure_dir, resolve_device, save_json


DIFFUSION_MODES = [
    "global_img2img",
    "inpaint_protected",
    "inpaint_reinsert",
]


def choose_torch_dtype(device: str):
    if device == "cuda":
        return torch.float16

    # Very important on Mac/MPS: float16 often causes black or broken images.
    return torch.float32


def make_generator(device: str, seed: int):
    generator_device = "cuda" if device == "cuda" else "cpu"
    return torch.Generator(device=generator_device).manual_seed(seed)


def yolo_labels_to_object_mask(
    labels,
    image_size: tuple[int, int],
    pixel_margin: int = 48,
    relative_margin: float = 3.0,
) -> Image.Image:
    """
    Object mask convention:
    white = protected object region
    black = editable background
    """
    width, height = image_size
    mask = np.zeros((height, width), dtype=np.uint8)

    for _, x, y, w, h in labels:
        box_w = w * width
        box_h = h * height

        margin_x = int(round(max(pixel_margin, box_w * relative_margin)))
        margin_y = int(round(max(pixel_margin, box_h * relative_margin)))

        x1 = int((x - w / 2) * width) - margin_x
        y1 = int((y - h / 2) * height) - margin_y
        x2 = int((x + w / 2) * width) + margin_x
        y2 = int((y + h / 2) * height) + margin_y

        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width, x2))
        y2 = max(0, min(height, y2))

        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 255

    return Image.fromarray(mask, mode="L")


def object_mask_to_inpaint_mask(object_mask: Image.Image) -> Image.Image:
    """
    Diffusers inpainting convention:
    white = repaint
    black = preserve

    Therefore we invert the protected object mask.
    """
    return ImageOps.invert(object_mask.convert("L"))


def reinsert_object_region(
    generated: Image.Image,
    original: Image.Image,
    object_mask: Image.Image,
) -> Image.Image:
    generated = generated.convert("RGB")
    original = original.convert("RGB")

    if generated.size != original.size:
        generated = generated.resize(original.size, Image.Resampling.LANCZOS)

    out = generated.copy()
    out.paste(original, (0, 0), object_mask.convert("L"))
    return out


def force_context_change_before_inpainting(
    original: Image.Image,
    object_mask: Image.Image,
    variant_index: int,
) -> Image.Image:
    """
    Force a night-time conditioning image before inpainting.

    The object region is preserved in the conditioning image.
    The editable background is transformed into a dark, blue, low-light
    night scene so that the diffusion model has a strong visual signal.
    """
    original_rgb = original.convert("RGB")
    arr = np.array(original_rgb).astype(np.float32)
    mask = np.array(object_mask.convert("L")) > 0

    # RGB channels.
    night = arr.copy()

    # Strong night transform:
    # reduce red/green, keep more blue, reduce global brightness.
    night[:, :, 0] = night[:, :, 0] * 0.12
    night[:, :, 1] = night[:, :, 1] * 0.18
    night[:, :, 2] = night[:, :, 2] * 0.38 + 18

    # Add slight low-light sensor noise.
    rng = np.random.default_rng(variant_index + 42)
    noise = rng.normal(loc=0.0, scale=5.0, size=night.shape)
    night = night + noise

    # Add subtle vertical moonlight / sensor gradient.
    height, width = night.shape[:2]
    y = np.linspace(1.15, 0.75, height).reshape(height, 1, 1)
    night = night * y

    # Add vignette to darken borders.
    yy, xx = np.mgrid[0:height, 0:width]
    cx, cy = width / 2, height / 2
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    dist = dist / dist.max()
    vignette = 1.0 - 0.35 * dist
    night = night * vignette[:, :, None]

    night = np.clip(night, 0, 255)

    conditioned = arr.copy()

    # Apply night transform only to editable background.
    # Keep the protected drone region unchanged.
    conditioned[~mask] = night[~mask]
    conditioned = np.clip(conditioned, 0, 255).astype(np.uint8)

    return Image.fromarray(conditioned)

def draw_boxes(image: Image.Image, labels) -> Image.Image:
    arr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    height, width = arr.shape[:2]

    for _, x, y, w, h in labels:
        x1 = int((x - w / 2) * width)
        y1 = int((y - h / 2) * height)
        x2 = int((x + w / 2) * width)
        y2 = int((y + h / 2) * height)

        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width - 1, x2))
        y2 = max(0, min(height - 1, y2))

        cv2.rectangle(arr, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


def overlay_mask(
    image: Image.Image,
    mask: Image.Image,
    color: tuple[int, int, int],
) -> Image.Image:
    image_arr = np.array(image.convert("RGB"))
    mask_arr = np.array(mask.convert("L"))

    overlay = image_arr.copy()
    overlay[mask_arr > 0] = color

    blended = cv2.addWeighted(image_arr, 0.60, overlay, 0.40, 0)
    return Image.fromarray(blended)


def image_pixel_stats(image: Image.Image) -> dict[str, float]:
    arr = np.array(image.convert("RGB"))
    return {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "nearly_black_pct": float(np.all(arr < 5, axis=2).mean() * 100.0),
    }


def is_suspicious_black_image(stats: dict[str, float]) -> bool:
    return stats["mean"] < 5 or stats["max"] < 30 or stats["std"] < 2


def compute_diff_stats(
    original: Image.Image,
    generated: Image.Image,
    object_mask: Image.Image,
) -> dict[str, float]:
    original_arr = np.array(original.convert("RGB")).astype(np.float32)
    generated_arr = np.array(generated.convert("RGB")).astype(np.float32)

    if generated_arr.shape != original_arr.shape:
        generated = generated.resize(original.size, Image.Resampling.LANCZOS)
        generated_arr = np.array(generated.convert("RGB")).astype(np.float32)

    object_mask_arr = np.array(object_mask.convert("L")) > 0
    background_mask_arr = ~object_mask_arr

    diff = np.abs(original_arr - generated_arr).mean(axis=2)

    object_diff = float(diff[object_mask_arr].mean()) if object_mask_arr.any() else 0.0
    background_diff = float(diff[background_mask_arr].mean()) if background_mask_arr.any() else 0.0

    return {
        "object_region_mean_abs_diff": object_diff,
        "background_region_mean_abs_diff": background_diff,
        "mask_coverage": float(object_mask_arr.mean()),
    }


class DiffusionBackend:
    def __init__(self, diffusion_config: dict[str, Any], device: str):
        self.config = diffusion_config
        self.device = device
        self.dtype = choose_torch_dtype(device)
        self.img2img_pipe = None
        self.inpaint_pipe = None

    def _setup_pipe(self, pipe):
        pipe = pipe.to(self.device)

        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()

        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()

        return pipe

    def load_img2img(self):
        if self.img2img_pipe is not None:
            return self.img2img_pipe

        model_name = self.config["img2img_model"]

        print(f"Loading img2img model: {model_name}")
        print(f"Device: {self.device}")
        print(f"Torch dtype: {self.dtype}")

        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )

        self.img2img_pipe = self._setup_pipe(pipe)
        return self.img2img_pipe

    def load_inpaint(self):
        if self.inpaint_pipe is not None:
            return self.inpaint_pipe

        model_name = self.config["inpaint_model"]

        print(f"Loading inpaint model: {model_name}")
        print(f"Device: {self.device}")
        print(f"Torch dtype: {self.dtype}")

        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )

        self.inpaint_pipe = self._setup_pipe(pipe)
        return self.inpaint_pipe

    def global_img2img(
        self,
        image: Image.Image,
        prompt: str,
        negative_prompt: str,
        strength: float,
        guidance_scale: float,
        steps: int,
        seed: int,
    ) -> Image.Image:
        pipe = self.load_img2img()
        generator = make_generator(self.device, seed)

        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image.convert("RGB"),
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            generator=generator,
        ).images[0].convert("RGB")

        if output.size != image.size:
            output = output.resize(image.size, Image.Resampling.LANCZOS)

        return output

    def inpaint(
        self,
        image: Image.Image,
        inpaint_mask: Image.Image,
        prompt: str,
        negative_prompt: str,
        strength: float,
        guidance_scale: float,
        steps: int,
        seed: int,
    ) -> Image.Image:
        pipe = self.load_inpaint()
        generator = make_generator(self.device, seed)

        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image.convert("RGB"),
            mask_image=inpaint_mask.convert("L"),
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            generator=generator,
        ).images[0].convert("RGB")

        if output.size != image.size:
            output = output.resize(image.size, Image.Resampling.LANCZOS)

        return output


def generate_with_mode(
    backend: DiffusionBackend,
    mode: str,
    original: Image.Image,
    conditioned_for_inpaint: Image.Image,
    object_mask: Image.Image,
    inpaint_mask: Image.Image,
    config: dict[str, Any],
    seed: int,
) -> tuple[Image.Image, bool, str]:
    diffusion_cfg = config["diffusion"]

    prompt = diffusion_cfg["prompt"]
    negative_prompt = diffusion_cfg["negative_prompt"]

    if mode == "global_img2img":
        generated = backend.global_img2img(
            image=original,
            prompt=prompt,
            negative_prompt=negative_prompt,
            strength=float(diffusion_cfg["global_strength"]),
            guidance_scale=float(diffusion_cfg["global_guidance_scale"]),
            steps=int(diffusion_cfg["global_steps"]),
            seed=seed,
        )
        return generated, False, "global image-to-image; object not explicitly protected"

    if mode == "inpaint_protected":
        generated = backend.inpaint(
            image=conditioned_for_inpaint,
            inpaint_mask=inpaint_mask,
            prompt=prompt,
            negative_prompt=negative_prompt,
            strength=float(diffusion_cfg["inpaint_strength"]),
            guidance_scale=float(diffusion_cfg["inpaint_guidance_scale"]),
            steps=int(diffusion_cfg["inpaint_steps"]),
            seed=seed,
        )
        return generated, False, "background inpainting with protected object mask"

    if mode == "inpaint_reinsert":
        generated_before_reinsert = backend.inpaint(
            image=conditioned_for_inpaint,
            inpaint_mask=inpaint_mask,
            prompt=prompt,
            negative_prompt=negative_prompt,
            strength=float(diffusion_cfg["inpaint_strength"]),
            guidance_scale=float(diffusion_cfg["inpaint_guidance_scale"]),
            steps=int(diffusion_cfg["inpaint_steps"]),
            seed=seed,
        )

        generated_after_reinsert = reinsert_object_region(
            generated=generated_before_reinsert,
            original=original,
            object_mask=object_mask,
        )

        return generated_after_reinsert, True, "background inpainting followed by exact object reinsertion"

    raise ValueError(f"Unknown diffusion mode: {mode}")


def make_single_image_grid(
    original: Image.Image,
    labels,
    object_mask: Image.Image,
    inpaint_mask: Image.Image,
    conditioned_for_inpaint: Image.Image,
    outputs: dict[str, Image.Image],
    output_path: Path,
) -> None:
    ensure_dir(output_path.parent)

    tiles = [
        ("original", original),
        ("original + box", draw_boxes(original, labels)),
        ("protected object mask", overlay_mask(original, object_mask, (255, 40, 40))),
        ("editable inpaint mask", overlay_mask(original, inpaint_mask, (255, 120, 40))),
        ("inpaint condition", conditioned_for_inpaint),
    ]

    for mode in DIFFUSION_MODES:
        tiles.append((mode, draw_boxes(outputs[mode], labels)))

    tile_w = 256
    tile_h = 256
    caption_h = 40

    canvas = Image.new(
        "RGB",
        (tile_w * len(tiles), tile_h + caption_h),
        (245, 245, 245),
    )

    draw = ImageDraw.Draw(canvas)

    for i, (caption, tile) in enumerate(tiles):
        x = i * tile_w
        tile = tile.convert("RGB").resize((tile_w, tile_h), Image.Resampling.LANCZOS)
        canvas.paste(tile, (x, caption_h))
        draw.text((x + 8, 10), caption, fill=(0, 0, 0))

    canvas.save(output_path, quality=95)


def select_source_images(
    baseline_root: Path,
    start_index: int,
    max_images: int,
) -> list[Path]:
    all_images = sorted((baseline_root / "images" / "train").glob("*.jpg"))

    if start_index >= len(all_images):
        raise ValueError(
            f"start_index={start_index} is too large. Dataset has {len(all_images)} train images."
        )

    return all_images[start_index:start_index + max_images]


def make_diffusion_grid(
    config: dict[str, Any],
    overwrite: bool = False,
    max_images: int | None = None,
    start_index: int = 0,
) -> dict[str, Any]:
    baseline_root = Path(config["experiments"]["baseline"]["dataset_root"])
    output_root = Path(config["diffusion"]["output_root"])

    reports_dir = ensure_dir(config["paths"]["reports"])
    tables_dir = ensure_dir(config["paths"]["tables"])
    previews_dir = ensure_dir(config["paths"]["previews"])
    grid_dir = ensure_dir(previews_dir / "diffusion_grid")

    max_images = int(max_images or config["diffusion"]["max_images"])
    device = resolve_device(config["yolo"]["device"])

    if output_root.exists():
        if overwrite:
            shutil.rmtree(output_root)
        else:
            raise FileExistsError(f"Output dataset already exists: {output_root}. Use --overwrite.")

    for split in ["train", "val", "test"]:
        ensure_dir(output_root / "images" / split)
        ensure_dir(output_root / "labels" / split)

    copied_train = copy_split(baseline_root, output_root, "train")
    copied_val = copy_split(baseline_root, output_root, "val")
    copied_test = copy_split(baseline_root, output_root, "test")

    backend = DiffusionBackend(
        diffusion_config=config["diffusion"],
        device=device,
    )

    source_images = select_source_images(
        baseline_root=baseline_root,
        start_index=start_index,
        max_images=max_images,
    )
    source_label_dir = baseline_root / "labels" / "train"

    print("Selected source images:")
    for image_path in source_images:
        print(f"- {image_path}")

    rows = []

    for local_index, image_path in enumerate(tqdm(source_images, desc="Generating diffusion grid")):
        global_index = start_index + local_index
        label_path = source_label_dir / f"{image_path.stem}.txt"
        labels = read_yolo_labels(label_path)

        if not labels:
            continue

        original = Image.open(image_path).convert("RGB")

        object_mask = yolo_labels_to_object_mask(
            labels=labels,
            image_size=original.size,
            pixel_margin=int(config["diffusion"]["box_margin_px"]),
            relative_margin=float(config["diffusion"]["box_margin_ratio"]),
        )

        inpaint_mask = object_mask_to_inpaint_mask(object_mask)

        conditioned_for_inpaint = force_context_change_before_inpainting(
            original=original,
            object_mask=object_mask,
            variant_index=global_index,
        )

        outputs = {}

        for mode in DIFFUSION_MODES:
            seed = int(config["yolo"]["seed"]) + global_index * 100 + DIFFUSION_MODES.index(mode)

            generated, reinsertion_used, strategy = generate_with_mode(
                backend=backend,
                mode=mode,
                original=original,
                conditioned_for_inpaint=conditioned_for_inpaint,
                object_mask=object_mask,
                inpaint_mask=inpaint_mask,
                config=config,
                seed=seed,
            )

            output_stem = f"diffusion_{mode}_{global_index:03d}_{image_path.stem}"
            output_image_path = output_root / "images" / "train" / f"{output_stem}.jpg"
            output_label_path = output_root / "labels" / "train" / f"{output_stem}.txt"
            metadata_path = output_root / "metadata" / f"{output_stem}.json"

            ensure_dir(output_image_path.parent)
            ensure_dir(output_label_path.parent)
            ensure_dir(metadata_path.parent)

            generated.save(output_image_path, quality=95)
            shutil.copy2(label_path, output_label_path)

            stats = image_pixel_stats(generated)
            diff_stats = compute_diff_stats(
                original=original,
                generated=generated,
                object_mask=object_mask,
            )

            metadata = {
                "source_image": str(image_path),
                "source_label": str(label_path),
                "generated_image": str(output_image_path),
                "generated_label": str(output_label_path),
                "mode": mode,
                "strategy": strategy,
                "reinsertion_used": reinsertion_used,
                "device": device,
                "torch_dtype": str(choose_torch_dtype(device)),
                "seed": seed,
                "start_index": start_index,
                "global_index": global_index,
                "output_pixel_stats": stats,
                "suspicious_black_image": is_suspicious_black_image(stats),
                **diff_stats,
            }

            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

            rows.append(metadata)
            outputs[mode] = generated

            print(
                f"\n{mode} | "
                f"background_diff={metadata['background_region_mean_abs_diff']:.2f} | "
                f"object_diff={metadata['object_region_mean_abs_diff']:.2f} | "
                f"black={metadata['suspicious_black_image']}"
            )

        grid_path = grid_dir / f"diffusion_grid_{global_index:03d}_{image_path.stem}.jpg"
        make_single_image_grid(
            original=original,
            labels=labels,
            object_mask=object_mask,
            inpaint_mask=inpaint_mask,
            conditioned_for_inpaint=conditioned_for_inpaint,
            outputs=outputs,
            output_path=grid_path,
        )

    dataset_yaml = write_dataset_yaml(output_root)
    validation_report = validate_yolo_dataset(output_root)

    table_path = tables_dir / "diffusion_grid_index.csv"
    pd.DataFrame(rows).to_csv(table_path, index=False)

    summary_by_mode = {}
    df = pd.DataFrame(rows)

    if not df.empty:
        for mode in DIFFUSION_MODES:
            mode_df = df[df["mode"] == mode]
            summary_by_mode[mode] = {
                "count": int(len(mode_df)),
                "mean_background_diff": float(mode_df["background_region_mean_abs_diff"].mean()),
                "mean_object_diff": float(mode_df["object_region_mean_abs_diff"].mean()),
                "num_black_images": int(mode_df["suspicious_black_image"].sum()),
            }

    report = {
        "type": "three_mode_diffusion_grid",
        "modes": DIFFUSION_MODES,
        "start_index": start_index,
        "max_images": max_images,
        "input_root": str(baseline_root),
        "output_root": str(output_root),
        "dataset_yaml": str(dataset_yaml),
        "copied_real_train": copied_train,
        "copied_val": copied_val,
        "copied_test": copied_test,
        "generated_train": len(rows),
        "device": device,
        "torch_dtype": str(choose_torch_dtype(device)),
        "summary_by_mode": summary_by_mode,
        "validation": validation_report,
        "table": str(table_path),
        "grid_dir": str(grid_dir),
    }

    save_json(report, reports_dir / "diffusion_grid_report.json")

    print("\nDiffusion grid completed.")
    print(f"Grid directory: {grid_dir}")
    print(f"Generated diffusion images: {len(rows)}")
    print(f"Validation valid: {validation_report['is_valid']}")
    print("Summary by mode:")
    print(json.dumps(summary_by_mode, indent=2))

    return report
