from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.augment import read_yolo_labels
from src.config import load_config
from src.diffusion import DiffusionBackend
from src.diffusion_v2 import (
    apply_lab_delta_to_object,
    composite_with_alpha,
    draw_boxes,
    erode_mask,
    make_feather_alpha,
    make_night_condition,
    make_object_matte,
    overlay_mask,
    preview_alpha,
    quality_metadata,
)
from src.utils import ensure_dir, resolve_device


# -----------------------------
# Helpers
# -----------------------------
def run_img2img_backend(
    backend,
    image,
    prompt: str,
    negative_prompt: str,
    strength: float,
    guidance_scale: float,
    steps: int,
    seed: int,
):
    if hasattr(backend, "img2img"):
        return backend.img2img(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            strength=strength,
            guidance_scale=guidance_scale,
            steps=steps,
            seed=seed,
        )

    if hasattr(backend, "global_img2img"):
        return backend.global_img2img(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            strength=strength,
            guidance_scale=guidance_scale,
            steps=steps,
            seed=seed,
        )

    if hasattr(backend, "generate_img2img"):
        return backend.generate_img2img(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            strength=strength,
            guidance_scale=guidance_scale,
            steps=steps,
            seed=seed,
        )

    available = [name for name in dir(backend) if "img" in name.lower() or "paint" in name.lower()]
    raise AttributeError(
        "No compatible img2img method found on DiffusionBackend. "
        f"Available image-related methods: {available}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate readable V2.2 diffusion LAB-delta grids.")
    parser.add_argument("--max-images", type=int, default=8)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--presets", nargs="+", default=["conservative", "medium"])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def yolo_to_xyxy(label, width, height):
    cls_id, x_c, y_c, w, h = label
    x1 = int((x_c - w / 2) * width)
    y1 = int((y_c - h / 2) * height)
    x2 = int((x_c + w / 2) * width)
    y2 = int((y_c + h / 2) * height)
    return max(0, x1), max(0, y1), min(width - 1, x2), min(height - 1, y2)


def crop_around_box(image: Image.Image, label, pad_factor: float = 8.0, min_size: int = 160) -> Image.Image:
    w, h = image.size
    x1, y1, x2, y2 = yolo_to_xyxy(label, w, h)

    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    side = int(max(min_size, max(bw, bh) * pad_factor))

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    left = max(0, cx - side // 2)
    top = max(0, cy - side // 2)
    right = min(w, left + side)
    bottom = min(h, top + side)

    # readjust if clipped
    left = max(0, right - side)
    top = max(0, bottom - side)

    crop = image.crop((left, top, right, bottom))
    return crop


def annotate_tile(image: Image.Image, title: str, tile_size=(260, 260), title_h: int = 28) -> Image.Image:
    img = image.copy().convert("RGB")
    img.thumbnail(tile_size, Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", (tile_size[0], tile_size[1] + title_h), "white")
    x = (tile_size[0] - img.width) // 2
    y = title_h + (tile_size[1] - img.height) // 2
    canvas.paste(img, (x, y))

    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, tile_size[0], title_h], fill=(240, 240, 240))
    draw.text((6, 6), title, fill="black")
    return canvas


def make_grid(rows, output_path: Path, tile_size=(260, 260), title_h: int = 28, row_gap: int = 10, col_gap: int = 10):
    prepared_rows = []
    max_cols = max(len(row) for row in rows)

    for row in rows:
        prepared = [annotate_tile(img, title, tile_size=tile_size, title_h=title_h) for title, img in row]
        prepared_rows.append(prepared)

    cell_w = tile_size[0]
    cell_h = tile_size[1] + title_h

    grid_w = max_cols * cell_w + (max_cols - 1) * col_gap
    grid_h = len(prepared_rows) * cell_h + (len(prepared_rows) - 1) * row_gap

    canvas = Image.new("RGB", (grid_w, grid_h), "white")

    for r, row in enumerate(prepared_rows):
        for c, tile in enumerate(row):
            x = c * (cell_w + col_gap)
            y = r * (cell_h + row_gap)
            canvas.paste(tile, (x, y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=95)


def first_label(labels):
    return labels[0]


def build_zoom_row(row):
    zoom_row = []
    for title, img, label in row:
        crop = crop_around_box(img, label, pad_factor=10.0, min_size=180)
        zoom_row.append((title, crop))
    return zoom_row


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    args = parse_args()

    project_cfg = load_config()
    v2_cfg = yaml.safe_load(Path("configs/diffusion_v2.yaml").read_text())["diffusion_v2"]

    accepted_sources_path = Path(project_cfg["paths"]["tables"]) / "diffusion_v2_source_accepted.csv"
    if not accepted_sources_path.exists():
        raise FileNotFoundError(
            f"Missing accepted source file: {accepted_sources_path}. "
            f"Run scripts/15_prepare_diffusion_sources.py first."
        )

    sources = pd.read_csv(accepted_sources_path)
    sources = sources.iloc[args.start:args.start + args.max_images].copy()

    output_root = Path("data/augmented/diffusion_v2_grid")
    previews_dir = ensure_dir(Path(project_cfg["paths"]["previews"]) / "diffusion_v2_grid")
    tables_dir = ensure_dir(project_cfg["paths"]["tables"])

    if output_root.exists() and args.overwrite:
        import shutil
        shutil.rmtree(output_root)

    ensure_dir(output_root)

    device = resolve_device(project_cfg["yolo"]["device"])

    backend = DiffusionBackend(
        diffusion_config={"img2img_model": v2_cfg["img2img_model"]},
        device=device,
    )

    prompt = v2_cfg["prompt"]["positive"]
    negative_prompt = v2_cfg["prompt"]["negative"]
    lab_cfg = v2_cfg["lab_delta"]

    rows = []

    for _, row in tqdm(list(sources.iterrows()), desc="V2.2 diffusion LAB delta grid"):
        image_path = Path(row["image_path"])
        label_path = Path(row["label_path"])

        labels = read_yolo_labels(label_path)
        if not labels:
            continue

        label0 = first_label(labels)
        original = Image.open(image_path).convert("RGB")

        object_matte = make_object_matte(original, labels)
        object_overlay = overlay_mask(original, object_matte, (255, 50, 50))

        paste_mask = erode_mask(object_matte, erode_px=int(lab_cfg["mask_erode_px"]))
        hard_alpha = make_feather_alpha(paste_mask, feather_px=0)
        micro_alpha = make_feather_alpha(paste_mask, feather_px=int(lab_cfg["micro_feather_px"]))

        top_row_full = [
            ("original", original),
            ("original + box", draw_boxes(original.copy(), labels)),
            ("object matte", object_overlay),
        ]

        top_row_zoom = build_zoom_row([
            ("original", original, label0),
            ("original + box", draw_boxes(original.copy(), labels), label0),
            ("object matte", object_overlay, label0),
        ])

        full_rows = [top_row_full]
        zoom_rows = [top_row_zoom]

        for preset_name in args.presets:
            preset = v2_cfg["presets"][preset_name]

            condition = make_night_condition(
                image=original,
                darkness=float(preset["darkness"]),
            )

            seed = int(project_cfg["yolo"]["seed"]) + int(row["source_index"]) * 100 + (17 if preset_name == "medium" else 0)

            generated_context = run_img2img_backend(
                backend=backend,
                image=condition,
                prompt=prompt,
                negative_prompt=negative_prompt,
                strength=float(preset["img2img_strength"]),
                guidance_scale=float(preset["guidance_scale"]),
                steps=int(preset["steps"]),
                seed=seed,
            )

            lab_delta_object = apply_lab_delta_to_object(
                original=original,
                generated_context=generated_context,
                object_mask=paste_mask,
                ring_inner_px=int(lab_cfg["ring_inner_px"]),
                ring_outer_px=int(lab_cfg["ring_outer_px"]),
                luminance_strength=float(lab_cfg["luminance_strength"]),
                contrast_strength=float(lab_cfg["contrast_strength"]),
                chroma_strength=float(lab_cfg["chroma_strength"]),
                night_blue_bias=float(lab_cfg["night_blue_bias"]),
                max_l_shift=float(lab_cfg["max_l_shift"]),
                max_ab_shift=float(lab_cfg["max_ab_shift"]),
                min_l_scale=float(lab_cfg["min_l_scale"]),
                max_l_scale=float(lab_cfg["max_l_scale"]),
            )

            hard_original = composite_with_alpha(
                background=generated_context,
                foreground=original,
                alpha_mask=hard_alpha,
            )

            hard_lab_delta = composite_with_alpha(
                background=generated_context,
                foreground=lab_delta_object,
                alpha_mask=hard_alpha,
            )

            microfeather_lab_delta = composite_with_alpha(
                background=generated_context,
                foreground=lab_delta_object,
                alpha_mask=micro_alpha,
            )

            base_stem = f"v2_{preset_name}_{int(row['source_index']):04d}_{image_path.stem}"

            variants = [
                ("hard_original", hard_original, hard_alpha),
                ("hard_lab_delta", hard_lab_delta, hard_alpha),
                ("microfeather_lab_delta", microfeather_lab_delta, micro_alpha),
            ]

            for mode_name, mode_image, mode_alpha in variants:
                out_img = output_root / preset_name / mode_name / "images" / f"{base_stem}.jpg"
                out_json = output_root / preset_name / mode_name / "metadata" / f"{base_stem}.json"

                ensure_dir(out_img.parent)
                ensure_dir(out_json.parent)

                mode_image.save(out_img, quality=95)

                meta = quality_metadata(
                    original=original,
                    generated_context=generated_context,
                    composite=mode_image,
                    object_mask=paste_mask,
                    alpha_mask=mode_alpha,
                    quality_cfg=v2_cfg["quality_gate"],
                )

                payload = {
                    "preset": preset_name,
                    "mode": mode_name,
                    "source_index": int(row["source_index"]),
                    "source_image": str(image_path),
                    "source_label": str(label_path),
                    "generated_image": str(out_img),
                    "seed": seed,
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    **lab_cfg,
                    **preset,
                    **meta,
                }

                out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                rows.append(payload)

            row_full = [
                (f"{preset_name} condition", condition),
                (f"{preset_name} paste mask", preview_alpha(hard_alpha)),
                (f"{preset_name} delta object", draw_boxes(lab_delta_object.copy(), labels)),
                (f"{preset_name} hard original", draw_boxes(hard_original.copy(), labels)),
                (f"{preset_name} hard lab delta", draw_boxes(hard_lab_delta.copy(), labels)),
                (f"{preset_name} microfeather", draw_boxes(microfeather_lab_delta.copy(), labels)),
            ]
            full_rows.append(row_full)

            row_zoom = build_zoom_row([
                (f"{preset_name} condition", condition, label0),
                (f"{preset_name} paste mask", preview_alpha(hard_alpha), label0),
                (f"{preset_name} delta object", draw_boxes(lab_delta_object.copy(), labels), label0),
                (f"{preset_name} hard original", draw_boxes(hard_original.copy(), labels), label0),
                (f"{preset_name} hard lab delta", draw_boxes(hard_lab_delta.copy(), labels), label0),
                (f"{preset_name} microfeather", draw_boxes(microfeather_lab_delta.copy(), labels), label0),
            ])
            zoom_rows.append(row_zoom)

        grid_path_full = previews_dir / f"v2_grid_full_{int(row['source_index']):04d}_{image_path.stem}.jpg"
        grid_path_zoom = previews_dir / f"v2_grid_zoom_{int(row['source_index']):04d}_{image_path.stem}.jpg"

        make_grid(full_rows, grid_path_full, tile_size=(260, 220), title_h=28, row_gap=14, col_gap=10)
        make_grid(zoom_rows, grid_path_zoom, tile_size=(260, 220), title_h=28, row_gap=14, col_gap=10)

    df = pd.DataFrame(rows)
    results_path = tables_dir / "diffusion_v2_grid_results.csv"
    df.to_csv(results_path, index=False)

    summary = (
        df.groupby(["preset", "mode"])
        .agg(
            count=("mode", "count"),
            auto_gate_pass_count=("accepted_by_auto_gate", "sum"),
            auto_gate_pass_rate=("accepted_by_auto_gate", "mean"),
            mean_background_diff=("background_region_mean_abs_diff", "mean"),
            mean_object_diff=("object_region_mean_abs_diff", "mean"),
            mean_context_diff=("context_region_mean_abs_diff", "mean"),
            mean_halo_score=("halo_score", "mean"),
        )
        .reset_index()
    )

    summary_path = tables_dir / "diffusion_v2_grid_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\\nV2.2 diffusion LAB delta grid completed.")
    print(f"Results: {results_path}")
    print(f"Summary: {summary_path}")
    print(f"Preview dir: {previews_dir}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
