from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.augment import read_yolo_labels
from src.config import load_config
from src.diffusion import DiffusionBackend
from src.diffusion_v2 import (
    composite_with_alpha,
    draw_boxes,
    labels_box_stats,
    make_feather_alpha,
    make_night_condition,
    make_object_matte,
    overlay_mask,
    preview_alpha,
    quality_metadata,
    relight_object_to_context,
    save_grid,
)
from src.utils import ensure_dir, resolve_device


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


def adaptive_feather_px(max_box_side_px: float, preset_cap: int) -> int:
    """
    Tiny drones should not get huge feather radii.
    """
    value = int(round(max_box_side_px * 0.15))
    value = max(2, value)
    value = min(value, preset_cap)
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate V2.2 diffusion hyperparameter grid.")
    parser.add_argument("--max-images", type=int, default=8)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--presets", nargs="+", default=["conservative", "medium"])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


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

    rows = []

    for _, row in tqdm(list(sources.iterrows()), desc="V2.2 diffusion grid"):
        image_path = Path(row["image_path"])
        label_path = Path(row["label_path"])

        labels = read_yolo_labels(label_path)
        if not labels:
            continue

        original = Image.open(image_path).convert("RGB")
        object_matte = make_object_matte(original, labels)
        object_overlay = overlay_mask(original, object_matte, (255, 50, 50))

        box_stats = labels_box_stats(labels, image_size=original.size)
        max_box_side = float(box_stats["max_box_side_px"])

        base_tiles = [
            ("original", original),
            ("original + box", draw_boxes(original.copy(), labels)),
            ("object matte", object_overlay),
        ]

        preset_tiles = []

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

            feather_px = adaptive_feather_px(
                max_box_side_px=max_box_side,
                preset_cap=int(preset["feather_px"]),
            )

            hard_alpha = make_feather_alpha(object_matte, feather_px=0)
            feather_alpha = make_feather_alpha(object_matte, feather_px=feather_px)

            hard_reinsert = composite_with_alpha(
                background=generated_context,
                foreground=original,
                alpha_mask=hard_alpha,
            )

            feather_reinsert = composite_with_alpha(
                background=generated_context,
                foreground=original,
                alpha_mask=feather_alpha,
            )

            relit_object = relight_object_to_context(
                original=original,
                generated_context=generated_context,
                object_mask=object_matte,
                ring_inner_px=int(preset["ring_inner_px"]),
                ring_outer_px=int(preset["ring_outer_px"]),
                relight_strength=float(preset.get("relight_strength", 0.25)),
            )

            relight_feather_reinsert = composite_with_alpha(
                background=generated_context,
                foreground=relit_object,
                alpha_mask=feather_alpha,
            )

            base_stem = f"v2_{preset_name}_{int(row['source_index']):04d}_{image_path.stem}"

            variants = [
                ("hard_reinsert", hard_reinsert, hard_alpha),
                ("feather_reinsert", feather_reinsert, feather_alpha),
                ("relight_feather_reinsert", relight_feather_reinsert, feather_alpha),
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
                    object_mask=object_matte,
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
                    "adaptive_feather_px": feather_px,
                    **preset,
                    **meta,
                }

                out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                rows.append(payload)

            preset_tiles.extend(
                [
                    (f"{preset_name} condition", condition),
                    (f"{preset_name} alpha", preview_alpha(feather_alpha)),
                    (f"{preset_name} hard", draw_boxes(hard_reinsert.copy(), labels)),
                    (f"{preset_name} feather", draw_boxes(feather_reinsert.copy(), labels)),
                    (f"{preset_name} relight", draw_boxes(relight_feather_reinsert.copy(), labels)),
                ]
            )

        grid_path = previews_dir / f"v2_grid_{int(row['source_index']):04d}_{image_path.stem}.jpg"
        save_grid(
            tiles=[*base_tiles, *preset_tiles],
            output_path=grid_path,
            tile_size=220,
        )

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
            mean_adaptive_feather_px=("adaptive_feather_px", "mean"),
        )
        .reset_index()
    )

    summary_path = tables_dir / "diffusion_v2_grid_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\nV2.2 diffusion grid completed.")
    print(f"Results: {results_path}")
    print(f"Summary: {summary_path}")
    print(f"Preview dir: {previews_dir}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
