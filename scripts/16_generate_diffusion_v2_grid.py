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
    draw_boxes,
    make_box_mask,
    make_inpaint_mask_from_protection,
    make_night_condition,
    overlay_mask,
    quality_metadata,
    save_grid,
    soft_reinsert_object,
)
from src.utils import ensure_dir, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate V2 diffusion hyperparameter grid.")
    parser.add_argument("--max-images", type=int, default=20)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--presets", nargs="+", default=["conservative", "medium", "strong"])
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
        diffusion_config={
            "inpaint_model": v2_cfg["inpaint_model"],
            "img2img_model": "runwayml/stable-diffusion-v1-5",
        },
        device=device,
    )

    prompt = v2_cfg["prompt"]["positive"]
    negative_prompt = v2_cfg["prompt"]["negative"]

    rows = []

    for row_idx, row in tqdm(list(sources.iterrows()), desc="V2 diffusion grid"):
        image_path = Path(row["image_path"])
        label_path = Path(row["label_path"])

        labels = read_yolo_labels(label_path)

        if not labels:
            continue

        original = Image.open(image_path).convert("RGB")

        preset_outputs = []

        base_tiles = [
            ("original", original),
            ("original + box", draw_boxes(original, labels)),
        ]

        for preset_name in args.presets:
            preset = v2_cfg["presets"][preset_name]

            strict_mask = make_box_mask(
                labels=labels,
                image_size=original.size,
                margin_px=int(preset["strict_object_margin_px"]),
                relative_margin=0.8,
                blur_px=0,
            )

            protection_mask = make_box_mask(
                labels=labels,
                image_size=original.size,
                margin_px=int(preset["protection_margin_px"]),
                relative_margin=1.5,
                blur_px=int(preset["mask_blur_px"]),
            )

            blend_mask = make_box_mask(
                labels=labels,
                image_size=original.size,
                margin_px=int(preset["blend_margin_px"]),
                relative_margin=2.0,
                blur_px=int(preset["mask_blur_px"]),
            )

            inpaint_mask = make_inpaint_mask_from_protection(protection_mask)

            condition = make_night_condition(
                original=original,
                protection_mask=protection_mask,
                condition_strength=float(preset["condition_strength"]),
                variant_index=int(row["source_index"]),
            )

            seed = int(project_cfg["yolo"]["seed"]) + int(row["source_index"]) * 100 + hash(preset_name) % 97

            generated_before = backend.inpaint(
                image=condition,
                inpaint_mask=inpaint_mask,
                prompt=prompt,
                negative_prompt=negative_prompt,
                strength=float(preset["strength"]),
                guidance_scale=float(preset["guidance_scale"]),
                steps=int(preset["steps"]),
                seed=seed,
            )

            generated_after = soft_reinsert_object(
                original=original,
                generated=generated_before,
                strict_mask=strict_mask,
                blend_mask=blend_mask,
                adapt_luminance=True,
            )

            meta = quality_metadata(
                original=original,
                generated=generated_after,
                strict_mask=strict_mask,
                quality_cfg=v2_cfg["quality_gate"],
            )

            output_stem = f"v2_{preset_name}_{int(row['source_index']):04d}_{image_path.stem}"
            output_image_path = output_root / preset_name / "images" / f"{output_stem}.jpg"
            output_meta_path = output_root / preset_name / "metadata" / f"{output_stem}.json"

            ensure_dir(output_image_path.parent)
            ensure_dir(output_meta_path.parent)

            generated_after.save(output_image_path, quality=95)

            metadata = {
                "preset": preset_name,
                "source_index": int(row["source_index"]),
                "source_image": str(image_path),
                "source_label": str(label_path),
                "generated_image": str(output_image_path),
                "device": device,
                "seed": seed,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                **preset,
                **meta,
            }

            output_meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            rows.append(metadata)

            preset_outputs.append((preset_name, draw_boxes(generated_after, labels)))

            print(
                f"\n{preset_name} | "
                f"accepted={meta['accepted_by_auto_gate']} | "
                f"bg_diff={meta['background_region_mean_abs_diff']:.2f} | "
                f"obj_diff={meta['object_region_mean_abs_diff']:.2f} | "
                f"reasons={meta['rejection_reasons']}"
            )

        grid_path = previews_dir / f"v2_grid_{int(row['source_index']):04d}_{image_path.stem}.jpg"

        save_grid(
            tiles=[
                *base_tiles,
                ("condition", condition),
                ("protect mask", overlay_mask(original, protection_mask, (255, 40, 40))),
                *preset_outputs,
            ],
            output_path=grid_path,
            tile_size=220,
        )

    df = pd.DataFrame(rows)

    results_path = tables_dir / "diffusion_v2_grid_results.csv"
    df.to_csv(results_path, index=False)

    summary = (
        df.groupby("preset")
        .agg(
            count=("preset", "count"),
            accepted=("accepted_by_auto_gate", "sum"),
            mean_background_diff=("background_region_mean_abs_diff", "mean"),
            mean_object_diff=("object_region_mean_abs_diff", "mean"),
            mean_mask_coverage=("mask_coverage", "mean"),
        )
        .reset_index()
    )

    summary_path = tables_dir / "diffusion_v2_grid_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\nV2 diffusion grid completed.")
    print(f"Results: {results_path}")
    print(f"Summary: {summary_path}")
    print(f"Preview dir: {previews_dir}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
