from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.augment import read_yolo_labels
from src.config import load_config
from src.diffusion import (
    DiffusionBackend,
    compute_diff_stats,
    force_context_change_before_inpainting,
    image_pixel_stats,
    is_suspicious_black_image,
    object_mask_to_inpaint_mask,
    reinsert_object_region,
    yolo_labels_to_object_mask,
)
from src.utils import ensure_dir, resolve_device, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate accepted inpaint-reinsert night diffusion pool.")

    parser.add_argument("--target-count", type=int, default=300)
    parser.add_argument("--variants-per-image", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--min-background-diff", type=float, default=8.0)
    parser.add_argument("--max-object-diff", type=float, default=3.0)
    parser.add_argument("--max-mask-coverage", type=float, default=0.35)

    return parser.parse_args()


def passes_quality_gate(
    metadata: dict,
    min_background_diff: float,
    max_object_diff: float,
    max_mask_coverage: float,
) -> tuple[bool, list[str]]:
    reasons = []

    if metadata["suspicious_black_image"]:
        reasons.append("suspicious_black_image")

    if metadata["background_region_mean_abs_diff"] < min_background_diff:
        reasons.append("background_diff_too_low")

    if metadata["object_region_mean_abs_diff"] > max_object_diff:
        reasons.append("object_diff_too_high")

    if metadata["mask_coverage"] > max_mask_coverage:
        reasons.append("mask_coverage_too_high")

    return len(reasons) == 0, reasons


def main() -> None:
    args = parse_args()
    config = load_config()

    baseline_root = Path(config["experiments"]["baseline"]["dataset_root"])
    pool_root = Path("data/augmented/diffusion_reinsert_night_pool")

    images_dir = pool_root / "images"
    labels_dir = pool_root / "labels"
    metadata_dir = pool_root / "metadata"

    reports_dir = ensure_dir(config["paths"]["reports"])
    tables_dir = ensure_dir(config["paths"]["tables"])

    if pool_root.exists() and args.overwrite:
        shutil.rmtree(pool_root)

    ensure_dir(images_dir)
    ensure_dir(labels_dir)
    ensure_dir(metadata_dir)

    device = resolve_device(config["yolo"]["device"])

    backend = DiffusionBackend(
        diffusion_config=config["diffusion"],
        device=device,
    )

    source_images = sorted((baseline_root / "images" / "train").glob("*.jpg"))

    if args.start_index >= len(source_images):
        raise ValueError(f"start-index={args.start_index} >= number of train images={len(source_images)}")

    selected_images = source_images[args.start_index:]
    source_label_dir = baseline_root / "labels" / "train"

    accepted_rows = []
    rejected_rows = []

    print(f"Target accepted diffusion images: {args.target_count}")
    print(f"Variants per source image: {args.variants_per_image}")
    print(f"Source images available from index {args.start_index}: {len(selected_images)}")
    print(f"Pool root: {pool_root}")
    print(f"Device: {device}")

    progress = tqdm(total=args.target_count, desc="Accepted diffusion images")

    for source_idx, image_path in enumerate(selected_images, start=args.start_index):
        if len(accepted_rows) >= args.target_count:
            break

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

        for variant_idx in range(args.variants_per_image):
            if len(accepted_rows) >= args.target_count:
                break

            global_variant_idx = source_idx * 1000 + variant_idx

            conditioned = force_context_change_before_inpainting(
                original=original,
                object_mask=object_mask,
                variant_index=global_variant_idx,
            )

            seed = int(config["yolo"]["seed"]) + source_idx * 100 + variant_idx

            generated_before = backend.inpaint(
                image=conditioned,
                inpaint_mask=inpaint_mask,
                prompt=config["diffusion"]["prompt"],
                negative_prompt=config["diffusion"]["negative_prompt"],
                strength=float(config["diffusion"]["inpaint_strength"]),
                guidance_scale=float(config["diffusion"]["inpaint_guidance_scale"]),
                steps=int(config["diffusion"]["inpaint_steps"]),
                seed=seed,
            )

            generated_after = reinsert_object_region(
                generated=generated_before,
                original=original,
                object_mask=object_mask,
            )

            stats = image_pixel_stats(generated_after)
            diff_stats = compute_diff_stats(
                original=original,
                generated=generated_after,
                object_mask=object_mask,
            )

            output_stem = f"diffusion_reinsert_night_{len(accepted_rows):05d}_{image_path.stem}_v{variant_idx:02d}"

            output_image_path = images_dir / f"{output_stem}.jpg"
            output_label_path = labels_dir / f"{output_stem}.txt"
            output_metadata_path = metadata_dir / f"{output_stem}.json"

            metadata = {
                "pool_index": len(accepted_rows),
                "source_index": source_idx,
                "variant_index": variant_idx,
                "source_image": str(image_path),
                "source_label": str(label_path),
                "generated_image": str(output_image_path),
                "generated_label": str(output_label_path),
                "metadata_path": str(output_metadata_path),
                "mode": "inpaint_reinsert_night",
                "reinsertion_used": True,
                "device": device,
                "seed": seed,
                "quality_gate": {
                    "min_background_diff": args.min_background_diff,
                    "max_object_diff": args.max_object_diff,
                    "max_mask_coverage": args.max_mask_coverage,
                },
                "output_pixel_stats": stats,
                "suspicious_black_image": is_suspicious_black_image(stats),
                **diff_stats,
            }

            accepted, rejection_reasons = passes_quality_gate(
                metadata=metadata,
                min_background_diff=args.min_background_diff,
                max_object_diff=args.max_object_diff,
                max_mask_coverage=args.max_mask_coverage,
            )

            if accepted:
                generated_after.save(output_image_path, quality=95)
                shutil.copy2(label_path, output_label_path)

                metadata["accepted"] = True
                metadata["rejection_reasons"] = []

                output_metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

                accepted_rows.append(metadata)
                progress.update(1)

                print(
                    f"\nACCEPTED {len(accepted_rows)}/{args.target_count} | "
                    f"bg_diff={metadata['background_region_mean_abs_diff']:.2f} | "
                    f"obj_diff={metadata['object_region_mean_abs_diff']:.2f} | "
                    f"mask={metadata['mask_coverage']:.3f}"
                )

            else:
                metadata["accepted"] = False
                metadata["rejection_reasons"] = rejection_reasons
                rejected_rows.append(metadata)

                print(
                    f"\nREJECTED | reasons={rejection_reasons} | "
                    f"bg_diff={metadata['background_region_mean_abs_diff']:.2f} | "
                    f"obj_diff={metadata['object_region_mean_abs_diff']:.2f} | "
                    f"mask={metadata['mask_coverage']:.3f}"
                )

    progress.close()

    accepted_df = pd.DataFrame(accepted_rows)
    rejected_df = pd.DataFrame(rejected_rows)

    accepted_csv = tables_dir / "diffusion_reinsert_night_pool_accepted.csv"
    rejected_csv = tables_dir / "diffusion_reinsert_night_pool_rejected.csv"

    accepted_df.to_csv(accepted_csv, index=False)
    rejected_df.to_csv(rejected_csv, index=False)

    report = {
        "target_count": args.target_count,
        "accepted_count": len(accepted_rows),
        "rejected_count": len(rejected_rows),
        "pool_root": str(pool_root),
        "accepted_csv": str(accepted_csv),
        "rejected_csv": str(rejected_csv),
        "quality_gate": {
            "min_background_diff": args.min_background_diff,
            "max_object_diff": args.max_object_diff,
            "max_mask_coverage": args.max_mask_coverage,
        },
    }

    save_json(report, reports_dir / "diffusion_reinsert_night_pool_report.json")

    print("\nDiffusion pool generation completed.")
    print(json.dumps(report, indent=2))

    if len(accepted_rows) < args.target_count:
        print("WARNING: target count was not reached.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
