"""Inspect generated diffusion images for black/degenerate outputs."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image


def image_stats(path: Path) -> dict:
    image = Image.open(path)
    arr = np.array(image.convert("RGB"))
    nearly_black = np.all(arr < 5, axis=2)
    return {
        "path": str(path),
        "mode": image.mode,
        "size": f"{image.width}x{image.height}",
        "dtype": str(arr.dtype),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "nearly_black_pct": float(nearly_black.mean() * 100.0),
    }


def is_suspicious(stats: dict) -> bool:
    return stats["mean"] < 5 or stats["max"] < 30 or stats["std"] < 2


def read_manifest(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def inspect_manifest(manifest_path: Path, report_path: Path) -> None:
    rows = read_manifest(manifest_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "index",
        "source_path",
        "generated_path",
        "source_mode",
        "source_size",
        "source_dtype",
        "source_min",
        "source_max",
        "source_mean",
        "source_std",
        "source_nearly_black_pct",
        "generated_mode",
        "generated_size",
        "generated_dtype",
        "generated_min",
        "generated_max",
        "generated_mean",
        "generated_std",
        "generated_nearly_black_pct",
        "suspicious",
    ]

    with report_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, row in enumerate(rows, start=1):
            source_path = Path(row["source_image_path"])
            generated_path = Path(row["output_image_path"])
            source_stats = image_stats(source_path)
            generated_stats = image_stats(generated_path)
            suspicious = is_suspicious(generated_stats)

            print(f"[{idx}] source: {source_path}")
            print(f"    generated: {generated_path}")
            print(
                "    generated "
                f"mode={generated_stats['mode']} size={generated_stats['size']} "
                f"dtype={generated_stats['dtype']} min={generated_stats['min']:.1f} "
                f"max={generated_stats['max']:.1f} mean={generated_stats['mean']:.2f} "
                f"std={generated_stats['std']:.2f} "
                f"nearly_black={generated_stats['nearly_black_pct']:.2f}% "
                f"suspicious={suspicious}"
            )

            writer.writerow(
                {
                    "index": idx,
                    "source_path": source_stats["path"],
                    "generated_path": generated_stats["path"],
                    "source_mode": source_stats["mode"],
                    "source_size": source_stats["size"],
                    "source_dtype": source_stats["dtype"],
                    "source_min": source_stats["min"],
                    "source_max": source_stats["max"],
                    "source_mean": source_stats["mean"],
                    "source_std": source_stats["std"],
                    "source_nearly_black_pct": source_stats["nearly_black_pct"],
                    "generated_mode": generated_stats["mode"],
                    "generated_size": generated_stats["size"],
                    "generated_dtype": generated_stats["dtype"],
                    "generated_min": generated_stats["min"],
                    "generated_max": generated_stats["max"],
                    "generated_mean": generated_stats["mean"],
                    "generated_std": generated_stats["std"],
                    "generated_nearly_black_pct": generated_stats["nearly_black_pct"],
                    "suspicious": suspicious,
                }
            )

    print(f"CSV report written: {report_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect diffusion-generated images for black outputs.")
    parser.add_argument("--manifest", default="data/synthetic/diffusion_grid/manifest.jsonl")
    parser.add_argument("--report", default="reports/generated_image_diagnostics.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    inspect_manifest(Path(args.manifest), Path(args.report))


if __name__ == "__main__":
    main()
