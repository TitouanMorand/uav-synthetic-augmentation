from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.train_eval import evaluate_yolo, train_yolo
from src.utils import save_json, set_seed


def count_train_images(dataset_root: Path) -> int:
    return len(list((dataset_root / "images" / "train").glob("*.jpg")))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO on diffusion ablation datasets.")

    parser.add_argument("--sizes", nargs="+", type=int, default=[75, 150, 165])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--equal-compute", action="store_true")
    parser.add_argument("--target-exposures", type=int, default=4500)

    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()

    set_seed(int(config["yolo"]["seed"]))

    all_results = {}

    for size in args.sizes:
        dataset_root = Path(f"data/augmented/yolo_drone_diffusion_reinsert_night_n{size:03d}")
        dataset_yaml = dataset_root / "dataset.yaml"

        if not dataset_yaml.exists():
            raise FileNotFoundError(
                f"Missing dataset for N={size}: {dataset_yaml}. "
                f"Run scripts/12_build_diffusion_ablation_datasets.py first."
            )

        train_images = count_train_images(dataset_root)

        if args.epochs is not None:
            epochs = args.epochs
            training_budget = "manual_epochs"
        elif args.equal_compute:
            epochs = max(1, round(args.target_exposures / train_images))
            training_budget = "equal_compute"
        else:
            epochs = int(config["yolo"]["epochs_augmented"])
            training_budget = "full_budget"

        run_name = f"real_plus_diffusion_reinsert_night_n{size:03d}_hf_drone_300"

        print("\n" + "=" * 100)
        print(f"Training diffusion ablation N={size}")
        print(f"Dataset: {dataset_root}")
        print(f"Train images: {train_images}")
        print(f"Epochs: {epochs}")
        print(f"Training budget: {training_budget}")
        print("=" * 100)

        train_summary = train_yolo(
            config=config,
            dataset_yaml=dataset_yaml,
            run_name=run_name,
            epochs=epochs,
            batch=args.batch,
            workers=args.workers,
            device=args.device,
        )

        eval_report = evaluate_yolo(
            config=config,
            dataset_yaml=dataset_yaml,
            weights_path=Path(train_summary["best_weights"]),
            run_name=run_name,
            splits=["val", "test"],
            device=args.device,
        )

        result = {
            "size": size,
            "dataset_root": str(dataset_root),
            "dataset_yaml": str(dataset_yaml),
            "run_name": run_name,
            "train_images": train_images,
            "epochs": epochs,
            "training_budget": training_budget,
            "training": train_summary,
            "evaluation": eval_report,
        }

        all_results[f"n{size:03d}"] = result

        save_json(
            result,
            Path(config["paths"]["reports"]) / f"{run_name}_full_result.json",
        )

    save_json(
        all_results,
        Path(config["paths"]["reports"]) / "diffusion_ablation_training_report.json",
    )

    print("Step 13 completed successfully.")


if __name__ == "__main__":
    main()
