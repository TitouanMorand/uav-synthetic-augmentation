from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.size_eval import evaluate_size_stratified, save_size_eval_outputs
from src.train_eval import metrics_to_dict
from src.utils import ensure_dir, resolve_device, save_json


def find_weights(run_name: str) -> Path:
    candidates = [
        Path("runs/detect") / run_name / "weights" / "best.pt",
        Path("runs/detect/runs/detect") / run_name / "weights" / "best.pt",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Could not find best.pt for run={run_name}. Tried: {candidates}"
    )


def dataset_map(config: dict) -> dict[str, Path]:
    return {
        "real": Path(config["experiments"]["baseline"]["dataset_root"]),
        "stress_night": Path("data/stress/yolo_drone_hf_300_night"),
        "stress_haze": Path("data/stress/yolo_drone_hf_300_haze"),
        "stress_low_contrast": Path("data/stress/yolo_drone_hf_300_low_contrast"),
    }


def experiment_map(config: dict) -> dict[str, str]:
    experiments = {
        "baseline_real_only": config["experiments"]["baseline"]["name"],
        "real_plus_classic": config["experiments"]["classic"]["name"],
        "real_plus_object_preserving": config["experiments"]["object_preserving"]["name"],
    }

    # Automatically include diffusion ablation runs if their weights exist.
    diffusion_weight_paths = sorted(
        Path("runs/detect").glob(
            "real_plus_diffusion_reinsert_night_n*_hf_drone_300/weights/best.pt"
        )
    )

    for weights_path in diffusion_weight_paths:
        run_name = weights_path.parents[1].name

        # Example:
        # real_plus_diffusion_reinsert_night_n075_hf_drone_300
        marker = "night_"
        suffix = "_hf_drone_300"

        if marker in run_name and suffix in run_name:
            n_tag = run_name.split(marker, 1)[1].split(suffix, 1)[0]
            experiment_name = f"real_plus_diffusion_{n_tag}"
        else:
            experiment_name = run_name

        experiments[experiment_name] = run_name

    return experiments


def evaluate_standard_metrics(
    config: dict,
    experiments: dict[str, str],
    datasets: dict[str, Path],
    splits: list[str],
) -> pd.DataFrame:
    rows = []

    device = resolve_device(config["yolo"]["device"])
    imgsz = int(config["yolo"]["image_size"])
    batch = int(config["yolo"]["batch"])
    workers = int(config["yolo"]["workers"])

    for experiment_name, run_name in experiments.items():
        weights_path = find_weights(run_name)
        model = YOLO(str(weights_path))

        for dataset_name, dataset_root in datasets.items():
            dataset_yaml = dataset_root / "dataset.yaml"

            if not dataset_yaml.exists():
                raise FileNotFoundError(f"Missing dataset.yaml: {dataset_yaml}")

            for split in splits:
                print(f"Standard eval | {experiment_name} | {dataset_name} | {split}")

                metrics = model.val(
                    data=str(dataset_yaml),
                    split=split,
                    imgsz=imgsz,
                    batch=batch,
                    device=device,
                    workers=workers,
                    project="runs/detect/eval_protocol",
                    name=f"{run_name}_{dataset_name}_{split}",
                    exist_ok=True,
                    verbose=False,
                    plots=False,
                )

                row = {
                    "experiment": experiment_name,
                    "run_name": run_name,
                    "dataset": dataset_name,
                    "split": split,
                    **metrics_to_dict(metrics),
                }

                rows.append(row)

    return pd.DataFrame(rows)


def compute_deltas(df: pd.DataFrame) -> pd.DataFrame:
    baseline = df[df["experiment"] == "baseline_real_only"].copy()

    rows = []

    for experiment in sorted(df["experiment"].unique()):
        if experiment == "baseline_real_only":
            continue

        current = df[df["experiment"] == experiment].copy()

        merged = current.merge(
            baseline,
            on=["dataset", "split"],
            suffixes=("_experiment", "_baseline"),
        )

        for _, row in merged.iterrows():
            rows.append(
                {
                    "experiment": experiment,
                    "dataset": row["dataset"],
                    "split": row["split"],
                    "precision_delta": row["precision_experiment"] - row["precision_baseline"],
                    "recall_delta": row["recall_experiment"] - row["recall_baseline"],
                    "map50_delta": row["map50_experiment"] - row["map50_baseline"],
                    "map50_95_delta": row["map50_95_experiment"] - row["map50_95_baseline"],
                }
            )

    return pd.DataFrame(rows)


def make_protocol_plots(config: dict, standard_df: pd.DataFrame) -> None:
    previews_dir = ensure_dir(config["paths"]["previews"])

    for metric in ["precision", "recall", "map50", "map50_95"]:
        for split in ["val", "test"]:
            subset = standard_df[standard_df["split"] == split]

            if subset.empty:
                continue

            pivot = subset.pivot_table(
                index="dataset",
                columns="experiment",
                values=metric,
                aggfunc="mean",
            )

            ax = pivot.plot(kind="bar", figsize=(11, 5))
            ax.set_title(f"{metric} by dataset — {split}")
            ax.set_xlabel("Evaluation dataset")
            ax.set_ylabel(metric)
            ax.set_ylim(0, 1.0)
            ax.grid(axis="y", alpha=0.3)
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()

            output_path = previews_dir / f"protocol_{metric}_{split}.png"
            plt.savefig(output_path, dpi=160)
            plt.close()


def evaluate_size_metrics(
    config: dict,
    experiments: dict[str, str],
    datasets: dict[str, Path],
    splits: list[str],
) -> list[dict]:
    reports = []

    for experiment_name, run_name in experiments.items():
        weights_path = find_weights(run_name)

        for dataset_name, dataset_root in datasets.items():
            for split in splits:
                report = evaluate_size_stratified(
                    config=config,
                    run_name=run_name,
                    weights_path=weights_path,
                    dataset_root=dataset_root,
                    dataset_name=dataset_name,
                    split=split,
                )
                report["experiment"] = experiment_name
                reports.append(report)

    return reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full evaluation protocol.")
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument("--skip-size-eval", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()

    tables_dir = ensure_dir(config["paths"]["tables"])
    reports_dir = ensure_dir(config["paths"]["reports"])

    experiments = experiment_map(config)
    datasets = dataset_map(config)

    print("Experiments:")
    for k, v in experiments.items():
        print(f"- {k}: {v}")

    print("Datasets:")
    for k, v in datasets.items():
        print(f"- {k}: {v}")

    standard_df = evaluate_standard_metrics(
        config=config,
        experiments=experiments,
        datasets=datasets,
        splits=args.splits,
    )

    standard_path = tables_dir / "protocol_standard_metrics.csv"
    standard_df.to_csv(standard_path, index=False)

    delta_df = compute_deltas(standard_df)
    delta_path = tables_dir / "protocol_standard_metric_deltas.csv"
    delta_df.to_csv(delta_path, index=False)

    make_protocol_plots(config, standard_df)

    size_reports = []

    if not args.skip_size_eval:
        size_reports = evaluate_size_metrics(
            config=config,
            experiments=experiments,
            datasets=datasets,
            splits=args.splits,
        )
        save_size_eval_outputs(config, size_reports)

    report = {
        "experiments": experiments,
        "datasets": {k: str(v) for k, v in datasets.items()},
        "splits": args.splits,
        "standard_metrics_table": str(standard_path),
        "standard_deltas_table": str(delta_path),
        "size_eval_enabled": not args.skip_size_eval,
        "num_size_eval_reports": len(size_reports),
    }

    save_json(report, reports_dir / "protocol_evaluation_report.json")

    print("\nProtocol evaluation completed.")
    print(f"Standard metrics: {standard_path}")
    print(f"Deltas: {delta_path}")
    print("Step 09 completed successfully.")


if __name__ == "__main__":
    main()
