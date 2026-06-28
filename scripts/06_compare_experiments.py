from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.utils import ensure_dir, save_json


def load_eval_table(tables_dir: Path, run_name: str) -> pd.DataFrame:
    path = tables_dir / f"{run_name}_eval_metrics.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing evaluation table: {path}\n"
            f"Run the corresponding training script first."
        )

    return pd.read_csv(path)


def make_metric_plot(df: pd.DataFrame, metric: str, output_path: Path) -> None:
    ensure_dir(output_path.parent)

    pivot = df.pivot(index="split", columns="experiment", values=metric)

    ax = pivot.plot(kind="bar", figsize=(9, 5))
    ax.set_title(f"{metric} comparison")
    ax.set_xlabel("Split")
    ax.set_ylabel(metric)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)

    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def compute_deltas(comparison: pd.DataFrame) -> pd.DataFrame:
    baseline = comparison[comparison["experiment"] == "baseline_real_only"].copy()

    delta_rows = []

    for experiment in sorted(comparison["experiment"].unique()):
        if experiment == "baseline_real_only":
            continue

        current = comparison[comparison["experiment"] == experiment].copy()

        merged = current.merge(
            baseline,
            on="split",
            suffixes=("_experiment", "_baseline"),
        )

        for _, row in merged.iterrows():
            delta_rows.append(
                {
                    "experiment": experiment,
                    "split": row["split"],
                    "precision_delta": row["precision_experiment"] - row["precision_baseline"],
                    "recall_delta": row["recall_experiment"] - row["recall_baseline"],
                    "map50_delta": row["map50_experiment"] - row["map50_baseline"],
                    "map50_95_delta": row["map50_95_experiment"] - row["map50_95_baseline"],
                }
            )

    return pd.DataFrame(delta_rows)


def main() -> None:
    config = load_config()

    tables_dir = Path(config["paths"]["tables"])
    reports_dir = Path(config["paths"]["reports"])
    previews_dir = Path(config["paths"]["previews"])

    experiments = {
        "baseline_real_only": config["experiments"]["baseline"]["name"],
        "real_plus_classic": config["experiments"]["classic"]["name"],
        "real_plus_object_preserving": config["experiments"]["object_preserving"]["name"],
    }

    frames = []

    for experiment_name, run_name in experiments.items():
        df = load_eval_table(tables_dir, run_name)
        df["experiment"] = experiment_name
        frames.append(df)

    comparison = pd.concat(frames, ignore_index=True)

    comparison_path = tables_dir / "comparison_all_experiments.csv"
    comparison.to_csv(comparison_path, index=False)

    delta_df = compute_deltas(comparison)

    delta_path = tables_dir / "comparison_all_experiments_deltas.csv"
    delta_df.to_csv(delta_path, index=False)

    for metric in ["precision", "recall", "map50", "map50_95"]:
        make_metric_plot(
            comparison,
            metric=metric,
            output_path=previews_dir / f"comparison_all_experiments_{metric}.png",
        )

    report = {
        "experiments": experiments,
        "comparison_table": str(comparison_path),
        "delta_table": str(delta_path),
        "summary": delta_df.to_dict(orient="records"),
    }

    report_path = save_json(report, reports_dir / "comparison_all_experiments_report.json")

    print("Comparison completed.")
    print(f"Comparison table: {comparison_path}")
    print(f"Delta table: {delta_path}")
    print(f"Report: {report_path}")
    print("\nDeltas vs baseline:")
    print(delta_df.to_string(index=False))
    print("Step 06 completed successfully.")


if __name__ == "__main__":
    main()
