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

    ax = pivot.plot(kind="bar", figsize=(8, 5))
    ax.set_title(f"{metric} comparison")
    ax.set_xlabel("Split")
    ax.set_ylabel(metric)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)

    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    config = load_config()

    tables_dir = Path(config["paths"]["tables"])
    reports_dir = Path(config["paths"]["reports"])
    previews_dir = Path(config["paths"]["previews"])

    baseline_run = config["experiments"]["baseline"]["name"]
    classic_run = config["experiments"]["classic"]["name"]

    experiments = {
        "baseline_real_only": baseline_run,
        "real_plus_classic": classic_run,
    }

    frames = []

    for experiment_name, run_name in experiments.items():
        df = load_eval_table(tables_dir, run_name)
        df["experiment"] = experiment_name
        frames.append(df)

    comparison = pd.concat(frames, ignore_index=True)

    comparison_path = tables_dir / "comparison_baseline_vs_classic.csv"
    comparison.to_csv(comparison_path, index=False)

    baseline = comparison[comparison["experiment"] == "baseline_real_only"].copy()
    classic = comparison[comparison["experiment"] == "real_plus_classic"].copy()

    deltas = classic.merge(
        baseline,
        on="split",
        suffixes=("_classic", "_baseline"),
    )

    delta_rows = []

    for _, row in deltas.iterrows():
        delta_rows.append(
            {
                "split": row["split"],
                "precision_delta": row["precision_classic"] - row["precision_baseline"],
                "recall_delta": row["recall_classic"] - row["recall_baseline"],
                "map50_delta": row["map50_classic"] - row["map50_baseline"],
                "map50_95_delta": row["map50_95_classic"] - row["map50_95_baseline"],
            }
        )

    delta_df = pd.DataFrame(delta_rows)
    delta_path = tables_dir / "comparison_baseline_vs_classic_deltas.csv"
    delta_df.to_csv(delta_path, index=False)

    for metric in ["precision", "recall", "map50", "map50_95"]:
        make_metric_plot(
            comparison,
            metric=metric,
            output_path=previews_dir / f"comparison_baseline_vs_classic_{metric}.png",
        )

    report = {
        "baseline_run": baseline_run,
        "classic_run": classic_run,
        "comparison_table": str(comparison_path),
        "delta_table": str(delta_path),
        "summary": delta_df.to_dict(orient="records"),
    }

    report_path = save_json(report, reports_dir / "comparison_baseline_vs_classic_report.json")

    print("Comparison completed.")
    print(f"Comparison table: {comparison_path}")
    print(f"Delta table: {delta_path}")
    print(f"Report: {report_path}")
    print("\nDeltas:")
    print(delta_df.to_string(index=False))
    print("Step 06 completed successfully.")


if __name__ == "__main__":
    main()
