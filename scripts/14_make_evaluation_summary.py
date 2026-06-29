from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.utils import ensure_dir


BUCKET_ORDER = ["very_tiny", "tiny", "small", "medium_plus"]
DATASET_ORDER = ["real", "stress_night", "stress_haze", "stress_low_contrast"]


def experiment_sort_key(name: str) -> tuple[int, int, str]:
    if name == "baseline_real_only":
        return (0, 0, name)
    if name == "real_plus_classic":
        return (1, 0, name)
    if name == "real_plus_object_preserving":
        return (2, 0, name)

    match = re.search(r"n(\d+)", name)
    if "real_plus_diffusion" in name and match:
        return (3, int(match.group(1)), name)

    return (9, 0, name)


def order_experiments(df: pd.DataFrame) -> pd.DataFrame:
    if "experiment" not in df.columns:
        return df

    order = sorted(df["experiment"].dropna().unique(), key=experiment_sort_key)
    df = df.copy()
    df["experiment"] = pd.Categorical(df["experiment"], categories=order, ordered=True)
    return df.sort_values("experiment")


def round_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].round(4)
    return df


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No data available._"

    df = round_metrics(df)
    columns = list(df.columns)

    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"

    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(row[col]) for col in columns) + " |")

    return "\n".join([header, sep, *rows])


def save_table(df: pd.DataFrame, path: Path) -> Path:
    ensure_dir(path.parent)
    round_metrics(df).to_csv(path, index=False)
    return path


def make_bar_plot(df: pd.DataFrame, x: str, y: str, title: str, output_path: Path) -> None:
    ensure_dir(output_path.parent)

    if df.empty:
        return

    plot_df = order_experiments(df)

    ax = plot_df.plot(
        kind="barh",
        x=x,
        y=y,
        figsize=(10, max(4, 0.5 * len(plot_df))),
        legend=False,
    )

    ax.set_title(title)
    ax.set_xlabel(y)
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def make_grouped_plot(pivot: pd.DataFrame, title: str, output_path: Path) -> None:
    ensure_dir(output_path.parent)

    if pivot.empty:
        return

    ax = pivot.plot(kind="bar", figsize=(12, 5))
    ax.set_title(title)
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def load_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required file: {path}\n"
            f"Run: python scripts/09_evaluate_protocol.py --splits test"
        )

    return pd.read_csv(path)


def maybe_load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def add_experiment_from_run_name(df: pd.DataFrame, run_to_exp: dict[str, str]) -> pd.DataFrame:
    df = df.copy()

    if "experiment" not in df.columns:
        df["experiment"] = df["run_name"].map(run_to_exp)

    return df


def main() -> None:
    args = parse_args()
    config = load_config()

    tables_dir = Path(config["paths"]["tables"])
    previews_dir = Path(config["paths"]["previews"])

    standard_path = tables_dir / "protocol_standard_metrics.csv"
    delta_path = tables_dir / "protocol_standard_metric_deltas.csv"
    size_path = tables_dir / "size_stratified_metrics.csv"
    fp_path = tables_dir / "fp_per_image_metrics.csv"

    standard_df = load_required_csv(standard_path)
    delta_df = load_required_csv(delta_path)

    size_df = maybe_load_csv(size_path)
    fp_df = maybe_load_csv(fp_path)

    run_to_exp = (
        standard_df[["run_name", "experiment"]]
        .drop_duplicates()
        .set_index("run_name")["experiment"]
        .to_dict()
    )

    if size_df is not None:
        size_df = add_experiment_from_run_name(size_df, run_to_exp)

    if fp_df is not None:
        fp_df = add_experiment_from_run_name(fp_df, run_to_exp)

    split = args.split

    standard_split = standard_df[standard_df["split"] == split].copy()
    delta_split = delta_df[delta_df["split"] == split].copy()

    standard_split = order_experiments(standard_split)
    delta_split = order_experiments(delta_split)

    # ------------------------------------------------------------------
    # 1. Real test main comparison
    # ------------------------------------------------------------------
    real_main = standard_split[standard_split["dataset"] == "real"][
        ["experiment", "run_name", "precision", "recall", "map50", "map50_95"]
    ].copy()

    save_table(real_main, tables_dir / f"summary_{split}_real_main_metrics.csv")

    make_bar_plot(
        real_main,
        x="experiment",
        y="map50_95",
        title=f"Real {split} mAP50-95 by experiment",
        output_path=previews_dir / f"summary_{split}_real_map50_95.png",
    )

    make_bar_plot(
        real_main,
        x="experiment",
        y="recall",
        title=f"Real {split} recall by experiment",
        output_path=previews_dir / f"summary_{split}_real_recall.png",
    )

    # ------------------------------------------------------------------
    # 2. Stress-test comparison
    # ------------------------------------------------------------------
    stress_map = standard_split.pivot_table(
        index="experiment",
        columns="dataset",
        values="map50_95",
        aggfunc="mean",
    ).reindex(columns=DATASET_ORDER)

    stress_recall = standard_split.pivot_table(
        index="experiment",
        columns="dataset",
        values="recall",
        aggfunc="mean",
    ).reindex(columns=DATASET_ORDER)

    stress_map = stress_map.reset_index()
    stress_recall = stress_recall.reset_index()

    stress_map = order_experiments(stress_map)
    stress_recall = order_experiments(stress_recall)

    save_table(stress_map, tables_dir / f"summary_{split}_stress_map50_95.csv")
    save_table(stress_recall, tables_dir / f"summary_{split}_stress_recall.csv")

    make_grouped_plot(
        stress_map.set_index("experiment"),
        title=f"{split} mAP50-95 across real and stress datasets",
        output_path=previews_dir / f"summary_{split}_stress_map50_95.png",
    )

    make_grouped_plot(
        stress_recall.set_index("experiment"),
        title=f"{split} recall across real and stress datasets",
        output_path=previews_dir / f"summary_{split}_stress_recall.png",
    )

    # ------------------------------------------------------------------
    # 3. Deltas vs baseline
    # ------------------------------------------------------------------
    delta_summary = delta_split[
        [
            "experiment",
            "dataset",
            "precision_delta",
            "recall_delta",
            "map50_delta",
            "map50_95_delta",
        ]
    ].copy()

    save_table(delta_summary, tables_dir / f"summary_{split}_deltas_vs_baseline.csv")

    # ------------------------------------------------------------------
    # 4. AP by object size
    # ------------------------------------------------------------------
    size_real_ap = pd.DataFrame()
    size_real_recall = pd.DataFrame()
    size_night_ap = pd.DataFrame()

    if size_df is not None:
        size_split = size_df[size_df["split"] == split].copy()
        size_split = order_experiments(size_split)

        size_real = size_split[size_split["dataset"] == "real"].copy()
        size_night = size_split[size_split["dataset"] == "stress_night"].copy()

        size_real_ap = size_real.pivot_table(
            index="experiment",
            columns="bucket",
            values="ap50",
            aggfunc="mean",
        ).reindex(columns=BUCKET_ORDER).reset_index()

        size_real_recall = size_real.pivot_table(
            index="experiment",
            columns="bucket",
            values="recall50",
            aggfunc="mean",
        ).reindex(columns=BUCKET_ORDER).reset_index()

        size_night_ap = size_night.pivot_table(
            index="experiment",
            columns="bucket",
            values="ap50",
            aggfunc="mean",
        ).reindex(columns=BUCKET_ORDER).reset_index()

        size_real_ap = order_experiments(size_real_ap)
        size_real_recall = order_experiments(size_real_recall)
        size_night_ap = order_experiments(size_night_ap)

        save_table(size_real_ap, tables_dir / f"summary_{split}_size_ap50_real.csv")
        save_table(size_real_recall, tables_dir / f"summary_{split}_size_recall_real.csv")
        save_table(size_night_ap, tables_dir / f"summary_{split}_size_ap50_stress_night.csv")

        make_grouped_plot(
            size_real_ap.set_index("experiment"),
            title=f"Real {split} AP50 by object size",
            output_path=previews_dir / f"summary_{split}_size_ap50_real.png",
        )

        make_grouped_plot(
            size_night_ap.set_index("experiment"),
            title=f"Stress night {split} AP50 by object size",
            output_path=previews_dir / f"summary_{split}_size_ap50_stress_night.png",
        )

    # ------------------------------------------------------------------
    # 5. FP per image
    # ------------------------------------------------------------------
    fp_pivot = pd.DataFrame()

    if fp_df is not None:
        fp_split = fp_df[fp_df["split"] == split].copy()
        fp_split = order_experiments(fp_split)

        fp_pivot = fp_split.pivot_table(
            index="experiment",
            columns="dataset",
            values="fp_per_image_conf_0_25",
            aggfunc="mean",
        ).reindex(columns=DATASET_ORDER).reset_index()

        fp_pivot = order_experiments(fp_pivot)

        save_table(fp_pivot, tables_dir / f"summary_{split}_fp_per_image.csv")

        make_grouped_plot(
            fp_pivot.set_index("experiment"),
            title=f"{split} false positives per image",
            output_path=previews_dir / f"summary_{split}_fp_per_image.png",
        )

    # ------------------------------------------------------------------
    # 6. Markdown summary
    # ------------------------------------------------------------------
    output_md = Path(args.output)

    parts = []

    parts.append(f"# Evaluation Summary — {split}\n")

    parts.append(
        "This report summarizes the augmentation evaluation protocol. "
        "It compares the real-only baseline against classic augmentation, "
        "object-preserving augmentation, and diffusion ablation runs when available.\n"
    )

    parts.append("## 1. Real test metrics\n")
    parts.append(markdown_table(real_main))

    parts.append("\n## 2. Stress-test mAP50-95\n")
    parts.append(markdown_table(stress_map))

    parts.append("\n## 3. Stress-test recall\n")
    parts.append(markdown_table(stress_recall))

    parts.append("\n## 4. Deltas vs baseline\n")
    parts.append(markdown_table(delta_summary))

    parts.append("\n## 5. AP50 by object size — real test\n")
    if size_df is None:
        parts.append("_Size-stratified evaluation not found. Run `python scripts/09_evaluate_protocol.py --splits test`._")
    else:
        parts.append(markdown_table(size_real_ap))

    parts.append("\n## 6. Recall by object size — real test\n")
    if size_df is None:
        parts.append("_Size-stratified evaluation not found._")
    else:
        parts.append(markdown_table(size_real_recall))

    parts.append("\n## 7. AP50 by object size — stress night\n")
    if size_df is None:
        parts.append("_Size-stratified evaluation not found._")
    else:
        parts.append(markdown_table(size_night_ap))

    parts.append("\n## 8. False positives per image\n")
    if fp_df is None:
        parts.append("_FP/image evaluation not found. Run size-stratified evaluation first._")
    else:
        parts.append(markdown_table(fp_pivot))

    parts.append("\n## Generated files\n")
    parts.append(
        f"- Standard metrics: `{standard_path}`\n"
        f"- Deltas: `{delta_path}`\n"
        f"- Size metrics: `{size_path}`\n"
        f"- FP/image metrics: `{fp_path}`\n"
        f"- Summary tables: `artifacts/tables/summary_{split}_*.csv`\n"
        f"- Summary plots: `artifacts/previews/summary_{split}_*.png`\n"
    )

    output_md.write_text("\n\n".join(parts), encoding="utf-8")

    print(f"Evaluation summary written to: {output_md}")
    print(f"Summary tables written to: {tables_dir}/summary_{split}_*.csv")
    print(f"Summary plots written to: {previews_dir}/summary_{split}_*.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create readable evaluation summary tables and plots.")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--output", type=str, default="EVALUATION_SUMMARY.md")
    return parser.parse_args()


if __name__ == "__main__":
    main()
