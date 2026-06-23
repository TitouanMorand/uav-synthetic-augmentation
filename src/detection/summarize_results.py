"""Create a readable Markdown summary from an Ultralytics YOLO results.csv."""
import argparse
import csv
from pathlib import Path


KEY_METRICS = [
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
    "train/box_loss",
    "train/cls_loss",
    "val/box_loss",
    "val/cls_loss",
]


def clean_row(row: dict[str, str]) -> dict[str, str]:
    return {key.strip(): value.strip() for key, value in row.items()}


def as_float(value: str, default: float = float("-inf")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def format_value(value: str) -> str:
    number = as_float(value, default=float("nan"))
    if number != number:
        return value
    if abs(number) >= 100:
        return f"{number:.2f}"
    return f"{number:.4f}"


def row_to_markdown(title: str, row: dict[str, str]) -> list[str]:
    lines = [f"## {title}", "", "| Metric | Value |", "| --- | ---: |"]
    lines.append(f"| epoch | {row.get('epoch', 'n/a')} |")
    for metric in KEY_METRICS:
        if metric in row:
            lines.append(f"| {metric} | {format_value(row[metric])} |")
    return lines


def summarize_results(results_csv: Path, output: Path | None = None) -> Path:
    if not results_csv.exists():
        raise FileNotFoundError(f"results.csv not found: {results_csv}")

    with results_csv.open("r", newline="") as f:
        rows = [clean_row(row) for row in csv.DictReader(f)]
    if not rows:
        raise RuntimeError(f"No rows found in {results_csv}")

    last = rows[-1]
    best = max(rows, key=lambda row: as_float(row.get("metrics/mAP50-95(B)", "")))
    output_path = output or results_csv.with_name("metrics_summary.md")

    lines = [
        "# YOLO Metrics Summary",
        "",
        f"Source: `{results_csv}`",
        "",
        "Use the final epoch for the actual run endpoint, and the best mAP50-95 row",
        "to quickly track the strongest checkpoint during the smoke experiment.",
        "",
    ]
    lines.extend(row_to_markdown("Final Epoch", last))
    lines.extend([""])
    lines.extend(row_to_markdown("Best mAP50-95 Epoch", best))
    lines.extend([""])

    output_path.write_text("\n".join(lines))
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize an Ultralytics results.csv into Markdown.")
    parser.add_argument("results_csv", nargs="?", default="runs/baseline/yolov8n_smoke/results.csv")
    parser.add_argument("--output", default=None, help="Optional Markdown output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = summarize_results(
        results_csv=Path(args.results_csv),
        output=Path(args.output) if args.output else None,
    )
    print(f"Wrote readable metrics summary to {output}")


if __name__ == "__main__":
    main()
