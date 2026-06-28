from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from ultralytics import YOLO

from src.config import project_root
from src.utils import ensure_dir, resolve_device, save_json


def metrics_to_dict(metrics: Any) -> dict[str, float | None]:
    box = getattr(metrics, "box", None)

    if box is None:
        return {
            "precision": None,
            "recall": None,
            "map50": None,
            "map50_95": None,
        }

    return {
        "precision": float(getattr(box, "mp", 0.0)),
        "recall": float(getattr(box, "mr", 0.0)),
        "map50": float(getattr(box, "map50", 0.0)),
        "map50_95": float(getattr(box, "map", 0.0)),
    }


def train_yolo(
    config: dict[str, Any],
    dataset_yaml: str | Path,
    run_name: str,
    epochs: int | None = None,
    batch: int | None = None,
    workers: int | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    root = project_root()
    yolo_cfg = config["yolo"]
    paths_cfg = config["paths"]

    dataset_yaml = root / dataset_yaml

    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {dataset_yaml}")

    epochs = int(epochs or yolo_cfg["epochs_baseline"])
    batch = int(batch or yolo_cfg["batch"])
    workers = int(workers if workers is not None else yolo_cfg["workers"])
    device = resolve_device(device or yolo_cfg["device"])

    image_size = int(yolo_cfg["image_size"])
    model_name = yolo_cfg["model"]
    seed = int(yolo_cfg["seed"])
    patience = int(yolo_cfg["patience"])

    runs_project_dir = root / paths_cfg["runs"] / "detect"

    print("Training YOLO")
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_yaml}")
    print(f"Run name: {run_name}")
    print(f"Epochs: {epochs}")
    print(f"Device: {device}")
    print(f"YOLO project dir: {runs_project_dir}")

    model = YOLO(model_name)

    model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        imgsz=image_size,
        batch=batch,
        workers=workers,
        device=device,
        project=str(runs_project_dir),
        name=run_name,
        exist_ok=True,
        patience=patience,
        seed=seed,
        verbose=True,
    )

    trainer = getattr(model, "trainer", None)
    actual_run_dir = Path(getattr(trainer, "save_dir", runs_project_dir / run_name))

    best_weights = actual_run_dir / "weights" / "best.pt"
    last_weights = actual_run_dir / "weights" / "last.pt"

    if not best_weights.exists():
        raise FileNotFoundError(
            f"Training finished, but best.pt was not found at: {best_weights}"
        )

    summary = {
        "run_name": run_name,
        "dataset_yaml": str(dataset_yaml),
        "model": model_name,
        "epochs": epochs,
        "image_size": image_size,
        "batch": batch,
        "workers": workers,
        "device": device,
        "run_dir": str(actual_run_dir),
        "best_weights": str(best_weights),
        "last_weights": str(last_weights),
        "best_weights_exists": best_weights.exists(),
        "last_weights_exists": last_weights.exists(),
    }

    reports_dir = ensure_dir(root / paths_cfg["reports"])
    save_json(summary, reports_dir / f"{run_name}_training_summary.json")

    return summary


def evaluate_yolo(
    config: dict[str, Any],
    dataset_yaml: str | Path,
    weights_path: str | Path,
    run_name: str,
    splits: list[str] | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    root = project_root()
    yolo_cfg = config["yolo"]
    paths_cfg = config["paths"]

    dataset_yaml = Path(dataset_yaml)
    weights_path = Path(weights_path)

    if not dataset_yaml.is_absolute():
        dataset_yaml = root / dataset_yaml

    if not weights_path.is_absolute():
        weights_path = root / weights_path

    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {dataset_yaml}")

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    splits = splits or ["val", "test"]
    device = resolve_device(device or yolo_cfg["device"])
    image_size = int(yolo_cfg["image_size"])

    model = YOLO(str(weights_path))

    rows = []
    report = {
        "run_name": run_name,
        "weights_path": str(weights_path),
        "dataset_yaml": str(dataset_yaml),
        "device": device,
        "splits": {},
    }

    for split in splits:
        print(f"Evaluating {run_name} on split={split}")

        metrics = model.val(
            data=str(dataset_yaml),
            split=split,
            imgsz=image_size,
            batch=int(yolo_cfg["batch"]),
            device=device,
            workers=int(yolo_cfg["workers"]),
            verbose=False,
        )

        split_metrics = metrics_to_dict(metrics)
        report["splits"][split] = split_metrics

        rows.append(
            {
                "run_name": run_name,
                "split": split,
                **split_metrics,
            }
        )

    reports_dir = ensure_dir(root / paths_cfg["reports"])
    tables_dir = ensure_dir(root / paths_cfg["tables"])

    report_path = save_json(report, reports_dir / f"{run_name}_eval_report.json")

    table_path = tables_dir / f"{run_name}_eval_metrics.csv"
    pd.DataFrame(rows).to_csv(table_path, index=False)

    print(f"Evaluation report saved to: {report_path}")
    print(f"Evaluation table saved to: {table_path}")

    return report


def train_and_evaluate_baseline(
    config: dict[str, Any],
    run_name: str,
    epochs: int | None = None,
    batch: int | None = None,
    workers: int | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    dataset_root = Path(config["experiments"]["baseline"]["dataset_root"])
    dataset_yaml = dataset_root / "dataset.yaml"

    train_summary = train_yolo(
        config=config,
        dataset_yaml=dataset_yaml,
        run_name=run_name,
        epochs=epochs,
        batch=batch,
        workers=workers,
        device=device,
    )

    best_weights = Path(train_summary["best_weights"])

    eval_report = evaluate_yolo(
        config=config,
        dataset_yaml=dataset_yaml,
        weights_path=best_weights,
        run_name=run_name,
        splits=["val", "test"],
        device=device,
    )

    return {
        "training": train_summary,
        "evaluation": eval_report,
    }
