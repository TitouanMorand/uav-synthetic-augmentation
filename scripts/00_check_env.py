from __future__ import annotations

import json
import platform
import sys
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.utils import ensure_dir, resolve_device, save_json, set_seed


REQUIRED_PACKAGES = [
    "numpy",
    "opencv-python",
    "torch",
    "torchvision",
    "ultralytics",
    "albumentations",
    "matplotlib",
    "pandas",
    "pyyaml",
    "tqdm",
    "pillow",
    "datasets",
    "huggingface_hub",
    "rich",
]


def package_version(package_name: str) -> str:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "NOT INSTALLED"


def main() -> None:
    set_seed(42)

    config = load_config()
    reports_dir = ensure_dir(config["paths"]["reports"])

    versions = {}
    missing = []

    for package in REQUIRED_PACKAGES:
        package_ver = package_version(package)
        versions[package] = package_ver

        if package_ver == "NOT INSTALLED":
            missing.append(package)

    if missing:
        print("Missing packages:")
        for package in missing:
            print(f"- {package}")
        print("\nRun: pip install -r requirements.txt")
        raise SystemExit(1)

    import albumentations
    import cv2
    import datasets
    import matplotlib
    import numpy
    import pandas
    import PIL
    import torch
    import torchvision
    import ultralytics
    import yaml

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "project": config["project"]["name"],
        "workflow": "python-first, huggingface-first",
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "recommended_device": resolve_device(config["yolo"]["device"]),
        "imports": {
            "numpy": numpy.__version__,
            "opencv": cv2.__version__,
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "ultralytics": ultralytics.__version__,
            "albumentations": albumentations.__version__,
            "matplotlib": matplotlib.__version__,
            "pandas": pandas.__version__,
            "pillow": PIL.__version__,
            "datasets": datasets.__version__,
            "yaml": yaml.__name__,
        },
        "package_versions": versions,
    }

    report_path = save_json(report, reports_dir / "env_report.json")

    print(json.dumps(report, indent=2))
    print(f"\nEnvironment report saved to: {report_path}")
    print("Step 00 completed successfully.")


if __name__ == "__main__":
    main()
