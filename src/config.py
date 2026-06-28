from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    current = Path.cwd().resolve()

    for candidate in [current, *current.parents]:
        if (candidate / "configs" / "project.yaml").exists():
            return candidate

    raise FileNotFoundError("Could not find project root with configs/project.yaml")


def load_config(config_path: str | Path = "configs/project.yaml") -> dict[str, Any]:
    root = project_root()
    path = root / config_path

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
