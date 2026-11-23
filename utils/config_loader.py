from pathlib import Path
from typing import Any

import yaml


def load_config(config_name: str = "params.yaml") -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "config" / config_name

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found at: {config_path}")

    return yaml.safe_load(config_path.read_text(encoding="utf-8"))
