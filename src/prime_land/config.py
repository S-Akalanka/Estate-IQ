from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"


def load_config(filename: str = "config.yaml") -> dict:
    """Load a YAML file from the config folder."""
    with open(CONFIG_DIR / filename, encoding="utf-8") as file:
        return yaml.safe_load(file)


CONFIG = load_config()


def get_path(name: str) -> Path:
    """Resolve a path from the paths section of config.yaml."""
    return PROJECT_ROOT / CONFIG["paths"][name]
