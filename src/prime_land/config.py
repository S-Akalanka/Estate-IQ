import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"

load_dotenv(PROJECT_ROOT / ".env")


def load_config(filename: str = "config.yaml") -> dict:
    """Load a YAML file from the config folder."""
    with open(CONFIG_DIR / filename, encoding="utf-8") as file:
        return yaml.safe_load(file)


CONFIG = load_config()


def get_path(name: str) -> Path:
    """Resolve a path from the paths section of config.yaml."""
    return PROJECT_ROOT / CONFIG["paths"][name]


def get_api_key() -> str:
    """Read the Google AI Studio key from the environment."""
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        raise ValueError("GOOGLE_API_KEY is missing, add it to the .env file")
    return key


def load_faqs() -> list[str]:
    """Read the FAQ questions used to warm the cache."""
    return load_config("faqs.yaml")["faqs"]
