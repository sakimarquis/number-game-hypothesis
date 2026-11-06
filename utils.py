"""Project utilities: path resolution, config loading, reproducibility."""

import json
import os
import random

import numpy as np
import torch
from ruamel.yaml import YAML

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def seed_everything(seed: int):
    """Set random seeds for Python, NumPy, and PyTorch (CPU + CUDA)."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def resolve_path(path: str) -> str:
    """Resolve a path relative to the project root."""
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def load_yaml(file_path: str) -> dict:
    """Load a YAML file, preserving quotes."""
    yaml = YAML()
    yaml.preserve_quotes = True
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.load(f)


def load_json(path: str) -> dict | None:
    """Load JSON file, or None if it doesn't exist."""
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _to_serializable(obj):
    """Recursively convert numpy/torch types to JSON-safe Python types."""
    if isinstance(obj, torch.Tensor):
        return obj.cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serializable(i) for i in obj]
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def save_json(path: str, data):
    """Save data as JSON, creating parent dirs. Converts numpy types automatically.

    Writes to a sibling temp file then os.replace() onto the destination so a crash
    mid-write cannot truncate an existing file (atomic on the same filesystem, Windows ok).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_to_serializable(data), f, indent=2)
    os.replace(tmp, path)


def group_by(items, key_fn):
    """Group items by key function, preserving insertion order. Returns (key, list) pairs."""
    groups = {}
    for item in items:
        groups.setdefault(key_fn(item), []).append(item)
    return groups.items()
