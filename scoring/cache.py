"""Raw prefix-cache codec shared by scoring and analysis."""

import json


def serialize_prefix_cache(cache: dict[tuple[int, ...], object]) -> dict[str, object]:
    """Encode tuple prefix keys for the on-disk JSON payload."""
    return {json.dumps(list(prefix), separators=(",", ":")): result for prefix, result in cache.items()}


def load_prefix_cache(raw: dict) -> dict[tuple[int, ...], object]:
    """Decode the on-disk flat prefix-cache payload."""
    return {tuple(json.loads(k)): v for k, v in raw["prefix_cache"].items()}
