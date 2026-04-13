from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.config import settings
from app.core.exceptions import RegistryNotFoundError


def read_registry(path: Path | None = None) -> dict[str, Any]:
    registry_path = path or settings.MODEL_REGISTRY_PATH
    if not registry_path.exists():
        raise RegistryNotFoundError(f"Registry file not found at {registry_path}")

    with registry_path.open("r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)

    if isinstance(payload, list):
        return {"models": payload}
    if not isinstance(payload, dict):
        return {"models": []}
    return payload


def write_registry(payload: dict[str, Any], path: Path | None = None) -> Path:
    registry_path = path or settings.MODEL_REGISTRY_PATH
    registry_path.parent.mkdir(parents=True, exist_ok=True)

    with registry_path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2, default=str)

    return registry_path


def get_best_model_details(path: Path | None = None) -> dict[str, Any]:
    registry = read_registry(path)

    if isinstance(registry.get("best_model"), dict):
        return registry["best_model"]

    candidates: list[dict[str, Any]] = []
    for key in ("models", "all_models"):
        value = registry.get(key)
        if isinstance(value, list):
            candidates.extend([item for item in value if isinstance(item, dict)])

    if not candidates:
        raise RegistryNotFoundError("No model entries found in registry")

    def sort_tuple(item: dict[str, Any]) -> tuple[float, float, float]:
        rmse = float(item.get("rmse", float("inf")))
        mae = float(item.get("mae", float("inf")))
        nse = float(item.get("nse", float("-inf")))
        return rmse, mae, -nse

    return sorted(candidates, key=sort_tuple)[0]
