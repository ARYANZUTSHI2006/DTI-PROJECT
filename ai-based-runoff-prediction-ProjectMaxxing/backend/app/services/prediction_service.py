from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from app.config import settings
from app.core.exceptions import BackendException, ModelArtifactError
from app.core.logger import get_logger
from app.schemas import PredictRequest
from app.services.registry_service import get_best_model_details, read_registry


logger = get_logger(__name__)


def _resolve_artifact_path(best_model: dict[str, Any], registry_payload: dict[str, Any]) -> Path:
    artifact_path = (
        best_model.get("artifact_path")
        or best_model.get("model_path")
        or registry_payload.get("artifact_path")
    )
    if not artifact_path:
        raise ModelArtifactError("No artifact path found in registry or best model details")

    path = Path(str(artifact_path))
    if not path.is_absolute():
        path = settings.BASE_DIR / path

    if not path.exists():
        raise ModelArtifactError(f"Model artifact not found at {path}")

    return path


def _load_model(path: Path):
    if path.suffix.lower() in {".joblib", ".pkl", ".pickle"}:
        try:
            return joblib.load(path)
        except Exception:
            with path.open("rb") as file_obj:
                return pickle.load(file_obj)

    if path.suffix.lower() == ".json":
        try:
            import xgboost as xgb

            model = xgb.XGBRegressor()
            model.load_model(str(path))
            return model
        except Exception as exc:
            raise ModelArtifactError(f"Unable to load JSON model artifact: {exc}") from exc

    with path.open("rb") as file_obj:
        return pickle.load(file_obj)


def run_prediction(payload: PredictRequest) -> dict[str, Any]:
    logger.info("Prediction request received | feature_count=%s", len(payload.features))

    registry_payload = read_registry()
    best_model = get_best_model_details()
    artifact_path = _resolve_artifact_path(best_model, registry_payload)
    model = _load_model(artifact_path)

    feature_order = best_model.get("feature_cols") or best_model.get("feature_columns")
    if isinstance(feature_order, list) and feature_order:
        try:
            values = [payload.features[name] for name in feature_order]
        except KeyError as exc:
            raise BackendException(422, f"Missing required feature: {exc.args[0]}") from exc
    else:
        values = list(payload.features.values())

    feature_vector = np.array(values, dtype=float).reshape(1, -1)

    if not hasattr(model, "predict"):
        raise ModelArtifactError("Loaded artifact does not expose predict()")

    try:
        prediction_raw = model.predict(feature_vector)
    except Exception as exc:
        raise BackendException(422, f"Prediction failed for provided features: {exc}") from exc
    prediction_value = float(np.array(prediction_raw).reshape(-1)[0])

    target_transform = str(best_model.get("target_transform", "none")).lower()
    if target_transform == "log1p":
        prediction_value = float(np.expm1(max(prediction_value, 0.0)))

    prediction_value = max(0.0, prediction_value)

    return {
        "prediction": prediction_value,
        "model_name": best_model.get("model_name") or best_model.get("name"),
    }
