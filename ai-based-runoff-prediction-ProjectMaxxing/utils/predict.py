from pathlib import Path
from typing import Any, Dict
import json
import warnings
from functools import lru_cache
import logging

import joblib
import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)
MAX_PHYSICAL_DISCHARGE_CUMEC = 100_000.0
MAX_REASONABLE_FEATURE_ABS = 1_000_000.0
PRED_SCALED_CLIP = (-20.0, 20.0)
_PRED_SCALE_WARNING_EMITTED = False
BASIN_MAX_DISCHARGE_CUMEC = {
    "small": 2_000.0,
    "medium": 10_000.0,
    "large": 50_000.0,
}


def _normalize_name(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _extract_numeric_series(feature_df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    if feature_df is None or feature_df.empty:
        return pd.Series(dtype=float)

    normalized_columns = {_normalize_name(col): col for col in feature_df.columns}
    for candidate in candidates:
        resolved_col = normalized_columns.get(_normalize_name(candidate))
        if resolved_col is None:
            continue
        series = pd.to_numeric(feature_df[resolved_col], errors="coerce")
        if series.notna().any():
            return series
    return pd.Series(dtype=float)


def _clip_by_candidates(feature_df: pd.DataFrame, candidates: list[str], low: float, high: float) -> None:
    normalized_columns = {_normalize_name(col): col for col in feature_df.columns}
    for candidate in candidates:
        resolved_col = normalized_columns.get(_normalize_name(candidate))
        if resolved_col is None:
            continue
        feature_df[resolved_col] = pd.to_numeric(feature_df[resolved_col], errors="coerce").clip(lower=low, upper=high)


def _soft_clip_to_training_bounds(
    feature_df: pd.DataFrame,
    feature_bounds: dict[str, tuple[float, float]] | None,
    alpha: float = 0.2,
) -> pd.DataFrame:
    if feature_df is None or feature_df.empty or not feature_bounds:
        return feature_df

    adjusted = feature_df.copy()
    norm_to_col = {_normalize_name(col): col for col in adjusted.columns}
    for feature_name, bounds in feature_bounds.items():
        if not isinstance(bounds, tuple) or len(bounds) != 2:
            continue
        lower, upper = bounds
        if not np.isfinite(lower) or not np.isfinite(upper) or upper < lower:
            continue
        col = norm_to_col.get(_normalize_name(feature_name))
        if col is None:
            continue
        series = pd.to_numeric(adjusted[col], errors="coerce")
        low_mask = series < lower
        high_mask = series > upper
        if low_mask.any():
            series.loc[low_mask] = lower + alpha * (series.loc[low_mask] - lower)
        if high_mask.any():
            series.loc[high_mask] = upper + alpha * (series.loc[high_mask] - upper)
        adjusted[col] = series.clip(lower=lower, upper=upper)
    return adjusted


def _apply_hydrologic_input_constraints(feature_df: pd.DataFrame) -> pd.DataFrame:
    if feature_df is None or feature_df.empty:
        return pd.DataFrame()

    clipped = feature_df.copy()

    # Rainfall constraints.
    _clip_by_candidates(clipped, ["Mean_PCP", "PCP", "P1", "P2", "P3", "PCP_roll_mean_3"], 0.0, 200.0)
    _clip_by_candidates(clipped, ["PCP_roll_sum_7"], 0.0, 500.0)

    # Temperature constraints.
    _clip_by_candidates(clipped, ["Mean_Tmin", "TMIN"], -10.0, 35.0)
    _clip_by_candidates(clipped, ["Mean_Tmax", "TMAX"], 0.0, 50.0)

    # Humidity constraints (supports fraction or percent input).
    rh_series = _extract_numeric_series(clipped, ["rh", "RH"])
    if not rh_series.empty:
        max_rh = float(rh_series.max(skipna=True)) if rh_series.notna().any() else 1.0
        if max_rh > 1.5:
            rh_series = rh_series / 100.0
        rh_series = rh_series.clip(lower=0.3, upper=1.0)
        _clip_by_candidates(clipped, ["rh", "RH"], 0.3, 1.0)
        for col in clipped.columns:
            if _normalize_name(col) in {"rh"}:
                clipped[col] = rh_series

    # Solar constraints. If source is low-scale daily-energy (~0-30), map W/m² bounds into that unit.
    solar_series = _extract_numeric_series(clipped, ["solar"])
    if not solar_series.empty:
        median_solar = float(solar_series.median(skipna=True))
        if median_solar <= 30.0:
            # Equivalent of ~50-350 W/m² in MJ/m²/day (approx divide by 11.57)
            _clip_by_candidates(clipped, ["solar"], 4.3, 30.3)
        else:
            _clip_by_candidates(clipped, ["solar"], 50.0, 350.0)

    # Wind constraints.
    _clip_by_candidates(clipped, ["wind", "wind "], 0.0, 20.0)

    # Hydrologic memory constraints.
    _clip_by_candidates(
        clipped,
        ["lag_discharge_1", "lag_discharge_2", "lag_discharge_3", "discharge_roll_mean_3"],
        0.0,
        50_000.0,
    )
    _clip_by_candidates(clipped, ["discharge_roll_std_3"], 0.0, 50_000.0)
    return clipped


def _infer_basin_scale(feature_df: pd.DataFrame | None) -> str:
    if feature_df is None or feature_df.empty:
        return "large"

    area_series = _extract_numeric_series(feature_df, ["watershed_area", "basin_area", "area", "catchment_area"])
    if area_series.empty:
        return "large"

    area_median = float(area_series.median(skipna=True))
    if not np.isfinite(area_median) or area_median <= 0:
        return "large"

    area_km2 = area_median / 1_000_000.0 if area_median > 100_000.0 else area_median
    if area_km2 <= 500.0:
        return "small"
    if area_km2 <= 5_000.0:
        return "medium"
    return "large"


def _enforce_rainfall_runoff_monotonicity(preds: np.ndarray, feature_df: pd.DataFrame | None) -> np.ndarray:
    if preds.size <= 1 or feature_df is None or feature_df.empty:
        return preds

    adjusted = preds.copy()
    precip = _extract_numeric_series(feature_df, ["Mean_PCP", "PCP", "P1", "P2", "P3"])
    lag1 = _extract_numeric_series(feature_df, ["lag_discharge_1"])
    if precip.empty:
        return adjusted

    precip = precip.reindex(feature_df.index).ffill().bfill().fillna(0.0)
    lag1 = lag1.reindex(feature_df.index).ffill().bfill().fillna(0.0) if not lag1.empty else pd.Series(0.0, index=feature_df.index)

    limit = min(len(adjusted), len(precip))
    for i in range(1, limit):
        p_prev = float(precip.iloc[i - 1])
        p_now = float(precip.iloc[i])
        if p_now > p_prev and adjusted[i] < adjusted[i - 1] * 0.9:
            rainfall_ratio = (p_now - p_prev) / max(abs(p_prev), 1.0)
            rainfall_ratio = float(np.clip(rainfall_ratio, 0.0, 5.0))
            adjusted[i] = adjusted[i] * (1.0 + 0.3 * rainfall_ratio)

        if p_now >= 80.0 and adjusted[i] < 1.0:
            lag_floor = float(max(1.0, 0.2 * float(lag1.iloc[i])))
            adjusted[i] = max(adjusted[i], lag_floor)
    return adjusted


def _stabilize_hydrograph(preds: np.ndarray, feature_df: pd.DataFrame | None) -> np.ndarray:
    if preds.size <= 1:
        return preds

    stabilized = preds.copy()
    std_series = _extract_numeric_series(feature_df, ["discharge_roll_std_3"]) if feature_df is not None else pd.Series(dtype=float)
    if std_series.empty:
        std_vals = np.full_like(stabilized, max(np.std(stabilized[: min(10, len(stabilized))]), 10.0), dtype=float)
    else:
        std_vals = np.asarray(std_series.ffill().bfill().fillna(10.0), dtype=float)

    limit = min(len(stabilized), len(std_vals))
    for i in range(1, limit):
        step_limit = max(3.0 * float(std_vals[i]), 10.0)
        delta = stabilized[i] - stabilized[i - 1]
        if abs(delta) > step_limit:
            stabilized[i] = stabilized[i - 1] + np.sign(delta) * step_limit
    return stabilized


def _sanitize_feature_frame(
    feature_df: pd.DataFrame,
    feature_bounds: dict[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    if feature_df is None or feature_df.empty:
        return pd.DataFrame()

    numeric = feature_df.copy()
    for col in numeric.columns:
        numeric[col] = pd.to_numeric(numeric[col], errors="coerce")

    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    medians = numeric.median(numeric_only=True)
    numeric = numeric.fillna(medians).fillna(0.0)
    return numeric


def _apply_inverse_target_transform(
    y_pred_scaled: np.ndarray,
    target_transform: str,
    best_model_meta: dict[str, Any] | None = None,
) -> np.ndarray:
    global _PRED_SCALE_WARNING_EMITTED
    preds = np.asarray(y_pred_scaled, dtype=float).reshape(-1)
    preds = np.where(np.isfinite(preds), preds, np.nan)

    if preds.size:
        pred_min = float(np.nanmin(preds))
        pred_max = float(np.nanmax(preds))
        pred_mean = float(np.nanmean(preds))
        if pred_min < -10.0 or pred_max > 10.0:
            if not _PRED_SCALE_WARNING_EMITTED:
                LOGGER.warning(
                    "Scaled predictions exceeded [-10, 10]. Returning raw model outputs without clipping. "
                    "Further identical warnings are suppressed.",
                )
                _PRED_SCALE_WARNING_EMITTED = True

    transform = str(target_transform or "none").strip().lower()
    meta = best_model_meta if isinstance(best_model_meta, dict) else {}

    if transform in {"none", ""}:
        return preds

    if transform == "log":
        return np.exp(preds)

    if transform == "log1p":
        return np.expm1(preds)

    if transform == "sqrt":
        return np.square(preds)

    if transform in {"boxcox", "box-cox"}:
        lam = float(meta.get("target_boxcox_lambda", meta.get("boxcox_lambda", 0.0)) or 0.0)
        shift = float(meta.get("target_boxcox_shift", 0.0) or 0.0)
        if lam == 0.0:
            return np.exp(preds) - shift
        shifted = (lam * preds) + 1.0
        shifted = np.where(shifted > 0.0, shifted, np.nan)
        return np.power(shifted, 1.0 / lam) - shift

    if transform in {"standard", "standardize", "zscore", "standard_scaler"}:
        std = float(meta.get("target_std", 1.0) or 1.0)
        mean = float(meta.get("target_mean", 0.0) or 0.0)
        return preds * std + mean

    if transform in {"minmax", "min_max", "minmax_scaler"}:
        y_min = float(meta.get("target_min", 0.0) or 0.0)
        y_max = float(meta.get("target_max", 1.0) or 1.0)
        if y_max <= y_min:
            return preds
        return preds * (y_max - y_min) + y_min

    return preds


def _finalize_discharge_predictions(preds: np.ndarray, feature_df: pd.DataFrame | None = None) -> np.ndarray:
    arr = np.asarray(preds, dtype=float).reshape(-1)
    arr = np.nan_to_num(arr, nan=0.0, posinf=MAX_PHYSICAL_DISCHARGE_CUMEC, neginf=0.0)
    arr = np.clip(arr, 0.0, MAX_PHYSICAL_DISCHARGE_CUMEC)

    arr = _enforce_rainfall_runoff_monotonicity(arr, feature_df)
    arr = _stabilize_hydrograph(arr, feature_df)
    basin_scale = _infer_basin_scale(feature_df)
    basin_max = float(BASIN_MAX_DISCHARGE_CUMEC.get(basin_scale, BASIN_MAX_DISCHARGE_CUMEC["large"]))
    arr = np.clip(arr, 0.0, min(MAX_PHYSICAL_DISCHARGE_CUMEC, basin_max))

    # Safety assertions requested for physically realistic runoff outputs.
    assert np.all(np.isfinite(arr)), "Non-finite discharge values after post-processing."
    assert np.all(arr >= 0.0), "Negative discharge detected after post-processing."
    assert np.all(arr <= MAX_PHYSICAL_DISCHARGE_CUMEC), "Discharge exceeds configured physical upper bound."
    if feature_df is not None and not feature_df.empty:
        precip = _extract_numeric_series(feature_df, ["Mean_PCP", "PCP", "P1", "P2", "P3"])
        if not precip.empty:
            precip = precip.reindex(feature_df.index).ffill().bfill().fillna(0.0)
            mask = (precip.to_numpy(dtype=float)[: arr.size] >= 80.0) & (arr <= 0.0)
            if np.any(mask):
                arr[mask] = 1.0

    assert np.all(arr <= 100_000.0), "Discharge exceeds hard hydrologic cap (100000 CUMEC)."
    return arr


@lru_cache(maxsize=1)
def _get_lstm_loader():
    try:
        import importlib

        keras_models = importlib.import_module("tensorflow.keras.models")
        raw_loader = getattr(keras_models, "load_model", None)
        if raw_loader is None:
            return None

        def _loader(model_path):
            # Load for inference only; avoids legacy H5 compile-time metric deserialization issues.
            try:
                return raw_loader(model_path, compile=False)
            except TypeError:
                return raw_loader(model_path)

        return _loader
    except Exception:
        return None


def _safe_load_joblib(path: Path) -> Any:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


def _safe_load_json(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_registry(model_dir: Path) -> dict[str, Any]:
    candidates = [
        model_dir.parent / "backend" / "model_registry.json",
        model_dir / "model_registry.json",
    ]
    for registry_path in candidates:
        payload = _safe_load_json(registry_path)
        if payload:
            return payload
    return {}


def _resolve_artifact_path(
    model_dir: Path,
    best_model: dict[str, Any],
    registry_payload: dict[str, Any],
) -> Path | None:
    artifact = best_model.get("artifact_path") or registry_payload.get("artifact_path")
    if not artifact:
        return None

    artifact_path = Path(str(artifact))
    if artifact_path.is_absolute():
        return artifact_path if artifact_path.exists() else None

    candidates = [
        model_dir.parent / artifact_path,
        model_dir.parent / "backend" / artifact_path,
        model_dir / artifact_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_generic_model(path: Path | None) -> Any:
    if path is None or not path.exists() or path.stat().st_size == 0:
        return None
    suffix = path.suffix.lower()
    if suffix == ".json":
        return _load_xgboost_model(path)
    if suffix in {".joblib", ".pkl"}:
        return _safe_load_joblib(path)
    return None


def _load_xgboost_model(path: Path | None) -> Any:
    if path is None or not path.exists() or path.suffix.lower() != ".json":
        return None
    try:
        import xgboost as xgb

        model = xgb.XGBRegressor()
        model.load_model(str(path))
        return model
    except Exception:
        return None


def _safe_load_lstm(path: Path | None) -> Any:
    if path is None or not path.exists() or path.stat().st_size == 0:
        return None

    load_model = _get_lstm_loader()
    if load_model is None:
        return None

    try:
        return load_model(path)
    except Exception:
        return None


def load_artifacts(model_dir: Path, load_lstm: bool = True) -> Dict[str, Any]:
    rf_model = _safe_load_joblib(model_dir / "rf_model.pkl")
    features = _safe_load_joblib(model_dir / "features.pkl")
    scaler = _safe_load_joblib(model_dir / "scaler.pkl")
    model_metadata = _safe_load_json(model_dir / "model_metadata.json")
    lstm_model_path = model_dir / "lstm_model.h5"
    lstm_model = _safe_load_lstm(lstm_model_path) if load_lstm else None

    registry_payload = _read_registry(model_dir)
    best_model = registry_payload.get("best_model") if isinstance(registry_payload.get("best_model"), dict) else {}
    best_model_name = str(best_model.get("model_name", ""))
    target_transform = str(best_model.get("target_transform", "none"))
    registry_feature_cols = _normalize_feature_list(best_model.get("feature_cols"))

    best_model_path = _resolve_artifact_path(model_dir, best_model, registry_payload)
    best_model_obj = _load_generic_model(best_model_path)

    xgb_model_path = best_model_path
    xgb_model = best_model_obj if best_model_name.lower().startswith("xgboost") else _load_xgboost_model(xgb_model_path)

    if xgb_model is None:
        fallback_paths = [
            model_dir.parent / "backend" / "models" / "best_xgboost_log1p.json",
            model_dir.parent / "backend" / "models" / "best_xgboost.json",
            model_dir / "best_xgboost_log1p.json",
            model_dir / "best_xgboost.json",
        ]
        for fallback_path in fallback_paths:
            xgb_model = _load_xgboost_model(fallback_path)
            if xgb_model is not None:
                xgb_model_path = fallback_path
                if not best_model_name:
                    best_model_name = "xgboost"
                break

    return {
        "rf_model": rf_model,
        "features": features,
        "scaler": scaler,
        "lstm_model": lstm_model,
        "lstm_model_path": lstm_model_path,
        "lstm_load_attempted": bool(load_lstm),
        "best_model_obj": best_model_obj,
        "best_model_path": best_model_path,
        "best_model_name": best_model_name,
        "xgb_model": xgb_model,
        "xgb_model_path": xgb_model_path,
        "xgb_model_name": best_model_name,
        "target_transform": target_transform,
        "registry_feature_cols": registry_feature_cols,
        "best_model": best_model,
        "model_metadata": model_metadata,
    }


def ensure_lstm_loaded(artifacts: Dict[str, Any]) -> Any:
    if artifacts.get("lstm_model") is not None:
        return artifacts.get("lstm_model")

    if artifacts.get("lstm_load_attempted"):
        return None

    artifacts["lstm_load_attempted"] = True
    lstm_path = artifacts.get("lstm_model_path")
    loaded = _safe_load_lstm(lstm_path if isinstance(lstm_path, Path) else None)
    artifacts["lstm_model"] = loaded
    return loaded


def _extract_feature_bounds_from_artifacts(artifacts: Dict[str, Any]) -> dict[str, tuple[float, float]]:
    payload = artifacts.get("best_model") if isinstance(artifacts.get("best_model"), dict) else {}
    raw_bounds = payload.get("feature_bounds")
    if not isinstance(raw_bounds, dict):
        return {}

    parsed: dict[str, tuple[float, float]] = {}
    for key, value in raw_bounds.items():
        if isinstance(value, dict):
            low = value.get("min")
            high = value.get("max")
        elif isinstance(value, (list, tuple)) and len(value) >= 2:
            low, high = value[0], value[1]
        else:
            continue

        try:
            low_f = float(low)
            high_f = float(high)
        except Exception:
            continue

        if not np.isfinite(low_f) or not np.isfinite(high_f) or high_f < low_f:
            continue
        parsed[str(key)] = (low_f, high_f)
    return parsed


def _normalize_feature_list(raw_features) -> list[str]:
    if raw_features is None:
        return []
    if isinstance(raw_features, pd.Index):
        return raw_features.tolist()
    if isinstance(raw_features, np.ndarray):
        return raw_features.tolist()
    if isinstance(raw_features, (list, tuple, set)):
        return list(raw_features)
    if isinstance(raw_features, dict):
        return list(raw_features.keys())
    return []


def _resolve_feature_column(feature_df: pd.DataFrame, feature_name: str) -> str | None:
    if feature_name in feature_df.columns:
        return feature_name
    norm_map = {_normalize_name(col): col for col in feature_df.columns}
    return norm_map.get(_normalize_name(feature_name))


def _apply_target_mode_to_predictions(
    preds: np.ndarray,
    feature_df: pd.DataFrame,
    model_metadata: dict[str, Any] | None = None,
) -> np.ndarray:
    metadata = model_metadata if isinstance(model_metadata, dict) else {}
    target_mode = str(metadata.get("target_mode", "absolute")).strip().lower()
    if target_mode != "delta_from_lag1":
        return np.asarray(preds, dtype=float).reshape(-1)

    base_feature = str(metadata.get("target_base_feature", "lag_discharge_1"))
    resolved_col = _resolve_feature_column(feature_df, base_feature)
    if resolved_col is None:
        return np.asarray(preds, dtype=float).reshape(-1)

    base = pd.to_numeric(feature_df[resolved_col], errors="coerce").to_numpy(dtype=float).reshape(-1)
    out = np.asarray(preds, dtype=float).reshape(-1)
    length = min(len(out), len(base))
    if length == 0:
        return out
    out[:length] = out[:length] + base[:length]
    return out


def get_feature_list(artifacts: Dict[str, Any]) -> list[str]:
    registry_features = _normalize_feature_list(artifacts.get("registry_feature_cols"))
    if registry_features:
        return registry_features
    serialized_features = _normalize_feature_list(artifacts.get("features"))
    if serialized_features:
        return serialized_features
    metadata = artifacts.get("model_metadata") if isinstance(artifacts.get("model_metadata"), dict) else {}
    return _normalize_feature_list(metadata.get("feature_columns"))


def align_features(feature_df: pd.DataFrame, feature_list: list[str]) -> pd.DataFrame:
    if not feature_list:
        return feature_df.copy()

    aligned = feature_df.copy()
    for feature_name in feature_list:
        if feature_name not in aligned.columns:
            aligned[feature_name] = 0.0

    return aligned[feature_list]


def _align_by_count(feature_df: pd.DataFrame, expected_features: int | None) -> pd.DataFrame:
    if expected_features is None or expected_features <= 0:
        return feature_df

    current = feature_df.copy()
    current_cols = list(current.columns)
    if len(current_cols) < expected_features:
        for idx in range(len(current_cols), expected_features):
            current[f"__pad_{idx}"] = 0.0
    elif len(current_cols) > expected_features:
        current = current.iloc[:, :expected_features].copy()
    return current


def _extract_lstm_feature_count(lstm_model: Any) -> int | None:
    input_shape = getattr(lstm_model, "input_shape", None)
    if isinstance(input_shape, list) and input_shape:
        input_shape = input_shape[0]
    if isinstance(input_shape, tuple) and len(input_shape) >= 3:
        try:
            return int(input_shape[-1]) if input_shape[-1] is not None else None
        except Exception:
            return None
    return None


def _predict_with_lstm(
    feature_df: pd.DataFrame,
    lstm_model: Any,
    lstm_feature_names: list[str] | None = None,
    scaler: Any = None,
    lstm_target_transform: str = "none",
) -> np.ndarray:
    if lstm_model is None:
        return np.array([])

    try:
        prepared = feature_df.copy()
        if lstm_feature_names:
            prepared = align_features(prepared, lstm_feature_names)
        else:
            prepared = _align_by_count(prepared, _extract_lstm_feature_count(lstm_model))

        prepared = _sanitize_feature_frame(prepared)
        if scaler is not None:
            scaler_features = getattr(scaler, "n_features_in_", None)
            if isinstance(scaler_features, int) and scaler_features > 0:
                prepared = _align_by_count(prepared, scaler_features)

        input_data = prepared.to_numpy(dtype=float)
        if scaler is not None and hasattr(scaler, "transform"):
            flat = input_data.reshape(input_data.shape[0], -1)
            flat = np.asarray(scaler.transform(flat), dtype=float)
            flat = np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
            input_data = flat.reshape(input_data.shape[0], 1, flat.shape[1])
        else:
            input_data = input_data.reshape((input_data.shape[0], 1, input_data.shape[1]))
        preds = np.asarray(lstm_model.predict(input_data, verbose=0)).reshape(-1)
        preds = _apply_inverse_target_transform(preds, lstm_target_transform)
        return preds
    except Exception:
        return np.array([])


def _predict_with_rf(feature_df: pd.DataFrame, rf_model: Any, rf_feature_names: list[str] | None = None) -> np.ndarray:
    if rf_model is not None and hasattr(rf_model, "predict"):
        old_verbose = None
        old_n_jobs = None
        try:
            prepared = feature_df.copy()
            if rf_feature_names:
                prepared = align_features(prepared, rf_feature_names)
            else:
                expected = getattr(rf_model, "n_features_in_", None)
                if isinstance(expected, int):
                    prepared = _align_by_count(prepared, expected)
            prepared = _sanitize_feature_frame(prepared)

            old_verbose = getattr(rf_model, "verbose", None)
            if old_verbose not in (None, 0):
                try:
                    rf_model.verbose = 0
                except Exception:
                    old_verbose = None
            old_n_jobs = getattr(rf_model, "n_jobs", None)
            if isinstance(old_n_jobs, int) and old_n_jobs != 1:
                try:
                    rf_model.n_jobs = 1
                except Exception:
                    old_n_jobs = None

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    return np.asarray(rf_model.predict(prepared)).reshape(-1)
            except ValueError:
                expected = getattr(rf_model, "n_features_in_", None)
                if isinstance(expected, int):
                    repaired = _align_by_count(feature_df, expected)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)
                        return np.asarray(rf_model.predict(repaired)).reshape(-1)
                raise
        except Exception:
            return np.array([])
        finally:
            if old_verbose not in (None, 0):
                try:
                    rf_model.verbose = old_verbose
                except Exception:
                    pass
            if old_n_jobs not in (None, 1):
                try:
                    rf_model.n_jobs = old_n_jobs
                except Exception:
                    pass
    if feature_df.shape[1] == 0:
        return np.array([])
    return feature_df.mean(axis=1).to_numpy()


def _predict_with_xgboost(
    feature_df: pd.DataFrame,
    xgb_model: Any,
    target_transform: str,
    xgb_feature_names: list[str] | None = None,
) -> np.ndarray:
    if xgb_model is None or not hasattr(xgb_model, "predict"):
        return np.array([])

    try:
        prepared = feature_df.copy()
        if xgb_feature_names:
            prepared = align_features(prepared, xgb_feature_names)
        prepared = _sanitize_feature_frame(prepared)
        preds = np.asarray(xgb_model.predict(prepared)).reshape(-1)
        preds = _apply_inverse_target_transform(preds, target_transform)
        return preds
    except Exception:
        return np.array([])


def _predict_with_best_model(
    feature_df: pd.DataFrame,
    best_model: Any,
    best_model_name: str,
    target_transform: str,
    best_feature_names: list[str] | None = None,
    best_model_meta: dict[str, Any] | None = None,
) -> np.ndarray:
    if best_model is None:
        return np.array([])

    model_name = str(best_model_name or "").strip().lower()
    if "xgboost" in model_name:
        return _predict_with_xgboost(
            feature_df,
            best_model,
            target_transform,
            xgb_feature_names=best_feature_names,
        )

    old_n_jobs = None
    try:
        prepared = feature_df.copy()
        if best_feature_names:
            prepared = align_features(prepared, best_feature_names)
        else:
            expected = getattr(best_model, "n_features_in_", None)
            if isinstance(expected, int):
                prepared = _align_by_count(prepared, expected)
        prepared = _sanitize_feature_frame(prepared)

        old_n_jobs = getattr(best_model, "n_jobs", None)
        if isinstance(old_n_jobs, int) and old_n_jobs != 1:
            try:
                best_model.n_jobs = 1
            except Exception:
                old_n_jobs = None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            preds = np.asarray(best_model.predict(prepared)).reshape(-1)
        preds = _apply_inverse_target_transform(preds, target_transform, best_model_meta=best_model_meta)
        return preds
    except Exception:
        return np.array([])
    finally:
        if old_n_jobs not in (None, 1):
            try:
                best_model.n_jobs = old_n_jobs
            except Exception:
                pass


def predict_runoff(feature_df: pd.DataFrame, artifacts: Dict[str, Any], model_choice: str = "Random Forest") -> np.ndarray:
    rf_model = artifacts.get("rf_model")
    lstm_model = artifacts.get("lstm_model")
    scaler = artifacts.get("scaler")
    best_model_obj = artifacts.get("best_model_obj")
    best_model_name = str(artifacts.get("best_model_name", ""))
    xgb_model = artifacts.get("xgb_model")
    target_transform = str(artifacts.get("target_transform", "none"))
    best_model_meta = artifacts.get("best_model") if isinstance(artifacts.get("best_model"), dict) else {}
    model_metadata = artifacts.get("model_metadata") if isinstance(artifacts.get("model_metadata"), dict) else {}
    lstm_target_transform = str(model_metadata.get("lstm_target_transform", "none")).strip().lower()
    feature_bounds = _extract_feature_bounds_from_artifacts(artifacts)
    rf_feature_names = _normalize_feature_list(artifacts.get("features"))
    best_feature_names = _normalize_feature_list(artifacts.get("registry_feature_cols"))
    xgb_feature_names = best_feature_names
    validated_features = _sanitize_feature_frame(feature_df, feature_bounds=feature_bounds)
    choice = str(model_choice).strip().lower()

    if choice in {"best model", "xgboost (best)", "xgboost", "xgboost_log1p"}:
        best_preds = _predict_with_best_model(
            validated_features,
            best_model_obj,
            best_model_name,
            target_transform,
            best_feature_names=best_feature_names,
            best_model_meta=best_model_meta,
        )
        if best_preds.size > 0:
            return best_preds
        xgb_preds = _predict_with_xgboost(
            validated_features,
            xgb_model,
            target_transform,
            xgb_feature_names=xgb_feature_names,
        )
        if xgb_preds.size > 0:
            return xgb_preds

    if choice == "lstm":
        if lstm_model is None:
            lstm_model = ensure_lstm_loaded(artifacts)
        lstm_preds = _predict_with_lstm(
            validated_features,
            lstm_model,
            lstm_feature_names=rf_feature_names,
            scaler=scaler,
            lstm_target_transform=lstm_target_transform,
        )
        if lstm_preds.size > 0:
            return _apply_target_mode_to_predictions(lstm_preds, validated_features, model_metadata=model_metadata)
        rf_preds = _predict_with_rf(validated_features, rf_model, rf_feature_names=rf_feature_names)
        if rf_preds.size > 0:
            return _apply_target_mode_to_predictions(rf_preds, validated_features, model_metadata=model_metadata)
        xgb_preds = _predict_with_xgboost(
            validated_features,
            xgb_model,
            target_transform,
            xgb_feature_names=xgb_feature_names,
        )
        if xgb_preds.size > 0:
            return xgb_preds

    if choice in {"random forest", "rf"} and (rf_model is None or not hasattr(rf_model, "predict")):
        xgb_preds = _predict_with_xgboost(
            validated_features,
            xgb_model,
            target_transform,
            xgb_feature_names=xgb_feature_names,
        )
        if xgb_preds.size > 0:
            return xgb_preds

    if choice not in {"random forest", "rf", "lstm"}:
        best_preds = _predict_with_best_model(
            validated_features,
            best_model_obj,
            best_model_name,
            target_transform,
            best_feature_names=best_feature_names,
            best_model_meta=best_model_meta,
        )
        if best_preds.size > 0:
            return best_preds
        xgb_preds = _predict_with_xgboost(
            validated_features,
            xgb_model,
            target_transform,
            xgb_feature_names=xgb_feature_names,
        )
        if xgb_preds.size > 0:
            return xgb_preds

    rf_preds = _predict_with_rf(validated_features, rf_model, rf_feature_names=rf_feature_names)
    return _apply_target_mode_to_predictions(rf_preds, validated_features, model_metadata=model_metadata)


def predict_batch(batch_df: pd.DataFrame, artifacts: Dict[str, Any], model_choice: str = "Random Forest") -> pd.DataFrame:
    feature_list = get_feature_list(artifacts)
    prepared = align_features(batch_df, feature_list) if feature_list else batch_df.select_dtypes(include=["number"])
    predictions = predict_runoff(prepared, artifacts, model_choice=model_choice)

    output_df = batch_df.copy()
    output_df["predicted_discharge_cumec"] = predictions
    return output_df


def get_feature_importance(artifacts: Dict[str, Any], fallback_features: list[str] | None = None) -> pd.DataFrame:
    rf_model = artifacts.get("rf_model")
    feature_list = _normalize_feature_list(artifacts.get("features"))

    if not feature_list and fallback_features:
        feature_list = fallback_features

    if rf_model is None or not hasattr(rf_model, "feature_importances_") or not feature_list:
        return pd.DataFrame(columns=["feature", "importance"])

    try:
        importances = np.asarray(rf_model.feature_importances_).reshape(-1)
    except Exception:
        return pd.DataFrame(columns=["feature", "importance"])

    if len(importances) != len(feature_list):
        size = min(len(importances), len(feature_list))
        importances = importances[:size]
        feature_list = feature_list[:size]

    imp_df = pd.DataFrame({"feature": feature_list, "importance": importances})
    return imp_df.sort_values("importance", ascending=False)
