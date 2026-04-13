from __future__ import annotations

import importlib.util
import inspect
import pickle
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.config import settings
from app.core.exceptions import BackendException
from app.core.logger import get_logger
from app.schemas import TrainAutoRequest
from app.services.registry_service import write_registry


logger = get_logger(__name__)
REALISTIC_DISCHARGE_MAX_CUMEC = 500000.0
SECONDS_PER_DAY = 86400.0


def _load_function_from_file(file_path: Path, module_name: str, function_name: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise BackendException(500, f"Could not load module spec from {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    fn = getattr(module, function_name, None)
    if fn is None:
        raise BackendException(500, f"Function {function_name} not found in {file_path}")
    return fn


def _get_training_functions():
    backend_training_file = settings.BASE_DIR / "model_training_backend.py"
    root_training_file = settings.PROJECT_ROOT / "model_training_backend.py"
    backend_viz_file = settings.BASE_DIR / "visualization.py"
    root_viz_file = settings.PROJECT_ROOT / "visualization.py"

    training_file = root_training_file if root_training_file.exists() else backend_training_file
    viz_file = root_viz_file if root_viz_file.exists() else backend_viz_file

    if not training_file.exists():
        raise BackendException(
            500,
            "model_training_backend.py not found in backend/ or project root",
        )
    if not viz_file.exists():
        raise BackendException(
            500,
            "visualization.py not found in backend/ or project root",
        )

    auto_train = _load_function_from_file(
        training_file,
        "model_training_backend_module",
        "auto_train_best_model",
    )
    generate_plots = _load_function_from_file(
        viz_file,
        "visualization_module",
        "generate_all_plots",
    )
    return auto_train, generate_plots


def _discover_dataset() -> Path:
    candidates: list[Path] = []
    for data_dir in settings.DEFAULT_DATA_DIRS:
        if data_dir.exists() and data_dir.is_dir():
            candidates.extend(sorted(data_dir.glob("*.csv")))
            candidates.extend(sorted(data_dir.glob("*.xlsx")))

    if not candidates:
        raise BackendException(404, "No dataset file found in configured data folders")

    kasol_xlsx = [p for p in candidates if p.stem.lower() == "kasol" and p.suffix.lower() == ".xlsx"]
    if kasol_xlsx:
        return kasol_xlsx[0]

    for file_path in candidates:
        if file_path.stem.lower() == "kasol":
            return file_path

    return candidates[0]


def _normalize_name(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _detect_discharge_column(df: pd.DataFrame) -> str | None:
    preferred = ["Discharge (CUMEC)", "Discharge", "DisCUMEC", "discharge", "runoff", "cumec", "streamflow", "flow"]
    lowered = {_normalize_name(col): col for col in df.columns}
    for candidate in preferred:
        resolved = lowered.get(_normalize_name(candidate))
        if resolved is not None:
            return resolved
    for col in df.select_dtypes(include=["number"]).columns:
        normalized = _normalize_name(col)
        if any(token in normalized for token in ("discharge", "cumec", "runoff", "streamflow", "flow")):
            return str(col)
    return None


def _infer_discharge_factor(series: pd.Series, column_name: str | None) -> float:
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return 1.0

    positive = clean[clean > 0]
    sample = positive if not positive.empty else clean.abs()
    median_val = float(sample.median())
    normalized_col = _normalize_name(column_name or "")

    if any(token in normalized_col for token in ("literpersecond", "litrepersecond", "lps", "lsec")):
        return 0.001
    if any(token in normalized_col for token in ("m3perday", "m3day", "dailyvolume", "cumecday", "m3d")):
        return 1.0 / SECONDS_PER_DAY
    if median_val <= REALISTIC_DISCHARGE_MAX_CUMEC:
        return 1.0
    if median_val >= 1000000.0 and (median_val / SECONDS_PER_DAY) <= REALISTIC_DISCHARGE_MAX_CUMEC:
        return 1.0 / SECONDS_PER_DAY
    if (median_val / 1000.0) <= REALISTIC_DISCHARGE_MAX_CUMEC:
        return 0.001
    if (median_val / 1000000.0) <= REALISTIC_DISCHARGE_MAX_CUMEC:
        return 0.000001
    return 1.0


def _normalize_discharge_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    working = df.copy()
    discharge_col = _detect_discharge_column(working)
    if discharge_col is None or discharge_col not in working.columns:
        return working
    factor = _infer_discharge_factor(working[discharge_col], discharge_col)
    if factor != 1.0:
        working[discharge_col] = pd.to_numeric(working[discharge_col], errors="coerce") * factor
    return working


def load_training_dataset() -> pd.DataFrame:
    dataset_path = _discover_dataset()
    if dataset_path.suffix.lower() == ".xlsx":
        excel_file = pd.ExcelFile(dataset_path)
        sheet_name: str | None = None

        if "Sheet1" in excel_file.sheet_names:
            sheet_name = "Sheet1"
        else:
            for candidate_sheet in excel_file.sheet_names:
                probe = pd.read_excel(dataset_path, sheet_name=candidate_sheet, nrows=1)
                cols = {str(col).strip().upper() for col in probe.columns}
                if "DATE" in cols:
                    sheet_name = candidate_sheet
                    break

        dataframe = pd.read_excel(dataset_path, sheet_name=sheet_name)
    else:
        dataframe = pd.read_csv(dataset_path)

    if "DATE" in dataframe.columns:
        dataframe["DATE"] = pd.to_datetime(dataframe["DATE"], errors="coerce")
        dataframe = dataframe.sort_values("DATE").reset_index(drop=True)

    return _normalize_discharge_dataframe(dataframe)


def _coerce_plot_result(plot_result: Any) -> list[str]:
    if plot_result is None:
        return []
    if isinstance(plot_result, str):
        return [plot_result]
    if isinstance(plot_result, (list, tuple, set)):
        return [str(item) for item in plot_result]
    if isinstance(plot_result, dict):
        values: list[str] = []
        for value in plot_result.values():
            if isinstance(value, (list, tuple, set)):
                values.extend([str(item) for item in value])
            elif value is not None:
                values.append(str(value))
        return values
    return [str(plot_result)]


def _resolve_artifact_path(train_result: dict[str, Any]) -> Path:
    artifact_path = train_result.get("artifact_path") or train_result.get("best_model_path")
    if not artifact_path:
        raise ValueError("No artifact path present in training result")

    path = Path(str(artifact_path))
    if not path.is_absolute():
        base_candidate = settings.BASE_DIR / path
        root_candidate = settings.PROJECT_ROOT / path
        if base_candidate.exists():
            path = base_candidate
        elif root_candidate.exists():
            path = root_candidate

    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found at {path}")

    return path


def _load_estimator(path: Path, model_name: str | None):
    suffix = path.suffix.lower()
    if suffix in {".joblib", ".pkl", ".pickle"}:
        try:
            return joblib.load(path)
        except Exception:
            with path.open("rb") as file_obj:
                return pickle.load(file_obj)

    normalized_name = (model_name or "").lower()
    if suffix == ".json" and normalized_name.startswith("xgboost"):
        try:
            import xgboost as xgb

            model = xgb.XGBRegressor()
            model.load_model(str(path))
            return model
        except Exception as exc:
            raise RuntimeError(f"Unable to load XGBoost artifact: {exc}") from exc

    with path.open("rb") as file_obj:
        return pickle.load(file_obj)


def _extract_training_columns(df: pd.DataFrame) -> tuple[str, list[str]]:
    target_col = "Discharge" if "Discharge" in df.columns else "Discharge (CUMEC)"
    feature_cols = [
        col
        for col in df.columns
        if col not in {"DATE", "Target_t_plus_3", "Discharge", "Discharge (CUMEC)"}
        and pd.api.types.is_numeric_dtype(df[col])
    ]
    return target_col, feature_cols


def _prepare_plot_inputs(
    train_result: dict[str, Any],
    df: pd.DataFrame,
    payload: TrainAutoRequest,
) -> dict[str, Any]:
    target_col, feature_cols = _extract_training_columns(df)
    if target_col not in df.columns:
        raise ValueError(f"Target column not found in dataset: {target_col}")
    if not feature_cols:
        raise ValueError("No numeric feature columns available for plotting")

    artifact_path = _resolve_artifact_path(train_result)
    model_name = train_result.get("best_model", {}).get("model_name")
    target_transform = str(train_result.get("best_model", {}).get("target_transform", "none")).lower()
    model = _load_estimator(artifact_path, model_name)

    eval_df = df.copy()
    if "DATE" in eval_df.columns:
        eval_df["DATE"] = pd.to_datetime(eval_df["DATE"], errors="coerce")
        eval_df["Year"] = eval_df["DATE"].dt.year
        if payload.test_start_year is not None:
            filtered_df = eval_df[eval_df["Year"] >= payload.test_start_year].copy()
            if not filtered_df.empty:
                eval_df = filtered_df

    eval_df = eval_df.dropna(subset=feature_cols + [target_col])
    if eval_df.empty:
        raise ValueError("No rows available after dropping missing values for plotting")

    X_eval = eval_df[feature_cols]
    y_true = eval_df[target_col]
    y_pred = np.asarray(model.predict(X_eval)).reshape(-1)
    if target_transform == "log1p":
        y_pred = np.expm1(np.maximum(y_pred, 0.0))

    return {
        "model": model,
        "X": X_eval,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def _first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _invoke_auto_train(df: pd.DataFrame, payload: TrainAutoRequest) -> dict[str, Any]:
    auto_train_best_model, _ = _get_training_functions()
    signature = inspect.signature(auto_train_best_model)
    target_col, feature_cols = _extract_training_columns(df)

    arg_map: dict[str, Any] = {
        "df": df,
        "train_end_year": payload.train_end_year,
        "test_start_year": payload.test_start_year,
        "search_type": payload.search_type,
        "n_iter": payload.n_iter,
        "test_years_start": payload.test_start_year,
        "test_years_end": None,
        "target_col": target_col,
        "feature_cols": feature_cols,
        "train_size": 0.8,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": 1,
    }

    kwargs: dict[str, Any] = {}
    missing_required: list[str] = []

    for name, param in signature.parameters.items():
        if name in arg_map and arg_map[name] is not None:
            kwargs[name] = arg_map[name]
        elif param.default is inspect._empty:
            missing_required.append(name)

    if missing_required:
        raise BackendException(
            500,
            f"auto_train_best_model requires unsupported parameters: {', '.join(missing_required)}",
        )

    return auto_train_best_model(**kwargs)


def _invoke_generate_plots(
    train_result: dict[str, Any],
    df: pd.DataFrame,
    payload: TrainAutoRequest,
) -> list[str]:
    _, generate_all_plots = _get_training_functions()
    settings.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    signature = inspect.signature(generate_all_plots)

    plot_inputs: dict[str, Any] = {}
    try:
        plot_inputs = _prepare_plot_inputs(train_result, df, payload)
    except Exception as exc:
        logger.warning("Skipping plot generation input preparation: %s", exc)

    candidates: dict[str, Any] = {
        "output_dir": str(settings.PLOTS_DIR),
        "save_dir": str(settings.PLOTS_DIR),
        "result": train_result,
        "results": train_result,
        "best_model_result": train_result,
        "best_model": train_result.get("best_model"),
        "all_models": train_result.get("all_models"),
        "df": df,
        "data": df,
        "metrics": train_result.get("metrics"),
        "y_test": train_result.get("y_test"),
        "y_true": _first_not_none(plot_inputs.get("y_true"), train_result.get("y_test")),
        "y_pred": _first_not_none(plot_inputs.get("y_pred"), train_result.get("y_pred")),
        "model": plot_inputs.get("model"),
        "X": _first_not_none(plot_inputs.get("X"), df),
        "residuals": train_result.get("residuals"),
        "history": train_result.get("history"),
    }

    kwargs: dict[str, Any] = {}
    missing_required: list[str] = []

    for name, param in signature.parameters.items():
        if name in candidates and candidates[name] is not None:
            kwargs[name] = candidates[name]
        elif param.default is inspect._empty:
            missing_required.append(name)

    if missing_required:
        logger.warning(
            "Skipping plot generation due to missing parameters: %s",
            ", ".join(missing_required),
        )
        return []

    try:
        result = generate_all_plots(**kwargs)
        return _coerce_plot_result(result)
    except Exception as exc:
        logger.warning("Plot generation failed and was skipped: %s", exc)
        return []


def run_auto_training(payload: TrainAutoRequest) -> dict[str, Any]:
    logger.info("Training started | train_end_year=%s test_start_year=%s search_type=%s n_iter=%s", payload.train_end_year, payload.test_start_year, payload.search_type, payload.n_iter)

    dataframe = load_training_dataset()
    train_result = _invoke_auto_train(dataframe, payload)

    if not isinstance(train_result, dict):
        raise BackendException(500, "auto_train_best_model returned unexpected response type")

    best_model = train_result.get("best_model")
    if best_model is None:
        raise BackendException(500, "No best_model key found in training result")

    logger.info("Best model selected | %s", best_model)

    plots = _invoke_generate_plots(train_result, dataframe, payload)

    response = {
        "best_model": best_model,
        "all_models": train_result.get("all_models", []),
        "artifact_path": train_result.get("artifact_path") or train_result.get("best_model_path"),
        "plots": plots,
    }

    registry_payload = {
        "best_model": response["best_model"],
        "all_models": response["all_models"],
        "artifact_path": response["artifact_path"],
    }
    write_registry(registry_payload)

    logger.info("Training completed")
    return response
