from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit

from utils.preprocessing import get_discharge_normalization_report, normalize_discharge_dataframe
from xgboost import XGBRegressor


SearchType = Literal["grid", "randomized"]
TargetTransform = Literal["none", "log1p"]
CANONICAL_FEATURE_COLS = [
    "Mean_PCP",
    "Mean_Tmax",
    "Mean_Tmin",
    "rh",
    "solar",
    "wind",
    "lag_discharge_1",
    "lag_discharge_2",
    "lag_discharge_3",
    "season_sin",
    "season_cos",
    "discharge_roll_mean_3",
    "discharge_roll_std_3",
    "PCP_roll_mean_3",
    "PCP_roll_sum_7",
]


@dataclass
class ModelMetrics:
    rmse: float
    mae: float
    r2: float
    nse: float


@dataclass
class TunedModelResult:
    model_name: str
    estimator: Any
    best_params: dict[str, Any]
    cv_score_rmse: float
    metrics: ModelMetrics
    target_transform: TargetTransform


def nse(observed: np.ndarray, predicted: np.ndarray) -> float:
    observed = np.asarray(observed)
    predicted = np.asarray(predicted)
    denominator = np.sum((observed - np.mean(observed)) ** 2)
    if denominator == 0:
        return float("nan")
    return float(1 - (np.sum((observed - predicted) ** 2) / denominator))


def evaluate_regression(y_true: Any, y_pred: Any) -> ModelMetrics:
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    rmse_value = float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)))
    mae_value = float(mean_absolute_error(y_true_arr, y_pred_arr))
    r2_value = float(r2_score(y_true_arr, y_pred_arr))
    nse_value = nse(y_true_arr, y_pred_arr)
    return ModelMetrics(rmse=rmse_value, mae=mae_value, r2=r2_value, nse=nse_value)


def _resolve_target_transform(y_train: pd.Series, target_transform: TargetTransform) -> TargetTransform:
    if target_transform != "log1p":
        return "none"
    y_min = float(np.nanmin(np.asarray(y_train, dtype=float)))
    return "log1p" if y_min >= 0 else "none"


def _apply_target_transform(y: pd.Series, target_transform: TargetTransform) -> np.ndarray:
    y_array = np.asarray(y, dtype=float)
    if target_transform == "log1p":
        return np.log1p(y_array)
    return y_array


def _inverse_target_transform(y: np.ndarray, target_transform: TargetTransform) -> np.ndarray:
    if target_transform == "log1p":
        return np.expm1(y)
    return y


def _build_sample_weights(y_train: pd.Series) -> np.ndarray:
    y_array = np.asarray(y_train, dtype=float)
    weights = np.ones_like(y_array, dtype=float)
    q90 = float(np.nanquantile(y_array, 0.90))
    q99 = float(np.nanquantile(y_array, 0.99))
    weights[y_array >= q90] = 2.0
    weights[y_array >= q99] = 3.0
    return weights


def _resolve_cv_splits(requested_splits: int, n_train_samples: int) -> int:
    if n_train_samples <= 10:
        return 2
    upper_bound = min(6, n_train_samples - 1)
    return max(2, min(requested_splits, upper_bound))


def _build_search(
    estimator: Any,
    param_grid: dict[str, list[Any]],
    cv: TimeSeriesSplit,
    random_state: int,
    search_type: SearchType,
    n_iter: int,
    n_jobs: int,
) -> GridSearchCV | RandomizedSearchCV:
    common_kwargs = {
        "estimator": estimator,
        "cv": cv,
        "scoring": "neg_root_mean_squared_error",
        "n_jobs": n_jobs,
        "verbose": 1,
    }

    if search_type == "grid":
        return GridSearchCV(param_grid=param_grid, **common_kwargs)

    return RandomizedSearchCV(
        param_distributions=param_grid,
        n_iter=min(n_iter, int(np.prod([len(v) for v in param_grid.values()]))),
        random_state=random_state,
        **common_kwargs,
    )


def tune_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int = 42,
    cv_splits: int = 3,
    search_type: SearchType = "randomized",
    n_iter: int = 20,
    n_jobs: int = -1,
    target_transform: TargetTransform = "none",
) -> TunedModelResult:
    rf = RandomForestRegressor(random_state=random_state, n_jobs=-1)
    rf_grid = {
        "n_estimators": [300, 600, 900, 1200],
        "max_depth": [None, 10, 14, 18],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", 0.7, 1.0],
    }

    resolved_transform = _resolve_target_transform(y_train, target_transform)
    y_train_fit = _apply_target_transform(y_train, resolved_transform)
    sample_weights = _build_sample_weights(y_train)

    cv = TimeSeriesSplit(n_splits=cv_splits)
    search = _build_search(
        estimator=rf,
        param_grid=rf_grid,
        cv=cv,
        random_state=random_state,
        search_type=search_type,
        n_iter=n_iter,
        n_jobs=n_jobs,
    )
    search.fit(X_train, y_train_fit, sample_weight=sample_weights)

    best_estimator = search.best_estimator_
    y_pred = np.asarray(best_estimator.predict(X_test)).reshape(-1)  # type: ignore
    y_pred = _inverse_target_transform(y_pred, resolved_transform)
    metrics = evaluate_regression(y_test, y_pred)

    model_name = "random_forest" if resolved_transform == "none" else "random_forest_log1p"
    return TunedModelResult(
        model_name=model_name,
        estimator=best_estimator,
        best_params=search.best_params_,
        cv_score_rmse=float(-search.best_score_),
        metrics=metrics,
        target_transform=resolved_transform,
    )


def tune_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int = 42,
    cv_splits: int = 3,
    search_type: SearchType = "randomized",
    n_iter: int = 20,
    n_jobs: int = -1,
    target_transform: TargetTransform = "none",
) -> TunedModelResult:
    xgb = XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        random_state=random_state,
        n_jobs=n_jobs,
    )
    xgb_grid = {
        "n_estimators": [300, 500, 800, 1100],
        "learning_rate": [0.01, 0.03, 0.05, 0.08],
        "max_depth": [3, 5, 7, 9],
        "min_child_weight": [1, 3, 5],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.65, 0.8, 1.0],
        "reg_alpha": [0.0, 0.1, 0.4],
        "reg_lambda": [1.0, 2.0, 5.0],
        "gamma": [0.0, 0.1, 0.3],
    }

    resolved_transform = _resolve_target_transform(y_train, target_transform)
    y_train_fit = _apply_target_transform(y_train, resolved_transform)
    sample_weights = _build_sample_weights(y_train)

    cv = TimeSeriesSplit(n_splits=cv_splits)
    search = _build_search(
        estimator=xgb,
        param_grid=xgb_grid,
        cv=cv,
        random_state=random_state,
        search_type=search_type,
        n_iter=n_iter,
        n_jobs=n_jobs,
    )
    search.fit(X_train, y_train_fit, sample_weight=sample_weights)

    best_estimator = search.best_estimator_
    y_pred = np.asarray(best_estimator.predict(X_test)).reshape(-1)  # type: ignore
    y_pred = _inverse_target_transform(y_pred, resolved_transform)
    metrics = evaluate_regression(y_test, y_pred)

    model_name = "xgboost" if resolved_transform == "none" else "xgboost_log1p"
    return TunedModelResult(
        model_name=model_name,
        estimator=best_estimator,
        best_params=search.best_params_,
        cv_score_rmse=float(-search.best_score_),
        metrics=metrics,
        target_transform=resolved_transform,
    )


def tune_extra_trees(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int = 42,
    cv_splits: int = 3,
    search_type: SearchType = "randomized",
    n_iter: int = 20,
    n_jobs: int = -1,
    target_transform: TargetTransform = "none",
) -> TunedModelResult:
    extra = ExtraTreesRegressor(random_state=random_state, n_jobs=n_jobs)
    extra_grid = {
        "n_estimators": [300, 600, 900, 1200],
        "max_depth": [None, 10, 14, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", 0.7, 1.0],
    }

    resolved_transform = _resolve_target_transform(y_train, target_transform)
    y_train_fit = _apply_target_transform(y_train, resolved_transform)
    sample_weights = _build_sample_weights(y_train)

    cv = TimeSeriesSplit(n_splits=cv_splits)
    search = _build_search(
        estimator=extra,
        param_grid=extra_grid,
        cv=cv,
        random_state=random_state,
        search_type=search_type,
        n_iter=n_iter,
        n_jobs=n_jobs,
    )
    search.fit(X_train, y_train_fit, sample_weight=sample_weights)

    best_estimator = search.best_estimator_
    y_pred = np.asarray(best_estimator.predict(X_test)).reshape(-1)  # type: ignore
    y_pred = _inverse_target_transform(y_pred, resolved_transform)
    metrics = evaluate_regression(y_test, y_pred)

    model_name = "extra_trees" if resolved_transform == "none" else "extra_trees_log1p"
    return TunedModelResult(
        model_name=model_name,
        estimator=best_estimator,
        best_params=search.best_params_,
        cv_score_rmse=float(-search.best_score_),
        metrics=metrics,
        target_transform=resolved_transform,
    )


def build_lstm_early_stopping(patience: int = 5):
    try:
        from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
    except Exception:
        from keras.callbacks import EarlyStopping

    return EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
    )


def train_lstm_with_early_stopping(
    model,
    X_train: np.ndarray,
    y_train_scaled: np.ndarray,
    validation_split: float = 0.1,
    epochs: int = 50,
    batch_size: int = 32,
    patience: int = 5,
    verbose: int = 1,
):
    early_stop = build_lstm_early_stopping(patience=patience)
    history = model.fit(
        X_train,
        y_train_scaled,
        validation_split=validation_split,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=verbose,
    )
    model.history = history
    return history


def _save_model(model_name: str, estimator: Any, models_dir: Path) -> str:
    models_dir.mkdir(parents=True, exist_ok=True)

    normalized_name = model_name.lower().replace(" ", "_")
    if normalized_name.startswith("xgboost"):
        file_path = models_dir / f"best_{normalized_name}.json"
        estimator.save_model(str(file_path))
        return file_path.as_posix()

    file_path = models_dir / f"best_{normalized_name}.joblib"
    import joblib

    joblib.dump(estimator, file_path)
    return file_path.as_posix()


def _update_model_registry(
    best_result: TunedModelResult,
    comparison: list[dict[str, Any]],
    artifact_path: str,
    registry_path: Path,
    feature_cols: list[str],
    target_col: str,
    discharge_report: dict[str, Any] | None = None,
) -> None:
    if registry_path.exists():
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    else:
        registry = {}

    registry["best_model"] = {
        "model_name": best_result.model_name,
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "artifact_path": artifact_path,
        "best_params": best_result.best_params,
        "cv_score_rmse": best_result.cv_score_rmse,
        "target_transform": best_result.target_transform,
        "feature_cols": feature_cols,
        "target_col": target_col,
        "target_unit": "CUMEC",
        "target_already_normalized_to_cumec": True,
        "source_discharge_assumption": (discharge_report or {}).get("assumed_unit", "unknown"),
        "source_discharge_factor_to_cumec": float((discharge_report or {}).get("applied_factor_to_cumec", 1.0) or 1.0),
        "final_metrics": asdict(best_result.metrics),
    }
    registry["model_comparison_rmse"] = comparison

    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")


def _prepare_splits(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    date_col: str,
    train_end_year: int,
    test_start_year: int,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    if not feature_cols:
        raise ValueError("feature_cols cannot be empty.")
    if test_start_year <= train_end_year:
        raise ValueError(
            "test_start_year must be greater than train_end_year to avoid train/test overlap."
        )

    work_df = df.copy()
    work_df[date_col] = pd.to_datetime(work_df[date_col], errors="coerce")
    work_df = work_df.dropna(subset=[date_col, target_col]).copy()
    work_df["Year"] = work_df[date_col].dt.year

    train_df = work_df[work_df["Year"] <= train_end_year].copy()
    test_df = work_df[work_df["Year"] >= test_start_year].copy()

    if train_df.empty or test_df.empty:
        raise ValueError(
            "Train/Test split produced empty dataframes. "
            f"Check train_end_year={train_end_year} and test_start_year={test_start_year}."
        )

    X_train = train_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    X_test = test_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    y_train = pd.to_numeric(train_df[target_col], errors="coerce")
    y_test = pd.to_numeric(test_df[target_col], errors="coerce")

    valid_train = y_train.notna()
    valid_test = y_test.notna()
    X_train = X_train.loc[valid_train].copy()
    y_train = y_train.loc[valid_train].copy()
    X_test = X_test.loc[valid_test].copy()
    y_test = y_test.loc[valid_test].copy()

    train_medians = X_train.median(numeric_only=True)
    X_train = X_train.fillna(train_medians).fillna(0.0)
    X_test = X_test.fillna(train_medians).fillna(0.0)

    if X_train.empty or X_test.empty:
        raise ValueError("No samples available after cleaning and imputation.")

    return X_train, y_train, X_test, y_test


def _resolve_column_name(df: pd.DataFrame, preferred: str, fallback: list[str] | None = None) -> str | None:
    choices = [preferred] + (fallback or [])
    normalized = {str(col).strip().lower(): str(col) for col in df.columns}
    for choice in choices:
        key = str(choice).strip().lower()
        if key in normalized:
            return normalized[key]
    return None


def _prepare_modeling_frame(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    date_col: str,
) -> tuple[pd.DataFrame, list[str], str, str]:
    work_df = df.copy()
    work_df = work_df.rename(columns={col: str(col).strip() for col in work_df.columns})
    work_df = normalize_discharge_dataframe(work_df)

    resolved_date_col = _resolve_column_name(work_df, date_col, ["DATE", "Date", "date", "Datetime", "datetime"])
    if resolved_date_col is None:
        raise ValueError("Date column not found for modeling.")

    resolved_target_col = _resolve_column_name(
        work_df,
        target_col,
        ["Discharge (CUMEC)", "Discharge", "discharge", "DisCUMEC"],
    )
    if resolved_target_col is None:
        raise ValueError("Target discharge column not found for modeling.")

    work_df[resolved_date_col] = pd.to_datetime(work_df[resolved_date_col], errors="coerce")
    work_df = work_df.sort_values(resolved_date_col).reset_index(drop=True)

    for col in ["PCP", "P1", "P2", "P3", "TMAX", "TMIN", "rh", "solar", "wind", "wind "]:
        if col in work_df.columns:
            work_df[col] = pd.to_numeric(work_df[col], errors="coerce")

    if "Mean_PCP" not in work_df.columns:
        if "PCP" in work_df.columns:
            work_df["Mean_PCP"] = work_df["PCP"]
        else:
            precip_parts = [c for c in ["P1", "P2", "P3"] if c in work_df.columns]
            if precip_parts:
                work_df["Mean_PCP"] = work_df[precip_parts].mean(axis=1)

    if "Mean_Tmax" not in work_df.columns and "TMAX" in work_df.columns:
        work_df["Mean_Tmax"] = work_df["TMAX"]
    if "Mean_Tmin" not in work_df.columns and "TMIN" in work_df.columns:
        work_df["Mean_Tmin"] = work_df["TMIN"]

    if "wind" not in work_df.columns and "wind " in work_df.columns:
        work_df["wind"] = work_df["wind "]

    discharge_series = pd.to_numeric(work_df[resolved_target_col], errors="coerce")
    work_df["lag_discharge_1"] = discharge_series.shift(1)
    work_df["lag_discharge_2"] = discharge_series.shift(2)
    work_df["lag_discharge_3"] = discharge_series.shift(3)

    discharge_hist = discharge_series.shift(1)
    work_df["discharge_roll_mean_3"] = discharge_hist.rolling(window=3, min_periods=1).mean()
    work_df["discharge_roll_std_3"] = discharge_hist.rolling(window=3, min_periods=2).std(ddof=0)

    precip_base = pd.to_numeric(work_df.get("Mean_PCP"), errors="coerce") if "Mean_PCP" in work_df.columns else None
    if precip_base is not None:
        precip_hist = precip_base.shift(1)
        work_df["PCP_roll_mean_3"] = precip_hist.rolling(window=3, min_periods=1).mean()
        work_df["PCP_roll_sum_7"] = precip_hist.rolling(window=7, min_periods=1).sum()

    day_of_year = work_df[resolved_date_col].dt.dayofyear
    angle = (2 * np.pi * day_of_year) / 365.25
    work_df["season_sin"] = np.sin(angle)
    work_df["season_cos"] = np.cos(angle)

    available_canonical = [col for col in CANONICAL_FEATURE_COLS if col in work_df.columns]
    if len(available_canonical) == len(CANONICAL_FEATURE_COLS):
        resolved_feature_cols = CANONICAL_FEATURE_COLS.copy()
    else:
        numeric_existing = [
            col
            for col in feature_cols
            if col in work_df.columns and pd.api.types.is_numeric_dtype(work_df[col])
        ]
        resolved_feature_cols = numeric_existing or available_canonical

    if not resolved_feature_cols:
        raise ValueError("No usable feature columns available after feature engineering.")

    return work_df, resolved_feature_cols, resolved_target_col, resolved_date_col


def auto_train_best_model(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "Target_t_plus_3",
    date_col: str = "DATE",
    train_end_year: int = 2000,
    test_start_year: int = 2006,
    random_state: int = 42,
    search_type: SearchType = "randomized",
    n_iter: int = 20,
    cv_splits: int = 3,
    n_jobs: int = -1,
    registry_path: str = "model_registry.json",
    models_dir: str = "models",
) -> dict[str, Any]:
    """
    Train and tune RF + XGBoost using TimeSeriesSplit, compare by RMSE,
    register best model in model_registry.json, and return full summary.
    """

    prepared_df, resolved_feature_cols, resolved_target_col, resolved_date_col = _prepare_modeling_frame(
        df=df,
        feature_cols=feature_cols,
        target_col=target_col,
        date_col=date_col,
    )
    discharge_report = get_discharge_normalization_report(prepared_df, resolved_target_col)

    X_train, y_train, X_test, y_test = _prepare_splits(
        df=prepared_df,
        feature_cols=resolved_feature_cols,
        target_col=resolved_target_col,
        date_col=resolved_date_col,
        train_end_year=train_end_year,
        test_start_year=test_start_year,
    )

    effective_cv_splits = _resolve_cv_splits(cv_splits, len(X_train))

    candidate_results: list[TunedModelResult] = []
    for transform in ("none", "log1p"):
        candidate_results.append(
            tune_random_forest(
                X_train,
                y_train,
                X_test,
                y_test,
                random_state=random_state,
                cv_splits=effective_cv_splits,
                search_type=search_type,
                n_iter=n_iter,
                n_jobs=n_jobs,
                target_transform=transform,  # type: ignore[arg-type]
            )
        )
        candidate_results.append(
            tune_xgboost(
                X_train,
                y_train,
                X_test,
                y_test,
                random_state=random_state,
                cv_splits=effective_cv_splits,
                search_type=search_type,
                n_iter=n_iter,
                n_jobs=n_jobs,
                target_transform=transform,  # type: ignore[arg-type]
            )
        )
        candidate_results.append(
            tune_extra_trees(
                X_train,
                y_train,
                X_test,
                y_test,
                random_state=random_state,
                cv_splits=effective_cv_splits,
                search_type=search_type,
                n_iter=n_iter,
                n_jobs=n_jobs,
                target_transform=transform,  # type: ignore[arg-type]
            )
        )

    deduped_results: dict[str, TunedModelResult] = {}
    for result in candidate_results:
        existing = deduped_results.get(result.model_name)
        if existing is None or result.metrics.rmse < existing.metrics.rmse:
            deduped_results[result.model_name] = result

    results = list(deduped_results.values())
    results_sorted = sorted(results, key=lambda item: item.metrics.rmse)
    best_result = results_sorted[0]

    artifact_path = _save_model(
        model_name=best_result.model_name,
        estimator=best_result.estimator,
        models_dir=Path(models_dir),
    )

    comparison = [
        {
            "model_name": r.model_name,
            "rmse": r.metrics.rmse,
            "mae": r.metrics.mae,
            "r2": r.metrics.r2,
            "nse": r.metrics.nse,
            "cv_score_rmse": r.cv_score_rmse,
            "best_params": r.best_params,
            "target_transform": r.target_transform,
            "feature_cols": resolved_feature_cols,
            "target_col": resolved_target_col,
        }
        for r in results_sorted
    ]

    _update_model_registry(
        best_result=best_result,
        comparison=comparison,
        artifact_path=artifact_path,
        registry_path=Path(registry_path),
        feature_cols=resolved_feature_cols,
        target_col=resolved_target_col,
        discharge_report=discharge_report,
    )

    return {
        "best_model": comparison[0],
        "all_models": comparison,
        "artifact_path": artifact_path,
        "registry_path": Path(registry_path).as_posix(),
    }
