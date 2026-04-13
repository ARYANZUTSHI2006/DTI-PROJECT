"""
Training script for runoff forecasting models (Random Forest and LSTM).
Uses observed discharge from Kasol.xlsx (normalized to CUMEC) and trains
models for 3-day ahead forecasting using a stability-aware target:
delta_from_lag1 = discharge_t+3 - lag_discharge_1
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from utils.preprocessing import detect_discharge_column, load_dataset, prepare_features


FORECAST_HORIZON_DAYS = 3
TRAIN_END_YEAR = 2000
TEST_START_YEAR = 2006
RANDOM_STATE = 42
TARGET_MODE = "delta_from_lag1"
TARGET_BASE_FEATURE = "lag_discharge_1"
LSTM_TARGET_TRANSFORM = "none"


def nse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    denom = np.sum((y_true_arr - np.mean(y_true_arr)) ** 2)
    if denom == 0:
        return float("nan")
    return float(1 - (np.sum((y_true_arr - y_pred_arr) ** 2) / denom))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr))),
        "mae": float(mean_absolute_error(y_true_arr, y_pred_arr)),
        "r2": float(r2_score(y_true_arr, y_pred_arr)),
        "nse": nse(y_true_arr, y_pred_arr),
    }


def resolve_dataset_path(base_dir: Path) -> Path:
    preferred = [base_dir / "datasets" / "Kasol.xlsx", base_dir / "datasets" / "Kasol.csv"]
    for path in preferred:
        if path.exists() and path.stat().st_size > 0:
            return path
    raise FileNotFoundError("No dataset found. Expected datasets/Kasol.xlsx or datasets/Kasol.csv")


def resolve_date_column(df: pd.DataFrame) -> str:
    for candidate in ("DATE", "Date", "date"):
        if candidate in df.columns:
            return candidate
    raise ValueError("No date column found in dataset (expected DATE/Date/date).")


def build_modeling_frame(df: pd.DataFrame, horizon_days: int) -> tuple[pd.DataFrame, list[str], str]:
    date_col = resolve_date_column(df)
    target_col = detect_discharge_column(df)
    if target_col is None:
        raise ValueError("No discharge target column found in dataset.")

    frame = df.copy()
    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
    frame = frame.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

    feature_df = prepare_features(frame)
    target_abs = pd.to_numeric(frame[target_col], errors="coerce").shift(-horizon_days)

    if TARGET_BASE_FEATURE not in feature_df.columns:
        raise ValueError(f"Required feature not found for target mode: {TARGET_BASE_FEATURE}")
    base_lag = pd.to_numeric(feature_df[TARGET_BASE_FEATURE], errors="coerce")
    target_delta = target_abs - base_lag

    modeling = feature_df.copy()
    modeling["target_abs"] = target_abs
    modeling["target_delta"] = target_delta
    modeling["year"] = frame[date_col].dt.year
    modeling = modeling.dropna().reset_index(drop=True)

    feature_columns = [col for col in modeling.columns if col not in {"target_abs", "target_delta", "year"}]
    if not feature_columns:
        raise ValueError("No feature columns available after preprocessing.")

    return modeling, feature_columns, target_col


def split_train_test(
    modeling: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    train_mask = modeling["year"] <= TRAIN_END_YEAR
    test_mask = modeling["year"] >= TEST_START_YEAR

    if train_mask.sum() < 300 or test_mask.sum() < 120:
        split_idx = int(len(modeling) * 0.8)
        split_idx = max(1, min(split_idx, len(modeling) - 1))
        train_df = modeling.iloc[:split_idx].copy()
        test_df = modeling.iloc[split_idx:].copy()
        split_info = {
            "strategy": "chronological_80_20_fallback",
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "train_year_max": int(train_df["year"].max()) if len(train_df) else None,
            "test_year_min": int(test_df["year"].min()) if len(test_df) else None,
        }
    else:
        train_df = modeling.loc[train_mask].copy()
        test_df = modeling.loc[test_mask].copy()
        split_info = {
            "strategy": "year_boundary",
            "train_end_year": TRAIN_END_YEAR,
            "test_start_year": TEST_START_YEAR,
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "train_year_max": int(train_df["year"].max()) if len(train_df) else None,
            "test_year_min": int(test_df["year"].min()) if len(test_df) else None,
        }

    x_train = train_df[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y_train_delta = train_df["target_delta"].to_numpy(dtype=float)
    y_train_abs = train_df["target_abs"].to_numpy(dtype=float)

    x_test = test_df[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y_test_delta = test_df["target_delta"].to_numpy(dtype=float)
    y_test_abs = test_df["target_abs"].to_numpy(dtype=float)
    y_test_base = pd.to_numeric(test_df[TARGET_BASE_FEATURE], errors="coerce").to_numpy(dtype=float)

    if len(x_train) == 0 or len(x_test) == 0:
        raise ValueError("Train/test split produced empty partitions.")

    return (
        x_train,
        y_train_delta,
        y_train_abs,
        x_test,
        y_test_delta,
        y_test_abs,
        y_test_base,
        split_info,
    )


def train_random_forest(
    x_train: pd.DataFrame,
    y_train_delta: np.ndarray,
    x_test: pd.DataFrame,
    y_test_abs: np.ndarray,
    y_test_base: np.ndarray,
) -> tuple[Any, dict[str, Any], dict[str, float]]:
    base_model = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1)
    param_grid = {
        "n_estimators": [300, 500, 700, 900],
        "max_depth": [None, 10, 14, 18],
        "min_samples_split": [2, 4, 8, 12],
        "min_samples_leaf": [1, 2, 3, 4],
        "max_features": ["sqrt", 0.7, 1.0],
    }

    n_splits = min(5, max(2, len(x_train) // 500))
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_grid,
        n_iter=24,
        scoring="neg_root_mean_squared_error",
        cv=TimeSeriesSplit(n_splits=n_splits),
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=1,
    )
    search.fit(x_train, y_train_delta)

    best_model = search.best_estimator_
    y_pred_delta = np.asarray(best_model.predict(x_test)).reshape(-1)
    y_pred_abs = y_pred_delta + np.asarray(y_test_base, dtype=float)
    metric_payload = regression_metrics(y_test_abs, y_pred_abs)
    best_params = dict(search.best_params_)
    best_params["cv_score_rmse"] = float(-search.best_score_)

    return best_model, best_params, metric_payload


def train_lstm(
    x_train: pd.DataFrame,
    y_train_delta: np.ndarray,
    x_test: pd.DataFrame,
    y_test_abs: np.ndarray,
    y_test_base: np.ndarray,
) -> tuple[Any, Any, dict[str, float] | None]:
    try:
        from tensorflow.keras.callbacks import EarlyStopping
        from tensorflow.keras.layers import Dense, Dropout, LSTM
        from tensorflow.keras.losses import Huber
        from tensorflow.keras.models import Sequential
        import tensorflow as tf
    except Exception as exc:  # pragma: no cover
        print(f"LSTM training skipped (TensorFlow not available): {exc}")
        return None, None, None

    tf.keras.utils.set_random_seed(RANDOM_STATE)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train.to_numpy(dtype=float))
    x_test_scaled = scaler.transform(x_test.to_numpy(dtype=float))

    x_train_lstm = x_train_scaled.reshape((x_train_scaled.shape[0], 1, x_train_scaled.shape[1]))
    x_test_lstm = x_test_scaled.reshape((x_test_scaled.shape[0], 1, x_test_scaled.shape[1]))

    y_train_fit = np.asarray(y_train_delta, dtype=float)

    model = Sequential(
        [
            LSTM(64, input_shape=(1, x_train_lstm.shape[2]), return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss=Huber())

    early_stop = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
    model.fit(
        x_train_lstm,
        y_train_fit,
        validation_split=0.1,
        epochs=40,
        batch_size=64,
        callbacks=[early_stop],
        verbose=1,
    )

    y_pred_delta = np.asarray(model.predict(x_test_lstm, verbose=0)).reshape(-1)
    y_pred_abs = y_pred_delta + np.asarray(y_test_base, dtype=float)
    lstm_metrics = regression_metrics(y_test_abs, y_pred_abs)
    return model, scaler, lstm_metrics


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    models_dir = base_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = resolve_dataset_path(base_dir)
    print(f"Using dataset: {dataset_path}")

    dataset = load_dataset(dataset_path)
    modeling, feature_columns, target_col = build_modeling_frame(dataset, FORECAST_HORIZON_DAYS)
    (
        x_train,
        y_train_delta,
        y_train_abs,
        x_test,
        y_test_delta,
        y_test_abs,
        y_test_base,
        split_info,
    ) = split_train_test(modeling, feature_columns)

    print(f"Train rows: {len(x_train)} | Test rows: {len(x_test)}")
    print("Training Random Forest...")
    rf_model, rf_params, rf_metrics = train_random_forest(
        x_train,
        y_train_delta,
        x_test,
        y_test_abs,
        y_test_base,
    )
    joblib.dump(rf_model, models_dir / "rf_model.pkl")
    print(f"RF metrics (absolute discharge): {json.dumps(rf_metrics)}")

    print("Training LSTM...")
    lstm_model, scaler, lstm_metrics = train_lstm(
        x_train,
        y_train_delta,
        x_test,
        y_test_abs,
        y_test_base,
    )
    if lstm_model is not None and scaler is not None:
        lstm_model.save(models_dir / "lstm_model.h5")
        joblib.dump(scaler, models_dir / "scaler.pkl")
        print(f"LSTM metrics (absolute discharge): {json.dumps(lstm_metrics or {})}")
    else:
        print("LSTM model not saved.")

    joblib.dump(feature_columns, models_dir / "features.pkl")

    metadata = {
        "updated_at": pd.Timestamp.utcnow().isoformat(),
        "dataset_path": str(dataset_path),
        "target_column": target_col,
        "forecast_horizon_days": FORECAST_HORIZON_DAYS,
        "target_mode": TARGET_MODE,
        "target_base_feature": TARGET_BASE_FEATURE,
        "split": split_info,
        "feature_columns": feature_columns,
        "lstm_target_transform": LSTM_TARGET_TRANSFORM if lstm_model is not None else "none",
        "target_delta_stats": {
            "train_mean": float(np.mean(y_train_delta)),
            "train_std": float(np.std(y_train_delta)),
            "train_min": float(np.min(y_train_delta)),
            "train_max": float(np.max(y_train_delta)),
        },
        "metrics": {
            "random_forest": rf_metrics,
            "lstm": lstm_metrics,
        },
        "random_forest": {
            "best_params": rf_params,
        },
    }
    (models_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Saved artifacts:")
    print(f"  - {models_dir / 'rf_model.pkl'}")
    print(f"  - {models_dir / 'lstm_model.h5'}" if lstm_model is not None else "  - LSTM unavailable")
    print(f"  - {models_dir / 'scaler.pkl'}" if scaler is not None else "  - scaler unavailable")
    print(f"  - {models_dir / 'features.pkl'}")
    print(f"  - {models_dir / 'model_metadata.json'}")


if __name__ == "__main__":
    main()
