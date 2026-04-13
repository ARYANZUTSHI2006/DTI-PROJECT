from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from utils.preprocessing import detect_discharge_column, load_dataset

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover
    XGBRegressor = None  # type: ignore[assignment]


BASELINE_RMSE = 101.87
BASELINE_TRAIN_RMSE = 91.64
BASELINE_TEST_RMSE = 126.99
RANDOM_STATE = 42
TARGET_HORIZON_DAYS = 3
TRAIN_END_YEAR = 2000
TEST_START_YEAR = 2006


@dataclass
class ModelScore:
    name: str
    train_metrics: dict[str, float]
    test_metrics: dict[str, float]
    overall_metrics: dict[str, float]
    best_params: dict[str, Any]


def nse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    if denom == 0:
        return float("nan")
    return float(1.0 - (np.sum((y_true - y_pred) ** 2) / denom))


def pbias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.sum(y_true)
    if np.isclose(denom, 0.0):
        return float("nan")
    return float(100.0 * np.sum(y_pred - y_true) / denom)


def peak_error_pct(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0:
        return float("nan")
    peak_idx = int(np.argmax(y_true))
    peak_true = float(y_true[peak_idx])
    if np.isclose(peak_true, 0.0):
        return float("nan")
    return float(abs((float(y_pred[peak_idx]) - peak_true) / peak_true) * 100.0)


def metrics_bundle(y_true: np.ndarray, y_pred: np.ndarray, high_flow_quantile: float = 0.90) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mean_y = float(np.mean(y_true))
    nrmse = float((rmse / mean_y) * 100.0) if not np.isclose(mean_y, 0.0) else float("nan")

    threshold = float(np.quantile(y_true, high_flow_quantile))
    high_mask = y_true >= threshold
    if np.any(high_mask):
        high_rmse = float(np.sqrt(mean_squared_error(y_true[high_mask], y_pred[high_mask])))
    else:
        high_rmse = float("nan")

    return {
        "rmse": rmse,
        "nrmse_mean_pct": nrmse,
        "r2": float(r2_score(y_true, y_pred)),
        "nse": nse(y_true, y_pred),
        "pbias_pct": pbias(y_true, y_pred),
        "peak_error_pct": peak_error_pct(y_true, y_pred),
        "high_flow_rmse": high_rmse,
    }


def build_model_score(
    name: str,
    y_train_true: np.ndarray,
    y_train_pred: np.ndarray,
    y_test_true: np.ndarray,
    y_test_pred: np.ndarray,
    best_params: dict[str, Any],
) -> ModelScore:
    train_metrics = metrics_bundle(y_train_true, y_train_pred)
    test_metrics = metrics_bundle(y_test_true, y_test_pred)
    overall_true = np.concatenate([np.asarray(y_train_true, dtype=float), np.asarray(y_test_true, dtype=float)])
    overall_pred = np.concatenate([np.asarray(y_train_pred, dtype=float), np.asarray(y_test_pred, dtype=float)])
    overall_metrics = metrics_bundle(overall_true, overall_pred)
    return ModelScore(
        name=name,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        overall_metrics=overall_metrics,
        best_params=best_params,
    )


def resolve_dataset_path(base_dir: Path) -> Path:
    preferred = [base_dir / "datasets" / "Kasol.xlsx", base_dir / "datasets" / "Kasol.csv"]
    for candidate in preferred:
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    raise FileNotFoundError("No dataset found. Expected datasets/Kasol.xlsx or datasets/Kasol.csv")


def resolve_date_column(df: pd.DataFrame) -> str:
    for col in ["DATE", "Date", "date", "Datetime", "datetime"]:
        if col in df.columns:
            return col
    raise ValueError("Date column not found in dataset.")


def resolve_rainfall_series(df: pd.DataFrame) -> pd.Series:
    if "Mean_PCP" in df.columns:
        return pd.to_numeric(df["Mean_PCP"], errors="coerce")
    if "PCP" in df.columns:
        return pd.to_numeric(df["PCP"], errors="coerce")

    rain_parts = [col for col in ["P1", "P2", "P3"] if col in df.columns]
    if rain_parts:
        return df[rain_parts].apply(pd.to_numeric, errors="coerce").mean(axis=1)

    return pd.Series(0.0, index=df.index, dtype=float)


def add_hydrology_features(df: pd.DataFrame, horizon_days: int) -> tuple[pd.DataFrame, str, str, list[str]]:
    work = df.copy()
    date_col = resolve_date_column(work)
    target_col = detect_discharge_column(work)
    if target_col is None:
        raise ValueError("Could not detect discharge target column.")

    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work = work.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

    runoff = pd.to_numeric(work[target_col], errors="coerce")
    rain = resolve_rainfall_series(work)

    work["rain"] = rain
    work["q_lag_1"] = runoff.shift(1)
    work["q_lag_2"] = runoff.shift(2)

    work["rain_lag_1"] = rain.shift(1)
    work["rain_lag_2"] = rain.shift(2)
    work["rain_lag_3"] = rain.shift(3)
    work["rain_lag_5"] = rain.shift(5)

    rain_hist = rain.shift(1)
    work["rain_sum_3"] = rain_hist.rolling(window=3, min_periods=1).sum()
    work["rain_sum_5"] = rain_hist.rolling(window=5, min_periods=1).sum()
    work["rain_sum_7"] = rain_hist.rolling(window=7, min_periods=1).sum()

    # Exponentially weighted antecedent rainfall proxy for catchment wetness memory.
    work["antecedent_moisture_index"] = rain_hist.ewm(alpha=0.15, adjust=False).mean()

    work["month"] = work[date_col].dt.month
    work["monsoon_flag"] = work["month"].isin([6, 7, 8, 9]).astype(int)

    day_of_year = work[date_col].dt.dayofyear
    angle = (2.0 * np.pi * day_of_year) / 365.25
    work["season_sin"] = np.sin(angle)
    work["season_cos"] = np.cos(angle)

    work["rain_intensity_3d"] = work["rain_sum_3"] / 3.0
    work["runoff_recession"] = work["q_lag_1"] - work["q_lag_2"]
    work["runoff_rain_ratio"] = work["q_lag_1"] / (work["rain_sum_7"] + 1.0)

    for col in ["Mean_Tmax", "TMAX", "Mean_Tmin", "TMIN", "rh", "solar", "wind", "wind "]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    if "wind" not in work.columns and "wind " in work.columns:
        work["wind"] = work["wind "]

    if "Mean_Tmax" not in work.columns and "TMAX" in work.columns:
        work["Mean_Tmax"] = work["TMAX"]
    if "Mean_Tmin" not in work.columns and "TMIN" in work.columns:
        work["Mean_Tmin"] = work["TMIN"]

    work["target"] = runoff.shift(-horizon_days)

    feature_cols = [
        "rain",
        "Mean_Tmax",
        "Mean_Tmin",
        "rh",
        "solar",
        "wind",
        "q_lag_1",
        "q_lag_2",
        "rain_lag_1",
        "rain_lag_2",
        "rain_lag_3",
        "rain_lag_5",
        "rain_sum_3",
        "rain_sum_5",
        "rain_sum_7",
        "antecedent_moisture_index",
        "month",
        "monsoon_flag",
        "season_sin",
        "season_cos",
        "rain_intensity_3d",
        "runoff_recession",
        "runoff_rain_ratio",
    ]

    available = [c for c in feature_cols if c in work.columns]
    modeling = work[[date_col, target_col, "target", *available]].copy()
    modeling = modeling.dropna().reset_index(drop=True)

    if modeling.empty:
        raise ValueError("No rows remain after feature engineering and target shift.")

    return modeling, date_col, target_col, available


def chronological_split(modeling: pd.DataFrame, date_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    modeling = modeling.copy()
    modeling["year"] = modeling[date_col].dt.year

    train_mask = modeling["year"] <= TRAIN_END_YEAR
    test_mask = modeling["year"] >= TEST_START_YEAR

    if int(train_mask.sum()) >= 500 and int(test_mask.sum()) >= 200:
        train_df = modeling.loc[train_mask].copy()
        test_df = modeling.loc[test_mask].copy()
    else:
        split_idx = int(len(modeling) * 0.8)
        split_idx = max(1, min(split_idx, len(modeling) - 1))
        train_df = modeling.iloc[:split_idx].copy()
        test_df = modeling.iloc[split_idx:].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Chronological split produced empty partitions.")

    return train_df, test_df


def build_sample_weights(y_train: np.ndarray) -> np.ndarray:
    y_train = np.asarray(y_train, dtype=float)
    weights = np.ones_like(y_train, dtype=float)
    q80 = float(np.quantile(y_train, 0.80))
    q90 = float(np.quantile(y_train, 0.90))
    q95 = float(np.quantile(y_train, 0.95))

    weights[y_train >= q80] = 1.3
    weights[y_train >= q90] = 1.6
    weights[y_train >= q95] = 2.0
    return weights


def select_features(X_train: pd.DataFrame, y_train_log: np.ndarray) -> list[str]:
    selector = RandomForestRegressor(
        n_estimators=500,
        max_depth=14,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    selector.fit(X_train, y_train_log)
    importances = pd.Series(selector.feature_importances_, index=X_train.columns).sort_values(ascending=False)

    selected = importances[importances >= 0.001].index.tolist()

    must_keep = [
        "rain",
        "Mean_Tmax",
        "Mean_Tmin",
        "rh",
        "solar",
        "wind",
        "q_lag_1",
        "q_lag_2",
        "rain_lag_1",
        "rain_lag_2",
        "rain_lag_3",
        "rain_lag_5",
        "rain_sum_3",
        "rain_sum_5",
        "rain_sum_7",
        "antecedent_moisture_index",
        "month",
        "monsoon_flag",
    ]
    for col in must_keep:
        if col in X_train.columns and col not in selected:
            selected.append(col)

    if len(selected) < 16:
        selected = importances.head(min(20, len(importances))).index.tolist()

    return selected


def optimize_blend_weight(
    rf_model: RandomForestRegressor,
    xgb_model: Any,
    X_train: pd.DataFrame,
    y_train_log: np.ndarray,
    sample_weights: np.ndarray,
) -> float:
    split_idx = int(len(X_train) * 0.85)
    split_idx = max(200, min(split_idx, len(X_train) - 100))

    X_sub = X_train.iloc[:split_idx]
    y_sub = y_train_log[:split_idx]
    w_sub = sample_weights[:split_idx]

    X_val = X_train.iloc[split_idx:]
    y_val = np.expm1(y_train_log[split_idx:])

    rf_shadow = RandomForestRegressor(**rf_model.get_params())
    rf_shadow.fit(X_sub, y_sub, sample_weight=w_sub)
    rf_val = np.expm1(np.clip(np.asarray(rf_shadow.predict(X_val)).reshape(-1), -20.0, 20.0))

    xgb_shadow = XGBRegressor(**xgb_model.get_params())
    xgb_shadow.fit(X_sub, y_sub, sample_weight=w_sub)
    xgb_val = np.expm1(np.clip(np.asarray(xgb_shadow.predict(X_val)).reshape(-1), -20.0, 20.0))

    best_weight = 0.5
    best_score = float("inf")

    for w in np.linspace(0.0, 1.0, 41):
        blend_val = w * rf_val + (1.0 - w) * xgb_val
        rmse_val = float(np.sqrt(mean_squared_error(y_val, blend_val)))
        high_mask = y_val >= np.quantile(y_val, 0.90)
        if np.any(high_mask):
            high_rmse = float(np.sqrt(mean_squared_error(y_val[high_mask], blend_val[high_mask])))
        else:
            high_rmse = rmse_val

        score = rmse_val + 0.35 * high_rmse
        if score < best_score:
            best_score = score
            best_weight = float(w)

    return best_weight


def train_lstm_model(
    X_train: pd.DataFrame,
    y_train_log: np.ndarray,
    X_test: pd.DataFrame,
    y_train_abs: np.ndarray,
    y_test_abs: np.ndarray,
    sample_weights: np.ndarray,
) -> ModelScore | None:
    try:
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping
        from tensorflow.keras.layers import Dense, Dropout, LSTM
        from tensorflow.keras.models import Sequential
    except Exception:
        return None

    tf.keras.utils.set_random_seed(RANDOM_STATE)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lookback = 7
    if len(X_train_scaled) <= lookback + 5:
        return None

    X_train_seq: list[np.ndarray] = []
    y_train_seq: list[float] = []
    w_train_seq: list[float] = []

    for i in range(lookback, len(X_train_scaled)):
        X_train_seq.append(X_train_scaled[i - lookback : i, :])
        y_train_seq.append(float(y_train_log[i]))
        w_train_seq.append(float(sample_weights[i]))

    if not X_train_seq:
        return None

    bridge = np.vstack([X_train_scaled[-lookback:, :], X_test_scaled])
    X_test_seq: list[np.ndarray] = []
    for i in range(lookback, len(bridge)):
        X_test_seq.append(bridge[i - lookback : i, :])

    X_train_arr = np.asarray(X_train_seq, dtype=float)
    y_train_arr = np.asarray(y_train_seq, dtype=float)
    w_train_arr = np.asarray(w_train_seq, dtype=float)
    X_test_arr = np.asarray(X_test_seq, dtype=float)

    model = Sequential(
        [
            LSTM(64, return_sequences=True, input_shape=(lookback, X_train_arr.shape[2])),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="huber")

    stop = EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True)
    model.fit(
        X_train_arr,
        y_train_arr,
        sample_weight=w_train_arr,
        validation_split=0.1,
        epochs=35,
        batch_size=64,
        callbacks=[stop],
        verbose=0,
    )

    train_pred_log = np.asarray(model.predict(X_train_arr, verbose=0)).reshape(-1)
    test_pred_log = np.asarray(model.predict(X_test_arr, verbose=0)).reshape(-1)

    train_pred = np.expm1(np.clip(train_pred_log, -20.0, 20.0))
    test_pred = np.expm1(np.clip(test_pred_log, -20.0, 20.0))

    y_train_cmp = y_train_abs[lookback:]
    train_metrics = metrics_bundle(y_train_cmp, train_pred)
    test_metrics = metrics_bundle(y_test_abs, test_pred)

    return ModelScore(
        name="lstm_log1p_weighted",
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        overall_metrics=metrics_bundle(
            np.concatenate([y_train_cmp, y_test_abs]),
            np.concatenate([train_pred, test_pred]),
        ),
        best_params={"lookback": lookback, "epochs": 35, "batch_size": 64},
    )


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    models_dir = base_dir / "models"
    reports_dir = base_dir / "reports"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = resolve_dataset_path(base_dir)
    raw = load_dataset(dataset_path)

    modeling, date_col, target_col, feature_cols = add_hydrology_features(raw, TARGET_HORIZON_DAYS)
    train_df, test_df = chronological_split(modeling, date_col)

    X_train_all = train_df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X_test_all = test_df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y_train = pd.to_numeric(train_df["target"], errors="coerce").to_numpy(dtype=float)
    y_test = pd.to_numeric(test_df["target"], errors="coerce").to_numpy(dtype=float)

    y_train_log = np.log1p(np.clip(y_train, a_min=0.0, a_max=None))
    sample_weights = build_sample_weights(y_train)

    selected_features = select_features(X_train_all, y_train_log)
    X_train = X_train_all[selected_features]
    X_test = X_test_all[selected_features]

    # Reference model for peak-flow and RMSE improvement attribution.
    baseline_feature_set = [
        col
        for col in [
            "rain",
            "Mean_Tmax",
            "Mean_Tmin",
            "rh",
            "solar",
            "wind",
            "q_lag_1",
            "q_lag_2",
            "season_sin",
            "season_cos",
        ]
        if col in X_train_all.columns
    ]
    baseline_rf = RandomForestRegressor(
        n_estimators=500,
        max_depth=10,
        min_samples_split=4,
        min_samples_leaf=8,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    baseline_rf.fit(
        X_train_all[baseline_feature_set],
        y_train_log,
    )
    baseline_train_pred = np.expm1(
        np.clip(np.asarray(baseline_rf.predict(X_train_all[baseline_feature_set])).reshape(-1), -20.0, 20.0)
    )
    baseline_test_pred = np.expm1(
        np.clip(np.asarray(baseline_rf.predict(X_test_all[baseline_feature_set])).reshape(-1), -20.0, 20.0)
    )
    reference_score = build_model_score(
        name="rf_reference_baseline_log1p",
        y_train_true=y_train,
        y_train_pred=baseline_train_pred,
        y_test_true=y_test,
        y_test_pred=baseline_test_pred,
        best_params={
            "n_estimators": 500,
            "max_depth": 10,
            "min_samples_split": 4,
            "min_samples_leaf": 8,
            "max_features": "sqrt",
            "feature_count": len(baseline_feature_set),
        },
    )

    n_splits = max(3, min(4, len(X_train) // 1400))
    cv = TimeSeriesSplit(n_splits=n_splits)

    rf_search = RandomizedSearchCV(
        estimator=RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        param_distributions={
            "n_estimators": [400, 600, 800],
            "max_depth": [8, 10, 12],
            "min_samples_split": [4, 8, 12],
            "min_samples_leaf": [4, 8, 12, 16],
            "max_features": ["sqrt", 0.7],
        },
        n_iter=12,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1,
    )
    rf_search.fit(X_train, y_train_log, sample_weight=sample_weights)
    rf_model = rf_search.best_estimator_

    rf_train_pred = np.expm1(np.clip(np.asarray(rf_model.predict(X_train)).reshape(-1), -20.0, 20.0))
    rf_test_pred = np.expm1(np.clip(np.asarray(rf_model.predict(X_test)).reshape(-1), -20.0, 20.0))

    rf_score = build_model_score(
        name="random_forest_log1p_weighted",
        y_train_true=y_train,
        y_train_pred=rf_train_pred,
        y_test_true=y_test,
        y_test_pred=rf_test_pred,
        best_params={**rf_search.best_params_, "cv_rmse": float(-rf_search.best_score_)},
    )

    rf_unweighted = RandomForestRegressor(
        n_estimators=500,
        max_depth=10,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf_unweighted.fit(X_train, y_train_log)
    rf_unw_train_pred = np.expm1(
        np.clip(np.asarray(rf_unweighted.predict(X_train)).reshape(-1), -20.0, 20.0)
    )
    rf_unw_test_pred = np.expm1(
        np.clip(np.asarray(rf_unweighted.predict(X_test)).reshape(-1), -20.0, 20.0)
    )
    rf_unweighted_score = build_model_score(
        name="random_forest_log1p_unweighted",
        y_train_true=y_train,
        y_train_pred=rf_unw_train_pred,
        y_test_true=y_test,
        y_test_pred=rf_unw_test_pred,
        best_params={
            "n_estimators": 500,
            "max_depth": 10,
            "min_samples_split": 4,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
        },
    )

    scores: list[ModelScore] = [rf_score, rf_unweighted_score, reference_score]

    if XGBRegressor is not None:
        xgb_search = RandomizedSearchCV(
            estimator=XGBRegressor(
                objective="reg:squarederror",
                eval_metric="rmse",
                tree_method="hist",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            param_distributions={
                "n_estimators": [300, 500, 700, 900],
                "learning_rate": [0.01, 0.03, 0.05, 0.08],
                "max_depth": [3, 5, 7, 9],
                "min_child_weight": [1, 3, 5],
                "subsample": [0.7, 0.85, 1.0],
                "colsample_bytree": [0.7, 0.85, 1.0],
                "reg_alpha": [0.0, 0.1, 0.5],
                "reg_lambda": [1.0, 2.0, 5.0],
                "gamma": [0.0, 0.1, 0.3],
            },
            n_iter=12,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=1,
        )
        xgb_search.fit(X_train, y_train_log, sample_weight=sample_weights)
        xgb_model = xgb_search.best_estimator_

        xgb_train_pred = np.expm1(np.clip(np.asarray(xgb_model.predict(X_train)).reshape(-1), -20.0, 20.0))
        xgb_test_pred = np.expm1(np.clip(np.asarray(xgb_model.predict(X_test)).reshape(-1), -20.0, 20.0))

        xgb_score = build_model_score(
            name="xgboost_log1p_weighted",
            y_train_true=y_train,
            y_train_pred=xgb_train_pred,
            y_test_true=y_test,
            y_test_pred=xgb_test_pred,
            best_params={**xgb_search.best_params_, "cv_rmse": float(-xgb_search.best_score_)},
        )
        scores.append(xgb_score)

        blend_weight = optimize_blend_weight(
            rf_model=rf_model,
            xgb_model=xgb_model,
            X_train=X_train,
            y_train_log=y_train_log,
            sample_weights=sample_weights,
        )

        hybrid_train_pred = blend_weight * rf_train_pred + (1.0 - blend_weight) * xgb_train_pred
        hybrid_test_pred = blend_weight * rf_test_pred + (1.0 - blend_weight) * xgb_test_pred

        hybrid_score = build_model_score(
            name="hybrid_rf_xgb_log1p_weighted",
            y_train_true=y_train,
            y_train_pred=hybrid_train_pred,
            y_test_true=y_test,
            y_test_pred=hybrid_test_pred,
            best_params={"rf_weight": blend_weight, "xgb_weight": 1.0 - blend_weight},
        )
        scores.append(hybrid_score)

        rf_artifact = models_dir / "improved_random_forest_log1p.joblib"
        xgb_artifact = models_dir / "improved_xgboost_log1p.json"
        joblib.dump(rf_model, rf_artifact)
        xgb_model.save_model(str(xgb_artifact))
    else:
        xgb_score = None
        rf_artifact = models_dir / "improved_random_forest_log1p.joblib"
        joblib.dump(rf_model, rf_artifact)

    lstm_score = train_lstm_model(
        X_train=X_train,
        y_train_log=y_train_log,
        X_test=X_test,
        y_train_abs=y_train,
        y_test_abs=y_test,
        sample_weights=sample_weights,
    )
    if lstm_score is not None:
        scores.append(lstm_score)

    candidate_scores = [s for s in scores if s.name != "rf_reference_baseline_log1p"]
    scores_sorted = sorted(candidate_scores, key=lambda item: item.test_metrics["rmse"])
    best = scores_sorted[0]

    best_overall_rmse = float(best.overall_metrics["rmse"])
    rmse_improvement_pct = float((BASELINE_RMSE - best_overall_rmse) / BASELINE_RMSE * 100.0)

    baseline_gap = BASELINE_TEST_RMSE - BASELINE_TRAIN_RMSE
    new_gap = float(best.test_metrics["rmse"] - best.train_metrics["rmse"])
    peak_error_improvement_pct = float(
        (reference_score.test_metrics["peak_error_pct"] - best.test_metrics["peak_error_pct"])
        / max(reference_score.test_metrics["peak_error_pct"], 1e-6)
        * 100.0
    )
    high_flow_rmse_improvement_pct = float(
        (reference_score.test_metrics["high_flow_rmse"] - best.test_metrics["high_flow_rmse"])
        / max(reference_score.test_metrics["high_flow_rmse"], 1e-6)
        * 100.0
    )

    summary = {
        "dataset_path": str(dataset_path.as_posix()),
        "target_column": target_col,
        "forecast_horizon_days": TARGET_HORIZON_DAYS,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "selected_feature_count": int(len(selected_features)),
        "selected_features": selected_features,
        "baseline": {
            "rmse": BASELINE_RMSE,
            "train_rmse": BASELINE_TRAIN_RMSE,
            "test_rmse": BASELINE_TEST_RMSE,
            "nrmse_pct": 24.87,
            "nse": 0.935,
            "r2": 0.935,
        },
        "best_model": {
            "name": best.name,
            "train_metrics": best.train_metrics,
            "test_metrics": best.test_metrics,
            "overall_metrics": best.overall_metrics,
            "best_params": best.best_params,
        },
        "all_models": [
            {
                "name": score.name,
                "train_metrics": score.train_metrics,
                "test_metrics": score.test_metrics,
                "overall_metrics": score.overall_metrics,
                "best_params": score.best_params,
            }
            for score in sorted(scores, key=lambda item: item.test_metrics["rmse"])
        ],
        "improvements": {
            "rmse_improvement_pct_vs_baseline": rmse_improvement_pct,
            "peak_error_improvement_pct_vs_reference": peak_error_improvement_pct,
            "high_flow_rmse_improvement_pct_vs_reference": high_flow_rmse_improvement_pct,
            "overfitting_gap_baseline": baseline_gap,
            "overfitting_gap_new": new_gap,
            "overfitting_reduced": bool(abs(new_gap) < abs(baseline_gap)),
        },
    }

    summary_path = reports_dir / "improved_model_results.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=== Improved Runoff Modeling Summary ===")
    print(f"Dataset: {dataset_path.as_posix()}")
    print(f"Train rows: {len(train_df)} | Test rows: {len(test_df)}")
    print(f"Selected features ({len(selected_features)}): {selected_features}")
    print(f"Best model: {best.name}")
    print(f"Overall RMSE: {best.overall_metrics['rmse']:.3f}")
    print(f"Overall NRMSE (% mean): {best.overall_metrics['nrmse_mean_pct']:.3f}")
    print(f"Overall NSE: {best.overall_metrics['nse']:.4f}")
    print(f"Test RMSE: {best.test_metrics['rmse']:.3f}")
    print(f"Test NRMSE (% mean): {best.test_metrics['nrmse_mean_pct']:.3f}")
    print(f"Test NSE: {best.test_metrics['nse']:.4f}")
    print(f"Test R2: {best.test_metrics['r2']:.4f}")
    print(f"Test High-flow RMSE: {best.test_metrics['high_flow_rmse']:.3f}")
    print(f"Test Peak Error (%): {best.test_metrics['peak_error_pct']:.3f}")
    print(f"Test PBIAS (%): {best.test_metrics['pbias_pct']:.3f}")
    print(f"RMSE improvement vs baseline (%): {rmse_improvement_pct:.2f}")
    print(f"Peak Error improvement vs reference RF (%): {peak_error_improvement_pct:.2f}")
    print(f"High-flow RMSE improvement vs reference RF (%): {high_flow_rmse_improvement_pct:.2f}")
    print(f"Overfitting gap baseline/new: {baseline_gap:.3f} -> {new_gap:.3f}")
    print(f"Summary file: {summary_path.as_posix()}")


if __name__ == "__main__":
    main()
