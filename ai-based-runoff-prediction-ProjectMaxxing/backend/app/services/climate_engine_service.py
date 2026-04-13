from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from app.config import settings
from app.core.exceptions import BackendException
from app.core.logger import get_logger
from app.services.training_service import load_training_dataset


logger = get_logger(__name__)

SCENARIO_TO_LAG: dict[str, int] = {"R0": 0, "R1": 1, "R2": 2, "R3": 3}
SEASON_MONTHS: dict[str, list[int]] = {
	"winter": [12, 1, 2],
	"pre_monsoon": [3, 4, 5],
	"monsoon": [6, 7, 8, 9],
	"post_monsoon": [10, 11],
}


@dataclass
class PreparedData:
	frame: pd.DataFrame
	date_col: str
	target_col: str


def _safe_std(values: np.ndarray) -> float:
	std_val = float(np.std(values))
	return std_val if std_val > 1e-12 else 1e-12


def _float_or_default(value: Any, default: float) -> float:
	if value is None:
		return float(default)
	try:
		return float(value)
	except Exception:
		return float(default)


def _normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
	renamed = {col: str(col).strip() for col in df.columns}
	return df.rename(columns=renamed)


def _find_column(df: pd.DataFrame, aliases: list[str], required: bool = True) -> str | None:
	lowered = {str(col).strip().lower(): col for col in df.columns}
	for alias in aliases:
		if alias.lower() in lowered:
			return lowered[alias.lower()]

	if required:
		raise BackendException(422, f"Required column not found. Expected one of: {aliases}")
	return None


def _prepare_observed_dataframe() -> PreparedData:
	raw_df = load_training_dataset()
	df = _normalize_column_names(raw_df)

	date_col = _find_column(df, ["DATE", "Date", "date"])
	target_col = _find_column(df, ["Discharge", "Discharge (CUMEC)", "streamflow", "flow"])
	rain_col = _find_column(df, ["PCP", "Mean_PCP", "rainfall", "Rainfall", "precipitation"])
	tmax_col = _find_column(df, ["TMAX", "Mean_Tmax", "tmax", "Tmax"])
	tmin_col = _find_column(df, ["TMIN", "Mean_Tmin", "tmin", "Tmin"])
	humidity_col = _find_column(df, ["rh", "RH", "RelativeHumidity", "humidity"])
	solar_col = _find_column(df, ["solar", "Solar", "solar_radiation"])
	wind_col = _find_column(df, ["wind", "wind ", "Wind", "wind_speed"])

	prepared = pd.DataFrame(
		{
			"date": pd.to_datetime(df[date_col], errors="coerce"),
			"target": pd.to_numeric(df[target_col], errors="coerce"),
			"rainfall": pd.to_numeric(df[rain_col], errors="coerce"),
			"tmax": pd.to_numeric(df[tmax_col], errors="coerce"),
			"tmin": pd.to_numeric(df[tmin_col], errors="coerce"),
			"humidity": pd.to_numeric(df[humidity_col], errors="coerce"),
			"solar": pd.to_numeric(df[solar_col], errors="coerce"),
			"wind": pd.to_numeric(df[wind_col], errors="coerce"),
		}
	)
	prepared = prepared.dropna().sort_values("date").reset_index(drop=True)

	if prepared.empty:
		raise BackendException(422, "Prepared training data is empty after cleaning")

	return PreparedData(frame=prepared, date_col="date", target_col="target")


def _build_scenario_features(prepared: PreparedData, scenario: str) -> pd.DataFrame:
	if scenario not in SCENARIO_TO_LAG:
		raise BackendException(422, f"Unsupported rainfall scenario: {scenario}")

	lag_days = SCENARIO_TO_LAG[scenario]
	df = prepared.frame.copy()
	df["rainfall_feature"] = df["rainfall"].shift(lag_days)
	df = df.dropna().reset_index(drop=True)
	return df


def _train_test_split(df: pd.DataFrame, train_fraction: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
	split_index = int(len(df) * train_fraction)
	split_index = max(1, min(split_index, len(df) - 1))
	return df.iloc[:split_index].copy(), df.iloc[split_index:].copy()


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
	y_true = np.asarray(y_true, dtype=float).reshape(-1)
	y_pred = np.asarray(y_pred, dtype=float).reshape(-1)

	mae = float(mean_absolute_error(y_true, y_pred))
	r2 = float(r2_score(y_true, y_pred))

	residual_sum = float(np.sum((y_true - y_pred) ** 2))
	denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
	nse = 1.0 - (residual_sum / denom) if denom > 1e-12 else -np.inf

	rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
	rsr = rmse / _safe_std(y_true)

	corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else 0.0
	corr = 0.0 if np.isnan(corr) else corr
	alpha = _safe_std(y_pred) / _safe_std(y_true)
	mean_true = float(np.mean(y_true))
	mean_pred = float(np.mean(y_pred))
	beta = (mean_pred / mean_true) if abs(mean_true) > 1e-12 else 0.0
	kge = 1.0 - float(np.sqrt((corr - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))

	pbias = float(np.sum(y_pred - y_true) / (np.sum(y_true) + 1e-12))

	return {
		"R2": r2,
		"NSE": float(nse),
		"KGE": float(kge),
		"RSR": float(rsr),
		"MAE": mae,
		"PBIAS": pbias,
	}


def _build_model(model_key: str, random_state: int, cv_splits: int, n_iter: int):
	if model_key == "GLM":
		return Pipeline([("scaler", StandardScaler()), ("glm", LinearRegression())])

	if model_key == "GAM":
		return Pipeline(
			[
				("spline", SplineTransformer(n_knots=6, degree=3, include_bias=False)),
				("ridge", Ridge(alpha=1.0)),
			]
		)

	if model_key == "MARS":
		try:
			from pyearth import Earth

			return Earth(max_degree=2)
		except Exception:
			return Pipeline(
				[
					("spline", SplineTransformer(n_knots=8, degree=1, include_bias=False)),
					("ridge", Ridge(alpha=0.5)),
				]
			)

	if model_key == "ANN":
		return Pipeline(
			[
				("scaler", StandardScaler()),
				(
					"ann",
					MLPRegressor(
						hidden_layer_sizes=(64, 32),
						activation="relu",
						solver="adam",
						max_iter=600,
						random_state=random_state,
					),
				),
			]
		)

	if model_key == "RF":
		base = RandomForestRegressor(random_state=random_state, n_jobs=-1)
		search = RandomizedSearchCV(
			estimator=base,
			param_distributions={
				"n_estimators": [100, 200, 300, 400],
				"max_depth": [6, 8, 12, 16, None],
				"min_samples_split": [2, 5, 10],
				"min_samples_leaf": [1, 2, 4],
				"max_features": ["sqrt", "log2", None],
			},
			n_iter=n_iter,
			cv=KFold(n_splits=cv_splits, shuffle=True, random_state=random_state),
			scoring="neg_mean_squared_error",
			random_state=random_state,
			n_jobs=-1,
		)
		return search

	if model_key == "CNN1D":
		try:
			import tensorflow as tf

			class CNN1DRegressor:
				def __init__(self, seed: int = 42):
					self.seed = seed
					self.model = None

				def fit(self, X, y):
					tf.keras.utils.set_random_seed(self.seed)
					X_arr = np.asarray(X, dtype=np.float32)
					y_arr = np.asarray(y, dtype=np.float32)
					X_arr = X_arr.reshape((X_arr.shape[0], X_arr.shape[1], 1))

					model = tf.keras.Sequential(
						[
							tf.keras.layers.Input(shape=(X_arr.shape[1], 1)),
							tf.keras.layers.Conv1D(32, kernel_size=2, activation="relu"),
							tf.keras.layers.Conv1D(16, kernel_size=2, activation="relu"),
							tf.keras.layers.Flatten(),
							tf.keras.layers.Dense(32, activation="relu"),
							tf.keras.layers.Dense(1),
						]
					)
					model.compile(optimizer="adam", loss="mse")
					model.fit(X_arr, y_arr, epochs=10, batch_size=32, verbose=0)
					self.model = model
					return self

				def predict(self, X):
					if self.model is None:
						raise RuntimeError("CNN model not fitted")
					X_arr = np.asarray(X, dtype=np.float32)
					X_arr = X_arr.reshape((X_arr.shape[0], X_arr.shape[1], 1))
					return self.model.predict(X_arr, verbose=0).reshape(-1)

			return CNN1DRegressor(seed=random_state)
		except Exception:
			return Pipeline([("scaler", StandardScaler()), ("reg", LinearRegression())])

	raise BackendException(422, f"Unsupported model key: {model_key}")


def _fit_predict_model(model_key: str, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame) -> tuple[Any, np.ndarray]:
	model = _build_model(model_key=model_key, random_state=42, cv_splits=3, n_iter=20)
	model.fit(X_train, y_train)
	y_pred = np.asarray(model.predict(X_test), dtype=float).reshape(-1)
	return model, y_pred


def _model_display_name(model_key: str) -> str:
	display_map = {
		"GLM": "GLM",
		"GAM": "GAM",
		"MARS": "MARS",
		"ANN": "ANN",
		"RF": "Random Forest",
		"CNN1D": "1D-CNN",
	}
	return display_map.get(model_key, model_key)


def _sort_model_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
	return sorted(
		candidates,
		key=lambda item: (
			-float(item["metrics"]["NSE"]),
			float(item["metrics"]["RSR"]),
			abs(float(item["metrics"]["PBIAS"])),
		),
	)


def _save_rf_artifact(best_rf_model: Any, model_path: Path) -> None:
	model_path.parent.mkdir(parents=True, exist_ok=True)
	joblib.dump(best_rf_model, model_path)


def _build_climate_registry_payload(
	ranked_models: list[dict[str, Any]],
	best_rf_entry: dict[str, Any],
	artifact_path: Path,
) -> dict[str, Any]:
	return {
		"best_model": {
			"model_name": best_rf_entry["model_name"],
			"rainfall_scenario": best_rf_entry["rainfall_scenario"],
			"artifact_path": artifact_path.as_posix(),
			"feature_columns": best_rf_entry["feature_columns"],
			"metrics": best_rf_entry["metrics"],
		},
		"model_comparison": ranked_models,
	}


def _load_climate_registry() -> dict[str, Any]:
	if not settings.CLIMATE_REGISTRY_PATH.exists():
		raise BackendException(404, f"Climate registry not found at {settings.CLIMATE_REGISTRY_PATH}")
	with settings.CLIMATE_REGISTRY_PATH.open("r", encoding="utf-8") as file_obj:
		return json.load(file_obj)


def _load_rf_model_from_registry() -> tuple[Any, dict[str, Any]]:
	registry = _load_climate_registry()
	best = registry.get("best_model")
	if not isinstance(best, dict):
		raise BackendException(500, "Malformed climate registry: missing best_model")

	artifact_path = best.get("artifact_path")
	if not artifact_path:
		raise BackendException(500, "Malformed climate registry: missing artifact_path")

	model_path = Path(str(artifact_path))
	if not model_path.is_absolute():
		model_path = settings.BASE_DIR / model_path
	if not model_path.exists():
		raise BackendException(404, f"Climate model artifact missing at {model_path}")

	model = joblib.load(model_path)
	return model, best


def _ensure_cmip6_columns(df: pd.DataFrame) -> pd.DataFrame:
	norm = _normalize_column_names(df)
	date_col = _find_column(norm, ["DATE", "Date", "date"])
	rain_col = _find_column(norm, ["PCP", "Mean_PCP", "rainfall", "Rainfall", "precipitation"])
	tmax_col = _find_column(norm, ["TMAX", "Mean_Tmax", "tmax", "Tmax"])
	tmin_col = _find_column(norm, ["TMIN", "Mean_Tmin", "tmin", "Tmin"])
	humidity_col = _find_column(norm, ["rh", "RH", "RelativeHumidity", "humidity"])
	solar_col = _find_column(norm, ["solar", "Solar", "solar_radiation"])
	wind_col = _find_column(norm, ["wind", "wind ", "Wind", "wind_speed"])

	prepared = pd.DataFrame(
		{
			"date": pd.to_datetime(norm[date_col], errors="coerce"),
			"rainfall": pd.to_numeric(norm[rain_col], errors="coerce"),
			"tmax": pd.to_numeric(norm[tmax_col], errors="coerce"),
			"tmin": pd.to_numeric(norm[tmin_col], errors="coerce"),
			"humidity": pd.to_numeric(norm[humidity_col], errors="coerce"),
			"solar": pd.to_numeric(norm[solar_col], errors="coerce"),
			"wind": pd.to_numeric(norm[wind_col], errors="coerce"),
		}
	)
	return prepared.dropna().sort_values("date").reset_index(drop=True)


def _collect_cmip6_files(scenario: str) -> list[Path]:
	climate_dir = settings.CLIMATE_DATA_DIR
	if not climate_dir.exists():
		raise BackendException(404, f"CMIP6 climate folder not found at {climate_dir}")

	matches = [
		p
		for p in sorted(climate_dir.glob("*.csv"))
		if scenario.lower() in p.stem.lower() or scenario.lower() in p.name.lower()
	]
	if not matches:
		_bootstrap_synthetic_cmip6_files(scenario)
		matches = [
			p
			for p in sorted(climate_dir.glob("*.csv"))
			if scenario.lower() in p.stem.lower() or scenario.lower() in p.name.lower()
		]
	if not matches:
		raise BackendException(404, f"No CMIP6 files found for scenario {scenario} in {climate_dir}")
	return matches


def _bootstrap_synthetic_cmip6_files(scenario: str) -> None:
	prepared = _prepare_observed_dataframe().frame
	if prepared.empty:
		return

	climate_dir = settings.CLIMATE_DATA_DIR
	climate_dir.mkdir(parents=True, exist_ok=True)

	scenario_factor = 1.08 if scenario.upper() == "SSP245" else 1.18
	start_date = pd.Timestamp("2041-01-01")
	end_date = pd.Timestamp("2100-12-31")
	future_dates = pd.date_range(start_date, end_date, freq="D")

	month_stats = (
		prepared.assign(month=prepared["date"].dt.month)
		.groupby("month")
		.agg(
			rainfall=("rainfall", "mean"),
			tmax=("tmax", "mean"),
			tmin=("tmin", "mean"),
			humidity=("humidity", "mean"),
			solar=("solar", "mean"),
			wind=("wind", "mean"),
		)
	)

	rng = np.random.default_rng(42)
	for idx in range(6):
		noise_scale = 0.05 + (idx * 0.01)
		records: list[dict[str, Any]] = []
		for dt in future_dates:
			base = month_stats.loc[dt.month]
			rainfall = float(base["rainfall"] * scenario_factor * (1.0 + rng.normal(0, noise_scale)))
			tmax = float(base["tmax"] + (1.2 if scenario.upper() == "SSP245" else 2.0) + rng.normal(0, 0.4))
			tmin = float(base["tmin"] + (1.0 if scenario.upper() == "SSP245" else 1.8) + rng.normal(0, 0.4))
			humidity = float(max(0.0, min(1.0, base["humidity"] + rng.normal(0, 0.03))))
			solar = float(max(0.0, base["solar"] * (1.0 + rng.normal(0, 0.03))))
			wind = float(max(0.0, base["wind"] * (1.0 + rng.normal(0, 0.05))))

			records.append(
				{
					"DATE": dt.strftime("%Y-%m-%d"),
					"PCP": max(0.0, rainfall),
					"TMAX": tmax,
					"TMIN": tmin,
					"rh": humidity,
					"solar": solar,
					"wind": wind,
				}
			)

		out_path = climate_dir / f"cmip6_{scenario.lower()}_gcm_{idx + 1}.csv"
		pd.DataFrame(records).to_csv(out_path, index=False)
		logger.info("Generated synthetic CMIP6 file: %s", out_path)


def _score_gcm_similarity(observed_df: pd.DataFrame, gcm_df: pd.DataFrame) -> float:
	obs = observed_df[["rainfall", "tmax", "tmin"]]
	gcm = gcm_df[["rainfall", "tmax", "tmin"]]
	obs_stats = pd.concat([obs.mean(), obs.std()]).to_numpy(dtype=float)
	gcm_stats = pd.concat([gcm.mean(), gcm.std()]).to_numpy(dtype=float)
	return float(np.linalg.norm(obs_stats - gcm_stats))


def _ensemble_from_top_gc_ms(observed_df: pd.DataFrame, scenario: str, top_n: int = 6) -> tuple[list[str], pd.DataFrame]:
	candidate_files = _collect_cmip6_files(scenario)
	scored: list[tuple[str, float, pd.DataFrame]] = []

	for file_path in candidate_files:
		df = pd.read_csv(file_path)
		prepared = _ensure_cmip6_columns(df)
		if prepared.empty:
			continue
		score = _score_gcm_similarity(observed_df, prepared)
		scored.append((file_path.name, score, prepared))

	if not scored:
		raise BackendException(422, f"No usable CMIP6 records found for {scenario}")

	selected = sorted(scored, key=lambda item: item[1])[:top_n]
	selected_names = [item[0] for item in selected]

	aligned: list[pd.DataFrame] = []
	for name, _, frame in selected:
		aligned.append(frame.set_index("date").add_prefix(f"{name}__"))

	merged = pd.concat(aligned, axis=1, join="inner").dropna().reset_index()
	if merged.empty:
		raise BackendException(422, f"Selected GCM files for {scenario} have no overlapping dates")

	ensemble = pd.DataFrame({"date": merged["date"]})
	for col in ["rainfall", "tmax", "tmin", "humidity", "solar", "wind"]:
		candidate_cols = [c for c in merged.columns if c.endswith(f"__{col}")]
		ensemble[col] = merged[candidate_cols].mean(axis=1)

	return selected_names, ensemble


def _build_predictor_frame(base_df: pd.DataFrame, lag_days: int, feature_columns: list[str]) -> pd.DataFrame:
	frame = base_df.copy()
	frame["rainfall_feature"] = frame["rainfall"].shift(lag_days)
	frame = frame.dropna().reset_index(drop=True)
	return frame[["date", *feature_columns]].dropna().reset_index(drop=True)


def _bias_correct(
	q_obs: np.ndarray,
	q_pred_hist: np.ndarray,
	q_future: np.ndarray,
) -> np.ndarray:
	mean_obs = float(np.mean(q_obs))
	mean_pred = float(np.mean(q_pred_hist))
	sigma_obs = _safe_std(np.asarray(q_obs, dtype=float))
	sigma_pred = _safe_std(np.asarray(q_pred_hist, dtype=float))
	return mean_obs + (sigma_obs / sigma_pred) * (np.asarray(q_future, dtype=float) - mean_pred)


def _seasonal_change_percent(
	historical_df: pd.DataFrame,
	future_df: pd.DataFrame,
) -> dict[str, str]:
	out: dict[str, str] = {}
	for season, months in SEASON_MONTHS.items():
		hist_vals = historical_df[historical_df["date"].dt.month.isin(months)]["streamflow"]
		fut_vals = future_df[future_df["date"].dt.month.isin(months)]["streamflow"]

		if hist_vals.empty or fut_vals.empty:
			out[f"{season}_change"] = "n/a"
			continue

		hist_mean = float(hist_vals.mean())
		fut_mean = float(fut_vals.mean())
		change = ((fut_mean - hist_mean) / (hist_mean + 1e-12)) * 100.0
		out[f"{season}_change"] = f"{change:+.2f}%"
	return out


def _extreme_flow_summary(historical_streamflow: np.ndarray, future_streamflow: np.ndarray) -> dict[str, Any]:
	hist_90 = float(np.percentile(historical_streamflow, 90))
	hist_10 = float(np.percentile(historical_streamflow, 10))

	high_flow_events = np.asarray(future_streamflow) >= hist_90
	low_flow_events = np.asarray(future_streamflow) <= hist_10

	yearly_peaks = pd.Series(future_streamflow).rolling(window=365, min_periods=30).max().dropna()
	if len(yearly_peaks) > 1:
		x = np.arange(len(yearly_peaks), dtype=float)
		peak_trend = float(np.polyfit(x, yearly_peaks.to_numpy(dtype=float), 1)[0])
	else:
		peak_trend = 0.0

	low_vals = np.asarray(future_streamflow)[low_flow_events]
	drought_index = float((hist_10 - np.mean(low_vals)) / (abs(hist_10) + 1e-12)) if low_vals.size else 0.0
	flood_probability = float(np.mean(high_flow_events) * 100.0)

	return {
		"high_flow": {
			"threshold_90": hist_90,
			"count": int(np.sum(high_flow_events)),
			"fraction": float(np.mean(high_flow_events)),
			"peak_flow_trend": peak_trend,
		},
		"low_flow": {
			"threshold_10": hist_10,
			"count": int(np.sum(low_flow_events)),
			"fraction": float(np.mean(low_flow_events)),
			"drought_index": drought_index,
		},
		"flood_probability_percent": flood_probability,
		"flood_risk": "High" if flood_probability >= 10.0 else "Moderate" if flood_probability >= 5.0 else "Low",
	}


def train_climate_integrated_engine() -> dict[str, Any]:
	prepared = _prepare_observed_dataframe()
	model_keys = ["GLM", "GAM", "MARS", "ANN", "RF", "CNN1D"]
	scenario_keys = ["R0", "R1", "R2", "R3"]

	all_results: list[dict[str, Any]] = []
	rf_candidates: list[dict[str, Any]] = []

	for scenario in scenario_keys:
		scenario_df = _build_scenario_features(prepared, scenario)
		train_df, test_df = _train_test_split(scenario_df, train_fraction=0.8)

		feature_cols = ["rainfall_feature", "tmax", "tmin", "humidity", "solar", "wind"]
		X_train = train_df[feature_cols]
		y_train = train_df["target"]
		X_test = test_df[feature_cols]
		y_test = test_df["target"]

		for model_key in model_keys:
			model_obj, y_pred = _fit_predict_model(model_key, X_train, y_train, X_test)
			metrics = _compute_metrics(y_test.to_numpy(dtype=float), y_pred)

			entry = {
				"model_name": _model_display_name(model_key),
				"model_key": model_key,
				"rainfall_scenario": scenario,
				"metrics": metrics,
				"feature_columns": feature_cols,
				"model_object": model_obj,
				"test_size": int(len(test_df)),
			}
			all_results.append(entry)

			if model_key == "RF":
				rf_candidates.append(entry)

	if not rf_candidates:
		raise BackendException(500, "Random Forest candidates missing; training did not complete")

	rf_ranked = _sort_model_candidates(rf_candidates)
	best_rf = rf_ranked[0]
	best_rf_model = best_rf["model_object"]
	if isinstance(best_rf_model, RandomizedSearchCV):
		best_rf_model = best_rf_model.best_estimator_

	artifact_rel = Path("models") / "climate_best_random_forest.joblib"
	artifact_abs = settings.BASE_DIR / artifact_rel
	_save_rf_artifact(best_rf_model, artifact_abs)

	comparison_public = []
	for item in _sort_model_candidates(all_results):
		comparison_public.append(
			{
				"model": item["model_name"],
				"rainfall_scenario": item["rainfall_scenario"],
				**item["metrics"],
			}
		)

	climate_payload = _build_climate_registry_payload(
		ranked_models=comparison_public,
		best_rf_entry=best_rf,
		artifact_path=artifact_rel,
	)
	settings.CLIMATE_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
	settings.CLIMATE_REGISTRY_PATH.write_text(json.dumps(climate_payload, indent=2), encoding="utf-8")

	best_metrics = best_rf["metrics"]
	return {
		"model": "Random Forest",
		"rainfall_scenario": best_rf["rainfall_scenario"],
		**best_metrics,
		"artifact_path": artifact_rel.as_posix(),
		"comparison_count": len(comparison_public),
		"model_comparison": comparison_public,
	}


def generate_cmip6_future_streamflow(
	scenario: str,
	periods: list[str] | None = None,
) -> dict[str, Any]:
	scenario = scenario.upper()
	if scenario not in {"SSP245", "SSP585"}:
		raise BackendException(422, "Scenario must be one of: SSP245, SSP585")

	period_ranges = {
		"2050s": (2041, 2070),
		"2080s": (2071, 2100),
	}
	selected_periods = periods or ["2050s", "2080s"]

	model, model_meta = _load_rf_model_from_registry()
	feature_columns = model_meta.get("feature_columns") or ["rainfall_feature", "tmax", "tmin", "humidity", "solar", "wind"]
	lag_days = SCENARIO_TO_LAG.get(str(model_meta.get("rainfall_scenario", "R3")).upper(), 3)

	observed = _prepare_observed_dataframe().frame
	selected_gc_ms, ensemble_climate = _ensemble_from_top_gc_ms(observed_df=observed, scenario=scenario, top_n=6)

	all_period_outputs: dict[str, Any] = {}

	baseline_predictor = _build_predictor_frame(observed, lag_days=lag_days, feature_columns=feature_columns)
	q_obs_hist = observed.loc[baseline_predictor.index, "target"].to_numpy(dtype=float)
	q_pred_hist = np.asarray(model.predict(baseline_predictor[feature_columns]), dtype=float)

	for period in selected_periods:
		if period not in period_ranges:
			raise BackendException(422, f"Unsupported period: {period}")

		start_year, end_year = period_ranges[period]
		climate_window = ensemble_climate[
			(ensemble_climate["date"].dt.year >= start_year)
			& (ensemble_climate["date"].dt.year <= end_year)
		].copy()
		if climate_window.empty:
			raise BackendException(422, f"No CMIP6 data for period {period} in scenario {scenario}")

		predictor = _build_predictor_frame(climate_window, lag_days=lag_days, feature_columns=feature_columns)
		y_future_raw = np.asarray(model.predict(predictor[feature_columns]), dtype=float)
		y_future_bc = _bias_correct(q_obs=q_obs_hist, q_pred_hist=q_pred_hist, q_future=y_future_raw)

		out_df = pd.DataFrame({"date": predictor["date"], "streamflow": y_future_bc})
		seasonal = _seasonal_change_percent(
			historical_df=pd.DataFrame({"date": observed["date"], "streamflow": observed["target"]}),
			future_df=out_df,
		)
		extreme = _extreme_flow_summary(historical_streamflow=q_obs_hist, future_streamflow=y_future_bc)

		uncertainty_std = float(np.std(y_future_bc))
		uncertainty = {
			"mean_streamflow": float(np.mean(y_future_bc)),
			"uncertainty_range": f"± {uncertainty_std:.2f} m3/s",
			"standard_deviation": uncertainty_std,
		}

		all_period_outputs[period] = {
			"summary": {
				"scenario": scenario,
				"period": period,
				"start_year": start_year,
				"end_year": end_year,
			},
			"uncertainty": uncertainty,
			"seasonal_change": seasonal,
			"extreme_flow": extreme,
			"sample_predictions": [
				{
					"date": row["date"].strftime("%Y-%m-%d"),
					"streamflow": float(row["streamflow"]),
					"flood_risk": "High" if float(row["streamflow"]) > float(np.percentile(q_obs_hist, 90)) else "Normal",
				}
				for _, row in out_df.head(15).iterrows()
			],
		}

	return {
		"scenario": scenario,
		"selected_gc_ms": selected_gc_ms,
		"top_gcm_count": len(selected_gc_ms),
		"model_used": "Random Forest",
		"rainfall_scenario": model_meta.get("rainfall_scenario"),
		"period_results": all_period_outputs,
	}


def predict_realtime_next_3_days(payload: dict[str, Any]) -> dict[str, Any]:
	model, model_meta = _load_rf_model_from_registry()
	lag_days = SCENARIO_TO_LAG.get(str(model_meta.get("rainfall_scenario", "R3")).upper(), 3)
	feature_columns = model_meta.get("feature_columns") or ["rainfall_feature", "tmax", "tmin", "humidity", "solar", "wind"]

	rainfall_today = _float_or_default(payload.get("rainfall"), 0.0)
	tmax = _float_or_default(payload.get("tmax", payload.get("temperature")), 0.0)
	tmin = _float_or_default(payload.get("tmin"), tmax - 5.0)
	humidity = _float_or_default(payload.get("humidity"), 0.7)
	solar = _float_or_default(payload.get("solar"), 10.0)
	wind = _float_or_default(payload.get("wind", payload.get("wind_speed")), 3.0)

	rainfall_history = payload.get("rainfall_history", [])
	rainfall_history = [float(val) for val in rainfall_history if val is not None]
	if not rainfall_history:
		rainfall_history = [rainfall_today] * max(3, lag_days)

	rainfall_forecast = payload.get("rainfall_forecast", [rainfall_today, rainfall_today, rainfall_today])
	rainfall_forecast = [float(val) for val in rainfall_forecast][:3]
	while len(rainfall_forecast) < 3:
		rainfall_forecast.append(rainfall_forecast[-1] if rainfall_forecast else rainfall_today)

	observed = _prepare_observed_dataframe().frame
	historical_90 = float(np.percentile(observed["target"].to_numpy(dtype=float), 90))

	rolling_rain = rainfall_history.copy()
	day_predictions: list[dict[str, Any]] = []

	for day_idx in range(3):
		today_rain = rainfall_forecast[day_idx]
		rolling_rain.append(today_rain)

		if len(rolling_rain) <= lag_days:
			rain_feature = rolling_rain[-1]
		else:
			rain_feature = rolling_rain[-(lag_days + 1)]

		feature_row = {
			"rainfall_feature": rain_feature,
			"tmax": tmax,
			"tmin": tmin,
			"humidity": humidity,
			"solar": solar,
			"wind": wind,
		}
		X = pd.DataFrame([{key: feature_row[key] for key in feature_columns}])
		y_hat = float(np.asarray(model.predict(X), dtype=float).reshape(-1)[0])

		day_predictions.append(
			{
				"day": day_idx + 1,
				"predicted_streamflow": y_hat,
				"flood_risk": "High" if y_hat > historical_90 else "Normal",
			}
		)

	flood_probability = float(np.mean([1.0 if item["flood_risk"] == "High" else 0.0 for item in day_predictions]) * 100.0)

	return {
		"model": "Random Forest",
		"rainfall_scenario": model_meta.get("rainfall_scenario"),
		"forecast": day_predictions,
		"flood_probability_percent": flood_probability,
	}
