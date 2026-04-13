from datetime import datetime
from io import BytesIO
from pathlib import Path
import logging
import os
import re
import warnings
from urllib.parse import quote

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)
logging.getLogger("tensorflow").disabled = True
logging.getLogger("absl").disabled = True
warnings.filterwarnings("ignore", message=".*reset_default_graph.*")

from utils.dashboard import (
    create_glass_card,
    create_kpi_card,
    detect_datetime_column,
    detect_discharge_column,
    feature_importance_chart,
    get_risk_label,
    historical_discharge_chart,
    inject_custom_css,
    predicted_vs_observed_chart,
    sparkline_chart,
)
from utils.metrics import nse, r2, rmse
from utils.predict import (
    align_features,
    get_feature_importance,
    get_feature_list,
    load_artifacts,
    predict_runoff,
)
from utils.preprocessing import (
    REALISTIC_DISCHARGE_MAX_CUMEC,
    get_discharge_normalization_report,
    load_dataset,
    prepare_features,
)


st.set_page_config(page_title="Hydrological Intelligence Platform", page_icon="🌊", layout="wide")
inject_custom_css()

base_dir = Path(__file__).resolve().parent
model_dir = base_dir / "models"
dataset_path = base_dir / "datasets" / "Kasol.xlsx"
dataset_csv_path = base_dir / "datasets" / "Kasol.csv"
DATASET_CACHE_VERSION = "20260327-dashboard-discharge-fix"


@st.cache_resource(show_spinner=False)
def _get_artifacts(path: Path):
    return load_artifacts(path, load_lstm=True)


@st.cache_data(show_spinner=False)
def _get_default_dataset(path: Path, cache_version: str, modified_ns: int, file_size: int):
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return load_dataset(path)
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def _get_csv_dataset(path: Path, cache_version: str, modified_ns: int, file_size: int):
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return load_dataset(path)
    except Exception:
        return pd.DataFrame()


artifacts = _get_artifacts(model_dir)
feature_list = get_feature_list(artifacts)
best_model_ready = artifacts.get("best_model_obj") is not None
best_model_label = "BEST"
rf_model_ready = artifacts.get("rf_model") is not None
lstm_model_ready = artifacts.get("lstm_model") is not None
model_options: list[str] = []
if rf_model_ready:
    model_options.append("Random Forest")
if lstm_model_ready:
    model_options.append("LSTM")
if not model_options:
    model_options = ["Random Forest"]
model_metadata = artifacts.get("model_metadata") if isinstance(artifacts.get("model_metadata"), dict) else {}


def _safe_int(value: object, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _model_forecast_horizon_days(model_name: str | None) -> int:
    normalized = str(model_name or "").strip().lower()
    rf_lstm_horizon = max(0, _safe_int(model_metadata.get("forecast_horizon_days", 3), 3))

    if normalized in {"random forest", "rf", "lstm"}:
        return rf_lstm_horizon

    if normalized in {best_model_label.lower(), "best model", "xgboost (best)", "xgboost", "xgboost_log1p"}:
        best_meta = artifacts.get("best_model") if isinstance(artifacts.get("best_model"), dict) else {}
        return max(0, _safe_int(best_meta.get("forecast_horizon_days", 0), 0))

    return rf_lstm_horizon

if not feature_list:
    feature_list = [
        "PCP",
        "P1",
        "P2",
        "P3",
        "TMAX",
        "TMIN",
        "rh",
        "solar",
        "wind",
        "lag_discharge_1",
        "lag_discharge_2",
        "lag_discharge_3",
        "season_sin",
        "season_cos",
    ]

default_token = (
    DATASET_CACHE_VERSION,
    dataset_path.stat().st_mtime_ns if dataset_path.exists() else 0,
    dataset_path.stat().st_size if dataset_path.exists() else 0,
)
csv_token = (
    DATASET_CACHE_VERSION,
    dataset_csv_path.stat().st_mtime_ns if dataset_csv_path.exists() else 0,
    dataset_csv_path.stat().st_size if dataset_csv_path.exists() else 0,
)

default_data = _get_default_dataset(dataset_path, *default_token)
csv_data = _get_csv_dataset(dataset_csv_path, *csv_token)
discharge_col = detect_discharge_column(default_data) if not default_data.empty else None
if (default_data.empty or discharge_col is None) and dataset_path.exists() and dataset_path.stat().st_size > 0:
    try:
        refreshed_default_data = load_dataset(dataset_path)
        refreshed_discharge_col = detect_discharge_column(refreshed_default_data) if not refreshed_default_data.empty else None
        if not refreshed_default_data.empty and refreshed_discharge_col is not None:
            default_data = refreshed_default_data
            discharge_col = refreshed_discharge_col
    except Exception:
        pass
discharge_report = get_discharge_normalization_report(default_data, discharge_col) if not default_data.empty else {}


def _discharge_display_factor(df: pd.DataFrame, discharge_column: str | None) -> float:
    return 1.0


discharge_factor = _discharge_display_factor(default_data, discharge_col)
discharge_unit_label = "CUMEC"
MAX_PREDICTION_DISCHARGE_CUMEC = 100_000.0
LSTM_SCENARIO_MAX_DAYS = 360
source_discharge_factor = float(discharge_report.get("applied_factor_to_cumec", 1.0) or 1.0)


def _artifacts_use_normalized_cumec(artifact_payload: dict) -> bool:
    best_model_meta = artifact_payload.get("best_model") if isinstance(artifact_payload.get("best_model"), dict) else {}
    return bool(best_model_meta.get("target_already_normalized_to_cumec"))


legacy_prediction_factor = (
    1.0
)
legacy_input_factor = 1.0


def _prediction_display_factor(model_name: str | None, value: float | None = None) -> float:
    return 1.0


def _to_display_discharge(value: float | None, model_name: str | None = None) -> float | None:
    if value is None:
        return None
    return float(value) * _prediction_display_factor(model_name, value)


def _adapt_feature_units_for_loaded_models(feature_df: pd.DataFrame) -> pd.DataFrame:
    return feature_df.copy() if feature_df is not None else pd.DataFrame()


def _predict_runoff_with_loaded_model_units(feature_df: pd.DataFrame, model_choice: str) -> np.ndarray:
    prepared = _adapt_feature_units_for_loaded_models(feature_df)
    return predict_runoff(prepared, artifacts, model_choice=model_choice)


def _coerce_numeric_values(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return numeric

    text = series.astype(str).str.strip()
    text = text.replace({"": np.nan, "nan": np.nan, "none": np.nan, "None": np.nan, "-": np.nan})
    cleaned = text.str.replace(r"[^0-9,.\-+]", "", regex=True)

    has_comma = cleaned.str.contains(",", na=False)
    has_dot = cleaned.str.contains(r"\.", regex=True, na=False)

    thousands_mask = has_comma & has_dot
    cleaned.loc[thousands_mask] = cleaned.loc[thousands_mask].str.replace(",", "", regex=False)

    decimal_comma_mask = has_comma & ~has_dot
    cleaned.loc[decimal_comma_mask] = cleaned.loc[decimal_comma_mask].str.replace(",", ".", regex=False)

    return pd.to_numeric(cleaned, errors="coerce")


@st.cache_data(show_spinner=False)
def _load_discharge_from_workbook(path: Path, modified_ns: int, file_size: int) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()

    try:
        workbook = pd.read_excel(path, sheet_name=None)
    except Exception:
        return pd.DataFrame()

    preferred_order = ["Sheet1", "sheet1", "Sheet5", "sheet5"]
    ordered_sheet_names = [name for name in preferred_order if name in workbook]
    ordered_sheet_names += [name for name in workbook.keys() if name not in ordered_sheet_names]

    for sheet_name in ordered_sheet_names:
        frame = workbook.get(sheet_name)
        if frame is None or frame.empty:
            continue

        normalized_columns = {"".join(ch for ch in str(col).lower() if ch.isalnum()): col for col in frame.columns}
        date_col = next(
            (
                col
                for token, col in normalized_columns.items()
                if "date" in token or "time" in token
            ),
            None,
        )
        discharge_col = next(
            (
                col
                for token, col in normalized_columns.items()
                if (
                    "discharge" in token
                    or "cumec" in token
                    or "runoff" in token
                    or "streamflow" in token
                    or token in {"flow", "discumec"}
                )
            ),
            None,
        )

        if discharge_col is None:
            continue

        discharge_values = _coerce_numeric_values(frame[discharge_col])
        if date_col is not None:
            parsed_dates = pd.to_datetime(frame[date_col], errors="coerce", dayfirst=True)
            if parsed_dates.notna().sum() == 0:
                parsed_dates = pd.to_datetime(frame[date_col], errors="coerce")
        else:
            parsed_dates = pd.Series(pd.NaT, index=frame.index)

        if parsed_dates.notna().sum() == 0:
            parsed_dates = pd.date_range(start="1970-01-01", periods=len(frame), freq="D")

        discharge_df = pd.DataFrame(
            {
                "Date": parsed_dates,
                "Discharge (CUMEC)": discharge_values,
            }
        ).dropna(subset=["Date", "Discharge (CUMEC)"])

        if not discharge_df.empty:
            return discharge_df.reset_index(drop=True)

    return pd.DataFrame()


def _prepare_dashboard_discharge_frame(frame: pd.DataFrame, column: str | None) -> tuple[pd.DataFrame, str | None]:
    if frame is None or frame.empty or column is None or column not in frame.columns:
        return pd.DataFrame(), None

    prepared = frame.copy()
    prepared[column] = _coerce_numeric_values(prepared[column])
    date_col = detect_datetime_column(prepared)

    if date_col and date_col in prepared.columns:
        parsed_dates = pd.to_datetime(prepared[date_col], errors="coerce", dayfirst=True)
        if parsed_dates.notna().sum() == 0:
            parsed_dates = pd.to_datetime(prepared[date_col], errors="coerce")
        prepared[date_col] = parsed_dates
        prepared = prepared.dropna(subset=[date_col, column]).sort_values(date_col)
    else:
        prepared = prepared.dropna(subset=[column]).reset_index(drop=True)

    if prepared.empty:
        return pd.DataFrame(), None
    return prepared, column


def _load_dashboard_discharge_source() -> tuple[pd.DataFrame, str | None]:
    frame = default_data
    column = discharge_col if frame is not None and not frame.empty else None
    if frame is not None and not frame.empty and column is None:
        column = detect_discharge_column(frame)

    prepared_frame, prepared_col = _prepare_dashboard_discharge_frame(frame, column)
    if not prepared_frame.empty and prepared_col is not None:
        return prepared_frame, prepared_col

    try:
        refreshed_frame = load_dataset(dataset_path)
    except Exception:
        refreshed_frame = pd.DataFrame()

    refreshed_col = detect_discharge_column(refreshed_frame) if not refreshed_frame.empty else None
    prepared_refreshed, prepared_refreshed_col = _prepare_dashboard_discharge_frame(refreshed_frame, refreshed_col)
    if not prepared_refreshed.empty and prepared_refreshed_col is not None:
        return prepared_refreshed, prepared_refreshed_col

    fallback_frame = _load_discharge_from_workbook(
        dataset_path,
        dataset_path.stat().st_mtime_ns if dataset_path.exists() else 0,
        dataset_path.stat().st_size if dataset_path.exists() else 0,
    )
    prepared_fallback, prepared_fallback_col = _prepare_dashboard_discharge_frame(fallback_frame, "Discharge (CUMEC)")
    if not prepared_fallback.empty and prepared_fallback_col is not None:
        return prepared_fallback, prepared_fallback_col

    return pd.DataFrame(), None


def _normalize_name(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _extract_numeric_series(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)

    normalized_columns = {_normalize_name(col): col for col in df.columns}
    for candidate in candidates:
        resolved_col = normalized_columns.get(_normalize_name(candidate))
        if resolved_col is None:
            continue
        series = pd.to_numeric(df[resolved_col], errors="coerce").dropna()
        if not series.empty:
            return series
    return pd.Series(dtype=float)


def _infer_training_unit_profile(df: pd.DataFrame) -> tuple[bool, bool]:
    """
    Returns:
    - rh_is_fraction: True when RH appears in 0..1 scale.
    - solar_is_low_scale: True when solar appears in low-scale daily-energy style values.
    """
    rh_series = _extract_numeric_series(df, ["rh", "RH"])
    if rh_series.empty:
        rh_is_fraction = True
    else:
        rh_q95 = float(rh_series.quantile(0.95))
        rh_is_fraction = rh_q95 <= 1.5

    solar_series = _extract_numeric_series(df, ["solar"])
    if solar_series.empty:
        solar_is_low_scale = True
    else:
        solar_median = float(solar_series.median())
        solar_is_low_scale = solar_median <= 40.0

    return rh_is_fraction, solar_is_low_scale


RH_IS_FRACTION, SOLAR_IS_LOW_SCALE = _infer_training_unit_profile(default_data)


def _seasonal_terms(date_value) -> tuple[float, float]:
    ts = pd.Timestamp(date_value)
    angle = (2 * np.pi * ts.dayofyear) / 365.25
    return float(np.sin(angle)), float(np.cos(angle))


def _feature_default_value(feature_name: str, date_value, df: pd.DataFrame, discharge_column: str | None) -> float:
    lower_name = feature_name.lower()
    season_sin, season_cos = _seasonal_terms(date_value)

    if lower_name == "season_sin":
        return season_sin
    if lower_name == "season_cos":
        return season_cos

    if lower_name.startswith("lag_discharge_") and discharge_column and discharge_column in df.columns:
        lag_steps = int(lower_name.rsplit("_", 1)[-1]) if lower_name.rsplit("_", 1)[-1].isdigit() else 1
        discharge_series = pd.to_numeric(df[discharge_column], errors="coerce").dropna()
        if discharge_series.size >= lag_steps:
            return float(discharge_series.iloc[-lag_steps])
        if not discharge_series.empty:
            return float(discharge_series.median())
        return 0.0

    if lower_name == "discharge_roll_mean_3" and discharge_column and discharge_column in df.columns:
        discharge_series = pd.to_numeric(df[discharge_column], errors="coerce").dropna()
        if not discharge_series.empty:
            return float(discharge_series.tail(3).mean())
        return 0.0

    if lower_name == "discharge_roll_std_3" and discharge_column and discharge_column in df.columns:
        discharge_series = pd.to_numeric(df[discharge_column], errors="coerce").dropna()
        if discharge_series.size >= 2:
            return float(discharge_series.tail(3).std(ddof=0))
        return 0.0

    feature_candidates = {
        "mean_pcp": ["Mean_PCP", "PCP", "P1", "P2", "P3"],
        "mean_tmax": ["Mean_Tmax", "TMAX"],
        "mean_tmin": ["Mean_Tmin", "TMIN"],
        "rh": ["rh", "RH"],
        "solar": ["solar"],
        "wind": ["wind", "wind "],
        "pcp_roll_mean_3": ["PCP_roll_mean_3", "PCP"],
        "pcp_roll_sum_7": ["PCP_roll_sum_7", "PCP"],
    }

    candidates = feature_candidates.get(lower_name, [feature_name])
    series = _extract_numeric_series(df, candidates)
    if series.empty:
        return 0.0

    if lower_name == "pcp_roll_mean_3":
        return float(series.tail(3).mean())
    if lower_name == "pcp_roll_sum_7":
        return float(series.tail(7).sum())
    return float(series.median())


def _feature_group(feature_name: str) -> str:
    name = feature_name.lower()
    if "season_" in name:
        return "Seasonality (Auto)"
    if "lag_discharge" in name or "discharge_roll" in name:
        return "Hydrology Memory"
    if "pcp" in name or name in {"p1", "p2", "p3"}:
        return "Rainfall Inputs"
    if "tmax" in name or "tmin" in name or "temp" in name:
        return "Temperature"
    if "rh" in name or "solar" in name or "wind" in name:
        return "Atmosphere"
    return "Other Inputs"


def _feature_label(feature_name: str) -> str:
    rh_label = "Relative Humidity (fraction 0-1)" if RH_IS_FRACTION else "Relative Humidity (%)"
    solar_label = "Solar Radiation (MJ/m2/day)" if SOLAR_IS_LOW_SCALE else "Solar Radiation (W/m2)"
    labels = {
        "Mean_PCP": "Mean Precipitation (mm/day)",
        "Mean_Tmax": "Mean Tmax (deg C)",
        "Mean_Tmin": "Mean Tmin (deg C)",
        "rh": rh_label,
        "solar": solar_label,
        "wind": "Wind Speed (m/s)",
        "lag_discharge_1": "Discharge Lag-1 (CUMEC)",
        "lag_discharge_2": "Discharge Lag-2 (CUMEC)",
        "lag_discharge_3": "Discharge Lag-3 (CUMEC)",
        "season_sin": "Season Sinusoid",
        "season_cos": "Season Cosine",
        "discharge_roll_mean_3": "Discharge Rolling Mean-3 (CUMEC)",
        "discharge_roll_std_3": "Discharge Rolling Std-3 (CUMEC)",
        "PCP_roll_mean_3": "Precip Rolling Mean-3 (mm/day)",
        "PCP_roll_sum_7": "Precip Rolling Sum-7 (mm)",
    }
    return labels.get(feature_name, feature_name.replace("_", " ").title())


def _feature_help_text(feature_name: str) -> str:
    rh_help = (
        "Training data uses humidity as fraction (0-1). If you enter values >1, they will be interpreted as % and converted."
        if RH_IS_FRACTION
        else "Training data uses humidity as %. If you enter <=1 values, they will be interpreted as fraction and converted."
    )
    solar_help = (
        "Training data solar scale is low-range daily-energy values. High W/m2-style entries are converted automatically."
        if SOLAR_IS_LOW_SCALE
        else "Training data solar scale is W/m2-like values. Very small daily-energy-style entries are converted automatically."
    )
    help_text = {
        "season_sin": "Auto-derived from Forecast Date.",
        "season_cos": "Auto-derived from Forecast Date.",
        "lag_discharge_1": "Most recent observed discharge, normalized to CUMEC.",
        "lag_discharge_2": "Discharge from two timesteps before, normalized to CUMEC.",
        "lag_discharge_3": "Discharge from three timesteps before, normalized to CUMEC.",
        "rh": rh_help,
        "solar": solar_help,
    }
    return help_text.get(feature_name, "Pre-filled from historical values; discharge-derived defaults are normalized to CUMEC.")


def _input_step(default_value: float) -> float:
    magnitude = abs(default_value)
    if magnitude >= 1000:
        return 1.0
    if magnitude >= 100:
        return 0.5
    if magnitude >= 10:
        return 0.1
    return 0.01


def _discharge_input_alerts(input_values: dict[str, float]) -> list[str]:
    alerts: list[str] = []
    for feature_name, value in input_values.items():
        if "discharge" not in feature_name.lower():
            continue
        if abs(float(value)) > REALISTIC_DISCHARGE_MAX_CUMEC:
            alerts.append(f"{_feature_label(feature_name)} = {float(value):,.2f} {discharge_unit_label}")
    return alerts


def _normalize_user_input_units(input_values: dict[str, float]) -> tuple[dict[str, float], list[str]]:
    normalized = {key: float(value) for key, value in input_values.items()}
    adjustments: list[str] = []

    if "rh" in normalized:
        rh_value = float(normalized["rh"])
        if RH_IS_FRACTION and rh_value > 1.5:
            normalized["rh"] = rh_value / 100.0
            adjustments.append(f"Relative Humidity converted from {rh_value:.3f}% to {normalized['rh']:.4f} fraction.")
        elif (not RH_IS_FRACTION) and rh_value <= 1.5:
            normalized["rh"] = rh_value * 100.0
            adjustments.append(f"Relative Humidity converted from {rh_value:.4f} fraction to {normalized['rh']:.2f}%.")

    if "solar" in normalized:
        solar_value = float(normalized["solar"])
        if SOLAR_IS_LOW_SCALE and solar_value > 80.0:
            normalized["solar"] = solar_value / 11.57
            adjustments.append(
                f"Solar Radiation converted from {solar_value:.2f} W/m2-style to {normalized['solar']:.3f} low-scale units."
            )
        elif (not SOLAR_IS_LOW_SCALE) and solar_value < 40.0:
            normalized["solar"] = solar_value * 11.57
            adjustments.append(
                f"Solar Radiation converted from {solar_value:.3f} low-scale units to {normalized['solar']:.2f} W/m2-style."
            )

    return normalized, adjustments


def _build_feature_reference_ranges(feature_df: pd.DataFrame) -> dict[str, tuple[float, float, float, float]]:
    if feature_df is None or feature_df.empty:
        return {}

    ranges: dict[str, tuple[float, float, float, float]] = {}
    numeric = feature_df.apply(pd.to_numeric, errors="coerce")
    for col in numeric.columns:
        series = numeric[col].replace([np.inf, -np.inf], np.nan).dropna()
        if series.empty:
            continue
        p01 = float(series.quantile(0.01))
        p99 = float(series.quantile(0.99))
        vmin = float(series.min())
        vmax = float(series.max())
        ranges[str(col)] = (p01, p99, vmin, vmax)
    return ranges


def _input_ood_report(
    input_row: pd.DataFrame,
    reference_ranges: dict[str, tuple[float, float, float, float]],
) -> pd.DataFrame:
    if input_row is None or input_row.empty or not reference_ranges:
        return pd.DataFrame(columns=["Feature", "Input", "P01", "P99", "Train Min", "Train Max"])

    rows: list[dict[str, float | str]] = []
    row = input_row.iloc[0]
    for feature_name, value in row.items():
        if feature_name not in reference_ranges:
            continue
        if not np.isfinite(float(value)):
            continue
        p01, p99, vmin, vmax = reference_ranges[feature_name]
        if float(value) < p01 or float(value) > p99:
            rows.append(
                {
                    "Feature": feature_name,
                    "Input": float(value),
                    "P01": p01,
                    "P99": p99,
                    "Train Min": vmin,
                    "Train Max": vmax,
                }
            )
    return pd.DataFrame(rows)


if "auth_username" not in st.session_state:
    st.session_state["auth_username"] = "hydro_user"
if "auth_display_name" not in st.session_state:
    st.session_state["auth_display_name"] = "Hydro User"

if "user_profile" not in st.session_state or not isinstance(st.session_state["user_profile"], dict):
    st.session_state["user_profile"] = {
        "full_name": st.session_state.get("auth_display_name", ""),
        "mobile": "",
        "email": "",
        "organization": "",
        "designation": "",
        "location": "",
        "emergency_name": "",
        "emergency_mobile": "",
        "alert_channel": "Both",
        "notes": "",
    }


st.markdown(
    """
    <div class="hero">
        <h1 class="hero-title">Hydrological Intelligence Platform</h1>
        <p class="hero-subtitle">AI-Powered Runoff &amp; Flood Risk Forecasting</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if discharge_report:
    discharge_warning = discharge_report.get("warning")
    if discharge_warning:
        st.warning(str(discharge_warning))
    if legacy_prediction_factor != 1.0:
        st.info(
            "Loaded model artifacts appear to use the older raw discharge scale. "
            "Predictions are being compatibility-scaled to CUMEC; retrain the models on the fixed pipeline to remove this fallback."
        )

nav_descriptions = {
    "Dashboard": "Overview cards and latest risk snapshot.",
    "Analytics": "Inspect RMSE, R2, NSE and model comparison.",
    "Single Forecast": "Configure parameters and run one forecast.",
    "Batch Forecast": "Upload CSV and export bulk predictions.",
    "Scenario Simulator": "Run what-if climate scenarios and compare runoff impact.",
    "Profile": "Profile details and active session information.",
    "Model Status": "Inspect loaded artifacts and model readiness.",
    "Report Center": "Generate downloadable summary reports (CSV/PDF).",
    "About System": "System scope, approach, and platform summary.",
}
nav_config = [
    ("Menu", "Dashboard", "D"),
    ("Menu", "Analytics", "A"),
    ("Menu", "Single Forecast", "S"),
    ("Menu", "Batch Forecast", "B"),
    ("Apps", "Scenario Simulator", "C"),
    ("Tools", "Model Status", "M"),
    ("Tools", "Report Center", "R"),
    ("Tools", "About System", "I"),
]
nav_groups: dict[str, list[str]] = {}
nav_icons: dict[str, str] = {}
for section_name, nav_name, nav_icon in nav_config:
    nav_groups.setdefault(section_name, []).append(nav_name)
    nav_icons[nav_name] = nav_icon
nav_items = [nav_name for _, nav_name, _ in nav_config]
nav_route_map = {
    "Dashboard": "Dashboard",
    "Analytics": "Model Analytics",
    "Single Forecast": "Single Prediction",
    "Batch Forecast": "Batch Prediction",
    "Scenario Simulator": "Scenario Simulator",
    "Profile": "Profile",
    "Model Status": "Model Status",
    "Report Center": "Report Center",
    "About System": "About System",
}

selected_nav_label = st.query_params.get("nav", "Dashboard")
if isinstance(selected_nav_label, list):
    selected_nav_label = selected_nav_label[0] if selected_nav_label else "Dashboard"
allowed_direct_pages = {"Profile"}
if selected_nav_label not in set(nav_items).union(allowed_direct_pages):
    selected_nav_label = "Dashboard"
if "Profile" not in nav_icons:
    nav_icons["Profile"] = "U"

auth_display_name = st.session_state.get("auth_display_name", "User")
auth_username = st.session_state.get("auth_username", "user")
profile_data = st.session_state.get("user_profile", {})
profile_card_name = str(profile_data.get("full_name", "")).strip() or auth_display_name
profile_card_subtitle = str(profile_data.get("email", "")).strip() or f"@{auth_username}"
user_profile_href = f"?nav={quote('Profile')}"
st.sidebar.markdown(
    f"""
    <a class="sidebar-user-link" href="{user_profile_href}" target="_self">
        <div class="sidebar-user-card">
            <div class="sidebar-logo">AI</div>
            <div>
                <div class="sidebar-brand-title">{profile_card_name}</div>
                <div class="sidebar-brand-subtitle">{profile_card_subtitle}</div>
            </div>
        </div>
    </a>
    """,
    unsafe_allow_html=True,
)
st.sidebar.markdown("### Navigation")
for group_title, group_items in nav_groups.items():
    st.sidebar.markdown(f"<div class='nav-section-label'>{group_title}</div>", unsafe_allow_html=True)
    links_html = []
    for nav_name in group_items:
        active_class = " active" if selected_nav_label == nav_name else ""
        nav_icon = nav_icons.get(nav_name, ".")
        nav_href = f"?nav={quote(nav_name)}"
        links_html.append(
            f"<a class='sidebar-nav-link{active_class}' href='{nav_href}' target='_self'>"
            f"<span class='sidebar-nav-icon'>{nav_icon}</span>"
            f"<span class='sidebar-nav-text'>{nav_name}</span>"
            "</a>"
        )
    st.sidebar.markdown("".join(links_html), unsafe_allow_html=True)

selected_nav = nav_route_map.get(selected_nav_label, "Dashboard")
st.sidebar.markdown("---")
st.sidebar.markdown(
    f"""
    <div class="nav-active-note">
        <div class="nav-active-title">{nav_icons.get(selected_nav_label, '-')}  {selected_nav_label}</div>
        <div class="nav-active-subtitle">{nav_descriptions.get(selected_nav_label, "")}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

rf_ready = artifacts.get("rf_model") is not None
lstm_ready = artifacts.get("lstm_model") is not None
best_model_ready = artifacts.get("best_model_obj") is not None
rf_class = "status-ready" if rf_ready else "status-missing"
lstm_class = "status-ready" if lstm_ready else "status-missing"
st.sidebar.markdown(
    f"""
    <div class="status-row">
        <span class="status-chip {rf_class}">RF {'Ready' if rf_ready else 'Missing'}</span>
        <span class="status-chip {lstm_class}">LSTM {'Ready' if lstm_ready else 'Missing'}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

if st.sidebar.button("Log Out", width="stretch", type="secondary"):
    st.session_state["auth_username"] = "hydro_user"
    st.session_state["auth_display_name"] = "Hydro User"
    st.session_state["user_profile"] = {
        "full_name": "Hydro User",
        "mobile": "",
        "email": "",
        "organization": "",
        "designation": "",
        "location": "",
        "emergency_name": "",
        "emergency_mobile": "",
        "alert_channel": "Both",
        "notes": "",
    }
    st.query_params["nav"] = "Dashboard"
    st.rerun()

if "forecast_date" not in st.session_state:
    st.session_state["forecast_date"] = pd.Timestamp.today().date()
if "forecast_model" not in st.session_state:
    st.session_state["forecast_model"] = "Random Forest" if "Random Forest" in model_options else model_options[0]
if "single_prediction" not in st.session_state:
    st.session_state["single_prediction"] = None
if "risk_label" not in st.session_state:
    st.session_state["risk_label"] = "Normal"
if "risk_class" not in st.session_state:
    st.session_state["risk_class"] = "risk-normal"
if "risk_icon" not in st.session_state:
    st.session_state["risk_icon"] = "N"

selected_date = st.session_state["forecast_date"]
selected_model = st.session_state["forecast_model"]
single_prediction = st.session_state["single_prediction"]
risk_label = st.session_state["risk_label"]
risk_class = st.session_state["risk_class"]
risk_icon = st.session_state["risk_icon"]


def _is_valid_email(email_value: str) -> bool:
    if not email_value:
        return True
    return bool(re.match(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$", email_value))


def _is_valid_mobile(phone_value: str) -> bool:
    if not phone_value:
        return True
    digits_only = "".join(ch for ch in phone_value if ch.isdigit())
    return 10 <= len(digits_only) <= 15


def _downsample_df(df: pd.DataFrame, max_points: int = 1500) -> pd.DataFrame:
    if df is None or df.empty or len(df) <= max_points:
        return df
    step = max(1, int(np.ceil(len(df) / max_points)))
    return df.iloc[::step].copy()


def _timeseries_plot(df: pd.DataFrame, y_title: str = "", height: int = 360) -> go.Figure:
    fig = go.Figure()
    if df is None or df.empty:
        fig.update_layout(
            template="plotly_dark",
            height=height,
            margin={"l": 16, "r": 12, "t": 10, "b": 16},
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    plot_df = df.copy()
    x_values = plot_df.index if plot_df.index is not None else np.arange(len(plot_df))
    for col in plot_df.columns:
        y_vals = pd.to_numeric(plot_df[col], errors="coerce")
        trace_df = pd.DataFrame({"x": x_values, "y": y_vals}).dropna(subset=["y"])
        if trace_df.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=trace_df["x"],
                y=trace_df["y"],
                mode="lines",
                name=str(col),
                line={"width": 1.8},
            )
        )

    if not fig.data:
        fig.add_annotation(
            text="No chartable numeric data available.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"size": 14},
        )

    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin={"l": 16, "r": 12, "t": 10, "b": 16},
        legend={"orientation": "h", "y": -0.18, "x": 0.0},
        hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False, title=y_title)
    return fig


def _align_metric_inputs(observed: np.ndarray | None, predicted: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    observed_arr = np.asarray(observed if observed is not None else [], dtype=float).reshape(-1)
    predicted_arr = np.asarray(predicted if predicted is not None else [], dtype=float).reshape(-1)
    compare_len = min(observed_arr.size, predicted_arr.size)
    if compare_len == 0:
        return np.array([]), np.array([])

    observed_arr = observed_arr[:compare_len]
    predicted_arr = predicted_arr[:compare_len]
    valid_mask = np.isfinite(observed_arr) & np.isfinite(predicted_arr)
    if not np.any(valid_mask):
        return np.array([]), np.array([])
    return observed_arr[valid_mask], predicted_arr[valid_mask]


def _align_metric_inputs_with_horizon(
    observed: np.ndarray | pd.Series | None,
    predicted: np.ndarray | pd.Series | None,
    horizon_days: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    observed_arr = np.asarray(observed if observed is not None else [], dtype=float).reshape(-1)
    predicted_arr = np.asarray(predicted if predicted is not None else [], dtype=float).reshape(-1)
    if observed_arr.size == 0 or predicted_arr.size == 0:
        return np.array([]), np.array([])

    horizon = max(0, int(horizon_days))
    if horizon > 0:
        shifted = np.full(observed_arr.shape, np.nan, dtype=float)
        if observed_arr.size > horizon:
            shifted[:-horizon] = observed_arr[horizon:]
        observed_arr = shifted

    compare_len = min(observed_arr.size, predicted_arr.size)
    observed_arr = observed_arr[:compare_len]
    predicted_arr = predicted_arr[:compare_len]
    valid_mask = np.isfinite(observed_arr) & np.isfinite(predicted_arr)
    if not np.any(valid_mask):
        return np.array([]), np.array([])
    return observed_arr[valid_mask], predicted_arr[valid_mask]


def _percent_change_series(baseline: np.ndarray, scenario: np.ndarray) -> np.ndarray:
    baseline_arr = np.asarray(baseline, dtype=float).reshape(-1)
    scenario_arr = np.asarray(scenario, dtype=float).reshape(-1)
    compare_len = min(baseline_arr.size, scenario_arr.size)
    if compare_len == 0:
        return np.array([])
    baseline_arr = baseline_arr[:compare_len]
    scenario_arr = scenario_arr[:compare_len]
    denom = np.maximum(np.abs(baseline_arr), 1e-9)
    return ((scenario_arr - baseline_arr) / denom) * 100.0


def _resolve_column_name(df: pd.DataFrame, candidates: list[str]) -> str | None:
    normalized_columns = {_normalize_name(col): col for col in df.columns}
    for candidate in candidates:
        normalized_candidate = _normalize_name(candidate)
        if normalized_candidate in normalized_columns:
            return normalized_columns[normalized_candidate]
    return None


def _sanitize_simulation_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()

    clean = frame.copy()
    for col in clean.columns:
        if pd.api.types.is_numeric_dtype(clean[col]):
            clean[col] = pd.to_numeric(clean[col], errors="coerce").replace([np.inf, -np.inf], np.nan)

    numeric_cols = [col for col in clean.columns if pd.api.types.is_numeric_dtype(clean[col])]
    if numeric_cols:
        medians = clean[numeric_cols].median(numeric_only=True)
        clean[numeric_cols] = clean[numeric_cols].fillna(medians).fillna(0.0)
        clean[numeric_cols] = clean[numeric_cols].clip(lower=-1_000_000.0, upper=1_000_000.0)

    return clean.reset_index(drop=True)


def _scenario_range_drift_report(reference_df: pd.DataFrame, scenario_df: pd.DataFrame) -> pd.DataFrame:
    if reference_df is None or reference_df.empty or scenario_df is None or scenario_df.empty:
        return pd.DataFrame(columns=["Feature", "Column", "Train Min", "Train Max", "Scenario Min", "Scenario Max"])

    feature_tokens = {
        "precipitation": ["pcp", "precip", "rain", "p1", "p2", "p3"],
        "temperature": ["tmax", "tmin", "temp"],
        "evapotranspiration": ["evapo", "et", "eto"],
        "soil moisture": ["soil", "moist"],
        "watershed area": ["area", "basin"],
        "slope": ["slope", "gradient"],
        "land use": ["landuse", "landcover", "lulc"],
        "humidity": ["rh", "humid"],
        "solar radiation": ["solar", "radiation"],
        "wind": ["wind"],
        "discharge memory": ["lagdischarge", "dischargeroll", "discharge"],
    }

    rows: list[dict[str, object]] = []
    for col in scenario_df.columns:
        normalized_col = _normalize_name(col)
        feature_group = None
        for label, tokens in feature_tokens.items():
            if any(token in normalized_col for token in tokens):
                feature_group = label
                break
        if feature_group is None:
            continue

        resolved_ref_col = _resolve_column_name(reference_df, [str(col)])
        if resolved_ref_col is None:
            continue

        ref_series = pd.to_numeric(reference_df[resolved_ref_col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        scen_series = pd.to_numeric(scenario_df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if ref_series.empty or scen_series.empty:
            continue

        ref_min = float(ref_series.min())
        ref_max = float(ref_series.max())
        scen_min = float(scen_series.min())
        scen_max = float(scen_series.max())
        if scen_min < ref_min or scen_max > ref_max:
            rows.append(
                {
                    "Feature": feature_group.title(),
                    "Column": str(col),
                    "Train Min": ref_min,
                    "Train Max": ref_max,
                    "Scenario Min": scen_min,
                    "Scenario Max": scen_max,
                }
            )

    return pd.DataFrame(rows)


def _unit_consistency_warnings(reference_df: pd.DataFrame, scenario_df: pd.DataFrame) -> list[str]:
    if reference_df is None or reference_df.empty or scenario_df is None or scenario_df.empty:
        return []

    warnings_list: list[str] = []

    def _median(df: pd.DataFrame, col: str) -> float | None:
        if col not in df.columns:
            return None
        series = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if series.empty:
            return None
        return float(series.median())

    # Temperature consistency (deg C vs Kelvin).
    for temp_col in ["Mean_Tmax", "Mean_Tmin", "TMAX", "TMIN"]:
        ref = _median(reference_df, temp_col)
        scen = _median(scenario_df, temp_col)
        if ref is None or scen is None:
            continue
        if (ref < 100.0 and scen > 200.0) or (ref > 200.0 and scen < 100.0):
            warnings_list.append(f"{temp_col}: potential temperature unit mismatch (deg C vs K).")

    # Rainfall consistency (mm vs m) and discharge (m3/s vs m3/day), area (km2 vs m2).
    checks = [
        ("rainfall", ["Mean_PCP", "PCP", "P1", "P2", "P3"], 1000.0, "possible mm vs m conversion drift"),
        ("discharge", ["Discharge (CUMEC)", "Discharge", "discharge", "DisCUMEC"], 86400.0, "possible m3/s vs m3/day drift"),
        ("area", ["watershed_area", "basin_area", "area"], 1_000_000.0, "possible km2 vs m2 drift"),
    ]

    for feature_name, columns, expected_ratio, message in checks:
        for col in columns:
            ref = _median(reference_df, col)
            scen = _median(scenario_df, col)
            if ref is None or scen is None or ref == 0.0:
                continue
            ratio = abs(scen / ref)
            if ratio <= 0.0:
                continue
            log_distance = abs(np.log10(ratio) - np.log10(expected_ratio))
            if log_distance < 0.25:
                warnings_list.append(f"{feature_name.title()} ({col}): {message}.")
            break

    return warnings_list


def _feature_bounds_from_frame(feature_df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    if feature_df is None or feature_df.empty:
        return {}

    bounds: dict[str, tuple[float, float]] = {}
    numeric_df = feature_df.apply(pd.to_numeric, errors="coerce")
    for col in numeric_df.columns:
        series = numeric_df[col].replace([np.inf, -np.inf], np.nan).dropna()
        if series.empty:
            continue
        bounds[str(col)] = (float(series.min()), float(series.max()))
    return bounds


def _soft_clip_feature_bounds(
    feature_df: pd.DataFrame,
    feature_bounds: dict[str, tuple[float, float]] | None,
    alpha: float = 0.2,
) -> pd.DataFrame:
    if feature_df is None or feature_df.empty or not feature_bounds:
        return feature_df.copy() if feature_df is not None else pd.DataFrame()

    adjusted = feature_df.copy()
    normalized_columns = {_normalize_name(col): col for col in adjusted.columns}
    for feature_name, bounds in feature_bounds.items():
        if not isinstance(bounds, tuple) or len(bounds) != 2:
            continue
        lower, upper = bounds
        if not np.isfinite(lower) or not np.isfinite(upper) or upper < lower:
            continue

        resolved_col = normalized_columns.get(_normalize_name(feature_name))
        if resolved_col is None:
            continue

        series = pd.to_numeric(adjusted[resolved_col], errors="coerce")
        low_mask = series < lower
        high_mask = series > upper
        if low_mask.any():
            series.loc[low_mask] = lower + alpha * (series.loc[low_mask] - lower)
        if high_mask.any():
            series.loc[high_mask] = upper + alpha * (series.loc[high_mask] - upper)
        adjusted[resolved_col] = series.clip(lower=lower, upper=upper)

    return adjusted


def _apply_scenario_to_raw_frame(
    raw_df: pd.DataFrame,
    precip_change_pct: int,
    temp_shift_c: float,
    humidity_shift_pct: int,
    solar_change_pct: int,
    wind_change_pct: int,
) -> pd.DataFrame:
    scenario_df = raw_df.copy()

    precip_cols = [
        col
        for col in [
            _resolve_column_name(scenario_df, ["Mean_PCP"]),
            _resolve_column_name(scenario_df, ["PCP"]),
            _resolve_column_name(scenario_df, ["P1"]),
            _resolve_column_name(scenario_df, ["P2"]),
            _resolve_column_name(scenario_df, ["P3"]),
        ]
        if col is not None
    ]
    temp_cols = [
        col
        for col in [
            _resolve_column_name(scenario_df, ["Mean_Tmax"]),
            _resolve_column_name(scenario_df, ["Mean_Tmin"]),
            _resolve_column_name(scenario_df, ["TMAX"]),
            _resolve_column_name(scenario_df, ["TMIN"]),
        ]
        if col is not None
    ]
    humidity_cols = [col for col in [_resolve_column_name(scenario_df, ["rh", "RH"])] if col is not None]
    solar_cols = [col for col in [_resolve_column_name(scenario_df, ["solar"])] if col is not None]
    wind_cols = [col for col in [_resolve_column_name(scenario_df, ["wind", "wind "])] if col is not None]

    precip_factor = 1.0 + (precip_change_pct / 100.0)
    solar_factor = 1.0 + (solar_change_pct / 100.0)
    wind_factor = 1.0 + (wind_change_pct / 100.0)

    for col in precip_cols:
        scenario_df[col] = pd.to_numeric(scenario_df[col], errors="coerce").fillna(0.0) * precip_factor
        scenario_df[col] = scenario_df[col].clip(lower=0.0)
    for col in temp_cols:
        scenario_df[col] = pd.to_numeric(scenario_df[col], errors="coerce").fillna(0.0) + temp_shift_c
    for col in humidity_cols:
        hum_series = pd.to_numeric(scenario_df[col], errors="coerce").fillna(0.0)
        if hum_series.max() <= 1.5:
            hum_series = hum_series + (humidity_shift_pct / 100.0)
            scenario_df[col] = hum_series.clip(lower=0.0, upper=1.0)
        else:
            hum_series = hum_series + humidity_shift_pct
            scenario_df[col] = hum_series.clip(lower=0.0, upper=100.0)
    for col in solar_cols:
        scenario_df[col] = pd.to_numeric(scenario_df[col], errors="coerce").fillna(0.0) * solar_factor
        scenario_df[col] = scenario_df[col].clip(lower=0.0)
    for col in wind_cols:
        scenario_df[col] = pd.to_numeric(scenario_df[col], errors="coerce").fillna(0.0) * wind_factor
        scenario_df[col] = scenario_df[col].clip(lower=0.0)

    return scenario_df


def _simulate_bounded_runoff(
    raw_df: pd.DataFrame,
    model_choice: str,
    anchor_horizon: int = 14,
    feature_bounds: dict[str, tuple[float, float]] | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    prepared = prepare_features(raw_df)
    aligned = align_features(prepared, feature_list)
    aligned = aligned.replace([np.inf, -np.inf], np.nan)
    aligned = aligned.apply(pd.to_numeric, errors="coerce").ffill().bfill().fillna(0.0)
    if aligned.empty:
        return aligned, np.array([])

    predictions = np.asarray(
        _predict_runoff_with_loaded_model_units(aligned, model_choice=model_choice),
        dtype=float,
    ).reshape(-1)
    return aligned.reset_index(drop=True), predictions


def _section_header(title: str, subtitle: str) -> None:
    st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)
    st.caption(subtitle)


def _build_report_data(
    default_frame: pd.DataFrame,
    csv_frame: pd.DataFrame,
    discharge_column: str | None,
    selected_model_name: str,
    prediction_value: float | None,
    risk_value: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, str]] = []
    summary_rows.append({"Metric": "Generated At", "Value": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
    summary_rows.append({"Metric": "Selected Model", "Value": selected_model_name})
    summary_rows.append({"Metric": "XLSX Records", "Value": str(len(default_frame))})
    summary_rows.append({"Metric": "CSV Records", "Value": str(len(csv_frame))})
    summary_rows.append({"Metric": "Tracked Features", "Value": str(len(feature_list))})
    summary_rows.append({"Metric": "RF Loaded", "Value": "Yes" if rf_ready else "No"})
    summary_rows.append({"Metric": "LSTM Loaded", "Value": "Yes" if lstm_ready else "No"})
    summary_rows.append(
        {
            "Metric": f"Latest Prediction ({discharge_unit_label})",
            "Value": f"{prediction_value:,.2f}" if prediction_value is not None else "Not generated",
        }
    )
    summary_rows.append({"Metric": "Latest Risk Label", "Value": risk_value if prediction_value is not None else "Not assessed"})

    if not default_frame.empty:
        date_col = detect_datetime_column(default_frame)
        if date_col and date_col in default_frame.columns:
            dt_series = pd.to_datetime(default_frame[date_col], errors="coerce").dropna()
            if not dt_series.empty:
                summary_rows.append({"Metric": "Historical Range Start", "Value": str(dt_series.min().date())})
                summary_rows.append({"Metric": "Historical Range End", "Value": str(dt_series.max().date())})

    if not csv_frame.empty:
        climate_cols = [col for col in ["Mean_PCP", "Mean_Tmax", "Mean_Tmin", "rh", "solar", "wind"] if col in csv_frame.columns]
        for col in climate_cols:
            col_series = pd.to_numeric(csv_frame[col], errors="coerce").dropna()
            if not col_series.empty:
                summary_rows.append({"Metric": f"{col} Mean", "Value": f"{float(col_series.mean()):.3f}"})

    compare_df = pd.DataFrame(columns=["Model", "RMSE", "R2", "NSE"])
    if not default_frame.empty and discharge_column and discharge_column in default_frame.columns:
        observed_vals = pd.to_numeric(default_frame[discharge_column], errors="coerce").dropna().to_numpy()
        observed_vals = np.asarray(observed_vals, dtype=float) * discharge_factor
        if observed_vals.size > 5:
            model_frame = align_features(prepare_features(default_frame), feature_list)
            selected_preds = np.asarray(_predict_runoff_with_loaded_model_units(model_frame, model_choice=selected_model_name))
            selected_preds = selected_preds * _prediction_display_factor(selected_model_name)
            selected_horizon = _model_forecast_horizon_days(selected_model_name)
            observed_aligned, selected_preds = _align_metric_inputs_with_horizon(
                observed_vals,
                selected_preds,
                horizon_days=selected_horizon,
            )
            if observed_aligned.size > 5:
                summary_rows.append({"Metric": "Selected Model RMSE", "Value": f"{rmse(observed_aligned, selected_preds):.3f}"})
                summary_rows.append({"Metric": "Selected Model R2", "Value": f"{r2(observed_aligned, selected_preds):.3f}"})
                summary_rows.append({"Metric": "Selected Model NSE", "Value": f"{nse(observed_aligned, selected_preds):.3f}"})

                comparison_rows: list[dict[str, float | str]] = []
                for candidate_model, is_ready in [
                    ("Random Forest", rf_ready),
                    ("LSTM", lstm_ready),
                ]:
                    if not is_ready:
                        continue
                    candidate_preds = np.asarray(
                        _predict_runoff_with_loaded_model_units(model_frame, model_choice=candidate_model)
                    ) * _prediction_display_factor(candidate_model)
                    candidate_horizon = _model_forecast_horizon_days(candidate_model)
                    observed_candidate, candidate_preds = _align_metric_inputs_with_horizon(
                        observed_vals,
                        candidate_preds,
                        horizon_days=candidate_horizon,
                    )
                    if observed_candidate.size <= 5:
                        continue
                    comparison_rows.append(
                        {
                            "Model": candidate_model,
                            "RMSE": rmse(observed_candidate, candidate_preds),
                            "R2": r2(observed_candidate, candidate_preds),
                            "NSE": nse(observed_candidate, candidate_preds),
                        }
                    )

                if comparison_rows:
                    compare_df = pd.DataFrame(comparison_rows)

    return pd.DataFrame(summary_rows), compare_df


def _build_pdf_report(summary_df: pd.DataFrame, compare_df: pd.DataFrame) -> bytes | None:
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
    except Exception:
        return None

    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    y = height - 40
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(40, y, "Runoff Forecast Summary Report")
    y -= 24
    pdf.setFont("Helvetica", 10)
    for _, row in summary_df.iterrows():
        line = f"{row['Metric']}: {row['Value']}"
        pdf.drawString(40, y, line[:115])
        y -= 14
        if y < 80:
            pdf.showPage()
            y = height - 40
            pdf.setFont("Helvetica", 10)

    if not compare_df.empty:
        y -= 10
        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(40, y, "Model Comparison")
        y -= 18
        pdf.setFont("Helvetica", 10)
        for _, row in compare_df.iterrows():
            line = f"{row['Model']}: RMSE={row['RMSE']:.3f}, R2={row['R2']:.3f}, NSE={row['NSE']:.3f}"
            pdf.drawString(40, y, line[:115])
            y -= 14
            if y < 80:
                pdf.showPage()
                y = height - 40
                pdf.setFont("Helvetica", 10)

    pdf.save()
    return buffer.getvalue()


show_main_inputs = selected_nav == "Single Prediction"
if show_main_inputs:
    _section_header("Forecast Input Console", "Configure meteorological parameters on the main workspace.")
    st.caption(
        "Defaults come from the historical dataset. Discharge-derived fields are normalized to CUMEC before they are shown here."
    )
    control_col_1, control_col_2 = st.columns(2)
    with control_col_1:
        selected_date = st.date_input("Forecast Date", value=selected_date, key="main_forecast_date")
    with control_col_2:
        selected_model = st.selectbox(
            "Model Selection",
            options=model_options,
            index=model_options.index(selected_model) if selected_model in model_options else 0,
            key="main_forecast_model",
        )

    st.session_state["forecast_date"] = selected_date
    st.session_state["forecast_model"] = selected_model

    group_order = [
        "Rainfall Inputs",
        "Temperature",
        "Atmosphere",
        "Hydrology Memory",
        "Seasonality (Auto)",
        "Other Inputs",
    ]
    grouped_features = {group_name: [] for group_name in group_order}
    for feature_name in feature_list:
        grouped_features.setdefault(_feature_group(feature_name), []).append(feature_name)

    input_values: dict[str, float] = {}
    predict_clicked = False
    reset_clicked = False
    with st.form("main_prediction_form", clear_on_submit=False):
        for group_name in group_order:
            features_in_group = grouped_features.get(group_name, [])
            if not features_in_group:
                continue

            with st.expander(group_name, expanded=group_name in {"Rainfall Inputs", "Hydrology Memory"}):
                cols = st.columns(2)
                for index, feature_name in enumerate(features_in_group):
                    default_value = _feature_default_value(feature_name, selected_date, default_data, discharge_col)
                    is_auto = feature_name.lower() in {"season_sin", "season_cos"}
                    input_format = "%.4f" if "season_" in feature_name.lower() else "%.3f"
                    with cols[index % 2]:
                        if is_auto:
                            st.markdown(f"**{_feature_label(feature_name)}**")
                            st.caption(f"{default_value:.4f} (auto from forecast date)")
                            input_values[feature_name] = float(default_value)
                        else:
                            state_key = f"main_feature_{feature_name}"
                            if state_key not in st.session_state:
                                st.session_state[state_key] = float(default_value)
                            widget_value = st.number_input(
                                label=_feature_label(feature_name),
                                step=_input_step(default_value),
                                format=input_format,
                                help=_feature_help_text(feature_name),
                                key=state_key,
                            )
                            input_values[feature_name] = float(widget_value)

        action_col_1, action_col_2, _ = st.columns([1, 1, 4])
        with action_col_1:
            predict_clicked = st.form_submit_button("Run Prediction", width="stretch")
        with action_col_2:
            reset_clicked = st.form_submit_button("Reset Defaults", width="stretch")

    discharge_input_alerts = _discharge_input_alerts(input_values)
    if discharge_input_alerts:
        preview = "; ".join(discharge_input_alerts[:3])
        if len(discharge_input_alerts) > 3:
            preview += "; ..."
        st.warning(
            (
                f"One or more discharge inputs exceed the realistic screening threshold of "
                f"{REALISTIC_DISCHARGE_MAX_CUMEC:,.0f} {discharge_unit_label}. Review the entered values: {preview}"
            )
        )

    if reset_clicked:
        for feature_name in feature_list:
            if feature_name.lower() in {"season_sin", "season_cos"}:
                continue
            state_key = f"main_feature_{feature_name}"
            st.session_state[state_key] = float(
                _feature_default_value(feature_name, selected_date, default_data, discharge_col)
            )
        st.rerun()

    if predict_clicked:
        normalized_input_values, unit_adjustments = _normalize_user_input_units(input_values)
        input_row_df = pd.DataFrame([normalized_input_values])
        aligned_input = align_features(input_row_df, feature_list)
        reference_feature_frame = (
            align_features(prepare_features(default_data), feature_list)
            if not default_data.empty
            else pd.DataFrame()
        )
        reference_ranges = _build_feature_reference_ranges(reference_feature_frame)
        ood_df = _input_ood_report(aligned_input, reference_ranges)

        if unit_adjustments:
            st.info("Input units were auto-aligned to match model training units.")
            for adjustment in unit_adjustments:
                st.caption(f"- {adjustment}")

        if not ood_df.empty:
            st.warning(
                "One or more inputs are outside the model's typical training distribution (P01-P99). "
                "Prediction confidence may be lower."
            )
            with st.expander("Input Distribution Diagnostics", expanded=False):
                st.dataframe(ood_df, width="stretch", hide_index=True)

        prediction_arr = _predict_runoff_with_loaded_model_units(aligned_input, model_choice=selected_model)
        single_prediction = float(prediction_arr[0]) if len(prediction_arr) else 0.0

        historical_values = default_data[discharge_col].dropna().to_numpy() if discharge_col else None
        historical_display_values = (
            np.asarray(historical_values, dtype=float) * discharge_factor
            if historical_values is not None
            else None
        )
        risk_label, risk_class, risk_icon = get_risk_label(
            _to_display_discharge(single_prediction, selected_model) or 0.0,
            historical_display_values,
        )

        st.session_state["single_prediction"] = single_prediction
        st.session_state["risk_label"] = risk_label
        st.session_state["risk_class"] = risk_class
        st.session_state["risk_icon"] = risk_icon

if selected_nav == "Dashboard":
    _section_header("Historical Trends", "Historical views from both Kasol.xlsx and Kasol.csv datasets.")
    k1, k2, k3 = st.columns(3)
    k1.markdown(create_kpi_card("XLSX Records", str(len(default_data))), unsafe_allow_html=True)
    k2.markdown(create_kpi_card("CSV Records", str(len(csv_data))), unsafe_allow_html=True)
    k3.markdown(create_kpi_card("Active Model", selected_model), unsafe_allow_html=True)

    st.markdown("<div style='height: 0.6rem;'></div>", unsafe_allow_html=True)
    chart_height = 360
    dashboard_discharge_data, dashboard_discharge_col = _load_dashboard_discharge_source()
    if dashboard_discharge_col is None or dashboard_discharge_data.empty:
        with st.expander("Discharge Chart Diagnostics", expanded=False):
            valid_discharge_points = 0
            if dashboard_discharge_col and dashboard_discharge_col in dashboard_discharge_data.columns:
                valid_discharge_points = int(
                    _coerce_numeric_values(dashboard_discharge_data[dashboard_discharge_col]).dropna().shape[0]
                )
            diagnostic_df = pd.DataFrame(
                [
                    {"Metric": "Rows", "Value": str(len(dashboard_discharge_data))},
                    {"Metric": "Columns", "Value": ", ".join(map(str, dashboard_discharge_data.columns[:12])) if not dashboard_discharge_data.empty else "(none)"},
                    {"Metric": "Detected Discharge Column", "Value": str(dashboard_discharge_col)},
                    {"Metric": "Valid Numeric Discharge Values", "Value": str(valid_discharge_points)},
                ]
            )
            st.dataframe(diagnostic_df, width="stretch", hide_index=True)
    top_left, top_right = st.columns(2)
    with top_left:
        with st.container(border=True):
            st.markdown("**Historical Discharge**")
            hist_fig = historical_discharge_chart(dashboard_discharge_data, dashboard_discharge_col)
            hist_fig.update_layout(title_text="", height=chart_height, margin={"l": 16, "r": 12, "t": 10, "b": 16})
            st.plotly_chart(hist_fig, width="stretch")

    with top_right:
        with st.container(border=True):
            st.markdown("**Climate Variables (Mean PCP, Mean Tmax, Mean Tmin)**")
            if not csv_data.empty:
                csv_frame = csv_data.copy()
                if "Date" in csv_frame.columns:
                    csv_frame["Date"] = pd.to_datetime(csv_frame["Date"], errors="coerce", dayfirst=True)
                    csv_frame = csv_frame.dropna(subset=["Date"]).set_index("Date")

                climate_cols = [col for col in ["Mean_PCP", "Mean_Tmax", "Mean_Tmin"] if col in csv_frame.columns]
                if climate_cols:
                    climate_frame = _downsample_df(csv_frame[climate_cols], max_points=1200)
                    climate_fig = _timeseries_plot(climate_frame, y_title="Climate", height=chart_height)
                    st.plotly_chart(climate_fig, width="stretch")
                else:
                    st.info("Kasol.csv does not contain the expected climate columns.")
            else:
                st.info("Kasol.csv is unavailable or empty.")

    bottom_left, bottom_right = st.columns(2)
    with bottom_left:
        with st.container(border=True):
            st.markdown("**Discharge Trend Smoothing (7/30 day)**")
            if dashboard_discharge_col and not dashboard_discharge_data.empty:
                discharge_series = (
                    _coerce_numeric_values(dashboard_discharge_data[dashboard_discharge_col])
                    .dropna()
                    .reset_index(drop=True)
                )
                if not discharge_series.empty:
                    trend_df = pd.DataFrame(
                        {
                            "Observed": discharge_series,
                            "Rolling Mean (7)": discharge_series.rolling(7, min_periods=1).mean(),
                            "Rolling Mean (30)": discharge_series.rolling(30, min_periods=1).mean(),
                        }
                    )
                    trend_df = _downsample_df(trend_df, max_points=1200)
                    trend_fig = _timeseries_plot(trend_df, y_title="Discharge", height=chart_height)
                    st.plotly_chart(trend_fig, width="stretch")
                else:
                    st.info("Discharge column exists but contains no valid numeric values.")
            else:
                st.info("No valid discharge data is currently available for the trend chart.")

    with bottom_right:
        with st.container(border=True):
            st.markdown("**Atmospheric Trends (RH, Solar, Wind)**")
            if not csv_data.empty:
                csv_atm = csv_data.copy()
                if "Date" in csv_atm.columns:
                    csv_atm["Date"] = pd.to_datetime(csv_atm["Date"], errors="coerce", dayfirst=True)
                    csv_atm = csv_atm.dropna(subset=["Date"]).set_index("Date")
                atm_cols = [col for col in ["rh", "solar", "wind"] if col in csv_atm.columns]
                if atm_cols:
                    atm_frame = _downsample_df(csv_atm[atm_cols], max_points=1200)
                    atm_fig = _timeseries_plot(atm_frame, y_title="Atmosphere", height=chart_height)
                    st.plotly_chart(atm_fig, width="stretch")
                else:
                    st.info("Kasol.csv does not contain rh/solar/wind columns.")
            else:
                st.info("Kasol.csv is unavailable or empty.")

    if selected_model == "Random Forest":
        importance_df = get_feature_importance(artifacts, fallback_features=feature_list)
        if not importance_df.empty:
            st.plotly_chart(feature_importance_chart(importance_df), width="stretch")
        else:
            st.info("Feature importance unavailable. Add a valid `rf_model.pkl` and `features.pkl`.")

elif selected_nav == "Single Prediction":
    _section_header("Single Forecast", "Generate a 3-day ahead discharge prediction from the main input console.")
    if single_prediction is None:
        st.info("Use the main input console above and click Run Prediction to generate a forecast.")
    else:
        display_prediction = _to_display_discharge(single_prediction, selected_model) or 0.0
        card_html = (
            "<div style='font-size:0.9rem; color: var(--text-secondary);'>⚠️ Predicted 3-Day Ahead Discharge</div>"
            f"<div class='prediction-value'>{display_prediction:,.2f} {discharge_unit_label}</div>"
            f"<div style='margin-top:0.5rem;'><span class='risk-badge {risk_class}'>{risk_icon} {risk_label}</span></div>"
        )
        st.markdown(create_glass_card("Forecast Output", card_html, animate=True), unsafe_allow_html=True)

elif selected_nav == "Batch Prediction":
    _section_header("Batch Prediction", "Upload a CSV for bulk runoff forecasting and export results.")
    template_df = pd.DataFrame([{feature: 0.0 for feature in feature_list}])
    st.download_button(
        label="Download CSV Template",
        data=template_df.to_csv(index=False).encode("utf-8"),
        file_name="runoff_batch_template.csv",
        mime="text/csv",
    )

    batch_file = st.file_uploader("Drag and drop your CSV here", type=["csv"], key="batch_upload")

    if batch_file is not None:
        batch_df = pd.read_csv(batch_file)
        batch_features = align_features(batch_df, feature_list) if feature_list else batch_df.select_dtypes(include=["number"])
        batch_predictions = _predict_runoff_with_loaded_model_units(batch_features, model_choice=selected_model)
        batch_output = batch_df.copy()
        batch_output["predicted_discharge_cumec"] = batch_predictions
        if "predicted_discharge_cumec" in batch_output.columns:
            output_factor = _prediction_display_factor(selected_model)
            batch_output["predicted_discharge_cumec"] = (
                pd.to_numeric(batch_output["predicted_discharge_cumec"], errors="coerce") * output_factor
            )

        st.markdown(create_glass_card("Batch Preview", "<div style='color: var(--text-secondary);'>Previewing first 25 rows.</div>"), unsafe_allow_html=True)
        st.dataframe(batch_output.head(25), width="stretch", height=320)

        st.download_button(
            label="Download Batch Predictions",
            data=batch_output.to_csv(index=False).encode("utf-8"),
            file_name="batch_runoff_predictions.csv",
            mime="text/csv",
        )

elif selected_nav == "Model Analytics":
    _section_header("Model Analytics", "Performance metrics and model comparison insights.")
    metric_col_1, metric_col_2, metric_col_3 = st.columns(3)

    observed_vals = None
    observed_vals_raw = None
    eval_preds = None
    model_frame = None
    if not default_data.empty and discharge_col:
        observed_vals_raw = pd.to_numeric(default_data[discharge_col], errors="coerce").dropna().to_numpy()
        observed_vals_raw = np.asarray(observed_vals_raw, dtype=float) * discharge_factor
        if observed_vals_raw.size > 5:
            model_frame = align_features(prepare_features(default_data), feature_list)
            eval_preds = _predict_runoff_with_loaded_model_units(model_frame, model_choice=selected_model)
            eval_preds = np.asarray(eval_preds, dtype=float) * _prediction_display_factor(selected_model)
            selected_horizon = _model_forecast_horizon_days(selected_model)
            observed_vals, eval_preds = _align_metric_inputs_with_horizon(
                observed_vals_raw,
                eval_preds,
                horizon_days=selected_horizon,
            )

    if observed_vals is not None and eval_preds is not None and observed_vals.size > 5:
        rmse_val = rmse(observed_vals, eval_preds)
        r2_val = r2(observed_vals, eval_preds)
        nse_val = nse(observed_vals, eval_preds)
        spark_source = observed_vals[-20:] if observed_vals.size >= 20 else observed_vals

        metric_col_1.markdown(create_kpi_card("RMSE", f"{rmse_val:.3f}"), unsafe_allow_html=True)
        metric_col_1.plotly_chart(sparkline_chart(spark_source, color="#45c0ff"), width="stretch")

        metric_col_2.markdown(create_kpi_card("R²", f"{r2_val:.3f}"), unsafe_allow_html=True)
        metric_col_2.plotly_chart(sparkline_chart(spark_source, color="#7c6cff"), width="stretch")

        metric_col_3.markdown(create_kpi_card("NSE", f"{nse_val:.3f}"), unsafe_allow_html=True)
        metric_col_3.plotly_chart(sparkline_chart(spark_source, color="#ffb35a"), width="stretch")
    else:
        st.info("Upload historical data with observed discharge to populate model analytics.")

    st.markdown("<div style='height: 0.6rem;'></div>", unsafe_allow_html=True)
    st.markdown(create_glass_card("Hydrological Metrics", """
        <ul style='margin:0; color: var(--text-secondary);'>
            <li><strong>RMSE</strong> measures typical forecast error magnitude.</li>
            <li><strong>R²</strong> indicates explanatory power versus variance.</li>
            <li><strong>NSE</strong> benchmarks predictive skill against mean discharge.</li>
        </ul>
    """), unsafe_allow_html=True)

    rf_loaded = artifacts.get("rf_model") is not None
    lstm_loaded = artifacts.get("lstm_model") is not None
    if observed_vals_raw is not None and observed_vals_raw.size > 5 and model_frame is not None:
        comparison_rows: list[dict[str, float | str]] = []
        for candidate_model, is_ready in [
            ("Random Forest", rf_loaded),
            ("LSTM", lstm_loaded),
        ]:
            if not is_ready:
                continue
            candidate_preds = np.asarray(
                _predict_runoff_with_loaded_model_units(model_frame, model_choice=candidate_model)
            ) * _prediction_display_factor(candidate_model)
            candidate_horizon = _model_forecast_horizon_days(candidate_model)
            observed_candidate, candidate_preds = _align_metric_inputs_with_horizon(
                observed_vals_raw,
                candidate_preds,
                horizon_days=candidate_horizon,
            )
            if observed_candidate.size <= 5:
                continue
            comparison_rows.append(
                {
                    "Model": candidate_model,
                    "RMSE": rmse(observed_candidate, candidate_preds),
                    "R²": r2(observed_candidate, candidate_preds),
                    "NSE": nse(observed_candidate, candidate_preds),
                }
            )

        if comparison_rows:
            compare_df = pd.DataFrame(comparison_rows)
            st.dataframe(compare_df, width="stretch")

elif selected_nav == "Scenario Simulator":
    _section_header("Scenario Simulator", "Apply climate perturbations and compare baseline vs scenario runoff.")
    if default_data.empty:
        st.info("Historical data is unavailable. Add data to run scenario simulation.")
    else:
        max_sim_window = max(30, min(len(default_data), 5000))
        default_sim_window = min(180, max_sim_window)
        control_col_1, control_col_2, control_col_3 = st.columns(3)
        with control_col_1:
            scenario_model = st.selectbox(
                "Scenario Model",
                options=model_options,
                index=model_options.index(selected_model) if selected_model in model_options else 0,
                key="scenario_model",
            )
            if scenario_model == "LSTM" and max_sim_window > LSTM_SCENARIO_MAX_DAYS:
                st.caption(
                    f"LSTM scenario runs are capped at {LSTM_SCENARIO_MAX_DAYS} days for responsiveness. "
                    "Use Random Forest or BEST for longer windows."
                )
            precip_change_pct = st.slider("Precipitation Change (%)", min_value=-50, max_value=200, value=20, step=5)
            temp_shift_c = st.slider("Temperature Shift (deg C)", min_value=-5.0, max_value=5.0, value=0.5, step=0.1)
        with control_col_2:
            humidity_shift_pct = st.slider("Humidity Change (%)", min_value=-30, max_value=30, value=0, step=1)
            solar_change_pct = st.slider("Solar Radiation Change (%)", min_value=-40, max_value=40, value=0, step=2)
        with control_col_3:
            wind_change_pct = st.slider("Wind Speed Change (%)", min_value=-40, max_value=40, value=0, step=2)
            effective_max_window = (
                min(max_sim_window, LSTM_SCENARIO_MAX_DAYS)
                if scenario_model == "LSTM"
                else max_sim_window
            )
            effective_default_window = min(default_sim_window, effective_max_window)
            sim_window = st.slider(
                "Simulation Window (days)",
                min_value=30,
                max_value=effective_max_window,
                value=effective_default_window,
                step=30,
            )
        run_scenario_simulation = st.button("Run Scenario Simulation", type="primary", width="stretch")

        if not run_scenario_simulation:
            st.info("Adjust the scenario inputs and click `Run Scenario Simulation` to generate the comparison chart.")
            st.stop()

        progress_text = st.empty()
        progress_bar = st.progress(0)

        progress_text.info("Generating scenario results: preparing climate inputs...")
        baseline_raw = _sanitize_simulation_frame(default_data.tail(sim_window).reset_index(drop=True).copy())
        scenario_raw = _apply_scenario_to_raw_frame(
            baseline_raw,
            precip_change_pct=precip_change_pct,
            temp_shift_c=temp_shift_c,
            humidity_shift_pct=humidity_shift_pct,
            solar_change_pct=solar_change_pct,
            wind_change_pct=wind_change_pct,
        )
        scenario_raw = _sanitize_simulation_frame(scenario_raw)
        baseline_feature_frame = align_features(prepare_features(baseline_raw), feature_list)
        training_feature_bounds = _feature_bounds_from_frame(baseline_feature_frame)
        scenario_feature_frame = align_features(prepare_features(scenario_raw), feature_list)
        drift_df = _scenario_range_drift_report(baseline_raw, scenario_raw)
        if not drift_df.empty:
            st.warning(
                "Scenario inputs are outside the historical training distribution for one or more variables. "
                "Predictions are stabilized with clipping to reduce numerical blow-ups."
            )
            with st.expander("Out-of-Distribution Feature Ranges", expanded=False):
                st.dataframe(drift_df, width="stretch", hide_index=True)
        feature_drift_df = _scenario_range_drift_report(baseline_feature_frame, scenario_feature_frame)
        if not feature_drift_df.empty:
            with st.expander("Feature-Space Drift vs Training Range", expanded=False):
                st.dataframe(feature_drift_df, width="stretch", hide_index=True)
        unit_warnings = _unit_consistency_warnings(baseline_raw, scenario_raw)
        if unit_warnings:
            st.warning("Potential unit consistency issues detected:")
            for item in unit_warnings:
                st.caption(f"- {item}")
        scenario_anchor_horizon = min(21, max(7, sim_window // 12))
        progress_bar.progress(20)

        progress_text.info("Generating scenario results: running baseline simulation...")
        _, baseline_preds = _simulate_bounded_runoff(
            baseline_raw,
            model_choice=scenario_model,
            anchor_horizon=scenario_anchor_horizon,
            feature_bounds=training_feature_bounds,
        )
        progress_bar.progress(50)

        progress_text.info("Generating scenario results: running scenario simulation...")
        _, scenario_preds = _simulate_bounded_runoff(
            scenario_raw,
            model_choice=scenario_model,
            anchor_horizon=scenario_anchor_horizon,
            feature_bounds=training_feature_bounds,
        )
        progress_bar.progress(75)

        progress_text.info("Generating scenario results: finalizing metrics and charts...")
        compare_len = min(len(baseline_preds), len(scenario_preds))
        if compare_len == 0:
            progress_bar.empty()
            progress_text.warning("Scenario simulation could not generate predictions for the selected data window.")
            st.stop()
        baseline_preds = baseline_preds[:compare_len]
        scenario_preds = scenario_preds[:compare_len]
        delta_preds = scenario_preds - baseline_preds
        scenario_factor = _prediction_display_factor(scenario_model)
        baseline_preds_display = baseline_preds * scenario_factor
        scenario_preds_display = scenario_preds * scenario_factor
        delta_preds_display = delta_preds * scenario_factor
        delta_pct_series = _percent_change_series(baseline_preds_display, scenario_preds_display)

        avg_baseline = float(np.mean(baseline_preds_display)) if compare_len else 0.0
        avg_scenario = float(np.mean(scenario_preds_display)) if compare_len else 0.0
        delta_abs = avg_scenario - avg_baseline
        delta_pct = (delta_abs / avg_baseline * 100.0) if avg_baseline != 0 else 0.0
        higher_flow_days = int(np.sum(delta_preds_display > 0))
        lower_flow_days = int(np.sum(delta_preds_display < 0))
        peak_increase = float(np.max(delta_preds_display)) if compare_len else 0.0
        peak_decrease = float(np.min(delta_preds_display)) if compare_len else 0.0

        progress_bar.progress(100)
        progress_text.success("Scenario results are ready.")

        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.markdown(create_kpi_card("Baseline Mean", f"{avg_baseline:,.2f}"), unsafe_allow_html=True)
        k2.markdown(create_kpi_card("Scenario Mean", f"{avg_scenario:,.2f}"), unsafe_allow_html=True)
        k3.markdown(create_kpi_card("Delta", f"{delta_abs:,.2f}"), unsafe_allow_html=True)
        k4.markdown(create_kpi_card("Delta %", f"{delta_pct:,.2f}%"), unsafe_allow_html=True)
        k5.markdown(create_kpi_card("Higher-Flow Days", str(higher_flow_days)), unsafe_allow_html=True)
        k6.markdown(create_kpi_card("Lower-Flow Days", str(lower_flow_days)), unsafe_allow_html=True)

        sim_df = pd.DataFrame({"Baseline": baseline_preds_display, "Scenario": scenario_preds_display})
        delta_df = pd.DataFrame({"Delta": delta_preds_display})
        pct_df = pd.DataFrame({"Percent Change": delta_pct_series})
        if not baseline_raw.empty:
            date_col = detect_datetime_column(baseline_raw)
            if date_col and date_col in baseline_raw.columns:
                dt_series = pd.to_datetime(baseline_raw[date_col], errors="coerce").dropna()
                if len(dt_series) >= compare_len:
                    sim_df.index = dt_series.iloc[:compare_len].to_numpy()
                    delta_df.index = sim_df.index
                    pct_df.index = sim_df.index

        sim_df = _downsample_df(sim_df, max_points=1200)
        delta_df = _downsample_df(delta_df, max_points=1200)
        pct_df = _downsample_df(pct_df, max_points=1200)
        with st.container(border=True):
            st.markdown("**Baseline vs Scenario Runoff**")
            sim_fig = _timeseries_plot(sim_df, y_title="Discharge", height=380)
            st.plotly_chart(sim_fig, width="stretch")

        delta_col, pct_col = st.columns(2)
        with delta_col:
            with st.container(border=True):
                st.markdown("**Scenario Delta**")
                delta_fig = _timeseries_plot(delta_df, y_title="Scenario - Baseline", height=300)
                st.plotly_chart(delta_fig, width="stretch")
        with pct_col:
            with st.container(border=True):
                st.markdown("**Scenario Percent Change**")
                pct_fig = _timeseries_plot(pct_df, y_title="Percent", height=300)
                st.plotly_chart(pct_fig, width="stretch")

        with st.container(border=True):
            st.markdown("**Scenario Response Summary**")
            scenario_summary = pd.DataFrame(
                [
                    {"Parameter": "Scenario Model", "Applied Change": scenario_model},
                    {"Parameter": "Simulation Window", "Applied Change": f"{sim_window} days"},
                    {"Parameter": "Precipitation", "Applied Change": f"{precip_change_pct:+d}%"},
                    {"Parameter": "Temperature", "Applied Change": f"{temp_shift_c:+.1f} deg C"},
                    {"Parameter": "Humidity", "Applied Change": f"{humidity_shift_pct:+d}%"},
                    {"Parameter": "Solar Radiation", "Applied Change": f"{solar_change_pct:+d}%"},
                    {"Parameter": "Wind Speed", "Applied Change": f"{wind_change_pct:+d}%"},
                    {"Parameter": "Runoff Memory", "Applied Change": f"Bounded recursive discharge (re-anchored every {scenario_anchor_horizon} days)"},
                    {"Parameter": "Peak Increase", "Applied Change": f"{peak_increase:,.2f} {discharge_unit_label}"},
                    {"Parameter": "Peak Decrease", "Applied Change": f"{peak_decrease:,.2f} {discharge_unit_label}"},
                ]
            )
            st.dataframe(scenario_summary, width="stretch", hide_index=True)

        if discharge_col and discharge_col in default_data.columns:
            observed_vals = pd.to_numeric(default_data[discharge_col], errors="coerce").dropna().to_numpy()
            if observed_vals.size > 20:
                threshold_raw = float(np.quantile(observed_vals, 0.90))
                threshold_display = threshold_raw * discharge_factor
                baseline_exceed = int(np.sum(baseline_preds_display > threshold_display))
                scenario_exceed = int(np.sum(scenario_preds_display > threshold_display))
                st.markdown(
                    create_glass_card(
                        "Extreme Flow Impact",
                        f"""
                        <p style='color: var(--text-secondary); margin:0.1rem 0;'>90th percentile threshold: <strong>{threshold_display:,.2f}</strong> {discharge_unit_label}</p>
                        <p style='color: var(--text-secondary); margin:0.1rem 0;'>Baseline exceedance count: <strong>{baseline_exceed}</strong></p>
                        <p style='color: var(--text-secondary); margin:0.1rem 0;'>Scenario exceedance count: <strong>{scenario_exceed}</strong></p>
                        """,
                    ),
                    unsafe_allow_html=True,
                )

elif selected_nav == "Model Status":
    _section_header("Model Status", "Artifact readiness, feature schema, and deployment health.")
    best_model_artifact_path = artifacts.get("best_model_path")
    if not isinstance(best_model_artifact_path, Path):
        best_model_artifact_path = base_dir / "backend" / "models" / "best_extra_trees_log1p.joblib"
    artifact_specs = [
        (best_model_label, best_model_artifact_path),
        ("Random Forest", model_dir / "rf_model.pkl"),
        ("LSTM", model_dir / "lstm_model.h5"),
        ("Feature Schema", model_dir / "features.pkl"),
        ("Scaler", model_dir / "scaler.pkl"),
    ]
    status_rows = []
    for artifact_name, artifact_path in artifact_specs:
        exists = artifact_path.exists() and artifact_path.stat().st_size > 0
        size_kb = round(artifact_path.stat().st_size / 1024, 2) if exists else 0.0
        modified = pd.to_datetime(artifact_path.stat().st_mtime, unit="s") if exists else pd.NaT
        status_rows.append(
            {
                "Artifact": artifact_name,
                "Status": "Ready" if exists else "Missing",
                "Size (KB)": size_kb,
                "Last Modified": modified,
            }
        )

    status_df = pd.DataFrame(status_rows)
    st.dataframe(status_df, width="stretch", hide_index=True)

    meta_col_1, meta_col_2, meta_col_3, meta_col_4 = st.columns(4)
    meta_col_1.markdown(create_kpi_card("Features", str(len(feature_list))), unsafe_allow_html=True)
    meta_col_2.markdown(create_kpi_card("Best Loaded", "Yes" if best_model_ready else "No"), unsafe_allow_html=True)
    meta_col_3.markdown(create_kpi_card("RF Loaded", "Yes" if rf_ready else "No"), unsafe_allow_html=True)
    meta_col_4.markdown(create_kpi_card("LSTM Loaded", "Yes" if lstm_ready else "No"), unsafe_allow_html=True)

    if discharge_report:
        unit_audit_df = pd.DataFrame(
            [
                {"Metric": "Detected Column", "Value": str(discharge_report.get("column_name", "n/a"))},
                {"Metric": "Assumed Source Unit", "Value": str(discharge_report.get("assumed_unit", "unknown"))},
                {
                    "Metric": "Factor to CUMEC",
                    "Value": f"{float(discharge_report.get('applied_factor_to_cumec', 1.0) or 1.0):.8f}",
                },
                {"Metric": "Raw Median", "Value": f"{float(discharge_report.get('raw_median', 0.0) or 0.0):,.2f}"},
                {
                    "Metric": "Normalized Median",
                    "Value": f"{float(discharge_report.get('converted_median', 0.0) or 0.0):,.2f} {discharge_unit_label}",
                },
            ]
        )
        with st.expander("Discharge Unit Audit", expanded=False):
            st.dataframe(unit_audit_df, width="stretch", hide_index=True)

    with st.expander("Feature Schema", expanded=False):
        st.dataframe(pd.DataFrame({"feature": feature_list}), width="stretch", hide_index=True)

elif selected_nav == "Profile":
    _section_header("Profile", "Manage account identity and communication details.")
    profile_state = st.session_state.get("user_profile", {})
    user_col_1, user_col_2, user_col_3 = st.columns(3)
    user_col_1.markdown(
        create_kpi_card("Display Name", profile_state.get("full_name", "") or auth_display_name),
        unsafe_allow_html=True,
    )
    user_col_2.markdown(create_kpi_card("Username", auth_username), unsafe_allow_html=True)
    user_col_3.markdown(create_kpi_card("Role", "Hydrology Analyst"), unsafe_allow_html=True)

    with st.form("profile_form", clear_on_submit=False):
        p_col_1, p_col_2 = st.columns(2)
        with p_col_1:
            full_name = st.text_input("Full Name", value=profile_state.get("full_name", ""))
            mobile = st.text_input("Mobile Number", value=profile_state.get("mobile", ""))
            email = st.text_input("Email ID", value=profile_state.get("email", ""))
            organization = st.text_input("Organization", value=profile_state.get("organization", ""))
            designation = st.text_input("Designation", value=profile_state.get("designation", ""))
        with p_col_2:
            location = st.text_input("Location / City", value=profile_state.get("location", ""))
            emergency_name = st.text_input("Emergency Contact Name", value=profile_state.get("emergency_name", ""))
            emergency_mobile = st.text_input("Emergency Contact Mobile", value=profile_state.get("emergency_mobile", ""))
            alert_channel = st.selectbox(
                "Preferred Alert Channel",
                options=["Email", "SMS", "Both"],
                index=["Email", "SMS", "Both"].index(profile_state.get("alert_channel", "Both")),
            )
            notes = st.text_area(
                "Profile Notes",
                value=profile_state.get("notes", ""),
                placeholder="Add role context, basin responsibility, or operational notes.",
                height=100,
            )

        save_col, reset_col, _ = st.columns([1, 1, 4])
        with save_col:
            save_profile = st.form_submit_button("Save Profile", width="stretch")
        with reset_col:
            reset_profile = st.form_submit_button("Reset", width="stretch")

    if reset_profile:
        st.session_state["user_profile"] = {
            "full_name": st.session_state.get("auth_display_name", ""),
            "mobile": "",
            "email": "",
            "organization": "",
            "designation": "",
            "location": "",
            "emergency_name": "",
            "emergency_mobile": "",
            "alert_channel": "Both",
            "notes": "",
        }
        st.rerun()

    if save_profile:
        validation_errors = []
        if not full_name.strip():
            validation_errors.append("Full Name is required.")
        if not _is_valid_email(email.strip()):
            validation_errors.append("Enter a valid Email ID.")
        if not _is_valid_mobile(mobile.strip()):
            validation_errors.append("Enter a valid Mobile Number (10-15 digits).")
        if emergency_mobile.strip() and not _is_valid_mobile(emergency_mobile.strip()):
            validation_errors.append("Enter a valid Emergency Contact Mobile Number (10-15 digits).")

        if validation_errors:
            for err in validation_errors:
                st.error(err)
        else:
            st.session_state["user_profile"] = {
                "full_name": full_name.strip(),
                "mobile": mobile.strip(),
                "email": email.strip(),
                "organization": organization.strip(),
                "designation": designation.strip(),
                "location": location.strip(),
                "emergency_name": emergency_name.strip(),
                "emergency_mobile": emergency_mobile.strip(),
                "alert_channel": alert_channel,
                "notes": notes.strip(),
            }
            st.session_state["auth_display_name"] = full_name.strip()
            st.success("Profile saved successfully.")

    latest_profile = st.session_state.get("user_profile", {})
    st.markdown(
        create_glass_card(
            "Profile Summary",
            f"""
            <p style='color: var(--text-secondary); margin:0.1rem 0;'><strong>Name:</strong> {latest_profile.get("full_name", "-")}</p>
            <p style='color: var(--text-secondary); margin:0.1rem 0;'><strong>Email:</strong> {latest_profile.get("email", "-")}</p>
            <p style='color: var(--text-secondary); margin:0.1rem 0;'><strong>Mobile:</strong> {latest_profile.get("mobile", "-")}</p>
            <p style='color: var(--text-secondary); margin:0.1rem 0;'><strong>Organization:</strong> {latest_profile.get("organization", "-")}</p>
            <p style='color: var(--text-secondary); margin:0.1rem 0;'><strong>Role:</strong> {latest_profile.get("designation", "-")}</p>
            <p style='color: var(--text-secondary); margin:0.1rem 0;'><strong>Location:</strong> {latest_profile.get("location", "-")}</p>
            <p style='color: var(--text-secondary); margin:0.1rem 0;'><strong>Alert Channel:</strong> {latest_profile.get("alert_channel", "-")}</p>
            """,
        ),
        unsafe_allow_html=True,
    )

elif selected_nav == "Report Center":
    _section_header("Report Center", "One-click downloadable summary report (CSV and PDF).")
    summary_df, compare_df = _build_report_data(
        default_frame=default_data,
        csv_frame=csv_data,
        discharge_column=discharge_col,
        selected_model_name=selected_model,
        prediction_value=_to_display_discharge(single_prediction, selected_model),
        risk_value=risk_label,
    )
    pdf_bytes = _build_pdf_report(summary_df, compare_df)

    top_col_1, top_col_2 = st.columns(2)
    with top_col_1:
        st.download_button(
            label="Download Summary CSV",
            data=summary_df.to_csv(index=False).encode("utf-8"),
            file_name=f"runoff_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            width="stretch",
        )
    with top_col_2:
        if pdf_bytes is not None:
            st.download_button(
                label="Download PDF Report",
                data=pdf_bytes,
                file_name=f"runoff_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
                width="stretch",
            )
        else:
            st.info("PDF export engine is unavailable. CSV export is still available.")

    st.markdown("<div style='height: 0.4rem;'></div>", unsafe_allow_html=True)
    st.markdown(create_glass_card("Summary Preview", "<div style='color: var(--text-secondary);'>Key report metrics.</div>"), unsafe_allow_html=True)
    st.dataframe(summary_df, width="stretch", hide_index=True, height=320)

    if not compare_df.empty:
        st.markdown(create_glass_card("Model Comparison Preview", "<div style='color: var(--text-secondary);'>Comparative metrics for available models.</div>"), unsafe_allow_html=True)
        st.dataframe(compare_df, width="stretch", hide_index=True)

elif selected_nav == "About System":
    _section_header("About the System", "Hydrological intelligence designed for climate-resilient planning.")
    about_date_col = detect_datetime_column(default_data) if not default_data.empty else None
    historical_start = "Not available"
    historical_end = "Not available"
    if about_date_col and about_date_col in default_data.columns:
        about_dates = pd.to_datetime(default_data[about_date_col], errors="coerce").dropna()
        if not about_dates.empty:
            historical_start = str(about_dates.min().date())
            historical_end = str(about_dates.max().date())

    loaded_models = []
    if best_model_ready:
        loaded_models.append(best_model_label)
    if rf_ready:
        loaded_models.append("Random Forest")
    if lstm_ready:
        loaded_models.append("LSTM")
    loaded_models_label = ", ".join(loaded_models) if loaded_models else "No model artifacts loaded"

    st.markdown(
        create_glass_card(
            "Project Mission",
            """
            <p style='color: var(--text-secondary); margin-bottom:0.5rem;'>
                This platform forecasts river discharge for flood preparedness, hydrological monitoring, and climate resilience planning.
                It combines machine learning and sequence modeling with domain-aware feature engineering so teams can move from raw
                weather observations to actionable runoff insights.
            </p>
            <p style='color: var(--text-secondary); margin-bottom:0;'>
                Primary operational output: <strong>3-day ahead discharge estimate</strong> in CUMEC, with risk labels and model diagnostics.
            </p>
            """,
        ),
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height: 0.35rem;'></div>", unsafe_allow_html=True)

    about_col_1, about_col_2 = st.columns(2, gap="large")
    with about_col_1:
        st.markdown(
            create_glass_card(
                "Data Foundation",
                f"""
                <p style='color: var(--text-secondary); margin:0.1rem 0;'><strong>Historical XLSX Records:</strong> {len(default_data):,}</p>
                <p style='color: var(--text-secondary); margin:0.1rem 0;'><strong>Climate CSV Records:</strong> {len(csv_data):,}</p>
                <p style='color: var(--text-secondary); margin:0.1rem 0;'><strong>Historical Coverage:</strong> {historical_start} to {historical_end}</p>
                <p style='color: var(--text-secondary); margin:0.1rem 0;'>
                    Typical drivers include precipitation (PCP/P1-P3), temperature (TMAX/TMIN), humidity, solar radiation, wind,
                    and lagged discharge memory terms.
                </p>
                """,
            ),
            unsafe_allow_html=True,
        )

    with about_col_2:
        st.markdown(
            create_glass_card(
                "Model Stack",
                f"""
                <p style='color: var(--text-secondary); margin:0.1rem 0;'><strong>Available Models:</strong> {loaded_models_label}</p>
                <p style='color: var(--text-secondary); margin:0.1rem 0;'>
                    <strong>{best_model_label}</strong> is the primary production candidate when available.
                    <strong>Random Forest</strong> provides robust nonlinear baseline behavior and feature interpretability.
                    <strong>LSTM</strong> captures temporal sequence dynamics in evolving hydro-climatic patterns.
                </p>
                <p style='color: var(--text-secondary); margin:0.1rem 0;'>
                    Forecast quality is monitored through RMSE, R², and NSE in the Analytics and Report Center modules.
                </p>
                """,
            ),
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 0.35rem;'></div>", unsafe_allow_html=True)
    st.markdown(
        create_glass_card(
            "End-to-End Workflow",
            """
            <ol style='margin:0; color: var(--text-secondary); padding-left:1.2rem;'>
                <li><strong>Data ingestion and normalization:</strong> load XLSX/CSV sources, detect columns, normalize discharge units to CUMEC.</li>
                <li><strong>Feature preparation:</strong> align weather, lag-discharge memory, rolling summaries, and seasonal cyclic terms.</li>
                <li><strong>Prediction and scoring:</strong> run selected model, compute risk label, and show model health indicators.</li>
                <li><strong>Operational analysis:</strong> compare baseline vs climate-perturbed scenarios and inspect percent change profiles.</li>
                <li><strong>Reporting:</strong> export summary CSV/PDF with model metrics, data coverage, and latest forecast context.</li>
            </ol>
            """,
        ),
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height: 0.35rem;'></div>", unsafe_allow_html=True)

    st.markdown(
        create_glass_card(
            "How to Read Results",
            f"""
            <p style='color: var(--text-secondary); margin:0.1rem 0;'>
                <strong>Single Forecast:</strong> use this for immediate decision support on expected discharge ({discharge_unit_label}) and risk category.
            </p>
            <p style='color: var(--text-secondary); margin:0.1rem 0;'>
                <strong>Model Analytics:</strong> lower RMSE and higher R²/NSE generally indicate better historical fit.
            </p>
            <p style='color: var(--text-secondary); margin:0.1rem 0;'>
                <strong>Scenario Simulator:</strong> best used for directional what-if planning (sensitivity to rainfall/temperature/humidity/solar/wind changes),
                not as a substitute for full physically based flood routing models.
            </p>
            <p style='color: var(--text-secondary); margin:0.1rem 0;'>
                <strong>Report Center:</strong> generates auditable snapshots for communication, review meetings, and model-governance tracking.
            </p>
            """,
        ),
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height: 0.35rem;'></div>", unsafe_allow_html=True)

    with st.expander("Model Assumptions and Operational Boundaries", expanded=False):
        st.markdown(
            """
            - Forecast skill depends on data quality, representativeness, and stable sensor behavior.
            - Extreme events outside historical patterns can increase uncertainty.
            - Scenario perturbations apply controlled changes to observed drivers and are intended for planning sensitivity analysis.
            - For high-stakes flood operations, combine this platform with domain expert review and independent hydrodynamic validation.
            """
        )

st.markdown(
    "<div class='footer'>AI Runoff Forecasting System | Built for Hydrological Intelligence | © 2026</div>",
    unsafe_allow_html=True,
)


