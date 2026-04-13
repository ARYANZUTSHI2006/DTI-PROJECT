from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


SECONDS_PER_DAY = 86400.0
REALISTIC_DISCHARGE_MAX_CUMEC = 500000.0


@dataclass(frozen=True)
class DischargeNormalizationReport:
    column_name: str | None
    raw_median: float
    raw_max: float
    applied_factor_to_cumec: float
    assumed_unit: str
    rationale: str
    converted_median: float
    converted_max: float
    warning: str | None = None


def load_dataset(file_path: Path) -> pd.DataFrame:
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        dataset = pd.read_csv(file_path)
        return normalize_discharge_dataframe(dataset)
    if suffix in {".xlsx", ".xls"}:
        dataset = _load_excel_dataset(file_path)
        return normalize_discharge_dataframe(dataset)
    raise ValueError(f"Unsupported dataset format: {suffix}")


def _normalize_name(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _resolve_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    normalized_columns = {_normalize_name(col): col for col in df.columns}
    for candidate in candidates:
        normalized_candidate = _normalize_name(candidate)
        if normalized_candidate in normalized_columns:
            return normalized_columns[normalized_candidate]
    return None


def _numeric_series(df: pd.DataFrame, column_name: str | None, default_value: float = 0.0) -> pd.Series:
    if column_name is None or column_name not in df.columns:
        return pd.Series(default_value, index=df.index, dtype=float)
    return pd.to_numeric(df[column_name], errors="coerce")


def _reshape_yearly_discharge_pivot(frame: pd.DataFrame) -> pd.DataFrame | None:
    if frame is None or frame.empty or frame.shape[1] < 2:
        return None

    candidate = frame.iloc[:, :2].copy()
    first_col = candidate.iloc[:, 0]
    second_col = candidate.iloc[:, 1]

    discharge_signals = [
        str(col) for col in candidate.columns
    ] + second_col.astype(str).tolist()
    has_discharge_signal = any("discharge" in _normalize_name(value) for value in discharge_signals)
    if not has_discharge_signal:
        return None

    years = pd.to_numeric(first_col, errors="coerce")
    valid_years = years.dropna()
    if valid_years.empty:
        return None

    valid_years = valid_years[(valid_years >= 1800) & (valid_years <= 2200)]
    if valid_years.size < 3:
        return None

    discharge = pd.to_numeric(second_col, errors="coerce")
    reshaped = pd.DataFrame(
        {
            "Date": pd.to_datetime(years.astype("Int64").astype(str) + "-01-01", errors="coerce"),
            "Discharge (CUMEC)": discharge,
        }
    ).dropna(subset=["Date", "Discharge (CUMEC)"])

    if reshaped.empty:
        return None

    return reshaped.reset_index(drop=True)


def _load_excel_dataset(file_path: Path) -> pd.DataFrame:
    workbook = pd.read_excel(file_path, sheet_name=None)
    preferred_sheets = ["Sheet1", "sheet1", "DATA", "Data"]
    for sheet_name in preferred_sheets:
        if sheet_name in workbook:
            preferred_frame = workbook[sheet_name]
            reshaped = _reshape_yearly_discharge_pivot(preferred_frame)
            return reshaped if reshaped is not None else preferred_frame

    for sheet_name, frame in workbook.items():
        normalized_cols = {_normalize_name(col) for col in frame.columns}
        if "date" in normalized_cols:
            return frame
        reshaped = _reshape_yearly_discharge_pivot(frame)
        if reshaped is not None:
            return reshaped

    first_sheet = next(iter(workbook.values()), pd.DataFrame())
    reshaped = _reshape_yearly_discharge_pivot(first_sheet)
    return reshaped if reshaped is not None else first_sheet


def detect_discharge_column(df: pd.DataFrame) -> str | None:
    if df is None or df.empty:
        return None

    preferred_candidates = [
        "Discharge (CUMEC)",
        "Discharge",
        "DisCUMEC",
        "discharge",
        "runoff",
        "cumec",
        "streamflow",
        "flow",
    ]
    resolved = _resolve_column(df, preferred_candidates)
    if resolved is not None:
        return resolved

    for col in df.select_dtypes(include=["number"]).columns:
        normalized = _normalize_name(col)
        if any(token in normalized for token in ("discharge", "cumec", "runoff", "streamflow", "flow")):
            return str(col)
    return None


def _infer_discharge_factor(series: pd.Series, column_name: str | None) -> tuple[float, str, str]:
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return 1.0, "unknown", "No numeric discharge values were available to infer units."

    positive = clean[clean > 0]
    sample = positive if not positive.empty else clean.abs()
    raw_median = float(sample.median())
    normalized_col = _normalize_name(column_name or "")

    if any(token in normalized_col for token in ("literpersecond", "litrepersecond", "lps", "lsec")):
        return 0.001, "L/s", "Column name indicates liters per second; converted to CUMEC by dividing by 1000."
    if any(token in normalized_col for token in ("m3perday", "m3day", "dailyvolume", "cumecday", "m3d")):
        return 1.0 / SECONDS_PER_DAY, "m3/day", "Column name indicates daily cubic-meter volumes; converted to CUMEC by dividing by 86400."
    if any(token in normalized_col for token in ("x1000", "times1000", "scaled1000")):
        return 0.001, "scaled x1000", "Column name indicates values scaled by 1000; converted to CUMEC."
    if any(token in normalized_col for token in ("x1000000", "times1000000", "scaled1000000")):
        return 0.000001, "scaled x1000000", "Column name indicates values scaled by 1,000,000; converted to CUMEC."

    if raw_median <= REALISTIC_DISCHARGE_MAX_CUMEC:
        return 1.0, "CUMEC", "Observed discharge magnitudes already fall within a realistic CUMEC range."

    daily_median = raw_median / SECONDS_PER_DAY
    thousand_median = raw_median / 1000.0
    million_median = raw_median / 1000000.0

    if raw_median >= 1000000.0 and daily_median <= REALISTIC_DISCHARGE_MAX_CUMEC:
        return (
            1.0 / SECONDS_PER_DAY,
            "m3/day",
            (
                "Raw discharge magnitudes are far above realistic CUMEC values. "
                "Treating them as daily cubic-meter volumes yields plausible flows."
            ),
        )
    if thousand_median <= REALISTIC_DISCHARGE_MAX_CUMEC:
        return 0.001, "L/s or x1000", "Raw discharge magnitudes exceed realistic CUMEC values; dividing by 1000 yields plausible flows."
    if million_median <= REALISTIC_DISCHARGE_MAX_CUMEC:
        return 0.000001, "scaled x1000000", "Raw discharge magnitudes exceed realistic CUMEC values; dividing by 1,000,000 yields plausible flows."
    return 1.0, "unknown", "Discharge units could not be inferred confidently; values were left unchanged."


def get_discharge_normalization_report(
    df: pd.DataFrame,
    discharge_column: str | None = None,
) -> dict[str, object]:
    existing = df.attrs.get("discharge_normalization") if hasattr(df, "attrs") else None
    if isinstance(existing, dict):
        return existing

    discharge_col = discharge_column or detect_discharge_column(df)
    series = _numeric_series(df, discharge_col)
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        report = DischargeNormalizationReport(
            column_name=discharge_col,
            raw_median=0.0,
            raw_max=0.0,
            applied_factor_to_cumec=1.0,
            assumed_unit="unknown",
            rationale="No discharge values were available for unit inspection.",
            converted_median=0.0,
            converted_max=0.0,
            warning=None,
        )
        return asdict(report)

    factor, assumed_unit, rationale = _infer_discharge_factor(clean, discharge_col)
    raw_median = float(clean.median())
    raw_max = float(clean.max())
    converted = clean * factor
    converted_median = float(converted.median())
    converted_max = float(converted.max())

    warning = None
    if converted_max > REALISTIC_DISCHARGE_MAX_CUMEC:
        warning = (
            f"Discharge values still exceed {REALISTIC_DISCHARGE_MAX_CUMEC:,.0f} CUMEC after normalization. "
            "Verify the source units or scaling in the dataset."
        )

    report = DischargeNormalizationReport(
        column_name=discharge_col,
        raw_median=raw_median,
        raw_max=raw_max,
        applied_factor_to_cumec=factor,
        assumed_unit=assumed_unit,
        rationale=rationale,
        converted_median=converted_median,
        converted_max=converted_max,
        warning=warning,
    )
    return asdict(report)


def normalize_discharge_dataframe(
    df: pd.DataFrame,
    discharge_column: str | None = None,
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df.copy()

    working = df.copy()
    existing_report = working.attrs.get("discharge_normalization") if hasattr(working, "attrs") else None
    if isinstance(existing_report, dict) and bool(existing_report.get("already_normalized_to_cumec")):
        return working

    report = get_discharge_normalization_report(working, discharge_column)
    discharge_col = report.get("column_name")
    factor = float(report.get("applied_factor_to_cumec", 1.0) or 1.0)

    if isinstance(discharge_col, str) and discharge_col in working.columns and factor != 1.0:
        converted = pd.to_numeric(working[discharge_col], errors="coerce") * factor
        working[discharge_col] = converted

    normalized_report = dict(report)
    normalized_report["already_normalized_to_cumec"] = True
    working.attrs["discharge_normalization"] = normalized_report
    return working


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    working = normalize_discharge_dataframe(df)
    date_col = _resolve_column(working, ["DATE", "Date", "time", "timestamp"])
    discharge_col = detect_discharge_column(working)
    pcp_col = _resolve_column(working, ["Mean_PCP", "PCP"])
    tmax_col = _resolve_column(working, ["Mean_Tmax", "TMAX"])
    tmin_col = _resolve_column(working, ["Mean_Tmin", "TMIN"])
    rh_col = _resolve_column(working, ["rh", "RH"])
    solar_col = _resolve_column(working, ["solar"])
    wind_col = _resolve_column(working, ["wind", "wind "])
    p1_col = _resolve_column(working, ["P1"])
    p2_col = _resolve_column(working, ["P2"])
    p3_col = _resolve_column(working, ["P3"])

    feature_df = pd.DataFrame(index=working.index)

    pcp_series = _numeric_series(working, pcp_col)
    if pcp_col is None:
        precip_parts = [
            _numeric_series(working, p1_col),
            _numeric_series(working, p2_col),
            _numeric_series(working, p3_col),
        ]
        precip_frame = pd.concat(precip_parts, axis=1)
        pcp_series = precip_frame.mean(axis=1, skipna=True)

    tmax_series = _numeric_series(working, tmax_col)
    tmin_series = _numeric_series(working, tmin_col)
    rh_series = _numeric_series(working, rh_col)
    solar_series = _numeric_series(working, solar_col)
    wind_series = _numeric_series(working, wind_col)
    discharge_series = _numeric_series(working, discharge_col)

    pcp_series = pcp_series.ffill().bfill().fillna(0.0)
    tmax_series = tmax_series.ffill().bfill().fillna(0.0)
    tmin_series = tmin_series.ffill().bfill().fillna(0.0)
    rh_series = rh_series.ffill().bfill().fillna(0.0)
    solar_series = solar_series.ffill().bfill().fillna(0.0)
    wind_series = wind_series.ffill().bfill().fillna(0.0)
    discharge_series = discharge_series.ffill().bfill().fillna(0.0)

    feature_df["Mean_PCP"] = pcp_series
    feature_df["Mean_Tmax"] = tmax_series
    feature_df["Mean_Tmin"] = tmin_series
    feature_df["rh"] = rh_series
    feature_df["solar"] = solar_series
    feature_df["wind"] = wind_series

    history_source = discharge_series.shift(1)
    discharge_fill = float(discharge_series.median()) if not discharge_series.empty else 0.0
    lag1 = history_source.shift(0)
    lag2 = history_source.shift(1)
    lag3 = history_source.shift(2)

    feature_df["lag_discharge_1"] = lag1.ffill().bfill().fillna(discharge_fill)
    feature_df["lag_discharge_2"] = lag2.ffill().bfill().fillna(discharge_fill)
    feature_df["lag_discharge_3"] = lag3.ffill().bfill().fillna(discharge_fill)

    discharge_roll = history_source.ffill().bfill().fillna(discharge_fill)
    feature_df["discharge_roll_mean_3"] = discharge_roll.rolling(3, min_periods=1).mean().fillna(discharge_fill)
    feature_df["discharge_roll_std_3"] = discharge_roll.rolling(3, min_periods=1).std(ddof=0).fillna(0.0)
    feature_df["PCP_roll_mean_3"] = pcp_series.rolling(3, min_periods=1).mean().fillna(0.0)
    feature_df["PCP_roll_sum_7"] = pcp_series.rolling(7, min_periods=1).sum().fillna(0.0)

    if date_col and date_col in working.columns:
        date_series = pd.to_datetime(working[date_col], errors="coerce")
        if date_series.notna().any():
            day_of_year = date_series.dt.dayofyear.ffill().bfill().fillna(1.0)
            angle = (2 * np.pi * day_of_year.astype(float)) / 365.25
            feature_df["season_sin"] = np.sin(angle)
            feature_df["season_cos"] = np.cos(angle)
        else:
            feature_df["season_sin"] = 0.0
            feature_df["season_cos"] = 1.0
    else:
        feature_df["season_sin"] = 0.0
        feature_df["season_cos"] = 1.0

    return feature_df.ffill().bfill().fillna(0.0)
