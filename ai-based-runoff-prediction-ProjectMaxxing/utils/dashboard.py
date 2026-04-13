from __future__ import annotations
from typing import Iterable, Sequence
import math
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def inject_custom_css() -> None:
    import streamlit as st
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=Space+Grotesk:wght@500;600;700&display=swap');
            :root {
                --bg-primary: #0b1324;
                --bg-secondary: #0f1b35;
                --card-bg: rgba(20, 30, 55, 0.55);
                --card-border: rgba(120, 160, 255, 0.18);
                --accent: #45c0ff;
                --accent-2: #7c6cff;
                --text-primary: #eaf1ff;
                --text-secondary: rgba(234, 241, 255, 0.72);
            }

            html, body, [class*="css"]  {
                font-family: 'IBM Plex Sans', sans-serif;
                color: var(--text-primary);
            }

            .stApp {
                background: radial-gradient(circle at top left, rgba(69, 192, 255, 0.12), transparent 40%),
                            radial-gradient(circle at 20% 20%, rgba(124, 108, 255, 0.16), transparent 45%),
                            linear-gradient(160deg, #0b1324 0%, #0f1b35 60%, #0b1324 100%);
            }

            section[data-testid="stSidebar"] {
                background: linear-gradient(180deg, rgba(12, 22, 42, 0.98), rgba(12, 22, 42, 0.96));
                border-right: 1px solid rgba(100, 140, 220, 0.15);
            }

            section[data-testid="stSidebar"] .block-container {
                padding-top: 1rem;
                padding-left: 0.8rem;
                padding-right: 0.8rem;
            }

            .sidebar-brand,
            .sidebar-user-card {
                display: flex;
                align-items: center;
                gap: 0.65rem;
                border: 1px solid rgba(130, 160, 230, 0.22);
                border-radius: 14px;
                padding: 0.75rem 0.8rem;
                background: linear-gradient(140deg, rgba(74, 108, 255, 0.18), rgba(36, 220, 180, 0.12));
                box-shadow: 0 10px 24px rgba(5, 10, 22, 0.36);
                margin-bottom: 0.7rem;
            }

            .sidebar-user-link,
            .sidebar-user-link:visited,
            .sidebar-user-link:hover {
                text-decoration: none !important;
                color: inherit !important;
                display: block;
            }

            .sidebar-logo {
                width: 1.9rem;
                height: 1.9rem;
                border-radius: 10px;
                display: flex;
                align-items: center;
                justify-content: center;
                background: linear-gradient(145deg, #58d7ff, #7b8bff);
                color: #0f1730;
                font-size: 1rem;
                font-weight: 700;
            }

            .sidebar-brand-title {
                font-family: 'Space Grotesk', sans-serif;
                font-size: 0.98rem;
                font-weight: 700;
                letter-spacing: 0.02em;
                line-height: 1.1;
            }

            .sidebar-brand-subtitle {
                font-size: 0.72rem;
                color: var(--text-secondary);
                margin-top: 0.05rem;
                letter-spacing: 0.03em;
            }

            .nav-section-label {
                font-size: 0.68rem;
                text-transform: uppercase;
                letter-spacing: 0.13em;
                color: rgba(214, 225, 255, 0.5);
                margin: 0.72rem 0 0.3rem 0.15rem;
                font-weight: 600;
            }

            section[data-testid="stSidebar"] a.sidebar-nav-link,
            section[data-testid="stSidebar"] a.sidebar-nav-link:visited {
                display: flex;
                align-items: center;
                gap: 0.58rem;
                width: 100%;
                text-decoration: none !important;
                color: rgba(232, 239, 255, 0.86) !important;
                border: 1px solid transparent;
                border-radius: 11px;
                padding: 0.5rem 0.62rem;
                margin: 0.18rem 0;
                font-size: 0.9rem;
                font-weight: 500;
                transition: all 0.16s ease;
                background: rgba(255, 255, 255, 0.02);
                white-space: nowrap;
            }

            section[data-testid="stSidebar"] a.sidebar-nav-link:hover {
                border-color: rgba(130, 160, 230, 0.33);
                background: rgba(114, 134, 200, 0.14);
                color: #f2f7ff !important;
                text-decoration: none !important;
            }

            section[data-testid="stSidebar"] a.sidebar-nav-link.active,
            section[data-testid="stSidebar"] a.sidebar-nav-link.active:visited {
                background: #eef2ff;
                color: #161b33 !important;
                border-color: rgba(255, 255, 255, 0.72);
                font-weight: 600;
                box-shadow: 0 8px 20px rgba(6, 10, 22, 0.35);
                text-decoration: none !important;
            }

            .sidebar-nav-icon {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                min-width: 1.15rem;
                width: 1.15rem;
                opacity: 0.92;
                font-weight: 700;
            }

            .sidebar-nav-text {
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
            }

            section[data-testid="stSidebar"] div.stButton > button {
                width: 100%;
                justify-content: flex-start;
                border-radius: 12px;
                border: 1px solid transparent;
                background: rgba(255, 255, 255, 0.02);
                color: rgba(227, 236, 255, 0.9);
                padding: 0.56rem 0.7rem;
                font-size: 0.92rem;
                font-weight: 500;
                transition: all 0.18s ease;
            }

            section[data-testid="stSidebar"] div.stButton > button[kind="secondary"]:hover {
                border-color: rgba(130, 160, 230, 0.35);
                background: rgba(114, 134, 200, 0.18);
                color: #f1f6ff;
            }

            section[data-testid="stSidebar"] div.stButton > button[kind="primary"] {
                background: #eef2ff;
                color: #141a31;
                border-color: rgba(255, 255, 255, 0.72);
                box-shadow: 0 8px 20px rgba(6, 10, 22, 0.35);
                font-weight: 600;
            }

            .nav-active-note {
                margin-top: 0.7rem;
                padding: 0.62rem 0.72rem;
                border: 1px solid rgba(128, 160, 240, 0.25);
                border-radius: 12px;
                background: rgba(56, 72, 120, 0.25);
            }

            .nav-active-title {
                font-size: 0.84rem;
                font-weight: 600;
                color: #f2f6ff;
                margin-bottom: 0.2rem;
            }

            .nav-active-subtitle {
                font-size: 0.75rem;
                color: rgba(225, 235, 255, 0.72);
                line-height: 1.35;
            }

            .status-row {
                display: flex;
                gap: 0.35rem;
                margin-top: 0.2rem;
            }

            .status-chip {
                display: inline-flex;
                align-items: center;
                gap: 0.28rem;
                border-radius: 999px;
                padding: 0.2rem 0.5rem;
                font-size: 0.72rem;
                font-weight: 600;
                border: 1px solid transparent;
            }

            .status-ready {
                color: #9cf3cb;
                background: rgba(48, 188, 134, 0.14);
                border-color: rgba(109, 240, 190, 0.28);
            }

            .status-missing {
                color: #ffc184;
                background: rgba(218, 128, 38, 0.14);
                border-color: rgba(255, 188, 122, 0.28);
            }

            .main > div {
                padding-top: 1.1rem;
            }

            .hero {
                border: 1px solid var(--card-border);
                border-radius: 18px;
                padding: 1.6rem 1.8rem;
                background: linear-gradient(120deg, rgba(69, 192, 255, 0.18), rgba(124, 108, 255, 0.08));
                box-shadow: 0 12px 30px rgba(8, 12, 24, 0.45);
                position: relative;
                overflow: hidden;
            }

            .hero:before {
                content: "";
                position: absolute;
                inset: 0;
                background: linear-gradient(120deg, rgba(255,255,255,0.12), rgba(255,255,255,0));
                opacity: 0.6;
                animation: shimmer 6s ease-in-out infinite;
            }

            .hero-title {
                font-family: 'Space Grotesk', sans-serif;
                font-size: 2.1rem;
                font-weight: 700;
                margin: 0;
            }

            .hero-subtitle {
                margin: 0.4rem 0 0 0;
                font-size: 1.05rem;
                color: var(--text-secondary);
            }

            .glass-card {
                border-radius: 16px;
                border: 1px solid var(--card-border);
                background: var(--card-bg);
                backdrop-filter: blur(14px);
                padding: 1.1rem 1.25rem;
                box-shadow: 0 10px 26px rgba(8, 12, 24, 0.35);
                margin-bottom: 0.9rem;
                line-height: 1.45;
            }

            .prediction-value {
                font-size: 2.4rem;
                font-weight: 700;
                font-family: 'Space Grotesk', sans-serif;
                line-height: 1.1;
                margin-top: 0.35rem;
            }

            .risk-badge {
                display: inline-flex;
                align-items: center;
                gap: 0.4rem;
                padding: 0.3rem 0.8rem;
                border-radius: 999px;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.04em;
                font-size: 0.75rem;
                transition: all 0.35s ease;
            }

            .risk-normal {
                background: rgba(34, 210, 143, 0.18);
                color: #5ff3c0;
                box-shadow: 0 0 14px rgba(34, 210, 143, 0.6);
            }

            .risk-moderate {
                background: rgba(255, 170, 60, 0.18);
                color: #ffb35a;
                box-shadow: 0 0 14px rgba(255, 170, 60, 0.55);
            }

            .risk-high {
                background: rgba(255, 118, 69, 0.22);
                color: #ff9d6c;
                box-shadow: 0 0 16px rgba(255, 118, 69, 0.62);
            }

            .risk-extreme {
                background: rgba(255, 90, 90, 0.18);
                color: #ff9aa0;
                box-shadow: 0 0 16px rgba(255, 90, 90, 0.65);
            }

            .kpi-title {
                font-size: 0.85rem;
                color: var(--text-secondary);
                text-transform: uppercase;
                letter-spacing: 0.08em;
                margin-bottom: 0.3rem;
            }

            .kpi-value {
                font-size: 1.6rem;
                font-weight: 700;
                font-family: 'Space Grotesk', sans-serif;
            }

            .section-title {
                font-family: 'Space Grotesk', sans-serif;
                font-size: 1.4rem;
                margin: 0.4rem 0 0.6rem;
            }

            .footer {
                text-align: center;
                opacity: 0.75;
                margin-top: 2rem;
                padding-top: 0.8rem;
                border-top: 1px solid rgba(120,120,120,0.20);
                font-size: 0.9rem;
            }

            @keyframes shimmer {
                0% { transform: translateX(-30%); }
                50% { transform: translateX(10%); }
                100% { transform: translateX(-30%); }
            }

            @keyframes glowPulse {
                0% { box-shadow: 0 0 14px rgba(69, 192, 255, 0.35); }
                50% { box-shadow: 0 0 22px rgba(69, 192, 255, 0.6); }
                100% { box-shadow: 0 0 14px rgba(69, 192, 255, 0.35); }
            }

            .animate-update {
                animation: glowPulse 1.6s ease-in-out;
            }

            div[data-testid="stFileUploader"] {
                background: rgba(20, 30, 55, 0.45);
                border: 1px dashed rgba(124, 200, 255, 0.4);
                border-radius: 14px;
                padding: 1rem;
            }

            div[data-testid="stDownloadButton"] button {
                background: linear-gradient(90deg, #45c0ff, #7c6cff);
                color: #0b1324;
                border: none;
                font-weight: 600;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def detect_discharge_column(df: pd.DataFrame) -> str | None:
    if df is None or df.empty:
        return None

    report = df.attrs.get("discharge_normalization") if hasattr(df, "attrs") else None
    if isinstance(report, dict):
        reported_col = report.get("column_name")
        if isinstance(reported_col, str) and reported_col in df.columns:
            return reported_col

    candidate_names = [
        "discharge",
        "dischargecumec",
        "runoff",
        "q",
        "cumec",
        "observeddischarge",
        "riverdischarge",
        "streamflow",
        "flow",
    ]
    lowered = {str(col).lower(): col for col in df.columns}
    normalized = {"".join(ch for ch in str(col).lower() if ch.isalnum()): col for col in df.columns}

    for candidate in candidate_names:
        if candidate in lowered:
            return lowered[candidate]
        if candidate in normalized:
            return normalized[candidate]

    for col in df.select_dtypes(include=["number"]).columns:
        col_name = str(col).lower()
        if (
            "discharge" in col_name
            or "cumec" in col_name
            or "runoff" in col_name
            or "streamflow" in col_name
            or col_name == "flow"
        ):
            return col

    return None


def detect_datetime_column(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        col_lower = col.lower()
        if "date" in col_lower or "time" in col_lower:
            return col
    return None


def get_risk_label(predicted_discharge: float, historical_series: Iterable[float] | None = None) -> tuple[str, str, str]:
    hist = np.asarray(list(historical_series), dtype=float) if historical_series is not None else np.array([])
    hist = hist[~np.isnan(hist)]

    if hist.size >= 10:
        moderate_threshold = float(np.quantile(hist, 0.70))
        high_threshold = float(np.quantile(hist, 0.80))
        extreme_threshold = float(np.quantile(hist, 0.90))
    else:
        moderate_threshold = 300.0
        high_threshold = 500.0
        extreme_threshold = 800.0

    if predicted_discharge < moderate_threshold:
        return "Normal", "risk-normal", "🟢"
    if predicted_discharge < high_threshold:
        return "Moderate", "risk-moderate", "🟠"
    if predicted_discharge < extreme_threshold:
        return "High", "risk-high", "🟠"
    return "Extreme", "risk-extreme", "🔴"


def create_glass_card(title: str, body_html: str, animate: bool = False) -> str:
    anim_class = " animate-update" if animate else ""
    return (
        f"<div class=\"glass-card{anim_class}\">"
        f"<div class=\"kpi-title\">{title}</div>"
        f"{body_html}"
        "</div>"
    )


def create_kpi_card(title: str, value: str) -> str:
    return (
        "<div class=\"glass-card\">"
        f"<div class=\"kpi-title\">{title}</div>"
        f"<div class=\"kpi-value\">{value}</div>"
        "</div>"
    )


def sparkline_chart(series: Sequence[float], color: str = "#45c0ff") -> go.Figure:
    x_vals = list(range(len(series)))
    fig = go.Figure(
        data=[
            go.Scatter(
                x=x_vals,
                y=series,
                mode="lines",
                line={"color": color, "width": 2},
                hoverinfo="skip",
            )
        ]
    )
    fig.update_layout(
        template="plotly_dark",
        height=120,
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        xaxis={"visible": False},
        yaxis={"visible": False},
    )
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig


def historical_discharge_chart(df: pd.DataFrame, discharge_col: str | None) -> go.Figure:
    discharge_col = discharge_col or detect_discharge_column(df)
    if discharge_col is None or discharge_col not in df.columns:
        fallback = pd.DataFrame({"Index": np.arange(1, len(df) + 1), "Discharge": np.zeros(len(df))})
        fig = px.line(
            fallback,
            x="Index",
            y="Discharge",
            title="📊 Historical Discharge",
            template="plotly_dark",
        )
        fig.update_layout(height=360)
        return fig

    date_col = detect_datetime_column(df)
    chart_df = df.copy()
    chart_df[discharge_col] = pd.to_numeric(chart_df[discharge_col], errors="coerce")
    max_points = 2000

    if date_col:
        chart_df[date_col] = pd.to_datetime(chart_df[date_col], errors="coerce")
        chart_df = chart_df.dropna(subset=[date_col, discharge_col]).sort_values(date_col)
        if chart_df.empty:
            chart_df = pd.DataFrame({"Index": np.arange(1, len(df) + 1), "Discharge": np.zeros(len(df))})
            fig = px.line(chart_df, x="Index", y="Discharge", title="📊 Historical Discharge", template="plotly_dark")
            fig.update_layout(height=360)
            return fig
        if len(chart_df) > max_points:
            step = max(1, math.ceil(len(chart_df) / max_points))
            chart_df = chart_df.iloc[::step].copy()
        fig = px.line(
            chart_df,
            x=date_col,
            y=discharge_col,
            title="📊 Historical Discharge",
            template="plotly_dark",
        )
    else:
        chart_df = chart_df.dropna(subset=[discharge_col]).reset_index(drop=True)
        if chart_df.empty:
            chart_df = pd.DataFrame({"Index": np.arange(1, len(df) + 1), "Discharge": np.zeros(len(df))})
            fig = px.line(chart_df, x="Index", y="Discharge", title="📊 Historical Discharge", template="plotly_dark")
            fig.update_layout(height=360)
            return fig
        chart_df["Index"] = chart_df.index + 1
        if len(chart_df) > max_points:
            step = max(1, math.ceil(len(chart_df) / max_points))
            chart_df = chart_df.iloc[::step].copy()
        fig = px.line(
            chart_df,
            x="Index",
            y=discharge_col,
            title="📊 Historical Discharge",
            template="plotly_dark",
        )

    fig.update_layout(height=360, legend_title_text="", hovermode="x unified")
    fig.update_xaxes(showgrid=False, rangeslider_visible=False)
    fig.update_yaxes(showgrid=False)
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig


def predicted_vs_observed_chart(df: pd.DataFrame, observed_col: str | None, predicted: np.ndarray) -> go.Figure:
    chart_df = pd.DataFrame({"step": np.arange(1, len(predicted) + 1), "Predicted": predicted})
    max_points = 2000

    if observed_col and observed_col in df.columns:
        observed_values = pd.to_numeric(df[observed_col], errors="coerce").to_numpy()
        aligned_observed = observed_values[: len(predicted)]
        chart_df["Observed"] = aligned_observed

    chart_df["Predicted"] = pd.to_numeric(chart_df["Predicted"], errors="coerce")
    chart_df = chart_df.dropna(subset=["Predicted"])

    if len(chart_df) > max_points:
        step = max(1, math.ceil(len(chart_df) / max_points))
        chart_df = chart_df.iloc[::step].copy()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=chart_df["step"],
            y=chart_df["Predicted"],
            mode="lines+markers",
            name="Predicted",
            line={"color": "#45c0ff"},
            marker={"size": 4},
        )
    )

    if "Observed" in chart_df.columns:
        fig.add_trace(
            go.Scatter(
                x=chart_df["step"],
                y=chart_df["Observed"],
                mode="lines+markers",
                name="Observed",
                line={"color": "#9f8cff"},
                marker={"size": 4},
            )
        )

    fig.update_layout(
        title="📈 Predicted vs Observed",
        xaxis_title="Time Step",
        yaxis_title="Discharge (CUMEC)",
        height=360,
        template="plotly_dark",
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig


def feature_importance_chart(importance_df: pd.DataFrame) -> go.Figure:
    top_df = importance_df.head(15)
    fig = px.bar(
        top_df,
        x="importance",
        y="feature",
        orientation="h",
        title="🌧 Feature Importance (Random Forest)",
        color="importance",
        color_continuous_scale=["#0b1324", "#45c0ff", "#7c6cff"],
        template="plotly_dark",
    )
    fig.update_layout(height=420, yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig
