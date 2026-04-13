from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TrainAutoRequest(BaseModel):
    train_end_year: int = Field(..., description="Last year to include in training split")
    test_start_year: int = Field(..., description="First year to include in test split")
    search_type: str = Field(default="randomized", description="Hyperparameter search strategy")
    n_iter: int = Field(default=50, ge=1, description="Number of search iterations")


class TrainAutoResponse(BaseModel):
    best_model: dict[str, Any]
    all_models: list[dict[str, Any]]
    artifact_path: str | None = None
    plots: list[str] = Field(default_factory=list)


class PredictRequest(BaseModel):
    features: dict[str, float] = Field(..., description="Feature dictionary expected by best model")


class PredictResponse(BaseModel):
    prediction: float
    model_name: str | None = None


class RealtimePredictRequest(BaseModel):
    rainfall: float
    temperature: float
    humidity: float
    tmax: float | None = None
    tmin: float | None = None
    solar: float | None = None
    wind: float | None = None
    rainfall_history: list[float] = Field(default_factory=list)
    rainfall_forecast: list[float] = Field(default_factory=list)


class RealtimePredictResponse(BaseModel):
    model: str
    rainfall_scenario: str | None = None
    forecast: list[dict[str, Any]]
    flood_probability_percent: float


class MetricsLatestResponse(BaseModel):
    best_model: dict[str, Any]


class ClimateTrainResponse(BaseModel):
    model: str
    rainfall_scenario: str
    R2: float
    NSE: float
    KGE: float
    RSR: float
    MAE: float
    PBIAS: float
    artifact_path: str
    comparison_count: int
    model_comparison: list[dict[str, Any]]


class ClimateFutureRequest(BaseModel):
    scenario: str = Field(..., description="Climate scenario: SSP245 or SSP585")
    periods: list[str] = Field(default_factory=lambda: ["2050s", "2080s"])


class ClimateFutureResponse(BaseModel):
    scenario: str
    selected_gc_ms: list[str]
    top_gcm_count: int
    model_used: str
    rainfall_scenario: str | None = None
    period_results: dict[str, Any]


class HealthResponse(BaseModel):
    status: str
