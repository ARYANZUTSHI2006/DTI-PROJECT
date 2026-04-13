from __future__ import annotations

from fastapi import APIRouter

from app.schemas import (
    ClimateFutureRequest,
    ClimateFutureResponse,
    ClimateTrainResponse,
)
from app.services.climate_engine_service import (
    generate_cmip6_future_streamflow,
    train_climate_integrated_engine,
)


router = APIRouter(prefix="/climate", tags=["climate"])


@router.post("/train-engine", response_model=ClimateTrainResponse)
def train_engine() -> ClimateTrainResponse:
    result = train_climate_integrated_engine()
    return ClimateTrainResponse(**result)


@router.post("/future-streamflow", response_model=ClimateFutureResponse)
def future_streamflow(payload: ClimateFutureRequest) -> ClimateFutureResponse:
    result = generate_cmip6_future_streamflow(payload.scenario, payload.periods)
    return ClimateFutureResponse(**result)
