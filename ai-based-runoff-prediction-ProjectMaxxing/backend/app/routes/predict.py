from __future__ import annotations

from fastapi import APIRouter

from app.schemas import (
    PredictRequest,
    PredictResponse,
    RealtimePredictRequest,
    RealtimePredictResponse,
)
from app.services.climate_engine_service import predict_realtime_next_3_days
from app.services.prediction_service import run_prediction


router = APIRouter(tags=["predict"])


@router.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    result = run_prediction(payload)
    return PredictResponse(**result)


@router.post("/predict-realtime", response_model=RealtimePredictResponse)
def predict_realtime(payload: RealtimePredictRequest) -> RealtimePredictResponse:
    result = predict_realtime_next_3_days(payload.model_dump())
    return RealtimePredictResponse(**result)
