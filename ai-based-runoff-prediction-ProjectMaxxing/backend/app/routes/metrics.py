from __future__ import annotations

from fastapi import APIRouter

from app.schemas import MetricsLatestResponse
from app.services.registry_service import get_best_model_details


router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get("/latest", response_model=MetricsLatestResponse)
def metrics_latest() -> MetricsLatestResponse:
    best_model = get_best_model_details()
    return MetricsLatestResponse(best_model=best_model)
