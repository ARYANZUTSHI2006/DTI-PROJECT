from __future__ import annotations

from fastapi import APIRouter

from app.schemas import TrainAutoRequest, TrainAutoResponse
from app.services.training_service import run_auto_training


router = APIRouter(prefix="/train", tags=["train"])


@router.post("/auto", response_model=TrainAutoResponse)
def train_auto(payload: TrainAutoRequest) -> TrainAutoResponse:
    result = run_auto_training(payload)
    return TrainAutoResponse(**result)
