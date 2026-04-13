from __future__ import annotations

import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.core.exceptions import register_exception_handlers
from app.core.logger import setup_logging, get_logger
from app.routes.climate import router as climate_router
from app.routes.health import router as health_router
from app.routes.metrics import router as metrics_router
from app.routes.predict import router as predict_router
from app.routes.train import router as train_router


setup_logging(settings.LOG_DIR, settings.LOG_FILE)
logger = get_logger(__name__)

app = FastAPI(
    title="Runoff Prediction Backend",
    description="Production-ready FastAPI backend for runoff model training and inference.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    start = time.perf_counter()
    try:
        response = await call_next(request)
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info("%s %s completed in %.2fms", request.method, request.url.path, elapsed_ms)
    return response


app.include_router(health_router)
app.include_router(train_router)
app.include_router(predict_router)
app.include_router(metrics_router)
app.include_router(climate_router)

register_exception_handlers(app)
