from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    BASE_DIR: Path = Path(__file__).resolve().parents[1]
    PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

    LOG_DIR: Path = BASE_DIR / "logs"
    LOG_FILE: str = "backend.log"
    PLOTS_DIR: Path = BASE_DIR / "plots"

    MODEL_REGISTRY_PATH: Path = BASE_DIR / "model_registry.json"
    MODEL_ARTIFACTS_DIR: Path = BASE_DIR / "artifacts"
    CLIMATE_REGISTRY_PATH: Path = BASE_DIR / "climate" / "climate_registry.json"
    CLIMATE_DATA_DIR: Path = BASE_DIR / "climate"

    DEFAULT_DATA_DIRS: tuple[Path, ...] = (
        PROJECT_ROOT / "data",
        PROJECT_ROOT / "dataset",
        PROJECT_ROOT / "datasets",
        BASE_DIR / "data",
    )

    CORS_ORIGINS: list[str] = None

    def __post_init__(self):
        object.__setattr__(
            self,
            "CORS_ORIGINS",
            [origin.strip() for origin in os.getenv("CORS_ORIGINS", "*").split(",") if origin.strip()],
        )


settings = Settings()
