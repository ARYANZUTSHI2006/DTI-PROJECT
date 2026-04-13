from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


class BackendException(HTTPException):
    def __init__(self, status_code: int, detail: str):
        super().__init__(status_code=status_code, detail=detail)


class RegistryNotFoundError(BackendException):
    def __init__(self, detail: str = "Model registry not found"):
        super().__init__(status_code=404, detail=detail)


class ModelArtifactError(BackendException):
    def __init__(self, detail: str = "Model artifact loading failed"):
        super().__init__(status_code=500, detail=detail)


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={
                "message": "Validation error",
                "errors": exc.errors(),
                "path": request.url.path,
            },
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"message": exc.detail, "path": request.url.path},
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=500,
            content={
                "message": "Internal server error",
                "details": str(exc),
                "path": request.url.path,
            },
        )
