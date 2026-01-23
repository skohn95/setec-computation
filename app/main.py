"""FastAPI application entry point for Setec Computation Service."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.routers import control_charts, health, msa

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup and shutdown events."""
    logger.info("Starting Setec Computation Service")
    yield
    logger.info("Shutting down Setec Computation Service")


app = FastAPI(
    title="Setec Computation Service",
    description="Statistical computation service for Setec AI Hub",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _serialize_validation_errors(errors: list) -> list:
    """Convert validation errors to JSON-serializable format."""
    serialized = []
    for error in errors:
        serialized_error = {
            "type": error.get("type", "unknown"),
            "loc": error.get("loc", []),
            "msg": error.get("msg", "Error de validación"),
        }
        # Only include 'input' if it's a simple type
        if "input" in error and isinstance(error["input"], (str, int, float, bool, type(None), list, dict)):
            serialized_error["input"] = error["input"]
        serialized.append(serialized_error)
    return serialized


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle FastAPI request validation errors with Spanish messages."""
    logger.warning(f"Request validation error: {exc.errors()}")
    serialized_errors = _serialize_validation_errors(exc.errors())
    return JSONResponse(
        status_code=422,
        content={
            "status": "error",
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Datos de entrada inválidos. Verifica el formato de los datos.",
                "details": serialized_errors,
            },
        },
    )


@app.exception_handler(ValueError)
async def validation_exception_handler(request: Request, exc: ValueError):
    """Handle validation errors with user-friendly Spanish messages."""
    logger.warning(f"Value error: {exc}")
    return JSONResponse(
        status_code=400,
        content={
            "status": "error",
            "error": {
                "code": "VALIDATION_ERROR",
                "message": f"Datos de entrada inválidos: {str(exc)}",
            },
        },
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Handle unexpected errors with generic Spanish message.

    Note: HTTPException is handled by FastAPI's default handler and won't reach here.
    """
    # Skip HTTPException - let FastAPI handle it
    if isinstance(exc, HTTPException):
        raise exc

    # Log the actual error for debugging (English)
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "Error interno del servidor. Por favor, intenta de nuevo.",
            },
        },
    )


# Include routers
app.include_router(health.router)
app.include_router(msa.router, prefix="/api/msa", tags=["MSA"])
app.include_router(control_charts.router, prefix="/api/control-charts", tags=["Control Charts"])
