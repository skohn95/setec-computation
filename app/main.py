"""FastAPI application entry point for Setec Computation Service."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.routers import control_charts, health, msa


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup and shutdown events."""
    # Startup: Load any resources if needed
    yield
    # Shutdown: Clean up resources if needed


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


@app.exception_handler(ValueError)
async def validation_exception_handler(request: Request, exc: ValueError):
    """Handle validation errors with user-friendly Spanish messages."""
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
    """Handle unexpected errors with generic Spanish message."""
    # Log the actual error for debugging (English)
    print(f"Unexpected error: {exc}")
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
