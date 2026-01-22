"""Health check endpoint for service monitoring."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check():
    """Return service health status for Railway monitoring."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "service": "setec-computation",
    }
