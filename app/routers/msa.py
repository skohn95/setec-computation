"""MSA (Measurement System Analysis) computation router.

Placeholder for Story 4.2 - MSA Calculation Engine.
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()


@router.post("/compute")
async def compute_msa():
    """Compute MSA (Gauge R&R) analysis.

    This endpoint will be implemented in Story 4.2.
    """
    return JSONResponse(
        status_code=501,
        content={
            "status": "error",
            "error": {
                "code": "NOT_IMPLEMENTED",
                "message": "El cálculo MSA aún no está implementado. Disponible próximamente.",
            },
        },
    )
