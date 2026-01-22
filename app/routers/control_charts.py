"""Control Charts (SPC) computation router.

Placeholder for Epic 5 - Control Charts Agent.
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()


@router.post("/compute")
async def compute_control_charts():
    """Compute Control Charts (SPC) analysis.

    This endpoint will be implemented in Epic 5.
    """
    return JSONResponse(
        status_code=501,
        content={
            "status": "error",
            "error": {
                "code": "NOT_IMPLEMENTED",
                "message": "El cálculo de gráficos de control aún no está implementado. Disponible próximamente.",
            },
        },
    )
