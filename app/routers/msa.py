"""MSA (Measurement System Analysis) computation router.

Provides the POST /api/msa/compute endpoint for Gauge R&R analysis.
"""

import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.models.msa import MSAInput, MSAResult
from app.services.msa_calculator import MSACalculator

logger = logging.getLogger(__name__)

router = APIRouter()

# Singleton calculator instance
_calculator = MSACalculator()


@router.post(
    "/compute",
    response_model=None,
    summary="Calcular análisis MSA (Gauge R&R)",
    description="""
    Realiza un análisis de Gauge R&R (Repetibilidad y Reproducibilidad)
    basado en ANOVA para evaluar el sistema de medición.

    **Requisitos mínimos:**
    - 2 partes
    - 1 operador
    - 2 repeticiones (trials)

    **Categorías de clasificación:**
    - `excellent`: %GRR < 10% - Sistema de medición aceptable
    - `marginal`: 10% ≤ %GRR ≤ 30% - Puede ser aceptable según la aplicación
    - `unacceptable`: %GRR > 30% - El sistema necesita mejoras
    """,
    responses={
        200: {
            "description": "Análisis MSA completado exitosamente",
            "content": {
                "application/json": {
                    "example": {
                        "status": "success",
                        "data": {
                            "grr_percent": 18.2,
                            "repeatability_percent": 12.5,
                            "reproducibility_percent": 5.7,
                            "part_variation_percent": 81.8,
                            "ndc": 7,
                            "category": "marginal",
                            "components": {
                                "equipment_variation": 0.0234,
                                "operator_variation": 0.0156,
                                "part_variation": 0.1823,
                                "total_variation": 0.2013,
                            },
                        },
                    }
                }
            },
        },
        400: {
            "description": "Error de validación en los datos de entrada",
            "content": {
                "application/json": {
                    "example": {
                        "status": "error",
                        "error": {
                            "code": "VALIDATION_ERROR",
                            "message": "Datos de entrada inválidos: Se requieren al menos 2 partes para el análisis",
                        },
                    }
                }
            },
        },
        422: {
            "description": "Error de formato en los datos de entrada",
            "content": {
                "application/json": {
                    "example": {
                        "status": "error",
                        "error": {
                            "code": "VALIDATION_ERROR",
                            "message": "Datos de entrada inválidos. Verifica el formato de los datos.",
                            "details": [],
                        },
                    }
                }
            },
        },
        500: {
            "description": "Error interno del servidor",
            "content": {
                "application/json": {
                    "example": {
                        "status": "error",
                        "error": {
                            "code": "INTERNAL_ERROR",
                            "message": "Error interno del servidor. Por favor, intenta de nuevo.",
                        },
                    }
                }
            },
        },
    },
)
async def compute_msa(data: MSAInput) -> JSONResponse:
    """Compute MSA (Gauge R&R) analysis.

    Args:
        data: Input data with parts, operators, trials, and measurements

    Returns:
        JSON response with MSA results or error
    """
    try:
        # Perform calculation
        result: MSAResult = _calculator.calculate(data)

        # Return success response
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "data": result.model_dump(),
            },
        )

    except ValueError as e:
        # Validation error - already has Spanish message from model validators
        logger.warning(f"MSA validation error: {e}")
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": f"Datos de entrada inválidos: {str(e)}",
                },
            },
        )

    except Exception as e:
        # Unexpected error - log in English, respond in Spanish
        logger.error(f"MSA calculation error: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": {
                    "code": "CALCULATION_ERROR",
                    "message": "Error al calcular el análisis MSA. Por favor, verifica los datos e intenta de nuevo.",
                },
            },
        )
