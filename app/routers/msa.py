"""MSA (Measurement System Analysis) computation router.

Provides MSA analysis endpoints:
- POST /api/msa/compute - Basic Gauge R&R analysis
- POST /api/msa/compute-extended - Extended analysis with ANOVA, charts, etc.
- POST /api/msa/detect-structure - Detect data structure (parts, operators, trials)
"""

import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.models.msa import MSAInput, MSAResult
from app.models.extended_msa import ExtendedMSAInput, ExtendedMSAResult
from app.services.msa_calculator import MSACalculator
from app.services.extended_msa_calculator import ExtendedMSACalculator

logger = logging.getLogger(__name__)

router = APIRouter()

# Singleton calculator instances
_calculator = MSACalculator()
_extended_calculator = ExtendedMSACalculator()


# Response model for detect-structure endpoint
class DataStructureResult(BaseModel):
    """Result of data structure detection."""
    n_parts: int = Field(..., description="Número de partes detectadas")
    k_operators: int = Field(..., description="Número de operadores detectados")
    r_trials: int = Field(..., description="Número de repeticiones por operador por parte")
    total_measurements: int = Field(..., description="Total de mediciones")
    operator_ids: list[str] = Field(..., description="IDs de operadores detectados")
    part_ids: list[str] = Field(..., description="IDs de partes detectadas")


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


@router.post(
    "/compute-extended",
    response_model=None,
    summary="Calcular análisis MSA extendido con ANOVA completo",
    description="""
    Realiza un análisis de Gauge R&R extendido con:
    - Tabla ANOVA completa con valores F y P
    - Componentes de varianza detallados (%Contribución, %Estudio, %Tolerancia)
    - Métricas por operador con ranking de consistencia
    - Análisis de estabilidad (detección de deriva)
    - Análisis de linealidad y sesgo
    - Análisis de resolución (cuando se proporciona tolerancia)
    - Datos para gráficos (componentes, R-chart, X-bar, box plots, interacción)

    **Requisitos mínimos:**
    - 2 partes
    - 1 operador
    - 2 repeticiones (trials)

    **Campo opcional:**
    - `tolerance`: Tolerancia de especificación para cálculos de %Tolerancia
    """,
    responses={
        200: {
            "description": "Análisis MSA extendido completado exitosamente",
            "content": {
                "application/json": {
                    "example": {
                        "status": "success",
                        "data": {
                            "grr_percent": 18.2,
                            "category": "marginal",
                            "anova": {"rows": [], "significant_effects": ["Part"]},
                            "variance_components": {},
                            "operator_metrics": [],
                            "reference_operator": "Op1",
                            "charts": {},
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
        500: {
            "description": "Error interno del servidor",
            "content": {
                "application/json": {
                    "example": {
                        "status": "error",
                        "error": {
                            "code": "CALCULATION_ERROR",
                            "message": "Error al calcular el análisis MSA extendido.",
                        },
                    }
                }
            },
        },
    },
)
async def compute_extended_msa(data: ExtendedMSAInput) -> JSONResponse:
    """Compute extended MSA (Gauge R&R) analysis with full ANOVA and charts.

    Args:
        data: Input data with parts, operators, trials, measurements, and optional tolerance

    Returns:
        JSON response with extended MSA results or error
    """
    try:
        # Perform extended calculation
        result: ExtendedMSAResult = _extended_calculator.calculate(data)

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
        logger.warning(f"Extended MSA validation error: {e}")
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
        logger.error(f"Extended MSA calculation error: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": {
                    "code": "CALCULATION_ERROR",
                    "message": "Error al calcular el análisis MSA extendido. Por favor, verifica los datos e intenta de nuevo.",
                },
            },
        )


@router.post(
    "/detect-structure",
    response_model=None,
    summary="Detectar estructura de datos MSA",
    description="""
    Analiza los datos de entrada y retorna la estructura detectada:
    - Número de partes
    - Número de operadores
    - Número de repeticiones (trials)
    - Total de mediciones
    - IDs de operadores y partes

    Útil para validación y configuración de la interfaz antes de ejecutar el análisis completo.
    """,
    responses={
        200: {
            "description": "Estructura detectada exitosamente",
            "content": {
                "application/json": {
                    "example": {
                        "status": "success",
                        "data": {
                            "n_parts": 5,
                            "k_operators": 3,
                            "r_trials": 2,
                            "total_measurements": 30,
                            "operator_ids": ["Op1", "Op2", "Op3"],
                            "part_ids": ["P1", "P2", "P3", "P4", "P5"],
                        },
                    }
                }
            },
        },
        400: {
            "description": "Error de validación en los datos de entrada",
        },
    },
)
async def detect_structure(data: ExtendedMSAInput) -> JSONResponse:
    """Detect the structure of MSA input data.

    Args:
        data: Input data with parts, operators, trials, and measurements

    Returns:
        JSON response with detected structure
    """
    try:
        n_parts = len(data.parts)
        k_operators = len(data.operators)
        r_trials = data.trials
        total_measurements = n_parts * k_operators * r_trials

        result = DataStructureResult(
            n_parts=n_parts,
            k_operators=k_operators,
            r_trials=r_trials,
            total_measurements=total_measurements,
            operator_ids=data.operators,
            part_ids=data.parts,
        )

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "data": result.model_dump(),
            },
        )

    except ValueError as e:
        logger.warning(f"Structure detection validation error: {e}")
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
        logger.error(f"Structure detection error: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "Error al detectar la estructura de datos.",
                },
            },
        )
