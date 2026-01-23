"""Pydantic models for MSA (Measurement System Analysis) computation.

Models for Gauge R&R input data and results.
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class MSAInput(BaseModel):
    """Input data for MSA Gauge R&R analysis.

    Attributes:
        parts: List of part identifiers (nombres de las partes)
        operators: List of operator identifiers (nombres de los operadores)
        trials: Number of measurement trials per operator per part (repeticiones)
        measurements: 3D array [parts][operators][trials] of measurement values
    """

    parts: list[str] = Field(
        ...,
        min_length=2,
        description="Lista de identificadores de partes (mínimo 2)",
        json_schema_extra={"example": ["Part1", "Part2", "Part3", "Part4", "Part5"]},
    )
    operators: list[str] = Field(
        ...,
        min_length=1,
        description="Lista de identificadores de operadores (mínimo 1)",
        json_schema_extra={"example": ["Op1", "Op2", "Op3"]},
    )
    trials: int = Field(
        ...,
        ge=2,
        description="Número de repeticiones por operador por parte (mínimo 2)",
        json_schema_extra={"example": 3},
    )
    measurements: list[list[list[float]]] = Field(
        ...,
        description="Matriz 3D de mediciones [partes][operadores][repeticiones]",
        json_schema_extra={
            "example": [
                [[2.5, 2.6, 2.5], [2.4, 2.5, 2.6], [2.5, 2.5, 2.4]],
                [[3.1, 3.0, 3.2], [3.0, 3.1, 3.0], [3.2, 3.1, 3.1]],
            ]
        },
    )

    @model_validator(mode="after")
    def validate_measurements_shape(self) -> "MSAInput":
        """Validate measurements array shape matches parts × operators × trials."""
        n_parts = len(self.parts)
        n_operators = len(self.operators)
        n_trials = self.trials

        # Check parts dimension
        if len(self.measurements) != n_parts:
            raise ValueError(
                f"La matriz de mediciones debe tener {n_parts} partes, "
                f"pero tiene {len(self.measurements)}"
            )

        # Check operators and trials dimensions for each part
        for i, part_measurements in enumerate(self.measurements):
            if len(part_measurements) != n_operators:
                raise ValueError(
                    f"La parte {i + 1} debe tener {n_operators} operadores, "
                    f"pero tiene {len(part_measurements)}"
                )

            for j, operator_trials in enumerate(part_measurements):
                if len(operator_trials) != n_trials:
                    raise ValueError(
                        f"La parte {i + 1}, operador {j + 1} debe tener {n_trials} "
                        f"repeticiones (trials), pero tiene {len(operator_trials)}"
                    )

                # Validate numeric values
                for k, value in enumerate(operator_trials):
                    if value is None or (isinstance(value, float) and (value != value)):  # NaN check
                        raise ValueError(
                            f"Valor inválido en parte {i + 1}, operador {j + 1}, "
                            f"repetición {k + 1}: se requiere un valor numérico"
                        )

        return self


class VarianceComponents(BaseModel):
    """Variance components from ANOVA decomposition.

    Attributes:
        equipment_variation: σ² for equipment (repeatability)
        operator_variation: σ² for operator (reproducibility)
        part_variation: σ² for parts
        total_variation: Total σ²
    """

    equipment_variation: float = Field(
        ...,
        ge=0,
        description="Variación del equipo (repetibilidad) - σ²",
        json_schema_extra={"example": 0.0234},
    )
    operator_variation: float = Field(
        ...,
        ge=0,
        description="Variación del operador (reproducibilidad) - σ²",
        json_schema_extra={"example": 0.0156},
    )
    part_variation: float = Field(
        ...,
        ge=0,
        description="Variación parte a parte - σ²",
        json_schema_extra={"example": 0.1823},
    )
    total_variation: float = Field(
        ...,
        ge=0,
        description="Variación total - σ²",
        json_schema_extra={"example": 0.2013},
    )


class MSAResult(BaseModel):
    """Result of MSA Gauge R&R analysis.

    Attributes:
        grr_percent: Total Gauge R&R as percentage of total variation
        repeatability_percent: Equipment variation as percentage
        reproducibility_percent: Operator variation as percentage
        part_variation_percent: Part-to-part variation as percentage
        ndc: Number of distinct categories
        category: Classification (excellent, marginal, unacceptable)
        components: Raw variance components
    """

    grr_percent: float = Field(
        ...,
        ge=0,
        description="Porcentaje total de Gauge R&R (%GRR)",
        json_schema_extra={"example": 18.2},
    )
    repeatability_percent: float = Field(
        ...,
        ge=0,
        description="Porcentaje de repetibilidad (variación del equipo)",
        json_schema_extra={"example": 12.5},
    )
    reproducibility_percent: float = Field(
        ...,
        ge=0,
        description="Porcentaje de reproducibilidad (variación del operador)",
        json_schema_extra={"example": 5.7},
    )
    part_variation_percent: float = Field(
        ...,
        ge=0,
        description="Porcentaje de variación parte a parte",
        json_schema_extra={"example": 81.8},
    )
    ndc: int = Field(
        ...,
        ge=0,
        description="Número de categorías distintas (ndc)",
        json_schema_extra={"example": 7},
    )
    category: Literal["excellent", "marginal", "unacceptable"] = Field(
        ...,
        description="Clasificación del sistema de medición según %GRR",
        json_schema_extra={"example": "marginal"},
    )
    components: VarianceComponents = Field(
        ...,
        description="Componentes de varianza del análisis ANOVA",
    )
