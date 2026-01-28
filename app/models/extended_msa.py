"""Extended Pydantic models for MSA (Measurement System Analysis) computation.

Models for extended Gauge R&R analysis including ANOVA tables, detailed variance
components, operator metrics, stability analysis, linearity/bias, resolution
analysis, and chart data.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

from app.models.msa import MSAInput


class ExtendedMSAInput(MSAInput):
    """Extended input data for MSA analysis with optional tolerance.

    Inherits from MSAInput and adds optional tolerance field for
    %Tolerance calculations.

    Attributes:
        tolerance: Optional tolerance value for calculating %Tolerance
    """

    tolerance: Optional[float] = Field(
        default=None,
        gt=0,
        description="Tolerancia opcional para calcular %Tolerancia (debe ser positiva)",
        json_schema_extra={"example": 0.5},
    )


# -----------------------------------------------------------------------------
# ANOVA Models
# -----------------------------------------------------------------------------


class ANOVARow(BaseModel):
    """Single row in the ANOVA table.

    Attributes:
        source: Source of variation (Part, Operator, Part × Operator, Equipment, Total)
        df: Degrees of freedom
        ss: Sum of squares
        ms: Mean square (SS/df), None for Total
        f_value: F-statistic, None for Equipment and Total
        p_value: P-value, None for Equipment and Total
        is_significant: True if P < 0.05, None for Equipment and Total
    """

    source: str = Field(
        ...,
        description="Fuente de variación",
        json_schema_extra={"example": "Part"},
    )
    df: int = Field(
        ...,
        ge=0,
        description="Grados de libertad",
        json_schema_extra={"example": 4},
    )
    ss: float = Field(
        ...,
        ge=0,
        description="Suma de cuadrados",
        json_schema_extra={"example": 10.5},
    )
    ms: Optional[float] = Field(
        default=None,
        ge=0,
        description="Cuadrado medio (SS/df)",
        json_schema_extra={"example": 2.625},
    )
    f_value: Optional[float] = Field(
        default=None,
        ge=0,
        description="Estadístico F",
        json_schema_extra={"example": 15.3},
    )
    p_value: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="Valor P",
        json_schema_extra={"example": 0.001},
    )
    is_significant: Optional[bool] = Field(
        default=None,
        description="True si P < 0.05",
        json_schema_extra={"example": True},
    )


class ANOVAResult(BaseModel):
    """Complete ANOVA table result.

    Attributes:
        rows: List of ANOVA rows for each source
        significant_effects: List of source names where P < 0.05
    """

    rows: list[ANOVARow] = Field(
        ...,
        description="Filas de la tabla ANOVA",
    )
    significant_effects: list[str] = Field(
        default_factory=list,
        description="Lista de efectos significativos (P < 0.05)",
        json_schema_extra={"example": ["Part", "Operator"]},
    )


# -----------------------------------------------------------------------------
# Extended Variance Components Models
# -----------------------------------------------------------------------------


class ExtendedVarianceComponent(BaseModel):
    """Single variance component with all percentage metrics.

    Attributes:
        variance: σ² value
        std_dev: σ value (standard deviation)
        pct_contribution: Percentage of total variance (σ²/σ²_total × 100)
        pct_study: Percentage of study variation (σ/σ_total × 100)
        pct_tolerance: Percentage of tolerance (6σ/tolerance × 100), None if no tolerance
    """

    variance: float = Field(
        ...,
        ge=0,
        description="Varianza (σ²)",
        json_schema_extra={"example": 0.05},
    )
    std_dev: float = Field(
        ...,
        ge=0,
        description="Desviación estándar (σ)",
        json_schema_extra={"example": 0.2236},
    )
    pct_contribution: float = Field(
        ...,
        ge=0,
        description="Porcentaje de contribución (σ²/σ²_total × 100)",
        json_schema_extra={"example": 25.0},
    )
    pct_study: float = Field(
        ...,
        ge=0,
        description="Porcentaje de variación del estudio (σ/σ_total × 100)",
        json_schema_extra={"example": 30.0},
    )
    pct_tolerance: Optional[float] = Field(
        default=None,
        ge=0,
        description="Porcentaje de tolerancia (6σ/tolerancia × 100)",
        json_schema_extra={"example": 15.0},
    )


class ExtendedVarianceComponents(BaseModel):
    """Extended variance components with detailed breakdown.

    Attributes:
        total_grr: Total Gauge R&R (repeatability + reproducibility)
        repeatability: Equipment variation
        reproducibility: Operator variation (including interaction)
        part_to_part: Part-to-part variation
        total_variation: Total system variation
        interaction: Part × Operator interaction (optional, only if significant)
    """

    total_grr: ExtendedVarianceComponent = Field(
        ...,
        description="Gauge R&R total (repetibilidad + reproducibilidad)",
    )
    repeatability: ExtendedVarianceComponent = Field(
        ...,
        description="Variación del equipo (repetibilidad)",
    )
    reproducibility: ExtendedVarianceComponent = Field(
        ...,
        description="Variación del operador (reproducibilidad)",
    )
    part_to_part: ExtendedVarianceComponent = Field(
        ...,
        description="Variación parte a parte",
    )
    total_variation: ExtendedVarianceComponent = Field(
        ...,
        description="Variación total del sistema",
    )
    interaction: Optional[ExtendedVarianceComponent] = Field(
        default=None,
        description="Interacción Parte × Operador (solo si es significativa)",
    )


# -----------------------------------------------------------------------------
# Operator Metrics Models
# -----------------------------------------------------------------------------


class OperatorMetrics(BaseModel):
    """Per-operator statistical metrics.

    Attributes:
        operator_id: Operator identifier
        mean: Mean of all measurements by this operator
        std_dev: Standard deviation of this operator's measurements
        range_avg: Average range of this operator's measurements
        bias_estimate: Bias estimate (operator_mean - grand_mean)
        consistency_rank: Rank by consistency (1 = most consistent, lowest std_dev)
    """

    operator_id: str = Field(
        ...,
        description="Identificador del operador",
        json_schema_extra={"example": "Op1"},
    )
    mean: float = Field(
        ...,
        description="Media de las mediciones del operador",
        json_schema_extra={"example": 5.25},
    )
    std_dev: float = Field(
        ...,
        ge=0,
        description="Desviación estándar del operador",
        json_schema_extra={"example": 0.15},
    )
    range_avg: float = Field(
        ...,
        ge=0,
        description="Rango promedio del operador",
        json_schema_extra={"example": 0.35},
    )
    bias_estimate: float = Field(
        ...,
        description="Estimación del sesgo (media_operador - media_global)",
        json_schema_extra={"example": 0.02},
    )
    consistency_rank: int = Field(
        ...,
        ge=1,
        description="Ranking de consistencia (1 = más consistente)",
        json_schema_extra={"example": 1},
    )


# -----------------------------------------------------------------------------
# Stability Analysis Models
# -----------------------------------------------------------------------------


class StabilityAnalysis(BaseModel):
    """Stability analysis results for drift detection.

    Attributes:
        has_drift: True if significant drift detected
        drift_direction: 'increasing' or 'decreasing' if drift exists
        drift_magnitude: Slope of trend line if drift exists
        per_trial_means: Mean value for each trial
        trend_p_value: P-value from linear regression
    """

    has_drift: bool = Field(
        ...,
        description="True si se detectó deriva significativa",
        json_schema_extra={"example": False},
    )
    drift_direction: Optional[Literal["increasing", "decreasing"]] = Field(
        default=None,
        description="Dirección de la deriva ('increasing' o 'decreasing')",
        json_schema_extra={"example": "increasing"},
    )
    drift_magnitude: Optional[float] = Field(
        default=None,
        description="Magnitud de la deriva (pendiente de la línea de tendencia)",
        json_schema_extra={"example": 0.05},
    )
    per_trial_means: list[float] = Field(
        ...,
        description="Media por cada repetición/trial",
        json_schema_extra={"example": [5.0, 5.1, 4.9, 5.0, 5.05]},
    )
    trend_p_value: float = Field(
        ...,
        ge=0,
        le=1,
        description="Valor P de la regresión lineal",
        json_schema_extra={"example": 0.45},
    )


# -----------------------------------------------------------------------------
# Linearity and Bias Models
# -----------------------------------------------------------------------------


class LinearityBiasResult(BaseModel):
    """Linearity and bias analysis results.

    Attributes:
        bias: Overall bias (average of absolute biases)
        bias_percent: Bias as percentage of tolerance (if provided)
        linearity: Linearity value (range of biases)
        linearity_percent: Linearity as percentage of tolerance
        per_part_bias: Bias for each part
        regression_slope: Slope from bias vs reference regression
        regression_intercept: Intercept from regression
        reference_values_used: 'provided' or 'estimated' (part means)
    """

    bias: float = Field(
        ...,
        description="Sesgo general",
        json_schema_extra={"example": 0.05},
    )
    bias_percent: Optional[float] = Field(
        default=None,
        description="Sesgo como porcentaje de tolerancia",
        json_schema_extra={"example": 2.5},
    )
    linearity: float = Field(
        ...,
        description="Linealidad (rango de sesgos)",
        json_schema_extra={"example": 0.02},
    )
    linearity_percent: Optional[float] = Field(
        default=None,
        description="Linealidad como porcentaje de tolerancia",
        json_schema_extra={"example": 1.0},
    )
    per_part_bias: list[float] = Field(
        ...,
        description="Sesgo por cada parte",
        json_schema_extra={"example": [0.03, 0.05, 0.07, 0.04, 0.06]},
    )
    regression_slope: float = Field(
        ...,
        description="Pendiente de la regresión sesgo vs referencia",
        json_schema_extra={"example": 0.002},
    )
    regression_intercept: float = Field(
        ...,
        description="Intercepto de la regresión",
        json_schema_extra={"example": 0.01},
    )
    reference_values_used: Literal["provided", "estimated"] = Field(
        ...,
        description="'provided' si se dieron valores de referencia, 'estimated' si se usaron medias de partes",
        json_schema_extra={"example": "estimated"},
    )


# -----------------------------------------------------------------------------
# Resolution Analysis Models
# -----------------------------------------------------------------------------


class ResolutionAnalysis(BaseModel):
    """Measurement resolution analysis results.

    Attributes:
        measurement_resolution: Detected measurement resolution (smallest difference)
        resolution_ratio: resolution / (tolerance or 6σ)
        is_adequate: True if ratio < 0.1
        recommendation: Spanish recommendation text
    """

    measurement_resolution: float = Field(
        ...,
        ge=0,
        description="Resolución de medición detectada",
        json_schema_extra={"example": 0.01},
    )
    resolution_ratio: float = Field(
        ...,
        ge=0,
        description="Razón de resolución = resolución / (tolerancia o 6σ)",
        json_schema_extra={"example": 0.05},
    )
    is_adequate: bool = Field(
        ...,
        description="True si la razón < 0.1",
        json_schema_extra={"example": True},
    )
    recommendation: str = Field(
        ...,
        description="Recomendación en español",
        json_schema_extra={"example": "La resolución del instrumento es adecuada para este estudio MSA."},
    )


# -----------------------------------------------------------------------------
# Chart Data Models
# -----------------------------------------------------------------------------


class ComponentsChartData(BaseModel):
    """Data for variance components bar chart.

    Attributes:
        categories: Component names
        pct_contribution: %Contribution values
        pct_study_var: %Study Variation values
        pct_tolerance: %Tolerance values (None if no tolerance provided)
    """

    categories: list[str] = Field(
        ...,
        description="Nombres de los componentes",
        json_schema_extra={"example": ["Total GRR", "Repeatability", "Reproducibility", "Part-to-Part"]},
    )
    pct_contribution: list[float] = Field(
        ...,
        description="Valores de %Contribución",
        json_schema_extra={"example": [20.0, 15.0, 5.0, 80.0]},
    )
    pct_study_var: list[float] = Field(
        ...,
        description="Valores de %Variación del Estudio",
        json_schema_extra={"example": [25.0, 20.0, 10.0, 75.0]},
    )
    pct_tolerance: Optional[list[Optional[float]]] = Field(
        default=None,
        description="Valores de %Tolerancia (None si no se proporcionó tolerancia)",
        json_schema_extra={"example": [10.0, 8.0, 4.0, None]},
    )


class ControlChartData(BaseModel):
    """Data for R-chart or X-bar chart.

    Attributes:
        data_points: Chart data points
        center_line: Center line value (R-bar or X-double-bar)
        ucl: Upper control limit
        lcl: Lower control limit
        out_of_control_indices: Indices of out-of-control points
    """

    data_points: list[float] = Field(
        ...,
        description="Puntos de datos del gráfico",
        json_schema_extra={"example": [0.2, 0.3, 0.15, 0.25, 0.35]},
    )
    center_line: float = Field(
        ...,
        description="Línea central (R-bar o X-doble-barra)",
        json_schema_extra={"example": 0.25},
    )
    ucl: float = Field(
        ...,
        description="Límite de control superior",
        json_schema_extra={"example": 0.5},
    )
    lcl: float = Field(
        ...,
        description="Límite de control inferior",
        json_schema_extra={"example": 0.0},
    )
    out_of_control_indices: list[int] = Field(
        default_factory=list,
        description="Índices de puntos fuera de control",
        json_schema_extra={"example": [4]},
    )


class BoxPlotData(BaseModel):
    """Data for box plot visualization (by part or by operator).

    Attributes:
        labels: Part or operator labels
        min_values: Minimum values
        q1_values: First quartile values
        median_values: Median values
        q3_values: Third quartile values
        max_values: Maximum values
        mean_values: Mean values
    """

    labels: list[str] = Field(
        ...,
        description="Etiquetas (partes u operadores)",
        json_schema_extra={"example": ["Part1", "Part2", "Part3"]},
    )
    min_values: list[float] = Field(
        ...,
        description="Valores mínimos",
        json_schema_extra={"example": [4.8, 5.7, 6.5]},
    )
    q1_values: list[float] = Field(
        ...,
        description="Valores del primer cuartil",
        json_schema_extra={"example": [4.9, 5.8, 6.6]},
    )
    median_values: list[float] = Field(
        ...,
        description="Valores de la mediana",
        json_schema_extra={"example": [5.0, 5.9, 6.7]},
    )
    q3_values: list[float] = Field(
        ...,
        description="Valores del tercer cuartil",
        json_schema_extra={"example": [5.1, 6.0, 6.8]},
    )
    max_values: list[float] = Field(
        ...,
        description="Valores máximos",
        json_schema_extra={"example": [5.2, 6.1, 6.9]},
    )
    mean_values: list[float] = Field(
        ...,
        description="Valores de la media",
        json_schema_extra={"example": [5.0, 5.9, 6.7]},
    )


class InteractionPlotData(BaseModel):
    """Data for interaction plot (parts × operators).

    Attributes:
        parts: Part labels
        operators: Operator labels
        means_matrix: 2D array [operator][part] of means
    """

    parts: list[str] = Field(
        ...,
        description="Etiquetas de partes",
        json_schema_extra={"example": ["P1", "P2", "P3"]},
    )
    operators: list[str] = Field(
        ...,
        description="Etiquetas de operadores",
        json_schema_extra={"example": ["Op1", "Op2"]},
    )
    means_matrix: list[list[float]] = Field(
        ...,
        description="Matriz 2D [operador][parte] de medias",
        json_schema_extra={"example": [[5.0, 5.5, 6.0], [5.1, 5.4, 6.1]]},
    )


class ChartData(BaseModel):
    """Complete chart data for all visualization types.

    Attributes:
        components_chart: Variance components bar chart data
        r_chart: R-chart (range chart) data
        xbar_chart: X-bar chart data
        by_part: Box plot data by part
        by_operator: Box plot data by operator
        interaction_plot: Interaction plot data
    """

    components_chart: ComponentsChartData = Field(
        ...,
        description="Datos para gráfico de componentes de varianza",
    )
    r_chart: ControlChartData = Field(
        ...,
        description="Datos para gráfico R (rangos)",
    )
    xbar_chart: ControlChartData = Field(
        ...,
        description="Datos para gráfico X-barra",
    )
    by_part: BoxPlotData = Field(
        ...,
        description="Datos de diagrama de caja por parte",
    )
    by_operator: BoxPlotData = Field(
        ...,
        description="Datos de diagrama de caja por operador",
    )
    interaction_plot: InteractionPlotData = Field(
        ...,
        description="Datos para gráfico de interacción",
    )


# -----------------------------------------------------------------------------
# Complete Extended MSA Result
# -----------------------------------------------------------------------------


class ExtendedMSAResult(BaseModel):
    """Complete result of extended MSA analysis.

    Includes all metrics from the basic MSA result plus detailed ANOVA,
    extended variance components, operator metrics, optional stability,
    linearity/bias, resolution analyses, and chart data.

    Attributes:
        grr_percent: Total Gauge R&R as percentage of total variation
        repeatability_percent: Equipment variation as percentage
        reproducibility_percent: Operator variation as percentage
        part_variation_percent: Part-to-part variation as percentage
        ndc: Number of distinct categories
        category: Classification (excellent, marginal, unacceptable)
        anova: Complete ANOVA table
        variance_components: Extended variance components with all percentages
        operator_metrics: Per-operator statistical metrics
        reference_operator: ID of the most consistent operator
        charts: Data for all chart types
        stability: Stability analysis (optional)
        linearity_bias: Linearity and bias analysis (optional)
        resolution: Resolution analysis (optional, requires tolerance)
    """

    # Basic metrics (same as MSAResult)
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

    # Extended analysis components
    anova: ANOVAResult = Field(
        ...,
        description="Tabla ANOVA completa",
    )
    variance_components: ExtendedVarianceComponents = Field(
        ...,
        description="Componentes de varianza extendidos",
    )
    operator_metrics: list[OperatorMetrics] = Field(
        ...,
        description="Métricas estadísticas por operador",
    )
    reference_operator: str = Field(
        ...,
        description="ID del operador más consistente (referencia)",
        json_schema_extra={"example": "Op1"},
    )
    charts: ChartData = Field(
        ...,
        description="Datos para todos los gráficos",
    )

    # Optional analyses
    stability: Optional[StabilityAnalysis] = Field(
        default=None,
        description="Análisis de estabilidad (opcional)",
    )
    linearity_bias: Optional[LinearityBiasResult] = Field(
        default=None,
        description="Análisis de linealidad y sesgo (opcional)",
    )
    resolution: Optional[ResolutionAnalysis] = Field(
        default=None,
        description="Análisis de resolución (requiere tolerancia)",
    )
