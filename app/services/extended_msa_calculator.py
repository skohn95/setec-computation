"""Extended MSA (Measurement System Analysis) Calculator Service.

Implements complete ANOVA-based Gauge R&R calculations with extended metrics
including detailed variance components, operator analysis, stability analysis,
linearity/bias analysis, resolution analysis, and chart data generation.
"""

import logging
import math
from typing import Literal, Optional

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

from app.models.extended_msa import (
    ExtendedMSAInput,
    ExtendedMSAResult,
    ANOVARow,
    ANOVAResult,
    ExtendedVarianceComponent,
    ExtendedVarianceComponents,
    OperatorMetrics,
    StabilityAnalysis,
    LinearityBiasResult,
    ResolutionAnalysis,
    ComponentsChartData,
    ControlChartData,
    BoxPlotData,
    InteractionPlotData,
    ChartData,
)

# Control chart constants (for subgroup sizes 2-10)
# A2, D3, D4 from standard control chart tables
CONTROL_CHART_CONSTANTS = {
    2: {"A2": 1.880, "D3": 0.000, "D4": 3.267},
    3: {"A2": 1.023, "D3": 0.000, "D4": 2.574},
    4: {"A2": 0.729, "D3": 0.000, "D4": 2.282},
    5: {"A2": 0.577, "D3": 0.000, "D4": 2.114},
    6: {"A2": 0.483, "D3": 0.000, "D4": 2.004},
    7: {"A2": 0.419, "D3": 0.076, "D4": 1.924},
    8: {"A2": 0.373, "D3": 0.136, "D4": 1.864},
    9: {"A2": 0.337, "D3": 0.184, "D4": 1.816},
    10: {"A2": 0.308, "D3": 0.223, "D4": 1.777},
}


class ExtendedMSACalculator:
    """Calculator for extended Gauge R&R analysis using ANOVA method.

    Implements the AIAG MSA Manual methodology with extended metrics for
    detailed analysis of measurement system variation.
    """

    def calculate(self, data: ExtendedMSAInput) -> ExtendedMSAResult:
        """Perform complete extended Gauge R&R analysis.

        Args:
            data: ExtendedMSAInput with parts, operators, trials, measurements, and optional tolerance

        Returns:
            ExtendedMSAResult with all metrics and analyses
        """
        # Convert to numpy array: shape (parts, operators, trials)
        measurements = np.array(data.measurements)

        n_parts = len(data.parts)
        n_operators = len(data.operators)
        n_trials = data.trials
        tolerance = data.tolerance

        # Calculate ANOVA
        anova_result, raw_anova = self._calculate_anova(
            measurements, n_parts, n_operators, n_trials
        )

        # Calculate variance components
        variance_data = self._calculate_variance_components_raw(
            raw_anova, n_parts, n_operators, n_trials
        )

        # Calculate extended variance components with percentages
        variance_components = self._build_extended_variance_components(
            variance_data, tolerance
        )

        # Calculate basic metrics
        grr_percent = variance_components.total_grr.pct_study
        repeatability_percent = variance_components.repeatability.pct_study
        reproducibility_percent = variance_components.reproducibility.pct_study
        part_variation_percent = variance_components.part_to_part.pct_study

        # Calculate NDC
        sigma_part = variance_components.part_to_part.std_dev
        sigma_grr = variance_components.total_grr.std_dev
        ndc = self._calculate_ndc(sigma_part, sigma_grr)

        # Classify
        category = self._classify_grr(grr_percent)

        # Calculate operator metrics
        operator_metrics, reference_operator = self._calculate_operator_metrics(
            measurements, data.operators, n_parts, n_trials
        )

        # Calculate stability analysis
        stability = self._calculate_stability(measurements, n_trials)

        # Calculate linearity and bias
        linearity_bias = self._calculate_linearity_bias(
            measurements, data.parts, tolerance
        )

        # Calculate resolution analysis (only if tolerance provided)
        resolution = None
        if tolerance is not None:
            resolution = self._calculate_resolution(
                measurements, tolerance, variance_components.total_variation.std_dev
            )

        # Generate chart data
        charts = self._generate_chart_data(
            measurements, data.parts, data.operators, n_trials,
            variance_components, tolerance
        )

        return ExtendedMSAResult(
            grr_percent=round(grr_percent, 2),
            repeatability_percent=round(repeatability_percent, 2),
            reproducibility_percent=round(reproducibility_percent, 2),
            part_variation_percent=round(part_variation_percent, 2),
            ndc=ndc,
            category=category,
            anova=anova_result,
            variance_components=variance_components,
            operator_metrics=operator_metrics,
            reference_operator=reference_operator,
            charts=charts,
            stability=stability,
            linearity_bias=linearity_bias,
            resolution=resolution,
        )

    # =========================================================================
    # ANOVA Calculations (Task 2)
    # =========================================================================

    def _calculate_anova(
        self,
        measurements: np.ndarray,
        n_parts: int,
        n_operators: int,
        n_trials: int,
    ) -> tuple[ANOVAResult, dict]:
        """Calculate complete ANOVA table.

        Returns:
            Tuple of (ANOVAResult, raw_anova_dict for variance component calculation)
        """
        grand_mean = measurements.mean()

        # Calculate means at different levels
        part_means = measurements.mean(axis=(1, 2))  # Mean across operators and trials
        operator_means = measurements.mean(axis=(0, 2))  # Mean across parts and trials
        cell_means = measurements.mean(axis=2)  # Mean for each part-operator combination

        # Calculate Sum of Squares
        ss_total = np.sum((measurements - grand_mean) ** 2)

        ss_part = n_operators * n_trials * np.sum((part_means - grand_mean) ** 2)

        if n_operators > 1:
            ss_operator = n_parts * n_trials * np.sum((operator_means - grand_mean) ** 2)
        else:
            ss_operator = 0.0

        # Interaction: Part × Operator
        if n_operators > 1:
            expected_cell_means = (
                part_means[:, np.newaxis]
                + operator_means[np.newaxis, :]
                - grand_mean
            )
            ss_interaction = n_trials * np.sum((cell_means - expected_cell_means) ** 2)
        else:
            ss_interaction = 0.0

        # Equipment (within / error)
        ss_equipment = np.sum((measurements - cell_means[:, :, np.newaxis]) ** 2)

        # Degrees of freedom
        df_part = n_parts - 1
        df_operator = n_operators - 1 if n_operators > 1 else 0
        df_interaction = df_part * df_operator if n_operators > 1 else 0
        df_equipment = n_parts * n_operators * (n_trials - 1)
        df_total = n_parts * n_operators * n_trials - 1

        # Mean Squares
        ms_part = ss_part / df_part if df_part > 0 else 0
        ms_operator = ss_operator / df_operator if df_operator > 0 else 0
        ms_interaction = ss_interaction / df_interaction if df_interaction > 0 else 0
        ms_equipment = ss_equipment / df_equipment if df_equipment > 0 else 0

        # F-statistics and P-values
        rows = []
        significant_effects = []

        # Part row
        if df_interaction > 0 and ms_interaction > 0:
            f_part = ms_part / ms_interaction
            df_denom = df_interaction
        else:
            f_part = ms_part / ms_equipment if ms_equipment > 0 else 0
            df_denom = df_equipment

        p_part = stats.f.sf(f_part, df_part, df_denom) if f_part > 0 and df_denom > 0 else 1.0
        is_sig_part = p_part < 0.05

        rows.append(ANOVARow(
            source="Part",
            df=df_part,
            ss=round(ss_part, 6),
            ms=round(ms_part, 6),
            f_value=round(f_part, 4) if f_part > 0 else None,
            p_value=round(p_part, 6) if f_part > 0 else None,
            is_significant=is_sig_part if f_part > 0 else None,
        ))
        if is_sig_part:
            significant_effects.append("Part")

        # Operator row
        if n_operators > 1:
            if df_interaction > 0 and ms_interaction > 0:
                f_operator = ms_operator / ms_interaction
                df_denom_op = df_interaction
            else:
                f_operator = ms_operator / ms_equipment if ms_equipment > 0 else 0
                df_denom_op = df_equipment

            p_operator = stats.f.sf(f_operator, df_operator, df_denom_op) if f_operator > 0 and df_denom_op > 0 else 1.0
            is_sig_operator = p_operator < 0.05

            rows.append(ANOVARow(
                source="Operator",
                df=df_operator,
                ss=round(ss_operator, 6),
                ms=round(ms_operator, 6),
                f_value=round(f_operator, 4) if f_operator > 0 else None,
                p_value=round(p_operator, 6) if f_operator > 0 else None,
                is_significant=is_sig_operator if f_operator > 0 else None,
            ))
            if is_sig_operator:
                significant_effects.append("Operator")
        else:
            rows.append(ANOVARow(
                source="Operator",
                df=0,
                ss=0.0,
                ms=None,
                f_value=None,
                p_value=None,
                is_significant=None,
            ))

        # Interaction row
        if n_operators > 1 and df_interaction > 0:
            f_interaction = ms_interaction / ms_equipment if ms_equipment > 0 else 0
            p_interaction = stats.f.sf(f_interaction, df_interaction, df_equipment) if f_interaction > 0 and df_equipment > 0 else 1.0
            is_sig_interaction = p_interaction < 0.05

            rows.append(ANOVARow(
                source="Part × Operator",
                df=df_interaction,
                ss=round(ss_interaction, 6),
                ms=round(ms_interaction, 6),
                f_value=round(f_interaction, 4) if f_interaction > 0 else None,
                p_value=round(p_interaction, 6) if f_interaction > 0 else None,
                is_significant=is_sig_interaction if f_interaction > 0 else None,
            ))
            if is_sig_interaction:
                significant_effects.append("Part × Operator")
        else:
            rows.append(ANOVARow(
                source="Part × Operator",
                df=0,
                ss=0.0,
                ms=None,
                f_value=None,
                p_value=None,
                is_significant=None,
            ))

        # Equipment row (no F-test for error term)
        rows.append(ANOVARow(
            source="Equipment",
            df=df_equipment,
            ss=round(ss_equipment, 6),
            ms=round(ms_equipment, 6),
            f_value=None,
            p_value=None,
            is_significant=None,
        ))

        # Total row
        rows.append(ANOVARow(
            source="Total",
            df=df_total,
            ss=round(ss_total, 6),
            ms=None,
            f_value=None,
            p_value=None,
            is_significant=None,
        ))

        # Raw ANOVA data for variance component calculation
        raw_anova = {
            "ss_part": ss_part,
            "ss_operator": ss_operator,
            "ss_interaction": ss_interaction,
            "ss_equipment": ss_equipment,
            "ss_total": ss_total,
            "ms_part": ms_part,
            "ms_operator": ms_operator,
            "ms_interaction": ms_interaction,
            "ms_equipment": ms_equipment,
            "df_part": df_part,
            "df_operator": df_operator,
            "df_interaction": df_interaction,
            "df_equipment": df_equipment,
        }

        return ANOVAResult(rows=rows, significant_effects=significant_effects), raw_anova

    # =========================================================================
    # Variance Components Calculations (Task 3)
    # =========================================================================

    def _calculate_variance_components_raw(
        self,
        raw_anova: dict,
        n_parts: int,
        n_operators: int,
        n_trials: int,
    ) -> dict:
        """Calculate raw variance components from ANOVA results."""
        ms_part = raw_anova["ms_part"]
        ms_operator = raw_anova["ms_operator"]
        ms_interaction = raw_anova["ms_interaction"]
        ms_equipment = raw_anova["ms_equipment"]

        # σ²_Equipment (repeatability)
        var_repeatability = ms_equipment

        # Handle single operator case
        if n_operators == 1:
            var_reproducibility = 0.0
            var_interaction = 0.0
            var_part = max(0, (ms_part - ms_equipment) / n_trials)
        else:
            # Check if interaction is significant (use α=0.25 for pooling decision)
            df_interaction = raw_anova["df_interaction"]
            df_equipment = raw_anova["df_equipment"]

            if df_interaction > 0 and df_equipment > 0 and ms_equipment > 0:
                f_interaction = ms_interaction / ms_equipment
                p_value = stats.f.sf(f_interaction, df_interaction, df_equipment)
                interaction_significant = p_value < 0.25
            else:
                interaction_significant = False

            if interaction_significant:
                # Include interaction
                var_interaction = max(0, (ms_interaction - ms_equipment) / n_trials)
                var_reproducibility = max(0, (ms_operator - ms_interaction) / (n_parts * n_trials))
                var_part = max(0, (ms_part - ms_interaction) / (n_operators * n_trials))
                # Add interaction to reproducibility
                var_reproducibility = var_reproducibility + var_interaction
            else:
                # Pool interaction with error
                var_interaction = 0.0
                var_reproducibility = max(0, (ms_operator - ms_equipment) / (n_parts * n_trials))
                var_part = max(0, (ms_part - ms_equipment) / (n_operators * n_trials))

        var_grr = var_repeatability + var_reproducibility
        var_total = var_part + var_grr

        return {
            "repeatability": var_repeatability,
            "reproducibility": var_reproducibility,
            "interaction": var_interaction,
            "part": var_part,
            "grr": var_grr,
            "total": var_total,
        }

    def _build_extended_variance_components(
        self,
        variance_data: dict,
        tolerance: Optional[float],
    ) -> ExtendedVarianceComponents:
        """Build ExtendedVarianceComponents with all percentage metrics."""
        var_total = variance_data["total"]
        std_total = math.sqrt(var_total) if var_total > 0 else 0

        def build_component(var: float, name: str) -> ExtendedVarianceComponent:
            std = math.sqrt(var) if var > 0 else 0

            if var_total > 0:
                pct_contribution = (var / var_total) * 100
            else:
                pct_contribution = 0.0

            if std_total > 0:
                pct_study = (std / std_total) * 100
            else:
                pct_study = 0.0

            pct_tolerance = None
            if tolerance is not None and tolerance > 0:
                # Only calculate for GRR-related components
                if name in ["grr", "repeatability", "reproducibility"]:
                    pct_tolerance = (6 * std / tolerance) * 100

            return ExtendedVarianceComponent(
                variance=round(var, 8),
                std_dev=round(std, 8),
                pct_contribution=round(pct_contribution, 2),
                pct_study=round(pct_study, 2),
                pct_tolerance=round(pct_tolerance, 2) if pct_tolerance is not None else None,
            )

        interaction_component = None
        if variance_data["interaction"] > 0:
            interaction_component = build_component(variance_data["interaction"], "interaction")

        return ExtendedVarianceComponents(
            total_grr=build_component(variance_data["grr"], "grr"),
            repeatability=build_component(variance_data["repeatability"], "repeatability"),
            reproducibility=build_component(variance_data["reproducibility"], "reproducibility"),
            part_to_part=build_component(variance_data["part"], "part"),
            total_variation=build_component(variance_data["total"], "total"),
            interaction=interaction_component,
        )

    # =========================================================================
    # Operator Metrics (Task 4)
    # =========================================================================

    def _calculate_operator_metrics(
        self,
        measurements: np.ndarray,
        operators: list[str],
        n_parts: int,
        n_trials: int,
    ) -> tuple[list[OperatorMetrics], str]:
        """Calculate per-operator metrics and identify reference operator."""
        grand_mean = measurements.mean()
        n_operators = len(operators)

        metrics_list = []

        for j, op_id in enumerate(operators):
            # Get all measurements for this operator: shape (n_parts, n_trials)
            op_measurements = measurements[:, j, :]

            # Mean
            op_mean = op_measurements.mean()

            # Std dev
            op_std = op_measurements.std(ddof=1) if op_measurements.size > 1 else 0.0

            # Range average (average range within each part)
            ranges = []
            for i in range(n_parts):
                part_op_data = measurements[i, j, :]
                ranges.append(part_op_data.max() - part_op_data.min())
            range_avg = np.mean(ranges)

            # Bias estimate
            bias_estimate = op_mean - grand_mean

            metrics_list.append({
                "operator_id": op_id,
                "mean": op_mean,
                "std_dev": op_std,
                "range_avg": range_avg,
                "bias_estimate": bias_estimate,
            })

        # Sort by std_dev (then by range_avg for tiebreaker) to assign ranks
        sorted_metrics = sorted(metrics_list, key=lambda m: (m["std_dev"], m["range_avg"]))

        # Assign ranks
        result = []
        for rank, m in enumerate(sorted_metrics, start=1):
            result.append(OperatorMetrics(
                operator_id=m["operator_id"],
                mean=round(m["mean"], 6),
                std_dev=round(m["std_dev"], 6),
                range_avg=round(m["range_avg"], 6),
                bias_estimate=round(m["bias_estimate"], 6),
                consistency_rank=rank,
            ))

        # Reference operator is rank 1
        reference_operator = sorted_metrics[0]["operator_id"]

        # Return in original order for consistency
        ordered_result = []
        for op_id in operators:
            ordered_result.append(next(m for m in result if m.operator_id == op_id))

        return ordered_result, reference_operator

    # =========================================================================
    # Stability Analysis (Task 5)
    # =========================================================================

    def _calculate_stability(
        self,
        measurements: np.ndarray,
        n_trials: int,
    ) -> StabilityAnalysis:
        """Analyze stability by checking for drift across trials."""
        # Calculate mean per trial (across all parts and operators)
        # measurements shape: (n_parts, n_operators, n_trials)
        per_trial_means = []
        for t in range(n_trials):
            trial_mean = measurements[:, :, t].mean()
            per_trial_means.append(float(trial_mean))

        # Perform linear regression to detect trend
        x = np.arange(n_trials)
        y = np.array(per_trial_means)

        if n_trials > 2:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        else:
            # With only 2 points, use simple slope calculation
            slope = (y[-1] - y[0]) / (n_trials - 1) if n_trials > 1 else 0
            p_value = 1.0  # Cannot determine significance with 2 points

        has_drift = p_value < 0.05 and abs(slope) > 1e-10

        drift_direction = None
        drift_magnitude = None
        if has_drift:
            drift_direction = "increasing" if slope > 0 else "decreasing"
            drift_magnitude = float(slope)

        return StabilityAnalysis(
            has_drift=has_drift,
            drift_direction=drift_direction,
            drift_magnitude=drift_magnitude,
            per_trial_means=[round(m, 6) for m in per_trial_means],
            trend_p_value=round(float(p_value), 6),
        )

    # =========================================================================
    # Linearity & Bias Analysis (Task 6)
    # =========================================================================

    def _calculate_linearity_bias(
        self,
        measurements: np.ndarray,
        parts: list[str],
        tolerance: Optional[float],
    ) -> LinearityBiasResult:
        """Calculate linearity and bias analysis.

        When true reference values are not provided, this calculates:
        - Per-part bias: deviation of each part's mean from the grand mean
        - Overall bias: average of absolute per-part biases
        - Linearity: variation in bias across the measurement range

        Note: Without true reference standards, this provides an estimate
        of relative bias between parts, not absolute measurement accuracy.
        """
        n_parts = len(parts)
        grand_mean = measurements.mean()

        # Calculate mean for each part
        part_means = measurements.mean(axis=(1, 2))

        # Per-part bias: deviation of part mean from grand mean
        # This measures relative bias between parts
        per_part_bias = []
        for i in range(n_parts):
            # Bias = part mean - grand mean (relative to overall center)
            bias = float(part_means[i] - grand_mean)
            per_part_bias.append(bias)

        # Overall bias (average of absolute biases)
        overall_bias = np.mean(np.abs(per_part_bias))

        # Linearity via regression of bias vs reference
        part_range = part_means.max() - part_means.min()

        if n_parts > 1 and part_range > 1e-10:
            # Can only calculate linearity if part means have variation
            slope, intercept, r_value, p_value, std_err = stats.linregress(part_means, per_part_bias)
            linearity = abs(slope) * part_range
        else:
            # With single part or identical part means, linearity is undefined (set to 0)
            slope = 0.0
            intercept = per_part_bias[0] if per_part_bias else 0.0
            linearity = 0.0

        # Percentage values when tolerance provided
        bias_percent = None
        linearity_percent = None
        if tolerance is not None and tolerance > 0:
            bias_percent = (overall_bias / tolerance) * 100
            linearity_percent = (linearity / tolerance) * 100

        return LinearityBiasResult(
            bias=round(overall_bias, 6),
            bias_percent=round(bias_percent, 2) if bias_percent is not None else None,
            linearity=round(linearity, 6),
            linearity_percent=round(linearity_percent, 2) if linearity_percent is not None else None,
            per_part_bias=[round(b, 6) for b in per_part_bias],
            regression_slope=round(float(slope), 6),
            regression_intercept=round(float(intercept), 6),
            reference_values_used="estimated",
        )

    # =========================================================================
    # Resolution Analysis (Task 7)
    # =========================================================================

    def _calculate_resolution(
        self,
        measurements: np.ndarray,
        tolerance: float,
        sigma_total: float,
    ) -> ResolutionAnalysis:
        """Calculate measurement resolution analysis."""
        # Detect measurement resolution from data
        # Resolution = smallest difference between distinct values
        flat_values = measurements.flatten()
        unique_values = np.unique(flat_values)

        if len(unique_values) > 1:
            diffs = np.diff(np.sort(unique_values))
            # Filter out very small differences (floating point artifacts)
            significant_diffs = diffs[diffs > 1e-10]
            if len(significant_diffs) > 0:
                measurement_resolution = float(np.min(significant_diffs))
            else:
                measurement_resolution = float(np.min(diffs)) if len(diffs) > 0 else 0.0
        else:
            measurement_resolution = 0.0

        # Resolution ratio
        # Use tolerance if provided, otherwise use 6σ_total
        denominator = tolerance if tolerance else (6 * sigma_total)
        if denominator > 0:
            resolution_ratio = measurement_resolution / denominator
        else:
            resolution_ratio = 0.0

        is_adequate = resolution_ratio < 0.1

        # Spanish recommendation
        if is_adequate:
            recommendation = "La resolución del instrumento es adecuada para este estudio MSA."
        elif resolution_ratio < 0.25:
            recommendation = "La resolución del instrumento es marginalmente aceptable. Considere usar un instrumento con mejor resolución."
        else:
            recommendation = "La resolución del instrumento es insuficiente. Se recomienda un instrumento con mejor resolución para obtener resultados confiables."

        return ResolutionAnalysis(
            measurement_resolution=round(measurement_resolution, 6),
            resolution_ratio=round(resolution_ratio, 4),
            is_adequate=is_adequate,
            recommendation=recommendation,
        )

    # =========================================================================
    # Chart Data Generation (Task 8)
    # =========================================================================

    def _generate_chart_data(
        self,
        measurements: np.ndarray,
        parts: list[str],
        operators: list[str],
        n_trials: int,
        variance_components: ExtendedVarianceComponents,
        tolerance: Optional[float],
    ) -> ChartData:
        """Generate data for all chart types."""
        n_parts = len(parts)
        n_operators = len(operators)

        # Components chart
        components_chart = self._generate_components_chart(variance_components, tolerance)

        # R-chart and X-bar chart
        r_chart, xbar_chart = self._generate_control_charts(
            measurements, n_parts, n_operators, n_trials
        )

        # Box plots
        by_part = self._generate_box_plot_by_part(measurements, parts)
        by_operator = self._generate_box_plot_by_operator(measurements, operators)

        # Interaction plot
        interaction_plot = self._generate_interaction_plot(measurements, parts, operators)

        return ChartData(
            components_chart=components_chart,
            r_chart=r_chart,
            xbar_chart=xbar_chart,
            by_part=by_part,
            by_operator=by_operator,
            interaction_plot=interaction_plot,
        )

    def _generate_components_chart(
        self,
        vc: ExtendedVarianceComponents,
        tolerance: Optional[float],
    ) -> ComponentsChartData:
        """Generate variance components bar chart data."""
        categories = ["Total GRR", "Repeatability", "Reproducibility", "Part-to-Part"]
        pct_contribution = [
            vc.total_grr.pct_contribution,
            vc.repeatability.pct_contribution,
            vc.reproducibility.pct_contribution,
            vc.part_to_part.pct_contribution,
        ]
        pct_study_var = [
            vc.total_grr.pct_study,
            vc.repeatability.pct_study,
            vc.reproducibility.pct_study,
            vc.part_to_part.pct_study,
        ]

        pct_tolerance = None
        if tolerance is not None:
            pct_tolerance = [
                vc.total_grr.pct_tolerance,
                vc.repeatability.pct_tolerance,
                vc.reproducibility.pct_tolerance,
                None,  # Part-to-part doesn't have %Tolerance
            ]

        return ComponentsChartData(
            categories=categories,
            pct_contribution=pct_contribution,
            pct_study_var=pct_study_var,
            pct_tolerance=pct_tolerance,
        )

    def _generate_control_charts(
        self,
        measurements: np.ndarray,
        n_parts: int,
        n_operators: int,
        n_trials: int,
    ) -> tuple[ControlChartData, ControlChartData]:
        """Generate R-chart and X-bar chart data."""
        # Each subgroup is one part-operator combination
        # Subgroup size = n_trials

        subgroup_size = min(n_trials, 10)  # Limit to available constants
        if subgroup_size < 2:
            subgroup_size = 2

        # Warn if trials exceed available control chart constants
        if n_trials > 10:
            logger.warning(
                f"Control chart constants only available for n≤10. "
                f"Using n=10 constants for {n_trials} trials. "
                f"Control limits may be less accurate."
            )

        constants = CONTROL_CHART_CONSTANTS.get(subgroup_size, CONTROL_CHART_CONSTANTS[2])
        A2 = constants["A2"]
        D3 = constants["D3"]
        D4 = constants["D4"]

        # Calculate ranges and means for each subgroup
        ranges = []
        means = []

        for i in range(n_parts):
            for j in range(n_operators):
                subgroup = measurements[i, j, :subgroup_size]
                ranges.append(float(subgroup.max() - subgroup.min()))
                means.append(float(subgroup.mean()))

        ranges = np.array(ranges)
        means = np.array(means)

        # R-chart
        r_bar = ranges.mean()
        ucl_r = D4 * r_bar
        lcl_r = D3 * r_bar  # Often 0 for small subgroups

        r_ooc = [i for i, r in enumerate(ranges) if r > ucl_r or r < lcl_r]

        r_chart = ControlChartData(
            data_points=[round(r, 6) for r in ranges],
            center_line=round(r_bar, 6),
            ucl=round(ucl_r, 6),
            lcl=round(lcl_r, 6),
            out_of_control_indices=r_ooc,
        )

        # X-bar chart
        x_double_bar = means.mean()
        ucl_x = x_double_bar + A2 * r_bar
        lcl_x = x_double_bar - A2 * r_bar

        xbar_ooc = [i for i, m in enumerate(means) if m > ucl_x or m < lcl_x]

        xbar_chart = ControlChartData(
            data_points=[round(m, 6) for m in means],
            center_line=round(x_double_bar, 6),
            ucl=round(ucl_x, 6),
            lcl=round(lcl_x, 6),
            out_of_control_indices=xbar_ooc,
        )

        return r_chart, xbar_chart

    def _generate_box_plot_by_part(
        self,
        measurements: np.ndarray,
        parts: list[str],
    ) -> BoxPlotData:
        """Generate box plot data grouped by part."""
        labels = parts
        min_vals = []
        q1_vals = []
        median_vals = []
        q3_vals = []
        max_vals = []
        mean_vals = []

        for i in range(len(parts)):
            part_data = measurements[i, :, :].flatten()
            min_vals.append(float(part_data.min()))
            q1_vals.append(float(np.percentile(part_data, 25)))
            median_vals.append(float(np.median(part_data)))
            q3_vals.append(float(np.percentile(part_data, 75)))
            max_vals.append(float(part_data.max()))
            mean_vals.append(float(part_data.mean()))

        return BoxPlotData(
            labels=labels,
            min_values=[round(v, 6) for v in min_vals],
            q1_values=[round(v, 6) for v in q1_vals],
            median_values=[round(v, 6) for v in median_vals],
            q3_values=[round(v, 6) for v in q3_vals],
            max_values=[round(v, 6) for v in max_vals],
            mean_values=[round(v, 6) for v in mean_vals],
        )

    def _generate_box_plot_by_operator(
        self,
        measurements: np.ndarray,
        operators: list[str],
    ) -> BoxPlotData:
        """Generate box plot data grouped by operator."""
        labels = operators
        min_vals = []
        q1_vals = []
        median_vals = []
        q3_vals = []
        max_vals = []
        mean_vals = []

        for j in range(len(operators)):
            op_data = measurements[:, j, :].flatten()
            min_vals.append(float(op_data.min()))
            q1_vals.append(float(np.percentile(op_data, 25)))
            median_vals.append(float(np.median(op_data)))
            q3_vals.append(float(np.percentile(op_data, 75)))
            max_vals.append(float(op_data.max()))
            mean_vals.append(float(op_data.mean()))

        return BoxPlotData(
            labels=labels,
            min_values=[round(v, 6) for v in min_vals],
            q1_values=[round(v, 6) for v in q1_vals],
            median_values=[round(v, 6) for v in median_vals],
            q3_values=[round(v, 6) for v in q3_vals],
            max_values=[round(v, 6) for v in max_vals],
            mean_values=[round(v, 6) for v in mean_vals],
        )

    def _generate_interaction_plot(
        self,
        measurements: np.ndarray,
        parts: list[str],
        operators: list[str],
    ) -> InteractionPlotData:
        """Generate interaction plot data (parts × operators means)."""
        # means_matrix[operator][part]
        means_matrix = []

        for j in range(len(operators)):
            operator_row = []
            for i in range(len(parts)):
                cell_mean = measurements[i, j, :].mean()
                operator_row.append(round(float(cell_mean), 6))
            means_matrix.append(operator_row)

        return InteractionPlotData(
            parts=parts,
            operators=operators,
            means_matrix=means_matrix,
        )

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _calculate_ndc(self, sigma_part: float, sigma_grr: float) -> int:
        """Calculate Number of Distinct Categories (ndc)."""
        if sigma_grr <= 0:
            if sigma_part > 0:
                return 999  # Perfect measurement system
            else:
                return 1  # No variation
        ndc = int(1.41 * sigma_part / sigma_grr)
        return max(0, ndc)

    def _classify_grr(self, grr_percent: float) -> Literal["excellent", "marginal", "unacceptable"]:
        """Classify measurement system based on %GRR."""
        if grr_percent < 10.0:
            return "excellent"
        elif grr_percent <= 30.0:
            return "marginal"
        else:
            return "unacceptable"
