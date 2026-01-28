"""Unit tests for Extended MSA Pydantic models.

Tests for ExtendedMSAInput, ANOVAResult, ExtendedVarianceComponents,
OperatorMetrics, StabilityAnalysis, LinearityBiasResult, ResolutionAnalysis,
chart data models, and ExtendedMSAResult.
"""

import pytest
from pydantic import ValidationError

from app.models.extended_msa import (
    ExtendedMSAInput,
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
    ExtendedMSAResult,
)


class TestExtendedMSAInput:
    """Tests for ExtendedMSAInput model."""

    def test_valid_input_without_tolerance(self):
        """Test valid input without optional tolerance field."""
        data = ExtendedMSAInput(
            parts=["Part1", "Part2"],
            operators=["Op1"],
            trials=2,
            measurements=[
                [[2.0, 2.1]],
                [[3.0, 3.1]],
            ],
        )
        assert data.tolerance is None
        assert len(data.parts) == 2

    def test_valid_input_with_tolerance(self):
        """Test valid input with tolerance specified."""
        data = ExtendedMSAInput(
            parts=["Part1", "Part2"],
            operators=["Op1"],
            trials=2,
            measurements=[
                [[2.0, 2.1]],
                [[3.0, 3.1]],
            ],
            tolerance=0.5,
        )
        assert data.tolerance == 0.5

    def test_tolerance_must_be_positive(self):
        """Test tolerance must be positive when provided."""
        with pytest.raises(ValidationError):
            ExtendedMSAInput(
                parts=["Part1", "Part2"],
                operators=["Op1"],
                trials=2,
                measurements=[
                    [[2.0, 2.1]],
                    [[3.0, 3.1]],
                ],
                tolerance=-0.5,  # Invalid - negative
            )

    def test_measurements_dimension_mismatch_parts_rejected(self):
        """Test that mismatched parts dimension is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            ExtendedMSAInput(
                parts=["Part1", "Part2", "Part3"],  # 3 parts
                operators=["Op1"],
                trials=2,
                measurements=[
                    [[2.0, 2.1]],  # Only 2 parts in measurements
                    [[3.0, 3.1]],
                ],
            )
        assert "partes" in str(exc_info.value).lower()

    def test_measurements_dimension_mismatch_operators_rejected(self):
        """Test that mismatched operators dimension is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            ExtendedMSAInput(
                parts=["Part1", "Part2"],
                operators=["Op1", "Op2"],  # 2 operators
                trials=2,
                measurements=[
                    [[2.0, 2.1]],  # Only 1 operator per part
                    [[3.0, 3.1]],
                ],
            )
        assert "operador" in str(exc_info.value).lower()

    def test_measurements_dimension_mismatch_trials_rejected(self):
        """Test that mismatched trials dimension is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            ExtendedMSAInput(
                parts=["Part1", "Part2"],
                operators=["Op1"],
                trials=3,  # 3 trials declared
                measurements=[
                    [[2.0, 2.1]],  # Only 2 trials
                    [[3.0, 3.1]],
                ],
            )
        assert "repeticiones" in str(exc_info.value).lower() or "trials" in str(exc_info.value).lower()

    def test_minimum_two_parts_required(self):
        """Test that minimum 2 parts are required for MSA analysis."""
        with pytest.raises(ValidationError) as exc_info:
            ExtendedMSAInput(
                parts=["Part1"],  # Only 1 part - should fail
                operators=["Op1"],
                trials=2,
                measurements=[
                    [[2.0, 2.1]],
                ],
            )
        # Should fail due to min_length=2 constraint on parts
        error_str = str(exc_info.value).lower()
        assert "parts" in error_str or "list" in error_str or "2" in error_str

    def test_minimum_two_trials_required(self):
        """Test that minimum 2 trials are required for MSA analysis."""
        with pytest.raises(ValidationError) as exc_info:
            ExtendedMSAInput(
                parts=["Part1", "Part2"],
                operators=["Op1"],
                trials=1,  # Only 1 trial - should fail
                measurements=[
                    [[2.0]],
                    [[3.0]],
                ],
            )
        # Should fail due to ge=2 constraint on trials
        error_str = str(exc_info.value).lower()
        assert "trials" in error_str or "2" in error_str or "greater" in error_str


class TestANOVAModels:
    """Tests for ANOVA-related models."""

    def test_anova_row_valid(self):
        """Test valid ANOVA row."""
        row = ANOVARow(
            source="Part",
            df=4,
            ss=10.5,
            ms=2.625,
            f_value=15.3,
            p_value=0.001,
            is_significant=True,
        )
        assert row.source == "Part"
        assert row.df == 4
        assert row.is_significant is True

    def test_anova_row_without_f_and_p(self):
        """Test ANOVA row for total (no F or P values)."""
        row = ANOVARow(
            source="Total",
            df=29,
            ss=25.0,
            ms=None,
            f_value=None,
            p_value=None,
            is_significant=None,
        )
        assert row.f_value is None
        assert row.p_value is None

    def test_anova_result_valid(self):
        """Test valid ANOVAResult."""
        result = ANOVAResult(
            rows=[
                ANOVARow(source="Part", df=4, ss=10.0, ms=2.5, f_value=15.0, p_value=0.001, is_significant=True),
                ANOVARow(source="Operator", df=2, ss=2.0, ms=1.0, f_value=6.0, p_value=0.02, is_significant=True),
                ANOVARow(source="Part × Operator", df=8, ss=1.0, ms=0.125, f_value=0.75, p_value=0.65, is_significant=False),
                ANOVARow(source="Equipment", df=15, ss=2.5, ms=0.167, f_value=None, p_value=None, is_significant=None),
                ANOVARow(source="Total", df=29, ss=15.5, ms=None, f_value=None, p_value=None, is_significant=None),
            ],
            significant_effects=["Part", "Operator"],
        )
        assert len(result.rows) == 5
        assert result.significant_effects == ["Part", "Operator"]


class TestExtendedVarianceComponents:
    """Tests for ExtendedVarianceComponents model."""

    def test_extended_component_valid(self):
        """Test valid extended variance component."""
        component = ExtendedVarianceComponent(
            variance=0.05,
            std_dev=0.2236,
            pct_contribution=25.0,
            pct_study=30.0,
            pct_tolerance=15.0,
        )
        assert component.variance == 0.05
        assert component.pct_contribution == 25.0

    def test_extended_component_without_tolerance(self):
        """Test component without tolerance percentage."""
        component = ExtendedVarianceComponent(
            variance=0.05,
            std_dev=0.2236,
            pct_contribution=25.0,
            pct_study=30.0,
            pct_tolerance=None,
        )
        assert component.pct_tolerance is None

    def test_extended_variance_components_full(self):
        """Test full ExtendedVarianceComponents model."""
        components = ExtendedVarianceComponents(
            total_grr=ExtendedVarianceComponent(
                variance=0.04, std_dev=0.2, pct_contribution=20.0, pct_study=25.0, pct_tolerance=10.0
            ),
            repeatability=ExtendedVarianceComponent(
                variance=0.03, std_dev=0.173, pct_contribution=15.0, pct_study=20.0, pct_tolerance=8.0
            ),
            reproducibility=ExtendedVarianceComponent(
                variance=0.01, std_dev=0.1, pct_contribution=5.0, pct_study=10.0, pct_tolerance=4.0
            ),
            part_to_part=ExtendedVarianceComponent(
                variance=0.16, std_dev=0.4, pct_contribution=80.0, pct_study=75.0, pct_tolerance=None
            ),
            total_variation=ExtendedVarianceComponent(
                variance=0.2, std_dev=0.447, pct_contribution=100.0, pct_study=100.0, pct_tolerance=None
            ),
            interaction=ExtendedVarianceComponent(
                variance=0.005, std_dev=0.071, pct_contribution=2.5, pct_study=8.0, pct_tolerance=3.0
            ),
        )
        assert components.total_grr.pct_contribution == 20.0
        assert components.part_to_part.variance == 0.16


class TestOperatorMetrics:
    """Tests for OperatorMetrics model."""

    def test_operator_metrics_valid(self):
        """Test valid operator metrics."""
        metrics = OperatorMetrics(
            operator_id="Op1",
            mean=5.25,
            std_dev=0.15,
            range_avg=0.35,
            bias_estimate=0.02,
            consistency_rank=1,
        )
        assert metrics.operator_id == "Op1"
        assert metrics.consistency_rank == 1
        assert metrics.bias_estimate == 0.02

    def test_operator_metrics_all_operators(self):
        """Test list of operator metrics with reference operator."""
        operators = [
            OperatorMetrics(operator_id="Op1", mean=5.25, std_dev=0.15, range_avg=0.35, bias_estimate=0.02, consistency_rank=1),
            OperatorMetrics(operator_id="Op2", mean=5.30, std_dev=0.20, range_avg=0.40, bias_estimate=0.07, consistency_rank=2),
            OperatorMetrics(operator_id="Op3", mean=5.22, std_dev=0.25, range_avg=0.50, bias_estimate=-0.01, consistency_rank=3),
        ]
        # Op1 has lowest std_dev, so rank 1
        assert operators[0].consistency_rank == 1
        assert operators[2].consistency_rank == 3


class TestStabilityAnalysis:
    """Tests for StabilityAnalysis model."""

    def test_stability_no_drift(self):
        """Test stability analysis with no drift."""
        stability = StabilityAnalysis(
            has_drift=False,
            drift_direction=None,
            drift_magnitude=None,
            per_trial_means=[5.0, 5.1, 4.9, 5.0, 5.05],
            trend_p_value=0.45,
        )
        assert stability.has_drift is False
        assert stability.drift_direction is None

    def test_stability_with_drift(self):
        """Test stability analysis with increasing drift."""
        stability = StabilityAnalysis(
            has_drift=True,
            drift_direction="increasing",
            drift_magnitude=0.05,
            per_trial_means=[5.0, 5.1, 5.2, 5.3, 5.4],
            trend_p_value=0.01,
        )
        assert stability.has_drift is True
        assert stability.drift_direction == "increasing"
        assert stability.drift_magnitude == 0.05

    def test_stability_decreasing_drift(self):
        """Test stability analysis with decreasing drift."""
        stability = StabilityAnalysis(
            has_drift=True,
            drift_direction="decreasing",
            drift_magnitude=-0.03,
            per_trial_means=[5.4, 5.3, 5.2, 5.1, 5.0],
            trend_p_value=0.02,
        )
        assert stability.drift_direction == "decreasing"


class TestLinearityBiasResult:
    """Tests for LinearityBiasResult model."""

    def test_linearity_bias_valid(self):
        """Test valid linearity and bias result."""
        result = LinearityBiasResult(
            bias=0.05,
            bias_percent=2.5,
            linearity=0.02,
            linearity_percent=1.0,
            per_part_bias=[0.03, 0.05, 0.07, 0.04, 0.06],
            regression_slope=0.002,
            regression_intercept=0.01,
            reference_values_used="estimated",
        )
        assert result.bias == 0.05
        assert result.reference_values_used == "estimated"

    def test_linearity_bias_with_true_references(self):
        """Test linearity with provided reference values."""
        result = LinearityBiasResult(
            bias=0.03,
            bias_percent=1.5,
            linearity=0.01,
            linearity_percent=0.5,
            per_part_bias=[0.02, 0.03, 0.04],
            regression_slope=0.001,
            regression_intercept=0.005,
            reference_values_used="provided",
        )
        assert result.reference_values_used == "provided"


class TestResolutionAnalysis:
    """Tests for ResolutionAnalysis model."""

    def test_resolution_adequate(self):
        """Test adequate measurement resolution."""
        resolution = ResolutionAnalysis(
            measurement_resolution=0.01,
            resolution_ratio=0.05,
            is_adequate=True,
            recommendation="La resolución del instrumento es adecuada para este estudio MSA.",
        )
        assert resolution.is_adequate is True
        assert resolution.resolution_ratio < 0.1

    def test_resolution_inadequate(self):
        """Test inadequate measurement resolution."""
        resolution = ResolutionAnalysis(
            measurement_resolution=0.5,
            resolution_ratio=0.25,
            is_adequate=False,
            recommendation="La resolución del instrumento es insuficiente. Se recomienda un instrumento con mejor resolución.",
        )
        assert resolution.is_adequate is False
        assert "insuficiente" in resolution.recommendation


class TestChartDataModels:
    """Tests for chart data models."""

    def test_components_chart_data(self):
        """Test components chart data."""
        chart = ComponentsChartData(
            categories=["Total GRR", "Repeatability", "Reproducibility", "Part-to-Part"],
            pct_contribution=[20.0, 15.0, 5.0, 80.0],
            pct_study_var=[25.0, 20.0, 10.0, 75.0],
            pct_tolerance=[10.0, 8.0, 4.0, None],
        )
        assert len(chart.categories) == 4
        assert chart.pct_contribution[0] == 20.0

    def test_control_chart_data(self):
        """Test control chart data (R-chart or X-bar chart)."""
        chart = ControlChartData(
            data_points=[0.2, 0.3, 0.15, 0.25, 0.35],
            center_line=0.25,
            ucl=0.5,
            lcl=0.0,
            out_of_control_indices=[4],
        )
        assert chart.center_line == 0.25
        assert 4 in chart.out_of_control_indices

    def test_box_plot_data(self):
        """Test box plot data."""
        box = BoxPlotData(
            labels=["Part1", "Part2", "Part3"],
            min_values=[4.8, 5.7, 6.5],
            q1_values=[4.9, 5.8, 6.6],
            median_values=[5.0, 5.9, 6.7],
            q3_values=[5.1, 6.0, 6.8],
            max_values=[5.2, 6.1, 6.9],
            mean_values=[5.0, 5.9, 6.7],
        )
        assert len(box.labels) == 3
        assert box.median_values[0] == 5.0

    def test_interaction_plot_data(self):
        """Test interaction plot data."""
        plot = InteractionPlotData(
            parts=["P1", "P2", "P3"],
            operators=["Op1", "Op2"],
            means_matrix=[[5.0, 5.5, 6.0], [5.1, 5.4, 6.1]],
        )
        assert len(plot.parts) == 3
        assert len(plot.operators) == 2
        assert plot.means_matrix[0][1] == 5.5

    def test_chart_data_complete(self):
        """Test complete ChartData model."""
        charts = ChartData(
            components_chart=ComponentsChartData(
                categories=["GRR", "Part"],
                pct_contribution=[20.0, 80.0],
                pct_study_var=[25.0, 75.0],
                pct_tolerance=None,
            ),
            r_chart=ControlChartData(
                data_points=[0.2, 0.3],
                center_line=0.25,
                ucl=0.5,
                lcl=0.0,
                out_of_control_indices=[],
            ),
            xbar_chart=ControlChartData(
                data_points=[5.0, 5.1],
                center_line=5.05,
                ucl=5.5,
                lcl=4.6,
                out_of_control_indices=[],
            ),
            by_part=BoxPlotData(
                labels=["P1"],
                min_values=[4.8],
                q1_values=[4.9],
                median_values=[5.0],
                q3_values=[5.1],
                max_values=[5.2],
                mean_values=[5.0],
            ),
            by_operator=BoxPlotData(
                labels=["Op1"],
                min_values=[4.8],
                q1_values=[4.9],
                median_values=[5.0],
                q3_values=[5.1],
                max_values=[5.2],
                mean_values=[5.0],
            ),
            interaction_plot=InteractionPlotData(
                parts=["P1"],
                operators=["Op1"],
                means_matrix=[[5.0]],
            ),
        )
        assert charts.components_chart is not None
        assert charts.r_chart is not None


class TestExtendedMSAResult:
    """Tests for ExtendedMSAResult model."""

    def test_extended_result_minimal(self):
        """Test minimal extended MSA result."""
        result = ExtendedMSAResult(
            # Basic metrics (from original MSAResult)
            grr_percent=18.2,
            repeatability_percent=12.5,
            reproducibility_percent=5.7,
            part_variation_percent=81.8,
            ndc=7,
            category="marginal",
            # ANOVA
            anova=ANOVAResult(
                rows=[
                    ANOVARow(source="Total", df=29, ss=15.5, ms=None, f_value=None, p_value=None, is_significant=None),
                ],
                significant_effects=[],
            ),
            # Extended variance components
            variance_components=ExtendedVarianceComponents(
                total_grr=ExtendedVarianceComponent(variance=0.04, std_dev=0.2, pct_contribution=20.0, pct_study=25.0, pct_tolerance=None),
                repeatability=ExtendedVarianceComponent(variance=0.03, std_dev=0.173, pct_contribution=15.0, pct_study=20.0, pct_tolerance=None),
                reproducibility=ExtendedVarianceComponent(variance=0.01, std_dev=0.1, pct_contribution=5.0, pct_study=10.0, pct_tolerance=None),
                part_to_part=ExtendedVarianceComponent(variance=0.16, std_dev=0.4, pct_contribution=80.0, pct_study=75.0, pct_tolerance=None),
                total_variation=ExtendedVarianceComponent(variance=0.2, std_dev=0.447, pct_contribution=100.0, pct_study=100.0, pct_tolerance=None),
            ),
            # Operator metrics
            operator_metrics=[
                OperatorMetrics(operator_id="Op1", mean=5.0, std_dev=0.15, range_avg=0.3, bias_estimate=0.0, consistency_rank=1),
            ],
            reference_operator="Op1",
            # Charts
            charts=ChartData(
                components_chart=ComponentsChartData(categories=["GRR"], pct_contribution=[20.0], pct_study_var=[25.0], pct_tolerance=None),
                r_chart=ControlChartData(data_points=[0.2], center_line=0.2, ucl=0.4, lcl=0.0, out_of_control_indices=[]),
                xbar_chart=ControlChartData(data_points=[5.0], center_line=5.0, ucl=5.5, lcl=4.5, out_of_control_indices=[]),
                by_part=BoxPlotData(labels=["P1"], min_values=[4.8], q1_values=[4.9], median_values=[5.0], q3_values=[5.1], max_values=[5.2], mean_values=[5.0]),
                by_operator=BoxPlotData(labels=["Op1"], min_values=[4.8], q1_values=[4.9], median_values=[5.0], q3_values=[5.1], max_values=[5.2], mean_values=[5.0]),
                interaction_plot=InteractionPlotData(parts=["P1"], operators=["Op1"], means_matrix=[[5.0]]),
            ),
        )
        assert result.grr_percent == 18.2
        assert result.category == "marginal"
        assert result.reference_operator == "Op1"

    def test_extended_result_with_all_optional(self):
        """Test extended MSA result with all optional fields."""
        result = ExtendedMSAResult(
            grr_percent=15.0,
            repeatability_percent=10.0,
            reproducibility_percent=5.0,
            part_variation_percent=85.0,
            ndc=9,
            category="marginal",
            anova=ANOVAResult(
                rows=[ANOVARow(source="Total", df=29, ss=15.5, ms=None, f_value=None, p_value=None, is_significant=None)],
                significant_effects=["Part"],
            ),
            variance_components=ExtendedVarianceComponents(
                total_grr=ExtendedVarianceComponent(variance=0.04, std_dev=0.2, pct_contribution=20.0, pct_study=25.0, pct_tolerance=10.0),
                repeatability=ExtendedVarianceComponent(variance=0.03, std_dev=0.173, pct_contribution=15.0, pct_study=20.0, pct_tolerance=8.0),
                reproducibility=ExtendedVarianceComponent(variance=0.01, std_dev=0.1, pct_contribution=5.0, pct_study=10.0, pct_tolerance=4.0),
                part_to_part=ExtendedVarianceComponent(variance=0.16, std_dev=0.4, pct_contribution=80.0, pct_study=75.0, pct_tolerance=None),
                total_variation=ExtendedVarianceComponent(variance=0.2, std_dev=0.447, pct_contribution=100.0, pct_study=100.0, pct_tolerance=None),
            ),
            operator_metrics=[
                OperatorMetrics(operator_id="Op1", mean=5.0, std_dev=0.15, range_avg=0.3, bias_estimate=0.0, consistency_rank=1),
            ],
            reference_operator="Op1",
            charts=ChartData(
                components_chart=ComponentsChartData(categories=["GRR"], pct_contribution=[20.0], pct_study_var=[25.0], pct_tolerance=[10.0]),
                r_chart=ControlChartData(data_points=[0.2], center_line=0.2, ucl=0.4, lcl=0.0, out_of_control_indices=[]),
                xbar_chart=ControlChartData(data_points=[5.0], center_line=5.0, ucl=5.5, lcl=4.5, out_of_control_indices=[]),
                by_part=BoxPlotData(labels=["P1"], min_values=[4.8], q1_values=[4.9], median_values=[5.0], q3_values=[5.1], max_values=[5.2], mean_values=[5.0]),
                by_operator=BoxPlotData(labels=["Op1"], min_values=[4.8], q1_values=[4.9], median_values=[5.0], q3_values=[5.1], max_values=[5.2], mean_values=[5.0]),
                interaction_plot=InteractionPlotData(parts=["P1"], operators=["Op1"], means_matrix=[[5.0]]),
            ),
            # Optional analyses
            stability=StabilityAnalysis(
                has_drift=False,
                drift_direction=None,
                drift_magnitude=None,
                per_trial_means=[5.0, 5.1, 4.9],
                trend_p_value=0.5,
            ),
            linearity_bias=LinearityBiasResult(
                bias=0.02,
                bias_percent=1.0,
                linearity=0.01,
                linearity_percent=0.5,
                per_part_bias=[0.02],
                regression_slope=0.001,
                regression_intercept=0.01,
                reference_values_used="estimated",
            ),
            resolution=ResolutionAnalysis(
                measurement_resolution=0.01,
                resolution_ratio=0.05,
                is_adequate=True,
                recommendation="La resolución del instrumento es adecuada.",
            ),
        )
        assert result.stability is not None
        assert result.stability.has_drift is False
        assert result.linearity_bias is not None
        assert result.resolution is not None
        assert result.resolution.is_adequate is True
