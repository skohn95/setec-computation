"""Unit tests for MSA Calculator Service.

Tests for Gauge R&R ANOVA-based calculations with known results.
"""

import math

import pytest

from app.models.msa import MSAInput
from app.services.msa_calculator import MSACalculator


class TestMSACalculatorKnownResults:
    """Tests with known results to verify calculation accuracy."""

    @pytest.fixture
    def calculator(self) -> MSACalculator:
        """Create calculator instance."""
        return MSACalculator()

    @pytest.fixture
    def standard_test_data(self) -> MSAInput:
        """Standard test dataset: 5 parts, 3 operators, 2 trials.

        This dataset has:
        - High part variation (parts are 2,3,4,5,6 - intentionally different)
        - Low repeatability (same operator measures similarly)
        - Low reproducibility (operators measure similarly)
        - Expected: %GRR < 10% (excellent), ndc > 5
        """
        return MSAInput(
            parts=["Part1", "Part2", "Part3", "Part4", "Part5"],
            operators=["Op1", "Op2", "Op3"],
            trials=2,
            measurements=[
                # Part 1
                [[2.00, 2.10], [2.05, 2.15], [2.00, 2.10]],
                # Part 2
                [[3.00, 3.05], [2.95, 3.10], [3.00, 3.00]],
                # Part 3
                [[4.00, 4.10], [4.05, 4.00], [4.10, 4.05]],
                # Part 4
                [[5.00, 5.05], [5.10, 5.00], [5.00, 5.10]],
                # Part 5
                [[6.00, 5.95], [6.05, 6.00], [5.95, 6.00]],
            ],
        )

    def test_known_dataset_returns_excellent_category(
        self, calculator: MSACalculator, standard_test_data: MSAInput
    ):
        """Test standard dataset returns excellent category (%GRR < 10%)."""
        result = calculator.calculate(standard_test_data)

        assert result.category == "excellent"
        assert result.grr_percent < 10.0

    def test_known_dataset_has_high_ndc(
        self, calculator: MSACalculator, standard_test_data: MSAInput
    ):
        """Test standard dataset has good discrimination (ndc > 5)."""
        result = calculator.calculate(standard_test_data)

        assert result.ndc >= 5

    def test_percentages_are_reasonable(
        self, calculator: MSACalculator, standard_test_data: MSAInput
    ):
        """Test that percentages are within reasonable bounds.

        Note: In Gauge R&R, percentages are ratios of standard deviations to total
        standard deviation. Since σ²_total = σ²_part + σ²_grr, the std dev ratios
        can sum to more than 100% (sqrt(a) + sqrt(b) >= sqrt(a+b)).
        The key invariant is: repeatability%² + reproducibility%² ≈ grr%²
        """
        result = calculator.calculate(standard_test_data)

        # Each percentage should be non-negative and <= 100%
        assert 0 <= result.grr_percent <= 100.0
        assert 0 <= result.repeatability_percent <= 100.0
        assert 0 <= result.reproducibility_percent <= 100.0
        assert 0 <= result.part_variation_percent <= 100.0

    def test_repeatability_plus_reproducibility_equals_grr(
        self, calculator: MSACalculator, standard_test_data: MSAInput
    ):
        """Test repeatability% + reproducibility% ≈ GRR% (within tolerance)."""
        result = calculator.calculate(standard_test_data)

        # Due to how std devs combine (sqrt of sum of variances),
        # repeat%² + reprod%² ≈ grr%² (approximately)
        expected_grr_squared = result.repeatability_percent**2 + result.reproducibility_percent**2
        actual_grr_squared = result.grr_percent**2

        assert abs(expected_grr_squared - actual_grr_squared) < 1.0


class TestMSACalculatorCategories:
    """Tests for %GRR category classification."""

    @pytest.fixture
    def calculator(self) -> MSACalculator:
        """Create calculator instance."""
        return MSACalculator()

    def test_excellent_category_below_10_percent(self, calculator: MSACalculator):
        """Test excellent category when %GRR < 10%."""
        # Dataset with very low measurement variation relative to part variation
        data = MSAInput(
            parts=["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10"],
            operators=["Op1", "Op2"],
            trials=3,
            measurements=[
                [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]],
                [[3.0, 3.0, 3.0], [3.0, 3.0, 3.0]],
                [[4.0, 4.0, 4.0], [4.0, 4.0, 4.0]],
                [[5.0, 5.0, 5.0], [5.0, 5.0, 5.0]],
                [[6.0, 6.0, 6.0], [6.0, 6.0, 6.0]],
                [[7.0, 7.0, 7.0], [7.0, 7.0, 7.0]],
                [[8.0, 8.0, 8.0], [8.0, 8.0, 8.0]],
                [[9.0, 9.0, 9.0], [9.0, 9.0, 9.0]],
                [[10.0, 10.0, 10.0], [10.0, 10.0, 10.0]],
            ],
        )
        result = calculator.calculate(data)

        # Perfect measurements = 0% GRR, excellent category
        assert result.category == "excellent"
        assert result.grr_percent < 10.0

    def test_marginal_category_10_to_30_percent(self, calculator: MSACalculator):
        """Test marginal category when 10% <= %GRR <= 30%."""
        # Dataset designed to have moderate measurement variation
        data = MSAInput(
            parts=["P1", "P2", "P3", "P4", "P5"],
            operators=["Op1", "Op2"],
            trials=2,
            measurements=[
                [[1.0, 1.3], [1.2, 0.9]],
                [[2.0, 2.3], [2.2, 1.9]],
                [[3.0, 3.3], [3.2, 2.9]],
                [[4.0, 4.3], [4.2, 3.9]],
                [[5.0, 5.3], [5.2, 4.9]],
            ],
        )
        result = calculator.calculate(data)

        # Moderate variation should be marginal
        assert result.category == "marginal"
        assert 10.0 <= result.grr_percent <= 30.0

    def test_unacceptable_category_above_30_percent(self, calculator: MSACalculator):
        """Test unacceptable category when %GRR > 30%."""
        # Dataset with high measurement variation relative to part variation
        data = MSAInput(
            parts=["P1", "P2"],
            operators=["Op1", "Op2"],
            trials=2,
            measurements=[
                [[1.0, 2.0], [0.5, 2.5]],  # High within-part variation
                [[1.5, 2.5], [1.0, 3.0]],  # Similar parts, high measurement noise
            ],
        )
        result = calculator.calculate(data)

        assert result.category == "unacceptable"
        assert result.grr_percent > 30.0


class TestMSACalculatorEdgeCases:
    """Tests for edge cases and special conditions."""

    @pytest.fixture
    def calculator(self) -> MSACalculator:
        """Create calculator instance."""
        return MSACalculator()

    def test_single_operator_reproducibility_zero(self, calculator: MSACalculator):
        """Test single operator case has reproducibility = 0."""
        data = MSAInput(
            parts=["P1", "P2", "P3", "P4", "P5"],
            operators=["Op1"],  # Single operator
            trials=3,
            measurements=[
                [[2.0, 2.1, 2.0]],
                [[3.0, 3.1, 3.0]],
                [[4.0, 4.1, 4.0]],
                [[5.0, 5.1, 5.0]],
                [[6.0, 6.1, 6.0]],
            ],
        )
        result = calculator.calculate(data)

        # Single operator = no between-operator variation
        assert result.reproducibility_percent == 0.0
        assert result.components.operator_variation == 0.0

    def test_minimum_valid_dataset(self, calculator: MSACalculator):
        """Test minimum valid dataset: 2 parts, 1 operator, 2 trials."""
        data = MSAInput(
            parts=["P1", "P2"],
            operators=["Op1"],
            trials=2,
            measurements=[
                [[1.0, 1.1]],
                [[2.0, 2.1]],
            ],
        )
        result = calculator.calculate(data)

        # Should compute without error
        assert result.grr_percent >= 0
        assert result.ndc >= 0
        assert result.category in ["excellent", "marginal", "unacceptable"]

    def test_variance_components_non_negative(self, calculator: MSACalculator):
        """Test all variance components are clamped to >= 0."""
        # Any dataset - variance components should never be negative
        data = MSAInput(
            parts=["P1", "P2", "P3"],
            operators=["Op1", "Op2"],
            trials=2,
            measurements=[
                [[1.0, 1.05], [1.02, 1.03]],
                [[2.0, 2.05], [2.02, 2.03]],
                [[3.0, 3.05], [3.02, 3.03]],
            ],
        )
        result = calculator.calculate(data)

        assert result.components.equipment_variation >= 0
        assert result.components.operator_variation >= 0
        assert result.components.part_variation >= 0
        assert result.components.total_variation >= 0


class TestNDCCalculation:
    """Tests for Number of Distinct Categories (ndc) calculation."""

    @pytest.fixture
    def calculator(self) -> MSACalculator:
        """Create calculator instance."""
        return MSACalculator()

    def test_ndc_formula_correctness(self, calculator: MSACalculator):
        """Test ndc = floor(1.41 × (Part Std Dev / GRR Std Dev))."""
        data = MSAInput(
            parts=["P1", "P2", "P3", "P4", "P5"],
            operators=["Op1", "Op2", "Op3"],
            trials=2,
            measurements=[
                [[2.00, 2.10], [2.05, 2.15], [2.00, 2.10]],
                [[3.00, 3.05], [2.95, 3.10], [3.00, 3.00]],
                [[4.00, 4.10], [4.05, 4.00], [4.10, 4.05]],
                [[5.00, 5.05], [5.10, 5.00], [5.00, 5.10]],
                [[6.00, 5.95], [6.05, 6.00], [5.95, 6.00]],
            ],
        )
        result = calculator.calculate(data)

        # Verify ndc calculation: floor(1.41 × σ_part / σ_grr)
        sigma_part = math.sqrt(result.components.part_variation)
        sigma_grr = math.sqrt(
            result.components.equipment_variation + result.components.operator_variation
        )

        if sigma_grr > 0:
            expected_ndc = int(1.41 * sigma_part / sigma_grr)
            assert result.ndc == expected_ndc

    def test_ndc_minimum_zero(self, calculator: MSACalculator):
        """Test ndc cannot be negative."""
        # High GRR, low part variation
        data = MSAInput(
            parts=["P1", "P2"],
            operators=["Op1", "Op2"],
            trials=2,
            measurements=[
                [[1.0, 2.0], [0.5, 2.5]],
                [[1.2, 2.2], [0.7, 2.7]],
            ],
        )
        result = calculator.calculate(data)

        assert result.ndc >= 0
