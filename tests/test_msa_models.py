"""Unit tests for MSA Pydantic models.

Tests for MSAInput, MSAResult, and VarianceComponents models.
"""

import pytest
from pydantic import ValidationError

from app.models.msa import MSAInput, MSAResult, VarianceComponents


class TestMSAInput:
    """Tests for MSAInput model validation."""

    def test_valid_input_minimal(self):
        """Test minimum valid input: 2 parts, 1 operator, 2 trials."""
        data = MSAInput(
            parts=["Part1", "Part2"],
            operators=["Op1"],
            trials=2,
            measurements=[
                [[2.0, 2.1]],  # Part 1, Op1, 2 trials
                [[3.0, 3.1]],  # Part 2, Op1, 2 trials
            ],
        )
        assert len(data.parts) == 2
        assert len(data.operators) == 1
        assert data.trials == 2

    def test_valid_input_standard(self):
        """Test standard valid input: 5 parts, 3 operators, 2 trials."""
        data = MSAInput(
            parts=["P1", "P2", "P3", "P4", "P5"],
            operators=["Op1", "Op2", "Op3"],
            trials=3,
            measurements=[
                [[2.0, 2.1, 2.0], [2.1, 2.0, 2.1], [2.0, 2.0, 2.1]],
                [[3.0, 3.1, 3.0], [3.1, 3.0, 3.1], [3.0, 3.0, 3.1]],
                [[4.0, 4.1, 4.0], [4.1, 4.0, 4.1], [4.0, 4.0, 4.1]],
                [[5.0, 5.1, 5.0], [5.1, 5.0, 5.1], [5.0, 5.0, 5.1]],
                [[6.0, 6.1, 6.0], [6.1, 6.0, 6.1], [6.0, 6.0, 6.1]],
            ],
        )
        assert len(data.parts) == 5
        assert len(data.operators) == 3
        assert data.trials == 3

    def test_invalid_parts_count(self):
        """Test validation fails with fewer than 2 parts."""
        with pytest.raises(ValidationError) as exc_info:
            MSAInput(
                parts=["Part1"],  # Only 1 part - invalid
                operators=["Op1"],
                trials=2,
                measurements=[[[2.0, 2.1]]],
            )
        assert "al menos 2 partes" in str(exc_info.value).lower() or "parts" in str(exc_info.value).lower()

    def test_invalid_operators_count(self):
        """Test validation fails with zero operators."""
        with pytest.raises(ValidationError) as exc_info:
            MSAInput(
                parts=["Part1", "Part2"],
                operators=[],  # No operators - invalid
                trials=2,
                measurements=[[[2.0, 2.1]], [[3.0, 3.1]]],
            )
        assert "operador" in str(exc_info.value).lower() or "operators" in str(exc_info.value).lower()

    def test_invalid_trials_count(self):
        """Test validation fails with fewer than 2 trials."""
        with pytest.raises(ValidationError) as exc_info:
            MSAInput(
                parts=["Part1", "Part2"],
                operators=["Op1"],
                trials=1,  # Only 1 trial - invalid
                measurements=[[[2.0]], [[3.0]]],
            )
        assert "trial" in str(exc_info.value).lower() or "repeticiones" in str(exc_info.value).lower()

    def test_invalid_measurements_shape_parts(self):
        """Test validation fails when measurements don't match parts count."""
        with pytest.raises(ValidationError) as exc_info:
            MSAInput(
                parts=["Part1", "Part2"],
                operators=["Op1"],
                trials=2,
                measurements=[[[2.0, 2.1]]],  # Only 1 part measurement
            )
        assert "partes" in str(exc_info.value).lower() or "measurements" in str(exc_info.value).lower()

    def test_invalid_measurements_shape_operators(self):
        """Test validation fails when measurements don't match operators count."""
        with pytest.raises(ValidationError) as exc_info:
            MSAInput(
                parts=["Part1", "Part2"],
                operators=["Op1", "Op2"],
                trials=2,
                measurements=[
                    [[2.0, 2.1]],  # Only 1 operator
                    [[3.0, 3.1]],  # Only 1 operator
                ],
            )
        assert "operadores" in str(exc_info.value).lower() or "measurements" in str(exc_info.value).lower()

    def test_invalid_measurements_shape_trials(self):
        """Test validation fails when measurements don't match trials count."""
        with pytest.raises(ValidationError) as exc_info:
            MSAInput(
                parts=["Part1", "Part2"],
                operators=["Op1"],
                trials=3,
                measurements=[
                    [[2.0, 2.1]],  # Only 2 trials
                    [[3.0, 3.1]],  # Only 2 trials
                ],
            )
        assert "trial" in str(exc_info.value).lower() or "repeticiones" in str(exc_info.value).lower()

    def test_negative_measurements_are_valid(self):
        """Test negative measurements are accepted (valid for deviations, temperature, etc.)."""
        data = MSAInput(
            parts=["Part1", "Part2"],
            operators=["Op1"],
            trials=2,
            measurements=[
                [[-2.0, -2.1]],  # Negative values
                [[-3.0, -3.1]],
            ],
        )
        assert data.measurements[0][0][0] == -2.0


class TestVarianceComponents:
    """Tests for VarianceComponents model."""

    def test_valid_components(self):
        """Test valid variance components."""
        components = VarianceComponents(
            equipment_variation=0.0234,
            operator_variation=0.0156,
            part_variation=0.1823,
            total_variation=0.2013,
        )
        assert components.equipment_variation == 0.0234
        assert components.operator_variation == 0.0156
        assert components.part_variation == 0.1823
        assert components.total_variation == 0.2013

    def test_components_cannot_be_negative(self):
        """Test variance components cannot be negative."""
        with pytest.raises(ValidationError):
            VarianceComponents(
                equipment_variation=-0.01,  # Invalid - negative
                operator_variation=0.0156,
                part_variation=0.1823,
                total_variation=0.2013,
            )


class TestMSAResult:
    """Tests for MSAResult model."""

    def test_valid_result_excellent(self):
        """Test valid result in excellent category."""
        result = MSAResult(
            grr_percent=5.0,
            repeatability_percent=3.0,
            reproducibility_percent=2.0,
            part_variation_percent=95.0,
            ndc=15,
            category="excellent",
            components=VarianceComponents(
                equipment_variation=0.01,
                operator_variation=0.005,
                part_variation=0.2,
                total_variation=0.215,
            ),
        )
        assert result.grr_percent == 5.0
        assert result.category == "excellent"
        assert result.ndc == 15

    def test_valid_result_marginal(self):
        """Test valid result in marginal category."""
        result = MSAResult(
            grr_percent=18.2,
            repeatability_percent=12.5,
            reproducibility_percent=5.7,
            part_variation_percent=81.8,
            ndc=7,
            category="marginal",
            components=VarianceComponents(
                equipment_variation=0.0234,
                operator_variation=0.0156,
                part_variation=0.1823,
                total_variation=0.2013,
            ),
        )
        assert result.grr_percent == 18.2
        assert result.category == "marginal"

    def test_valid_result_unacceptable(self):
        """Test valid result in unacceptable category."""
        result = MSAResult(
            grr_percent=45.0,
            repeatability_percent=30.0,
            reproducibility_percent=15.0,
            part_variation_percent=55.0,
            ndc=2,
            category="unacceptable",
            components=VarianceComponents(
                equipment_variation=0.1,
                operator_variation=0.05,
                part_variation=0.12,
                total_variation=0.27,
            ),
        )
        assert result.grr_percent == 45.0
        assert result.category == "unacceptable"

    def test_invalid_category(self):
        """Test invalid category value."""
        with pytest.raises(ValidationError):
            MSAResult(
                grr_percent=18.2,
                repeatability_percent=12.5,
                reproducibility_percent=5.7,
                part_variation_percent=81.8,
                ndc=7,
                category="unknown",  # Invalid category
                components=VarianceComponents(
                    equipment_variation=0.0234,
                    operator_variation=0.0156,
                    part_variation=0.1823,
                    total_variation=0.2013,
                ),
            )

    def test_ndc_must_be_non_negative(self):
        """Test ndc cannot be negative."""
        with pytest.raises(ValidationError):
            MSAResult(
                grr_percent=18.2,
                repeatability_percent=12.5,
                reproducibility_percent=5.7,
                part_variation_percent=81.8,
                ndc=-1,  # Invalid - negative
                category="marginal",
                components=VarianceComponents(
                    equipment_variation=0.0234,
                    operator_variation=0.0156,
                    part_variation=0.1823,
                    total_variation=0.2013,
                ),
            )
