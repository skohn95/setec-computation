"""Integration tests for Extended MSA endpoints.

Tests for /msa/compute-extended and /msa/detect-structure endpoints.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def valid_extended_input():
    """Valid input data for extended MSA endpoint."""
    return {
        "parts": ["P1", "P2", "P3", "P4", "P5"],
        "operators": ["Op1", "Op2", "Op3"],
        "trials": 2,
        "measurements": [
            [[5.0, 5.1], [4.9, 5.0], [5.0, 5.1]],
            [[6.0, 6.1], [5.9, 6.0], [6.0, 6.0]],
            [[7.0, 7.1], [6.9, 7.0], [7.0, 7.1]],
            [[8.0, 8.1], [7.9, 8.0], [8.0, 8.0]],
            [[9.0, 9.1], [8.9, 9.0], [9.0, 9.1]],
        ],
    }


@pytest.fixture
def valid_input_with_tolerance(valid_extended_input):
    """Valid input with tolerance specified."""
    return {**valid_extended_input, "tolerance": 2.0}


# =============================================================================
# /msa/compute-extended Endpoint Tests
# =============================================================================


class TestComputeExtendedEndpoint:
    """Tests for POST /msa/compute-extended endpoint."""

    def test_returns_200_with_valid_input(self, client, valid_extended_input):
        """Test successful computation returns 200."""
        response = client.post("/api/msa/compute-extended", json=valid_extended_input)
        assert response.status_code == 200

    def test_response_has_success_status(self, client, valid_extended_input):
        """Test response has status: success."""
        response = client.post("/api/msa/compute-extended", json=valid_extended_input)
        data = response.json()
        assert data["status"] == "success"

    def test_response_contains_basic_metrics(self, client, valid_extended_input):
        """Test response contains basic MSA metrics."""
        response = client.post("/api/msa/compute-extended", json=valid_extended_input)
        data = response.json()["data"]

        assert "grr_percent" in data
        assert "repeatability_percent" in data
        assert "reproducibility_percent" in data
        assert "part_variation_percent" in data
        assert "ndc" in data
        assert "category" in data

    def test_response_contains_anova(self, client, valid_extended_input):
        """Test response contains ANOVA table."""
        response = client.post("/api/msa/compute-extended", json=valid_extended_input)
        data = response.json()["data"]

        assert "anova" in data
        assert "rows" in data["anova"]
        assert "significant_effects" in data["anova"]
        assert len(data["anova"]["rows"]) == 5  # Part, Operator, Interaction, Equipment, Total

    def test_anova_rows_have_required_fields(self, client, valid_extended_input):
        """Test ANOVA rows have all required fields."""
        response = client.post("/api/msa/compute-extended", json=valid_extended_input)
        rows = response.json()["data"]["anova"]["rows"]

        for row in rows:
            assert "source" in row
            assert "df" in row
            assert "ss" in row
            # ms, f_value, p_value, is_significant can be None for some rows

    def test_response_contains_variance_components(self, client, valid_extended_input):
        """Test response contains extended variance components."""
        response = client.post("/api/msa/compute-extended", json=valid_extended_input)
        data = response.json()["data"]

        assert "variance_components" in data
        vc = data["variance_components"]
        assert "total_grr" in vc
        assert "repeatability" in vc
        assert "reproducibility" in vc
        assert "part_to_part" in vc
        assert "total_variation" in vc

    def test_variance_component_has_percentages(self, client, valid_extended_input):
        """Test variance components have percentage fields."""
        response = client.post("/api/msa/compute-extended", json=valid_extended_input)
        vc = response.json()["data"]["variance_components"]["total_grr"]

        assert "variance" in vc
        assert "std_dev" in vc
        assert "pct_contribution" in vc
        assert "pct_study" in vc

    def test_pct_tolerance_included_with_tolerance(self, client, valid_input_with_tolerance):
        """Test %Tolerance is calculated when tolerance is provided."""
        response = client.post("/api/msa/compute-extended", json=valid_input_with_tolerance)
        vc = response.json()["data"]["variance_components"]["total_grr"]

        assert vc["pct_tolerance"] is not None

    def test_pct_tolerance_none_without_tolerance(self, client, valid_extended_input):
        """Test %Tolerance is None when tolerance not provided."""
        response = client.post("/api/msa/compute-extended", json=valid_extended_input)
        vc = response.json()["data"]["variance_components"]["total_grr"]

        assert vc["pct_tolerance"] is None

    def test_response_contains_operator_metrics(self, client, valid_extended_input):
        """Test response contains operator metrics."""
        response = client.post("/api/msa/compute-extended", json=valid_extended_input)
        data = response.json()["data"]

        assert "operator_metrics" in data
        assert len(data["operator_metrics"]) == 3  # 3 operators

    def test_operator_metrics_have_required_fields(self, client, valid_extended_input):
        """Test operator metrics have all required fields."""
        response = client.post("/api/msa/compute-extended", json=valid_extended_input)
        metrics = response.json()["data"]["operator_metrics"]

        for m in metrics:
            assert "operator_id" in m
            assert "mean" in m
            assert "std_dev" in m
            assert "range_avg" in m
            assert "bias_estimate" in m
            assert "consistency_rank" in m

    def test_reference_operator_included(self, client, valid_extended_input):
        """Test reference operator is identified."""
        response = client.post("/api/msa/compute-extended", json=valid_extended_input)
        data = response.json()["data"]

        assert "reference_operator" in data
        assert data["reference_operator"] in ["Op1", "Op2", "Op3"]

    def test_response_contains_charts(self, client, valid_extended_input):
        """Test response contains chart data."""
        response = client.post("/api/msa/compute-extended", json=valid_extended_input)
        data = response.json()["data"]

        assert "charts" in data
        charts = data["charts"]
        assert "components_chart" in charts
        assert "r_chart" in charts
        assert "xbar_chart" in charts
        assert "by_part" in charts
        assert "by_operator" in charts
        assert "interaction_plot" in charts

    def test_control_charts_have_limits(self, client, valid_extended_input):
        """Test control charts have center line and limits."""
        response = client.post("/api/msa/compute-extended", json=valid_extended_input)
        charts = response.json()["data"]["charts"]

        for chart_name in ["r_chart", "xbar_chart"]:
            chart = charts[chart_name]
            assert "data_points" in chart
            assert "center_line" in chart
            assert "ucl" in chart
            assert "lcl" in chart
            assert "out_of_control_indices" in chart

    def test_response_contains_stability_analysis(self, client, valid_extended_input):
        """Test response contains stability analysis."""
        response = client.post("/api/msa/compute-extended", json=valid_extended_input)
        data = response.json()["data"]

        assert "stability" in data
        stability = data["stability"]
        assert "has_drift" in stability
        assert "per_trial_means" in stability
        assert "trend_p_value" in stability

    def test_response_contains_linearity_bias(self, client, valid_extended_input):
        """Test response contains linearity/bias analysis."""
        response = client.post("/api/msa/compute-extended", json=valid_extended_input)
        data = response.json()["data"]

        assert "linearity_bias" in data
        lb = data["linearity_bias"]
        assert "bias" in lb
        assert "linearity" in lb
        assert "per_part_bias" in lb
        assert "reference_values_used" in lb

    def test_resolution_included_with_tolerance(self, client, valid_input_with_tolerance):
        """Test resolution analysis is included when tolerance provided."""
        response = client.post("/api/msa/compute-extended", json=valid_input_with_tolerance)
        data = response.json()["data"]

        assert "resolution" in data
        assert data["resolution"] is not None
        assert "measurement_resolution" in data["resolution"]
        assert "resolution_ratio" in data["resolution"]
        assert "is_adequate" in data["resolution"]
        assert "recommendation" in data["resolution"]

    def test_category_is_valid(self, client, valid_extended_input):
        """Test category is one of the valid values."""
        response = client.post("/api/msa/compute-extended", json=valid_extended_input)
        category = response.json()["data"]["category"]

        assert category in ["excellent", "marginal", "unacceptable"]

    def test_invalid_input_returns_422(self, client):
        """Test invalid input returns 422."""
        response = client.post("/api/msa/compute-extended", json={"invalid": "data"})
        assert response.status_code == 422

    def test_missing_measurements_returns_422(self, client, valid_extended_input):
        """Test missing measurements returns 422."""
        del valid_extended_input["measurements"]
        response = client.post("/api/msa/compute-extended", json=valid_extended_input)
        assert response.status_code == 422

    def test_too_few_parts_returns_422(self, client):
        """Test fewer than 2 parts returns 422."""
        data = {
            "parts": ["P1"],  # Only 1 part
            "operators": ["Op1"],
            "trials": 2,
            "measurements": [[[1.0, 1.1]]],
        }
        response = client.post("/api/msa/compute-extended", json=data)
        assert response.status_code == 422

    def test_negative_tolerance_returns_422(self, client, valid_extended_input):
        """Test negative tolerance returns 422."""
        valid_extended_input["tolerance"] = -1.0
        response = client.post("/api/msa/compute-extended", json=valid_extended_input)
        assert response.status_code == 422

    def test_response_time_reasonable(self, client, valid_extended_input):
        """Test response time is reasonable (< 5 seconds)."""
        import time
        start = time.time()
        response = client.post("/api/msa/compute-extended", json=valid_extended_input)
        elapsed = time.time() - start

        assert response.status_code == 200
        assert elapsed < 5.0


# =============================================================================
# /msa/detect-structure Endpoint Tests
# =============================================================================


class TestDetectStructureEndpoint:
    """Tests for POST /msa/detect-structure endpoint."""

    def test_returns_200_with_valid_input(self, client, valid_extended_input):
        """Test successful structure detection returns 200."""
        response = client.post("/api/msa/detect-structure", json=valid_extended_input)
        assert response.status_code == 200

    def test_response_has_success_status(self, client, valid_extended_input):
        """Test response has status: success."""
        response = client.post("/api/msa/detect-structure", json=valid_extended_input)
        data = response.json()
        assert data["status"] == "success"

    def test_detects_correct_part_count(self, client, valid_extended_input):
        """Test correct number of parts detected."""
        response = client.post("/api/msa/detect-structure", json=valid_extended_input)
        data = response.json()["data"]
        assert data["n_parts"] == 5

    def test_detects_correct_operator_count(self, client, valid_extended_input):
        """Test correct number of operators detected."""
        response = client.post("/api/msa/detect-structure", json=valid_extended_input)
        data = response.json()["data"]
        assert data["k_operators"] == 3

    def test_detects_correct_trial_count(self, client, valid_extended_input):
        """Test correct number of trials detected."""
        response = client.post("/api/msa/detect-structure", json=valid_extended_input)
        data = response.json()["data"]
        assert data["r_trials"] == 2

    def test_calculates_total_measurements(self, client, valid_extended_input):
        """Test total measurements is calculated correctly."""
        response = client.post("/api/msa/detect-structure", json=valid_extended_input)
        data = response.json()["data"]
        # 5 parts * 3 operators * 2 trials = 30
        assert data["total_measurements"] == 30

    def test_returns_operator_ids(self, client, valid_extended_input):
        """Test operator IDs are returned."""
        response = client.post("/api/msa/detect-structure", json=valid_extended_input)
        data = response.json()["data"]
        assert data["operator_ids"] == ["Op1", "Op2", "Op3"]

    def test_returns_part_ids(self, client, valid_extended_input):
        """Test part IDs are returned."""
        response = client.post("/api/msa/detect-structure", json=valid_extended_input)
        data = response.json()["data"]
        assert data["part_ids"] == ["P1", "P2", "P3", "P4", "P5"]

    def test_invalid_input_returns_422(self, client):
        """Test invalid input returns 422."""
        response = client.post("/api/msa/detect-structure", json={"invalid": "data"})
        assert response.status_code == 422


# =============================================================================
# Backward Compatibility Tests
# =============================================================================


class TestBackwardCompatibility:
    """Tests to ensure original /msa/compute endpoint still works."""

    def test_original_endpoint_still_works(self, client, valid_extended_input):
        """Test original /msa/compute endpoint is unchanged."""
        # Remove tolerance (not part of original MSAInput)
        input_data = {k: v for k, v in valid_extended_input.items() if k != "tolerance"}

        response = client.post("/api/msa/compute", json=input_data)
        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "success"
        assert "grr_percent" in data["data"]
        assert "category" in data["data"]
        assert "components" in data["data"]

    def test_original_endpoint_response_format(self, client, valid_extended_input):
        """Test original endpoint returns original response format (not extended)."""
        input_data = {k: v for k, v in valid_extended_input.items() if k != "tolerance"}

        response = client.post("/api/msa/compute", json=input_data)
        data = response.json()["data"]

        # Original format should have 'components' not 'variance_components'
        assert "components" in data
        # Should NOT have extended fields
        assert "anova" not in data
        assert "operator_metrics" not in data
        assert "charts" not in data
