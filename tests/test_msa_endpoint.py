"""Integration tests for MSA endpoint.

Tests for POST /api/msa/compute with various inputs.
"""

import time

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client() -> TestClient:
    """Create test client."""
    return TestClient(app)


class TestMSAEndpointSuccess:
    """Tests for successful MSA computations."""

    def test_compute_returns_200_with_valid_input(self, client: TestClient):
        """Test successful computation returns 200."""
        response = client.post(
            "/api/msa/compute",
            json={
                "parts": ["P1", "P2", "P3", "P4", "P5"],
                "operators": ["Op1", "Op2", "Op3"],
                "trials": 2,
                "measurements": [
                    [[2.00, 2.10], [2.05, 2.15], [2.00, 2.10]],
                    [[3.00, 3.05], [2.95, 3.10], [3.00, 3.00]],
                    [[4.00, 4.10], [4.05, 4.00], [4.10, 4.05]],
                    [[5.00, 5.05], [5.10, 5.00], [5.00, 5.10]],
                    [[6.00, 5.95], [6.05, 6.00], [5.95, 6.00]],
                ],
            },
        )

        assert response.status_code == 200

    def test_response_has_correct_structure(self, client: TestClient):
        """Test response structure matches AC #2."""
        response = client.post(
            "/api/msa/compute",
            json={
                "parts": ["P1", "P2", "P3", "P4", "P5"],
                "operators": ["Op1", "Op2"],
                "trials": 2,
                "measurements": [
                    [[2.0, 2.1], [2.0, 2.1]],
                    [[3.0, 3.1], [3.0, 3.1]],
                    [[4.0, 4.1], [4.0, 4.1]],
                    [[5.0, 5.1], [5.0, 5.1]],
                    [[6.0, 6.1], [6.0, 6.1]],
                ],
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Check top-level structure
        assert "status" in data
        assert data["status"] == "success"
        assert "data" in data

        # Check data fields per AC #2
        result = data["data"]
        assert "grr_percent" in result
        assert "repeatability_percent" in result
        assert "reproducibility_percent" in result
        assert "part_variation_percent" in result
        assert "ndc" in result
        assert "category" in result
        assert "components" in result

        # Check components structure
        components = result["components"]
        assert "equipment_variation" in components
        assert "operator_variation" in components
        assert "part_variation" in components
        assert "total_variation" in components

    def test_category_is_valid_value(self, client: TestClient):
        """Test category is one of: excellent, marginal, unacceptable."""
        response = client.post(
            "/api/msa/compute",
            json={
                "parts": ["P1", "P2", "P3"],
                "operators": ["Op1"],
                "trials": 2,
                "measurements": [
                    [[1.0, 1.1]],
                    [[2.0, 2.1]],
                    [[3.0, 3.1]],
                ],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["data"]["category"] in ["excellent", "marginal", "unacceptable"]

    def test_response_time_under_5_seconds(self, client: TestClient):
        """Test response time < 5 seconds for typical dataset."""
        start_time = time.time()

        response = client.post(
            "/api/msa/compute",
            json={
                "parts": [f"P{i}" for i in range(1, 11)],  # 10 parts
                "operators": ["Op1", "Op2", "Op3"],
                "trials": 3,
                "measurements": [
                    [[float(i + j * 0.1 + k * 0.01) for k in range(3)] for j in range(3)]
                    for i in range(1, 11)
                ],
            },
        )

        elapsed_time = time.time() - start_time

        assert response.status_code == 200
        assert elapsed_time < 5.0, f"Response took {elapsed_time:.2f} seconds"


class TestMSAEndpointValidationErrors:
    """Tests for validation error handling."""

    def test_invalid_input_returns_422(self, client: TestClient):
        """Test invalid input returns 422 with Spanish error message."""
        response = client.post(
            "/api/msa/compute",
            json={
                "parts": ["P1"],  # Invalid: only 1 part
                "operators": ["Op1"],
                "trials": 2,
                "measurements": [[[1.0, 1.1]]],
            },
        )

        assert response.status_code == 422  # Pydantic validation error
        data = response.json()
        assert "error" in data
        assert "message" in data["error"]
        # Check for Spanish message
        assert any(
            spanish_word in data["error"]["message"].lower()
            for spanish_word in ["inválidos", "verifica", "datos"]
        )

    def test_missing_parts_returns_422(self, client: TestClient):
        """Test missing required field returns 422."""
        response = client.post(
            "/api/msa/compute",
            json={
                "operators": ["Op1"],
                "trials": 2,
                "measurements": [[[1.0, 1.1]]],
            },
        )

        assert response.status_code == 422

    def test_missing_operators_returns_422(self, client: TestClient):
        """Test missing required field returns 422."""
        response = client.post(
            "/api/msa/compute",
            json={
                "parts": ["P1", "P2"],
                "trials": 2,
                "measurements": [[[1.0, 1.1]], [[2.0, 2.1]]],
            },
        )

        assert response.status_code == 422

    def test_missing_trials_returns_422(self, client: TestClient):
        """Test missing required field returns 422."""
        response = client.post(
            "/api/msa/compute",
            json={
                "parts": ["P1", "P2"],
                "operators": ["Op1"],
                "measurements": [[[1.0, 1.1]], [[2.0, 2.1]]],
            },
        )

        assert response.status_code == 422

    def test_missing_measurements_returns_422(self, client: TestClient):
        """Test missing required field returns 422."""
        response = client.post(
            "/api/msa/compute",
            json={
                "parts": ["P1", "P2"],
                "operators": ["Op1"],
                "trials": 2,
            },
        )

        assert response.status_code == 422

    def test_measurements_shape_mismatch_returns_422(self, client: TestClient):
        """Test measurements shape mismatch returns 422."""
        response = client.post(
            "/api/msa/compute",
            json={
                "parts": ["P1", "P2"],
                "operators": ["Op1", "Op2"],
                "trials": 2,
                "measurements": [
                    [[1.0, 1.1]],  # Missing second operator
                    [[2.0, 2.1]],  # Missing second operator
                ],
            },
        )

        assert response.status_code == 422

    def test_too_few_trials_returns_422(self, client: TestClient):
        """Test fewer than 2 trials returns 422."""
        response = client.post(
            "/api/msa/compute",
            json={
                "parts": ["P1", "P2"],
                "operators": ["Op1"],
                "trials": 1,  # Invalid: need at least 2
                "measurements": [[[1.0]], [[2.0]]],
            },
        )

        assert response.status_code == 422


class TestMSAEndpointEdgeCases:
    """Tests for edge cases."""

    def test_minimum_valid_dataset(self, client: TestClient):
        """Test minimum valid dataset: 2 parts, 1 operator, 2 trials."""
        response = client.post(
            "/api/msa/compute",
            json={
                "parts": ["P1", "P2"],
                "operators": ["Op1"],
                "trials": 2,
                "measurements": [
                    [[1.0, 1.1]],
                    [[2.0, 2.1]],
                ],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

    def test_single_operator_has_zero_reproducibility(self, client: TestClient):
        """Test single operator case returns reproducibility = 0."""
        response = client.post(
            "/api/msa/compute",
            json={
                "parts": ["P1", "P2", "P3", "P4", "P5"],
                "operators": ["Op1"],  # Single operator
                "trials": 3,
                "measurements": [
                    [[2.0, 2.1, 2.0]],
                    [[3.0, 3.1, 3.0]],
                    [[4.0, 4.1, 4.0]],
                    [[5.0, 5.1, 5.0]],
                    [[6.0, 6.1, 6.0]],
                ],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["data"]["reproducibility_percent"] == 0.0
        assert data["data"]["components"]["operator_variation"] == 0.0
