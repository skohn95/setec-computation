"""Tests for health check endpoint."""


def test_health_check_returns_healthy(client):
    """Health check should return healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == "1.0.0"
    assert data["service"] == "setec-computation"


def test_health_check_response_format(client):
    """Health check response should have expected structure."""
    response = client.get("/health")
    data = response.json()
    assert "status" in data
    assert "version" in data
    assert "service" in data
