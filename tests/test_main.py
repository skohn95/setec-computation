"""Tests for main application."""


def test_app_starts(client):
    """App should start without errors."""
    response = client.get("/health")
    assert response.status_code == 200


def test_cors_headers_present(client):
    """CORS headers should be present in preflight responses."""
    response = client.options(
        "/health",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert "access-control-allow-origin" in response.headers


def test_msa_placeholder_returns_501(client):
    """MSA endpoint should return 501 Not Implemented."""
    response = client.post("/api/msa/compute", json={})
    assert response.status_code == 501
    data = response.json()
    assert data["status"] == "error"
    assert data["error"]["code"] == "NOT_IMPLEMENTED"


def test_control_charts_placeholder_returns_501(client):
    """Control charts endpoint should return 501 Not Implemented."""
    response = client.post("/api/control-charts/compute", json={})
    assert response.status_code == 501
    data = response.json()
    assert data["status"] == "error"
    assert data["error"]["code"] == "NOT_IMPLEMENTED"
