"""Tests for main application."""

from fastapi.testclient import TestClient

from app.main import app


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


def test_msa_endpoint_returns_422_for_empty_body(client):
    """MSA endpoint should return 422 for empty request body (missing required fields)."""
    response = client.post("/api/msa/compute", json={})
    assert response.status_code == 422
    data = response.json()
    assert data["status"] == "error"
    assert data["error"]["code"] == "VALIDATION_ERROR"


def test_control_charts_placeholder_returns_501(client):
    """Control charts endpoint should return 501 Not Implemented."""
    response = client.post("/api/control-charts/compute", json={})
    assert response.status_code == 501
    data = response.json()
    assert data["status"] == "error"
    assert data["error"]["code"] == "NOT_IMPLEMENTED"


def test_request_validation_error_returns_422():
    """Request validation errors should return 422 with Spanish message."""
    from pydantic import BaseModel

    # Create a test endpoint with request validation
    class TestInput(BaseModel):
        value: int

    @app.post("/test-validation")
    async def validate_input(data: TestInput):
        return {"value": data.value}

    client = TestClient(app, raise_server_exceptions=False)
    # Send invalid data type to trigger validation error
    response = client.post(
        "/test-validation",
        json={"value": "not an integer"},
    )
    assert response.status_code == 422
    data = response.json()
    assert data["status"] == "error"
    assert data["error"]["code"] == "VALIDATION_ERROR"
    assert "Datos de entrada inválidos" in data["error"]["message"]


def test_generic_exception_handler_returns_500():
    """Generic exceptions should return 500 with Spanish message."""
    # Create a test route that raises an exception
    @app.get("/test-exception")
    async def raise_exception():
        raise RuntimeError("Test error")

    client = TestClient(app, raise_server_exceptions=False)
    response = client.get("/test-exception")
    assert response.status_code == 500
    data = response.json()
    assert data["status"] == "error"
    assert data["error"]["code"] == "INTERNAL_ERROR"
    assert "Error interno del servidor" in data["error"]["message"]


def test_value_error_handler_returns_400():
    """ValueError should return 400 with Spanish message."""
    # Create a test route that raises ValueError
    @app.get("/test-value-error")
    async def raise_value_error():
        raise ValueError("Invalid input data")

    client = TestClient(app, raise_server_exceptions=False)
    response = client.get("/test-value-error")
    assert response.status_code == 400
    data = response.json()
    assert data["status"] == "error"
    assert data["error"]["code"] == "VALIDATION_ERROR"
    assert "Datos de entrada inválidos" in data["error"]["message"]
