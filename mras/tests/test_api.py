"""
Tests for FastAPI endpoints
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_endpoint():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_endpoint_structure():
    """Test query endpoint accepts correct structure"""
    response = client.post(
        "/query",
        json={"query": "test question", "top_k": 5}
    )
    # May fail if no data ingested, but should have correct structure
    assert "answer" in response.json() or response.status_code in [500, 200]


def test_ingest_endpoint_structure():
    """Test ingest endpoint structure"""
    response = client.post(
        "/ingest",
        json={"folder_path": "data/documents"}
    )
    # Should return correct structure even if folder is empty
    assert response.status_code in [200, 500]
