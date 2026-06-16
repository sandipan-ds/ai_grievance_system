from __future__ import annotations

import os
import pytest
from fastapi.testclient import TestClient

# Set mock models env variable just to be double safe
os.environ["MOCK_MODELS"] = "1"

from src.main import app
from src.ml.model_loader import load_model_bundle
from src.ml.predictor import preprocess_text, predict_complaint, predict_department, predict_severity


@pytest.fixture(scope="module")
def client():
    """Fixture that initializes the FastAPI application lifespan context during testing."""
    with TestClient(app) as c:
        yield c


def test_preprocess_text():
    # Test lowercasing, links removal, special chars removal, and lemmatization/stopwords
    raw_text = "Check out http://google.com for some sewage water leaking on 4th block!!"
    cleaned = preprocess_text(raw_text)
    
    assert "http" not in cleaned
    assert "google" not in cleaned
    assert "!!" not in cleaned
    assert "sewage" in cleaned
    assert "leak" in cleaned or "leaking" in cleaned


def test_model_loader_mock():
    # Verify loader returns mock bundle in testing environment
    bundle = load_model_bundle()
    assert bundle.is_mock is True
    assert bundle.device == "cpu"


def test_predictor_mock():
    # Verify predictor works with mock models
    bundle = load_model_bundle()
    complaint = "Leaking water pipe in the street"
    
    dept = predict_department(bundle, complaint)
    assert dept in ["BBMP", "BCP", "BESCOM", "BTP", "BWSSB", "KSFES", "KSPCB", "Transport"]
    
    sev = predict_severity(bundle, complaint)
    assert sev in ["non-grievance", "low", "medium", "high", "critical"]
    
    d, s = predict_complaint(bundle, complaint)
    assert d == dept
    assert s == sev


def test_api_health(client):
    # Verify GET /health route
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["models_loaded"] is True
    assert data["is_mock"] is True


def test_api_predict(client):
    # Verify POST /predict with a valid payload
    payload = {"complaint": "Heavy smoke coming from factory chimneys near HSR."}
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert "predicted_department" in data
    assert "severity" in data
    assert data["severity"] in ["non-grievance", "low", "medium", "high", "critical"]


def test_api_predict_validation(client):
    # Verify POST /predict validation error on empty payload
    payload = {"complaint": "  "}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Pydantic validation error


def test_api_analytics(client):
    # Verify GET /analytics route returns structured metrics
    response = client.get("/analytics")
    assert response.status_code == 200
    data = response.json()
    
    assert "total_complaints" in data
    assert "active_complaints" in data
    assert "critical_complaints" in data
    assert "recent_complaints" in data
    assert "department_labels" in data
    assert "department_distribution" in data
    assert "severity_labels" in data
    assert "severity_distribution" in data
    assert "chart_matrix" in data
    
    assert len(data["recent_complaints"]) > 0
    assert len(data["department_labels"]) == 8
    assert len(data["severity_labels"]) == 5
