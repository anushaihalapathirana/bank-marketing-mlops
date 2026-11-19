import numpy as np
import pytest

from fastapi.testclient import TestClient
from src import api

# Mock model
class MockModel:
    def predict(self, X):
        return np.array([1])  # Always predict class 1

    def predict_proba(self, X):
        return np.array([[0.2, 0.8]])  # Always return a probability of 0.8 for class 1
    
@pytest.fixture
def client(monkeypatch):
    # set a dummy model URI for testing
    monkeypatch.setenv("MODEL_URI", "models:/test-model")
    
    # replace MLFlow model loading with a mock model
    monkeypatch.setattr(api.mlflow.sklearn, "load_model", lambda uri: MockModel())
    
    #Start and stop the FastAPI app with the lifespan
    with TestClient(api.app) as test_client:
        yield test_client
        
# fixture provides reusable test data or setup. When a test function includes valid_customer as a parameter, pytest automatically calls the fixture and supplies its result
@pytest.fixture
def valid_customer():
    return {
        "age": 36,
        "job": "housemaid",
        "marital": "married",
        "education": "basic.4y",
        "default": "yes",
        "housing": "yes",
        "loan": "yes",
        "contact": "telephone",
        "month": "may",
        "day_of_week": "mon",
        "campaign": 1,
        "pdays": 999,
        "previous": 1,
        "poutcome": "nonexistent",
        "emp.var.rate": 1.1,
        "cons.price.idx": 93.994,
        "cons.conf.idx": -36.4,
        "euribor3m": 4.857,
        "nr.employed": 5191.0
    }
    
def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    
def test_valid_prediction(client, valid_customer):
    response = client.post("/predict", json=valid_customer)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_class"] == 1
    assert data["probability"] == 0.8
    
def test_invalid_age(client, valid_customer):
    # Create an invalid customer data dictionary missing a required field
    invalid_customer = valid_customer.copy()
    invalid_customer["age"] = "age"

    response = client.post("/predict", json=invalid_customer)
    assert response.status_code == 422  # Unprocessable Entity
    
def test_missing_field(client, valid_customer):
    # Create an invalid customer data dictionary missing a required field
    invalid_customer = valid_customer.copy()
    del invalid_customer["job"]

    response = client.post("/predict", json=invalid_customer)
    assert response.status_code == 422  # Unprocessable Entity
    
def test_extra_field(client, valid_customer):
    # Create an invalid customer data dictionary with an extra field
    invalid_customer = valid_customer.copy()
    invalid_customer["extra_field"] = "extra_value"

    response = client.post("/predict", json=invalid_customer)
    assert response.status_code == 422  # Unprocessable Entity  
    
