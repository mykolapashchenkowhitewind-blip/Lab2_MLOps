from fastapi.testclient import TestClient

from app.main import MODEL_PATH, REFERENCE_PATH, app
from ml.train import train_and_save


if not MODEL_PATH.exists() or not REFERENCE_PATH.exists():
    train_and_save(MODEL_PATH, REFERENCE_PATH)


def test_root_endpoint():
    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["version"] == "2.0.0"


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["model_loaded"] is True
    assert body["drift_detector_ready"] is True


def test_predict_setosa():
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }

    with TestClient(app) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["class_name"] == "setosa"
    assert 0.0 <= body["probability"] <= 1.0


def test_predict_invalid_input():
    payload = {"sepal_length": "not-a-number"}

    with TestClient(app) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_check_drift_endpoint_detects_shift():
    payload = {
        "samples": [
            [9.0, 8.0, 8.0, 5.0],
            [9.5, 7.5, 8.5, 5.5],
            [8.5, 8.5, 7.5, 4.5],
            [9.2, 8.2, 8.2, 5.2],
            [9.8, 7.8, 8.8, 5.8],
            [8.8, 8.8, 7.8, 4.8],
            [9.4, 8.4, 8.4, 5.4],
            [9.6, 7.6, 8.6, 5.6],
            [8.6, 8.6, 7.6, 4.6],
            [9.1, 8.1, 8.1, 5.1],
        ],
        "alpha": 0.05,
    }

    with TestClient(app) as client:
        response = client.post("/check-drift", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["drift_detected"] is True
    assert body["n_drifted_features"] >= 1
