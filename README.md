# Iris ML API with Monitoring

[![CI](https://github.com/<your-github-username>/<your-repository>/actions/workflows/ci.yml/badge.svg)](https://github.com/<your-github-username>/<your-repository>/actions/workflows/ci.yml)

FastAPI service for Iris flower classification with MLOps monitoring features: Prometheus metrics, `/metrics`, drift detection through the Kolmogorov-Smirnov test, structured JSON logs, Docker, Docker Compose monitoring, and pytest coverage.

## Technology Stack

- Python 3.11
- FastAPI, Uvicorn, Pydantic
- scikit-learn, joblib, NumPy, SciPy
- prometheus-client
- python-json-logger
- pytest, httpx
- Docker, Docker Compose
- Prometheus
- GitHub Actions
- Evidently optional bonus report

## Repository Structure

```text
Lab2_MLOps/
|-- app/
|   |-- main.py
|   |-- schemas.py
|   |-- metrics.py
|   |-- drift.py
|   `-- logging_config.py
|-- ml/
|   `-- train.py
|-- monitoring/
|   |-- prometheus.yml
|   `-- docker-compose.monitoring.yml
|-- scripts/
|   `-- evidently_report.py
|-- tests/
|-- model.joblib
|-- reference_stats.joblib
|-- Dockerfile
|-- requirements.txt
`-- README.md
```

## Run Locally

Use Python 3.11 or 3.12. Avoid Python 3.14 for this lab because pinned scientific packages may not have matching wheels.

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python -m ml.train
uvicorn app.main:app --reload
```

Open:

```text
http://localhost:8000/docs
```

## API Endpoints

### Health

```http
GET /health
```

Example response:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "drift_detector_ready": true
}
```

### Prediction

```http
POST /predict
```

Example request:

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

### Metrics

```http
GET /metrics
```

Returns Prometheus exposition format.

### Drift Detection

```http
POST /check-drift
```

Example request:

```json
{
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
    [9.1, 8.1, 8.1, 5.1]
  ],
  "alpha": 0.05
}
```

## Implemented Metrics

- `ml_predictions_total{class_name,status}`: number of predictions by class and status.
- `ml_prediction_latency_seconds`: histogram of `/predict` latency.
- `ml_prediction_confidence`: histogram of selected-class probability.
- `ml_errors_total{error_type}`: number of API errors.
- `ml_model_loaded`: gauge, `1` when the model is loaded.
- `ml_drift_checks_total`: number of drift checks.
- `ml_drift_detected_total{feature}`: drift detections by feature.

## Prometheus Monitoring

Run the full monitoring stack:

```powershell
cd monitoring
docker-compose -f docker-compose.monitoring.yml up --build
```

Open:

```text
http://localhost:9090/targets
```

Expected result: target `ml-api` is `UP`.

Useful PromQL queries:

```promql
ml_predictions_total
rate(ml_predictions_total[1m])
sum by (class_name) (ml_predictions_total)
histogram_quantile(0.95, rate(ml_prediction_latency_seconds_bucket[5m]))
ml_drift_checks_total
ml_drift_detected_total
```

Add Prometheus screenshots here after running the monitoring stack:

```text
screenshots/prometheus-targets.png
screenshots/prometheus-predictions.png
screenshots/prometheus-latency.png
```

## Structured Logging

The service writes JSON logs to stdout for important events:

- startup completed
- prediction request
- drift check
- model/reference loading problems
- inference errors

Example event fields:

```json
{
  "event": "prediction",
  "class_name": "setosa",
  "probability": 0.9816
}
```

## Tests

```powershell
pytest -q
```

The tests cover:

- model training and artifact creation
- `/`, `/health`, `/predict`, `/check-drift`
- Prometheus `/metrics`
- KS-based drift detector behavior

## Docker

```powershell
docker build -t ml-api:lab3 .
docker run --rm -p 8000:8000 ml-api:lab3
```

## Evidently Bonus Report

Install optional dependencies:

```powershell
pip install evidently==0.4.40 pandas==2.2.3
```

Generate an HTML drift report:

```powershell
python scripts/evidently_report.py
```

Output:

```text
drift_report.html
```

## CI/CD

GitHub Actions runs on pushes and pull requests to `main`:

- installs dependencies
- trains the model and saves artifacts
- runs pytest
- builds the Docker image

## Deployment

Render deployment URL:

```text
https://your-render-service.onrender.com
```

Replace the placeholder after creating the Render Web Service from this repository.

## Conclusions

Lab 3 turns the basic ML API from Lab 2 into an observable service. Prometheus metrics show operational behavior and model-specific behavior, JSON logs preserve event context, and the KS-test based drift endpoint detects when live input distributions differ from the training reference data.
