# Plan for Lab 3: ML API Monitoring and Problem Detection

## Goal

Extend the Lab 2 Iris ML API into an observable ML service. Add Prometheus metrics, a `/metrics` endpoint, drift detection with the Kolmogorov-Smirnov test, structured JSON logging, local Prometheus monitoring through Docker Compose, extra tests, and README documentation.

## Starting Point

Use the existing Lab 2 project as the base:

- FastAPI API in `app/main.py`
- Pydantic schemas in `app/schemas.py`
- Iris training script in `ml/train.py`
- Tests in `tests/`
- Dockerfile and GitHub Actions CI

Recommended workflow:

```powershell
git checkout -b lab3-monitoring
```

If the project is not yet a Git repository, initialize/push it first, then create the branch.

## Target Project Structure

```text
Lab2_MLOps/
|-- .github/
|   `-- workflows/
|       `-- ci.yml
|-- app/
|   |-- __init__.py
|   |-- main.py
|   |-- schemas.py
|   |-- metrics.py
|   |-- drift.py
|   `-- logging_config.py
|-- ml/
|   |-- __init__.py
|   `-- train.py
|-- monitoring/
|   |-- prometheus.yml
|   `-- docker-compose.monitoring.yml
|-- tests/
|   |-- __init__.py
|   |-- test_api.py
|   |-- test_model.py
|   |-- test_metrics.py
|   `-- test_drift.py
|-- scripts/
|   `-- evidently_report.py
|-- model.joblib
|-- reference_stats.joblib
|-- requirements.txt
|-- Dockerfile
|-- .dockerignore
|-- README.md
|-- plan.md
`-- plan_lab3.md
```

## Implementation Checklist

### 1. Update dependencies

Update `requirements.txt` with the Lab 3 monitoring dependencies:

```text
fastapi==0.115.0
uvicorn[standard]==0.30.6
pydantic==2.9.2
scikit-learn==1.5.2
joblib==1.4.2
numpy==1.26.4
pytest==8.3.3
httpx==0.27.2
prometheus-client==0.21.0
scipy==1.14.1
python-json-logger==2.0.7
evidently==0.4.40
pandas==2.2.3
```

Notes:

- `prometheus-client` is used for counters, histograms, gauges, and the `/metrics` output.
- `scipy` is required for `scipy.stats.ks_2samp`.
- `python-json-logger` is used for structured JSON logs.
- `evidently` and `pandas` are optional bonus dependencies for the HTML drift report.

Install:

```powershell
pip install -r requirements.txt
```

### 2. Add Prometheus metrics module

Create `app/metrics.py`.

Define a custom `CollectorRegistry` and these metrics:

- `ml_predictions_total`: `Counter` with labels `class_name` and `status`.
- `ml_prediction_latency_seconds`: `Histogram` for prediction latency.
- `ml_prediction_confidence`: `Histogram` for `predict_proba` confidence values.
- `ml_errors_total`: `Counter` with label `error_type`.
- `ml_model_loaded`: `Gauge`, `1` when the model is loaded, `0` otherwise.
- `ml_drift_checks_total`: `Counter` for all drift checks.
- `ml_drift_detected_total`: `Counter` with label `feature`.

Important: define metrics at module level, not inside request handlers, because Prometheus metric objects are registered globally.

### 3. Add structured JSON logging

Create `app/logging_config.py`.

Use `pythonjsonlogger.JsonFormatter` with fields:

- `timestamp`
- `level`
- `logger`
- `message`
- extra fields such as `event`, `class_name`, `probability`, `features`, `drift_detected`

Configure logs to go to `stdout`, because Docker/Render/CI tools collect standard output naturally.

### 4. Add drift detector

Create `app/drift.py`.

Implement `DriftDetector`:

- Constructor accepts:
  - `reference: np.ndarray`
  - `feature_names: list[str]`
- Validate that reference data is two-dimensional.
- Validate that `len(feature_names)` matches the number of columns.
- Method `detect(current: np.ndarray, alpha: float = 0.05) -> dict`.
- For each feature, run:

```python
stats.ks_2samp(reference_column, current_column)
```

- Mark a feature as drifted when `p_value < alpha`.
- Return:
  - `drift_detected`
  - `n_drifted_features`
  - `drifted_features`
  - `per_feature`
  - `n_samples`
  - `alpha`

### 5. Extend Pydantic schemas

Update `app/schemas.py`.

Keep existing:

- `IrisFeatures`
- `PredictionResponse`

Add:

- `DriftRequest`
  - `samples`: list of observations
  - each observation must contain exactly 4 float values
  - request must contain at least 10 samples
  - `alpha`: float, default `0.05`, allowed range `0.001` to `0.5`
- `FeatureDriftInfo`
  - `statistic`
  - `p_value`
  - `drift_detected`
- `DriftResponse`
  - full drift result returned by `DriftDetector`

### 6. Update model training

Update `ml/train.py`.

It should now save two artifacts:

- `model.joblib`
- `reference_stats.joblib`

Store reference data as:

```python
{
    "X": X_train,
    "feature_names": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
}
```

Run:

```powershell
python -m ml.train
```

Expected result:

- `model.joblib` exists.
- `reference_stats.joblib` exists.
- accuracy is still above `0.8`.

### 7. Update FastAPI application

Update `app/main.py`.

New behavior:

- Load `model.joblib` at startup.
- Load `reference_stats.joblib` at startup.
- Initialize `DriftDetector`.
- Set `MODEL_LOADED` gauge to `1` after successful model loading.
- Add structured logs for startup, predictions, drift checks, and errors.

Endpoints:

- `GET /`
  - return service info and version `2.0.0`.
- `GET /health`
  - include `model_loaded`.
  - include `drift_detector_ready`.
- `GET /metrics`
  - return Prometheus exposition format using `generate_latest(REGISTRY)`.
- `POST /predict`
  - perform the prediction.
  - increment `ml_predictions_total`.
  - observe `ml_prediction_confidence`.
  - observe prediction latency.
  - log prediction details as JSON.
- `POST /check-drift`
  - accept a batch of samples.
  - run `DriftDetector.detect`.
  - increment `ml_drift_checks_total`.
  - increment `ml_drift_detected_total` for every drifted feature.
  - log drift check result as JSON.

### 8. Add Prometheus configuration

Create `monitoring/prometheus.yml`.

Scrape targets:

- `ml-api:8000` with `metrics_path: /metrics`
- `localhost:9090` for Prometheus self-monitoring

Use `scrape_interval: 10s`.

### 9. Add Docker Compose monitoring setup

Create `monitoring/docker-compose.monitoring.yml`.

Services:

- `ml-api`
  - build from project root
  - expose port `8000`
- `prometheus`
  - image `prom/prometheus:v2.55.0`
  - expose port `9090`
  - mount `monitoring/prometheus.yml`
  - depends on `ml-api`

Run:

```powershell
cd monitoring
docker-compose -f docker-compose.monitoring.yml up --build
```

Check:

- API: `http://localhost:8000/docs`
- metrics: `http://localhost:8000/metrics`
- Prometheus: `http://localhost:9090`
- targets: `http://localhost:9090/targets`

### 10. Add tests

Add `tests/test_metrics.py`:

- `GET /metrics` returns `200`.
- response contains `ml_predictions_total`.
- response contains `ml_prediction_latency_seconds`.
- after `/predict`, metrics output changes and contains a success counter for `setosa`.

Add `tests/test_drift.py`:

- no drift for two samples generated from the same distribution.
- drift detected for a strongly shifted distribution.
- invalid shape raises `ValueError`.

Update existing tests if needed so they account for:

- `reference_stats.joblib`
- `drift_detector_ready` in `/health`
- new startup behavior

Run:

```powershell
pytest -q
```

### 11. Verify manually

Start API:

```powershell
uvicorn app.main:app --reload
```

Test prediction:

```powershell
curl -X POST http://localhost:8000/predict `
  -H "Content-Type: application/json" `
  -d "{\"sepal_length\":5.1,\"sepal_width\":3.5,\"petal_length\":1.4,\"petal_width\":0.2}"
```

Check metrics:

```powershell
curl http://localhost:8000/metrics
```

Check healthy drift batch:

```powershell
curl -X POST http://localhost:8000/check-drift `
  -H "Content-Type: application/json" `
  -d "{\"samples\":[[5.1,3.5,1.4,0.2],[4.9,3.0,1.4,0.2],[4.7,3.2,1.3,0.2],[5.4,3.9,1.7,0.4],[5.0,3.6,1.4,0.2],[5.5,2.5,4.0,1.3],[6.1,2.9,4.7,1.4],[6.0,3.0,4.8,1.8],[6.3,2.5,5.0,1.9],[6.5,3.0,5.2,2.0]],\"alpha\":0.05}"
```

Check drifted batch:

```powershell
curl -X POST http://localhost:8000/check-drift `
  -H "Content-Type: application/json" `
  -d "{\"samples\":[[9.0,8.0,8.0,5.0],[9.5,7.5,8.5,5.5],[8.5,8.5,7.5,4.5],[9.2,8.2,8.2,5.2],[9.8,7.8,8.8,5.8],[8.8,8.8,7.8,4.8],[9.4,8.4,8.4,5.4],[9.6,7.6,8.6,5.6],[8.6,8.6,7.6,4.6],[9.1,8.1,8.1,5.1]],\"alpha\":0.05}"
```

### 12. Run Prometheus monitoring

Start the monitoring stack:

```powershell
cd monitoring
docker-compose -f docker-compose.monitoring.yml up --build
```

Open:

```text
http://localhost:9090/targets
```

Expected: target `ml-api` is `UP`.

Useful PromQL queries:

```promql
ml_predictions_total
rate(ml_predictions_total[1m])
sum by (class_name) (ml_predictions_total)
histogram_quantile(0.95, rate(ml_prediction_latency_seconds_bucket[5m]))
ml_drift_checks_total
ml_drift_detected_total
```

Take screenshots for the README/report.

### 13. Optional Evidently report

Create `scripts/evidently_report.py`.

The script should:

- Load `reference_stats.joblib`.
- Convert reference data into a pandas DataFrame.
- Create a simulated current dataset with an artificial shift.
- Generate an Evidently `DataDriftPreset` report.
- Save it as `drift_report.html`.

Run:

```powershell
python scripts/evidently_report.py
```

### 14. Update Dockerfile and CI

The existing Dockerfile should still work because `python -m ml.train` will now generate both artifacts.

Verify Docker build:

```powershell
docker build -t ml-api:lab3 .
```

CI workflow can remain mostly the same:

- install dependencies
- run `python -m ml.train`
- run `pytest -q`
- build Docker image

If Evidently causes slow or fragile CI installs, consider keeping it optional and documenting it as a bonus step.

### 15. Update README report

Update `README.md` with Lab 3 sections:

- System overview.
- New endpoints:
  - `/metrics`
  - `/check-drift`
- Implemented Prometheus metrics.
- Drift detection explanation.
- JSON logging explanation.
- Local run instructions.
- Prometheus/docker-compose run instructions.
- PromQL examples.
- Screenshots from Prometheus.
- Test instructions.
- Conclusions.

### 16. Final submission checklist

- `python -m ml.train` creates:
  - `model.joblib`
  - `reference_stats.joblib`
- `pytest -q` passes.
- `/predict` works.
- `/metrics` works.
- `/check-drift` works for normal and drifted samples.
- Prometheus target is `UP`.
- README contains screenshots and explanations.
- GitHub Actions workflow is green.
- Repository is pushed to GitHub.

## Control Questions to Prepare

1. Difference between monitoring a classic web service and monitoring an ML service.
2. Why latency and error rate are not enough for ML monitoring.
3. How Prometheus pull-model scraping works.
4. What happens when `/metrics` returns `500` or the API is unavailable.
5. Difference between `Counter`, `Gauge`, and `Histogram`.
6. What data drift is and why it can silently damage model quality.
7. Three possible causes of data drift.
8. How the Kolmogorov-Smirnov test works.
9. What p-value means in the KS-test result.
10. Why structured JSON logs are useful for ML APIs.
