# Plan for Lab 2: CI/CD and ML API

## Goal

Build a small end-to-end MLOps project: train an Iris classifier, expose it through a FastAPI `/predict` endpoint, cover the model and API with pytest tests, containerize the app with Docker, run CI with GitHub Actions, deploy the service to Render, and document everything in `README.md`.

## Required Project Structure

```text
Lab2_MLOps/
|-- .github/
|   `-- workflows/
|       `-- ci.yml
|-- app/
|   |-- __init__.py
|   |-- main.py
|   `-- schemas.py
|-- ml/
|   |-- __init__.py
|   `-- train.py
|-- tests/
|   |-- __init__.py
|   |-- test_api.py
|   `-- test_model.py
|-- model.joblib
|-- requirements.txt
|-- Dockerfile
|-- .dockerignore
|-- README.md
`-- plan.md
```

## Implementation Checklist

### 1. Prepare the repository and environment

- Create or confirm a public GitHub repository for the lab.
- Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

- Create the `app`, `ml`, `tests`, and `.github/workflows` directories.
- Add empty `__init__.py` files to `app`, `ml`, and `tests`.

### 2. Add Python dependencies

- Create `requirements.txt` with:

```text
fastapi==0.115.0
uvicorn[standard]==0.30.6
pydantic==2.9.2
scikit-learn==1.5.2
joblib==1.4.2
pytest==8.3.3
httpx==0.27.2
numpy
```

- Install dependencies:

```powershell
pip install -r requirements.txt
```

### 3. Implement model training

- Create `ml/train.py`.
- Use `sklearn.datasets.load_iris`.
- Split the dataset with `train_test_split(test_size=0.2, random_state=42, stratify=y)`.
- Train `LogisticRegression(max_iter=1000)`.
- Calculate accuracy on the test split.
- Save the trained model to `model.joblib` with `joblib.dump`.
- Expose a reusable function `train_and_save(model_path=MODEL_PATH) -> float`.
- Verify locally:

```powershell
python -m ml.train
```

Expected result: `model.joblib` appears in the project root and reported accuracy is above `0.8`.

### 4. Implement API schemas

- Create `app/schemas.py`.
- Add `IrisFeatures` Pydantic model with four float fields:
  - `sepal_length`
  - `sepal_width`
  - `petal_length`
  - `petal_width`
- Add validation bounds with `Field(..., ge=0, le=10)`.
- Add `PredictionResponse` with:
  - `class_id: int`
  - `class_name: str`
  - `probability: float`

### 5. Implement FastAPI app

- Create `app/main.py`.
- Load `model.joblib` once during application startup.
- Define class names: `setosa`, `versicolor`, `virginica`.
- Add endpoints:
  - `GET /` returns service status.
  - `GET /health` returns health status and whether the model is loaded.
  - `POST /predict` accepts `IrisFeatures` and returns `PredictionResponse`.
- Use `numpy.array([[...]])` so scikit-learn receives a two-dimensional input.
- Return HTTP `503` if prediction is requested while the model is unavailable.
- Run locally:

```powershell
uvicorn app.main:app --reload
```

- Check:
  - `http://localhost:8000/`
  - `http://localhost:8000/health`
  - `http://localhost:8000/docs`

### 6. Add tests

- Create `tests/test_model.py`:
  - Verify `train_and_save` creates a model file.
  - Verify returned accuracy is between `0.0` and `1.0`.
  - Verify accuracy is above `0.8`.
  - Verify the trained model predicts one of classes `0`, `1`, or `2`.

- Create `tests/test_api.py`:
  - Ensure `model.joblib` exists before API tests by calling `train_and_save` if needed.
  - Use `fastapi.testclient.TestClient`.
  - Test `GET /`.
  - Test `GET /health`.
  - Test valid `POST /predict` payload for a typical setosa sample.
  - Test invalid input returns HTTP `422`.

- Run tests:

```powershell
pytest -q
```

### 7. Add Docker support

- Create `Dockerfile` based on `python:3.11-slim`.
- Set:
  - `PYTHONDONTWRITEBYTECODE=1`
  - `PYTHONUNBUFFERED=1`
  - `PIP_NO_CACHE_DIR=1`
- Copy `requirements.txt` first and install dependencies.
- Copy `app` and `ml`.
- Run `python -m ml.train` during image build so `model.joblib` is included.
- Start the app with:

```text
uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
```

- Create `.dockerignore` excluding cache files, `.git`, `.github`, tests, markdown files, and environment files.
- Verify locally:

```powershell
docker build -t ml-api:lab2 .
docker run --rm -p 8000:8000 ml-api:lab2
```

- Check `http://localhost:8000/health`.

### 8. Configure GitHub Actions CI

- Create `.github/workflows/ci.yml`.
- Trigger on:
  - `push` to `main`
  - `pull_request` to `main`
- Add a `test` job:
  - Checkout repository.
  - Set up Python `3.11`.
  - Cache pip dependencies.
  - Install `requirements.txt`.
  - Run `python -m ml.train`.
  - Run `pytest -q`.
- Add a `docker-build` job after tests:
  - Checkout repository.
  - Run `docker build -t ml-api:ci .`.
- Push to GitHub and confirm the workflow is green.

### 9. Deploy to Render

- Log in to Render with GitHub.
- Create a new Web Service from the repository.
- Choose Docker environment so Render uses the `Dockerfile`.
- Select the free instance type.
- Wait for the first build and deployment.
- Save the public Render URL.
- Verify:

```powershell
curl https://lab2-ml-api.onrender.com/health
```

Expected response:

```json
{"status":"healthy","model_loaded":true}
```

### 10. Write the README report

Create `README.md` with these sections:

- Project description.
- Technology stack.
- Repository structure.
- Local setup and run instructions.
- Docker build and run instructions.
- Test instructions.
- API documentation and example `/predict` request.
- GitHub Actions badge.
- Render deployment link.
- Short explanation of CI/CD workflow.

### 11. Final verification before submission

- Run `pytest -q` locally.
- Confirm `python -m ml.train` creates or updates `model.joblib`.
- Confirm the API starts locally with Uvicorn.
- Confirm Docker image builds.
- Confirm GitHub Actions workflow is successful.
- Confirm Render `/health` and `/docs` are reachable.
- Submit:
  - GitHub repository URL.
  - Render deployment URL.

## Control Questions to Prepare

1. Difference between Continuous Integration and Continuous Deployment.
2. Relationship between Workflow, Job, and Step in GitHub Actions.
3. Why the model is loaded during FastAPI startup instead of inside every `/predict` request.
4. Why `COPY requirements.txt` happens before copying source code in the Dockerfile.
5. How FastAPI and Pydantic return HTTP `422` for invalid `/predict` input.
