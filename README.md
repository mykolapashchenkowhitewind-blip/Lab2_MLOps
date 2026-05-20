# Iris ML API

[![CI](https://github.com/<your-github-username>/<your-repository>/actions/workflows/ci.yml/badge.svg)](https://github.com/<your-github-username>/<your-repository>/actions/workflows/ci.yml)

REST API for Iris flower classification. The project trains a Logistic Regression model with scikit-learn, serves predictions with FastAPI, verifies behavior with pytest, builds as a Docker image, and runs checks in GitHub Actions.

## Technology Stack

- Python 3.11
- FastAPI
- Uvicorn
- Pydantic
- scikit-learn
- joblib
- pytest
- Docker
- GitHub Actions
- Render

## Repository Structure

```text
Lab2_MLOps/
|-- .github/workflows/ci.yml
|-- app/
|   |-- main.py
|   `-- schemas.py
|-- ml/
|   `-- train.py
|-- tests/
|   |-- test_api.py
|   `-- test_model.py
|-- model.joblib
|-- requirements.txt
|-- Dockerfile
|-- .dockerignore
|-- README.md
`-- plan.md
```

## Run Locally

Use Python 3.11 or 3.12. The dependency pins are intended for the same runtime used by Docker and GitHub Actions, so avoid creating the environment with Python 3.14.

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Train the model:

```powershell
python -m ml.train
```

Start the API:

```powershell
uvicorn app.main:app --reload
```

Open the interactive API docs:

```text
http://localhost:8000/docs
```

## API

Health check:

```http
GET /health
```

Prediction:

```http
POST /predict
Content-Type: application/json
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

Example response:

```json
{
  "class_id": 0,
  "class_name": "setosa",
  "probability": 0.98
}
```

## Run Tests

```powershell
pytest -q
```

## Docker

Build the image:

```powershell
docker build -t ml-api:lab2 .
```

Run the container:

```powershell
docker run --rm -p 8000:8000 ml-api:lab2
```

Check the service:

```powershell
curl http://localhost:8000/health
```

## CI/CD

The GitHub Actions workflow in `.github/workflows/ci.yml` runs on every push and pull request to `main`.

It performs two jobs:

- `test`: installs dependencies, trains the model, and runs `pytest`.
- `docker-build`: builds the Docker image after tests pass.

## Deployment

Render deployment URL:

```text
https://your-render-service.onrender.com
```

After deployment, verify:

```powershell
curl https://your-render-service.onrender.com/health
```

## Notes

- Replace the GitHub badge URL placeholders with the real repository owner and name after pushing to GitHub.
- Replace the Render URL placeholder after creating the Render Web Service.
