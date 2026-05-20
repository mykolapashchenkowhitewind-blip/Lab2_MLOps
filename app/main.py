import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from .drift import DriftDetector
from .logging_config import setup_logging
from .metrics import (
    DRIFT_CHECKS,
    DRIFT_DETECTED,
    ERROR_COUNTER,
    MODEL_LOADED,
    PREDICTION_CONFIDENCE,
    PREDICTION_COUNTER,
    PREDICTION_LATENCY,
    REGISTRY,
)
from .schemas import DriftRequest, DriftResponse, IrisFeatures, PredictionResponse


MODEL_PATH = Path(__file__).resolve().parent.parent / "model.joblib"
REFERENCE_PATH = Path(__file__).resolve().parent.parent / "reference_stats.joblib"
CLASS_NAMES = ["setosa", "versicolor", "virginica"]

setup_logging()
logger = logging.getLogger("ml-api")

model = None
drift_detector: DriftDetector | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global model, drift_detector

    if not MODEL_PATH.exists():
        MODEL_LOADED.set(0)
        logger.error(
            "model_file_missing",
            extra={"event": "startup_error", "model_path": str(MODEL_PATH)},
        )
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)
    MODEL_LOADED.set(1)

    if REFERENCE_PATH.exists():
        reference_data = joblib.load(REFERENCE_PATH)
        drift_detector = DriftDetector(
            reference=np.asarray(reference_data["X"]),
            feature_names=reference_data["feature_names"],
        )
        drift_ready = True
    else:
        drift_detector = None
        drift_ready = False
        logger.warning(
            "reference_file_missing",
            extra={"event": "startup_warning", "reference_path": str(REFERENCE_PATH)},
        )

    logger.info(
        "startup_complete",
        extra={
            "event": "startup",
            "model_loaded": model is not None,
            "drift_detector_ready": drift_ready,
        },
    )

    yield


app = FastAPI(
    title="Iris ML API with Monitoring",
    description="REST API for Iris flower classification with Prometheus metrics and drift detection",
    version="2.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start

    if request.url.path == "/predict":
        PREDICTION_LATENCY.observe(elapsed)
        if response.status_code >= 400:
            PREDICTION_COUNTER.labels(class_name="unknown", status="error").inc()

    if response.status_code >= 500:
        ERROR_COUNTER.labels(error_type=f"http_{response.status_code}").inc()

    return response


@app.get("/")
def root() -> dict[str, str]:
    return {"status": "ok", "service": "Iris ML API", "version": "2.0.0"}


@app.get("/health")
def health() -> dict[str, bool | str]:
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "drift_detector_ready": drift_detector is not None,
    }


@app.get("/metrics")
def metrics() -> Response:
    data = generate_latest(REGISTRY)
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionResponse)
def predict(features: IrisFeatures) -> PredictionResponse:
    if model is None:
        ERROR_COUNTER.labels(error_type="model_not_loaded").inc()
        raise HTTPException(status_code=503, detail="Model is not loaded")

    x = np.array(
        [
            [
                features.sepal_length,
                features.sepal_width,
                features.petal_length,
                features.petal_width,
            ]
        ]
    )

    try:
        class_id = int(model.predict(x)[0])
        probability = float(model.predict_proba(x)[0, class_id])
    except Exception as exc:
        ERROR_COUNTER.labels(error_type="inference_error").inc()
        logger.exception("inference_failed", extra={"event": "inference_error"})
        raise HTTPException(status_code=500, detail="Inference error") from exc

    class_name = CLASS_NAMES[class_id]
    PREDICTION_COUNTER.labels(class_name=class_name, status="success").inc()
    PREDICTION_CONFIDENCE.observe(probability)

    logger.info(
        "prediction",
        extra={
            "event": "prediction",
            "class_id": class_id,
            "class_name": class_name,
            "probability": round(probability, 4),
            "features": features.model_dump(),
        },
    )

    return PredictionResponse(
        class_id=class_id,
        class_name=class_name,
        probability=round(probability, 4),
    )


@app.post("/check-drift", response_model=DriftResponse)
def check_drift(payload: DriftRequest) -> DriftResponse:
    if drift_detector is None:
        ERROR_COUNTER.labels(error_type="drift_detector_not_ready").inc()
        raise HTTPException(status_code=503, detail="Drift detector is not ready")

    DRIFT_CHECKS.inc()
    current = np.asarray(payload.samples)

    try:
        result = drift_detector.detect(current, alpha=payload.alpha)
    except ValueError as exc:
        ERROR_COUNTER.labels(error_type="invalid_drift_payload").inc()
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    for feature, info in result["per_feature"].items():
        if info["drift_detected"]:
            DRIFT_DETECTED.labels(feature=feature).inc()

    logger.info(
        "drift_check",
        extra={
            "event": "drift_check",
            "n_samples": len(payload.samples),
            "alpha": payload.alpha,
            "drift_detected": result["drift_detected"],
            "drifted_features": result["drifted_features"],
        },
    )

    return DriftResponse(**result)
