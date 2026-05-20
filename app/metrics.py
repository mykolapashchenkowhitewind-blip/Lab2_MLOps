from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram


REGISTRY = CollectorRegistry()

PREDICTION_COUNTER = Counter(
    "ml_predictions_total",
    "Total number of model prediction requests",
    labelnames=["class_name", "status"],
    registry=REGISTRY,
)

PREDICTION_LATENCY = Histogram(
    "ml_prediction_latency_seconds",
    "Prediction request latency in seconds",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
    registry=REGISTRY,
)

PREDICTION_CONFIDENCE = Histogram(
    "ml_prediction_confidence",
    "Distribution of predict_proba values for the selected class",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0),
    registry=REGISTRY,
)

ERROR_COUNTER = Counter(
    "ml_errors_total",
    "Total number of API errors",
    labelnames=["error_type"],
    registry=REGISTRY,
)

MODEL_LOADED = Gauge(
    "ml_model_loaded",
    "Whether the model is loaded: 1 for loaded, 0 otherwise",
    registry=REGISTRY,
)

DRIFT_CHECKS = Counter(
    "ml_drift_checks_total",
    "Total number of drift checks",
    registry=REGISTRY,
)

DRIFT_DETECTED = Counter(
    "ml_drift_detected_total",
    "Total number of detected drift events by feature",
    labelnames=["feature"],
    registry=REGISTRY,
)
