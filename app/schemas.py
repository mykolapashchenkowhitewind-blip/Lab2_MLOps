from typing import Dict, List

from pydantic import BaseModel, Field


class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., ge=0, le=10, description="cm")
    sepal_width: float = Field(..., ge=0, le=10, description="cm")
    petal_length: float = Field(..., ge=0, le=10, description="cm")
    petal_width: float = Field(..., ge=0, le=10, description="cm")


class PredictionResponse(BaseModel):
    class_id: int
    class_name: str
    probability: float


class DriftRequest(BaseModel):
    samples: List[List[float]] = Field(
        ...,
        min_length=10,
        description="Batch of Iris observations. Each observation must contain 4 features.",
    )
    alpha: float = Field(
        default=0.05,
        ge=0.001,
        le=0.5,
        description="Significance threshold for the KS test.",
    )

    def model_post_init(self, __context) -> None:
        for sample in self.samples:
            if len(sample) != 4:
                raise ValueError("Each sample must contain exactly 4 features")


class FeatureDriftInfo(BaseModel):
    statistic: float
    p_value: float
    drift_detected: bool


class DriftResponse(BaseModel):
    drift_detected: bool
    n_drifted_features: int
    drifted_features: List[str]
    per_feature: Dict[str, FeatureDriftInfo]
    n_samples: int
    alpha: float
