from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

class SensorReading(BaseModel):
    x: float
    y: float
    z: float

class SensorSample(BaseModel):
    timestamp: Optional[int] = None
    accelerometer: SensorReading
    gyroscope: SensorReading

class SensorWindowRequest(BaseModel):
    window: List[SensorSample] = Field(..., description="List of raw sensor readings forming a rolling window")

class DescriptivePredictionRequest(BaseModel):
    intensity: str = Field(..., description="Low, Medium, or High")
    stability: str = Field(..., description="Low, Medium, or High")
    orientation: str = Field(..., description="Lying, Sitting, or Standing")
    rotation: str = Field(..., description="Low, Moderate, or High")
    pattern: str = Field(..., description="Still, Regular, or Rhythmic")

class PredictionResponse(BaseModel):
    activity: str
    confidence: float
    probabilities: Dict[str, float]
    model_type: str = "RandomForest"
    description_summary: Optional[Dict[str, str]] = None
    top_predictions: Optional[List[Dict[str, Any]]] = None

class BatchPredictionRequest(BaseModel):
    feature_vectors: List[List[float]] = Field(..., description="List of 561-dimensional feature vectors")

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    dataset_source: str
    timestamp: float

class ModelInfoResponse(BaseModel):
    model_name: str
    dataset: str
    activities_count: int
    features_count: int
    training_samples: int
    testing_samples: int
    accuracy: float
    dataset_source: str

class MetricsResponse(BaseModel):
    metrics: Dict[str, Any]
