import time
import logging
import os
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from typing import List, Optional, Dict, Any

from backend.schemas import (
    SensorWindowRequest,
    DescriptivePredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse
)
from backend.predictor import predictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("har_backend")

app = FastAPI(
    title="Human Activity Recognition (HAR) API",
    description="Production-grade FastAPI backend for HAR using UCI dataset and Random Forest ML model.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception on {request.url}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": f"Internal server error: {str(exc)}"}
    )

@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    model_loaded = predictor.is_loaded()
    dataset_source = predictor.metrics.get("dataset_source", "unknown")
    return {
        "status": "online" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "dataset_source": dataset_source,
        "timestamp": time.time()
    }

@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info():
    try:
        return predictor.get_model_info()
    except Exception as e:
        logger.error(f"Error fetching model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics", tags=["Model"])
async def get_metrics():
    try:
        return {"metrics": predictor.metrics}
    except Exception as e:
        logger.error(f"Error fetching metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_activity(payload: Dict[str, Any]):
    """
    Accepts:
    1. 561-dim features: `{"features": [...]}`
    2. Raw sensor window: `{"window": [...]}`
    3. Descriptive movement parameters: `{"intensity": "High", "stability": "Medium", "orientation": "Standing", "rotation": "Moderate", "pattern": "Regular"}`
    """
    try:
        if "features" in payload:
            return predictor.predict(payload["features"])
        elif "window" in payload:
            return predictor.predict_from_window(payload["window"])
        elif "intensity" in payload or "orientation" in payload:
            return predictor.predict_from_description(
                intensity=payload.get("intensity", "Medium"),
                stability=payload.get("stability", "Medium"),
                orientation=payload.get("orientation", "Standing"),
                rotation=payload.get("rotation", "Moderate"),
                pattern=payload.get("pattern", "Regular")
            )
        else:
            raise HTTPException(
                status_code=400,
                detail="Invalid payload. Provide 'features', 'window', or descriptive parameters ('intensity', 'stability', etc.)."
            )
    except ValueError as ve:
        logger.warning(f"Validation error: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/descriptive", response_model=PredictionResponse, tags=["Prediction"])
async def predict_descriptive(payload: DescriptivePredictionRequest):
    try:
        return predictor.predict_from_description(
            intensity=payload.intensity,
            stability=payload.stability,
            orientation=payload.orientation,
            rotation=payload.rotation,
            pattern=payload.pattern
        )
    except Exception as e:
        logger.error(f"Descriptive prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/dataset/sample/{index}", tags=["Dataset / Demo"])
async def get_dataset_sample(index: int):
    try:
        return predictor.get_sample(index)
    except IndexError as ie:
        raise HTTPException(status_code=404, detail=str(ie))
    except Exception as e:
        logger.error(f"Error fetching sample: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/dataset/total", tags=["Dataset / Demo"])
async def get_dataset_total_samples():
    total = len(predictor.X_test_cache) if predictor.X_test_cache is not None else 2947
    return {"total_test_samples": total}

# Mount frontend static files
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
