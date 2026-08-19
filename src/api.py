import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from predict import predict_timings_for_monitor

app = FastAPI(title="Opsway Forecaster API")

class TimingMetrics(BaseModel):
    response_time: float
    dns_lookup: float
    tcp_connection: float
    tls_handshake: float
    server_processing: float
    content_transfer: float
    created_at: Optional[str] = None

class PredictRequest(BaseModel):
    monitor_id: int
    timings: list[TimingMetrics]

class PredictResponse(BaseModel):
    anomalies: list[bool]
    predictions: list[float]
    upper_bounds: list[float]
    lower_bounds: list[float]

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # Predict anomalies and compute timing stats
    timings_dicts = [t.dict() for t in req.timings]
    anomalies, predictions, upper_bounds, lower_bounds = predict_timings_for_monitor(req.monitor_id, timings_dicts)
    return PredictResponse(
        anomalies=anomalies,
        predictions=predictions,
        upper_bounds=upper_bounds,
        lower_bounds=lower_bounds
    )

class ForecastRequest(BaseModel):
    monitor_id: int
    timestamps: list[str]

class ForecastResponse(BaseModel):
    predictions: list[float]
    upper_bounds: list[float]
    lower_bounds: list[float]

from predict import forecast_for_monitor

@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest):
    predictions, upper_bounds, lower_bounds = forecast_for_monitor(req.monitor_id, req.timestamps)
    return ForecastResponse(
        predictions=predictions,
        upper_bounds=upper_bounds,
        lower_bounds=lower_bounds
    )

@app.get("/health")
def health():
    return {"status": "ok"}

