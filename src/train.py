import os
import mlflow
import mlflow.sklearn
from sklearn.ensemble import IsolationForest
import pandas as pd
from datetime import datetime

from data import get_clickhouse_client, get_monitor_data

def train_forecaster_for_monitor(monitor_id: int):
    print(f"Starting training for monitor {monitor_id}")
    client = get_clickhouse_client()
    df = get_monitor_data(client, monitor_id)
    
    if df.empty or len(df) < 50:
        print(f"Not enough data to train monitor {monitor_id} (found {len(df)} records)")
        return
    
    # Feature engineering: extract hour of day and day of week
    df['hour'] = df['created_at'].dt.hour
    df['day_of_week'] = df['created_at'].dt.dayofweek
    
    features = [
        'response_time', 
        'dns_lookup', 
        'tcp_connection', 
        'tls_handshake', 
        'server_processing', 
        'content_transfer', 
        'hour', 
        'day_of_week'
    ]
    
    X = df[features].fillna(0)
    
    # Isolation forest
    model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
    model.fit(X)
    
    # Log to MLflow
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment("Monitor_Anomaly_Detection")
    
    with mlflow.start_run():
        mlflow.log_param("monitor_id", monitor_id)
        mlflow.log_param("model_type", "IsolationForest")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("contamination", 0.01)
        mlflow.log_metric("training_samples", len(df))
        
        # Log model
        model_name = f"forecaster_monitor_{monitor_id}"
        mlflow.sklearn.log_model(model, "model", registered_model_name=model_name)
        
        print(f"Successfully trained and logged model for monitor {monitor_id}")

if __name__ == "__main__":
    # Test script locally
    train_forecaster_for_monitor(1)
