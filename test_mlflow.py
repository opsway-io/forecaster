import mlflow
import os

os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"
mlflow.set_tracking_uri("http://localhost:5000")

try:
    model = mlflow.sklearn.load_model("models:/forecaster_monitor_1/latest")
    print("Success")
except Exception as e:
    print(f"Error: {e}")
