# scripts/setup_mlflow.py
import os
import mlflow
from mlflow.tracking import MlflowClient
from app.model import register_model

def setup_mlflow_and_register_model():
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    
    print(f"MLflow tracking URI: {tracking_uri}")
    
    try:
        client = MlflowClient()
        experiments = client.search_experiments()
        print(f"Connected to MLflow server. Found {len(experiments)} experiments.")
    except Exception as e:
        print(f"Error connecting to MLflow server: {e}")
        print("Make sure your MLflow server is running.")
        print("Start the server using docker-compose or directly with MySQL: ")
        print("mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri mysql+pymysql://root:@localhost:3306/mlflow --default-artifact-root ./artifacts")
        return
    
    weights_path = "models/resnet18_weights.pth"
    if not os.path.exists(weights_path):
        print(f"Error: Model weights not found at {weights_path}")
        return
    
    try:
        model_version = register_model(
            weights_path=weights_path,
            model_name="resnet18_classifier",
            promote_to_stage="Production"
        )
        print(f"Successfully registered model version: {model_version}")
        print(f"Model is now available in Production stage")
    except Exception as e:
        print(f"Error registering model: {e}")

if __name__ == "__main__":
    setup_mlflow_and_register_model()