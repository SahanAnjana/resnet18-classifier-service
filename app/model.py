# app/model.py
import numpy as np
import torch
from torchvision import models
import mlflow
import os
from torch.nn import Module
from mlflow.tracking import MlflowClient
from typing import Optional, Tuple
from datetime import datetime

_model = None
_model_version = None

def setup_mlflow(tracking_uri: str = "http://localhost:5000"):
    mlflow.set_tracking_uri(tracking_uri)
    return mlflow.get_tracking_uri()

def load_model(
    model_name: str = "resnet18_classifier",
    stage: str = "Production",
    fallback_weights_path: str = "models/resnet18_weights.pth"
) -> Tuple[torch.nn.Module, Optional[str]]:
    global _model, _model_version
    
    if _model is None:
        try:
            client = MlflowClient()
            model_version = None
            
            for mv in client.search_model_versions(f"name='{model_name}'"):
                if mv.current_stage == stage:
                    model_version = mv.version
                    break
            
            if model_version:
                model_uri = f"models:/{model_name}/{stage}"
                _model = mlflow.pytorch.load_model(model_uri)
                _model_version = model_version
                print(f"Loaded model {model_name} version {model_version} from {stage} stage")
            else:
                raise Exception("No model found in registry")
                
        except Exception as e:
            print(f"Failed to load from MLflow: {e}. Using local weights.")
            model = models.resnet18(pretrained=False)
            state_dict = torch.load(fallback_weights_path, map_location="cpu", weights_only=False)
            model.load_state_dict(state_dict)
            model.eval()
            _model = model
            _model_version = None
    
    return _model, _model_version

def register_model(
    weights_path: str = "models/resnet18_weights.pth",
    model_name: str = "resnet18_classifier",
    promote_to_stage: Optional[str] = "Production"
) -> str:
    model = models.resnet18(pretrained=False)
    
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    
    with mlflow.start_run() as run:
        mlflow.log_param("model_type", "resnet18")
        mlflow.log_param("weights_source", weights_path)
        
        model_info = mlflow.pytorch.log_model(
            model, 
            artifact_path="model",
            registered_model_name=model_name
        )
    
    client = MlflowClient()
    model_version = None
    
    for mv in client.search_model_versions(f"run_id='{run.info.run_id}'"):
        model_version = mv.version
        
    if promote_to_stage and model_version:
        client.transition_model_version_stage(
            name=model_name,
            version=model_version,
            stage=promote_to_stage
        )
        
    return model_version

def log_prediction(run_id: str, image_name: str, prediction: str, confidence: float):
    active_run = mlflow.active_run()
    
    if active_run and active_run.info.run_id == run_id:
        mlflow.log_param("image_name", image_name)
        mlflow.log_metric("confidence", confidence)
        mlflow.set_tag("prediction", prediction)
    else:
        with mlflow.start_run(run_id=run_id):
            mlflow.log_param("image_name", image_name)
            mlflow.log_metric("confidence", confidence)
            mlflow.set_tag("prediction", prediction)

def log_drift_metrics(image_tensor, model, prediction, confidence):
    active_run = mlflow.active_run()
    def _log_metrics():
        with torch.no_grad():
            brightness = image_tensor.mean(dim=1).mean()
            contrast = image_tensor.std()
            
            embeddings = _get_embeddings(model, image_tensor)
            
            numpy_embeddings = embeddings.cpu().numpy()
            
            drift_metrics = analyze_drift(numpy_embeddings)
            
            mlflow.log_metric("avg_brightness", brightness.item())
            mlflow.log_metric("contrast", contrast.item())
            
            for key, value in drift_metrics.items():
                if isinstance(value, bool):
                    mlflow.set_tag(key, str(value))
                elif isinstance(value, (int, float)):
                    mlflow.log_metric(key, value)
                else:
                    mlflow.set_tag(key, str(value))
            
            embeddings_path = save_embeddings_to_file(numpy_embeddings)
            mlflow.log_artifact(embeddings_path, "embeddings")
            
            os.remove(embeddings_path)
    
    if active_run:
        _log_metrics()
    else:
        with mlflow.start_run():
            _log_metrics()

def _get_embeddings(model: Module, image_tensor: torch.Tensor) -> torch.Tensor:
    original_forward = model.forward
    
    features = None
    
    def hook_fn(module, input, output):
        nonlocal features
        features = output.detach()
    
    handle = model.avgpool.register_forward_hook(hook_fn)
    
    with torch.no_grad():
        _ = model(image_tensor)
    
    handle.remove()
    
    features = features.view(features.size(0), -1)
    
    return features

def save_embeddings_to_file(embeddings: np.ndarray, prefix: str = "embedding") -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    temp_dir = "temp_embeddings"
    os.makedirs(temp_dir, exist_ok=True)
    
    filename = f"{temp_dir}/{prefix}_{timestamp}.npy"
    
    np.save(filename, embeddings)
    
    return filename

def analyze_drift(current_embeddings: np.ndarray, baseline_path: str = "baseline_embeddings.npy") -> dict:
    if os.path.exists(baseline_path):
        baseline_embeddings = np.load(baseline_path)
        
        current_mean = current_embeddings.mean(axis=0)
        baseline_mean = baseline_embeddings.mean(axis=0)
        
        similarity = np.dot(current_mean, baseline_mean) / (
            np.linalg.norm(current_mean) * np.linalg.norm(baseline_mean)
        )
        
        distance = np.linalg.norm(current_mean - baseline_mean)
        
        return {
            "cosine_similarity": float(similarity),
            "euclidean_distance": float(distance),
            "drift_detected": float(similarity) < 0.8  # Threshold for drift
        }
    else:
        np.save(baseline_path, current_embeddings)
        return {
            "status": "new_baseline_created",
            "drift_detected": False
        }

def log_drift_metrics(image_tensor, model, prediction, confidence):
    with torch.no_grad():
        brightness = image_tensor.mean(dim=1).mean()
        contrast = image_tensor.std()
        
        embeddings = _get_embeddings(model, image_tensor)
        
        numpy_embeddings = embeddings.cpu().numpy()
        
        drift_metrics = analyze_drift(numpy_embeddings)
        
        mlflow.log_metric("avg_brightness", brightness.item())
        mlflow.log_metric("contrast", contrast.item())
        
        for key, value in drift_metrics.items():
            if isinstance(value, (int, float)): 
                mlflow.log_metric(key, value)
            else:
                mlflow.set_tag(key, str(value))
        
        embeddings_path = save_embeddings_to_file(numpy_embeddings)
        mlflow.log_artifact(embeddings_path, "embeddings")
        
        os.remove(embeddings_path)