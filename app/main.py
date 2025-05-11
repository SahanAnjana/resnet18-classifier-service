# app/main.py
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from app.model import load_model, setup_mlflow, log_prediction, log_drift_metrics
from torchvision import transforms
from PIL import Image
import torch
import mlflow
from io import BytesIO
import uuid
from typing import Optional

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

with open("models/imagenet_classes.txt") as f:
    CLASS_NAMES = [line.strip() for line in f.readlines()]

_preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

active_run_id = None

@app.on_event("startup")
def _startup():
    tracking_uri = setup_mlflow(tracking_uri="http://localhost:5000")
    print(f"MLflow tracking URI: {tracking_uri}")
    
    global active_run_id
    with mlflow.start_run(run_name="api_service") as run:
        active_run_id = run.info.run_id
        mlflow.set_tag("service_type", "prediction_api")
    
    model, version = load_model()
    if version:
        print(f"Loaded model version: {version}")
    else:
        print("Using local model weights (not from MLflow)")

def log_prediction_task(run_id: str, filename: str, class_name: str, probability: float):
    log_prediction(run_id, filename, class_name, probability)

@app.post("/predict-image")
async def predict_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    try:
        data = await file.read()

        image = Image.open(BytesIO(data)).convert("RGB")
        
        tensor = _preprocess(image).unsqueeze(0)  # [1,C,H,W]
        
        model, _ = load_model()
        with torch.no_grad():
            logits = model(tensor)
        probs = torch.nn.functional.softmax(logits, dim=1)[0]
        
        top1 = torch.topk(probs, k=1)
        idx = top1.indices[0].item()
        prob = round(top1.values[0].item(), 4)
        class_name = CLASS_NAMES[idx]
        
        if active_run_id:
            background_tasks.add_task(
                log_prediction_task,
                active_run_id,
                file.filename or f"image_{uuid.uuid4()}",
                class_name,
                prob
            )
            
            background_tasks.add_task(
                log_drift_metrics,
                tensor,
                model,  
                class_name,
                prob
            )

        return {
            "class_name": class_name,
            "probability": prob,
            "model_version": _ if _ else "local",
            "run_id": active_run_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "mlflow": mlflow.get_tracking_uri()}