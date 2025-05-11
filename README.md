# resnet18-classifier-service

## Description

`resnet18-classifier-service` is a production-ready microservice for image classification, featuring:

* **Pretrained ResNet-18** model from PyTorch’s torchvision library for 1,000-class ImageNet inference.
* **FastAPI** REST endpoints with automatic Swagger (`/docs`) and ReDoc (`/redoc`) documentation.
* **MLflow** integration for experiment tracking, model versioning, and model registry backed by a MySQL metadata store.
* **Uvicorn** ASGI server for asynchronous, low-latency predictions (<100 ms).

## Features

* **Health Check**: `GET /health` endpoint to verify service liveness.
* **Image Prediction**: `POST /predict` accepts image files and returns top-K predictions with probabilities.
* **Interactive API Docs**: Auto-generated Swagger UI and ReDoc for easy testing and integration.
* **Experiment Tracking**: Logs parameters, metrics, and artifacts to MLflow UI.
* **Model Registry**: Register, stage, and manage model versions via MLflow.

## Tech Stack

| Component     | Technology            |
| ------------- | --------------------- |
| Model         | PyTorch ResNet-18     |
| API Framework | FastAPI               |
| MLOps         | MLflow                |
| Database      | MySQL                 |
| Server        | Uvicorn (ASGI)        |
| Packaging     | pip, requirements.txt |

## Prerequisites

* Python 3.8+
* MySQL server
* Git

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/SahanAnjana/resnet18-classifier-service.git
   cd resnet18-classifier-service
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure MySQL**

   * Create a database for MLflow:

     ```sql
     CREATE DATABASE IF NOT EXISTS mlflow;
     ```
   * Update `MLFLOW_TRACKING_URI` and database URI in `config.py` or environment variables.

## Usage

### 1. Start MLflow Server

```bash
mlflow server \
  --backend-store-uri mysql+pymysql://root:@localhost/mlflow \
  --default-artifact-root ./mlruns \
  --host 0.0.0.0 --port 5000
```

### 2. Register Initial Model

```bash
python -m scripts.setup_mlflow
```

### 3. Run the API Service

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Make Predictions

Use `curl` or any HTTP client to send images:

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@/path/to/image.jpg" 
```

## API Reference

### GET /health

* **Description**: Returns service status.
* **Response**: `200 OK` with JSON `{ "status": "ok" }`.

### POST /predict

* **Description**: Classify an image.
* **Parameters**:

  * `file` (form-data): Image file to classify.
  * `top_k` (form-data, optional): Number of top predictions to return (default: 5).
* **Response**: JSON array of objects `{ "label": "class name", "probability": float }`.
