# start_mlflow_local.py
import os
import sys
import subprocess

def start_mlflow_server():
    db_username = "root"
    db_password = ""  # Empty password
    db_host = "localhost"
    db_port = "3306"
    db_name = "mlflow"
    
    mlflow_port = "5000"
    artifact_path = "./artifacts"
    
    os.makedirs(artifact_path, exist_ok=True)
    
    db_uri = f"mysql+pymysql://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}"
    
    cmd = [
        "mlflow", "server",
        "--host", "0.0.0.0",
        "--port", mlflow_port,
        "--backend-store-uri", db_uri,
        "--default-artifact-root", artifact_path
    ]
    
    print(f"Starting MLflow server with MySQL backend:")
    print(f"Database URI: {db_uri}")
    print(f"Artifact root: {artifact_path}")
    print(f"Server will be available at http://localhost:{mlflow_port}")
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nMLflow server stopped")
    except Exception as e:
        print(f"Error starting MLflow server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    start_mlflow_server()