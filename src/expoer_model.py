import os
import mlflow
from mlflow.artifacts import download_artifacts
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]  

mlflow.set_tracking_uri(f"sqlite:///{ROOT/'mlflow.db'}")

model_uri = os.environ["MODEL_URI"]

downloaded_path = download_artifacts(artifact_uri=model_uri)
# Move the downloaded model to a specific directory
destination_path = ROOT / "deployment" / "model"
if destination_path.exists():
    shutil.rmtree(destination_path)

destination_path.parent.mkdir(parents=True, exist_ok=True)
shutil.copytree(downloaded_path, destination_path)

print(f"Model downloaded and moved to: {destination_path}")
print("MLmodel exists:", (destination_path / "MLmodel").exists())