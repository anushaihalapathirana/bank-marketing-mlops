import os
from pathlib import Path
import pandas as pd
import mlflow
import mlflow.sklearn
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from src.schemas import CustomerFearures, PredictionResponse

ROOT = Path(__file__).resolve().parents[1]  

@asynccontextmanager
async def lifespan(app: FastAPI):
    
    # load the model only once when the api starts
    
    MODEL_URI = os.environ["MODEL_URI"]
    
    # Only configure local MLflow tracking when
    # loading a model by its MLflow URI
    if MODEL_URI.startswith("models:/") or MODEL_URI.startswith("runs:/"):
        mlflow.set_tracking_uri(f"sqlite:///{ROOT/'mlflow.db'}")
        
    app.state.model = mlflow.sklearn.load_model(MODEL_URI)
    yield
    
    # release the reference during shutdown
    del app.state.model
    
app = FastAPI(
    title="Bank Marketing Prediction API",
    description="An API for predicting whether a customer will subscribe to a term deposit based on their features.",
    version="1.0.0",
    lifespan=lifespan
    )

@app.get("/health")
def health_check():
    return {"status": "ok"} 

@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFearures, request: Request):
    #convert the Pydantic model to a DataFrame. convert field names to match the model's expected input using by_alias=True
    input_df = pd.DataFrame([customer.model_dump(by_alias=True)])
    
    model = request.app.state.model
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0, 1]
    
    return PredictionResponse(
        predicted_class=int(prediction),
        probability=float(probability)
    )  
