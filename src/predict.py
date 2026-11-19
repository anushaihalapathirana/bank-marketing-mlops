import os
import mlflow
from pathlib import Path
import pandas as pd
import mlflow.sklearn

ROOT = Path(__file__).resolve().parents[1]

mlflow.set_tracking_uri(f"sqlite:///{ROOT/'mlflow.db'}")

# get the model URI from the environment variable
MODEL_URI = os.environ["MODEL_URI"]

# load the saved pipeline from MLflow
model = mlflow.sklearn.load_model(MODEL_URI)

# example input data for prediction
customer = {
    "age": 36,
    "job": "housemaid",
    "marital": "married",
    "education": "basic.4y",
    "default": "yes",
    "housing": "yes",
    "loan": "yes",
    "contact": "telephone",
    "month": "may",
    "day_of_week": "mon",
    "campaign": 1,
    "pdays": 999,
    "previous": 1,
    "poutcome": "nonexistent",
    "emp.var.rate": 1.1,
    "cons.price.idx": 93.994,
    "cons.conf.idx": -36.4,
    "euribor3m": 4.857,
    "nr.employed": 5191.0,
}

input_df = pd.DataFrame([customer])

# generate prediction
prediction = model.predict(input_df)
print(f"Predicted class: {prediction[0]}")
probability = model.predict_proba(input_df)[:, 1]
print(f"Predicted probability of having a subscription: {round(probability[0], 2)}")
    