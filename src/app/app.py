from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
from src.serving.inference import predict, InferenceRequest, HouseData


app= FastAPI(title="Ames Housing Price Prediction API")


@app.get("/")
def root():
    """
    Health check endpoint for monitoring and load balancer health checks.
    """
    return {"status": "ok"}

@app.post("/predict")
def predict_endpoint(request: InferenceRequest):
    """
    Endpoint to predict housing prices based on input features.

    Args:
        request (InferenceRequest): The request body containing the input features.

    Returns:
        dict: A dictionary containing the predicted housing prices.
    """
    predictions = predict(
        request=request,
        path_ordinal="/Users/sofianeagounikaci/RegressionModel-Mlops/mlartifacts/mlartifactsordinal_encoder.pkl",
        path_onehot="/Users/sofianeagounikaci/RegressionModel-Mlops/mlartifacts/mlartifactsonehot_encoder.pkl"
    )
    return {"predictions": predictions}