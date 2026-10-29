from fastapi import FastAPI
from src.serving.inference import predict, InferenceRequest

app = FastAPI(title="Ames Housing Price Prediction API")


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/predict")
def predict_endpoint(request: InferenceRequest):
    predictions = predict(request)
    return {"predictions": predictions}