from fastapi import FastAPI
import joblib
import pandas as pd
from src.config import FEATURES


app = FastAPI()
model = None

def get_model():
    global model
    if model is None:
        model = joblib.load("models/lgbm_model.pkl")
    return model

# health check
@app.get("/")
def home():
    return {"message": "Energy Demand Forecast API is running."}

@app.post("/predict")
def predict(data: dict):
    model = get_model()
    df = pd.DataFrame([data])
    df = df[FEATURES]
    pred = model.predict(df)[0]

    return {"prediction": float(pred)}