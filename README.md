# Energy Demand Forecasting MLOps Project
End-to-end machine learning pipeline for energy demand forecasting, including training, tracking, and deployment.

## Overview
This project predicts energy demand based on time-based and engineered features (hour, weekday, lag features, etc.).

The system is fully productionized:

- Data preprocessing & feature engineering (Python, Pandas, Numpy)
- Machine learning model training (LightGBM)
- REST API for inference (FastAPI)
- Containerization (Docker)
- Cloud deployment (Google Cloud Run)
- CI/CD automation (GitHub Actions)

## Project Architecture
User → FastAPI → LightGBM Model
↓
Docker Container
↓
Google Cloud Run
↓
GitHub Actions (CI/CD)

---

## Dataset
The dataset contains time-series energy demand data.

### Features used:
- hour
- weekday
- temperature
- lag features (lag\_1, lag\_24, etc.)
- rolling statistics

### Target:
- `demand`

## Model
- Model: **LightGBM Regressor**
- Framework: `lightgbm`
- Loss: Regression (MAE / RMSE)

### Performance:
- MAE: ~346
- RMSE: ~456

---

## Live API
👉 https://energy-api-281778059410.europe-west3.run.app/docs

- Prediction Endpoint: POST /predict
- Example request:
```json
{
  "hour": 14,
  "dayofweek": 2,
  "lag_24": 7000,
  "lag_168": 6800,
  "rolling_mean_24": 6900,
  "temperature": 12.5,
  "precipitation": 0.0,
  "wind_speed": 5.2
}
```
- Response:
```json
{
  "prediction": 7234.12
}
```

---
## Results


## Key Learnings
- End-to-end ML pipeline design
- Dockerization of ML services
- Cloud deployment (Cloud Run)
- CI/CD automation
- Time-series feature engineering


## Future Improvements
