from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, Field
from datetime import date, timedelta
import mlflow
import torch
import numpy as np
import pandas as pd
import logging
import os

# Configure isolated logging for the API microservice
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Weather Forecasting API", description="MaaS: Model-as-a-Service for PyTorch inference.")

# 1. State Initialization: Dynamically load the Production Model and its metadata
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MODEL_NAME = "WeatherForecastModel"
ALIAS = "Production"

try:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    # 1. Retrieve the model metadata using the Alias
    model_info = client.get_model_version_by_alias(name=MODEL_NAME, alias=ALIAS)
    model_version = model_info.version
    run_id = model_info.run_id

    # 2. Construct the strict versioned URI explicitly (as per docs)
    MODEL_URI = f"models:/{MODEL_NAME}/{model_version}"

    # 3. Load the model
    model = mlflow.pytorch.load_model(MODEL_URI)

    logger.info(f"Loaded {MODEL_NAME} | Alias: {ALIAS} | Version: {model_version} | Run ID: {run_id}")
except Exception as e:
    logger.error(f"Failed to load model from MLflow Registry: {e}")
    model = None


# 2. Define the exact JSON schema required from the client
class ForecastRequest(BaseModel):
    location: str = Field(..., description="Location of the weather forecast")
    start_date: date = Field(..., description="The future start date for the forecast.")
    end_date: date = Field(..., description="The future end date for the forecast.")


@app.post("/api/v1/forecast")
def generate_forecast(request: ForecastRequest):
    if not model:
        raise HTTPException(status_code=503, detail="Inference model is currently unavailable.")

    if request.start_date > request.end_date:
        raise HTTPException(status_code=400, detail="start_date must precede end_date.")

    try:
        # 1. Read the historical data from the mounted volume
        location = request.location.capitalize()
        try:
            df = pd.read_csv(f"/code/data/raw/historical_weather-{location}.csv")
        except FileNotFoundError:
            raise HTTPException(status_code=500, detail="Location not found.")

        sequence_length = 24  # Must match the sequence length your model was trained on
        feature_cols = ['temperature_2m', 'relative_humidity_2m', 'precipitation']

        # 2. Extract the last N rows of real data
        recent_data = df.tail(sequence_length)[feature_cols].values
        historical_context_tensor = torch.tensor(np.array([recent_data]), dtype=torch.float32)

        # 3. Execute PyTorch Inference
        predictions = []
        current_date = request.start_date

        with torch.no_grad():
            delta_days = (request.end_date - request.start_date).days + 1

            for _ in range(delta_days):
                # Predict the next time step based on the real historical context
                pred = model(historical_context_tensor)
                predicted_temp = float(pred.numpy()[0][0])

                predictions.append({
                    "date": current_date.isoformat(),
                    "forecasted_temperature_2m": round(predicted_temp, 2)
                })
                current_date += timedelta(days=1)

                # --- The Autoregressive Step ---
                # We feed the predicted temperature back into the model for the next day.
                # Since the model only predicts temperature, we "forward-fill" the last known
                # humidity and precipitation to keep the feature shape consistent.
                last_humidity = float(historical_context_tensor[0, -1, 1])
                last_precip = float(historical_context_tensor[0, -1, 2])

                new_step = torch.tensor([[[predicted_temp, last_humidity, last_precip]]], dtype=torch.float32)

                # Drop the oldest day (index 0) and append the new predicted day at the end
                historical_context_tensor = torch.cat((historical_context_tensor[:, 1:, :], new_step), dim=1)

        return {
            "model_version": model_version,  # (Or whatever variable you stored the version in)
            "location": request.location,
            "forecast": predictions
        }

    except Exception as e:
        logger.error(f"Inference error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal inference execution failed.")
