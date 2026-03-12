# src/evaluation/evaluate_model.py
import hydra
from PIL import Image
from omegaconf import DictConfig
import mlflow
import pandas as pd
import numpy as np
import torch
import plotly.graph_objects as go
import logging
import os
import time
import io
from datetime import datetime

from mlflow.pytorch import load_model as load_pytorch_model
from mlflow.prophet import load_model as load_prophet_model

logger = logging.getLogger(__name__)


def calculate_metrics(y_true, y_pred):
    y_true, y_pred = np.array(y_true).flatten(), np.array(y_pred).flatten()
    return np.sqrt(np.mean((y_true - y_pred) ** 2)), np.mean(np.abs(y_true - y_pred))


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def evaluate(cfg: DictConfig):
    logger.info(f"--- Starting Daily Evaluation Pipeline for {cfg.model.name} ---")

    mlflow.set_tracking_uri("http://mlflow_server:5000")
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("Weather_Forecasting_Models")

    # 1. Load the Shifting 30-day Test Set
    # Because ingestion runs daily, the tail of this CSV contains new unseen data every day!
    df = pd.read_csv("data/raw/historical_weather-Lisbon.csv")
    test_df = df.iloc[-720:].reset_index(drop=True)
    dates = pd.to_datetime(test_df['date'])
    y_true = test_df['temperature_2m'].values

    # 2. Find the latest trained run to evaluate
    query = f"tags.mlflow.runName LIKE '{cfg.model.name}_Training_Run_%'"
    runs = client.search_runs(experiment_ids=[experiment.experiment_id], filter_string=query, order_by=["start_time DESC"], max_results=1)

    if not runs:
        logger.error(f"No trained model found for {cfg.model.name}. Skipping evaluation.")
        return

    latest_run_id = runs[0].info.run_id

    # 3. Load the Model Artifact directly from MLflow Registry
    artifact_uri = f"runs:/{latest_run_id}/model_artifact"
    logger.info(f"Evaluating Model Artifact: {artifact_uri}")

    # Re-open the EXISTING training run to append today's metrics
    with mlflow.start_run(run_id=latest_run_id):
        if cfg.model.name in ["LSTM", "GRU"]:
            loaded_model = load_pytorch_model(artifact_uri)

            features = test_df[['temperature_2m', 'relative_humidity_2m', 'precipitation']].values
            seq_length = cfg.model.sequence_length

            y_pred = []
            with torch.no_grad():
                for i in range(len(features) - seq_length):
                    x_input = torch.tensor(np.array([features[i:(i + seq_length)]]), dtype=torch.float32)
                    pred = loaded_model(x_input)
                    y_pred.append(pred.numpy()[0][0])

            y_true_aligned = y_true[seq_length:]
            dates_aligned = dates[seq_length:]

        elif cfg.model.name == "Prophet":
            loaded_model = load_prophet_model(artifact_uri)
            prophet_test_df = pd.DataFrame({'ds': dates})
            forecast = loaded_model.predict(prophet_test_df)
            y_pred = forecast['yhat'].values

            y_true_aligned = y_true
            dates_aligned = dates

        # 4. Calculate and Log Metrics
        # By logging to the same run_id every day, MLflow builds a historical drift chart!
        rmse, mae = calculate_metrics(y_true_aligned, y_pred)

        # Use a timestamp as the step so MLflow tracks the progression over time
        current_time = int(datetime.now().strftime("%Y%m%d"))
        mlflow.log_metric("daily_test_rmse", float(rmse), step=current_time)
        mlflow.log_metric("daily_test_mae", float(mae), step=current_time)

        # 5. Generate and Log Forecast Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates_aligned, y=y_true_aligned, mode='lines', name='Actual Temp', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=dates_aligned, y=y_pred, mode='lines', name=f'{cfg.model.name} Forecast', line=dict(color='red', dash='dash')))
        fig.update_layout(title=f"{cfg.model.name} - 30 Day Forecast vs Actuals", xaxis_title="Date", yaxis_title="Temperature (°C)")

        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f"forecast_plot_{cfg.model.name}_{current_time}.html"  # HTML natively supported by MLflow UI
        mlflow.log_figure(fig, plot_filename)

        logger.info(f"Daily Evaluation Complete! Test RMSE: {rmse:.4f} | Test MAE: {mae:.4f}")


if __name__ == "__main__":
    evaluate()
