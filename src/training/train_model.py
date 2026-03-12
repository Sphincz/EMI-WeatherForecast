import hydra
from omegaconf import DictConfig, OmegaConf
import mlflow
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from datetime import datetime

from models.lstm import WeatherLSTM
from models.gru import WeatherGRU
from models.fbprophet import WeatherProphet

import logging
import warnings
import yaml

from mlflow.prophet import log_model as log_prophet_model
from mlflow.pytorch import log_model as log_pytorch_model

logger = logging.getLogger(__name__)


def create_sliding_windows(data, seq_length):
    """Transforms the multivariate time series data into sliding windows for sequence modeling."""
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return torch.tensor(np.array(xs), dtype=torch.float32), torch.tensor(np.array(ys), dtype=torch.float32)


def get_dvc_hash(dvc_file_path="data/raw.dvc"):
    """Extracts the exact md5 data hash from the DVC tracking file."""
    try:
        with open(dvc_file_path, 'r') as f:
            dvc_data = yaml.safe_load(f)
            # Access the md5 hash of the tracked directory/file
            return dvc_data['outs'][0]['md5']
    except Exception as e:
        logger.warning(f"Could not read DVC hash from {dvc_file_path}: {e}")
        return "unknown_dvc_hash"


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def train(cfg: DictConfig):
    logger.info(f"--- Starting Training Pipeline for {cfg.model.name} ---")

    mlflow.set_tracking_uri("http://mlflow_server:5000")
    mlflow.set_experiment("Weather_Forecasting_Models")

    # Use Lisbon for now, for this Lab
    data_path = "data/raw/historical_weather-Lisbon.csv"
    df = pd.read_csv(data_path)

    # Reserve the last 720 rows (30 days) for the Evaluation script
    train_df = df.iloc[:-720].reset_index(drop=True)

    # 2. Generate the dynamic run name with a timestamp
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    dynamic_run_name = f"{cfg.model.name}_Training_Run_{current_time}"

    # 3. Capture the active run context as 'run'
    with mlflow.start_run(run_name=dynamic_run_name) as run:
        # Log Hydra parameters in a clean, flattened format for MLflow
        mlflow.log_params(OmegaConf.to_container(cfg.model, resolve=True))
        # Log the DVC Data Hash for strict reproducibility
        data_hash = get_dvc_hash("data/raw.dvc")
        mlflow.log_param("data_dvc_hash", data_hash)
        # Document the split
        mlflow.log_param("test_set_reserved_hours", 720)

        if cfg.model.name in ["LSTM", "GRU"]:
            # Multivariate Forecasting
            features = df[['temperature_2m', 'relative_humidity_2m', 'precipitation']].values
            x, y = create_sliding_windows(features, cfg.model.sequence_length)
            dataset = TensorDataset(x, y)

            dataloader = DataLoader(
                dataset,
                batch_size=cfg.model.batch_size,
                shuffle=False,
                num_workers=4
            )

            model = WeatherLSTM(cfg.model) if cfg.model.name == "LSTM" else WeatherGRU(cfg.model)

            # Pass the active run_id to the Lightning Logger
            mlf_logger = MLFlowLogger(
                experiment_name="Weather_Forecasting_Models",
                tracking_uri="http://mlflow_server:5000",
                run_id=run.info.run_id  # <--- This bridges them perfectly together!
            )

            trainer = pl.Trainer(
                max_epochs=cfg.model.epochs,
                logger=mlf_logger,
                enable_checkpointing=False,
                log_every_n_steps=5
            )
            trainer.fit(model, dataloader)

            # Grab one batch of data to serve as the input example (required due to export_model=True)
            example_input, _ = next(iter(dataloader))
            input_example = example_input.numpy()

            # Save the PyTorch model artifact to MLflow
            log_pytorch_model(model, name="model_artifact", export_model=True, code_paths=["src/training/models"], input_example=input_example)

        elif cfg.model.name == "Prophet":
            # Since Prophet is a univariate forecasting model, we'll focus only on temperature
            prophet_model = WeatherProphet(cfg.model)
            fitted_model = prophet_model.fit(df, target_column='temperature_2m')

            # 1. Generate predictions on the training data to calculate error
            prophet_df = pd.DataFrame({'ds': pd.to_datetime(df['date'])})
            forecast = fitted_model.predict(prophet_df)

            # 2. Calculate Mean Squared Error (MSE) using NumPy
            y_true = df['temperature_2m'].values
            y_pred = forecast['yhat'].values
            train_loss = np.mean((y_true - y_pred) ** 2)

            # 3. Explicitly log the metric to MLflow!
            mlflow.log_metric("train_loss", float(train_loss))

            # Save the Prophet model artifact to MLflow
            log_prophet_model(fitted_model, name="model_artifact")
            logger.info(f"Prophet fitting complete. Train Loss (MSE): {train_loss:.4f}")

    logger.info("--- Training Pipeline Complete ---")


if __name__ == "__main__":
    train()
