import mlflow
import pandas as pd
import hydra
from omegaconf import DictConfig
from dvc.api import DVCFileSystem
import dvc.api
import os


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def run_dynamic_experiment(cfg: DictConfig):
    print(f"--- Starting Experiment for {cfg.project_name} ---")

    # 1. Setup MLflow dynamically using Hydra configurations
    mlflow.set_tracking_uri(cfg.tracking.uri)
    mlflow.set_experiment(cfg.tracking.experiment_name)

    # 2. Strict DVC Lineage Tracking (From Lab 2)
    try:
        resource_url = dvc.api.get_url(path=cfg.paths.raw_data_dir, repo='.')
        folder_hash = os.path.basename(resource_url)

        # Connect to the Virtual Filesystem at HEAD
        fs = DVCFileSystem(url=".", rev="HEAD")
        tracked_files = fs.ls(cfg.paths.raw_data_dir)

        csv_files = []
        for f in tracked_files:
            file_path = f["name"] if isinstance(f, dict) else f
            if file_path.endswith(".csv"):
                csv_files.append(file_path)

        if not csv_files:
            print("No CSV files are tracked by DVC yet. Please run 'dvc add' first.")
            return

        latest_tracked_file = sorted(csv_files)[-1]

        with fs.open(latest_tracked_file) as f:
            df = pd.read_csv(f)

    except Exception as e:
        print(f"Error accessing DVC FileSystem: {e}")
        return

    # 3. Log to MLflow
    with mlflow.start_run(run_name=cfg.tracking.default_run_name):
        # Log the DVC Hash
        mlflow.log_param("dvc_data_hash", folder_hash)

        # Log Hydra configuration parameters to MLflow for absolute reproducibility
        mlflow.log_param("api_latitude", cfg.api.params.latitude)
        mlflow.log_param("api_longitude", cfg.api.params.longitude)
        mlflow.log_param("input_file", os.path.basename(latest_tracked_file))
        mlflow.log_metric("dataset_rows", len(df))

        print(f"Successfully logged to MLflow at {cfg.tracking.uri}")
        print(f"Experiment: {cfg.tracking.experiment_name}")
        print(f"DVC Data Hash: {folder_hash}")


if __name__ == "__main__":
    run_dynamic_experiment()