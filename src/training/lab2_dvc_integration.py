import mlflow
import pandas as pd
import dvc.api
import yaml
import os

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Lab2_DVC_Integration")


def dvc_integration():
    # Use the DVC API to get the internal URL/Hash of the tracked folder
    try:
        # get_url returns the path in the .dvc/cache (which is named by the MD5 hash)
        resource_url = dvc.api.get_url(path='data/raw', repo='.')
        folder_hash = os.path.basename(resource_url)

    except Exception as e:
        print(f"Error accessing DVC API: {e}")
        return

    with mlflow.start_run(run_name="Lab2_DVC_Versioning") as run:
        # Select the latest file for the current run
        files = [f for f in os.listdir("data/raw") if f.endswith(".csv")]
        latest_file = sorted(files)[-1]
        df = pd.read_csv(os.path.join("data/raw", latest_file))

        # Log the DVC Hash for Data Lineage
        # This provides a 1:1 link between this MLflow run and the DVC data state
        mlflow.log_param("dvc_data_hash", folder_hash)
        mlflow.log_param("input_file", latest_file)
        mlflow.log_metric("row_count", len(df))
        mlflow.log_metric("average_temperature", df["temperature_2m"].mean())
        mlflow.log_metric("average_relative_humidity", df["relative_humidity_2m"].mean())
        mlflow.log_metric("average_precipitation", df["precipitation"].mean())

        print(f"Logged successfully!")
        print(f"DVC Data Hash extracted via API: {folder_hash}")


if __name__ == "__main__":
    dvc_integration()
