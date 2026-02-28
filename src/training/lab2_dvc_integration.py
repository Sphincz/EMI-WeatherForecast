import mlflow
import pandas as pd
import dvc.api
from dvc.api import DVCFileSystem
import os

from mlflow.data.filesystem_dataset_source import FileSystemDatasetSource
from mlflow.data.pandas_dataset import from_pandas

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Lab2_DVC_Integration")


def dvc_integration():
    try:
        # 1. Get the hash of the tracked folder (Current DVC state)
        resource_url = dvc.api.get_url(path='data/raw', repo='.')
        folder_hash = os.path.basename(resource_url)

        # 2. Connect to the DVC Virtual Filesystem at the last commit (HEAD)
        # This completely ignores any uncommitted changes on your local hard drive
        fs = DVCFileSystem(url=".", rev="HEAD")

        # 3. List ONLY the files that DVC is tracking at this exact commit
        tracked_files = fs.ls("data/raw")

        csv_files = []
        for f in tracked_files:
            # If DVC returns a dictionary, grab the 'name' key. Otherwise, use the string.
            file_path = f["name"] if isinstance(f, dict) else f

            if file_path.endswith(".csv"):
                csv_files.append(file_path)

        if not csv_files:
            print("No CSV files are currently tracked by DVC in data/raw.")
            return

        # Grab the latest file FROM THE TRACKED LIST
        latest_tracked_file = sorted(csv_files)[-1]

        if not csv_files:
            print("No CSV files are currently tracked by DVC in data/raw.")
            return

        # Grab the latest file FROM THE TRACKED LIST
        latest_tracked_file = sorted(csv_files)[-1]

        # 4. Read the file directly from the DVC cache using fs.open()
        with fs.open(latest_tracked_file) as f:
            df = pd.read_csv(f)

        # 5. Create a Dataset object and log it to MLflow (for traceability)
        fs_dataset_source = FileSystemDatasetSource()
        dataset = from_pandas(df, source=fs_dataset_source.load(latest_tracked_file), name="weather-sample", targets="temperature_2m")

    except Exception as e:
        print(f"Error accessing DVC API: {e}")
        return

    with mlflow.start_run(run_name="Lab2_DVC_Lineage") as run:
        # Log the DVC Hash for Data Lineage
        # This provides a 1:1 link between this MLflow run and the DVC data state
        mlflow.log_param("dvc_data_hash", folder_hash)
        mlflow.log_param("input_file", latest_tracked_file)
        mlflow.log_input(dataset, context="training")
        mlflow.log_artifact(latest_tracked_file, artifact_path="last_tracked_file")
        mlflow.log_metric("row_count", len(df))
        mlflow.log_metric("average_temperature", df["temperature_2m"].mean())
        mlflow.log_metric("average_relative_humidity", df["relative_humidity_2m"].mean())
        mlflow.log_metric("average_precipitation", df["precipitation"].mean())

        print(f"Logged successfully!")
        print(f"DVC Data Hash extracted via API: {folder_hash}")


if __name__ == "__main__":
    dvc_integration()
