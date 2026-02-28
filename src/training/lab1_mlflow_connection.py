import mlflow
from mlflow.data.filesystem_dataset_source import FileSystemDatasetSource
from mlflow.data.pandas_dataset import from_pandas
import pandas as pd
import os

# Set connection to the local server running in Docker
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Lab1_Connectivity_Test")


def run_test():
    # Load the sample weather data
    data_path = os.path.abspath("data/raw/weather.csv")
    df = pd.read_csv(data_path)

    # Create a Dataset object
    fs_dataset_source = FileSystemDatasetSource()
    dataset = from_pandas(df, source=fs_dataset_source.load(data_path), name="weather-sample", targets="temperature")

    with mlflow.start_run(run_name="Initial_CSV_Load"):
        # Log parameters (The "Intent")
        mlflow.log_param("df_shape", df.shape)
        mlflow.log_param("row_count", len(df))

        # Log metrics (The measurable results)
        mlflow.log_metric("average_temp", df["temperature"].mean())

        # Log the dataset (The file itself for total traceability)
        mlflow.log_input(dataset, context="training")  # as an input to the training process
        mlflow.log_artifact(data_path, artifact_path="raw_data")  # as a file artifact for reference


if __name__ == "__main__":
    run_test()
