import pandas as pd
import numpy as np
from evidently import Dataset, Report, DataDefinition, Regression
from evidently.presets import DataDriftPreset, RegressionPreset
from evidently.ui.workspace import Workspace
import logging

# Configure isolated logging for the API microservice
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_drift_analysis():
    logger.info("--- Initiating Covariate Shift & Performance Decay Analysis ---")

    # 1. Load the centralized dataset (Lisbon, for now, to exemplify)
    df = pd.read_csv("data/raw/historical_weather-Lisbon.csv")

    reference_data = df.iloc[:-720].copy()
    current_data = df.iloc[-720:].copy()

    # 2. SIMULATE HISTORICAL PREDICTIONS (Good Performance)
    # The model performed well on the reference data (predictions are very close to the target)
    reference_data['prediction'] = reference_data['temperature_2m'] + np.random.normal(0, 1.5, size=len(reference_data))

    # 3. PROVOKE DRIFT (The Anomaly)
    # We synthetically offset the actual ground truth temperature readings by +15.0°C
    current_data['temperature_2m'] = current_data['temperature_2m'] + 15.0
    logger.warning("ARTIFICIAL DRIFT INJECTED: Ground truth temperature offset by +15.0°C")

    # 4. SIMULATE RECENT PREDICTIONS (Catastrophic Failure)
    # The model, unaware of the broken sensors/heatwave, predicted normal temperatures.
    # Therefore, its predictions are roughly 15 degrees lower than the new ground truth!
    current_data['prediction'] = current_data['temperature_2m'] - 15.0 + np.random.normal(0, 1.5, size=len(current_data))
    logger.warning("ARTIFICIAL MODEL DECAY SIMULATED: Predictions are ~15°C lower than the new ground truth")

    # 5. Define Data Definition
    data_def = DataDefinition(
        numerical_columns=["temperature_2m", "relative_humidity_2m", "precipitation"],
        regression=[Regression(target="temperature_2m", prediction="prediction")]
    )

    # Wrap BOTH raw pandas DataFrames into Evidently Dataset objects
    ref_dataset = Dataset.from_pandas(reference_data, data_definition=data_def)
    curr_dataset = Dataset.from_pandas(current_data, data_definition=data_def)

    # 6. Initialize Workspace
    ws = Workspace.create("evidently_ui/workspace")
    project = ws.create_project("Weather Forecast Operational Monitoring")
    project.description = "Continuous Monitoring for Data Drift and Model Decay."
    project.save()

    # 7. Generate the Combined Report
    drift_report = Report(metrics=[DataDriftPreset(), RegressionPreset()])

    # Pass the Evidenlty Dataset objects
    drift_run = drift_report.run(
        reference_data=ref_dataset,
        current_data=curr_dataset,
        name="Weather Forecast Operational Monitoring"
    )

    # 8. Send to UI
    ws.add_run(project.id, drift_run)
    logger.info("Combined Drift & Performance Report generated! Check http://localhost:8081")


if __name__ == "__main__":
    run_drift_analysis()
