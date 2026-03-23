import mlflow
from mlflow.tracking import MlflowClient
import logging

# Configure isolated logging for the API microservice
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def promote_best_model(model_name="WeatherForecastModel"):
    """
    Programmatically identifies the best historical run and promotes
    its artifact to the Production stage in the Model Registry.
    """
    mlflow.set_tracking_uri("http://localhost:5000")  # Or, if inside a docker container, use the appropriate hostname for the Tracking Server
    client = MlflowClient()
    experiment = client.get_experiment_by_name("Weather_Forecasting_Models")

    # 1. Query the Tracking Server for the lowest Root Mean Squared Error
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.daily_test_rmse ASC"],
        max_results=1
    )

    if not runs:
        logger.error("No tracked runs found. Pipeline must execute training first.")
        return

    best_run = runs[0]
    best_run_id = best_run.info.run_id
    logger.info(f"Optimal run identified: {best_run_id} | RMSE: {best_run.data.metrics.get('daily_test_rmse'):.4f}")

    # 2. Register the Artifact from the Best Run
    logged_models = best_run.outputs.model_outputs
    last_model = logged_models[0] if len(logged_models) > 0 else None
    model_uri = f"models:/{last_model.model_id}"
    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=model_name,
        tags={"Run name": best_run.info.run_name, "RMSE": f"{best_run.data.metrics.get('daily_test_rmse'):.4f}"}
    )

    # 3. Apply an Alias to the Registered Model Version (e.g., "Production")
    client.set_registered_model_alias(name=model_name, alias="Production", version=registered_model.version)
    logger.info(f"Model '{model_name}' version {registered_model.version} promoted to Production stage.")


if __name__ == "__main__":
    promote_best_model()
