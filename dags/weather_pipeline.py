from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from datetime import datetime, timedelta

# Default arguments applied to all tasks in the DAG
default_args = {
    'owner': 'mlops_engineer',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG context
with DAG(
        dag_id='weather_forecast_mlops_pipeline',
        default_args=default_args,
        description='End-to-End MLOps Pipeline for Weather Forecasting',
        schedule='@daily',  # Runs once a day automatically
        start_date=datetime(2026, 3, 1),
        catchup=False,
        tags=['weather_capstone', 'training'],
) as dag:
    # Task 1: Data Ingestion
    # We use BashOperator to run our Python script from the /opt/airflow root
    ingest_data = BashOperator(
        task_id='ingest_weather_data',
        bash_command='cd /opt/airflow && python src/ingestion/get_data.py'
    )

    # Task 2: Model Training
    # Notice how we can leverage Hydra CLI overrides right here in the DAG!
    train_model = BashOperator(
        task_id='train_forecast_model',
        bash_command='cd /opt/airflow && python src/training/train_model.py tracking.experiment_name="Airflow_Automated_Run" tracking.uri="http://mlflow_server:5000"'
    )

    # Task 3: Model Evaluation (Lab 5 Placeholder)
    evaluate_model = BashOperator(
        task_id='evaluate_champion_model',
        bash_command='cd /opt/airflow && python src/evaluation/evaluate_model.py'
    )

    # Define the execution logic (The Graph Edges)
    # Ingestion must succeed before Training, which must succeed before Evaluation
    ingest_data >> train_model >> evaluate_model
