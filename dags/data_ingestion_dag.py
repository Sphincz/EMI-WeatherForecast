# dags/data_ingestion_dag.py
from airflow.sdk import DAG
from airflow.sdk import Param
from airflow.sdk.definitions.param import ParamsDict
from airflow.sdk.definitions.dag import DAG
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

# Define the DAG with UI Parameters
with DAG(
    'daily_weather_ingestion',
    default_args=default_args,
    description='Fetches historical weather data incrementally',
    schedule='@daily',  # Runs once a day automatically
    catchup=False,
    tags=['weather_capstone', 'ingestion'],
    params=ParamsDict({
        "start_date": Param("2026-02-01", type="string", format="date", description="Start date (YYYY-MM-DD)"),
        "end_date": Param("2026-02-28", type="string", format="date", description="End date (YYYY-MM-DD)"),
    })
) as dag:
    # Task 1: Ingest Data for Lisbon
    ingest_lisbon = BashOperator(
        task_id='ingest_weather_lisbon',
        bash_command='cd /opt/airflow && python src/ingestion/get_historical_data.py '
                     'api.target_location="Lisbon" '
                     'api.date_range.start_date="{{ params.start_date }}" '
                     'api.date_range.end_date="{{ params.end_date }}"'
    )

    # Task 2: Ingest Data for Porto (Runs in parallel with Lisbon)
    ingest_porto = BashOperator(
        task_id='ingest_weather_porto',
        bash_command='cd /opt/airflow && python src/ingestion/get_historical_data.py '
                     'api.target_location="Porto" '
                     'api.date_range.start_date="{{ params.start_date }}" '
                     'api.date_range.end_date="{{ params.end_date }}"'
    )

    # Task 3: Update DVC Tracking
    update_dvc = BashOperator(
        task_id='update_dvc_tracking',
        bash_command='cd /opt/airflow && dvc add data/raw'
    )

    # Define the execution flow
    [ingest_lisbon, ingest_porto] >> update_dvc
