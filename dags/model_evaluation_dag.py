# dags/model_evaluation_dag.py
import os
from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from datetime import datetime, timedelta

# Dynamically scan for available models
CONF_MODEL_DIR = "/opt/airflow/conf/model"
try:
    available_models = [f.replace('.yaml', '') for f in os.listdir(CONF_MODEL_DIR) if f.endswith('.yaml')]
except FileNotFoundError:
    available_models = ["lstm", "gru", "prophet"]

# Default arguments applied to all tasks in the DAG
default_args = {
    'owner': 'mlops_engineer',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 5,
    'retry_delay': timedelta(seconds=5),
}

# Define the Evaluation DAG to run DAILY
with DAG(
        'daily_model_evaluation',
        default_args=default_args,
        description='Evaluates active ML models daily',
        schedule='@daily',
        catchup=False,
        tags=['weather_capstone', 'evaluation'],
) as dag:
    # Dynamically generate an evaluation task for EVERY model
    for model_name in available_models:
        evaluate_model_task = BashOperator(
            task_id=f'evaluate_{model_name}_model',
            bash_command=f'cd /opt/airflow && python src/evaluation/evaluate_model.py model={model_name}'
        )
