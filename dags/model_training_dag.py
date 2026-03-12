# dags/model_training_dag.py
import os
from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from datetime import datetime, timedelta
from airflow.sdk import Param
from airflow.sdk.definitions.param import ParamsDict

# 1. Dynamically scan the Hydra configuration directory
CONF_MODEL_DIR = "/opt/airflow/conf/model"
try:
    available_models = [
        f.replace('.yaml', '')
        for f in os.listdir(CONF_MODEL_DIR)
        if f.endswith('.yaml')
    ]
except FileNotFoundError:
    # Fallback safety
    available_models = ["lstm", "gru", "prophet"]

# 2. Add 'all' as the default option for our UI dropdown
dropdown_options = ["all"] + available_models

# Default arguments applied to all tasks in the DAG
default_args = {
    'owner': 'mlops_engineer',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 5,
    'retry_delay': timedelta(seconds=5),
}

# 3. Define the DAG with the dynamic UI parameter
with DAG(
        'monthly_model_training',
        default_args=default_args,
        description='Trains all models autonomously, or a specific user-selected model.',
        schedule='@monthly',
        catchup=False,
        tags=['weather_capstone', 'training'],
        params=ParamsDict({
            "target_model": Param(
                default="all",
                enum=dropdown_options,
                description="Select 'all' to run every model, or pick a specific one to train."
            )
        })
) as dag:
    # 4. Dynamically generate a training task for EVERY model found
    for model_name in available_models:
        # Bash logic: If UI is 'all' OR matches this specific task's model, run the training.
        # Otherwise, exit with 99. Airflow interprets exit 99 as a "Skipped" task state.
        run_logic = (
            f"if [ '{{{{ params.target_model }}}}' = 'all' ] || [ '{{{{ params.target_model }}}}' = '{model_name}' ]; then "
            f"cd /opt/airflow && python src/training/train_model.py model={model_name}; "
            "else "
            f"echo 'Skipping {model_name} training based on UI selection.'; "
            "exit 99; "
            "fi"
        )

        train_model_task = BashOperator(
            task_id=f'train_{model_name}_model',
            bash_command=run_logic
        )
