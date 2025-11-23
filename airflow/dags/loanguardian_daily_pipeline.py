 
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os

# Add utils folder to path
dag_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dag_path, "utils"))

from data_cleaning import clean_data
from feature_engineering import engineer_features
from model_training import train_model
from prediction import generate_predictions

default_args = {
    'owner': 'Rajit R Krishna',
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=2)
}

with DAG(
    dag_id='loanguardian_daily_pipeline',
    default_args=default_args,
    description='Daily ML workflow for LoanGuardian',
    schedule_interval='@daily',
    start_date=datetime(2024, 1, 1),
    catchup=False
) as dag:

    clean_task = PythonOperator(
        task_id='clean_data',
        python_callable=clean_data,
        op_kwargs={
            'input_path': '/opt/data/raw/loan_raw.csv',
            'output_path': '/opt/data/processed/cleaned.csv'
        }
    )

    feature_task = PythonOperator(
        task_id='feature_engineering',
        python_callable=engineer_features,
        op_kwargs={
            'input_path': '/opt/data/processed/cleaned.csv',
            'output_path': '/opt/data/processed/features.csv'
        }
    )

    train_task = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
        op_kwargs={
            'train_data_path': '/opt/data/processed/features.csv',
            'model_output_path': '/opt/models/loan_rf.pkl'
        }
    )

    predict_task = PythonOperator(
        task_id='generate_predictions',
        python_callable=generate_predictions,
        op_kwargs={
            'model_path': '/opt/models/loan_rf.pkl',
            'input_path': '/opt/data/processed/features.csv',
            'output_path': '/opt/data/output/predictions.csv'
        }
    )

    clean_task >> feature_task >> train_task >> predict_task
