from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import requests

# /train API 호출 함수
def trigger_training():
    url = "http://127.0.0.1:8000/train"  # 서버1 FastAPI 주소
    payload = {"experiment_name": "airflow_daily_retrain"}
    response = requests.post(url, json=payload)
    print(response.status_code, response.json())

default_args = {
    "owner": "airflow_dag",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="airflow_daily_retrain",
    default_args=default_args,
    description="매일 새벽 모델 재학습",
    schedule_interval="54 9 * * *",
    start_date=datetime(2025, 8, 4, 0, 0),
    catchup=False,
) as dag:

    train_task = PythonOperator(
        task_id="airflow_daily_retrain_task",
        python_callable=trigger_training
    )

    train_task
