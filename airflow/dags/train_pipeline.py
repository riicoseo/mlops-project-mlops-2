import os
import sys
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
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
    run_crawler = BashOperator(
        task_id = "airflow_daily_reclawler_task"
        bash_command="python /data-prepare/main.py"
    )

    run_train = BashOperator(
        task_id = "airflow_daily_retrain_task"
        bash_command = """
rm -rf /src/dataset/cache && 
python /src/main.py train lightgbm &&
python /src/main.py train randomforest &&
python /src/main.py train xgboost &&
echo finish train!
"""
    )

    run_crawler >> run_train
