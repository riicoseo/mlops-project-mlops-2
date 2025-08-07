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

def trigger_crawling():
    url = "http://3.35.129.98:8000/airflow/crawling"  
    response = requests.post(url)
    print(response.status_code, response.json())

def trigger_train():
    url = "http://3.35.129.98:8000/airflow/train"  
    response = requests.post(url)
    print(response.status_code, response.json())    

default_args = {
    "owner": "admin",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="daily_crawling_and_train",
    default_args=default_args,
    description="매일 새벽 크롤링 & 모델 학습",
    schedule="54 9 * * *",
    start_date=datetime(2025, 8, 6),
    catchup=False,
) as dag:
    run_crawling = PythonOperator(
        task_id="run_crawling_task",
        python_callable=trigger_crawling
    )

    run_train = PythonOperator(
        task_id="run_train_task",
        python_callable=trigger_train
    )

    run_crawling >> run_train
