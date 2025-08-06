from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import requests

# /reolad API 호출 함수
def trigger_reload():
    url = "http://3.35.129.98:8000/reload"  # 서버1 FastAPI 주소
    response = requests.post(url)
    print(response.status_code, response.json())

default_args = {
    "owner": "test111",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="test111",
    default_args=default_args,
    description="test444",
    schedule_interval=None,
    start_date=None,
    catchup=False,
) as dag:

    reload_task = PythonOperator(
        task_id="test444_task",
        python_callable=trigger_reload
    )

    reload_task
