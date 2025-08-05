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
    "owner": "airflow_dag",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="airflow_reload_model",
    default_args=default_args,
    description="Fast API 의 model reload 실행",
    schedule_interval=None,
    start_date=None,
    catchup=False,
) as dag:

    reload_task = PythonOperator(
        task_id="airflow_reload_model_task",
        python_callable=trigger_reload
    )

    reload_task
