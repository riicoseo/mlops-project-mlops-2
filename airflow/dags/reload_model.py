from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

# /reolad API 호출 함수
def trigger_reload():
    import requests
    import logging
    logger = logging.getLogger("airflow.task")
    url = "http://3.35.129.98:8000/reload"
    try:
        logger.info(f"Calling reload API: {url}")
        response = requests.post(url, timeout=30)
        logger.info(f"Status Code: {response.status_code}")
        logger.info(f"Response: {response.text}")
        response.raise_for_status()
    except Exception as e:
        logger.error(f"Error calling reload API: {e}")
        raise

default_args = {
    "owner": "admin",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="reload_model",
    default_args=default_args,
    description="Fast API 의 model reload 실행",
    schedule="30 3 * * *",
    start_date=datetime(2025, 8, 6),
    catchup=False,
) as dag:

    reload_task = PythonOperator(
        task_id="reload_model_task",
        python_callable=trigger_reload
    )

    reload_task
