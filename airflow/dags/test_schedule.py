from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

def test():
    print("✅ DAG 실행됨!")

default_args = {
    "owner": "me",
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    dag_id="test_schedule_every_3min",
    default_args=default_args,
    start_date=datetime(2025, 8, 5),
    schedule="*/3 * * * *",  # 3분마다 실행
    catchup=False,
    tags=["test"],
) as dag:
    run = PythonOperator(
        task_id="run_test",
        python_callable=test,
    )

    run 
