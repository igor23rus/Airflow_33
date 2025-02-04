import datetime as dt
import os
import sys
from airflow.models import DAG
from airflow.operators.python import PythonOperator


path = '/opt/airflow'
os.environ['PROJECT_PATH'] = path
sys.path.insert(0, path)

# <YOUR_IMPORTS>
from modules.pipeline import pipeline
from modules.predict import predict

args = {
    'owner': 'airflow',
    'start_date': dt.datetime(2025, 1, 19),
    'retries': 1,
    'retry_delay': dt.timedelta(minutes=1),
    'depends_on_past': False,
}

with DAG(
        dag_id='car_price_prediction',
        schedule="00 15 * * *",
        default_args=args,
) as dag:
    pipeline = PythonOperator(
        task_id='pipeline',
        python_callable=pipeline,
        dag=dag
    )

    predict = PythonOperator(
        task_id='predict_price',
        python_callable=predict,
        dag=dag
    )

    pipeline >> predict
