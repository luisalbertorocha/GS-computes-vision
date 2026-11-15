"""DAG do Airflow (integração com a disciplina de Dados).

Copie/monte este arquivo na pasta dags/ do Airflow, com o repositório no
PYTHONPATH do worker. A DAG chama a mesma função usada pelo agendador simples.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator


def _run() -> dict:
    from src.config import load_settings
    from src.pipeline import run_once

    return run_once(load_settings(), trigger="airflow")


with DAG(
    dag_id="zeladoria_cv",
    description="Coleta imagens de rua, detecta danos e gera detections.parquet",
    start_date=datetime(2026, 11, 14),
    schedule="*/15 * * * *",
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=2)},
    tags=["visao-computacional", "zeladoria"],
) as dag:
    PythonOperator(task_id="coletar_processar_publicar", python_callable=_run)
