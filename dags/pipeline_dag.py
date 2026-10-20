import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow.decorators import dag, task

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.macro.data_extraction import extract_all_raw_data
from src.macro.data_processing import process_and_thin_data
from src.macro.feature_extraction import run_advanced_feature_extraction
from src.macro.generate_drought_labels import extract_drought_resilience_labels
from src.macro.inference import generate_risk_map
from src.macro.model_training import train_spatial_sdm
from utils.generate_future_risk_maps import generate_future_projections

default_args = {
    "owner": "danil",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=3),
    "retry_exponential_backoff": True,
}


@dag(
    dag_id="forest_drought_vulnerability_pipeline",
    default_args=default_args,
    description="Multi-Scale Cloud-Native GeoAI Pipeline: Forest Vulnerability to Hotter Droughts (CMIP6 / Zarr / STAC)",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=[
        "geoai",
        "forest-vulnerability",
        "hotter-droughts",
        "xgboost",
        "spatial-cv",
        "planetary-computer",
        "cmip6",
    ],
)
def forest_drought_vulnerability_dag():
    @task(task_id="ingest_forest_inventory_data")
    def task_extract():
        extract_all_raw_data()

    @task(task_id="spatial_rarefaction_1km")
    def task_thinning():
        process_and_thin_data()

    @task(task_id="stratify_drought_resilience_labels")
    def task_sampling():
        extract_drought_resilience_labels()

    @task(task_id="cloud_feature_engineering", execution_timeout=timedelta(hours=1))
    def task_features():
        run_advanced_feature_extraction()

    @task(task_id="train_spatial_xgboost")
    def task_train():
        train_spatial_sdm()

    @task(task_id="generate_spatial_vulnerability_map")
    def task_infer():
        generate_risk_map()

    @task(task_id="generate_cmip6_climate_projections")
    def task_projections():
        generate_future_projections()

    t1 = task_extract()
    t2 = task_thinning()
    t3 = task_sampling()
    t4 = task_features()
    t5 = task_train()
    t6 = task_infer()
    t7 = task_projections()

    t1 >> t2 >> t3 >> t4 >> t5 >> t6 >> t7


dag_instance = forest_drought_vulnerability_dag()
