# prefect_flows/nightly_feature_pipeline.py

import sys
from prefect import flow, task, get_run_logger
from prefect_shell import shell_run_command
import os
import datetime

@task(name="Run Primary Feature ETL", retries=1, tags=["etl", "features"])
async def run_primary_feature_etl_task():
    """
    Executes the script to process raw CSVs and generate primary features.
    """
    task_run_logger = get_run_logger()
    
    script_path = os.path.join("etl", "primary_feature_etl.py")
    command = f"python {script_path}"
    
    task_run_logger.info(f"Executing Primary Feature ETL command: '{command}'")
    
    try:
        stdout_result = await shell_run_command(command=command)
        task_run_logger.info(f"Primary Feature ETL STDOUT:\n{stdout_result}")
        return {"status": "SUCCESS", "message": "Primary Feature ETL completed."}
    except Exception as e:
        task_run_logger.error(f"Primary Feature ETL script failed: {e}", exc_info=True)
        raise

@task(name="Run TCA Feature ETL", retries=1, tags=["etl", "tca"])
async def run_tca_feature_etl_task():
    """
    Executes the script to process TCA logs and generate features.
    """
    task_run_logger = get_run_logger()
    
    script_path = os.path.join("etl", "tca_feature_etl.py")
    command = f"python {script_path}"
    
    task_run_logger.info(f"Executing TCA Feature ETL command: '{command}'")
    
    try:
        stdout_result = await shell_run_command(command=command)
        task_run_logger.info(f"TCA Feature ETL STDOUT:\n{stdout_result}")
        return {"status": "SUCCESS", "message": "TCA Feature ETL completed."}
    except Exception as e:
        task_run_logger.error(f"TCA Feature ETL script failed: {e}", exc_info=True)
        raise

@flow(name="nightly-feature-pipeline-flow",
      description="Orchestrates daily feature creation, TCA feature ETL, and monthly ML model training.",
      log_prints=True)
async def nightly_feature_pipeline_flow(config_path: str = "config.yaml"):
    """
    The main Prefect flow for nightly operations.
    """
    flow_logger = get_run_logger()
    flow_logger.info(f"Starting Nightly ETL & Training Pipeline Flow using config: {config_path}")

    primary_etl_future = run_primary_feature_etl_task.submit()
    tca_etl_future = run_tca_feature_etl_task.submit()
    
    # --- CORRECTED RESULT RETRIEVAL ---
    # The .result() method on a future blocks until the result is available.
    # It should NOT be awaited.
    primary_etl_result = primary_etl_future.result(raise_on_failure=False)
    tca_etl_result = tca_etl_future.result(raise_on_failure=False)
    
    ml_training_final_status_msg = "SKIPPED_NOT_FIRST_OF_MONTH"
    if datetime.date.today().day == 1:
        flow_logger.info("First of the month: Triggering ML model training cycle.")
        ml_training_final_status_msg = "ML Training would run here, now with all features available."

    flow_logger.info("--- Nightly Pipeline Finished ---")
    flow_logger.info(f"  Primary Feature ETL Status: {primary_etl_result}")
    flow_logger.info(f"  TCA Feature ETL Status: {tca_etl_result}")
    flow_logger.info(f"  ML Training Status: {ml_training_final_status_msg}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(nightly_feature_pipeline_flow())