# prefect_flows/nightly_feature_pipeline.py

import sys
from prefect import flow, task, get_run_logger
from prefect_shell import shell_run_command
import os
import datetime

@task(name="Run TCA Feature ETL", retries=1, tags=["etl", "tca"])
async def run_tca_feature_etl_task():
    """
    Executes the script to process TCA logs and generate features.
    """
    task_run_logger = get_run_logger()
    
    tca_etl_script_path = os.path.join("etl", "tca_feature_etl.py")
    command = f"python {tca_etl_script_path}"
    
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

    tca_etl_result = await run_tca_feature_etl_task()

    # If you needed to run tasks in parallel, the pattern would be:
    # future = task.submit()
    # result = await future.result()
    # The previous code was just missing the await on the final .result() call.
    # But for this simple flow, direct await is cleaner.
    
    ml_training_final_status_msg = "SKIPPED_NOT_FIRST_OF_MONTH"
    if datetime.date.today().day == 1:
        flow_logger.info("First of the month: Triggering ML model training cycle.")
        ml_training_final_status_msg = "ML Training would run here, now with TCA features available."

    flow_logger.info("--- Nightly Pipeline Finished ---")
    flow_logger.info(f"  TCA Feature ETL Status: {tca_etl_result}")
    flow_logger.info(f"  ML Training Status: {ml_training_final_status_msg}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(nightly_feature_pipeline_flow())