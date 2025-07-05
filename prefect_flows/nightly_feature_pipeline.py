# prefect_flows/nightly_feature_pipeline.py

import sys
from prefect import flow, task, get_run_logger
from prefect_shell import shell_run_command
import os
import datetime

# --- Project Root Setup ---
try:
    CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_SCRIPT_DIR))
except NameError: 
    PROJECT_ROOT = os.getcwd()

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Assume other tasks are defined correctly elsewhere
from etl.fetch_daily_bars import fetch_daily_bars_task

# --- MODIFICATION: Add new task for TCA Feature ETL ---
@task(name="Run TCA Feature ETL", retries=1, tags=["etl", "tca"])
async def run_tca_feature_etl_task():
    """
    Executes the script to process TCA logs and generate features.
    """
    task_run_logger = get_run_logger()
    tca_etl_script_path = os.path.join(PROJECT_ROOT, "etl", "tca_feature_etl.py")
    command = f"python {tca_etl_script_path}"
    
    task_run_logger.info(f"Executing TCA Feature ETL command: {command}")
    try:
        result = await shell_run_command(command=command, cwd=PROJECT_ROOT, return_all=True)
        stdout = "\n".join(res.stdout_lines for res in result) # type: ignore
        task_run_logger.info(f"TCA Feature ETL STDOUT:\n{stdout}")
        return {"status": "SUCCESS", "message": "TCA Feature ETL completed."}
    except Exception as e:
        task_run_logger.error(f"TCA Feature ETL script failed: {e}", exc_info=True)
        raise

@flow(name="nightly-feature-pipeline-flow",
      description="...",
      log_prints=True)
async def nightly_feature_pipeline_flow(config_path: str = "config.yaml"):
    flow_logger = get_run_logger()
    flow_logger.info(f"Starting Nightly ETL & Training Pipeline Flow using config: {config_path}")

    # --- MODIFICATION: Add the new TCA ETL task to the flow ---
    # This task can run in parallel with other feature generation tasks.
    tca_etl_future = run_tca_feature_etl_task.submit()

    # Placeholder for other daily tasks like materialization
    # e.g., materialize_future = await run_feast_materialize_task.submit(wait_for=[tca_etl_future])
    
    # Monthly Conditional ML Training
    ml_training_final_status_msg = "SKIPPED_NOT_FIRST_OF_MONTH"
    if datetime.date.today().day == 1:
        flow_logger.info("First of the month: Triggering ML model training cycle.")
        
        # The ML trainer task now depends on the TCA ETL being complete.
        # ml_trainer_future = await run_ml_model_trainer_task_flow.submit(wait_for=[tca_etl_future, materialize_future])
        # ml_training_final_status_msg = await ml_trainer_future.result(raise_on_failure=False)
        ml_training_final_status_msg = "ML Training would run here, now with TCA features available."

    tca_etl_result = await tca_etl_future.result(raise_on_failure=False)

    flow_logger.info("--- Nightly Pipeline Finished ---")
    flow_logger.info(f"  TCA Feature ETL Status: {tca_etl_result}")
    flow_logger.info(f"  ML Training Status: {ml_training_final_status_msg}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(nightly_feature_pipeline_flow())