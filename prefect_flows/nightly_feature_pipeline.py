# prefect_flows/nightly_feature_pipeline.py

import sys
from prefect import flow, task, get_run_logger
from prefect_shell import shell_run_command
import os
import datetime

# Note: The sys.path manipulation is no longer needed at the top level of this file.
# The Prefect worker's execution environment and the `git_clone` step ensure that
# the working directory is the project root, and individual scripts like
# `tca_feature_etl.py` handle their own necessary path adjustments if needed.

@task(name="Run TCA Feature ETL", retries=1, tags=["etl", "tca"])
async def run_tca_feature_etl_task():
    """
    Executes the script to process TCA logs and generate features.
    """
    task_run_logger = get_run_logger()

    # The Prefect deployment's `git_clone` step sets the working directory to the
    # root of the repository. Therefore, we can use a simple relative path.
    tca_etl_script_path = os.path.join("etl", "tca_feature_etl.py")
    command = f"python {tca_etl_script_path}"
    
    task_run_logger.info(f"Executing TCA Feature ETL command: '{command}' from current directory: '{os.getcwd()}'")
    
    try:
        # We no longer need the `cwd` argument because the worker is already in the correct directory.
        result = await shell_run_command(command=command, return_all=True)
        
        # The result from shell_run_command is a list of completed processes
        stdout = "\n".join(res.stdout for res in result)
        task_run_logger.info(f"TCA Feature ETL STDOUT:\n{stdout}")
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

    # Submit the task to run in the background. This is non-blocking.
    tca_etl_future = run_tca_feature_etl_task.submit()

    # Placeholder for other daily tasks that could run in parallel
    # e.g., materialize_future = run_feast_materialize_task.submit()
    
    # Monthly Conditional ML Training
    ml_training_final_status_msg = "SKIPPED_NOT_FIRST_OF_MONTH"
    if datetime.date.today().day == 1:
        flow_logger.info("First of the month: Triggering ML model training cycle.")
        
        # To enforce a dependency, you would pass the future object to the next task's submit call.
        # e.g., ml_trainer_future = run_ml_model_trainer_task_flow.submit(wait_for=[tca_etl_future])
        # ml_training_result = await ml_trainer_future.result()
        ml_training_final_status_msg = "ML Training would run here, now with TCA features available."

    # Now, explicitly wait for the TCA ETL task to finish and get its result.
    # This is a blocking call within the async flow.
    tca_etl_result = await tca_etl_future.result(raise_on_failure=False)

    flow_logger.info("--- Nightly Pipeline Finished ---")
    flow_logger.info(f"  TCA Feature ETL Status: {tca_etl_result}")
    flow_logger.info(f"  ML Training Status: {ml_training_final_status_msg}")

if __name__ == "__main__":
    # This block allows you to run the flow directly from the command line
    # for local testing, e.g., `python prefect_flows/nightly_feature_pipeline.py`
    import asyncio
    asyncio.run(nightly_feature_pipeline_flow())