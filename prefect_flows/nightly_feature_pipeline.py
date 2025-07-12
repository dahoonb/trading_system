# prefect_flows/nightly_feature_pipeline.py

import sys
import os
import datetime
from prefect import flow, task, get_run_logger
from prefect_shell import shell_run_command

# --- Project Root Setup ---
# This ensures that the script can find other modules in the project
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- Task Definitions ---

@task(name="Run Primary Feature ETL", retries=1, tags=["etl", "features"])
async def run_primary_feature_etl_task():
    """
    Executes the script to process raw CSVs and generate primary technical features.
    """
    task_run_logger = get_run_logger()
    script_path = os.path.join("etl", "primary_feature_etl.py")
    command = f"python {script_path}"
    
    task_run_logger.info(f"Executing Primary Feature ETL command: '{command}'")
    
    try:
        stdout_result = await shell_run_command(command=command, return_all=True)
        # Log the full output for better debugging
        for line in stdout_result:
            task_run_logger.info(line)
        return {"status": "SUCCESS", "message": "Primary Feature ETL completed."}
    except Exception as e:
        task_run_logger.error(f"Primary Feature ETL script failed: {e}", exc_info=True)
        raise

@task(name="Run TCA Feature ETL", retries=1, tags=["etl", "tca"])
async def run_tca_feature_etl_task():
    """
    Executes the script to process TCA logs and generate execution-based features.
    """
    task_run_logger = get_run_logger()
    script_path = os.path.join("etl", "tca_feature_etl.py")
    command = f"python {script_path}"
    
    task_run_logger.info(f"Executing TCA Feature ETL command: '{command}'")
    
    try:
        stdout_result = await shell_run_command(command=command, return_all=True)
        for line in stdout_result:
            task_run_logger.info(line)
        return {"status": "SUCCESS", "message": "TCA Feature ETL completed."}
    except Exception as e:
        task_run_logger.error(f"TCA Feature ETL script failed: {e}", exc_info=True)
        raise

@task(name="Run Fundamental Feature ETL", retries=1, tags=["etl", "fundamentals"])
async def run_fundamental_feature_etl_task():
    """
    Executes the script to process financial statements and generate fundamental features.
    """
    task_run_logger = get_run_logger()
    script_path = os.path.join("etl", "fundamental_feature_etl.py")
    command = f"python {script_path}"
    
    task_run_logger.info(f"Executing Fundamental Feature ETL command: '{command}'")
    
    try:
        stdout_result = await shell_run_command(command=command, return_all=True)
        for line in stdout_result:
            task_run_logger.info(line)
        return {"status": "SUCCESS", "message": "Fundamental Feature ETL completed."}
    except Exception as e:
        task_run_logger.error(f"Fundamental Feature ETL script failed: {e}", exc_info=True)
        raise

@task(name="Materialize Features to Online Store", retries=1, tags=["feast", "materialize"])
async def materialize_features_task():
    """
    Runs 'feast materialize' to sync the latest features from the offline
    store to the online store, making them available for live trading.
    """
    task_run_logger = get_run_logger()
    feature_repo_path = os.path.join(PROJECT_ROOT, "feature_repo")
    
    # The end date is today, and the start date is a few days ago to ensure
    # we capture all recent updates.
    end_date = datetime.datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.datetime.now() - datetime.timedelta(days=5)).strftime('%Y-%m-%d')
    
    command = f"feast materialize {start_date}T00:00:00 {end_date}T23:59:59"
    
    task_run_logger.info(f"Executing Feast Materialize command in '{feature_repo_path}': '{command}'")
    
    try:
        # The command must be run from within the feature repo directory
        stdout_result = await shell_run_command(command=command, return_all=True, cwd=feature_repo_path)
        for line in stdout_result:
            task_run_logger.info(line)
        return {"status": "SUCCESS", "message": "Feast materialize completed."}
    except Exception as e:
        task_run_logger.error(f"Feast materialize failed: {e}", exc_info=True)
        raise

@task(name="Train ML Models", retries=0, tags=["ml", "training"])
async def train_ml_models_task():
    """
    Executes the ML model training script. This is typically run less frequently.
    """
    task_run_logger = get_run_logger()
    script_path = os.path.join("ml", "ml_model_trainer.py")
    command = f"python {script_path}"
    
    task_run_logger.info(f"Executing ML Model Training command: '{command}'")
    
    try:
        stdout_result = await shell_run_command(command=command, return_all=True)
        for line in stdout_result:
            task_run_logger.info(line)
        return {"status": "SUCCESS", "message": "ML model training completed."}
    except Exception as e:
        task_run_logger.error(f"ML model training script failed: {e}", exc_info=True)
        raise

# --- Main Flow Definition ---

@flow(name="nightly-feature-pipeline-flow",
      description="Orchestrates daily feature creation, materialization, and monthly ML model training.",
      log_prints=True)
async def nightly_feature_pipeline_flow(config_path: str = "config.yaml"):
    """
    The main Prefect flow for nightly operations. This flow:
    1. Runs all feature engineering ETL tasks in parallel.
    2. Waits for them to complete, then materializes the new features to the online store.
    3. On the first day of the month, it triggers the ML model retraining pipeline.
    """
    flow_logger = get_run_logger()
    flow_logger.info(f"--- Starting Nightly ETL & Training Pipeline Flow (using config: {config_path}) ---")

    # --- Step 1: Run all feature engineering tasks in parallel ---
    primary_etl_future = run_primary_feature_etl_task.submit()
    tca_etl_future = run_tca_feature_etl_task.submit()
    fundamental_etl_future = run_fundamental_feature_etl_task.submit()
    
    # --- Step 2: Wait for ETL to finish, then materialize ---
    # The 'wait_for' argument ensures that materialize_features_task only starts
    # after all three ETL tasks have successfully completed.
    materialize_future = materialize_features_task.submit(
        wait_for=[primary_etl_future, tca_etl_future, fundamental_etl_future]
    )
    
    # --- Step 3: Conditionally run ML training ---
    # This logic runs independently but also waits for materialization to be done.
    ml_training_final_status_msg = "SKIPPED: Not the first day of the month."
    if datetime.date.today().day == 1:
        flow_logger.info("First of the month: Triggering ML model training cycle.")
        # The training task also waits for materialization to ensure it uses the latest data.
        ml_training_future = train_ml_models_task.submit(wait_for=[materialize_future])
        # We can retrieve the result to confirm completion
        ml_result = ml_training_future.result(raise_on_failure=False)
        ml_training_final_status_msg = f"ML Training run status: {ml_result}"

    # Retrieve final statuses for logging
    primary_etl_result = primary_etl_future.result(raise_on_failure=False)
    tca_etl_result = tca_etl_future.result(raise_on_failure=False)
    fundamental_etl_result = fundamental_etl_future.result(raise_on_failure=False)
    materialize_result = materialize_future.result(raise_on_failure=False)

    flow_logger.info("--- Nightly Pipeline Finished ---")
    flow_logger.info(f"  - Primary Feature ETL Status: {primary_etl_result}")
    flow_logger.info(f"  - TCA Feature ETL Status: {tca_etl_result}")
    flow_logger.info(f"  - Fundamental Feature ETL Status: {fundamental_etl_result}")
    flow_logger.info(f"  - Feast Materialize Status: {materialize_result}")
    flow_logger.info(f"  - ML Training Status: {ml_training_final_status_msg}")

if __name__ == "__main__":
    import asyncio
    # This allows running the flow directly from the command line for testing
    asyncio.run(nightly_feature_pipeline_flow())