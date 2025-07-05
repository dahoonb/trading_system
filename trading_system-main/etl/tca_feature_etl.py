# etl/tca_feature_etl.py

import duckdb
import pandas as pd
import os
import sys
from datetime import datetime, timedelta

# --- Project Root Setup ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.logger import setup_logger

logger = setup_logger(logger_name="TCA_Feature_ETL")

def run_tca_feature_etl(tca_db_path: str, output_path: str):
    """
    Extracts execution data from the TCA database, computes aggregate
    features, and saves them to a Parquet file for Feast.
    """
    logger.info("Starting TCA Feature ETL process...")

    if not os.path.exists(tca_db_path):
        logger.warning(f"TCA database not found at '{tca_db_path}'. Skipping ETL.")
        # Create an empty dataframe with the correct schema if the file doesn't exist
        # to ensure downstream tasks don't fail on file not found.
        schema = {
            'event_timestamp': pd.Series(dtype='datetime64[ns]'),
            'ticker_id': pd.Series(dtype='object'),
            'avg_slippage_5d': pd.Series(dtype='float64'),
            'created_timestamp': pd.Series(dtype='datetime64[ns, UTC]')
        }
        empty_df = pd.DataFrame(schema)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        empty_df.to_parquet(output_path, index=False)
        logger.info(f"Created empty TCA features file at '{output_path}'.")
        return

    try:
        with duckdb.connect(tca_db_path, read_only=True) as con:
            # Load all execution data
            query = "SELECT timestamp, symbol, slippage FROM execution_log"
            log_df = con.execute(query).fetchdf()
    except Exception as e:
        logger.error(f"Failed to read from TCA database: {e}", exc_info=True)
        return

    if log_df.empty:
        logger.info("No execution data in TCA database. Nothing to process.")
        return

    logger.info(f"Loaded {len(log_df)} execution records from the TCA database.")

    # --- Feature Engineering ---
    log_df['timestamp'] = pd.to_datetime(log_df['timestamp'])
    log_df.set_index('timestamp', inplace=True)
    log_df.rename(columns={'symbol': 'ticker_id'}, inplace=True)

    # Calculate 5-day rolling average slippage per symbol
    # The result will have a timestamp for each day a trade occurred.
    features_df = log_df.groupby('ticker_id')['slippage'].rolling(window='5D').mean().reset_index()
    features_df.rename(columns={'timestamp': 'event_timestamp', 'slippage': 'avg_slippage_5d'}, inplace=True)
    
    # Add the required created_timestamp for Feast
    features_df['created_timestamp'] = datetime.utcnow().replace(tzinfo=pd.Timestamp.now().tz)

    # Ensure data types are correct for Parquet and Feast
    features_df['event_timestamp'] = pd.to_datetime(features_df['event_timestamp'])
    features_df['avg_slippage_5d'] = features_df['avg_slippage_5d'].astype('float64')

    # Save to Parquet
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        features_df.to_parquet(output_path, index=False)
        logger.info(f"Successfully saved {len(features_df)} TCA features to '{output_path}'.")
    except Exception as e:
        logger.error(f"Failed to save TCA features to Parquet file: {e}", exc_info=True)

if __name__ == "__main__":
    # This allows the script to be run manually for testing
    TCA_DB_PATH = os.path.join(PROJECT_ROOT, "data", "tca_log.duckdb")
    OUTPUT_PARQUET_PATH = os.path.join(PROJECT_ROOT, "data", "tca_features.parquet")
    run_tca_feature_etl(tca_db_path=TCA_DB_PATH, output_path=OUTPUT_PARQUET_PATH)