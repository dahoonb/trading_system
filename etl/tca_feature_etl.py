# etl/tca_feature_etl.py

import duckdb
import pandas as pd
import numpy as np
from pandas.util import hash_pandas_object
import os
import sys
from datetime import datetime, timedelta, UTC
import hashlib

# --- Project Root Setup ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.logger import setup_logger

logger = setup_logger(logger_name="TCA_Feature_ETL")

def stable_id(symbol: str) -> int:
    # FIX: Generate a SIGNED 64-bit integer to ensure compatibility with Protobuf.
    return int.from_bytes(
        hashlib.blake2b(symbol.encode(), digest_size=8).digest(),
        byteorder="big",
        signed=True # Changed from False to True
    )

def run_tca_feature_etl(tca_db_path: str, output_path: str):
    """
    Reads raw execution logs from a DuckDB, calculates time-windowed
    TCA features, and saves them to a Parquet file.

    This version converts string tickers to 64-bit integer IDs and ensures
    all timestamps are timezone-aware to align with the Feast entity definition
    and prevent data type conflicts.
    """
    logger.info("Starting TCA Feature ETL process...")

    if not os.path.exists(tca_db_path):
        logger.warning(f"TCA database not found at '{tca_db_path}'. Skipping ETL.")
        # Create an empty DataFrame with the correct schema for a cold start.
        # This is crucial for the first run of `feast materialize`.
        empty_df = pd.DataFrame({
            'event_timestamp': pd.Series([], dtype='datetime64[ns, UTC]'),
            'ticker_id': pd.Series([], dtype='int64'),
            'avg_slippage_5d': pd.Series([], dtype='float64'),
            'created_timestamp': pd.Series([], dtype='datetime64[ns, UTC]')
        })
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        empty_df.to_parquet(
            output_path,
            engine='pyarrow',
            use_dictionary=False,
            compression='snappy',
            index=False
        )
        logger.info(f"Created empty TCA features file with correct schema at '{output_path}'.")
        return

    try:
        with duckdb.connect(tca_db_path, read_only=True) as con:
            query = "SELECT timestamp, symbol, slippage FROM execution_log"
            log_df = con.execute(query).fetchdf()
    except Exception as e:
        logger.error(f"Failed to read from TCA database: {e}", exc_info=True)
        return

    if log_df.empty:
        logger.info("No execution data in TCA database. Nothing to process.")
        return

    logger.info(f"Loaded {len(log_df)} execution records from the TCA database.")

    # --- Data Transformation and Feature Engineering ---

    # 1. Ensure timestamp is timezone-aware (UTC) for Feast compatibility.
    log_df['timestamp'] = pd.to_datetime(log_df['timestamp'], utc=True)
    log_df.set_index('timestamp', inplace=True)
    log_df = log_df.sort_index()

    # 2. Convert string symbol to a consistent 64-bit integer ID.
    log_df['ticker_id'] = log_df['symbol'].map(stable_id).astype(np.int64)
    log_df.drop(columns='symbol', inplace=True)

    # 3. Calculate the feature: 5-day rolling average of slippage per ticker.
    features_df = log_df.groupby('ticker_id')['slippage'].rolling(window='5D').mean().reset_index()
    
    # 4. Clean up columns for the final feature table.
    features_df.rename(columns={'timestamp': 'event_timestamp', 'slippage': 'avg_slippage_5d'}, inplace=True)
    features_df['created_timestamp'] = datetime.now(UTC)

    # 5. Enforce final data types for maximum compatibility.
    features_df['ticker_id'] = features_df['ticker_id'].astype('int64')
    features_df['event_timestamp'] = pd.to_datetime(features_df['event_timestamp'], utc=True)
    features_df['avg_slippage_5d'] = features_df['avg_slippage_5d'].astype('float64')

    # --- Save to Parquet ---
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        features_df.to_parquet(
            output_path,
            engine='pyarrow',
            use_dictionary=False,
            compression='snappy',
            index=False
        )
        logger.info(f"Successfully saved {len(features_df)} TCA features to '{output_path}'.")
    except Exception as e:
        logger.error(f"Failed to save TCA features to Parquet file: {e}", exc_info=True)

if __name__ == "__main__":
    TCA_DB_PATH = os.path.join(PROJECT_ROOT, "data", "tca_log.duckdb")
    OUTPUT_PARQUET_PATH = os.path.join(PROJECT_ROOT, "data", "tca_features.parquet")
    run_tca_feature_etl(tca_db_path=TCA_DB_PATH, output_path=OUTPUT_PARQUET_PATH)