# etl/primary_feature_etl.py

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
import talib

# --- Project Root Setup ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.logger import setup_logger

logger = setup_logger(logger_name="Primary_Feature_ETL")

def run_primary_feature_etl(data_dir: str, output_path: str):
    """
    Reads raw historical CSVs, computes technical indicators and other features,
    and saves them to a Parquet file for Feast.
    """
    logger.info("--- Starting Primary Feature ETL Process ---")
    
    all_features = []
    
    try:
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        if not csv_files:
            logger.error(f"No CSV files found in '{data_dir}'. Aborting.")
            return
            
        logger.info(f"Found {len(csv_files)} CSV files to process.")

        for file in csv_files:
            symbol = file.replace('.csv', '')
            file_path = os.path.join(data_dir, file)
            df = pd.read_csv(file_path, parse_dates=['date'])
            
            # --- Feature Engineering ---
            df['sma_20'] = talib.SMA(df['close'], timeperiod=20)
            df['sma_200'] = talib.SMA(df['close'], timeperiod=200)
            df['rsi_14'] = talib.RSI(df['close'], timeperiod=14)
            df['atr_20'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=20)
            
            # Volatility Regime (example: 1 if 20-day vol > 200-day vol, else 0)
            df['vol_20d'] = df['close'].pct_change().rolling(20).std()
            df['vol_200d'] = df['close'].pct_change().rolling(200).std()
            df['vol_regime'] = np.where(df['vol_20d'] > df['vol_200d'], 1, 0)
            
            # Other example features from your feature view
            df['realized_vol_20d'] = df['close'].pct_change().rolling(20).std() * np.sqrt(252)
            df['oer_5d'] = (df['high'] - df['low']) / df['open'].rolling(5).mean()
            
            # Add ticker_id for Feast
            df['ticker_id'] = symbol
            
            # Rename date column for Feast
            df.rename(columns={'date': 'event_timestamp'}, inplace=True)
            
            # Add created_timestamp for Feast
            df['created_timestamp'] = datetime.utcnow().replace(tzinfo=pd.Timestamp.now().tz)
            
            all_features.append(df)

        if not all_features:
            logger.warning("No features were generated.")
            return

        # Combine all dataframes and drop rows with NaNs from rolling calculations
        combined_df = pd.concat(all_features, ignore_index=True).dropna()
        
        # Ensure correct data types
        for col in ['open', 'high', 'low', 'close', 'volume', 'sma_20', 'sma_200', 'rsi_14', 'atr_20', 'realized_vol_20d', 'oer_5d']:
            if col in combined_df.columns:
                combined_df[col] = combined_df[col].astype('float64')
        
        for col in ['vol_regime']:
             if col in combined_df.columns:
                combined_df[col] = combined_df[col].astype('int64')

        # Save to Parquet
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        combined_df.to_parquet(output_path, index=False)
        logger.info(f"Successfully saved {len(combined_df)} primary features to '{output_path}'.")

    except Exception as e:
        logger.error(f"An error occurred during primary feature ETL: {e}", exc_info=True)

if __name__ == "__main__":
    DATA_DIR = os.path.join(PROJECT_ROOT, "data", "historical_csv")
    OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "features.parquet")
    run_primary_feature_etl(data_dir=DATA_DIR, output_path=OUTPUT_PATH)