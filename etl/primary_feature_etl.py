# etl/primary_feature_etl.py

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, UTC
import talib
import hashlib

# --- Project Root Setup ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.logger import setup_logger

logger = setup_logger(logger_name="Primary_Feature_ETL")

# ----------------------------------------------------------------------
# Stable 64-bit hash helper. 8-byte digest → signed int64
def stable_id(sym: str) -> int:
    return int.from_bytes(
        hashlib.blake2b(sym.encode(), digest_size=8).digest(),
        byteorder="big",
        signed=True
    )
# ----------------------------------------------------------------------

def run_primary_feature_etl(data_dir: str, output_path: str):
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
            ticker_id_int = stable_id(symbol)
            
            file_path = os.path.join(data_dir, file)
            df = pd.read_csv(file_path, parse_dates=['date'])
            
            # --- Standard Technical Indicators ---
            df['sma_20'] = talib.SMA(df['close'], timeperiod=20)
            df['sma_200'] = talib.SMA(df['close'], timeperiod=200)
            df['rsi_14'] = talib.RSI(df['close'], timeperiod=14)
            df['atr_20'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=20)
            
            df['vol_20d'] = df['close'].pct_change().rolling(20).std()
            df['vol_200d'] = df['close'].pct_change().rolling(200).std()
            df['vol_regime'] = np.where(df['vol_20d'] > df['vol_200d'], 1, 0)
            
            df['realized_vol_20d'] = df['close'].pct_change().rolling(20).std() * np.sqrt(252)
            df['oer_5d'] = (df['high'] - df['low']) / df['open'].rolling(5).mean()
            
            df['ticker_id'] = ticker_id_int
            df.rename(columns={'date': 'event_timestamp'}, inplace=True)
            df['event_timestamp'] = pd.to_datetime(df['event_timestamp'], utc=True)
            
            all_features.append(df)

        if not all_features:
            logger.warning("No features were generated. Aborting.")
            return

        combined_df = pd.concat(all_features, ignore_index=True)
        
        # --- FIX: Add Cross-Sectional Momentum Rank Features ---
        logger.info("Calculating cross-sectional momentum rank features...")
        lookback_periods = [21, 63, 126, 252] # Approx 1, 3, 6, 12 months
        
        # Calculate returns over lookback periods
        for period in lookback_periods:
            combined_df[f'return_{period}d'] = combined_df.groupby('ticker_id')['close'].pct_change(periods=period)

        # Calculate cross-sectional ranks for each day
        for period in lookback_periods:
            rank_col_name = f'cs_rank_{period}d'
            combined_df[rank_col_name] = combined_df.groupby('event_timestamp')[f'return_{period}d'].rank(pct=True)
        
        # Clean up intermediate return columns
        combined_df.drop(columns=[f'return_{period}d' for period in lookback_periods], inplace=True)
        logger.info("Finished calculating cross-sectional features.")

        combined_df['created_timestamp'] = datetime.now(UTC)
        combined_df.dropna(inplace=True)
        
        # --- Type Casting ---
        float_cols = [
            'open', 'high', 'low', 'close', 'volume', 'sma_20', 'sma_200', 'rsi_14', 
            'atr_20', 'realized_vol_20d', 'oer_5d', 'cs_rank_21d', 'cs_rank_63d', 
            'cs_rank_126d', 'cs_rank_252d'
        ]
        for col in float_cols:
            if col in combined_df.columns:
                combined_df[col] = combined_df[col].astype('float64')
        
        int_cols = ['vol_regime', 'ticker_id']
        for col in int_cols:
             if col in combined_df.columns:
                combined_df[col] = combined_df[col].astype('int64')

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        combined_df.to_parquet(
            output_path,
            index=False,
            engine='pyarrow',
            use_dictionary=False,
            compression='snappy',
        )
        logger.info(f"Successfully saved {len(combined_df)} primary features to '{output_path}'.")

    except Exception as e:
        logger.error(f"An error occurred during primary feature ETL: {e}", exc_info=True)

if __name__ == "__main__":
    DATA_DIR = os.path.join(PROJECT_ROOT, "data", "historical_csv")
    OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "features.parquet")
    run_primary_feature_etl(data_dir=DATA_DIR, output_path=OUTPUT_PATH)