# get_historical_data.py

import argparse
import asyncio
import os
import logging
import sys

# --- Project Root Setup ---
# This ensures the script can find all the necessary modules
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.config_loader import config
from etl.fetch_daily_bars import fetch_daily_bars_task
from utils.logger import setup_logger

# Setup a logger for this script
logger = setup_logger(logger_name="HistoricalDataDownloader", log_level=logging.INFO)

async def main_async(duration: str, symbols: list, output_dir: str):
    """
    Asynchronous main function to orchestrate the data fetching and saving.
    """
    logger.info("--- Starting Historical Data Download ---")
    
    # Get IB connection details from the global config
    ib_config = config.get('ib_connection', {})
    if not ib_config:
        logger.critical("IB connection settings not found in config.yaml. Aborting.")
        return

    try:
        # Await the Prefect task directly. It's just an async function.
        fetched_data = await fetch_daily_bars_task.fn(
            ib_config=ib_config,
            symbols=symbols,
            duration=duration
        )
    except ConnectionError as e:
        logger.critical(f"Failed to connect to Interactive Brokers: {e}")
        logger.critical("Please ensure TWS or IB Gateway is running and you are logged in.")
        return
    except Exception as e:
        logger.error(f"An unexpected error occurred during data fetching: {e}", exc_info=True)
        return

    if not fetched_data:
        logger.warning("No data was fetched. Please check symbols and connection.")
        return

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Saving historical data to directory: '{output_dir}'")

    for symbol, df in fetched_data.items():
        if df.empty:
            logger.warning(f"No data returned for {symbol}. Skipping.")
            continue

        output_path = os.path.join(output_dir, f"{symbol}.csv")
        
        # --- Standardize the DataFrame format ---
        # The rest of the system expects lowercase column names and a 'date' column.
        df.reset_index(inplace=True)
        df.rename(columns={
            'index': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }, inplace=True, errors='ignore') # Use errors='ignore' in case columns are already lowercase
        
        # Ensure the required columns exist
        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        df = df[[col for col in required_cols if col in df.columns]]

        # Save to CSV
        df.to_csv(output_path, index=False)
        logger.info(f"Successfully saved {len(df)} rows for {symbol} to {output_path}")

    logger.info("--- Historical Data Download Complete ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download historical daily bar data from Interactive Brokers.")
    
    parser.add_argument(
        "--duration",
        type=str,
        default="3 Y",
        help="The duration of historical data to fetch (e.g., '1 D', '5 Y'). Default: '3 Y'."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/historical_csv",
        help="The directory to save the output CSV files. Default: 'data/historical_csv'."
    )
    parser.add_argument(
        "--symbols",
        nargs='+',
        default=None,
        help="A space-separated list of symbols to fetch. Overrides symbols from config.yaml."
    )

    args = parser.parse_args()

    # Determine which symbols to use
    symbols_to_fetch = args.symbols if args.symbols else config.get('symbols', [])
    if not symbols_to_fetch:
        logger.critical("No symbols specified via command line or in config.yaml. Aborting.")
        sys.exit(1)

    # Run the asynchronous main function
    asyncio.run(main_async(args.duration, symbols_to_fetch, args.output_dir))