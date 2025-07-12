# get_historical_data.py

import argparse
import os
import logging
import sys
import pandas as pd
import yfinance as yf
from tqdm import tqdm
from datetime import datetime
import time

# --- Project Root Setup ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.config_loader import config
from utils.logger import setup_logger

logger = setup_logger(logger_name="HistoricalDataDownloader", log_level=logging.INFO)

def get_sp500_tickers() -> list[str]:
    """Fetches the current list of S&P 500 tickers from Wikipedia."""
    try:
        logger.info("Fetching S&P 500 ticker list from Wikipedia...")
        payload = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        tickers = payload[0]['Symbol'].tolist()
        tickers = [t.replace('.', '-') for t in tickers]
        logger.info(f"Successfully fetched {len(tickers)} S&P 500 tickers.")
        return tickers
    except Exception as e:
        logger.error(f"Could not fetch S&P 500 tickers: {e}. Returning empty list.")
        return []

def download_and_save_data(start_date: str, end_date: str, symbols: list, output_dir: str):
    """
    Downloads and saves historical stock data ONE TICKER AT A TIME to ensure
    maximum reliability and correct CSV formatting.
    """
    logger.info(f"--- Starting Historical Data Download for {len(symbols)} symbols (one by one) ---")
    logger.info(f"Period: {start_date} to {end_date}")

    os.makedirs(output_dir, exist_ok=True)
    successful_downloads = 0

    for symbol in tqdm(symbols, desc="Downloading Tickers"):
        try:
            # yfinance returns a DataFrame with a MultiIndex on columns, even for one ticker.
            data = yf.download(
                tickers=symbol,
                start=start_date,
                end=end_date,
                auto_adjust=True,
                progress=False,
            )

            if data.empty:
                logger.warning(f"No data returned for {symbol}. It may be delisted or have no data for the period.")
                continue

            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            # --- Standard Saving Logic ---
            data.rename(columns={
                'Open': 'open', 'High': 'high', 'Low': 'low',
                'Close': 'close', 'Volume': 'volume'
            }, inplace=True)
            
            data.index.name = 'date'
            output_path = os.path.join(output_dir, f"{symbol}.csv")
            data.reset_index(inplace=True)
            data.to_csv(output_path, index=False, date_format='%Y-%m-%d')
            successful_downloads += 1

        except Exception as e:
            logger.error(f"An error occurred while downloading {symbol}: {e}")
        
        time.sleep(0.1)

    logger.info(f"--- Historical Data Download Complete ---")
    logger.info(f"Successfully downloaded data for {successful_downloads} out of {len(symbols)} requested symbols.")
    logger.info(f"Files are located in '{output_dir}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download historical daily bar data using yfinance.")
    
    parser.add_argument("--start-date", type=str, default="2007-01-01", help="The start date (YYYY-MM-DD).")
    parser.add_argument("--end-date", type=str, default=datetime.now().strftime('%Y-%m-%d'), help="The end date (YYYY-MM-DD).")
    parser.add_argument("--output-dir", type=str, default="data/historical_csv", help="The output directory.")
    parser.add_argument("--universe", type=str, default="config", choices=['config', 'sp500'], help="The symbol universe to download.")
    parser.add_argument("--symbols", nargs='+', default=None, help="Custom list of symbols, overrides --universe.")

    args = parser.parse_args()

    symbols_to_fetch = []
    if args.symbols:
        logger.info(f"Using custom symbol list provided via --symbols argument.")
        symbols_to_fetch = args.symbols
    elif args.universe == 'sp500':
        logger.info("Selected 'sp500' universe.")
        symbols_to_fetch = get_sp500_tickers()
    else:
        logger.info("Selected 'config' universe. Using symbols from config.yaml.")
        symbols_to_fetch = config.get('symbols', [])

    if not symbols_to_fetch:
        logger.critical("No symbols specified or found. Aborting.")
        sys.exit(1)

    download_and_save_data(args.start_date, args.end_date, symbols_to_fetch, args.output_dir)