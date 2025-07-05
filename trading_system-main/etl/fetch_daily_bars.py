# equity_trading_system/etl/fetch_daily_bars.py
import logging
import os
import sys
import datetime
import time
import queue
from typing import Dict, List, Optional, Any
import pandas as pd
from prefect import task, get_run_logger
import threading

# --- Project Root Setup ---
# This remains important for ensuring modules can be found.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.ib_wrapper import IBWrapper
from utils.logger import setup_logger

# This is a local handler class specifically for this data download task.
# It encapsulates the state and logic needed to fetch historical data.
class HistoricalDataHandler:
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.data: Dict[str, List[Any]] = {symbol: [] for symbol in symbols}
        self.requests: Dict[int, str] = {}
        self.completion_events: Dict[int, threading.Event] = {}

    def on_historical_data(self, reqId: int, bar):
        """Callback to receive a single historical data bar."""
        symbol = self.requests.get(reqId)
        if symbol:
            self.data[symbol].append({
                "date": bar.date, "open": bar.open, "high": bar.high,
                "low": bar.low, "close": bar.close, "volume": bar.volume
            })

    def on_historical_data_end(self, reqId: int, start: str, end: str):
        """Callback to signal the end of a historical data request."""
        if reqId in self.completion_events:
            self.completion_events[reqId].set()

    def get_dataframe(self, symbol: str) -> pd.DataFrame:
        """Converts the collected list of bar data into a pandas DataFrame."""
        return pd.DataFrame(self.data.get(symbol, []))

@task(name="Fetch Daily IBKR Bars", retries=2, retry_delay_seconds=120)
async def fetch_daily_bars_task(
    ib_config: Dict[str, Any],
    symbols: List[str],
    duration: str = '1 D'
) -> Dict[str, pd.DataFrame]:
    """
    A Prefect task to fetch daily historical bars from Interactive Brokers.
    This task is now self-contained and uses the core IBWrapper correctly.
    """
    try:
        logger = get_run_logger()
    except Exception:
        logger = setup_logger(logger_name="FetchDailyBarsTaskStandalone")

    fetched_data: Dict[str, pd.DataFrame] = {}
    ib_wrapper = None
    
    try:
        host = ib_config.get('host', '127.0.0.1')
        port = ib_config.get('port', 7497)
        client_id = ib_config.get('etl_feature_pipeline_client_id', 9903)
        
        if not symbols:
            logger.warning("No symbols provided to fetch.")
            return {}

        logger.info(f"Attempting to connect to IBKR at {host}:{port} with Client ID {client_id}.")
        
        # Use the robust IBWrapper for the connection
        ib_wrapper = IBWrapper()
        
        # Instantiate our dedicated handler for this task
        handler = HistoricalDataHandler(symbols)
        
        # Register the handler's methods to listen for events from the wrapper
        ib_wrapper.add_callback("historicalData", handler.on_historical_data)
        ib_wrapper.add_callback("historicalDataEnd", handler.on_historical_data_end)

        # Start the connection
        ib_wrapper.start_connection(host, port, client_id)
        if not ib_wrapper.isConnected():
            raise ConnectionError("Failed to connect to IBKR for daily bar fetch.")

        end_date_time_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d %H:%M:%S %Z")
        logger.info(f"Requesting historical data for {len(symbols)} symbols with duration '{duration}' ending {end_date_time_str}.")

        for symbol in symbols:
            contract = ib_wrapper.create_stock_contract(symbol) # Helper method to create contracts
            req_id = ib_wrapper.get_next_req_id()
            
            handler.requests[req_id] = symbol
            handler.completion_events[req_id] = threading.Event()
            
            ib_wrapper.reqHistoricalData(
                reqId=req_id,
                contract=contract,
                endDateTime=end_date_time_str,
                durationStr=duration,
                barSizeSetting="1 day",
                whatToShow="TRADES",
                useRTH=1,
                formatDate=1,
                keepUpToDate=False,
                chartOptions=[]
            )
            
            # Wait for this specific request to complete
            completed = handler.completion_events[req_id].wait(timeout=90.0)
            if not completed:
                logger.warning(f"Timed out waiting for historical data for {symbol}.")
                continue

            df = handler.get_dataframe(symbol)
            if df is not None and not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                fetched_data[symbol] = df
                logger.info(f"Successfully fetched {len(df)} bars for {symbol}.")
            else:
                logger.warning(f"No data returned for symbol: {symbol}")

    except Exception as e:
        logger.error(f"An error occurred during the fetch_daily_bars_task: {e}", exc_info=True)
        raise
    finally:
        if ib_wrapper and ib_wrapper.isConnected():
            ib_wrapper.stop_connection()
            logger.info("IBKR connection closed.")
            
    return fetched_data

# Helper method needs to be added to IBWrapper for this to be fully clean,
# but for a quick fix, we can define it here or add it to the wrapper class.
# Let's assume it exists on the wrapper for this example.
def create_stock_contract(self, symbol: str):
    from ibapi.contract import Contract
    contract = Contract()
    contract.symbol = symbol
    contract.secType = "STK"
    contract.exchange = "SMART"
    contract.currency = "USD"
    return contract

IBWrapper.create_stock_contract = create_stock_contract