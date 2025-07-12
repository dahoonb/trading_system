# equity_trading_system/etl/fetch_daily_bars.py
import logging
import os
import sys
import datetime
import threading
from typing import Dict, List, Any, Optional
import pandas as pd
from prefect import task, get_run_logger

# --- Project Root Setup ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.ib_wrapper import IBWrapper
from utils.logger import setup_logger

class _HistoricalDataHandler:
    """A simple, local handler class for a single data download task."""
    def __init__(self):
        self.data: Dict[int, List[Any]] = {}
        self.completion_events: Dict[int, threading.Event] = {}
        self.lock = threading.Lock()

    def on_historical_data(self, reqId: int, bar):
        with self.lock:
            if reqId not in self.data: self.data[reqId] = []
            self.data[reqId].append({
                "date": bar.date, "open": bar.open, "high": bar.high,
                "low": bar.low, "close": bar.close, "volume": bar.volume
            })

    def on_historical_data_end(self, reqId: int, start: str, end: str):
        if reqId in self.completion_events:
            self.completion_events[reqId].set()

    def get_dataframe(self, reqId: int) -> pd.DataFrame:
        with self.lock:
            return pd.DataFrame(self.data.get(reqId, []))

@task(name="Fetch Daily IBKR Bars Chunk", retries=2, retry_delay_seconds=120)
async def fetch_daily_bars_task(
    ib_wrapper: IBWrapper,
    symbols: List[str],
    duration: str,
    end_date_time_str: str # This is now used for logging purposes only
) -> Dict[str, pd.DataFrame]:
    """
    A Prefect task to fetch a single chunk of daily historical bars from IBKR.
    It now passes an empty string for endDateTime to support ADJUSTED_LAST data.
    """
    try:
        logger = get_run_logger()
    except Exception:
        logger = setup_logger(logger_name="FetchDailyBarsTaskStandalone")

    fetched_data: Dict[str, pd.DataFrame] = {}
    handler = _HistoricalDataHandler()
    
    ib_wrapper.add_callback("historicalData", handler.on_historical_data)
    ib_wrapper.add_callback("historicalDataEnd", handler.on_historical_data_end)

    logger.info(f"Requesting chunk for {len(symbols)} with duration '{duration}' (End date is implicit for adjusted data).")

    request_ids = {}
    for symbol in symbols:
        contract = create_stock_contract(symbol)
        req_id = ib_wrapper.get_next_req_id()
        
        handler.completion_events[req_id] = threading.Event()
        request_ids[symbol] = req_id
        
        # --- FIX: Set endDateTime to "" as required by ADJUSTED_LAST ---
        ib_wrapper.reqHistoricalData(
            reqId=req_id,
            contract=contract,
            endDateTime="", # CRITICAL CHANGE: Must be empty for adjusted data.
            durationStr=duration,
            barSizeSetting="1 day",
            whatToShow="ADJUSTED_LAST",
            useRTH=1,
            formatDate=1,
            keepUpToDate=False,
            chartOptions=[]
        )

    for symbol in symbols:
        req_id = request_ids[symbol]
        completed = handler.completion_events[req_id].wait(timeout=90.0)
        
        if not completed:
            logger.warning(f"Timed out waiting for historical data for {symbol} (ReqId: {req_id}). Cancelling request.")
            ib_wrapper.cancelHistoricalData(req_id)
            continue

        df = handler.get_dataframe(req_id)
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df = df.tz_localize('UTC')
            fetched_data[symbol] = df
            logger.info(f"Successfully fetched {len(df)} adjusted bars for {symbol}.")
        else:
            logger.warning(f"No data returned for symbol: {symbol}.")

    ib_wrapper.unregister_listener("historicalData", handler.on_historical_data)
    ib_wrapper.unregister_listener("historicalDataEnd", handler.on_historical_data_end)
            
    return fetched_data

def create_stock_contract(symbol: str):
    """Helper function to create a standard stock contract."""
    from ibapi.contract import Contract
    contract = Contract()
    contract.symbol = symbol
    contract.secType = "STK"
    contract.exchange = "SMART"
    contract.currency = "USD"
    return contract