# tca/calculator.py
"""
Nightly or Ad-Hoc TCA Calculation Module.

Connects to the TCA log database, queries trades for a specific period,
calculates standard TCA metrics (Implementation Shortfall, Fill Delay,
Opportunity Cost for filled and unfilled quantities), and outputs the results.
This script is intended for offline analysis and reporting.
"""

import sys
import duckdb
import pandas as pd
import datetime
import logging
import argparse
import os
from typing import Optional, Dict, Any, List
import math 

# Use a dedicated logger for this script
logger = logging.getLogger("TCACalculator")
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("TCACalculator logger initialized with basicConfig.")

# Default paths, can be overridden by command-line arguments
TCA_DB_PATH_DEFAULT = "data/tca_log.duckdb"
HISTORICAL_DATA_DIR_DEFAULT = "data/historical_csv"

def connect_db(db_path: str, read_only: bool = True) -> Optional[duckdb.DuckDBPyConnection]:
    """Connects to the DuckDB database."""
    try:
        conn = duckdb.connect(database=db_path, read_only=read_only)
        logger.info(f"Connected to DuckDB database: '{db_path}' (Read-Only: {read_only})")
        return conn
    except ImportError:
        logger.error("DuckDB library not found. Please install it using 'pip install duckdb'.")
        return None
    except Exception as e:
        logger.error(f"Failed to connect to DuckDB '{db_path}': {e}", exc_info=True)
        return None

def calculate_implementation_shortfall(trade_data: pd.Series) -> Optional[float]:
    """Calculates implementation shortfall in basis points (bps)."""
    fill_price = pd.to_numeric(trade_data.get('fill_price'), errors='coerce')
    ref_price = pd.to_numeric(trade_data.get('benchmark_price'), errors='coerce')
    direction = trade_data.get('direction', '').upper()

    if pd.isna(fill_price) or pd.isna(ref_price) or ref_price <= 0:
        return None
    
    # Slippage is positive if it's an adverse move (buy higher, sell lower)
    slippage_pct = (fill_price - ref_price) / ref_price
    cost_factor = 1.0 if direction == 'BUY' else -1.0
    
    shortfall_bps = slippage_pct * cost_factor * 10000
    return shortfall_bps if not (math.isnan(shortfall_bps) or math.isinf(shortfall_bps)) else None

def _fetch_eod_close_from_csv(symbol: str, target_date: datetime.date, hist_data_path: str) -> Optional[float]:
    """Fetches the end-of-day closing price for a symbol on a specific date."""
    csv_file = os.path.join(hist_data_path, f"{symbol}.csv")
    if not os.path.exists(csv_file):
        logger.warning(f"EOD Fetch: File not found for {symbol} at {csv_file}")
        return None
    try:
        hist_df = pd.read_csv(csv_file, index_col='date', parse_dates=True, usecols=['date', 'close'])
        matching_rows = hist_df[hist_df.index.date == target_date]
        if matching_rows.empty:
            return None
        return float(matching_rows['close'].iloc[0])
    except Exception as e:
        logger.error(f"EOD Fetch: Error reading CSV for {symbol}: {e}")
        return None

def calculate_opportunity_cost(trade_data: pd.Series, hist_data_path: str) -> Optional[float]:
    """Calculates opportunity cost for missed trades."""
    symbol = trade_data.get('symbol')
    unfilled_qty = pd.to_numeric(trade_data.get('quantity'), errors='coerce')
    direction = trade_data.get('direction', '').upper()
    limit_price = pd.to_numeric(trade_data.get('limit_price'), errors='coerce')
    event_ts = pd.to_datetime(trade_data.get('timestamp'), errors='coerce', utc=True)

    if pd.isna(unfilled_qty) or pd.isna(limit_price) or pd.isna(event_ts):
        return None

    target_date = event_ts.date()
    eod_price = _fetch_eod_close_from_csv(symbol, target_date, hist_data_path)
    if eod_price is None:
        return None
    
    price_difference = eod_price - limit_price
    cost_factor = 1.0 if direction == 'BUY' else -1.0
    
    return unfilled_qty * price_difference * cost_factor

def generate_daily_tca_report(db_conn: duckdb.DuckDBPyConnection, calculation_date: datetime.date, hist_data_path: str) -> Optional[pd.DataFrame]:
    """Queries logs and calculates a comprehensive TCA report for a given day."""
    start_ts = datetime.datetime.combine(calculation_date, datetime.time.min)
    end_ts = datetime.datetime.combine(calculation_date, datetime.time.max)
    logger.info(f"Generating TCA report for date: {calculation_date.isoformat()}")

    try:
        fills_df = db_conn.execute("SELECT * FROM execution_log WHERE timestamp BETWEEN ? AND ?", [start_ts, end_ts]).fetchdf()
        misses_df = db_conn.execute("SELECT * FROM missed_trades WHERE timestamp BETWEEN ? AND ?", [start_ts, end_ts]).fetchdf()
    except duckdb.Error as e:
        logger.error(f"Database query failed: {e}")
        return None

    report_parts = []
    if not fills_df.empty:
        fills_df['tca_metric'] = 'Implementation Shortfall (bps)'
        fills_df['value'] = fills_df.apply(calculate_implementation_shortfall, axis=1)
        report_parts.append(fills_df[['timestamp', 'symbol', 'algo_used', 'quantity', 'tca_metric', 'value']])

    if not misses_df.empty:
        misses_df['tca_metric'] = 'Opportunity Cost ($)'
        misses_df['value'] = misses_df.apply(lambda row: calculate_opportunity_cost(row, hist_data_path), axis=1)
        report_parts.append(misses_df[['timestamp', 'symbol', 'algo_used', 'quantity', 'tca_metric', 'value']])

    if not report_parts:
        logger.info(f"No trades found for {calculation_date.isoformat()}.")
        return pd.DataFrame()

    full_report = pd.concat(report_parts, ignore_index=True).dropna(subset=['value'])
    return full_report

def main():
    parser = argparse.ArgumentParser(description="Calculate daily TCA metrics from DuckDB logs.")
    parser.add_argument("--date", type=str, default=(datetime.date.today() - datetime.timedelta(days=1)).isoformat(), help="Date (YYYY-MM-DD) for the report. Defaults to yesterday.")
    parser.add_argument("--db-path", type=str, default=TCA_DB_PATH_DEFAULT, help=f"Path to the TCA DuckDB file.")
    parser.add_argument("--hist-path", type=str, default=HISTORICAL_DATA_DIR_DEFAULT, help=f"Path to the directory with historical CSV data.")
    parser.add_argument("--output-csv", type=str, default=None, help="Optional path to save the report as a CSV file.")
    args = parser.parse_args()

    try:
        calc_date = datetime.date.fromisoformat(args.date)
    except ValueError:
        logger.error(f"Invalid date format: '{args.date}'. Please use YYYY-MM-DD."); sys.exit(1)

    db_connection = connect_db(args.db_path)
    if not db_connection:
        sys.exit(1)

    report_df = generate_daily_tca_report(db_connection, calc_date, args.hist_path)
    db_connection.close()

    if report_df is not None and not report_df.empty:
        print("\n" + "="*25 + f" TCA Report for {calc_date.isoformat()} " + "="*25)
        print(report_df.to_string(index=False))
        print("="*75)
        if args.output_csv:
            try:
                output_dir = os.path.dirname(args.output_csv)
                if output_dir: os.makedirs(output_dir, exist_ok=True)
                report_df.to_csv(args.output_csv, index=False, float_format='%.4f')
                logger.info(f"TCA report saved to: {os.path.abspath(args.output_csv)}")
            except Exception as e:
                logger.error(f"Failed to save TCA report to CSV '{args.output_csv}': {e}")
    else:
        logger.info("TCA report generation finished with no data to display.")

if __name__ == "__main__":
    logger.info("--- Starting TCA Calculator/Reporting Script (Standalone) ---")
    main()
    logger.info("--- TCA Calculator/Reporting Script Finished ---")