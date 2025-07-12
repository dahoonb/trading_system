# etl/fundamental_feature_etl.py

import pandas as pd
import numpy as np
import yfinance as yf
import os
import sys
import time
import random
from datetime import datetime, UTC
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import hashlib, time, random, logging
from requests.exceptions import HTTPError, ConnectionError as ReqConnectionError

# --- Project Root Setup ---
# This ensures the script can find other modules in the project, like the logger.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.logger import setup_logger

logger = setup_logger(logger_name="Fundamental_Feature_ETL")

def get_sp500_tickers() -> list[str]:
    """
    Fetches the current list of S&P 500 tickers from Wikipedia.
    This defines the "Alpha Universe" for model training.
    """
    try:
        # This requires 'lxml' to be installed: pip install lxml
        payload = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        tickers = payload[0]['Symbol'].tolist()
        # Clean up tickers that might have different representations in yfinance (e.g., BRK.B -> BRK-B)
        tickers = [t.replace('.', '-') for t in tickers]
        logger.info(f"Successfully fetched {len(tickers)} S&P 500 tickers for the Alpha Universe.")
        return tickers
    except Exception as e:
        logger.error(f"Could not fetch S&P 500 tickers: {e}. Returning empty list.")
        return []

def _safe_get(df: pd.DataFrame, key: str, year: pd.Timestamp, default=0.0) -> float:
    """Safely get a value from a financial statement DataFrame."""
    if key in df.index and year in df.columns:
        val = df.loc[key, year]
        return val if pd.notna(val) else default
    return default

def calculate_piotroski_f_score(ticker: yf.Ticker) -> tuple[int, dict]:
    """
    Calculates the Piotroski F-Score for a given stock.
    Returns the score (0-9) and a dictionary of the component values.
    FIX: Made robust to handle missing data points in financial statements.
    """
    try:
        income_stmt = ticker.financials
        balance_sheet = ticker.balance_sheet
        cash_flow = ticker.cashflow

        if income_stmt.empty or balance_sheet.empty or cash_flow.empty or len(income_stmt.columns) < 2:
            return 0, {}

        year1 = income_stmt.columns[0]
        year0 = income_stmt.columns[1]
        year_minus_1 = income_stmt.columns[2] if len(income_stmt.columns) > 2 else year0

        # --- Profitability Criteria (4 points) ---
        ni_y1 = _safe_get(income_stmt, 'Net Income', year1)
        assets_y0 = _safe_get(balance_sheet, 'Total Assets', year0)
        roa_y1 = ni_y1 / assets_y0 if assets_y0 != 0 else 0
        f_roa = 1 if roa_y1 > 0 else 0

        cfo_y1 = _safe_get(cash_flow, 'Total Cash From Operating Activities', year1)
        f_cfo = 1 if cfo_y1 > 0 else 0

        ni_y0 = _safe_get(income_stmt, 'Net Income', year0)
        assets_y_minus_1 = _safe_get(balance_sheet, 'Total Assets', year_minus_1)
        roa_y0 = ni_y0 / assets_y_minus_1 if assets_y_minus_1 != 0 else 0
        f_delta_roa = 1 if roa_y1 > roa_y0 else 0
        
        f_accruals = 1 if cfo_y1 > ni_y1 else 0

        # --- Leverage, Liquidity, and Source of Funds Criteria (3 points) ---
        long_term_debt_y1 = _safe_get(balance_sheet, 'Long Term Debt', year1)
        leverage_y1 = long_term_debt_y1 / assets_y0 if assets_y0 != 0 else 0
        
        long_term_debt_y0 = _safe_get(balance_sheet, 'Long Term Debt', year0)
        leverage_y0 = long_term_debt_y0 / assets_y_minus_1 if assets_y_minus_1 != 0 else 0
        f_delta_leverage = 1 if leverage_y1 < leverage_y0 else 0

        current_assets_y1 = _safe_get(balance_sheet, 'Total Current Assets', year1)
        current_liabilities_y1 = _safe_get(balance_sheet, 'Total Current Liabilities', year1)
        current_ratio_y1 = current_assets_y1 / current_liabilities_y1 if current_liabilities_y1 != 0 else 0
        
        current_assets_y0 = _safe_get(balance_sheet, 'Total Current Assets', year0)
        current_liabilities_y0 = _safe_get(balance_sheet, 'Total Current Liabilities', year0)
        current_ratio_y0 = current_assets_y0 / current_liabilities_y0 if current_liabilities_y0 != 0 else 0
        f_delta_liquidity = 1 if current_ratio_y1 > current_ratio_y0 else 0
        
        shares_y1 = _safe_get(balance_sheet, 'Share Issued', year1, default=None) or _safe_get(balance_sheet, 'Common Stock', year1)
        shares_y0 = _safe_get(balance_sheet, 'Share Issued', year0, default=None) or _safe_get(balance_sheet, 'Common Stock', year0)
        f_eq_offer = 1 if shares_y1 <= shares_y0 else 0

        # --- Operating Efficiency Criteria (2 points) ---
        gross_profit_y1 = _safe_get(income_stmt, 'Gross Profit', year1)
        total_revenue_y1 = _safe_get(income_stmt, 'Total Revenue', year1)
        gross_margin_y1 = gross_profit_y1 / total_revenue_y1 if total_revenue_y1 != 0 else 0
        
        gross_profit_y0 = _safe_get(income_stmt, 'Gross Profit', year0)
        total_revenue_y0 = _safe_get(income_stmt, 'Total Revenue', year0)
        gross_margin_y0 = gross_profit_y0 / total_revenue_y0 if total_revenue_y0 != 0 else 0
        f_delta_margin = 1 if gross_margin_y1 > gross_margin_y0 else 0

        asset_turnover_y1 = total_revenue_y1 / assets_y0 if assets_y0 != 0 else 0
        asset_turnover_y0 = total_revenue_y0 / assets_y_minus_1 if assets_y_minus_1 != 0 else 0
        f_delta_turnover = 1 if asset_turnover_y1 > asset_turnover_y0 else 0

        f_score = f_roa + f_cfo + f_delta_roa + f_accruals + f_delta_leverage + f_delta_liquidity + f_eq_offer + f_delta_margin + f_delta_turnover
        
        components = {
            'roa_y1': roa_y1, 'cfo_y1': cfo_y1, 'delta_roa': roa_y1 - roa_y0, 'accruals': cfo_y1 - ni_y1
        }
        return f_score, components
    except Exception as e:
        # This broad except is a final fallback, but the _safe_get should prevent most errors.
        logger.debug(f"Could not calculate F-Score for ticker due to an unexpected error: {e}")
        return 0, {}

def get_shareholder_yield(ticker: yf.Ticker) -> tuple[float, float]:
    """Calculates dividend yield and buyback yield."""
    try:
        info = ticker.info
        div_yield = info.get('dividendYield', 0.0) or 0.0

        cash_flow = ticker.cashflow
        if cash_flow.empty or len(cash_flow.columns) < 1:
            return div_yield, 0.0

        year1 = cash_flow.columns[0]
        repurchase = _safe_get(cash_flow, 'Repurchase Of Stock', year1)
        issuance = _safe_get(cash_flow, 'Issuance Of Stock', year1)
        
        net_buyback = abs(repurchase) - abs(issuance)
        market_cap = info.get('marketCap', 0)
        
        buyback_yield = net_buyback / market_cap if market_cap > 0 else 0.0
        
        return div_yield, buyback_yield
    except Exception:
        return 0.0, 0.0

# ----------------------------------------------------------------------
# Stable 64-bit hash helper. 8-byte digest → signed int64
def stable_id(symbol: str) -> int:
    h = hashlib.blake2b(symbol.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big", signed=True)
# ----------------------------------------------------------------------

def process_ticker(ticker_symbol: str, max_retries: int = 3, initial_delay: float = 1.0) -> dict | None:
    """
    Fetches all fundamental data for a single ticker, relying on yfinance's
    internal session management and implementing a robust retry mechanism.
    """
    for attempt in range(max_retries):
        try:
            ticker = yf.Ticker(ticker_symbol)
            
            if ticker.history(period="1d").empty:
                 raise ConnectionError(f"No recent history for {ticker_symbol}, likely an invalid ticker.")

            info = ticker.info
            if not info or info.get('regularMarketPrice') is None:
                raise ConnectionError(f"Invalid or empty info dict for {ticker_symbol}")

            pb_ratio = info.get('priceToBook')
            pe_ratio = info.get('trailingPE')
            
            f_score, f_components = calculate_piotroski_f_score(ticker)
            div_yield, buyback_yield = get_shareholder_yield(ticker)
            
            if pb_ratio is None and pe_ratio is None:
                logger.warning(f"Skipping {ticker_symbol} due to missing P/B and P/E ratios.")
                return None

            return {
                "ticker_id": stable_id(ticker_symbol),
                'price_to_book': pb_ratio,
                'price_to_earnings': pe_ratio,
                'dividend_yield': div_yield,
                'buyback_yield': buyback_yield,
                'shareholder_yield': div_yield + buyback_yield,
                'piotroski_f_score': f_score,
                'roa': f_components.get('roa_y1'),
                'cfo': f_components.get('cfo_y1'),
            }
        except Exception as e:
            if attempt < max_retries - 1:
                delay = initial_delay * (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"Error processing {ticker_symbol}: {e}. Retrying in {delay:.1f}s...")
                time.sleep(delay)
                continue
            else:
                logger.error(f"Failed to process ticker {ticker_symbol} after {max_retries} attempts: {e}")
                return None
    return None

def run_fundamental_feature_etl(output_path: str):
    """Main ETL function to fetch, process, and save fundamental features."""
    logger.info("--- Starting Fundamental Feature ETL Process ---")
    
    tickers = get_sp500_tickers()
    if not tickers:
        logger.critical("No tickers found. Aborting ETL.")
        return

    all_features = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_ticker = {executor.submit(process_ticker, t): t for t in tickers}
        for future in tqdm(as_completed(future_to_ticker), total=len(tickers), desc="Fetching Fundamental Data"):
            result = future.result()
            if result:
                all_features.append(result)
    
    if not all_features:
        logger.error("No fundamental features were generated. Aborting.")
        return

    df = pd.DataFrame(all_features)
    
    # Add timezone-aware UTC timestamps for Feast compatibility.
    ts_now = datetime.now(UTC)
    df["event_timestamp"] = ts_now
    df["created_timestamp"] = ts_now
    
    # Clean and type data for consistency
    df.dropna(subset=['ticker_id'], inplace=True)
    df["ticker_id"] = df["ticker_id"].astype("int64")
    
    numeric_cols = [
        'price_to_book', 'price_to_earnings', 'dividend_yield', 'buyback_yield',
        'shareholder_yield', 'piotroski_f_score', 'roa', 'cfo'
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Ensure key metrics required by strategies are present
    df.dropna(subset=['price_to_book', 'piotroski_f_score'], how='any', inplace=True)
    
    logger.info(f"Successfully processed and saving data for {len(df)} tickers.")

    # Save to Parquet
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_parquet(
            output_path,
            index=False,
            engine='pyarrow',
            use_dictionary=False,
            compression='snappy',
        )
        logger.info(f"Successfully saved {len(df)} fundamental features to '{output_path}'.")
    except Exception as e:
        logger.error(f"Failed to save fundamental features to Parquet file: {e}", exc_info=True)

if __name__ == "__main__":
    # Before running, ensure you have the necessary libraries:
    # pip install pandas yfinance tqdm lxml pyarrow
    
    OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "fundamental_features.parquet")
    run_fundamental_feature_etl(output_path=OUTPUT_PATH)