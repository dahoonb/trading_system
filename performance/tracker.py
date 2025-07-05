# performance/tracker.py

import pandas as pd
import numpy as np
import datetime
import math
import os
import sqlite3
from typing import Optional, Dict, List, Tuple, Any 
import logging

# Use the central config loader to make this module configurable
from core.config_loader import config

# Use a dedicated logger for this module for better log filtering
logger = logging.getLogger("PerformanceTracker")

class TCALogger:
    """
    Logs detailed trade execution records to a persistent SQLite database.
    This data is essential for post-trade analysis and for providing the
    learning data for the adaptive AlgoWheel.
    """
    def __init__(self, logger_instance: Optional[logging.Logger] = None):
        """
        Initializes the TCALogger. The database path is now loaded from the
        central configuration file.
        """
        # Get DB path from the central config, with a sensible default
        self.db_path = config.get('results', {}).get('tca_log_db_path', 'results/tca_log.db')
        self.conn: Optional[sqlite3.Connection] = None
        self.cursor: Optional[sqlite3.Cursor] = None
        
        self.logger = logger_instance or logger

        if self.db_path:
            try:
                db_dir = os.path.dirname(self.db_path)
                if db_dir: os.makedirs(db_dir, exist_ok=True)
                
                # Use check_same_thread=False for multi-threaded access if needed,
                # though a single writer is typical.
                self.conn = sqlite3.connect(self.db_path, check_same_thread=False) 
                self.cursor = self.conn.cursor()
                self._create_tca_tables()
                self.logger.info(f"TCALogger initialized. Database at: {self.db_path}")
            except Exception as e:
                self.logger.error(f"TCALogger failed to initialize database at '{self.db_path}': {e}", exc_info=True)
                self.conn = None
                self.cursor = None
        else:
            self.logger.warning("TCALogger: No database path provided in config. TCA logging is disabled.")

    def _create_tca_tables(self):
        """Ensures the necessary tables for TCA logging exist in the database."""
        if not self.cursor or not self.conn: return
        try:
            # Table for successfully filled trades
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS fills (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    exchange TEXT,
                    quantity REAL NOT NULL,
                    direction TEXT NOT NULL,
                    fill_price REAL NOT NULL,
                    commission REAL,
                    fill_cost REAL,
                    submit_timestamp TEXT,
                    order_method TEXT,
                    order_status TEXT,
                    broker_order_id TEXT,
                    execution_id TEXT UNIQUE,
                    decision_ref_price REAL,
                    decision_ref_spread REAL,
                    internal_order_ref TEXT,
                    strategy_id TEXT,
                    realized_pnl REAL,
                    holding_period_days REAL,
                    tca_slippage_bps REAL
                )""")
            
            # Table for orders that did not fill (cancelled, expired, etc.)
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS unfilled_orders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    original_order_ref TEXT,
                    symbol TEXT NOT NULL,
                    attempted_quantity REAL NOT NULL,
                    unfilled_quantity REAL,
                    error_message TEXT,
                    reference_price_at_decision REAL,
                    original_direction TEXT
                )""")
            self.conn.commit()
        except sqlite3.Error as e:
            self.logger.error(f"TCALogger: Error creating TCA tables: {e}", exc_info=True)

    def log_fill_tca(self, fill_event, realized_pnl: Optional[float] = None, holding_period_days: Optional[float] = None):
        """Logs a single filled trade record to the database."""
        if not self.cursor or not self.conn or not fill_event: return
        
        slippage_bps = None
        if hasattr(fill_event, 'order_decision_reference_price') and fill_event.order_decision_reference_price > 0:
            price_diff = fill_event.average_price - fill_event.order_decision_reference_price
            if fill_event.direction.upper() == 'SELL':
                price_diff = -price_diff
            slippage_bps = (price_diff / fill_event.order_decision_reference_price) * 10000

        try:
            self.cursor.execute("""
                INSERT INTO fills (
                    timestamp, symbol, exchange, quantity, direction, fill_price, commission, fill_cost,
                    submit_timestamp, order_method, order_status, broker_order_id, execution_id,
                    decision_ref_price, decision_ref_spread, internal_order_ref, strategy_id,
                    realized_pnl, holding_period_days, tca_slippage_bps
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                fill_event.timestamp.isoformat(), fill_event.symbol, getattr(fill_event, 'exchange', None), 
                fill_event.quantity, fill_event.direction, fill_event.average_price, 
                getattr(fill_event, 'commission', None), getattr(fill_event, 'fill_cost', None),
                getattr(fill_event, 'submit_timestamp', None), getattr(fill_event, 'order_method', None), 
                getattr(fill_event, 'order_status', None), getattr(fill_event, 'order_id', None), 
                getattr(fill_event, 'exec_id', None), getattr(fill_event, 'order_decision_reference_price', None), 
                getattr(fill_event, 'order_decision_reference_spread', None), getattr(fill_event, 'order_id_ref_internal', None), 
                getattr(fill_event, 'strategy_id_orig_order', None), realized_pnl, holding_period_days, slippage_bps
            ))
            self.conn.commit()
        except sqlite3.Error as e:
            self.logger.error(f"TCALogger: Error logging fill to SQLite: {e}", exc_info=True)

    def log_unfilled_order_tca(self, unfilled_event):
        """Logs an unfilled or cancelled order to the database."""
        if not self.cursor or not self.conn or not unfilled_event: return
        try:
            self.cursor.execute("""
                INSERT INTO unfilled_orders (
                    timestamp, original_order_ref, symbol, attempted_quantity, 
                    unfilled_quantity, error_message, reference_price_at_decision, original_direction
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                unfilled_event.timestamp.isoformat(), getattr(unfilled_event, 'order_id_ref_original', None), 
                unfilled_event.symbol, getattr(unfilled_event, 'attempted_quantity', None),
                getattr(unfilled_event, 'unfilled_quantity', None), getattr(unfilled_event, 'error_msg', None),
                getattr(unfilled_event, 'reference_price_at_decision', None), getattr(unfilled_event, 'original_direction', None)
            ))
            self.conn.commit()
        except sqlite3.Error as e:
            self.logger.error(f"TCALogger: Error logging unfilled order to SQLite: {e}", exc_info=True)

    def close(self):
        """Closes the database connection gracefully."""
        if self.conn:
            try:
                self.conn.close()
                self.logger.info("TCALogger: SQLite connection closed.")
            except sqlite3.Error as e:
                self.logger.error(f"TCALogger: Error closing SQLite connection: {e}", exc_info=True)
        self.conn = None
        self.cursor = None

class PerformanceTracker:
    """
    Calculates and tracks portfolio performance metrics over time. This class
    is intended to be used for generating daily or periodic performance reports.

    Integration Plan:
    1.  An instance of this class should be created within `LivePortfolioManager`.
    2.  The `record_equity` method should be called by `LivePortfolioManager` at the end of every
        trading day (or more frequently) with the current timestamp and total portfolio value.
    3.  The `record_trade_pnl` method should be called by `LivePortfolioManager`
        after processing a `FillEvent` that closes a position, passing the realized P&L.
    4.  A new method in `LivePortfolioManager`, like `generate_daily_report`, should be created.
        This method would call `calculate_metrics()` and `save_results()` and should be
        triggered once per day from the main loop in `main.py`.
    """
    def __init__(self, initial_capital: float, logger_instance: Optional[logging.Logger] = None):
        self.logger = logger_instance or logger
        
        self.initial_capital = float(initial_capital)
        performance_config = config.get('performance', {})
        self.risk_free_rate = float(performance_config.get('risk_free_rate', 0.02))
        
        self.equity_curve: List[Tuple[datetime.datetime, float]] = [(datetime.datetime.now(), initial_capital)]
        self.peak_equity = float(initial_capital)
        
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_commissions = 0.0
        self.individual_realized_pnls: List[float] = []

        self.logger.info(f"PerformanceTracker initialized. Initial Capital: ${self.initial_capital:,.2f}")

    def record_equity(self, timestamp: datetime.datetime, equity_value: float):
        """Records a snapshot of the portfolio's equity at a given time."""
        if not isinstance(timestamp, datetime.datetime) or not isinstance(equity_value, (int, float)):
            return
        
        ts_utc = timestamp.astimezone(datetime.timezone.utc) if timestamp.tzinfo else timestamp
        self.equity_curve.append((ts_utc, equity_value))

        if equity_value > self.peak_equity:
            self.peak_equity = equity_value

    def record_trade_pnl(self, pnl: float, commission: float):
        """Records the outcome of a single closed trade."""
        self.total_trades += 1
        self.total_commissions += commission
        self.individual_realized_pnls.append(pnl)
        if pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1

    def calculate_metrics(self, benchmark_returns: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Calculates a comprehensive set of performance metrics."""
        if len(self.equity_curve) < 2:
            return self._empty_metrics_dict("Not enough equity data points.")

        equity_df = pd.DataFrame(self.equity_curve, columns=['timestamp', 'equity']).set_index('timestamp')
        final_equity = equity_df['equity'].iloc[-1]
        total_return_pct = (final_equity / self.initial_capital - 1) * 100
        
        duration_years = (equity_df.index[-1] - equity_df.index[0]).days / 365.25
        cagr = ((final_equity / self.initial_capital) ** (1 / duration_years) - 1) * 100 if duration_years > 0 else 0

        daily_returns = equity_df['equity'].resample('D').last().pct_change().dropna()
        volatility = daily_returns.std() * np.sqrt(252) * 100 if not daily_returns.empty else 0
        sharpe_ratio = ((cagr / 100) - self.risk_free_rate) / (volatility / 100) if volatility > 0 else 0

        peak = equity_df['equity'].expanding(min_periods=1).max()
        drawdown = (equity_df['equity'] - peak) / peak
        max_drawdown = drawdown.min() * 100

        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        
        total_profit = sum(p for p in self.individual_realized_pnls if p > 0)
        total_loss = abs(sum(p for p in self.individual_realized_pnls if p < 0))
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')

        metrics = {
            "Total Return (%)": f"{total_return_pct:.2f}",
            "CAGR (%)": f"{cagr:.2f}",
            "Annualized Volatility (%)": f"{volatility:.2f}",
            "Sharpe Ratio": f"{sharpe_ratio:.3f}",
            "Max Drawdown (%)": f"{max_drawdown:.2f}",
            "Win Rate (%)": f"{win_rate:.2f}",
            "Profit Factor": f"{profit_factor:.2f}",
            "Total Trades": self.total_trades,
            "Total Commissions ($)": f"${self.total_commissions:,.2f}"
        }
        return metrics

    def _empty_metrics_dict(self, reason: str) -> Dict[str, Any]:
        """Returns a default dictionary when metrics cannot be calculated."""
        return {"Error": reason, "Total Trades": self.total_trades}

    def save_results(self, output_dir: Optional[str] = None, filename_suffix: Optional[str] = ""):
        """Saves the performance summary and equity curve to files."""
        if output_dir is None:
            output_dir = config.get('logging', {}).get('results_directory', 'results')
        os.makedirs(output_dir, exist_ok=True)

        metrics = self.calculate_metrics()
        
        summary_filename = os.path.join(output_dir, f"performance_summary{filename_suffix}.txt")
        with open(summary_filename, 'w') as f:
            f.write(f"Performance Report - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*40 + "\n")
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")
        self.logger.info(f"Performance summary saved to: {summary_filename}")

        equity_df = pd.DataFrame(self.equity_curve, columns=['timestamp', 'equity']).set_index('timestamp')
        equity_filename = os.path.join(output_dir, f"equity_curve{filename_suffix}.csv")
        equity_df.to_csv(equity_filename)
        self.logger.info(f"Equity curve saved to: {equity_filename}")