# backtester/engine.py

import queue
import pandas as pd
from datetime import datetime
import logging
import os
import duckdb
from typing import Dict, Any, Tuple, List

from core.events import MarketEvent, SignalEvent, OrderEvent, FillEvent
from portfolio.live_manager import LivePortfolioManager
from backtester.performance import calculate_performance_metrics
from core.config_loader import config as global_config
from strategy.base import BaseStrategy

logger = logging.getLogger("BacktestEngine")

class BacktestEngine:
    """
    A single-threaded, event-driven backtesting engine that simulates
    the entire integrated trading system.

    MODIFIED to backtest the new multi-factor portfolio which uses
    Hierarchical Risk Parity (HRP) for allocation. The strategies are
    now created and managed internally by the LivePortfolioManager.
    """
    def __init__(self,
                 historical_data: pd.DataFrame,
                 initial_capital: float,
                 trading_universe: List[str], # Move this parameter up
                 commission_per_share: float = 0.005,
                 slippage_pct: float = 0.0005,
                 config_override: Dict = None,
                 tca_log_path: str = None):
        """
        Initializes the backtest engine for the multi-factor portfolio.

        Args:
            historical_data (pd.DataFrame): DataFrame with multi-index columns for symbols.
            initial_capital (float): The starting capital for the simulation.
            commission_per_share (float): A fixed commission cost per share traded.
            slippage_pct (float): A percentage of the fill price to simulate slippage.
            config_override (Dict, optional): A dictionary to override global config values.
            tca_log_path (str, optional): If provided, path to log simulated fills to a DuckDB file.
            trading_universe (List[str]): The specific list of symbols to be traded.
        """
        self.historical_data = historical_data
        self.initial_capital = initial_capital
        self.commission_per_share = commission_per_share
        self.slippage_pct = slippage_pct
        self.config = config_override if config_override else global_config

        self.tca_log_path = tca_log_path
        if self.tca_log_path:
            db_dir = os.path.dirname(self.tca_log_path)
            if db_dir: os.makedirs(db_dir, exist_ok=True)
            if os.path.exists(self.tca_log_path):
                os.remove(self.tca_log_path)
            self._initialize_tca_db()

        self.event_queue = queue.Queue()
        
        # The engine's main symbol list is the full alpha universe
        self.symbols = list(set(col[0] for col in self.historical_data.columns))

        # --- CHANGE: Pass both universes to the portfolio manager ---
        self.portfolio_manager = LivePortfolioManager(
            event_queue=self.event_queue,
            symbol_list=self.symbols, # Full alpha universe for context
            trading_universe=trading_universe, # Smaller universe for execution
            initial_capital=self.initial_capital,
            feature_repo_path=self.config['feature_repo_path'],
            backtest_mode=True,
            config_override=self.config
        )
        
        # --- MODIFICATION: Give strategies access to historical data ---
        # The BaseStrategy needs historical data to be populated for its logic.
        # We initialize empty DataFrames that will be filled during the backtest run.
        for strategy in self.portfolio_manager.strategies.values():
            strategy.data = {s: pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume']) for s in self.symbols}

        self.equity_curve = []

    def _initialize_tca_db(self):
        """Creates the execution_log table in the specified DuckDB file."""
        logger.info(f"Initializing TCA log database at: {self.tca_log_path}")
        with duckdb.connect(self.tca_log_path) as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS execution_log (
                    timestamp TIMESTAMP,
                    symbol VARCHAR,
                    algo_used VARCHAR,
                    quantity INTEGER,
                    fill_price DOUBLE,
                    benchmark_price DOUBLE,
                    slippage DOUBLE
                );
            """)

    def run(self) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """
        Runs the system-level backtest from start to finish and returns
        performance metrics and the detailed equity curve.
        """
        logger.info("Starting system-level backtest with HRP multi-factor portfolio...")

        for timestamp, row in self.historical_data.iterrows():
            # --- MODIFICATION: Update strategy dataframes with the latest bar ---
            # This is crucial for strategies that need historical data for calculations.
            for symbol in self.symbols:
                if (symbol, 'close') in row.index and not pd.isna(row[(symbol, 'close')]):
                    market_data = {
                        'open': row[(symbol, 'open')], 'high': row[(symbol, 'high')],
                        'low': row[(symbol, 'low')], 'close': row[(symbol, 'close')],
                        'volume': row[(symbol, 'volume')]
                    }
                    # Update data for all strategies
                    for strategy in self.portfolio_manager.strategies.values():
                        new_row_df = pd.DataFrame([market_data], index=[timestamp])
                        strategy.data[symbol] = pd.concat([strategy.data[symbol], new_row_df])

            # 1. Create MarketEvents for all symbols for this timestamp
            for symbol in self.symbols:
                if (symbol, 'close') in row.index and not pd.isna(row[(symbol, 'close')]):
                    market_data = {
                        'open': row[(symbol, 'open')], 'high': row[(symbol, 'high')],
                        'low': row[(symbol, 'low')], 'close': row[(symbol, 'close')],
                        'volume': row[(symbol, 'volume')]
                    }
                    market_event = MarketEvent(symbol, timestamp, market_data)
                    self.event_queue.put(market_event)

            # --- MODIFICATION: Run periodic HRP allocation ---
            # This simulates the periodic re-weighting of strategies.
            self.portfolio_manager.run_allocation(timestamp.date())

            # 2. Process all events generated by this market tick
            while not self.event_queue.empty():
                event = self.event_queue.get()
                self._process_event(event, timestamp)

            # 3. Record portfolio equity at the end of the day
            self._update_portfolio_value(timestamp)

        logger.info("Backtest finished. Calculating final performance...")
        equity_df = pd.DataFrame(self.equity_curve, columns=['timestamp', 'equity']).set_index('timestamp')

        performance_metrics = calculate_performance_metrics(equity_df['equity'])
        return performance_metrics, equity_df

    def _process_event(self, event, current_timestamp):
        """Dispatches events to the appropriate handlers."""
        if event.type == 'MARKET':
            # The portfolio manager now fans out market events to its internal strategies
            self.portfolio_manager.process_market_data(event)
        elif event.type == 'SIGNAL':
            self.portfolio_manager.process_signal(event)
        elif event.type == 'ORDER':
            self._simulate_fill(event, current_timestamp)
        elif event.type == 'FILL':
            self.portfolio_manager.process_fill(event)

    def _simulate_fill(self, event: OrderEvent, current_timestamp):
        """Simulates the execution of an order with slippage and commission."""
        try:
            current_index = self.historical_data.index.get_loc(current_timestamp)
            if current_index + 1 >= len(self.historical_data):
                return # Cannot fill at the very end of the data

            fill_timestamp = self.historical_data.index[current_index + 1]
            # Assume fill happens at the next day's open price
            fill_price = self.historical_data.iloc[current_index + 1][(event.symbol, 'open')]

            if pd.isna(fill_price):
                logger.warning(f"Skipping fill for {event.symbol} on {fill_timestamp.date()} due to NaN price.")
                return

            # Apply slippage based on trade direction
            if event.direction == 'BUY':
                fill_price *= (1 + self.slippage_pct)
            else: # SELL
                fill_price *= (1 - self.slippage_pct)

            fill_cost = fill_price * event.quantity
            commission_cost = self.commission_per_share * event.quantity

            fill_event = FillEvent(
                timestamp=fill_timestamp,
                symbol=event.symbol,
                exchange='BACKTEST',
                quantity=event.quantity,
                direction=event.direction,
                fill_cost=fill_cost,
                commission=commission_cost,
                strategy_id=event.strategy_id
            )
            self.event_queue.put(fill_event)

            # Log the simulated fill if a path is provided
            if self.tca_log_path:
                self._log_simulated_trade(fill_event, event.arrival_price, event.order_type)

        except Exception as e:
            logger.error(f"Error simulating fill for {event.symbol}: {e}", exc_info=True)

    def _log_simulated_trade(self, fill: FillEvent, benchmark_price: float, algo_used: str):
        """Logs a simulated trade to the specified DuckDB database."""
        try:
            with duckdb.connect(self.tca_log_path) as con:
                direction_multiplier = 1 if fill.direction == 'BUY' else -1
                slippage = (fill.average_price - benchmark_price) * direction_multiplier

                con.execute(
                    "INSERT INTO execution_log VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (fill.timestamp, fill.symbol, algo_used, fill.quantity, fill.average_price, benchmark_price, slippage)
                )
        except Exception as e:
            logger.error(f"Could not log simulated trade to '{self.tca_log_path}': {e}")

    def _update_portfolio_value(self, timestamp: datetime):
        """Calculates and records the current total portfolio value."""
        self.portfolio_manager._update_total_equity()
        current_equity = self.portfolio_manager.risk_manager.current_equity
        self.equity_curve.append((timestamp, current_equity))