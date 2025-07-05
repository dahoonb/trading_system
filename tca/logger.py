# tca/logger.py

import duckdb
import os
from datetime import datetime

class TCALogger:
    """
    Logs both filled and missed trades to a DuckDB database for comprehensive
    Transaction Cost Analysis. This data is used by the adaptive AlgoWheel.
    """
    def __init__(self, db_path="data/tca_log.duckdb"):
        """
        Initializes the TCALogger and the database.

        Args:
            db_path (str): The file path for the DuckDB database.
        """
        self.db_path = db_path
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        self._initialize_db()

    def _initialize_db(self):
        """Initializes the database and creates tables if they don't exist."""
        with duckdb.connect(self.db_path) as con:
            # Table for successful fills
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
            # --- MODIFICATION: Add table for missed trades ---
            con.execute("""
                CREATE TABLE IF NOT EXISTS missed_trades (
                    timestamp TIMESTAMP,
                    symbol VARCHAR,
                    algo_used VARCHAR,
                    quantity INTEGER,
                    direction VARCHAR,
                    limit_price DOUBLE,
                    last_market_price DOUBLE,
                    opportunity_cost DOUBLE
                );
            """)

    def log_execution(self, timestamp: datetime, symbol: str, algo_used: str, quantity: int, fill_price: float, benchmark_price: float):
        """Logs a single successful execution record."""
        direction_multiplier = 1 if quantity > 0 else -1
        slippage = (fill_price - benchmark_price) * direction_multiplier

        try:
            with duckdb.connect(self.db_path) as con:
                con.execute(
                    "INSERT INTO execution_log VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (timestamp, symbol, algo_used, quantity, fill_price, benchmark_price, slippage)
                )
        except duckdb.Error as e:
            print(f"Error logging execution to TCA database: {e}")

    def log_missed_trade(self, timestamp: datetime, symbol: str, algo_used: str, quantity: int, direction: str, limit_price: float, last_market_price: float):
        """
        Logs an unfilled/cancelled order and calculates its opportunity cost.

        Args:
            timestamp (datetime): Time of cancellation/expiration.
            symbol (str): The ticker symbol.
            algo_used (str): The execution algorithm that failed to fill.
            quantity (int): The number of shares that were not filled.
            direction (str): 'BUY' or 'SELL'.
            limit_price (float): The price of the passive limit order.
            last_market_price (float): The market price at the time of cancellation.
        """
        # Opportunity cost: The adverse price movement since the order was placed.
        # For a missed BUY, cost is how much the price went UP from the limit.
        # For a missed SELL, cost is how much the price went DOWN from the limit.
        if direction == 'BUY':
            opportunity_cost_per_share = max(0, last_market_price - limit_price)
        elif direction == 'SELL':
            opportunity_cost_per_share = max(0, limit_price - last_market_price)
        else:
            opportunity_cost_per_share = 0

        total_opportunity_cost = opportunity_cost_per_share * quantity

        try:
            with duckdb.connect(self.db_path) as con:
                con.execute(
                    "INSERT INTO missed_trades VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (timestamp, symbol, algo_used, quantity, direction, limit_price, last_market_price, total_opportunity_cost)
                )
            print(f"Logged missed trade for {quantity} {symbol} ({algo_used}). Opportunity Cost: ${total_opportunity_cost:.2f}")
        except duckdb.Error as e:
            print(f"Error logging missed trade to TCA database: {e}")