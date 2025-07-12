# portfolio/live_manager.py

import queue
from datetime import datetime
import pandas as pd
import logging
from typing import Dict, List, Optional

from core.events import OrderEvent, SignalEvent, FillEvent, MarketEvent
from portfolio.risk_manager import RiskManager
from portfolio.allocation import get_hrp_weights
from strategy.base import BaseStrategy
from strategy.quality_value import QualityValueStrategy
from strategy.low_volatility import LowVolatilityStrategy
from strategy.cross_sectional_momentum import CrossSectionalMomentumStrategy
from core.config_loader import config as global_config

logger = logging.getLogger("TradingSystem")

class LivePortfolioManager:
    """
    The LivePortfolioManager orchestrates the new multi-factor portfolio. It is
    responsible for:
    1.  Instantiating and managing the underlying trading strategies.
    2.  Periodically calculating optimal capital allocation between strategies using
        Hierarchical Risk Parity (HRP).
    3.  Processing signals from strategies and sizing trades based on the
        strategy's allocated capital.
    4.  Applying portfolio-level risk management (e.g., drawdown controls).
    5.  Maintaining the current state of holdings and cash for each strategy sleeve.
    """
    def __init__(self,
                 event_queue: queue.Queue,
                 symbol_list: list, # This is the full alpha universe
                 trading_universe: list, # This is the smaller tradable set
                 initial_capital: float,
                 feature_repo_path: str,
                 backtest_mode: bool = False,
                 config_override: Optional[Dict] = None):
        """
        Initializes the LivePortfolioManager.

        Args:
            event_queue (queue.Queue): The main system event queue.
            symbol_list (list): The list of symbols for signal generation (alpha universe).
            trading_universe (list): The list of symbols the system can actually trade.
            initial_capital (float): The total starting capital for the portfolio.
            feature_repo_path (str): The path to the Feast feature repository.
            backtest_mode (bool): Flag to indicate if running in backtest mode.
            config_override (Optional[Dict]): A dictionary to override global config values.
        """
        self.config = config_override if config_override is not None else global_config
        self.event_queue = event_queue
        self.symbol_list = symbol_list # Store the alpha universe
        self.trading_universe = trading_universe # Store the trading universe
        self.initial_capital = initial_capital
        self.backtest_mode = backtest_mode

        self.risk_manager = RiskManager(initial_capital=initial_capital)

        # --- Instantiate the new strategies ---
        # In a live system, these strategies would use a real-time feature store.
        # In backtesting, their internal dataframes will be populated by the engine.
        self.strategies: Dict[str, BaseStrategy] = {
            'quality_value': QualityValueStrategy(self.event_queue, self.symbol_list, 'quality_value', feature_repo_path=feature_repo_path),
            'low_volatility': LowVolatilityStrategy(self.event_queue, self.symbol_list, 'low_volatility', feature_repo_path=feature_repo_path),
            'cross_sectional_momentum': CrossSectionalMomentumStrategy(self.event_queue, self.symbol_list, 'cross_sectional_momentum')
        }
        strategy_ids = list(self.strategies.keys())
        logger.info(f"PortfolioManager initialized with strategies: {strategy_ids}")

        # --- HRP-based allocation state ---
        # Start with equal weights until enough data is gathered for HRP.
        self.strategy_weights = {sid: 1.0 / len(strategy_ids) for sid in strategy_ids}
        self.strategy_returns_df = pd.DataFrame(columns=strategy_ids)
        self.allocation_lookback = self.config.get('portfolio_optimizer', {}).get('performance_lookback_days', 126) # Default to ~6 months
        self.last_allocation_date: Optional[datetime.date] = None

        # --- Portfolio State Tracking ---
        self.latest_market_data: Dict[str, Dict] = {symbol: {} for symbol in self.symbol_list}
        self.holdings: Dict[str, Dict] = {sid: {sym: {'shares': 0, 'cost_basis': 0.0} for sym in symbol_list} for sid in strategy_ids}
        self.strategy_cash: Dict[str, float] = {sid: initial_capital * self.strategy_weights[sid] for sid in strategy_ids}
        self.last_day_equity: Dict[str, float] = self.strategy_cash.copy()

    def process_market_data(self, event: MarketEvent):
        """
        Updates market data and fans out the event to all strategies so they
        can calculate their signals.
        """
        self.update_holdings(event)
        for strategy in self.strategies.values():
            strategy.calculate_signals(event)

    def update_holdings(self, event: MarketEvent):
        """Stores the latest market data for valuation purposes."""
        if event.type == 'MARKET':
            self.latest_market_data[event.symbol] = event.data

    def process_signal(self, event: SignalEvent):
        """
        Processes a signal from a strategy, sizes the trade based on the
        strategy's dedicated capital, and creates an OrderEvent.
        """
        strategy_id = event.strategy_id
        if strategy_id not in self.strategies:
            logger.warning(f"Received signal for unknown strategy_id: {strategy_id}")
            return

        price = self.latest_market_data.get(event.symbol, {}).get('close', 0)
        if price <= 0:
            logger.warning(f"Cannot process signal for {event.symbol}; price is zero or unavailable.")
            return

        # --- Position Sizing based on Strategy's Capital Sleeve ---
        strategy_market_value = sum(
            data['shares'] * self.latest_market_data.get(symbol, {}).get('close', 0)
            for symbol, data in self.holdings[strategy_id].items()
        )
        strategy_equity = self.strategy_cash[strategy_id] + strategy_market_value

        # Apply portfolio-level drawdown throttle to the strategy's capital base
        drawdown_scale = self.risk_manager.get_risk_scaling_factor()
        capital_base = strategy_equity * drawdown_scale

        # Determine position size. Here, we use a simple equal-weighting within the strategy.
        # Each stock in the strategy's target portfolio gets an equal slice of the strategy's capital.
        target_portfolio_size = getattr(self.strategies[strategy_id], 'max_portfolio_size', 20)
        dollar_amount_per_stock = capital_base / target_portfolio_size if target_portfolio_size > 0 else 0
        target_quantity = int(dollar_amount_per_stock / price) if price > 0 else 0

        if target_quantity == 0 and event.direction == 'LONG':
             logger.debug(f"Calculated target quantity for {event.symbol} is zero. No order generated.")
             return

        # --- NEW: Add the final execution filter ---
        if event.symbol not in self.trading_universe:
            logger.debug(f"Signal for {event.symbol} ignored: not in trading universe.")
            return

        # --- Generate Order based on Signal ---
        current_position = self.holdings[strategy_id][event.symbol]['shares']

        if event.direction == 'EXIT':
            if current_position > 0:
                order = OrderEvent(strategy_id, event.symbol, 'MKT', abs(current_position), 'SELL')
                self.event_queue.put(order)
                logger.info(f"Generated EXIT order for {strategy_id}: SELL {abs(current_position)} {event.symbol}")
        elif event.direction == 'LONG':
            trade_quantity = target_quantity - current_position
            if trade_quantity > 0:
                order = OrderEvent(strategy_id, event.symbol, 'MKT', trade_quantity, 'BUY')
                self.event_queue.put(order)
                logger.info(f"Generated LONG order for {strategy_id}: BUY {trade_quantity} {event.symbol}")
            elif trade_quantity < 0:
                # This can happen if the target size decreases due to capital changes
                order = OrderEvent(strategy_id, event.symbol, 'MKT', abs(trade_quantity), 'SELL')
                self.event_queue.put(order)
                logger.info(f"Generated REDUCE order for {strategy_id}: SELL {abs(trade_quantity)} {event.symbol}")

    def process_fill(self, event: FillEvent):
        """
        Updates the holdings and cash for the specific strategy that
        generated the trade.
        """
        strategy_id = event.strategy_id
        if not strategy_id or strategy_id not in self.holdings:
            logger.error(f"Fill event for {event.symbol} has invalid strategy_id '{strategy_id}'. Cannot attribute P&L.")
            return

        symbol_holdings = self.holdings[strategy_id][event.symbol]

        if event.direction == 'BUY':
            new_total_cost = symbol_holdings['cost_basis'] * symbol_holdings['shares'] + event.fill_cost
            symbol_holdings['shares'] += event.quantity
            symbol_holdings['cost_basis'] = new_total_cost / symbol_holdings['shares'] if symbol_holdings['shares'] != 0 else 0
            self.strategy_cash[strategy_id] -= (event.fill_cost + event.commission)
        else: # SELL
            pnl = (event.average_price - symbol_holdings['cost_basis']) * event.quantity - event.commission if symbol_holdings['cost_basis'] > 0 else 0
            logger.info(f"Realized PNL for {strategy_id} on {event.symbol}: ${pnl:,.2f}")
            symbol_holdings['shares'] -= event.quantity
            self.strategy_cash[strategy_id] += (event.fill_cost - event.commission)
            if symbol_holdings['shares'] == 0:
                symbol_holdings['cost_basis'] = 0.0

        self._update_total_equity()

    def _update_total_equity(self):
        """
        Calculates the total portfolio value across all strategy sleeves and
        updates the central RiskManager.
        """
        total_market_value = 0
        for sid in self.holdings:
            for symbol, data in self.holdings[sid].items():
                if data['shares'] != 0:
                    last_price = self.latest_market_data.get(symbol, {}).get('close', 0)
                    total_market_value += data['shares'] * last_price

        total_cash = sum(self.strategy_cash.values())
        total_portfolio_value = total_cash + total_market_value
        self.risk_manager.update_equity(total_portfolio_value)

    def run_allocation(self, current_date: datetime.date):
        """
        Periodically re-calculates strategy weights using HRP. In a live system,
        this might trigger rebalancing trades. In this simplified version, it
        updates the target weights for future capital allocation.
        """
        if self.last_allocation_date and (current_date - self.last_allocation_date).days < 30:
            return # Re-run HRP allocation monthly

        logger.info(f"Running HRP allocation on {current_date}...")
        self.last_allocation_date = current_date

        # 1. Calculate daily returns for each strategy to build a history
        for sid in self.strategies.keys():
            market_value = sum(
                data['shares'] * self.latest_market_data.get(symbol, {}).get('close', 0)
                for symbol, data in self.holdings[sid].items()
            )
            current_equity = self.strategy_cash[sid] + market_value
            daily_return = (current_equity / self.last_day_equity[sid] - 1) if self.last_day_equity[sid] != 0 else 0
            self.strategy_returns_df.loc[current_date, sid] = daily_return
            self.last_day_equity[sid] = current_equity

        # Keep a rolling window of returns for the covariance calculation
        self.strategy_returns_df = self.strategy_returns_df.tail(self.allocation_lookback)
        if len(self.strategy_returns_df) < 20: # Need a minimum history for a stable covariance matrix
            logger.warning("Not enough return history to run HRP. Using previous weights.")
            return

        # 2. Calculate new weights with HRP
        cov_matrix = self.strategy_returns_df.cov()
        new_weights = get_hrp_weights(cov_matrix)
        
        if not new_weights.empty:
            self.strategy_weights = new_weights.to_dict()
            logger.info(f"HRP allocation complete. New target weights: {self.strategy_weights}")
        else:
            logger.error("HRP allocation failed to produce weights. Retaining previous weights.")

        # In a live system, one might reallocate cash here. For simplicity,
        # we let cash drift and use the new weights for future calculations.