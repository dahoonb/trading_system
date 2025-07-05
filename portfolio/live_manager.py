# portfolio/live_manager.py

import queue
from datetime import datetime
import pandas as pd
import feast
import joblib
import os
import logging
from typing import Dict, List, Optional

from core.events import OrderEvent, SignalEvent, FillEvent, MarketEvent
from portfolio.shared_sizing_logic import get_atr_volatility_adjusted_size
from portfolio.algo_wheel import AlgoWheel
from portfolio.monthly_counter import MonthlyCounter
from portfolio.risk_manager import RiskManager
from portfolio.optimizer import PortfolioOptimizer
from core.config_loader import config as global_config
from strategy.momentum import MovingAverageCrossoverStrategy
from strategy.mean_reversion import RsiMeanReversionStrategy

logger = logging.getLogger("TradingSystem")

class LivePortfolioManager:
    """
    The LivePortfolioManager is the core strategic component of the trading
    system, orchestrating risk, optimization, and order generation.
    It can be initialized with a specific configuration for backtesting.
    """
    def __init__(self, event_queue: queue.Queue, symbol_list: list, initial_capital: float, 
                 feature_repo_path: str, backtest_mode: bool = False, config_override: Optional[Dict] = None):
        
        # Use the provided config override, or fall back to the global config
        self.config = config_override if config_override is not None else global_config
        
        self.event_queue = event_queue
        self.symbol_list = symbol_list
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.backtest_mode = backtest_mode
        
        self.fill_counter = MonthlyCounter(persistence_filepath="state/fill_counter_state.json", config_override=self.config)
        self.algo_wheel = AlgoWheel(tca_db_path="data/tca_log.duckdb", config_override=self.config)
        self.risk_manager = RiskManager(initial_capital=initial_capital, config_override=self.config)
        
        strategy_ids = list(self.config.get('strategies', {}).keys())
        self.optimizer = PortfolioOptimizer(strategy_ids=strategy_ids, config_override=self.config)
        
        if not self.backtest_mode:
            self.feature_store = feast.FeatureStore(repo_path=feature_repo_path)
            self.ml_model_path = self.config.get('ml_models', {}).get('signal_veto_model_path', 'models/ml_vetoing_model.pkl')
            try:
                self.ml_model = joblib.load(self.ml_model_path)
                logger.info(f"Successfully loaded ML vetoing model from {self.ml_model_path}.")
            except FileNotFoundError:
                self.ml_model = None
                logger.warning(f"ML vetoing model not found at '{self.ml_model_path}'.")
        else:
            self.feature_store = None
            self.ml_model = None
            logger.info("LivePortfolioManager running in backtest mode. ML/Feast disabled.")

        self.latest_market_data = {symbol: {} for symbol in self.symbol_list}
        self.holdings = {sid: {sym: {'shares': 0, 'cost_basis': 0.0} for sym in symbol_list} for sid in strategy_ids}
        self.strategy_cash = {sid: initial_capital / len(strategy_ids) if strategy_ids else 0 for sid in strategy_ids}
        self.last_day_equity = {sid: initial_capital / len(strategy_ids) if strategy_ids else 0 for sid in strategy_ids}
        
        self.strategies = {}
        strategy_configs = self.config.get('strategies', {})
        for strategy_name, details in strategy_configs.items():
            if details.get('enabled'):
                params = {k: v for k, v in details.items() if k != 'enabled'}
                if strategy_name == 'momentum_ma':
                    self.strategies[strategy_name] = MovingAverageCrossoverStrategy(self.event_queue, self.symbol_list, strategy_id=strategy_name, **params)
                elif strategy_name == 'mean_reversion_rsi':
                    self.strategies[strategy_name] = RsiMeanReversionStrategy(self.event_queue, self.symbol_list, strategy_id=strategy_name, **params)

    def process_market_data(self, event: MarketEvent):
        """Processes a market event by fanning it out to all strategies."""
        self.update_holdings(event)
        for strategy in self.strategies.values():
            strategy.calculate_signals(event)

    def update_holdings(self, event: MarketEvent):
        if event.type == 'MARKET':
            self.latest_market_data[event.symbol] = event.data

    def process_signal(self, event: SignalEvent):
        """Processes a signal, applying all layers of risk and optimization."""
        if self.fill_counter.is_budget_exceeded(): return

        if self.ml_model and not self._is_signal_ml_approved(event.symbol):
            logger.info(f"ML Model VETOED signal for {event.symbol}.")
            return

        price = self.latest_market_data.get(event.symbol, {}).get('close', 0)
        if price <= 0: return
            
        strategy_id = event.strategy_id
        strategy_weights = self.optimizer.get_weights()
        strategy_weight = strategy_weights.get(strategy_id, 1.0 / len(self.optimizer.strategy_ids) if self.optimizer.strategy_ids else 1.0)
        capital_base = self.risk_manager.current_equity * strategy_weight
        
        atr = self._get_latest_atr(event.symbol) or price * 0.02
        quantity = get_atr_volatility_adjusted_size(
            capital=capital_base,
            risk_per_trade_pct=self.config['portfolio']['risk_per_trade_pct'],
            price=price,
            atr=atr
        )

        drawdown_scale = self.risk_manager.get_risk_scaling_factor()
        volatility_scale = self.optimizer.get_volatility_scaling_factor()
        final_quantity = int(quantity * drawdown_scale * volatility_scale)

        if final_quantity == 0: return

        order_context = {'order_size': final_quantity, 'volatility': atr / price}
        order_type = self.algo_wheel.get_execution_decision(event.symbol, order_context)
        
        order_direction = 'BUY' if event.direction == 'LONG' else 'SELL'
        current_position = self.holdings[strategy_id][event.symbol]['shares']
        if (event.direction == 'LONG' and current_position < 0) or (event.direction == 'SHORT' and current_position > 0):
            final_quantity += abs(current_position)

        order = OrderEvent(strategy_id, event.symbol, order_type, final_quantity, order_direction, arrival_price=price)
        self.event_queue.put(order)
        logger.info(f"Portfolio Manager: Generated {order}")

    def process_fill(self, event: FillEvent):
        """Updates portfolio state on a per-strategy basis after a fill."""
        self.fill_counter.increment()
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
        """Calculates total portfolio value and updates the RiskManager."""
        total_market_value = 0
        for sid in self.holdings:
            for symbol, data in self.holdings[sid].items():
                if data['shares'] != 0:
                    last_price = self.latest_market_data.get(symbol, {}).get('close', 0)
                    total_market_value += data['shares'] * last_price
        
        total_cash = sum(self.strategy_cash.values())
        total_portfolio_value = total_cash + total_market_value
        self.risk_manager.update_equity(total_portfolio_value)

    def run_optimization(self):
        """Calculates daily returns for each strategy and triggers the optimizer."""
        logger.info("Running periodic performance tracking for optimizer...")
        daily_returns = {}
        for sid in self.optimizer.strategy_ids:
            market_value = sum(
                data['shares'] * self.latest_market_data.get(symbol, {}).get('close', 0)
                for symbol, data in self.holdings[sid].items() if data['shares'] != 0
            )
            current_equity = self.strategy_cash[sid] + market_value
            daily_return = (current_equity / self.last_day_equity[sid] - 1) if self.last_day_equity[sid] != 0 else 0
            daily_returns[sid] = daily_return
            self.last_day_equity[sid] = current_equity
            
        self.optimizer.track_daily_returns(daily_returns)
        self.optimizer.calculate_optimal_weights()

    def _get_latest_atr(self, symbol: str) -> Optional[float]:
        if self.backtest_mode or not self.feature_store:
            return None
        try:
            features = self.feature_store.get_online_features(features=["all_ticker_features:atr_20"], entity_rows=[{"ticker_id": symbol}]).to_dict()
            return features['atr_20'][0]
        except Exception:
            return None

    def _is_signal_ml_approved(self, symbol: str) -> bool:
        if self.backtest_mode or not self.ml_model:
            return True
        try:
            model_feature_refs = ["all_ticker_features:rsi_14", "all_ticker_features:sma_200", "all_ticker_features:vol_regime"]
            feature_vector = self.feature_store.get_online_features(features=model_feature_refs, entity_rows=[{"ticker_id": symbol}]).to_df()
            if feature_vector.empty: return True
            features_for_prediction = feature_vector[["rsi_14", "sma_200", "vol_regime"]]
            prediction = self.ml_model.predict(features_for_prediction)[0]
            return prediction == 1
        except Exception as e:
            logger.error(f"Error during ML prediction for {symbol}: {e}. Approving signal by default.")
            return True