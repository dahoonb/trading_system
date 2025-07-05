# main.py

import os
import queue
import signal
import time
import logging
from datetime import datetime, timezone

from core.config_loader import config
from core.event_queue import main_queue
from data.ib_handler import IBHandler
from execution.ib_executor import IBExecutionHandler
from portfolio.live_manager import LivePortfolioManager
from strategy.momentum import MovingAverageCrossoverStrategy
from strategy.mean_reversion import RsiMeanReversionStrategy
from utils.logger import setup_logger

# Setup logger
setup_logger(logger_name="TradingSystem", log_level=logging.INFO)
logger = logging.getLogger("TradingSystem")

class TradingSystem:
    def __init__(self):
        """Initializes the trading system components."""
        self.event_queue = main_queue
        self.symbols = config['symbols']
        self.running = True

        self.ib_handler = IBHandler(self.event_queue, self.symbols)
        
        self.portfolio_manager = LivePortfolioManager(
            self.event_queue,
            self.symbols,
            initial_capital=config['portfolio']['initial_capital'],
            feature_repo_path=config['feature_repo_path']
        )
        
        self.execution_handler = IBExecutionHandler(
            self.event_queue,
            self.ib_handler
        )
        
        self.ib_handler.execution_handler = self.execution_handler
        self.ib_handler.order_status_handler = self.execution_handler
        self.execution_handler.portfolio_manager = self.portfolio_manager
        
        self.strategies = {}
        strategy_configs = config.get('strategies', {})
        for strategy_name, details in strategy_configs.items():
            if details.get('enabled'):
                params = {k: v for k, v in details.items() if k != 'enabled'}
                if strategy_name == 'momentum_ma':
                    self.strategies[strategy_name] = MovingAverageCrossoverStrategy(self.event_queue, self.symbols, strategy_id=strategy_name, **params)
                elif strategy_name == 'mean_reversion_rsi':
                    self.strategies[strategy_name] = RsiMeanReversionStrategy(self.event_queue, self.symbols, strategy_id=strategy_name, **params)
        
        logger.info(f"Loaded {len(self.strategies)} strategies: {list(self.strategies.keys())}")
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)

    def run(self):
        """Runs the main event loop, periodic optimization, and heartbeat."""
        logger.info("Starting Trading System...")
        conn_conf = config['ib_connection']
        self.ib_handler.connect(conn_conf['host'], conn_conf['port'], conn_conf['main_client_id'])
        self.ib_handler.subscribe_live_data()
        
        last_optimization_time = time.time()
        optimization_frequency_seconds = config.get('portfolio_optimizer', {}).get('optimization_frequency_days', 7) * 86400

        # --- MODIFICATION: Heartbeat setup ---
        op_risk_config = config.get('operational_risk', {})
        heartbeat_path = op_risk_config.get('heartbeat_flag_path', 'state/heartbeat.flag')
        heartbeat_interval = 60  # Update heartbeat every 60 seconds
        last_heartbeat_time = 0

        while self.running:
            now = time.time()
            
            # Periodic portfolio optimization
            if now - last_optimization_time > optimization_frequency_seconds:
                logger.info("Scheduled portfolio optimization triggered.")
                self.portfolio_manager.run_optimization()
                last_optimization_time = now

            # --- MODIFICATION: Provide heartbeat ---
            if now - last_heartbeat_time > heartbeat_interval:
                try:
                    with open(heartbeat_path, "w") as f:
                        f.write(datetime.now(timezone.utc).isoformat())
                    last_heartbeat_time = now
                except IOError as e:
                    logger.error(f"Could not write heartbeat file at '{heartbeat_path}': {e}")

            if os.path.exists("kill.flag"):
                logger.info("Kill flag detected. Shutting down main application...")
                self.shutdown(None, None)
                break

            try:
                event = self.event_queue.get(block=True, timeout=1)
                if event: self.process_event(event)
            except queue.Empty:
                continue

    def process_event(self, event):
        """Dispatches events to the appropriate handlers."""
        if event.type == 'MARKET':
            self.portfolio_manager.process_market_data(event)
        elif event.type == 'SIGNAL':
            self.portfolio_manager.process_signal(event)
        elif event.type == 'ORDER':
            self.execution_handler.process_order(event)
        elif event.type == 'FILL':
            self.portfolio_manager.process_fill(event)

    def shutdown(self, signum, frame):
        """Handles the graceful shutdown of the trading system."""
        if not self.running: return
        logger.info("\nShutting down Trading System...")
        self.running = False
        for strategy in self.strategies.values():
            strategy.save_state()
        self.ib_handler.disconnect()
        
        # --- MODIFICATION: Clean up heartbeat file on shutdown ---
        heartbeat_path = config.get('operational_risk', {}).get('heartbeat_flag_path', 'state/heartbeat.flag')
        if os.path.exists(heartbeat_path):
            try:
                os.remove(heartbeat_path)
                logger.info(f"Removed heartbeat file: {heartbeat_path}")
            except OSError as e:
                logger.error(f"Error removing heartbeat file: {e}")

        logger.info("Trading System has been shut down.")

if __name__ == "__main__":
    os.makedirs("state", exist_ok=True)
    system = TradingSystem()
    try:
        system.run()
    except (KeyboardInterrupt, SystemExit) as e:
        logger.info(f"Caught shutdown signal: {e}")
    finally:
        if system.running:
            system.shutdown(None, None)