# execution/ib_executor.py

import queue
from datetime import datetime
import pandas as pd
import logging
import os
import fcntl
import sys

from ibapi.contract import Contract
from ibapi.order import Order
from ibapi.execution import Execution
from ibapi.tag_value import TagValue
from ibapi.common import TickerId

from core.events import FillEvent, OrderEvent
from tca.logger import TCALogger
from core.config_loader import config

logger = logging.getLogger("TradingSystem")

class IBExecutionHandler:
    """
    Handles the execution of orders by interacting with the Interactive
    Brokers API. It translates OrderEvents into IB-compatible orders,
    processes Execution objects and order statuses into FillEvents or TCA logs,
    and increments a shared counter for order rate monitoring.
    """
    def __init__(self, event_queue: queue.Queue, ib_handler):
        """
        Initializes the IBExecutionHandler.

        Args:
            event_queue (queue.Queue): The main event queue for the system.
            ib_handler (IBHandler): A reference to the connected IBHandler instance.
        """
        self.event_queue = event_queue
        self.ib_handler = ib_handler
        self.tca_logger = TCALogger(db_path="data/tca_log.duckdb")
        
        # This map stores the context of an order (strategy, benchmark price, etc.)
        # It's crucial for linking fills back to their original intent for TCA.
        self.order_context_map = {}
        
        # This reference will be set by the main TradingSystem class.
        self.portfolio_manager = None
        
        # Get order rate counter path from config and initialize it
        op_risk_config = config.get('operational_risk', {})
        self.order_rate_counter_path = op_risk_config.get('order_rate_counter_path', 'state/order_rate.counter')
        self._initialize_counter_file()

    def _initialize_counter_file(self):
        """Ensures the counter file exists and is set to 0."""
        try:
            os.makedirs(os.path.dirname(self.order_rate_counter_path), exist_ok=True)
            with open(self.order_rate_counter_path, "w") as f:
                f.write("0")
        except IOError as e:
            logger.critical(f"Could not initialize order rate counter file at '{self.order_rate_counter_path}': {e}")

    def _increment_order_counter(self):
        """Atomically increments the order counter file."""
        lock_path = self.order_rate_counter_path + ".lock"
        try:
            # Using a file lock for robustness in case of future multi-process expansion
            with open(lock_path, 'w') as lock_file:
                if 'fcntl' in sys.modules: # POSIX-specific file locking
                    fcntl.flock(lock_file, fcntl.LOCK_EX)
                
                # Perform read-increment-write operation
                with open(self.order_rate_counter_path, "r+") as f:
                    count = int(f.read() or 0)
                    f.seek(0)
                    f.write(str(count + 1))
                    f.truncate()

                if 'fcntl' in sys.modules:
                    fcntl.flock(lock_file, fcntl.LOCK_UN)
        except (IOError, ValueError) as e:
            logger.error(f"Could not increment order rate counter: {e}")
        finally:
            if os.path.exists(lock_path):
                try:
                    os.remove(lock_path)
                except OSError:
                    pass

    def process_order(self, event: OrderEvent):
        """
        Processes an OrderEvent, increments the order counter, creates the
        appropriate IB Order object, stores its context for TCA, and sends
        it to the IB API for execution.
        """
        if event.type != 'ORDER':
            return
            
        # Increment order rate counter before placing the order
        self._increment_order_counter()

        contract = self._create_stock_contract(event.symbol)
        
        order = Order()
        order.action = event.direction
        order.totalQuantity = event.quantity
        
        if event.order_type == 'ADAPTIVE':
            order.orderType = 'MKT'
            order.algoStrategy = 'Adaptive'
            order.algoParams = [TagValue("adaptivePriority", "Normal")]
        elif event.order_type == 'MOC':
            order.orderType = 'MOC'
        elif event.order_type == 'LOC':
            order.orderType = 'LMT'
            order.lmtPrice = event.arrival_price # Use arrival price as the limit for simplicity
            order.tif = "MOC"
        elif event.order_type == 'TWAP':
            order.orderType = 'MKT'
            order.algoStrategy = 'Twap'
            order.algoParams = [
                TagValue("strategyType", "Marketable"),
                TagValue("startTime", datetime.now().strftime("%Y%m%d %H:%M:%S")),
                TagValue("endTime", (datetime.now() + pd.Timedelta(minutes=5)).strftime("%Y%m%d %H:%M:%S"))
            ]
        else:
            order.orderType = 'MKT'
        
        order_id = self.ib_handler.next_order_id
        
        # Store context before placing the order
        self.order_context_map[order_id] = {
            'strategy_id': event.strategy_id,
            'symbol': event.symbol,
            'algo_used': event.order_type,
            'benchmark_price': event.arrival_price,
            'quantity': event.quantity,
            'direction': event.direction,
            'limit_price': order.lmtPrice if event.order_type == 'LOC' else None
        }
        
        self.ib_handler.placeOrder(order_id, contract, order)
        self.ib_handler.next_order_id += 1
        logger.info(f"Execution Handler: Placed Order ID {order_id} for {event}")

    def process_fill(self, execution: Execution):
        """
        Processes an Execution object, creates a FillEvent, and logs
        accurate TCA data using the stored order context.
        """
        fill_time = datetime.strptime(execution.time, '%Y%m%d  %H:%M:%S')
        order_id = execution.orderId
        context = self.order_context_map.get(order_id)

        if not context:
            logger.warning(f"Received fill for unknown Order ID {order_id}. Cannot log TCA.")
            return

        fill_event = FillEvent(
            timestamp=fill_time,
            symbol=execution.contract.symbol,
            exchange=execution.exchange,
            quantity=int(execution.shares),
            direction="BUY" if execution.side == "BOT" else "SELL",
            fill_cost=(execution.price * int(execution.shares)),
            commission=execution.commissionReport.commission if execution.commissionReport else 0.0,
            strategy_id=context.get('strategy_id')
        )
        self.event_queue.put(fill_event)

        self.tca_logger.log_execution(
            timestamp=fill_time,
            symbol=execution.contract.symbol,
            algo_used=context['algo_used'],
            quantity=fill_event.quantity,
            fill_price=execution.price,
            benchmark_price=context['benchmark_price']
        )
        logger.info(f"Execution Handler: Logged real TCA data for Order ID {order_id}.")

    def process_order_status(self, orderId: TickerId, status: str, filled: float, remaining: float, lastFillPrice: float):
        """
        Processes order status updates to detect cancelled orders
        and log them as missed trades for TCA.
        """
        if status == 'Cancelled' and remaining > 0 and orderId in self.order_context_map:
            context = self.order_context_map.pop(orderId) # Remove from map once handled
            
            if context.get('algo_used') == 'LOC' and self.portfolio_manager:
                symbol = context['symbol']
                last_market_price = self.portfolio_manager.latest_market_data.get(symbol, {}).get('close', 0)
                
                if last_market_price > 0 and context['limit_price']:
                    self.tca_logger.log_missed_trade(
                        timestamp=datetime.now(),
                        symbol=symbol,
                        algo_used=context['algo_used'],
                        quantity=int(remaining),
                        direction=context['direction'],
                        limit_price=context['limit_price'],
                        last_market_price=last_market_price
                    )
        elif status in ['Filled', 'Inactive'] and remaining == 0 and orderId in self.order_context_map:
            # Clean up the context map for fully filled or otherwise completed orders
            del self.order_context_map[orderId]

    def _create_stock_contract(self, symbol: str) -> Contract:
        """Helper function to create a stock contract object."""
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"
        return contract