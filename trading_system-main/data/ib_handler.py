# data/ib_handler.py

import queue
import threading
from datetime import datetime
import time

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.common import TickerId
from ibapi.execution import Execution

from core.events import MarketEvent

class IBHandler(EWrapper, EClient):
    """
    Handles the connection to Interactive Brokers, manages the data feed,
    and listens for all incoming messages from the broker. It runs in a
    separate thread to ensure non-blocking operation.
    """
    def __init__(self, event_queue: queue.Queue, symbols: list):
        EClient.__init__(self, self)
        self.event_queue = event_queue
        self.symbols = symbols
        self.next_order_id = None
        self.req_id_to_symbol = {}
        self.next_req_id = 0
        self.client_thread = None
        
        # Handlers to be set by the main TradingSystem
        self.execution_handler = None
        self.order_status_handler = None

    def connect(self, host: str = "127.0.0.1", port: int = 7497, client_id: int = 1):
        """Connects to the IB TWS/Gateway and starts the client thread."""
        print(f"Connecting to IB on {host}:{port}...")
        super().connect(host, port, clientId=client_id)
        self.client_thread = threading.Thread(target=self.run, daemon=True)
        self.client_thread.start()
        time.sleep(2)
        if not self.isConnected():
            raise ConnectionError("Failed to connect to Interactive Brokers.")
        print("Successfully connected to IB.")

    def disconnect(self):
        """Disconnects from the IB TWS/Gateway."""
        if self.isConnected():
            print("Disconnecting from IB...")
            super().disconnect()
            if self.client_thread and self.client_thread.is_alive():
                self.client_thread.join(timeout=2)

    def subscribe_live_data(self):
        """Subscribes to live 5-second bars for all symbols."""
        print("Subscribing to live market data...")
        for symbol in self.symbols:
            self.next_req_id += 1
            self.req_id_to_symbol[self.next_req_id] = symbol
            contract = self._create_stock_contract(symbol)
            self.reqRealTimeBars(self.next_req_id, contract, 5, "TRADES", True, [])
            print(f"  - Subscribed to 5-sec bars for {symbol}")

    def _create_stock_contract(self, symbol: str) -> Contract:
        """Helper function to create a stock contract object."""
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"
        return contract

    # --- EWrapper Callback Methods ---

    def execDetails(self, reqId: int, contract: Contract, execution: Execution):
        """Callback for receiving execution details (fills)."""
        super().execDetails(reqId, contract, execution)
        if self.execution_handler:
            self.execution_handler.process_fill(execution)
        else:
            print("ERROR: execDetails received but no execution_handler is set in IBHandler.")

    def orderStatus(self, orderId: TickerId, status: str, filled: float, remaining: float, avgFillPrice: float, permId: int, parentId: int, lastFillPrice: float, clientId: int, whyHeld: str, mktCapPrice: float):
        """Callback for receiving order status updates."""
        super().orderStatus(orderId, status, filled, remaining, avgFillPrice, permId, parentId, lastFillPrice, clientId, whyHeld, mktCapPrice)
        
        # --- MODIFICATION: Delegate to the order status handler ---
        if self.order_status_handler:
            self.order_status_handler.process_order_status(orderId, status, filled, remaining, lastFillPrice)

    def nextValidId(self, orderId: int):
        """Receives the next valid order ID from the TWS."""
        super().nextValidId(orderId)
        self.next_order_id = orderId
        print(f"Next valid order ID received: {orderId}")

    def realtimeBar(self, reqId: TickerId, time: int, open_: float, high: float, low: float, close: float, volume: int, wap: float, count: int):
        """Callback for receiving real-time bar data."""
        dt = datetime.fromtimestamp(time)
        symbol = self.req_id_to_symbol.get(reqId)
        if symbol:
            market_data = {'open': open_, 'high': high, 'low': low, 'close': close, 'volume': volume}
            market_event = MarketEvent(symbol, dt, market_data)
            self.event_queue.put(market_event)

    def error(self, reqId: TickerId, errorCode: int, errorString: str, advancedOrderRejectJson=""):
        """Handles errors from the TWS."""
        if errorCode not in [2104, 2106, 2158, 2108, 2100, 2150]:
            print(f"Error (Req ID: {reqId}, Code: {errorCode}): {errorString}")